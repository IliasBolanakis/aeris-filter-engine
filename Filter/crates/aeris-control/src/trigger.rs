//! Deterministic EWMA proactive expansion trigger and hysteresis-guarded
//! filter contraction state machine.
//!
//! # Mathematical Equations
//! Given slot table capacity `N = 2^q`, current load factor `alpha`, EWMA
//! ingestion rate `lambda` (keys/sec), Reverse Map entry count `|RM|`, and
//! observed bulk-load speed `v_bulk` (keys/sec):
//!
//! ```text
//! T_full    = (N * max(MAX_LOAD_FACTOR - alpha, 0)) / lambda   (INFINITY if lambda <= 0)
//! T_rebuild = |RM| / max(v_bulk, 1.0)
//! ```
//!
//! - **Proactive Expansion (`TriggerAction::ProactiveExpand`):** fires when
//!   `T_full <= T_rebuild * margin` (Eq. 3) OR `alpha >= hard_expand_alpha`,
//!   launching a background `q -> q + 1` expansion before Robin Hood cluster
//!   lengths explode near `MAX_LOAD_FACTOR = 0.95`.
//! - **Contraction (`TriggerAction::Contract`):** fires when
//!   `q > MIN_QUOTIENT_BITS`, `alpha <= contraction_alpha_threshold` sustained
//!   for `contraction_sustained_checks` consecutive evaluations, AND projected
//!   post-contraction load factor `alpha * 2.0 <= max_post_contraction_alpha`,
//!   reclaiming RAM after bulk deletes without expand/contract thrashing.
//!
//! # Purity & Determinism
//! Contains zero system-clock calls and zero I/O; every evaluation is a pure
//! state transition driven solely by explicit [`TriggerInput`] metrics.

use aeris_types::{
    AerisError, AerisResult, DEFAULT_TARGET_LOAD_FACTOR, MAX_GUARDRAIL_PROACTIVE_MARGIN,
    MAX_LOAD_FACTOR, MAX_QUOTIENT_BITS, MIN_GUARDRAIL_PROACTIVE_MARGIN, MIN_QUOTIENT_BITS,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Default proactive rebuild safety margin multiplier `k` (`T_full <= T_rebuild * k`).
pub const DEFAULT_PROACTIVE_MARGIN: f64 = 1.5;

/// Default hard load-factor ceiling that unconditionally forces expansion.
pub const DEFAULT_HARD_EXPAND_ALPHA: f64 = DEFAULT_TARGET_LOAD_FACTOR;

/// Default load-factor ceiling below which a filter becomes eligible for contraction.
pub const DEFAULT_CONTRACTION_ALPHA_THRESHOLD: f64 = 0.25;

/// Default number of consecutive low-load checks required before triggering contraction.
pub const DEFAULT_CONTRACTION_SUSTAINED_CHECKS: u32 = 4;

/// Default maximum projected load factor (`2 * alpha`) permitted after contraction
/// so contraction does not immediately re-trigger proactive expansion.
pub const DEFAULT_MAX_POST_CONTRACTION_ALPHA: f64 = 0.65;

/// Minimum bulk-load speed floor (keys/sec) preventing division by zero in `T_rebuild`.
const MIN_BULK_LOAD_SPEED_FLOOR: f64 = 1.0;

/// Configuration parameters for [`ProactiveTrigger`].
///
/// # Invariants
/// Validated via [`ProactiveTriggerConfig::validate`]:
/// - `margin` is finite and in `[1.0, 3.0]`.
/// - `hard_expand_alpha` is finite and in `(0.0, MAX_LOAD_FACTOR]`.
/// - `contraction_alpha_threshold` is finite and in `(0.0, 0.50)`.
/// - `contraction_sustained_checks >= 1`.
/// - `max_post_contraction_alpha` is finite, `> contraction_alpha_threshold`,
///   and `< hard_expand_alpha`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProactiveTriggerConfig {
    /// Safety margin multiplier `k` in `[1.0, 3.0]` for `T_full <= T_rebuild * k`.
    pub margin: f64,
    /// Hard load-factor ceiling (`alpha >= hard_expand_alpha`) forcing expansion
    /// even when ingestion rate is zero or bursty.
    pub hard_expand_alpha: f64,
    /// Maximum load factor eligible for contraction (`q -> q - 1`).
    pub contraction_alpha_threshold: f64,
    /// Consecutive eligible evaluations required before emitting [`TriggerAction::Contract`].
    pub contraction_sustained_checks: u32,
    /// Upper bound on projected post-contraction load factor (`alpha * 2.0`).
    pub max_post_contraction_alpha: f64,
}

impl Default for ProactiveTriggerConfig {
    fn default() -> Self {
        Self {
            margin: DEFAULT_PROACTIVE_MARGIN,
            hard_expand_alpha: DEFAULT_HARD_EXPAND_ALPHA,
            contraction_alpha_threshold: DEFAULT_CONTRACTION_ALPHA_THRESHOLD,
            contraction_sustained_checks: DEFAULT_CONTRACTION_SUSTAINED_CHECKS,
            max_post_contraction_alpha: DEFAULT_MAX_POST_CONTRACTION_ALPHA,
        }
    }
}

impl ProactiveTriggerConfig {
    /// Validates all safety and anti-thrashing bounds on the trigger configuration.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on the first violated bound.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    pub fn validate(&self) -> AerisResult<()> {
        if !self.margin.is_finite()
            || self.margin < MIN_GUARDRAIL_PROACTIVE_MARGIN
            || self.margin > MAX_GUARDRAIL_PROACTIVE_MARGIN
        {
            return Err(AerisError::InvalidConfig(format!(
                "proactive margin must be in [{MIN_GUARDRAIL_PROACTIVE_MARGIN}, {MAX_GUARDRAIL_PROACTIVE_MARGIN}], got {}",
                self.margin
            )));
        }
        if !self.hard_expand_alpha.is_finite()
            || self.hard_expand_alpha <= 0.0
            || self.hard_expand_alpha > MAX_LOAD_FACTOR
        {
            return Err(AerisError::InvalidConfig(format!(
                "hard_expand_alpha must be in (0.0, {MAX_LOAD_FACTOR}], got {}",
                self.hard_expand_alpha
            )));
        }
        if !self.contraction_alpha_threshold.is_finite()
            || self.contraction_alpha_threshold <= 0.0
            || self.contraction_alpha_threshold >= 0.50
        {
            return Err(AerisError::InvalidConfig(format!(
                "contraction_alpha_threshold must be in (0.0, 0.50), got {}",
                self.contraction_alpha_threshold
            )));
        }
        if self.contraction_sustained_checks == 0 {
            return Err(AerisError::InvalidConfig(
                "contraction_sustained_checks must be at least 1".to_string(),
            ));
        }
        if !self.max_post_contraction_alpha.is_finite()
            || self.max_post_contraction_alpha <= self.contraction_alpha_threshold
            || self.max_post_contraction_alpha >= self.hard_expand_alpha
        {
            return Err(AerisError::InvalidConfig(format!(
                "max_post_contraction_alpha ({}) must lie strictly between contraction_alpha_threshold ({}) and hard_expand_alpha ({})",
                self.max_post_contraction_alpha,
                self.contraction_alpha_threshold,
                self.hard_expand_alpha
            )));
        }
        Ok(())
    }
}

/// Pure input telemetry snapshot consumed by [`ProactiveTrigger::evaluate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriggerInput {
    /// Total Quotient Filter slot capacity `N = 2^q`.
    pub capacity_slots: usize,
    /// Active quotient bit width `q`.
    pub current_q: u8,
    /// Current Quotient Filter slot occupancy load factor `alpha = occupied / N`.
    pub current_alpha: f64,
    /// Current EWMA key ingestion rate (keys/sec).
    pub ingestion_rate_ewma: f64,
    /// Estimated number of live entries stored in the Reverse Map.
    pub reverse_map_entries: u64,
    /// Observed background bulk-load throughput (keys/sec).
    pub bulk_load_speed: f64,
}

/// Lifecycle resizing action emitted by [`ProactiveTrigger::evaluate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerAction {
    /// No capacity resize is warranted.
    None,
    /// Proactively double the filter (`q -> q + 1`) before saturation.
    ProactiveExpand,
    /// Halve the filter (`q -> q - 1`) to reclaim memory after sustained underutilization.
    Contract,
}

/// Detailed diagnostic output of a single [`ProactiveTrigger::evaluate`] call.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerEvaluation {
    /// Projected seconds until the filter reaches [`MAX_LOAD_FACTOR`] (`f64::INFINITY`
    /// when `ingestion_rate_ewma <= 0.0`).
    pub t_full_secs: f64,
    /// Estimated seconds required to rebuild from the Reverse Map.
    pub t_rebuild_secs: f64,
    /// Active safety margin multiplier `k` applied to `t_rebuild_secs`.
    pub margin: f64,
    /// Projected load factor (`current_alpha * 2.0`) if contraction (`q -> q - 1`) runs.
    pub projected_post_contraction_alpha: f64,
    /// Number of consecutive low-load observations recorded after this evaluation.
    pub low_alpha_streak: u32,
    /// Resulting resizing decision.
    pub action: TriggerAction,
}

/// Deterministic EWMA proactive expansion and contraction state machine.
#[derive(Clone, Debug)]
pub struct ProactiveTrigger {
    config: ProactiveTriggerConfig,
    low_alpha_streak: u32,
}

impl Default for ProactiveTrigger {
    fn default() -> Self {
        Self::new(ProactiveTriggerConfig::default())
            .expect("default ProactiveTriggerConfig is always valid")
    }
}

impl ProactiveTrigger {
    /// Constructs a validated [`ProactiveTrigger`] state machine.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` violates any bound.
    pub fn new(config: ProactiveTriggerConfig) -> AerisResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            low_alpha_streak: 0,
        })
    }

    /// Returns the active [`ProactiveTriggerConfig`].
    #[inline]
    #[must_use]
    pub fn config(&self) -> &ProactiveTriggerConfig {
        &self.config
    }

    /// Updates the proactive rebuild safety margin `k` after validating `[1.0, 3.0]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `margin` is outside `[1.0, 3.0]` or non-finite.
    pub fn set_margin(&mut self, margin: f64) -> AerisResult<()> {
        let mut next = self.config.clone();
        next.margin = margin;
        next.validate()?;
        self.config = next;
        debug!(margin, "updated proactive trigger safety margin");
        Ok(())
    }

    /// Updates the hard expansion ceiling `hard_expand_alpha` after validating config bounds.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `hard_expand_alpha` violates config invariants.
    pub fn set_hard_expand_alpha(&mut self, hard_expand_alpha: f64) -> AerisResult<()> {
        let mut next = self.config.clone();
        next.hard_expand_alpha = hard_expand_alpha;
        next.validate()?;
        self.config = next;
        debug!(
            hard_expand_alpha,
            "updated proactive trigger hard_expand_alpha"
        );
        Ok(())
    }

    /// Returns the current consecutive low-load observation streak.
    #[inline]
    #[must_use]
    pub const fn low_alpha_streak(&self) -> u32 {
        self.low_alpha_streak
    }

    /// Explicitly resets the contraction streak counter (e.g., after an external resize).
    #[inline]
    pub fn reset_contraction_streak(&mut self) {
        self.low_alpha_streak = 0;
    }

    /// Computes projected time-to-saturation `T_full` and rebuild time `T_rebuild`
    /// and transitions the deterministic trigger state machine.
    ///
    /// # Complexity
    /// `O(1)` time, `O(1)` space, zero allocations, zero clock calls.
    pub fn evaluate(&mut self, input: &TriggerInput) -> TriggerEvaluation {
        let alpha = if input.current_alpha.is_finite() {
            input.current_alpha.max(0.0)
        } else {
            MAX_LOAD_FACTOR
        };

        let remaining_slots = (input.capacity_slots as f64) * (MAX_LOAD_FACTOR - alpha).max(0.0);
        let t_full_secs =
            if input.ingestion_rate_ewma.is_finite() && input.ingestion_rate_ewma > 0.0 {
                remaining_slots / input.ingestion_rate_ewma
            } else {
                f64::INFINITY
            };

        let safe_bulk_speed = if input.bulk_load_speed.is_finite() {
            input.bulk_load_speed.max(MIN_BULK_LOAD_SPEED_FLOOR)
        } else {
            MIN_BULK_LOAD_SPEED_FLOOR
        };
        let t_rebuild_secs = (input.reverse_map_entries as f64) / safe_bulk_speed;
        let projected_post_contraction_alpha = alpha * 2.0;

        let proactive_expand_due = t_full_secs <= t_rebuild_secs * self.config.margin;
        let hard_expand_due = alpha >= self.config.hard_expand_alpha;

        if proactive_expand_due || hard_expand_due {
            self.low_alpha_streak = 0;
            if input.current_q >= MAX_QUOTIENT_BITS {
                warn!(
                    current_q = input.current_q,
                    alpha,
                    t_full_secs,
                    t_rebuild_secs,
                    "expansion condition reached at MAX_QUOTIENT_BITS ceiling"
                );
                return TriggerEvaluation {
                    t_full_secs,
                    t_rebuild_secs,
                    margin: self.config.margin,
                    projected_post_contraction_alpha,
                    low_alpha_streak: 0,
                    action: TriggerAction::None,
                };
            }
            info!(
                current_q = input.current_q,
                alpha,
                t_full_secs,
                t_rebuild_secs,
                margin = self.config.margin,
                proactive_expand_due,
                hard_expand_due,
                "proactive trigger fired ProactiveExpand"
            );
            return TriggerEvaluation {
                t_full_secs,
                t_rebuild_secs,
                margin: self.config.margin,
                projected_post_contraction_alpha,
                low_alpha_streak: 0,
                action: TriggerAction::ProactiveExpand,
            };
        }

        let contract_eligible = input.current_q > MIN_QUOTIENT_BITS
            && alpha <= self.config.contraction_alpha_threshold
            && projected_post_contraction_alpha <= self.config.max_post_contraction_alpha;

        if !contract_eligible {
            self.low_alpha_streak = 0;
            return TriggerEvaluation {
                t_full_secs,
                t_rebuild_secs,
                margin: self.config.margin,
                projected_post_contraction_alpha,
                low_alpha_streak: 0,
                action: TriggerAction::None,
            };
        }

        self.low_alpha_streak = self.low_alpha_streak.saturating_add(1);
        let streak = self.low_alpha_streak;

        if streak >= self.config.contraction_sustained_checks {
            self.low_alpha_streak = 0;
            info!(
                current_q = input.current_q,
                alpha,
                projected_post_contraction_alpha,
                sustained_checks = streak,
                "proactive trigger fired Contract"
            );
            return TriggerEvaluation {
                t_full_secs,
                t_rebuild_secs,
                margin: self.config.margin,
                projected_post_contraction_alpha,
                low_alpha_streak: streak,
                action: TriggerAction::Contract,
            };
        }

        debug!(
            current_q = input.current_q,
            alpha,
            streak,
            required = self.config.contraction_sustained_checks,
            "accumulating low-alpha contraction streak"
        );
        TriggerEvaluation {
            t_full_secs,
            t_rebuild_secs,
            margin: self.config.margin,
            projected_post_contraction_alpha,
            low_alpha_streak: streak,
            action: TriggerAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation_enforces_guardrail_and_anti_thrashing_bounds() {
        let base = ProactiveTriggerConfig::default();
        assert!(base.validate().is_ok());

        let cases = [
            ProactiveTriggerConfig {
                margin: 0.99,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                margin: 3.01,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                margin: f64::NAN,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                hard_expand_alpha: 0.96,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                hard_expand_alpha: 0.0,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                contraction_alpha_threshold: 0.50,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                contraction_sustained_checks: 0,
                ..base.clone()
            },
            ProactiveTriggerConfig {
                max_post_contraction_alpha: base.hard_expand_alpha,
                ..base.clone()
            },
        ];
        for bad in &cases {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn eq3_boundary_fires_proactive_expand_precisely_before_saturation() {
        let mut trigger = ProactiveTrigger::new(ProactiveTriggerConfig {
            margin: 1.5,
            hard_expand_alpha: 0.90,
            ..ProactiveTriggerConfig::default()
        })
        .unwrap();

        // N = 10_000, alpha = 0.70 -> remaining_slots = 10_000 * 0.25 = 2_500.
        // reverse_map_entries = 7_000, bulk_load_speed = 10_000 -> T_rebuild = 0.70s.
        // T_rebuild * margin = 1.05s.
        // At ingestion_rate = 2_000 keys/s -> T_full = 1.25s > 1.05s -> None.
        let safe_input = TriggerInput {
            capacity_slots: 10_000,
            current_q: 14,
            current_alpha: 0.70,
            ingestion_rate_ewma: 2_000.0,
            reverse_map_entries: 7_000,
            bulk_load_speed: 10_000.0,
        };
        let eval_safe = trigger.evaluate(&safe_input);
        assert_eq!(eval_safe.action, TriggerAction::None);
        assert!((eval_safe.t_full_secs - 1.25).abs() < 1e-9);
        assert!((eval_safe.t_rebuild_secs - 0.70).abs() < 1e-9);

        // Increase ingestion_rate so T_full = 2_500 / 2_500 = 1.00s <= 1.05s -> ProactiveExpand!
        let urgent_input = TriggerInput {
            ingestion_rate_ewma: 2_500.0,
            ..safe_input
        };
        let eval_urgent = trigger.evaluate(&urgent_input);
        assert_eq!(eval_urgent.action, TriggerAction::ProactiveExpand);
        assert!((eval_urgent.t_full_secs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_ingestion_returns_infinite_t_full_and_respects_hard_expand_alpha() {
        let mut trigger = ProactiveTrigger::default();

        let idle_low = TriggerInput {
            capacity_slots: 4096,
            current_q: 12,
            current_alpha: 0.85,
            ingestion_rate_ewma: 0.0,
            reverse_map_entries: 3400,
            bulk_load_speed: 500_000.0,
        };
        let eval_idle = trigger.evaluate(&idle_low);
        assert!(eval_idle.t_full_secs.is_infinite());
        assert_eq!(eval_idle.action, TriggerAction::None);

        let idle_hard = TriggerInput {
            current_alpha: 0.90,
            ..idle_low
        };
        let eval_hard = trigger.evaluate(&idle_hard);
        assert!(eval_hard.t_full_secs.is_infinite());
        assert_eq!(eval_hard.action, TriggerAction::ProactiveExpand);
    }

    #[test]
    fn contraction_requires_sustained_low_alpha_and_safe_projected_load() {
        let mut trigger = ProactiveTrigger::new(ProactiveTriggerConfig {
            contraction_alpha_threshold: 0.25,
            contraction_sustained_checks: 4,
            max_post_contraction_alpha: 0.65,
            ..ProactiveTriggerConfig::default()
        })
        .unwrap();

        let low = TriggerInput {
            capacity_slots: 4096,
            current_q: 12,
            current_alpha: 0.20,
            ingestion_rate_ewma: 10.0,
            reverse_map_entries: 800,
            bulk_load_speed: 500_000.0,
        };

        // 3 consecutive checks -> still None
        for expected_streak in 1..=3 {
            let eval = trigger.evaluate(&low);
            assert_eq!(eval.action, TriggerAction::None);
            assert_eq!(eval.low_alpha_streak, expected_streak);
        }

        // An interruption above threshold resets streak to 0.
        let mid = TriggerInput {
            current_alpha: 0.30,
            ..low
        };
        let eval_mid = trigger.evaluate(&mid);
        assert_eq!(eval_mid.action, TriggerAction::None);
        assert_eq!(eval_mid.low_alpha_streak, 0);

        // Now 4 consecutive low checks fire Contract on the 4th.
        for _ in 0..3 {
            assert_eq!(trigger.evaluate(&low).action, TriggerAction::None);
        }
        let fired = trigger.evaluate(&low);
        assert_eq!(fired.action, TriggerAction::Contract);
        assert!((fired.projected_post_contraction_alpha - 0.40).abs() < 1e-9);
        assert_eq!(trigger.low_alpha_streak(), 0);

        // Must never contract at MIN_QUOTIENT_BITS.
        let min_q_input = TriggerInput {
            current_q: MIN_QUOTIENT_BITS,
            ..low
        };
        for _ in 0..6 {
            assert_eq!(trigger.evaluate(&min_q_input).action, TriggerAction::None);
        }

        // Must not contract if projected post-contraction alpha (2 * alpha) exceeds max_post_contraction_alpha.
        let mut strict_trigger = ProactiveTrigger::new(ProactiveTriggerConfig {
            contraction_alpha_threshold: 0.25,
            contraction_sustained_checks: 2,
            max_post_contraction_alpha: 0.35,
            ..ProactiveTriggerConfig::default()
        })
        .unwrap();
        // alpha = 0.20 <= 0.25, but 2 * 0.20 = 0.40 > 0.35 -> blocked!
        for _ in 0..5 {
            assert_eq!(strict_trigger.evaluate(&low).action, TriggerAction::None);
        }
    }
}
