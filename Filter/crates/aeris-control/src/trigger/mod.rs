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

#[cfg(test)]
mod tests;

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

/// Intermediate pure timing metrics computed for a single [`TriggerInput`].
struct TriggerTimings {
    alpha: f64,
    t_full_secs: f64,
    t_rebuild_secs: f64,
    projected_post_contraction_alpha: f64,
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
        let timings = compute_trigger_timings(input);

        if let Some((action, streak)) = self.evaluate_expansion(input, &timings) {
            return self.make_evaluation(&timings, streak, action);
        }

        let (action, streak) = self.evaluate_contraction(input, &timings);
        self.make_evaluation(&timings, streak, action)
    }

    fn evaluate_expansion(
        &mut self,
        input: &TriggerInput,
        timings: &TriggerTimings,
    ) -> Option<(TriggerAction, u32)> {
        let proactive_expand_due =
            timings.t_full_secs <= timings.t_rebuild_secs * self.config.margin;
        let hard_expand_due = timings.alpha >= self.config.hard_expand_alpha;

        if !proactive_expand_due && !hard_expand_due {
            return None;
        }

        self.low_alpha_streak = 0;
        if input.current_q >= MAX_QUOTIENT_BITS {
            warn!(
                current_q = input.current_q,
                alpha = timings.alpha,
                t_full_secs = timings.t_full_secs,
                t_rebuild_secs = timings.t_rebuild_secs,
                "expansion condition reached at MAX_QUOTIENT_BITS ceiling"
            );
            return Some((TriggerAction::None, 0));
        }

        info!(
            current_q = input.current_q,
            alpha = timings.alpha,
            t_full_secs = timings.t_full_secs,
            t_rebuild_secs = timings.t_rebuild_secs,
            margin = self.config.margin,
            proactive_expand_due,
            hard_expand_due,
            "proactive trigger fired ProactiveExpand"
        );
        Some((TriggerAction::ProactiveExpand, 0))
    }

    fn evaluate_contraction(
        &mut self,
        input: &TriggerInput,
        timings: &TriggerTimings,
    ) -> (TriggerAction, u32) {
        let contract_eligible = input.current_q > MIN_QUOTIENT_BITS
            && timings.alpha <= self.config.contraction_alpha_threshold
            && timings.projected_post_contraction_alpha <= self.config.max_post_contraction_alpha;

        if !contract_eligible {
            self.low_alpha_streak = 0;
            return (TriggerAction::None, 0);
        }

        self.low_alpha_streak = self.low_alpha_streak.saturating_add(1);
        let streak = self.low_alpha_streak;

        if streak >= self.config.contraction_sustained_checks {
            self.low_alpha_streak = 0;
            info!(
                current_q = input.current_q,
                alpha = timings.alpha,
                projected_post_contraction_alpha = timings.projected_post_contraction_alpha,
                sustained_checks = streak,
                "proactive trigger fired Contract"
            );
            return (TriggerAction::Contract, streak);
        }

        debug!(
            current_q = input.current_q,
            alpha = timings.alpha,
            streak,
            required = self.config.contraction_sustained_checks,
            "accumulating low-alpha contraction streak"
        );
        (TriggerAction::None, streak)
    }

    #[inline]
    const fn make_evaluation(
        &self,
        timings: &TriggerTimings,
        low_alpha_streak: u32,
        action: TriggerAction,
    ) -> TriggerEvaluation {
        TriggerEvaluation {
            t_full_secs: timings.t_full_secs,
            t_rebuild_secs: timings.t_rebuild_secs,
            margin: self.config.margin,
            projected_post_contraction_alpha: timings.projected_post_contraction_alpha,
            low_alpha_streak,
            action,
        }
    }
}

fn compute_trigger_timings(input: &TriggerInput) -> TriggerTimings {
    let alpha = if input.current_alpha.is_finite() {
        input.current_alpha.max(0.0)
    } else {
        MAX_LOAD_FACTOR
    };

    let remaining_slots = (input.capacity_slots as f64) * (MAX_LOAD_FACTOR - alpha).max(0.0);
    let t_full_secs = if input.ingestion_rate_ewma.is_finite() && input.ingestion_rate_ewma > 0.0 {
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

    TriggerTimings {
        alpha,
        t_full_secs,
        t_rebuild_secs,
        projected_post_contraction_alpha,
    }
}
