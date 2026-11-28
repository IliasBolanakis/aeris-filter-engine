//! Hysteresis span-transition evaluation and EWMA rate helpers for
//! [`super::SlidingWindowEstimator`].

use super::quantile::{compute_clipped_quantiles, ClippedQuantiles};
use super::{BreachDirection, SlidingWindowEstimator};
use tracing::{info, warn};

impl SlidingWindowEstimator {
    /// Updates the EWMA estimate of query arrival rate (`queries / elapsed_secs`).
    ///
    /// Non-finite or non-positive `elapsed_secs` values are rejected with a
    /// structured `tracing::warn!` so EWMA state is never poisoned by `NaN`/`Inf`.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    pub fn record_query_rate(&mut self, query_count: u64, elapsed_secs: f64) {
        if !elapsed_secs.is_finite() || elapsed_secs <= 0.0 {
            warn!(
                query_count,
                elapsed_secs, "ignoring invalid elapsed_secs in record_query_rate"
            );
            return;
        }
        let sample_rate = (query_count as f64) / elapsed_secs;
        self.query_rate = update_ewma(
            self.query_rate,
            sample_rate,
            self.config.ewma_alpha,
            &mut self.has_query_rate_sample,
        );
    }

    /// Sets the average number of live keys per non-empty partition (`mu`).
    ///
    /// Non-finite or negative values are ignored with a `tracing::warn!`.
    pub fn set_avg_keys_per_partition(&mut self, avg_keys_per_partition: f64) {
        if !avg_keys_per_partition.is_finite() || avg_keys_per_partition < 0.0 {
            warn!(
                avg_keys_per_partition,
                "ignoring invalid avg_keys_per_partition"
            );
            return;
        }
        self.avg_keys_per_partition = avg_keys_per_partition;
    }

    /// Runs a single hysteresis/cooldown evaluation pass against the current
    /// outlier-clipped window quantiles, returning `true` if `effective_query_span`
    /// transitioned.
    ///
    /// # Transition Rules
    /// 1. If the window is under-populated or `self.queries_since_last_change() < self.config.cooldown_queries`,
    ///    resets any pending breach streak and returns `false`.
    /// 2. Computes (or reuses cached) candidate `p95` over the outlier-clipped window distribution.
    /// 3. Checks whether `p95` lies strictly outside
    ///    `[eff / hysteresis_ratio, eff * hysteresis_ratio]`.
    /// 4. Transitions `effective_query_span = p95` only after
    ///    `config.required_sustained_evals` consecutive evaluations in the same direction.
    ///
    /// # Complexity
    /// `O(1)` when quantiles are clean and in-band, otherwise `O(log W)` time and `O(1)` space.
    pub fn evaluate_span_transition(&mut self) -> bool {
        if self.eval_in_band_clean {
            return false;
        }
        if self.window_len < self.min_eval_samples
            || self.queries_since_last_change() < self.config.cooldown_queries
        {
            self.sustained_direction = None;
            self.sustained_evals = 0;
            return false;
        }

        let quantiles = self.refresh_cached_quantiles();
        let candidate = quantiles.p95.max(1);
        let Some(direction) = self.detect_breach_direction(candidate) else {
            self.sustained_direction = None;
            self.sustained_evals = 0;
            self.eval_in_band_clean = true;
            return false;
        };

        if self.sustained_direction == Some(direction) {
            self.sustained_evals = self.sustained_evals.saturating_add(1);
        } else {
            self.sustained_direction = Some(direction);
            self.sustained_evals = 1;
        }

        if self.sustained_evals < self.config.required_sustained_evals {
            return false;
        }

        self.commit_span_transition(candidate, quantiles);
        true
    }

    #[inline]
    fn refresh_cached_quantiles(&mut self) -> ClippedQuantiles {
        if self.quantiles_dirty {
            self.cached_quantiles = compute_clipped_quantiles(
                &self.sorted_lengths,
                &self.config,
                self.effective_min_clip_samples,
            );
            self.quantiles_dirty = false;
        }
        self.cached_quantiles
    }

    #[inline]
    fn detect_breach_direction(&self, candidate: u64) -> Option<BreachDirection> {
        let eff = self.effective_query_span.max(1) as f64;
        let cand_f = candidate as f64;
        let ratio = self.config.hysteresis_ratio;
        if cand_f > eff * ratio {
            Some(BreachDirection::Up)
        } else if cand_f < eff / ratio {
            Some(BreachDirection::Down)
        } else {
            None
        }
    }

    fn commit_span_transition(&mut self, candidate: u64, quantiles: ClippedQuantiles) {
        let old_span = self.effective_query_span;
        self.effective_query_span = candidate;
        self.last_change_seq = self.hot.total_queries;
        self.last_eval_seq = self.hot.total_queries;
        self.sustained_direction = None;
        self.sustained_evals = 0;
        self.eval_in_band_clean = false;
        self.span_transitions = self.span_transitions.saturating_add(1);

        info!(
            old_span,
            new_span = self.effective_query_span,
            p50 = quantiles.p50,
            p95 = quantiles.p95,
            p99 = quantiles.p99,
            clipped_outliers = quantiles.clipped_outliers,
            transitions = self.span_transitions,
            "transitioned effective_query_span after cooldown and hysteresis confirmation"
        );
    }
}

/// Computes an odd golden-ratio phase increment coprime to a power-of-two `sample_stride`,
/// ensuring stratified block offsets visit every residue `0..sample_stride` uniformly
/// without aliasing with periodic query bursts.
#[inline]
pub(super) fn golden_odd_phase_step(sample_stride: u64) -> u64 {
    if sample_stride <= 1 {
        return 0;
    }
    let approx = ((sample_stride as f64) * 0.618_033_988_749_894_9).round() as u64;
    (approx | 1) & (sample_stride - 1)
}

/// Pure EWMA update helper initializing directly from the first valid sample.
#[inline]
pub(super) fn update_ewma(current: f64, sample: f64, alpha: f64, initialized: &mut bool) -> f64 {
    if !*initialized {
        *initialized = true;
        sample
    } else {
        alpha * sample + (1.0 - alpha) * current
    }
}
