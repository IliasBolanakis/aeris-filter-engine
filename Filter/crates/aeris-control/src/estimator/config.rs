//! Configuration constants and validated [`EstimatorConfig`] for
//! [`super::SlidingWindowEstimator`].

use aeris_types::{AerisError, AerisResult, MAX_MEMENTO_BITS};
use serde::{Deserialize, Serialize};

/// Default sliding-window sample capacity for [`super::SlidingWindowEstimator`].
pub const DEFAULT_WINDOW_SIZE: usize = 4096;

/// Default minimum sample count before median-based outlier clipping activates.
pub const DEFAULT_MIN_SAMPLES_FOR_CLIPPING: usize = 64;

/// Default multiplier applied to `raw_p60` to form the outlier clipping ceiling.
pub const DEFAULT_OUTLIER_MEDIAN_MULTIPLIER: u64 = 16;

/// Default minimum clipping floor so tiny point-query medians do not clip
/// modest legitimate range queries.
pub const DEFAULT_OUTLIER_FLOOR_SPAN: u64 = 64;

/// Default hard upper bound on clipped query spans (`2^MAX_MEMENTO_BITS`).
pub const DEFAULT_MAX_CLIP_SPAN: u64 = 1u64 << MAX_MEMENTO_BITS;

/// Default hysteresis ratio guarding `effective_query_span` transitions.
pub const DEFAULT_HYSTERESIS_RATIO: f64 = 2.0;

/// Default minimum number of queries between `effective_query_span` changes.
pub const DEFAULT_COOLDOWN_QUERIES: u64 = 256;

/// Default consecutive out-of-band evaluations required before updating
/// `effective_query_span`.
pub const DEFAULT_REQUIRED_SUSTAINED_EVALS: u32 = 4;

/// Default EWMA smoothing factor (`alpha in (0, 1]`).
pub const DEFAULT_EWMA_ALPHA: f64 = 0.25;

/// Default power-of-two query-length sampling stride (`1` in `64` queries),
/// keeping the query hot path branch-light (`(seq & (stride - 1)) == 0`) and
/// zero-allocation while preserving quantile accuracy and shift detection speed.
pub const DEFAULT_SAMPLE_STRIDE: u64 = 64;

/// Upper cap on the automatic evaluation cadence stride (in queries).
const MAX_AUTO_EVAL_STRIDE: u64 = 64;

/// Serde default helper keeping `EstimatorConfig::enabled == true` for legacy configs.
const fn default_enabled() -> bool {
    true
}

/// Configuration parameters controlling [`super::SlidingWindowEstimator`].
///
/// # Invariants
/// Every configuration validated via [`EstimatorConfig::validate`] satisfies:
/// - `window_size >= 1`
/// - `outlier_median_multiplier >= 1`
/// - `outlier_floor_span >= 1`
/// - `max_clip_span >= 1`
/// - `hysteresis_ratio.is_finite() && hysteresis_ratio >= 1.0`
/// - `required_sustained_evals >= 1`
/// - `ewma_alpha.is_finite() && ewma_alpha > 0.0 && ewma_alpha <= 1.0`
/// - `sample_stride.is_power_of_two()` (e.g. `1, 2, 4, ..., 64, ...`)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimatorConfig {
    /// Maximum number of recent queries covered by the sliding window.
    pub window_size: usize,
    /// Minimum number of window queries required before median-anchored outlier
    /// clipping is enabled.
    pub min_samples_for_clipping: usize,
    /// Multiplier applied to raw window quantile (`p60`) to compute the outlier cap.
    pub outlier_median_multiplier: u64,
    /// Lower floor on the outlier clipping cap so small medians do not clip
    /// moderate range queries.
    pub outlier_floor_span: u64,
    /// Absolute upper ceiling on clipped query spans (defaults to `1 << MAX_MEMENTO_BITS`).
    pub max_clip_span: u64,
    /// Multiplicative deadband around `effective_query_span`: candidate `p95` must
    /// lie strictly outside `[eff / hysteresis_ratio, eff * hysteresis_ratio]` to
    /// count toward a transition.
    pub hysteresis_ratio: f64,
    /// Minimum number of queries that must arrive after the previous transition
    /// before another transition may begin accumulating sustained evaluations.
    pub cooldown_queries: u64,
    /// Number of consecutive out-of-band evaluations in the same direction
    /// required before `effective_query_span` updates.
    pub required_sustained_evals: u32,
    /// Exponentially Weighted Moving Average smoothing factor `alpha in (0, 1]`.
    pub ewma_alpha: f64,
    /// Power-of-two query-length sampling stride (`1` = sample every query;
    /// `64` = sample 1-in-64 queries via rotating stratified offset).
    pub sample_stride: u64,
    /// Enables or disables query-stream workload estimation (default `true`).
    /// When `false`, [`super::SlidingWindowEstimator::record_query`] executes an
    /// immediate zero-overhead early return for static-`R` deployments.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_WINDOW_SIZE,
            min_samples_for_clipping: DEFAULT_MIN_SAMPLES_FOR_CLIPPING,
            outlier_median_multiplier: DEFAULT_OUTLIER_MEDIAN_MULTIPLIER,
            outlier_floor_span: DEFAULT_OUTLIER_FLOOR_SPAN,
            max_clip_span: DEFAULT_MAX_CLIP_SPAN,
            hysteresis_ratio: DEFAULT_HYSTERESIS_RATIO,
            cooldown_queries: DEFAULT_COOLDOWN_QUERIES,
            required_sustained_evals: DEFAULT_REQUIRED_SUSTAINED_EVALS,
            ewma_alpha: DEFAULT_EWMA_ALPHA,
            sample_stride: DEFAULT_SAMPLE_STRIDE,
            enabled: true,
        }
    }
}

impl EstimatorConfig {
    /// Overrides [`Self::enabled`] (`false` activates the zero-overhead static-`R`
    /// query-recording fast-path bypass).
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Validates all numeric and structural bounds of the estimator configuration.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if any parameter is out of range,
    /// non-finite, or if `sample_stride` is not a non-zero power of two.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    pub fn validate(&self) -> AerisResult<()> {
        if self.window_size == 0 {
            return Err(AerisError::InvalidConfig(
                "estimator window_size must be at least 1".to_string(),
            ));
        }
        if self.outlier_median_multiplier == 0 {
            return Err(AerisError::InvalidConfig(
                "outlier_median_multiplier must be at least 1".to_string(),
            ));
        }
        if self.outlier_floor_span == 0 {
            return Err(AerisError::InvalidConfig(
                "outlier_floor_span must be at least 1".to_string(),
            ));
        }
        if self.max_clip_span == 0 {
            return Err(AerisError::InvalidConfig(
                "max_clip_span must be at least 1".to_string(),
            ));
        }
        if !self.hysteresis_ratio.is_finite() || self.hysteresis_ratio < 1.0 {
            return Err(AerisError::InvalidConfig(format!(
                "hysteresis_ratio must be finite and >= 1.0, got {}",
                self.hysteresis_ratio
            )));
        }
        if self.required_sustained_evals == 0 {
            return Err(AerisError::InvalidConfig(
                "required_sustained_evals must be at least 1".to_string(),
            ));
        }
        if !self.ewma_alpha.is_finite() || self.ewma_alpha <= 0.0 || self.ewma_alpha > 1.0 {
            return Err(AerisError::InvalidConfig(format!(
                "ewma_alpha must be finite and in (0.0, 1.0], got {}",
                self.ewma_alpha
            )));
        }
        if !self.sample_stride.is_power_of_two() {
            return Err(AerisError::InvalidConfig(format!(
                "sample_stride must be a non-zero power of two, got {}",
                self.sample_stride
            )));
        }
        Ok(())
    }

    /// Computes the sampled length-window capacity `max(1, window_size / sample_stride)`
    /// so regime shifts still flush the window in `window_size` queries regardless of
    /// `sample_stride`.
    #[inline]
    pub(super) fn sample_window_capacity(&self) -> usize {
        let stride = usize::try_from(self.sample_stride)
            .unwrap_or(usize::MAX)
            .max(1);
        (self.window_size / stride).max(1)
    }

    /// Computes the minimum number of sampled length observations required before
    /// median-anchored outlier clipping activates.
    #[inline]
    pub(super) fn effective_min_clip_samples(&self) -> usize {
        let cap = self.sample_window_capacity();
        if self.sample_stride == 1 {
            self.min_samples_for_clipping
        } else {
            let stride = usize::try_from(self.sample_stride)
                .unwrap_or(usize::MAX)
                .max(1);
            (self.min_samples_for_clipping / stride).max(4).min(cap)
        }
    }

    /// Computes the automatic evaluation stride (in queries) derived from
    /// `cooldown_queries` and `required_sustained_evals`.
    #[inline]
    pub(super) fn auto_eval_stride(&self) -> u64 {
        let denom = u64::from(self.required_sustained_evals.max(1));
        (self.cooldown_queries / denom).clamp(1, MAX_AUTO_EVAL_STRIDE)
    }
}
