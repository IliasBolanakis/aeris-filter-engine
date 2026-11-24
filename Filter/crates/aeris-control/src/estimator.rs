//! Sliding-window workload quantile estimator with outlier clipping,
//! hysteresis, cooldown, and EWMA rate tracking.
//!
//! # Poisoning Resistance & Breakdown Point
//! Sampled query range lengths are retained in a bounded sliding window covering
//! `W = config.window_size` queries (`max(1, W / sample_stride)` sampled entries).
//! Whenever at least the minimum clipping sample count is present, the estimator
//! anchors outlier detection on the 60th-percentile raw quantile (`raw_p60`),
//! providing a 40% statistical breakdown point against adversarial poison bursts
//! while remaining immune to single-sample jitter on balanced 50/50 bimodal traffic:
//!
//! ```text
//! outlier_cap = max(raw_p60 * outlier_median_multiplier, outlier_floor_span)
//! clip_cap    = min(outlier_cap, max_clip_span)
//! ```
//!
//! Samples exceeding `outlier_cap` are excluded from the inlier quantile
//! distribution as adversarial/anomalous spikes, and surviving inliers are
//! bounded by `max_clip_span`. Consequently:
//! - Adversarial outlier bursts (< 40% of the window, e.g. 5–20% `10^7`-span
//!   queries injected into `R = 32` traffic) are clipped without shifting
//!   `p95` or `effective_query_span`.
//! - Genuine workload regime shifts and balanced 50/50 bimodal workloads do not
//!   suffer discontinuous clipping cliffs under `sample_stride = 64` sampling.
//!
//! # Purity & Determinism
//! This module performs zero disk I/O and reads zero system clocks. All rate
//! updates receive explicit caller-supplied durations so the estimator executes
//! identically in production and deterministic simulation tests.

use aeris_types::{
    AerisError, AerisResult, QueryRange, WorkloadEstimator, WorkloadStats, MAX_MEMENTO_BITS,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, trace, warn};

/// Default sliding-window sample capacity for [`SlidingWindowEstimator`].
pub const DEFAULT_WINDOW_SIZE: usize = 4096;

/// Default minimum sample count before median-based outlier clipping activates.
pub const DEFAULT_MIN_SAMPLES_FOR_CLIPPING: usize = 64;

/// Default multiplier applied to `raw_p50` to form the outlier clipping ceiling.
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

/// Configuration parameters controlling [`SlidingWindowEstimator`].
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
    /// Multiplier applied to raw window median (`p50`) to compute the outlier cap.
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
    /// `64` = sample 1-in-64 queries via cheap bitmask `(seq & (stride - 1)) == 0`).
    pub sample_stride: u64,
    /// Enables or disables query-stream workload estimation (default `true`).
    /// When `false`, [`SlidingWindowEstimator::record_query`] executes an immediate
    /// zero-overhead early return for static-`R` deployments.
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
    fn sample_window_capacity(&self) -> usize {
        let stride = usize::try_from(self.sample_stride)
            .unwrap_or(usize::MAX)
            .max(1);
        (self.window_size / stride).max(1)
    }

    /// Computes the minimum number of sampled length observations required before
    /// median-anchored outlier clipping activates.
    #[inline]
    fn effective_min_clip_samples(&self) -> usize {
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
    fn auto_eval_stride(&self) -> u64 {
        let denom = u64::from(self.required_sustained_evals.max(1));
        (self.cooldown_queries / denom).clamp(1, MAX_AUTO_EVAL_STRIDE)
    }
}

/// Direction in which candidate `p95` has breached the hysteresis deadband.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BreachDirection {
    /// Candidate `p95 > effective_query_span * hysteresis_ratio`.
    Up,
    /// Candidate `p95 < effective_query_span / hysteresis_ratio`.
    Down,
}

/// Exact quantile summary over the outlier-clipped sliding window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClippedQuantiles {
    p50: u64,
    p95: u64,
    p99: u64,
    clipped_outliers: u64,
}

impl ClippedQuantiles {
    /// Safe empty-window default quantiles (`p50 = p95 = p99 = 1`, `0` clipped outliers).
    const EMPTY: Self = Self {
        p50: 1,
        p95: 1,
        p99: 1,
        clipped_outliers: 0,
    };
}

/// Cache-line-colocated per-query hot counters touched on every [`WorkloadEstimator::record_query`]
/// call. Keeping these fields contiguous at the head of [`SlidingWindowEstimator`] ensures
/// the static-`R` disabled bypass and non-sampled, non-FP fast path touch only a single L1
/// cache line.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HotQueryCounters {
    /// Cumulative queries recorded across the lifetime of the estimator.
    total_queries: u64,
    /// Exact sequence number of the next query to sample (single compare on hot path).
    next_sample_seq: u64,
    /// Hot-path mirror of `config.enabled` kept in sync by `with_initial_span` and `set_enabled`.
    enabled: bool,
}

/// Deterministic sliding-window workload estimator implementing [`WorkloadEstimator`].
///
/// Uses stratified power-of-two sampling (one query per `sample_stride`-query block
/// at a golden-ratio rotating offset via `sample_mask = sample_stride - 1`) to record
/// query range lengths into a compact sorted multiset (`<= max(1, window_size / sample_stride)`
/// entries), keeping the per-query hot path to a single counter increment and comparison
/// with zero heap allocations while preventing periodic workload aliasing.
///
/// # Invariants
/// - `self.window_len == self.sorted_lengths.len() <= self.window_ring.len()`.
/// - `self.sorted_lengths` is always sorted in non-decreasing order and equals
///   the exact multiset of sampled query lengths stored in the active slots of
///   `self.window_ring`.
/// - When `!self.quantiles_dirty`, `self.cached_quantiles` equals
///   `compute_clipped_quantiles(&self.sorted_lengths, &self.config, self.effective_min_clip_samples)`.
/// - When `self.eval_in_band_clean`, `!self.quantiles_dirty`, `self.sustained_direction.is_none()`,
///   `self.sustained_evals == 0`, and `self.cached_quantiles.p95.max(1)` lies within the
///   hysteresis deadband around `self.effective_query_span`.
/// - `self.fp_query_seqs` stores strictly increasing query sequence numbers of
///   false positives; only entries within `[total_queries - window_size, total_queries)`
///   contribute to `observed_fpr`.
/// - `self.effective_query_span >= 1` always holds.
#[derive(Clone, Debug)]
pub struct SlidingWindowEstimator {
    /// Per-query hot counters kept contiguous on a single cache line.
    hot: HotQueryCounters,
    /// Query sequence number at which the last `effective_query_span` transition occurred.
    last_change_seq: u64,
    /// Query sequence number at which the last automatic hysteresis evaluation occurred.
    last_eval_seq: u64,
    /// Power-of-two bitmask `config.sample_stride - 1` for stratified block offsets.
    sample_mask: u64,
    /// Odd golden-ratio phase increment coprime to `config.sample_stride` to eliminate
    /// periodic workload aliasing (e.g., alternating or bursty fixed-period traffic).
    sample_phase_step: u64,
    /// Current rotating offset inside the active `sample_stride`-query block.
    sample_phase: u64,
    /// Precomputed automatic evaluation stride in queries (`config.auto_eval_stride()`).
    auto_eval_stride: u64,
    /// Precomputed sliding window size as `u64` (`config.window_size as u64`).
    window_size_u64: u64,
    /// Precomputed minimum sampled entries required before automatic span transitions.
    min_eval_samples: usize,
    /// Precomputed minimum sampled entries required before outlier clipping activates.
    effective_min_clip_samples: usize,
    /// Fixed-capacity ring buffer storing sampled query lengths (FIFO eviction).
    window_ring: Box<[u64]>,
    /// Number of populated entries in `window_ring` (`0..=window_ring.len()`).
    window_len: usize,
    /// Next eviction/overwrite slot in `window_ring` once full (`0..window_ring.len()`).
    window_head: usize,
    /// Exact sorted multiset mirroring active `window_ring` lengths for quantile queries.
    sorted_lengths: Vec<u64>,
    /// Cached outlier-clipped quantiles for the current `sorted_lengths` multiset.
    cached_quantiles: ClippedQuantiles,
    /// `true` iff `sorted_lengths` changed since `cached_quantiles` was last computed.
    quantiles_dirty: bool,
    /// `true` iff `evaluate_span_transition` already confirmed the current multiset lies
    /// strictly inside the hysteresis deadband for `effective_query_span`.
    eval_in_band_clean: bool,
    /// Monotonic sequence numbers of false-positive queries (pushed only on FPs).
    fp_query_seqs: VecDeque<u64>,
    /// Hysteresis-stabilized effective query span `R_eff`.
    effective_query_span: u64,
    /// Current direction of consecutive hysteresis deadband breaches, if any.
    sustained_direction: Option<BreachDirection>,
    /// Count of consecutive evaluations breaching the deadband in `sustained_direction`.
    sustained_evals: u32,
    /// Total number of `effective_query_span` transitions performed.
    span_transitions: u64,
    /// EWMA key ingestion rate (keys/sec).
    ingestion_rate_ewma: f64,
    has_ingest_sample: bool,
    /// EWMA bulk-load throughput (keys/sec).
    bulk_load_speed: f64,
    has_bulk_load_sample: bool,
    /// EWMA query arrival rate (queries/sec).
    query_rate: f64,
    has_query_rate_sample: bool,
    /// Average keys per non-empty partition (`mu`).
    avg_keys_per_partition: f64,
    /// Immutable validated estimator configuration.
    config: EstimatorConfig,
}

impl Default for SlidingWindowEstimator {
    fn default() -> Self {
        Self::new(EstimatorConfig::default()).expect("default EstimatorConfig is always valid")
    }
}

impl SlidingWindowEstimator {
    /// Constructs a new [`SlidingWindowEstimator`] with validated `config` and
    /// an initial `effective_query_span` of `1`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` violates any invariant.
    ///
    /// # Complexity
    /// `O(window_size / sample_stride)` pre-allocated capacity, `O(1)` time.
    pub fn new(config: EstimatorConfig) -> AerisResult<Self> {
        Self::with_initial_span(config, 1)
    }

    /// Constructs a new [`SlidingWindowEstimator`] initialized to `initial_span`
    /// (clamped to `[1, config.max_clip_span]`), preventing unnecessary warmup
    /// transitions when starting alongside an existing [`aeris_types::FilterConfig`].
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` violates any invariant.
    ///
    /// # Complexity
    /// `O(window_size / sample_stride)` pre-allocated capacity, `O(1)` time.
    pub fn with_initial_span(config: EstimatorConfig, initial_span: u64) -> AerisResult<Self> {
        config.validate()?;
        let clamped_span = initial_span.clamp(1, config.max_clip_span);
        let sample_mask = config.sample_stride - 1;
        let sample_phase_step = golden_odd_phase_step(config.sample_stride);
        let sample_window_cap = config.sample_window_capacity();
        let auto_eval_stride = config.auto_eval_stride();
        let min_eval_samples = (sample_window_cap / 2).clamp(1, 16);
        let effective_min_clip_samples = config.effective_min_clip_samples();
        let window_size_u64 = config.window_size as u64;
        debug!(
            window_size = config.window_size,
            sample_stride = config.sample_stride,
            sample_window_cap,
            initial_span = clamped_span,
            hysteresis_ratio = config.hysteresis_ratio,
            cooldown_queries = config.cooldown_queries,
            required_sustained_evals = config.required_sustained_evals,
            "initialized SlidingWindowEstimator"
        );
        Ok(Self {
            hot: HotQueryCounters {
                total_queries: 0,
                next_sample_seq: 0,
                enabled: config.enabled,
            },
            last_change_seq: 0,
            last_eval_seq: 0,
            sample_mask,
            sample_phase_step,
            sample_phase: 0,
            auto_eval_stride,
            window_size_u64,
            min_eval_samples,
            effective_min_clip_samples,
            window_ring: vec![0u64; sample_window_cap].into_boxed_slice(),
            window_len: 0,
            window_head: 0,
            sorted_lengths: Vec::with_capacity(sample_window_cap),
            cached_quantiles: ClippedQuantiles::EMPTY,
            quantiles_dirty: false,
            eval_in_band_clean: false,
            fp_query_seqs: VecDeque::new(),
            effective_query_span: clamped_span,
            sustained_direction: None,
            sustained_evals: 0,
            span_transitions: 0,
            ingestion_rate_ewma: 0.0,
            has_ingest_sample: false,
            bulk_load_speed: WorkloadStats::default().bulk_load_speed,
            has_bulk_load_sample: false,
            query_rate: 0.0,
            has_query_rate_sample: false,
            avg_keys_per_partition: WorkloadStats::default().avg_keys_per_partition,
            config,
        })
    }

    /// Returns the active [`EstimatorConfig`].
    #[inline]
    #[must_use]
    pub fn config(&self) -> &EstimatorConfig {
        &self.config
    }

    /// Enables or disables query-stream recording in [`WorkloadEstimator::record_query`].
    ///
    /// Setting `enabled = false` activates the zero-overhead static-`R` fast-path bypass
    /// without discarding previously collected EWMA or quantile state.
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
        self.hot.enabled = enabled;
    }

    /// Returns `true` if query-stream workload estimation is currently active.
    #[inline]
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Returns the total number of queries recorded across the lifetime of the estimator.
    #[inline]
    #[must_use]
    pub const fn total_queries(&self) -> u64 {
        self.hot.total_queries
    }

    /// Returns the number of queries recorded since the last `effective_query_span` transition.
    #[inline]
    #[must_use]
    pub const fn queries_since_last_change(&self) -> u64 {
        self.hot.total_queries.saturating_sub(self.last_change_seq)
    }

    /// Returns the number of queries recorded since the last automatic hysteresis evaluation.
    #[inline]
    const fn queries_since_last_eval(&self) -> u64 {
        self.hot.total_queries.saturating_sub(self.last_eval_seq)
    }

    /// Returns the cumulative number of `effective_query_span` transitions performed.
    #[inline]
    #[must_use]
    pub const fn span_transitions(&self) -> u64 {
        self.span_transitions
    }

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

        if self.quantiles_dirty {
            self.cached_quantiles = compute_clipped_quantiles(
                &self.sorted_lengths,
                &self.config,
                self.effective_min_clip_samples,
            );
            self.quantiles_dirty = false;
        }

        let quantiles = self.cached_quantiles;
        let candidate = quantiles.p95.max(1);
        let eff = self.effective_query_span.max(1) as f64;
        let cand_f = candidate as f64;
        let ratio = self.config.hysteresis_ratio;

        let breach = if cand_f > eff * ratio {
            Some(BreachDirection::Up)
        } else if cand_f < eff / ratio {
            Some(BreachDirection::Down)
        } else {
            None
        };

        let Some(direction) = breach else {
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
        true
    }

    /// Inserts a sampled query length into the fixed-capacity ring buffer and
    /// compact sorted multiset while evicting the oldest sampled length if full.
    ///
    /// Short-circuits in `O(1)` with zero multiset mutations or cache invalidations
    /// whenever the evicted length equals `query_len` (steady-state workloads).
    #[inline]
    fn push_window_sample(&mut self, query_len: u64) {
        let cap = self.window_ring.len();
        if self.window_len == cap {
            let head = self.window_head;
            let evicted_len = self.window_ring[head];
            let next_head = head + 1;
            self.window_head = if next_head == cap { 0 } else { next_head };

            if evicted_len == query_len {
                return;
            }

            self.window_ring[head] = query_len;
            replace_sorted_sample(&mut self.sorted_lengths, evicted_len, query_len);
            self.quantiles_dirty = true;
            self.eval_in_band_clean = false;
            return;
        }

        self.window_ring[self.window_len] = query_len;
        self.window_len += 1;
        let insert_idx = self.sorted_lengths.partition_point(|&v| v < query_len);
        self.sorted_lengths.insert(insert_idx, query_len);
        self.quantiles_dirty = true;
        self.eval_in_band_clean = false;
    }

    /// Cold path invoked strictly on false positives to maintain exact sliding-window
    /// FPR with zero overhead on negative queries.
    #[cold]
    #[inline(never)]
    fn record_fp_seq(&mut self, seq: u64) {
        let cutoff = self.hot.total_queries.saturating_sub(self.window_size_u64);
        while let Some(&front) = self.fp_query_seqs.front() {
            if front < cutoff {
                self.fp_query_seqs.pop_front();
            } else {
                break;
            }
        }
        self.fp_query_seqs.push_back(seq);
    }

    /// Unified cold path invoked only on false positives or once per `sample_stride`
    /// queries, minimizing instruction-cache footprint and register pressure in the
    /// inlined [`WorkloadEstimator::record_query`] hot loop.
    #[cold]
    #[inline(never)]
    fn record_query_slow(
        &mut self,
        seq: u64,
        range_start: u64,
        range_end: u64,
        is_false_positive: bool,
    ) {
        if is_false_positive {
            self.record_fp_seq(seq);
        }
        if seq == self.hot.next_sample_seq {
            let len = range_end.wrapping_sub(range_start).saturating_add(1);
            self.record_sampled_query_slow(seq, len);
        }
    }

    /// Sampled path invoked once per `config.sample_stride`-query block at a
    /// golden-ratio rotating offset to update the sliding-window quantile multiset
    /// and trigger hysteresis evaluations without periodic aliasing.
    #[inline]
    fn record_sampled_query_slow(&mut self, seq: u64, len: u64) {
        self.sample_phase =
            self.sample_phase.wrapping_add(self.sample_phase_step) & self.sample_mask;
        let next_block = (seq & !self.sample_mask).wrapping_add(self.config.sample_stride);
        self.hot.next_sample_seq = next_block.wrapping_add(self.sample_phase);

        self.push_window_sample(len);

        trace!(
            query_len = len,
            sampled_window_len = self.window_len,
            sample_stride = self.config.sample_stride,
            "recorded sampled query length"
        );

        if self.eval_in_band_clean {
            if self.config.sample_stride < self.auto_eval_stride
                && self.queries_since_last_change() >= self.config.cooldown_queries
                && self.queries_since_last_eval() >= self.auto_eval_stride
            {
                self.last_eval_seq = self.hot.total_queries;
            }
            return;
        }

        if self.queries_since_last_change() >= self.config.cooldown_queries
            && (self.queries_since_last_eval() >= self.auto_eval_stride
                || self.config.sample_stride >= self.auto_eval_stride)
        {
            self.last_eval_seq = self.hot.total_queries;
            let _ = self.evaluate_span_transition();
        }
    }

    /// Counts false positives recorded within the active `config.window_size`-query
    /// sliding window in `O(1)` (fast paths) or `O(log F)` time without mutating state.
    #[inline]
    fn active_window_fp_count(&self) -> u64 {
        let len = self.fp_query_seqs.len();
        if len == 0 {
            return 0;
        }
        let cutoff = self.hot.total_queries.saturating_sub(self.window_size_u64);
        if let Some(&front) = self.fp_query_seqs.front() {
            if front >= cutoff {
                return len as u64;
            }
        }
        if let Some(&back) = self.fp_query_seqs.back() {
            if back < cutoff {
                return 0;
            }
        }
        let expired = self.fp_query_seqs.partition_point(|&seq| seq < cutoff);
        (len.saturating_sub(expired)) as u64
    }
}

impl WorkloadEstimator for SlidingWindowEstimator {
    #[inline(always)]
    fn record_query(&mut self, range: QueryRange, is_false_positive: bool) {
        if !self.hot.enabled {
            return;
        }
        let seq = self.hot.total_queries;
        self.hot.total_queries = seq.wrapping_add(1);

        if (seq == self.hot.next_sample_seq) | is_false_positive {
            self.record_query_slow(seq, range.start(), range.end(), is_false_positive);
        }
    }

    fn record_ingest(&mut self, count: u64, elapsed_secs: f64) {
        if !elapsed_secs.is_finite() || elapsed_secs <= 0.0 {
            warn!(
                count,
                elapsed_secs, "ignoring invalid elapsed_secs in record_ingest"
            );
            return;
        }
        let inst_rate = (count as f64) / elapsed_secs;
        self.ingestion_rate_ewma = update_ewma(
            self.ingestion_rate_ewma,
            inst_rate,
            self.config.ewma_alpha,
            &mut self.has_ingest_sample,
        );
        debug!(
            count,
            elapsed_secs,
            inst_rate,
            ingestion_rate_ewma = self.ingestion_rate_ewma,
            "updated ingestion rate EWMA"
        );
    }

    fn record_bulk_load_speed(&mut self, keys_per_sec: f64) {
        if !keys_per_sec.is_finite() || keys_per_sec <= 0.0 {
            warn!(
                keys_per_sec,
                "ignoring invalid keys_per_sec in record_bulk_load_speed"
            );
            return;
        }
        self.bulk_load_speed = update_ewma(
            self.bulk_load_speed,
            keys_per_sec,
            self.config.ewma_alpha,
            &mut self.has_bulk_load_sample,
        );
        debug!(
            keys_per_sec,
            bulk_load_speed_ewma = self.bulk_load_speed,
            "updated bulk load speed EWMA"
        );
    }

    fn snapshot(&self) -> WorkloadStats {
        let window_count = self.hot.total_queries.min(self.window_size_u64);
        let active_fps = self.active_window_fp_count();
        let observed_fpr = if window_count == 0 {
            0.0
        } else {
            (active_fps as f64) / (window_count as f64)
        };
        let quantiles = if self.quantiles_dirty {
            compute_clipped_quantiles(
                &self.sorted_lengths,
                &self.config,
                self.effective_min_clip_samples,
            )
        } else {
            self.cached_quantiles
        };
        let estimated_clipped = quantiles
            .clipped_outliers
            .saturating_mul(self.config.sample_stride)
            .min(window_count);
        WorkloadStats {
            window_query_count: window_count,
            observed_fpr,
            p50_query_len: quantiles.p50,
            p95_query_len: quantiles.p95,
            p99_query_len: quantiles.p99,
            effective_query_span: self.effective_query_span,
            clipped_outliers: estimated_clipped,
            ingestion_rate_ewma: self.ingestion_rate_ewma,
            bulk_load_speed: self.bulk_load_speed,
            query_rate: self.query_rate,
            avg_keys_per_partition: self.avg_keys_per_partition,
        }
    }

    #[inline]
    fn effective_query_span(&self) -> u64 {
        self.effective_query_span
    }

    #[inline]
    fn ingestion_rate_ewma(&self) -> f64 {
        self.ingestion_rate_ewma
    }
}

/// Replaces one occurrence of `evicted_len` with `new_len` inside the non-decreasing
/// sorted multiset `sorted` using at most a single contiguous `copy_within` shift
/// instead of separate `Vec::remove` and `Vec::insert` passes.
///
/// # Preconditions
/// `sorted` MUST be sorted in non-decreasing order and `evicted_len != new_len`.
///
/// # Complexity
/// `O(log N + |new_idx - old_idx|)` time and `O(1)` auxiliary space.
#[inline]
fn replace_sorted_sample(sorted: &mut Vec<u64>, evicted_len: u64, new_len: u64) {
    if let Ok(old_idx) = sorted.binary_search(&evicted_len) {
        if new_len > evicted_len {
            let rel = sorted[old_idx + 1..].partition_point(|&v| v < new_len);
            let new_idx = old_idx + rel;
            if new_idx > old_idx {
                sorted.copy_within((old_idx + 1)..=new_idx, old_idx);
            }
            sorted[new_idx] = new_len;
        } else {
            let new_idx = sorted[..old_idx].partition_point(|&v| v < new_len);
            if new_idx < old_idx {
                sorted.copy_within(new_idx..old_idx, new_idx + 1);
            }
            sorted[new_idx] = new_len;
        }
    } else {
        // Defensive fallback (unreachable while ring buffer mirrors `sorted`).
        let insert_idx = sorted.partition_point(|&v| v < new_len);
        sorted.insert(insert_idx, new_len);
    }
}

/// Computes an odd golden-ratio phase increment coprime to a power-of-two `sample_stride`,
/// ensuring stratified block offsets visit every residue `0..sample_stride` uniformly
/// without aliasing with periodic query bursts.
#[inline]
fn golden_odd_phase_step(sample_stride: u64) -> u64 {
    if sample_stride <= 1 {
        return 0;
    }
    let approx = ((sample_stride as f64) * 0.618_033_988_749_894_9).round() as u64;
    (approx | 1) & (sample_stride - 1)
}

/// Pure EWMA update helper initializing directly from the first valid sample.
#[inline]
fn update_ewma(current: f64, sample: f64, alpha: f64, initialized: &mut bool) -> f64 {
    if !*initialized {
        *initialized = true;
        sample
    } else {
        alpha * sample + (1.0 - alpha) * current
    }
}

/// Computes exact quantiles over `sorted` after applying 40%-breakdown `raw_p60`-anchored
/// outlier clipping and `max_clip_span` capping.
///
/// # Preconditions
/// `sorted` MUST be sorted in non-decreasing order.
///
/// # Complexity
/// `O(1)` when no outliers exceed `clip_cap`, otherwise `O(log N)` time and `O(1)` space.
fn compute_clipped_quantiles(
    sorted: &[u64],
    config: &EstimatorConfig,
    min_clip_samples: usize,
) -> ClippedQuantiles {
    let n = sorted.len();
    if n == 0 {
        return ClippedQuantiles::EMPTY;
    }

    let max_val = sorted[n - 1];
    let (inliers, clipped_outliers) = if n >= min_clip_samples {
        let raw_p60 = nearest_rank_quantile_ratio(sorted, 3, 5);
        let outlier_cap = raw_p60
            .saturating_mul(config.outlier_median_multiplier)
            .max(config.outlier_floor_span);
        let clip_cap = outlier_cap.min(config.max_clip_span);

        if max_val <= clip_cap {
            // Fast path: all samples lie within both `clip_cap` and `outlier_cap` (zero binary searches).
            (sorted, 0)
        } else {
            // Statistical outliers strictly above `outlier_cap` are excluded from the inlier
            // slice. Because `raw_p60 <= outlier_cap`, `within_outlier_cap >= ceil(0.60 * n) >= 1`.
            let within_outlier_cap = sorted.partition_point(|&v| v <= outlier_cap);
            let inlier_count = within_outlier_cap.max(1);
            let within_clip_cap = if clip_cap == outlier_cap {
                within_outlier_cap
            } else {
                sorted[..within_outlier_cap].partition_point(|&v| v <= clip_cap)
            };
            let clipped = (n - within_clip_cap) as u64;
            (&sorted[..inlier_count], clipped)
        }
    } else if max_val <= config.max_clip_span {
        (sorted, 0)
    } else {
        let within_max = sorted.partition_point(|&v| v <= config.max_clip_span);
        let clipped = (n - within_max) as u64;
        (sorted, clipped)
    };

    let cap = config.max_clip_span;
    ClippedQuantiles {
        p50: exact_median_sorted(inliers).min(cap).max(1),
        p95: nearest_rank_quantile_ratio(inliers, 19, 20).min(cap).max(1),
        p99: nearest_rank_quantile_ratio(inliers, 99, 100)
            .min(cap)
            .max(1),
        clipped_outliers,
    }
}

/// Returns the exact integer median of a non-empty sorted slice without overflow.
///
/// For even lengths `2k`, averages the two middle elements `(a + b) / 2` so exact
/// 50/50 bimodal distributions do not bias arbitrarily toward one side.
#[inline]
fn exact_median_sorted(sorted: &[u64]) -> u64 {
    let n = sorted.len();
    debug_assert!(n > 0);
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        let lo = sorted[(n / 2) - 1];
        let hi = sorted[n / 2];
        lo + ((hi - lo) / 2)
    }
}

/// Computes the exact nearest-rank percentile for rational `num / den in (0, 1]`
/// over a non-empty sorted slice using pure integer arithmetic (`div_ceil`),
/// avoiding external `f64::ceil()` libm calls on baseline x86-64 targets.
#[inline]
fn nearest_rank_quantile_ratio(sorted: &[u64], num: usize, den: usize) -> u64 {
    let n = sorted.len();
    debug_assert!(n > 0 && num > 0 && num <= den);
    let rank = n.saturating_mul(num).div_ceil(den);
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn config_validation_rejects_invalid_parameters() {
        assert!(EstimatorConfig::default().validate().is_ok());

        let cases = [
            EstimatorConfig {
                window_size: 0,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                outlier_median_multiplier: 0,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                outlier_floor_span: 0,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                max_clip_span: 0,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                hysteresis_ratio: 0.99,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                hysteresis_ratio: f64::NAN,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                required_sustained_evals: 0,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                ewma_alpha: 0.0,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                ewma_alpha: 1.01,
                ..EstimatorConfig::default()
            },
            EstimatorConfig {
                ewma_alpha: f64::INFINITY,
                ..EstimatorConfig::default()
            },
        ];
        for bad in &cases {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn empty_window_returns_safe_defaults() {
        let est = SlidingWindowEstimator::new(EstimatorConfig::default()).unwrap();
        let snap = est.snapshot();
        assert_eq!(snap.window_query_count, 0);
        assert_eq!(snap.observed_fpr, 0.0);
        assert_eq!(snap.p50_query_len, 1);
        assert_eq!(snap.p95_query_len, 1);
        assert_eq!(snap.p99_query_len, 1);
        assert_eq!(snap.effective_query_span, 1);
        assert_eq!(snap.clipped_outliers, 0);
    }

    #[test]
    fn outlier_clipping_resists_adversarial_poison_spikes() {
        let cfg = EstimatorConfig::default();
        let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0xA3E1_0001);

        // 85% baseline R=32 queries mixed with 15% adversarial 10^7-span poison queries.
        for i in 0..4096u64 {
            let span = if i % 20 < 3 {
                10_000_000
            } else {
                rng.gen_range(24..=40)
            };
            let start = rng.gen_range(0..1_000_000_000);
            let range = QueryRange::with_len(start, span).unwrap();
            est.record_query(range, false);
        }

        let snap = est.snapshot();
        assert_eq!(snap.window_query_count, 4096);
        assert!(
            snap.clipped_outliers >= 600,
            "expected ~15% poison spikes clipped"
        );
        assert!(
            (24..=40).contains(&snap.p50_query_len),
            "p50 shifted unexpectedly: {}",
            snap.p50_query_len
        );
        assert!(
            (32..=40).contains(&snap.p95_query_len),
            "p95 shifted under poison attack: {}",
            snap.p95_query_len
        );
        assert_eq!(
            snap.effective_query_span, 32,
            "effective_query_span must not shift under 15% poison spikes"
        );
        assert_eq!(est.span_transitions(), 0);
    }

    #[test]
    fn legitimate_regime_shift_updates_span_after_cooldown_and_hysteresis() {
        let cfg = EstimatorConfig {
            window_size: 256,
            min_samples_for_clipping: 32,
            cooldown_queries: 64,
            required_sustained_evals: 4,
            hysteresis_ratio: 2.0,
            ..EstimatorConfig::default()
        };
        let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();

        // Warm up with 256 queries of span 32.
        for i in 0..256u64 {
            est.record_query(QueryRange::with_len(i * 100, 32).unwrap(), false);
        }
        assert_eq!(est.effective_query_span(), 32);
        assert_eq!(est.span_transitions(), 0);

        // Shift workload legitimately to span 256 (> 50% of window -> median shifts).
        for i in 0..300u64 {
            est.record_query(QueryRange::with_len(i * 1000, 256).unwrap(), false);
        }
        assert_eq!(est.effective_query_span(), 256);
        assert_eq!(est.span_transitions(), 1);
    }

    #[test]
    fn cooldown_and_sustained_evals_prevent_premature_transition() {
        let cfg = EstimatorConfig {
            window_size: 64,
            min_samples_for_clipping: 16,
            cooldown_queries: 100,
            required_sustained_evals: 4,
            hysteresis_ratio: 2.0,
            ..EstimatorConfig::default()
        };
        let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();

        // Send 99 queries of span 512: still inside cooldown (< 100).
        for i in 0..99u64 {
            est.record_query(QueryRange::with_len(i * 1000, 512).unwrap(), false);
        }
        assert_eq!(est.effective_query_span(), 32);
        assert!(!est.evaluate_span_transition());

        // 100th query crosses cooldown; still needs 4 consecutive evaluations.
        est.record_query(QueryRange::with_len(100_000, 512).unwrap(), false);
        assert_eq!(est.effective_query_span(), 32);

        // Drive explicit evaluations: 1st, 2nd, 3rd must not transition; 4th transitions.
        // Note: query #100 already executed 1 evaluation (since stride=25 and 100 >= 25),
        // so let's reset cleanly by measuring remaining evaluations needed.
        let mut transitions = 0;
        for _ in 0..4 {
            if est.evaluate_span_transition() {
                transitions += 1;
            }
        }
        assert_eq!(transitions, 1);
        assert_eq!(est.effective_query_span(), 512);
    }

    #[test]
    fn ewma_and_sliding_fpr_tracking_handle_valid_and_invalid_inputs() {
        let cfg = EstimatorConfig {
            window_size: 100,
            ewma_alpha: 0.25,
            ..EstimatorConfig::default()
        };
        let mut est = SlidingWindowEstimator::new(cfg).unwrap();

        // First sample initializes directly; subsequent samples smooth with alpha = 0.25.
        est.record_ingest(1000, 1.0);
        assert!((est.ingestion_rate_ewma() - 1000.0).abs() < 1e-9);
        est.record_ingest(2000, 1.0);
        assert!((est.ingestion_rate_ewma() - 1250.0).abs() < 1e-9);

        // Invalid elapsed_secs must be ignored without corrupting EWMA.
        est.record_ingest(9999, 0.0);
        est.record_ingest(9999, -1.0);
        est.record_ingest(9999, f64::NAN);
        assert!((est.ingestion_rate_ewma() - 1250.0).abs() < 1e-9);

        est.record_bulk_load_speed(400_000.0);
        est.record_bulk_load_speed(800_000.0);
        assert!((est.snapshot().bulk_load_speed - 500_000.0).abs() < 1e-9);
        est.record_bulk_load_speed(-10.0);
        assert!((est.snapshot().bulk_load_speed - 500_000.0).abs() < 1e-9);

        est.record_query_rate(100, 0.5);
        assert!((est.snapshot().query_rate - 200.0).abs() < 1e-9);

        // Sliding window FPR: 10 FPs out of 100 queries -> 0.10, then evicted by 100 non-FPs -> 0.0.
        for i in 0..100u64 {
            est.record_query(QueryRange::point(i), i < 10);
        }
        assert!((est.snapshot().observed_fpr - 0.10).abs() < 1e-9);

        for i in 100..200u64 {
            est.record_query(QueryRange::point(i), false);
        }
        assert!((est.snapshot().observed_fpr - 0.0).abs() < 1e-9);
    }

    #[test]
    fn legitimate_full_table_scan_regime_caps_at_max_clip_span() {
        let cfg = EstimatorConfig {
            window_size: 128,
            min_samples_for_clipping: 32,
            cooldown_queries: 16,
            required_sustained_evals: 2,
            ..EstimatorConfig::default()
        };
        let max_cap = cfg.max_clip_span;
        let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();

        // > 50% of window genuinely issues 10^7-wide scans.
        for i in 0..160u64 {
            est.record_query(QueryRange::with_len(i, 10_000_000).unwrap(), false);
        }
        let snap = est.snapshot();
        assert_eq!(snap.p95_query_len, max_cap);
        assert_eq!(snap.effective_query_span, max_cap);
        assert_eq!(snap.clipped_outliers, 128);
    }

    #[test]
    fn sample_stride_validation_requires_power_of_two() {
        for bad in [0u64, 3, 5, 6, 48, 65, 100] {
            let cfg = EstimatorConfig {
                sample_stride: bad,
                ..EstimatorConfig::default()
            };
            assert!(
                cfg.validate().is_err(),
                "expected sample_stride={bad} to fail validation"
            );
        }
        for good in [1u64, 2, 4, 16, 64, 128] {
            let cfg = EstimatorConfig {
                sample_stride: good,
                ..EstimatorConfig::default()
            };
            assert!(
                cfg.validate().is_ok(),
                "expected sample_stride={good} to pass validation"
            );
        }
    }

    #[test]
    fn sampled_quantile_accuracy_matches_unsampled_and_detects_shifts() {
        let unsampled_cfg = EstimatorConfig {
            sample_stride: 1,
            ..EstimatorConfig::default()
        };
        let sampled_cfg = EstimatorConfig::default(); // sample_stride = 64
        assert_eq!(sampled_cfg.sample_stride, DEFAULT_SAMPLE_STRIDE);

        let mut exact = SlidingWindowEstimator::with_initial_span(unsampled_cfg, 16).unwrap();
        let mut sampled = SlidingWindowEstimator::with_initial_span(sampled_cfg, 16).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0x5A17_0064);

        // Phase 1: 8,192 queries centered around R=32 with 10% adversarial 10^7 poison spikes.
        for i in 0..8_192u64 {
            let span = if i % 10 == 0 {
                10_000_000
            } else {
                rng.gen_range(24..=40)
            };
            let q = QueryRange::with_len(i * 100, span).unwrap();
            exact.record_query(q, false);
            sampled.record_query(q, false);
        }

        let s_exact = exact.snapshot();
        let s_sampled = sampled.snapshot();
        assert_eq!(s_sampled.window_query_count, 4096);
        assert!(
            s_sampled.p50_query_len.abs_diff(s_exact.p50_query_len) <= 4,
            "sampled p50={} drifted from exact p50={}",
            s_sampled.p50_query_len,
            s_exact.p50_query_len
        );
        assert!(
            s_sampled.p95_query_len.abs_diff(s_exact.p95_query_len) <= 4,
            "sampled p95={} drifted from exact p95={}",
            s_sampled.p95_query_len,
            s_exact.p95_query_len
        );
        assert!(
            (24..=40).contains(&s_sampled.effective_query_span),
            "poisoned effective_query_span={}",
            s_sampled.effective_query_span
        );

        // Phase 2: genuine shift to R=256 (within 4,096 queries, both estimators detect shift).
        for i in 0..4_096u64 {
            let q = QueryRange::with_len(10_000_000 + i * 512, 256).unwrap();
            exact.record_query(q, false);
            sampled.record_query(q, false);
        }
        assert_eq!(exact.effective_query_span(), 256);
        assert_eq!(sampled.effective_query_span(), 256);

        // Phase 3: shift back down to R=32.
        for i in 0..4_096u64 {
            let q = QueryRange::with_len(20_000_000 + i * 64, 32).unwrap();
            exact.record_query(q, false);
            sampled.record_query(q, false);
        }
        assert_eq!(exact.effective_query_span(), 32);
        assert_eq!(sampled.effective_query_span(), 32);
    }

    #[test]
    fn integer_nearest_rank_quantile_matches_float_ceil_across_all_sizes() {
        let cases = [
            (3usize, 5usize, 0.60_f64),
            (19, 20, 0.95_f64),
            (99, 100, 0.99_f64),
        ];
        for &(num, den, p) in &cases {
            for n in 1..=100_000usize {
                let int_rank = n.saturating_mul(num).div_ceil(den);
                let int_idx = int_rank.saturating_sub(1).min(n - 1);
                let flt_rank = ((n as f64) * p).ceil() as usize;
                let flt_idx = flt_rank.saturating_sub(1).min(n - 1);
                assert_eq!(
                    int_idx, flt_idx,
                    "index mismatch at n={n}, p={p}: int={int_idx}, flt={flt_idx}"
                );
            }
        }
    }

    #[test]
    fn replace_sorted_sample_preserves_multiset_and_sorted_order_exhaustively() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE_50A7);
        for cap in [1usize, 2, 3, 7, 16, 64] {
            let mut ring: Vec<u64> = (0..cap)
                .map(|_| match rng.gen_range(0..6) {
                    0 => 1,
                    1 => u64::MAX,
                    2 => 16,
                    _ => rng.gen_range(1..=64),
                })
                .collect();
            let mut sorted = ring.clone();
            sorted.sort_unstable();

            let mut head = 0usize;
            for _ in 0..1_000 {
                let new_val = match rng.gen_range(0..8) {
                    0 => 1,
                    1 => u64::MAX,
                    2 => ring[head], // equal-value no-op
                    _ => rng.gen_range(1..=64),
                };
                let old_val = ring[head];
                ring[head] = new_val;
                head = (head + 1) % cap;

                if old_val != new_val {
                    replace_sorted_sample(&mut sorted, old_val, new_val);
                }

                let mut expected = ring.clone();
                expected.sort_unstable();
                assert_eq!(sorted, expected, "sorted multiset diverged for cap={cap}");
            }
        }
    }

    #[test]
    fn steady_state_equal_eviction_still_transitions_after_cooldown_and_breach_streak() {
        // Fill window completely with constant R=256 while inside cooldown (initial_span=16),
        // then continue feeding constant R=256 so EVERY post-cooldown sample hits evicted == len.
        let cfg = EstimatorConfig {
            window_size: 256,
            min_samples_for_clipping: 32,
            cooldown_queries: 512,
            required_sustained_evals: 4,
            hysteresis_ratio: 2.0,
            sample_stride: 64,
            ..EstimatorConfig::default()
        };
        let mut est = SlidingWindowEstimator::with_initial_span(cfg, 16).unwrap();

        for i in 0..511u64 {
            est.record_query(QueryRange::with_len(i * 512, 256).unwrap(), false);
        }
        assert_eq!(est.effective_query_span(), 16);
        assert_eq!(est.span_transitions(), 0);

        // Continue constant R=256 past cooldown through 4 sustained evals (4 * 64 = 256 queries).
        for i in 511..1_024u64 {
            est.record_query(QueryRange::with_len(i * 512, 256).unwrap(), false);
        }
        assert_eq!(est.effective_query_span(), 256);
        assert_eq!(est.span_transitions(), 1);
    }

    #[test]
    fn optimized_estimator_matches_reference_model_on_shifting_and_poisoned_workloads() {
        // Naive reference implementation computing quantiles via full sort + f64::ceil
        // without caching, equal-eviction skips, or eval_in_band_clean shortcuts.
        struct RefEstimator {
            cfg: EstimatorConfig,
            mask: u64,
            phase_step: u64,
            phase: u64,
            next_sample: u64,
            cap: usize,
            window: VecDeque<u64>,
            fps: VecDeque<u64>,
            total: u64,
            since_change: u64,
            since_eval: u64,
            eff: u64,
            dir: Option<BreachDirection>,
            evals: u32,
            transitions: u64,
        }
        impl RefEstimator {
            fn new(cfg: EstimatorConfig, initial: u64) -> Self {
                let cap = cfg.sample_window_capacity();
                Self {
                    mask: cfg.sample_stride - 1,
                    phase_step: golden_odd_phase_step(cfg.sample_stride),
                    phase: 0,
                    next_sample: 0,
                    cap,
                    window: VecDeque::with_capacity(cap),
                    fps: VecDeque::new(),
                    total: 0,
                    since_change: 0,
                    since_eval: 0,
                    eff: initial.clamp(1, cfg.max_clip_span),
                    dir: None,
                    evals: 0,
                    transitions: 0,
                    cfg,
                }
            }
            fn quantiles(&self) -> ClippedQuantiles {
                let mut sorted: Vec<u64> = self.window.iter().copied().collect();
                sorted.sort_unstable();
                let n = sorted.len();
                if n == 0 {
                    return ClippedQuantiles::EMPTY;
                }
                let q_at = |s: &[u64], p: f64| -> u64 {
                    let r = ((s.len() as f64) * p).ceil() as usize;
                    s[r.saturating_sub(1).min(s.len() - 1)]
                };
                let min_clip = self.cfg.effective_min_clip_samples();
                let (inliers, clipped) = if n >= min_clip {
                    let raw_p60 = q_at(&sorted, 0.60);
                    let outlier_cap = raw_p60
                        .saturating_mul(self.cfg.outlier_median_multiplier)
                        .max(self.cfg.outlier_floor_span);
                    let clip_cap = outlier_cap.min(self.cfg.max_clip_span);
                    let ic = sorted.partition_point(|&v| v <= outlier_cap).max(1);
                    let wc = sorted.partition_point(|&v| v <= clip_cap);
                    (&sorted[..ic], (n - wc) as u64)
                } else {
                    let wm = sorted.partition_point(|&v| v <= self.cfg.max_clip_span);
                    (&sorted[..], (n - wm) as u64)
                };
                let cap = self.cfg.max_clip_span;
                ClippedQuantiles {
                    p50: exact_median_sorted(inliers).min(cap).max(1),
                    p95: q_at(inliers, 0.95).min(cap).max(1),
                    p99: q_at(inliers, 0.99).min(cap).max(1),
                    clipped_outliers: clipped,
                }
            }
            fn eval(&mut self) {
                let min_eval = (self.cap / 2).clamp(1, 16);
                if self.window.len() < min_eval || self.since_change < self.cfg.cooldown_queries {
                    self.dir = None;
                    self.evals = 0;
                    return;
                }
                let q = self.quantiles();
                let cand = q.p95.max(1);
                let eff_f = self.eff.max(1) as f64;
                let cand_f = cand as f64;
                let breach = if cand_f > eff_f * self.cfg.hysteresis_ratio {
                    Some(BreachDirection::Up)
                } else if cand_f < eff_f / self.cfg.hysteresis_ratio {
                    Some(BreachDirection::Down)
                } else {
                    None
                };
                let Some(d) = breach else {
                    self.dir = None;
                    self.evals = 0;
                    return;
                };
                if self.dir == Some(d) {
                    self.evals += 1;
                } else {
                    self.dir = Some(d);
                    self.evals = 1;
                }
                if self.evals >= self.cfg.required_sustained_evals {
                    self.eff = cand;
                    self.since_change = 0;
                    self.since_eval = 0;
                    self.dir = None;
                    self.evals = 0;
                    self.transitions += 1;
                }
            }
            fn record(&mut self, range: QueryRange, is_fp: bool) {
                let seq = self.total;
                self.total += 1;
                self.since_change += 1;
                self.since_eval += 1;
                if is_fp {
                    let cutoff = self.total.saturating_sub(self.cfg.window_size as u64);
                    while self.fps.front().is_some_and(|&f| f < cutoff) {
                        self.fps.pop_front();
                    }
                    self.fps.push_back(seq);
                }
                if seq == self.next_sample {
                    self.phase = self.phase.wrapping_add(self.phase_step) & self.mask;
                    let next_block = (seq & !self.mask).saturating_add(self.cfg.sample_stride);
                    self.next_sample = next_block.saturating_add(self.phase);
                    if self.window.len() == self.cap {
                        self.window.pop_front();
                    }
                    self.window.push_back(range.len_saturated().max(1));
                    let stride = self.cfg.auto_eval_stride();
                    if self.since_change >= self.cfg.cooldown_queries
                        && (self.since_eval >= stride || self.cfg.sample_stride >= stride)
                    {
                        self.since_eval = 0;
                        self.eval();
                    }
                }
            }
            fn active_fps(&self) -> u64 {
                let cutoff = self.total.saturating_sub(self.cfg.window_size as u64);
                let exp = self.fps.partition_point(|&s| s < cutoff);
                (self.fps.len() - exp) as u64
            }
        }

        let configs = [
            EstimatorConfig::default(), // sample_stride=64, window_size=4096 (cap=64)
            EstimatorConfig {
                window_size: 64,
                sample_stride: 64,
                cooldown_queries: 128,
                ..EstimatorConfig::default()
            }, // cap=1 boundary
            EstimatorConfig {
                window_size: 256,
                sample_stride: 1,
                cooldown_queries: 100,
                required_sustained_evals: 4,
                ..EstimatorConfig::default()
            }, // sample_stride < auto_eval_stride (1 < 25)
        ];

        for cfg in configs {
            let mut opt = SlidingWindowEstimator::with_initial_span(cfg.clone(), 16).unwrap();
            let mut ref_est = RefEstimator::new(cfg.clone(), 16);
            let mut rng = ChaCha8Rng::seed_from_u64(0xE0A1_7E57);

            // Multi-phase trace: steady R=16 -> R=256 -> R=32+poison -> R=1024 -> steady R=16.
            for step in 0..20_000u64 {
                let span = match step / 4_000 {
                    0 => 16,
                    1 => 256,
                    2 => {
                        if step % 12 == 0 {
                            10_000_000
                        } else {
                            rng.gen_range(24..=40)
                        }
                    }
                    3 => 1024,
                    _ => 16,
                };
                let is_fp = step % 313 == 0;
                let range = QueryRange::with_len(step * 2_048, span).unwrap();
                opt.record_query(range, is_fp);
                ref_est.record(range, is_fp);

                if step % 64 == 0 || step == 19_999 {
                    let s_opt = opt.snapshot();
                    let q_ref = ref_est.quantiles();
                    let w_cnt = ref_est.total.min(cfg.window_size as u64);
                    let ref_fpr = if w_cnt == 0 {
                        0.0
                    } else {
                        (ref_est.active_fps() as f64) / (w_cnt as f64)
                    };
                    let ref_clipped = q_ref
                        .clipped_outliers
                        .saturating_mul(cfg.sample_stride)
                        .min(w_cnt);

                    assert_eq!(opt.total_queries(), ref_est.total);
                    assert_eq!(opt.queries_since_last_change(), ref_est.since_change);
                    assert_eq!(opt.effective_query_span(), ref_est.eff);
                    assert_eq!(opt.span_transitions(), ref_est.transitions);
                    assert_eq!(s_opt.window_query_count, w_cnt);
                    assert_eq!(s_opt.p50_query_len, q_ref.p50);
                    assert_eq!(s_opt.p95_query_len, q_ref.p95);
                    assert_eq!(s_opt.p99_query_len, q_ref.p99);
                    assert_eq!(s_opt.clipped_outliers, ref_clipped);
                    assert!((s_opt.observed_fpr - ref_fpr).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn disabled_estimator_fast_path_skips_recording_and_supports_dynamic_toggle() {
        let default_cfg = EstimatorConfig::default();
        assert!(default_cfg.enabled);

        let disabled_cfg = EstimatorConfig {
            window_size: 128,
            cooldown_queries: 32,
            required_sustained_evals: 2,
            ..EstimatorConfig::default()
        }
        .with_enabled(false);
        assert!(!disabled_cfg.enabled);

        let mut est = SlidingWindowEstimator::with_initial_span(disabled_cfg, 32).unwrap();
        assert!(!est.is_enabled());

        // Recording thousands of shifted spans and false positives must be a complete no-op.
        for i in 0..2_048u64 {
            est.record_query(QueryRange::with_len(i * 1_000, 512).unwrap(), i % 7 == 0);
        }
        assert_eq!(est.total_queries(), 0);
        assert_eq!(est.effective_query_span(), 32);
        assert_eq!(est.span_transitions(), 0);
        let snap = est.snapshot();
        assert_eq!(snap.window_query_count, 0);
        assert_eq!(snap.observed_fpr, 0.0);

        // Dynamically re-enabling resumes normal quantile & regime-shift tracking.
        est.set_enabled(true);
        assert!(est.is_enabled());
        assert!(est.config().enabled);
        for i in 0..512u64 {
            est.record_query(QueryRange::with_len(i * 1_000, 512).unwrap(), false);
        }
        assert_eq!(est.total_queries(), 512);
        assert_eq!(est.effective_query_span(), 512);
        assert_eq!(est.span_transitions(), 1);

        // Toggling off again freezes recording immediately.
        est.set_enabled(false);
        assert!(!est.is_enabled());
        for i in 0..512u64 {
            est.record_query(QueryRange::with_len(i * 1_000, 16).unwrap(), true);
        }
        assert_eq!(est.total_queries(), 512);
        assert_eq!(est.effective_query_span(), 512);
    }
}
