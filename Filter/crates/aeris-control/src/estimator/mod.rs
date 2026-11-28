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

mod config;
mod quantile;
mod transitions;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_equivalence;

pub use config::{
    EstimatorConfig, DEFAULT_COOLDOWN_QUERIES, DEFAULT_EWMA_ALPHA, DEFAULT_HYSTERESIS_RATIO,
    DEFAULT_MAX_CLIP_SPAN, DEFAULT_MIN_SAMPLES_FOR_CLIPPING, DEFAULT_OUTLIER_FLOOR_SPAN,
    DEFAULT_OUTLIER_MEDIAN_MULTIPLIER, DEFAULT_REQUIRED_SUSTAINED_EVALS, DEFAULT_SAMPLE_STRIDE,
    DEFAULT_WINDOW_SIZE,
};

use self::quantile::{compute_clipped_quantiles, replace_sorted_sample, ClippedQuantiles};
use self::transitions::{golden_odd_phase_step, update_ewma};
use aeris_types::{AerisResult, QueryRange, WorkloadEstimator, WorkloadStats};
use std::collections::VecDeque;
use tracing::{debug, trace, warn};

/// Direction in which candidate `p95` has breached the hysteresis deadband.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BreachDirection {
    /// Candidate `p95 > effective_query_span * hysteresis_ratio`.
    Up,
    /// Candidate `p95 < effective_query_span / hysteresis_ratio`.
    Down,
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
    /// Returns [`aeris_types::AerisError::InvalidConfig`] if `config` violates any invariant.
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
    /// Returns [`aeris_types::AerisError::InvalidConfig`] if `config` violates any invariant.
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
