//! Asynchronous background rebuild coordinator, concurrency delta buffer,
//! token-bucket I/O throttler, and unified [`ControlledAerisEngine`] facade.
//!
//! # Zero-Downtime Dynamic Rebuild Protocol (`F_old -> F_new`)
//! 1. Activate [`DeltaBuffer`] so concurrent `Insert`/`Delete` mutations arriving
//!    on `F_old` are durably buffered in FIFO arrival order.
//! 2. Spawn dedicated background thread (`"aeris-bg-rebuild"`) that streams all
//!    live keys from `source_reverse_map.iter_all_keys()` in batches paced by
//!    [`IoThrottler`].
//! 3. Bulk-load `F_new` (`AerisFilter::bulk_load`) under `target_config` into a
//!    fresh destination [`ReverseMapStore`], automatically expanding `q -> q + 1`
//!    if the live key set exceeds `target_config.max_load_factor`.
//! 4. Replay [`FalsePositiveLog`] onto `F_new` so every previously observed false
//!    positive is monotonically eliminated off the foreground critical path.
//! 5. On [`AsyncRebuildCoordinator::finish_and_drain_deltas`], join the worker,
//!    atomically drain + deactivate [`DeltaBuffer`], apply deltas idempotently
//!    against the loaded snapshot key set (preventing duplicate mementos across
//!    snapshot/delta overlap), execute catch-up [`FalsePositiveLog`] replay, and
//!    verify zero false negatives for all live delta-inserted keys before returning
//!    [`CompletedRebuild`] for atomic `F_old -> F_new` swap.
//!
//! # Lock Ordering
//! `AsyncRebuildCoordinator::in_flight` -> `DeltaBuffer::inner` (leaf) and
//! `FalsePositiveLog::inner` (leaf). Neither leaf lock is ever held while
//! invoking [`AerisFilter`] or [`ReverseMapStore`] operations.

use crate::estimator::{EstimatorConfig, SlidingWindowEstimator};
use crate::fp_log::{FalsePositiveLog, FpLogConfig, FpReplayReport};
use crate::optimizer::DiscreteCostOptimizer;
use crate::trigger::{
    ProactiveTrigger, ProactiveTriggerConfig, TriggerAction, TriggerEvaluation, TriggerInput,
};
use aeris_core::AerisFilter;
use aeris_storage::{KvStoreConfig, ReferenceKvStore, ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, AerisResult, CostModelParams, CostOptimizer, Epoch, FilterConfig, Key,
    OptimizationDecision, QueryRange, RangeFilter, ReverseMapStore, WorkloadEstimator,
    WorkloadStats, MAX_QUOTIENT_BITS,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info};

/// Default batch size (keys per pacing window) for [`IoThrottler`].
pub const DEFAULT_IO_THROTTLE_BATCH_SIZE: usize = 1024;

/// Default batch size (mutations per off-lock pass) for incremental [`DeltaBuffer`] pre-draining.
pub const DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE: usize = 1024;

/// Default residual [`DeltaBuffer`] tail threshold below which off-lock pre-draining stops
/// and hands off to the brief locked tail drain + atomic swap.
pub const DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD: usize = 256;

/// Default maximum number of off-lock pre-drain batch passes before entering the locked tail swap.
pub const DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES: usize = 64;

/// Dedicated OS thread name assigned to background rebuild workers.
pub const BG_REBUILD_THREAD_NAME: &str = "aeris-bg-rebuild";

/// Subdirectory name for the primary [`ReferenceKvStore`] under a [`ControlledAerisEngine`] root.
const ENGINE_KV_DIR_NAME: &str = "kv_db";

/// Filename for the persistent [`FalsePositiveLog`] under a [`ControlledAerisEngine`] root.
const ENGINE_FP_LOG_NAME: &str = "fp.log";

/// Prefix for generational [`RocksDbReverseMap`] subdirectories swapped across rebuilds.
const ENGINE_RM_GEN_PREFIX: &str = "reverse_map_gen_";

/// Minimum elapsed seconds floor when computing observed bulk-load throughput.
const MIN_BULK_LOAD_ELAPSED_SECS: f64 = 1e-6;

/// Single mutation captured in [`DeltaBuffer`] while an asynchronous rebuild is in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeltaMutation {
    /// Key inserted into `F_old` during background rebuild.
    Insert(Key),
    /// Key deleted from `F_old` during background rebuild.
    Delete(Key),
}

/// Internal mutex-protected state for [`DeltaBuffer`].
#[derive(Debug, Default)]
struct DeltaBufferInner {
    active: bool,
    ops: VecDeque<DeltaMutation>,
}

/// Thread-safe FIFO buffer recording concurrent mutations (`Insert` / `Delete`)
/// arriving while a background rebuild constructs `F_new`.
///
/// # Lock Ordering
/// `DeltaBuffer::inner` is a leaf mutex; it never acquires any other lock.
#[derive(Debug, Default)]
pub struct DeltaBuffer {
    inner: Mutex<DeltaBufferInner>,
}

impl DeltaBuffer {
    /// Creates an inactive, empty [`DeltaBuffer`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Activates mutation buffering and clears any leftover entries from prior runs.
    pub fn activate(&self) {
        let mut inner = self.inner.lock();
        inner.ops.clear();
        inner.active = true;
        debug!("activated DeltaBuffer for background rebuild");
    }

    /// Deactivates mutation buffering without discarding already-buffered operations.
    pub fn deactivate(&self) {
        let mut inner = self.inner.lock();
        inner.active = false;
        debug!(remaining_ops = inner.ops.len(), "deactivated DeltaBuffer");
    }

    /// Returns `true` if the buffer is currently capturing incoming mutations.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner.lock().active
    }

    /// Records an inserted `key` if the buffer is active, returning `true` if captured.
    pub fn record_insert(&self, key: Key) -> bool {
        let mut inner = self.inner.lock();
        if !inner.active {
            return false;
        }
        inner.ops.push_back(DeltaMutation::Insert(key));
        true
    }

    /// Records a deleted `key` if the buffer is active, returning `true` if captured.
    pub fn record_delete(&self, key: Key) -> bool {
        let mut inner = self.inner.lock();
        if !inner.active {
            return false;
        }
        inner.ops.push_back(DeltaMutation::Delete(key));
        true
    }

    /// Drains and returns up to `max_items` buffered mutations from the front of the
    /// queue in strict FIFO arrival order while keeping the buffer active for concurrent
    /// writers appending to the tail.
    ///
    /// # Complexity
    /// `O(min(max_items, len))` time and space; acquires `DeltaBuffer::inner` briefly.
    #[must_use]
    pub fn drain_batch(&self, max_items: usize) -> Vec<DeltaMutation> {
        if max_items == 0 {
            return Vec::new();
        }
        let mut inner = self.inner.lock();
        let take = max_items.min(inner.ops.len());
        inner.ops.drain(..take).collect()
    }

    /// Drains and returns all buffered mutations in strict FIFO arrival order.
    #[must_use]
    pub fn drain_all(&self) -> Vec<DeltaMutation> {
        let mut inner = self.inner.lock();
        inner.ops.drain(..).collect()
    }

    /// Atomically drains all buffered mutations in FIFO order AND deactivates the buffer
    /// under a single lock acquisition, preventing races at the swap boundary.
    #[must_use]
    pub fn drain_and_deactivate(&self) -> Vec<DeltaMutation> {
        let mut inner = self.inner.lock();
        inner.active = false;
        inner.ops.drain(..).collect()
    }

    /// Returns the number of mutations currently queued in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().ops.len()
    }

    /// Returns `true` if zero mutations are queued in the buffer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().ops.is_empty()
    }
}

/// Configuration for token-bucket background scan rate limiting ([`IoThrottler`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoThrottleConfig {
    /// Maximum sustained Reverse Map scan throughput in keys/second (`None` = unthrottled).
    pub max_keys_per_sec: Option<u64>,
    /// Number of keys streamed per pacing check.
    pub batch_size: usize,
}

impl Default for IoThrottleConfig {
    fn default() -> Self {
        Self {
            max_keys_per_sec: None,
            batch_size: DEFAULT_IO_THROTTLE_BATCH_SIZE,
        }
    }
}

impl IoThrottleConfig {
    /// Constructs a throttled configuration capping scan speed at `max_keys_per_sec`.
    #[must_use]
    pub const fn limited(max_keys_per_sec: u64, batch_size: usize) -> Self {
        Self {
            max_keys_per_sec: Some(max_keys_per_sec),
            batch_size,
        }
    }

    /// Validates that `batch_size >= 1` and `max_keys_per_sec` (when `Some`) is `>= 1`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on invalid parameters.
    pub fn validate(&self) -> AerisResult<()> {
        if self.batch_size == 0 {
            return Err(AerisError::InvalidConfig(
                "IoThrottleConfig::batch_size must be at least 1".to_string(),
            ));
        }
        if self.max_keys_per_sec == Some(0) {
            return Err(AerisError::InvalidConfig(
                "IoThrottleConfig::max_keys_per_sec must be at least 1 when set".to_string(),
            ));
        }
        Ok(())
    }
}

/// Telemetry counters recorded by [`IoThrottler`] during background scanning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThrottleMetrics {
    /// Total keys paced through the throttler.
    pub keys_paced: u64,
    /// Number of batches that required sleeping to honor the rate limit.
    pub throttle_events: u64,
    /// Cumulative pacing delay applied in nanoseconds.
    pub total_throttled_ns: u64,
}

/// Token-bucket rate limiter pacing streaming [`ReverseMapStore::iter_all_keys`]
/// consumption during background rebuilds so disk I/O does not starve foreground traffic.
///
/// Separates pure virtual-clock math ([`Self::consume_at`]) from real wall-clock
/// sleeping ([`Self::pace`]) for deterministic unit testing and real I/O enforcement.
#[derive(Clone, Debug)]
pub struct IoThrottler {
    config: IoThrottleConfig,
    bucket_capacity: f64,
    tokens: f64,
    last_ns: u64,
    started_at: Instant,
    metrics: ThrottleMetrics,
}

impl IoThrottler {
    /// Creates a validated [`IoThrottler`] starting with an empty token bucket so
    /// even short scans strictly obey `max_keys_per_sec`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` is invalid.
    pub fn new(config: IoThrottleConfig) -> AerisResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            bucket_capacity: config.batch_size as f64,
            tokens: 0.0,
            last_ns: 0,
            started_at: Instant::now(),
            metrics: ThrottleMetrics::default(),
        })
    }

    /// Returns the configured [`IoThrottleConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> IoThrottleConfig {
        self.config
    }

    /// Returns the configured batch size in keys.
    #[inline]
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Returns accumulated [`ThrottleMetrics`].
    #[inline]
    #[must_use]
    pub const fn metrics(&self) -> ThrottleMetrics {
        self.metrics
    }

    /// Pure deterministic virtual-clock token-bucket step.
    ///
    /// Advances the virtual clock to `now_ns`, refills tokens at `max_keys_per_sec`
    /// up to `batch_size`, deducts `keys`, updates [`ThrottleMetrics`], and returns
    /// the exact required wait duration in nanoseconds (`0` if unthrottled or within budget).
    ///
    /// # Complexity
    /// `O(1)` time, `O(1)` space, zero system-clock calls.
    pub fn consume_at(&mut self, keys: u64, now_ns: u64) -> u64 {
        self.metrics.keys_paced = self.metrics.keys_paced.saturating_add(keys);
        let Some(rate_u64) = self.config.max_keys_per_sec else {
            self.last_ns = self.last_ns.max(now_ns);
            return 0;
        };
        if keys == 0 {
            self.last_ns = self.last_ns.max(now_ns);
            return 0;
        }

        let rate = rate_u64 as f64;
        let elapsed_ns = now_ns.saturating_sub(self.last_ns);
        let refilled = (elapsed_ns as f64) * rate / 1_000_000_000.0;
        self.tokens = (self.tokens + refilled).min(self.bucket_capacity);
        self.tokens -= keys as f64;

        if self.tokens < 0.0 {
            let deficit = -self.tokens;
            let wait_ns = ((deficit / rate) * 1_000_000_000.0).ceil() as u64;
            self.tokens = 0.0;
            self.last_ns = now_ns.saturating_add(wait_ns);
            self.metrics.throttle_events = self.metrics.throttle_events.saturating_add(1);
            self.metrics.total_throttled_ns =
                self.metrics.total_throttled_ns.saturating_add(wait_ns);
            wait_ns
        } else {
            self.last_ns = now_ns;
            0
        }
    }

    /// Enforces the configured rate limit in real wall-clock time for a completed
    /// batch of `keys`, sleeping only the exact token deficit duration when needed.
    pub fn pace(&mut self, keys: u64) {
        let now_ns = self.started_at.elapsed().as_nanos() as u64;
        let wait_ns = self.consume_at(keys, now_ns);
        if wait_ns > 0 {
            thread::sleep(Duration::from_nanos(wait_ns));
        }
    }
}

/// Lifecycle cause for launching an asynchronous background rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebuildReason {
    /// Dynamic partition size reconfiguration (`m_old -> m_new`) under fixed budget `C`.
    DynamicRangeChange {
        /// Previous memento bit width `m = log2(P)`.
        old_m: u8,
        /// Target memento bit width `m* = log2(P*)`.
        new_m: u8,
    },
    /// Proactive EWMA-triggered capacity expansion (`q_old -> q_new`).
    ProactiveExpansion {
        /// Previous quotient bit width `q`.
        old_q: u8,
        /// Target quotient bit width `q_new`.
        new_q: u8,
    },
    /// Hysteresis-guarded capacity contraction (`q_old -> q_new`) after bulk deletes.
    Contraction {
        /// Previous quotient bit width `q`.
        old_q: u8,
        /// Target quotient bit width `q - 1`.
        new_q: u8,
    },
}

/// Comprehensive diagnostic report returned upon completing a background rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RebuildReport {
    /// Lifecycle trigger that initiated the rebuild.
    pub reason: RebuildReason,
    /// Number of live keys streamed from the source Reverse Map during bulk load.
    pub keys_loaded: usize,
    /// Number of net new delta inserts applied during final drain.
    pub delta_inserts_applied: usize,
    /// Number of net delta deletes applied during final drain.
    pub delta_deletes_applied: usize,
    /// False-Positive Log replay report from the background worker phase.
    pub fp_replay_initial: FpReplayReport,
    /// False-Positive Log replay report from the post-delta catch-up phase.
    pub fp_replay_catchup: FpReplayReport,
    /// Number of automatic `2x` expansions executed during bulk load or delta drain.
    pub auto_expansions: usize,
    /// Observed background bulk-load throughput (keys/sec) for feeding `T_rebuild` EWMA.
    pub observed_bulk_load_speed: f64,
    /// I/O throttler metrics collected during the Reverse Map streaming scan.
    pub throttle_metrics: ThrottleMetrics,
    /// End-to-end duration from rebuild start through delta drain completion in nanoseconds.
    pub duration_ns: u64,
}

/// Ready-to-swap rebuilt filter, populated destination Reverse Map, and report
/// returned by [`AsyncRebuildCoordinator::finish_and_drain_deltas`].
pub struct CompletedRebuild {
    /// Fully built, delta-reconciled, FP-replayed replacement [`AerisFilter`] (`F_new`).
    pub filter: AerisFilter,
    /// Destination [`ReverseMapStore`] aligned with `filter`'s canonical fingerprints.
    pub reverse_map: Arc<dyn ReverseMapStore>,
    /// Telemetry and correctness summary for the completed rebuild.
    pub report: RebuildReport,
}

/// Internal payload produced by the background rebuild worker thread.
struct BackgroundRebuildOutput {
    filter: AerisFilter,
    dest_reverse_map: Arc<dyn ReverseMapStore>,
    fp_log: Arc<FalsePositiveLog>,
    loaded_keys: BTreeSet<Key>,
    fp_snapshot_seq: u64,
    fp_replay_initial: FpReplayReport,
    auto_expansions: usize,
    observed_bulk_load_speed: f64,
    throttle_metrics: ThrottleMetrics,
    reason: RebuildReason,
}

/// Active background rebuild worker handle tracked by [`AsyncRebuildCoordinator`].
struct InFlightRebuild {
    reason: RebuildReason,
    done: Arc<AtomicBool>,
    handle: JoinHandle<AerisResult<BackgroundRebuildOutput>>,
    started_at: Instant,
}

/// RAII drop guard ensuring `done` flips to `true` even if the background thread
/// exits early via error or unwind.
struct CompletionSignal(Arc<AtomicBool>);

impl Drop for CompletionSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Lifecycle state machine stored inside [`AsyncRebuildCoordinator::in_flight`].
enum RebuildSlot {
    /// No background rebuild or staged delta drain is active.
    Idle,
    /// Background thread `"aeris-bg-rebuild"` is executing or waiting to be joined.
    Running(InFlightRebuild),
    /// Background thread has been joined into a [`StagedRebuild`] performing
    /// two-phase incremental delta draining before atomic swap.
    Draining,
}

/// RAII drop guard ensuring [`RebuildSlot::Draining`] always transitions back to
/// [`RebuildSlot::Idle`] and deactivates [`DeltaBuffer`] if a [`StagedRebuild`]
/// is dropped early due to error or panic.
struct DrainGuard {
    in_flight: Arc<Mutex<RebuildSlot>>,
    delta_buffer: Arc<DeltaBuffer>,
    active: bool,
}

impl DrainGuard {
    fn new(in_flight: Arc<Mutex<RebuildSlot>>, delta_buffer: Arc<DeltaBuffer>) -> Self {
        Self {
            in_flight,
            delta_buffer,
            active: true,
        }
    }

    fn release(mut self) {
        self.active = false;
        *self.in_flight.lock() = RebuildSlot::Idle;
    }
}

impl Drop for DrainGuard {
    fn drop(&mut self) {
        if self.active {
            self.delta_buffer.deactivate();
            *self.in_flight.lock() = RebuildSlot::Idle;
        }
    }
}

/// Joined background rebuild handle supporting two-phase incremental [`DeltaBuffer`]
/// draining so large concurrent mutation backlogs are applied to `F_new` *before*
/// acquiring exclusive swap / chunk-quiescence locks.
///
/// # Two-Phase Protocol
/// 1. **Off-Lock Pre-Drain:** Call [`Self::pre_drain_until_small`] (or repeated
///    [`Self::pre_drain_batch`] calls) followed by [`Self::verify_pre_drained_keys`]
///    while foreground writers continue executing on `F_old`.
/// 2. **Locked Tail Swap:** Acquire swap/quiescence locks and invoke
///    [`Self::finish_tail`] or [`Self::finish_tail_with_live_keys`] to drain only
///    the tiny residual delta tail, replay catch-up [`FalsePositiveLog`] entries,
///    and return [`CompletedRebuild`].
pub struct StagedRebuild {
    bg: BackgroundRebuildOutput,
    started_at: Instant,
    delta_buffer: Arc<DeltaBuffer>,
    drain_guard: DrainGuard,
    delta_inserts_applied: usize,
    delta_deletes_applied: usize,
    auto_expansions: usize,
    all_inserted_delta_keys: Vec<Key>,
    unverified_inserted_keys: Vec<Key>,
    pre_drain_passes: usize,
    pre_drained_mutations: usize,
}

impl StagedRebuild {
    /// Returns the number of mutations currently waiting in [`DeltaBuffer`].
    #[inline]
    #[must_use]
    pub fn pending_delta_len(&self) -> usize {
        self.delta_buffer.len()
    }

    /// Returns the number of off-lock pre-drain batch passes executed so far.
    #[inline]
    #[must_use]
    pub const fn pre_drain_passes(&self) -> usize {
        self.pre_drain_passes
    }

    /// Returns the cumulative number of delta mutations drained off-lock so far.
    #[inline]
    #[must_use]
    pub const fn pre_drained_mutations(&self) -> usize {
        self.pre_drained_mutations
    }

    /// Drains up to `max_items` mutations from [`DeltaBuffer`] and applies them
    /// idempotently onto `F_new` and `dest_reverse_map` (using batched Reverse Map
    /// inserts) while leaving [`DeltaBuffer`] active for concurrent writers.
    ///
    /// Returns the number of mutations popped and processed in this batch.
    ///
    /// # Errors
    /// Propagates any filter expansion or Reverse Map storage error.
    pub fn pre_drain_batch(&mut self, max_items: usize) -> AerisResult<usize> {
        let batch = self.delta_buffer.drain_batch(max_items);
        let count = batch.len();
        if count == 0 {
            return Ok(0);
        }
        self.apply_mutations(batch)?;
        self.pre_drain_passes = self.pre_drain_passes.saturating_add(1);
        self.pre_drained_mutations = self.pre_drained_mutations.saturating_add(count);
        Ok(count)
    }

    /// Incrementally drains [`DeltaBuffer`] in bounded `batch_size` passes *off-lock*
    /// until `delta_buffer.len() <= tail_threshold` or `max_passes` passes complete.
    ///
    /// Returns the total number of mutations drained off-lock.
    ///
    /// # Errors
    /// Propagates any filter expansion or Reverse Map storage error.
    pub fn pre_drain_until_small(
        &mut self,
        tail_threshold: usize,
        batch_size: usize,
        max_passes: usize,
    ) -> AerisResult<usize> {
        let effective_batch = batch_size.max(1);
        let mut drained_total = 0usize;
        let mut passes = 0usize;

        while passes < max_passes && self.delta_buffer.len() > tail_threshold {
            let drained = self.pre_drain_batch(effective_batch)?;
            if drained == 0 {
                break;
            }
            drained_total = drained_total.saturating_add(drained);
            passes += 1;
        }

        debug!(
            reason = ?self.bg.reason,
            passes,
            drained_total,
            remaining_tail = self.delta_buffer.len(),
            tail_threshold,
            "completed off-lock incremental DeltaBuffer pre-drain"
        );
        Ok(drained_total)
    }

    /// Verifies zero false negatives across all still-live delta-inserted keys drained
    /// so far, allowing verification of large pre-drained backlogs to run *off-lock*
    /// prior to acquiring swap/quiescence locks.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if any live delta-inserted key
    /// returns `false` on `query_point`.
    pub fn verify_pre_drained_keys(&mut self) -> AerisResult<()> {
        for key in self.unverified_inserted_keys.drain(..) {
            if self.bg.loaded_keys.contains(&key) && !self.bg.filter.query_point(key) {
                return Err(AerisError::InternalInvariant(format!(
                    "rebuilt filter produced false negative for pre-drained live key {key}"
                )));
            }
        }
        Ok(())
    }

    /// Completes the final locked phase: atomically drains and deactivates any
    /// residual [`DeltaBuffer`] tail, replays catch-up [`FalsePositiveLog`] entries
    /// using `verify_empty`, verifies zero false negatives, and returns [`CompletedRebuild`].
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on false-negative detection, or
    /// propagates storage, filter, or FP-log errors.
    pub fn finish_tail<F>(mut self, verify_empty: F) -> AerisResult<CompletedRebuild>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let expansions_before_tail = self.auto_expansions;
        let tail = self.delta_buffer.drain_and_deactivate();
        let tail_len = tail.len();
        if !tail.is_empty() {
            self.apply_mutations(tail)?;
        }

        self.finalize_with_verifier(expansions_before_tail, tail_len, verify_empty)
    }

    /// Variant of [`Self::finish_tail`] that uses the exact reconciled `live_keys`
    /// set (background snapshot plus all drained delta mutations) as the ground-truth
    /// emptiness oracle without requiring an extra Reverse Map full scan under lock.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on false-negative detection, or
    /// propagates storage, filter, or FP-log errors.
    pub fn finish_tail_with_live_keys(mut self) -> AerisResult<CompletedRebuild> {
        let expansions_before_tail = self.auto_expansions;
        let tail = self.delta_buffer.drain_and_deactivate();
        let tail_len = tail.len();
        if !tail.is_empty() {
            self.apply_mutations(tail)?;
        }

        let catchup_min_seq = if self.delta_inserts_applied > 0 || self.delta_deletes_applied > 0 {
            0
        } else {
            self.bg.fp_snapshot_seq
        };

        let live_keys_ref = &self.bg.loaded_keys;
        let fp_replay_catchup = self.bg.fp_log.replay_since(
            catchup_min_seq,
            &mut self.bg.filter,
            self.bg.dest_reverse_map.as_ref(),
            |range| {
                Ok(live_keys_ref
                    .range(range.start()..=range.end())
                    .next()
                    .is_none())
            },
        )?;
        self.auto_expansions += fp_replay_catchup.expansions_during_replay;

        self.verify_post_tail_and_build(expansions_before_tail, tail_len, fp_replay_catchup)
    }

    /// Applies a batch of FIFO [`DeltaMutation`]s idempotently onto `F_new` and
    /// `dest_reverse_map`, coalescing consecutive inserts into [`ReverseMapStore::append_batch`]
    /// and flushing pending Reverse Map inserts before any `expand()` or `Delete`.
    fn apply_mutations(&mut self, mutations: Vec<DeltaMutation>) -> AerisResult<()> {
        let mut rm_insert_batch: Vec<(aeris_types::Fingerprint, Key)> =
            Vec::with_capacity(mutations.len().min(DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE));

        for mutation in mutations {
            match mutation {
                DeltaMutation::Insert(key) => {
                    if self.bg.loaded_keys.insert(key) {
                        let fp = insert_with_auto_expand_flushing_rm(
                            &mut self.bg.filter,
                            self.bg.dest_reverse_map.as_ref(),
                            key,
                            &mut self.auto_expansions,
                            &mut rm_insert_batch,
                        )?;
                        rm_insert_batch.push((fp, key));
                        self.all_inserted_delta_keys.push(key);
                        self.unverified_inserted_keys.push(key);
                        self.delta_inserts_applied += 1;
                    }
                }
                DeltaMutation::Delete(key) => {
                    if self.bg.loaded_keys.remove(&key) {
                        if !rm_insert_batch.is_empty() {
                            self.bg.dest_reverse_map.append_batch(&rm_insert_batch)?;
                            rm_insert_batch.clear();
                        }
                        if let Some(fp) = self.bg.filter.delete(key)? {
                            self.bg.dest_reverse_map.remove_key(&fp, key)?;
                        }
                        self.delta_deletes_applied += 1;
                    }
                }
            }
        }

        if !rm_insert_batch.is_empty() {
            self.bg.dest_reverse_map.append_batch(&rm_insert_batch)?;
        }
        Ok(())
    }

    fn finalize_with_verifier<F>(
        mut self,
        expansions_before_tail: usize,
        tail_len: usize,
        verify_empty: F,
    ) -> AerisResult<CompletedRebuild>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let catchup_min_seq = if self.delta_inserts_applied > 0 || self.delta_deletes_applied > 0 {
            0
        } else {
            self.bg.fp_snapshot_seq
        };

        let fp_replay_catchup = self.bg.fp_log.replay_since(
            catchup_min_seq,
            &mut self.bg.filter,
            self.bg.dest_reverse_map.as_ref(),
            &verify_empty,
        )?;
        self.auto_expansions += fp_replay_catchup.expansions_during_replay;

        self.verify_post_tail_and_build(expansions_before_tail, tail_len, fp_replay_catchup)
    }

    fn verify_post_tail_and_build(
        self,
        expansions_before_tail: usize,
        tail_len: usize,
        fp_replay_catchup: FpReplayReport,
    ) -> AerisResult<CompletedRebuild> {
        let keys_to_verify: &[Key] = if self.auto_expansions > expansions_before_tail {
            &self.all_inserted_delta_keys
        } else {
            &self.unverified_inserted_keys
        };
        for &key in keys_to_verify {
            if self.bg.loaded_keys.contains(&key) && !self.bg.filter.query_point(key) {
                return Err(AerisError::InternalInvariant(format!(
                    "rebuilt filter produced false negative for live delta-inserted key {key}"
                )));
            }
        }

        let duration_ns = self.started_at.elapsed().as_nanos() as u64;
        let report = RebuildReport {
            reason: self.bg.reason,
            keys_loaded: self.bg.throttle_metrics.keys_paced as usize,
            delta_inserts_applied: self.delta_inserts_applied,
            delta_deletes_applied: self.delta_deletes_applied,
            fp_replay_initial: self.bg.fp_replay_initial,
            fp_replay_catchup,
            auto_expansions: self.auto_expansions,
            observed_bulk_load_speed: self.bg.observed_bulk_load_speed,
            throttle_metrics: self.bg.throttle_metrics,
            duration_ns,
        };

        info!(
            reason = ?report.reason,
            keys_loaded = report.keys_loaded,
            delta_inserts_applied = report.delta_inserts_applied,
            delta_deletes_applied = report.delta_deletes_applied,
            pre_drain_passes = self.pre_drain_passes,
            pre_drained_mutations = self.pre_drained_mutations,
            final_tail_mutations = tail_len,
            fp_collisions_resolved =
                report.fp_replay_initial.collisions_resolved
                    + report.fp_replay_catchup.collisions_resolved,
            auto_expansions = report.auto_expansions,
            final_q = self.bg.filter.config().q,
            final_m = self.bg.filter.config().m,
            duration_ns = report.duration_ns,
            "completed asynchronous rebuild and delta drain"
        );

        self.drain_guard.release();

        Ok(CompletedRebuild {
            filter: self.bg.filter,
            reverse_map: self.bg.dest_reverse_map,
            report,
        })
    }
}

/// Coordinates non-overlapping asynchronous rebuilds on a dedicated worker thread
/// (`"aeris-bg-rebuild"`), pairing throttled Reverse Map scans with [`DeltaBuffer`]
/// mutation capture and monotonic [`FalsePositiveLog`] replay.
pub struct AsyncRebuildCoordinator {
    delta_buffer: Arc<DeltaBuffer>,
    in_flight: Arc<Mutex<RebuildSlot>>,
}

impl Default for AsyncRebuildCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncRebuildCoordinator {
    /// Creates an idle [`AsyncRebuildCoordinator`] with a fresh inactive [`DeltaBuffer`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            delta_buffer: Arc::new(DeltaBuffer::new()),
            in_flight: Arc::new(Mutex::new(RebuildSlot::Idle)),
        }
    }

    /// Returns a shared handle to the coordinator's [`DeltaBuffer`].
    #[must_use]
    pub fn delta_buffer(&self) -> Arc<DeltaBuffer> {
        Arc::clone(&self.delta_buffer)
    }

    /// Returns `true` if a background rebuild is currently running or draining deltas.
    #[must_use]
    pub fn is_in_progress(&self) -> bool {
        !matches!(*self.in_flight.lock(), RebuildSlot::Idle)
    }

    /// Returns `true` if an in-flight background worker has completed its background
    /// phase and is ready for [`Self::prepare_finish`] / [`Self::finish_and_drain_deltas`]
    /// without long blocking.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(
            *self.in_flight.lock(),
            RebuildSlot::Running(ref job) if job.done.load(Ordering::Acquire)
        )
    }

    /// Launches a background rebuild constructing `F_new` under `target_config`
    /// from `source_reverse_map` into `dest_reverse_map`.
    ///
    /// Immediately activates [`DeltaBuffer`] before spawning `"aeris-bg-rebuild"` so
    /// concurrent foreground mutations are captured without loss.
    ///
    /// # Errors
    /// - Returns [`AerisError::RebuildInProgress`] if a rebuild is already in flight.
    /// - Returns [`AerisError::InvalidConfig`] if `target_config` or `throttle_config` is invalid.
    /// - Returns [`AerisError::InternalInvariant`] if the OS refuses to spawn the worker thread.
    pub fn start_rebuild(
        &self,
        source_reverse_map: Arc<dyn ReverseMapStore>,
        dest_reverse_map: Arc<dyn ReverseMapStore>,
        fp_log: Arc<FalsePositiveLog>,
        target_config: FilterConfig,
        reason: RebuildReason,
        throttle_config: IoThrottleConfig,
    ) -> AerisResult<()> {
        target_config.validate()?;
        throttle_config.validate()?;

        let mut slot = self.in_flight.lock();
        if !matches!(*slot, RebuildSlot::Idle) {
            return Err(AerisError::RebuildInProgress);
        }

        // Activate DeltaBuffer BEFORE spawning the worker so every mutation arriving
        // concurrently with or after the Reverse Map snapshot is captured.
        self.delta_buffer.activate();
        let done = Arc::new(AtomicBool::new(false));
        let done_worker = Arc::clone(&done);
        let started_at = Instant::now();

        info!(
            ?reason,
            target_q = target_config.q,
            target_r = target_config.r,
            target_m = target_config.m,
            max_keys_per_sec = throttle_config.max_keys_per_sec,
            batch_size = throttle_config.batch_size,
            "starting asynchronous background rebuild"
        );

        let spawn_res = thread::Builder::new()
            .name(BG_REBUILD_THREAD_NAME.to_string())
            .spawn(move || {
                let _signal = CompletionSignal(done_worker);
                execute_background_rebuild(
                    source_reverse_map,
                    dest_reverse_map,
                    fp_log,
                    target_config,
                    reason,
                    throttle_config,
                )
            });

        match spawn_res {
            Ok(handle) => {
                *slot = RebuildSlot::Running(InFlightRebuild {
                    reason,
                    done,
                    handle,
                    started_at,
                });
                Ok(())
            }
            Err(err) => {
                self.delta_buffer.deactivate();
                error!(error = %err, "failed to spawn background rebuild thread");
                Err(AerisError::InternalInvariant(format!(
                    "failed to spawn '{BG_REBUILD_THREAD_NAME}' thread: {err}"
                )))
            }
        }
    }

    /// Joins the active background rebuild worker off-lock and transitions the
    /// coordinator into the [`StagedRebuild`] two-phase draining stage while keeping
    /// [`DeltaBuffer`] active for concurrent writers.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if no rebuild worker is running or
    /// if the worker thread panicked; propagates any background worker error.
    pub fn prepare_finish(&self) -> AerisResult<StagedRebuild> {
        let job = {
            let mut slot = self.in_flight.lock();
            match std::mem::replace(&mut *slot, RebuildSlot::Draining) {
                RebuildSlot::Running(job) => job,
                prev => {
                    *slot = prev;
                    return Err(AerisError::InternalInvariant(
                        "prepare_finish called with no active rebuild worker".to_string(),
                    ));
                }
            }
        };

        let drain_guard =
            DrainGuard::new(Arc::clone(&self.in_flight), Arc::clone(&self.delta_buffer));

        let bg_result = match job.handle.join() {
            Ok(res) => res,
            Err(_) => {
                error!(reason = ?job.reason, "background rebuild thread panicked");
                return Err(AerisError::InternalInvariant(
                    "background rebuild worker thread panicked".to_string(),
                ));
            }
        };

        let bg = match bg_result {
            Ok(out) => out,
            Err(err) => {
                error!(reason = ?job.reason, error = %err, "background rebuild worker failed");
                return Err(err);
            }
        };

        let auto_expansions = bg.auto_expansions;
        Ok(StagedRebuild {
            bg,
            started_at: job.started_at,
            delta_buffer: Arc::clone(&self.delta_buffer),
            drain_guard,
            delta_inserts_applied: 0,
            delta_deletes_applied: 0,
            auto_expansions,
            all_inserted_delta_keys: Vec::new(),
            unverified_inserted_keys: Vec::new(),
            pre_drain_passes: 0,
            pre_drained_mutations: 0,
        })
    }

    /// Joins the active background rebuild worker, runs incremental off-lock delta
    /// pre-draining, atomically drains and deactivates [`DeltaBuffer`], replays
    /// catch-up [`FalsePositiveLog`] entries, and verifies zero false negatives for
    /// all live delta-inserted keys.
    ///
    /// `verify_empty` MUST return `Ok(true)` iff `range` currently contains zero live
    /// keys in the authoritative store (including all drained delta mutations).
    ///
    /// # Errors
    /// - Returns [`AerisError::InternalInvariant`] if no rebuild is in progress, if the
    ///   worker panicked, or if a post-drain false negative is detected.
    /// - Propagates any storage, filter, or FP log error encountered during rebuild or drain.
    pub fn finish_and_drain_deltas<F>(&self, verify_empty: F) -> AerisResult<CompletedRebuild>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let mut staged = self.prepare_finish()?;
        staged.pre_drain_until_small(
            DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
            DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
            DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
        )?;
        staged.verify_pre_drained_keys()?;
        staged.finish_tail(verify_empty)
    }
}

/// Core execution body for the `"aeris-bg-rebuild"` background thread.
fn execute_background_rebuild(
    source_reverse_map: Arc<dyn ReverseMapStore>,
    dest_reverse_map: Arc<dyn ReverseMapStore>,
    fp_log: Arc<FalsePositiveLog>,
    target_config: FilterConfig,
    reason: RebuildReason,
    throttle_config: IoThrottleConfig,
) -> AerisResult<BackgroundRebuildOutput> {
    let phase_start = Instant::now();
    let mut throttler = IoThrottler::new(throttle_config)?;
    let batch_size = throttler.batch_size() as u64;

    // 1. Stream all live keys from source Reverse Map with token-bucket I/O throttling.
    let mut loaded_vec: Vec<Key> = Vec::new();
    let mut batch_count = 0u64;
    {
        let iter = source_reverse_map.iter_all_keys()?;
        for key_res in iter {
            let key = key_res?;
            loaded_vec.push(key);
            batch_count += 1;
            if batch_count >= batch_size {
                throttler.pace(batch_count);
                batch_count = 0;
            }
        }
        if batch_count > 0 {
            throttler.pace(batch_count);
        }
    }

    let loaded_keys: BTreeSet<Key> = loaded_vec.iter().copied().collect();

    // 2. Bulk-load F_new and populate destination Reverse Map, auto-expanding if full.
    let bulk_start = Instant::now();
    let mut cfg = target_config;
    let mut auto_expansions = 0usize;
    let mut new_filter = loop {
        match AerisFilter::bulk_load(cfg.clone(), &loaded_vec, dest_reverse_map.as_ref()) {
            Ok(f) => break f,
            Err(AerisError::FilterFull { .. })
                if auto_expansions < usize::from(MAX_QUOTIENT_BITS) =>
            {
                cfg = cfg.expanded()?;
                auto_expansions += 1;
                debug!(
                    new_q = cfg.q,
                    keys = loaded_vec.len(),
                    "auto-expanded target_config during background bulk_load"
                );
            }
            Err(err) => return Err(err),
        }
    };
    let bulk_elapsed_secs = bulk_start
        .elapsed()
        .as_secs_f64()
        .max(MIN_BULK_LOAD_ELAPSED_SECS);
    let observed_bulk_load_speed = (loaded_vec.len().max(1) as f64) / bulk_elapsed_secs;

    // 3. Replay all currently logged FPs onto F_new against the exact loaded snapshot key set.
    let fp_replay_initial =
        fp_log.replay_onto(&mut new_filter, dest_reverse_map.as_ref(), |r| {
            Ok(loaded_keys.range(r.start()..=r.end()).next().is_none())
        })?;
    auto_expansions += fp_replay_initial.expansions_during_replay;
    let fp_snapshot_seq = fp_replay_initial.last_replayed_seq;

    let duration_ns = phase_start.elapsed().as_nanos() as u64;
    let throttle_metrics = throttler.metrics();
    info!(
        ?reason,
        keys_loaded = loaded_vec.len(),
        fp_collisions_resolved = fp_replay_initial.collisions_resolved,
        auto_expansions,
        observed_bulk_load_speed,
        throttle_events = throttle_metrics.throttle_events,
        duration_ns,
        "finished background rebuild worker phase"
    );

    Ok(BackgroundRebuildOutput {
        filter: new_filter,
        dest_reverse_map,
        fp_log,
        loaded_keys,
        fp_snapshot_seq,
        fp_replay_initial,
        auto_expansions,
        observed_bulk_load_speed,
        throttle_metrics,
        reason,
    })
}

/// Inserts `key` into `filter`, flushing `pending_rm_batch` into `reverse_map`
/// before invoking `filter.expand(reverse_map)` whenever [`AerisError::FilterFull`]
/// occurs so selective rejuvenation always observes every previously inserted key.
fn insert_with_auto_expand_flushing_rm(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    key: Key,
    expansions_counter: &mut usize,
    pending_rm_batch: &mut Vec<(aeris_types::Fingerprint, Key)>,
) -> AerisResult<aeris_types::Fingerprint> {
    let mut local_expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                if !pending_rm_batch.is_empty() {
                    reverse_map.append_batch(pending_rm_batch)?;
                    pending_rm_batch.clear();
                }
                filter.expand(reverse_map)?;
                local_expansions += 1;
                *expansions_counter += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Inserts `key` into `filter`, automatically invoking `filter.expand(reverse_map)`
/// whenever [`AerisError::FilterFull`] occurs.
fn insert_with_auto_expand(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    key: Key,
    expansions_counter: &mut usize,
) -> AerisResult<aeris_types::Fingerprint> {
    let mut local_expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                filter.expand(reverse_map)?;
                local_expansions += 1;
                *expansions_counter += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Adapts `filter` to `range`, automatically invoking `filter.expand(reverse_map)`
/// whenever [`AerisError::FilterFull`] occurs.
fn adapt_with_auto_expand(
    filter: &mut AerisFilter,
    range: QueryRange,
    reverse_map: &dyn ReverseMapStore,
) -> AerisResult<()> {
    let mut local_expansions = 0u8;
    loop {
        match filter.adapt(range, reverse_map) {
            Ok(_) => return Ok(()),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                filter.expand(reverse_map)?;
                local_expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Configuration bundle for [`ControlledAerisEngine`].
#[derive(Clone, Debug)]
pub struct ControlledEngineConfig {
    /// Initial [`FilterConfig`] for the active [`AerisFilter`].
    pub filter_config: FilterConfig,
    /// Configuration for [`SlidingWindowEstimator`].
    pub estimator_config: EstimatorConfig,
    /// Analytical cost model for [`DiscreteCostOptimizer`].
    pub cost_model: CostModelParams,
    /// Configuration for [`ProactiveTrigger`].
    pub trigger_config: ProactiveTriggerConfig,
    /// Configuration for [`FalsePositiveLog`].
    pub fp_log_config: FpLogConfig,
    /// I/O rate-limiting policy for background rebuild scans.
    pub throttle_config: IoThrottleConfig,
    /// RocksDB options for [`ReferenceKvStore`].
    pub kv_store_config: KvStoreConfig,
    /// RocksDB options for generational [`RocksDbReverseMap`] instances.
    pub reverse_map_config: ReverseMapConfig,
    /// Enables dynamic-`R` workload estimation and dynamic-`R` rebuild recommendations
    /// (default `true`). Setting `false` activates the zero-overhead static-`R` fast-path
    /// bypass while keeping monotonic FP adaptation and proactive capacity triggers active.
    pub dynamic_r_enabled: bool,
}

impl ControlledEngineConfig {
    /// Creates a [`ControlledEngineConfig`] with production defaults around `filter_config`.
    #[must_use]
    pub fn new(filter_config: FilterConfig) -> Self {
        Self {
            filter_config,
            estimator_config: EstimatorConfig::default(),
            cost_model: CostModelParams::default(),
            trigger_config: ProactiveTriggerConfig::default(),
            fp_log_config: FpLogConfig::default(),
            throttle_config: IoThrottleConfig::default(),
            kv_store_config: KvStoreConfig::default(),
            reverse_map_config: ReverseMapConfig::default(),
            dynamic_r_enabled: true,
        }
    }

    /// Overrides [`Self::dynamic_r_enabled`] and synchronizes `estimator_config.enabled`.
    #[must_use]
    pub const fn with_dynamic_r_enabled(mut self, enabled: bool) -> Self {
        self.dynamic_r_enabled = enabled;
        self.estimator_config.enabled = enabled;
        self
    }

    /// Creates a [`ControlledEngineConfig`] configured for static-`R` execution only:
    /// sets `dynamic_r_enabled = false` and `estimator_config.enabled = false` to bypass
    /// per-query workload estimation and suppress dynamic-`R` rebuilds.
    #[must_use]
    pub fn static_r_only(filter_config: FilterConfig) -> Self {
        Self::new(filter_config).with_dynamic_r_enabled(false)
    }

    /// Validates every sub-component configuration.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on the first violated invariant.
    pub fn validate(&self) -> AerisResult<()> {
        self.filter_config.validate()?;
        self.estimator_config.validate()?;
        self.trigger_config.validate()?;
        self.fp_log_config.validate()?;
        self.throttle_config.validate()?;
        Ok(())
    }
}

/// Result of answering a closed range query through [`ControlledAerisEngine::query_range`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryOutcome {
    /// Ground-truth emptiness answer (`true` = at least one live key exists in range).
    pub non_empty: bool,
    /// Whether the RAM [`AerisFilter`] returned `true` (always `false` when bypassed).
    pub filter_returned_positive: bool,
    /// `true` iff the filter returned `true` while [`ReferenceKvStore`] confirmed empty.
    pub was_false_positive: bool,
    /// `true` iff the filter was bypassed directly to [`ReferenceKvStore`].
    pub bypassed_filter: bool,
    /// `true` iff `adapt()` and [`FalsePositiveLog::record`] ran to eliminate the FP.
    pub adapted: bool,
}

/// Combined Control Plane evaluation produced by [`ControlledAerisEngine::evaluate_control_plane`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPlaneDecision {
    /// Sliding-window workload statistics snapshot.
    pub workload: WorkloadStats,
    /// Discrete Cost Optimizer recommendation (`m*`, `r*`, bypass, rebuild gate).
    pub optimization: OptimizationDecision,
    /// Proactive EWMA expansion / contraction state-machine evaluation.
    pub trigger: TriggerEvaluation,
}

/// Pending destination RocksDB Reverse Map awaiting atomic swap at rebuild finish.
struct PendingReverseMap {
    generation: u64,
    store: Arc<RocksDbReverseMap>,
}

/// Unified high-level facade coupling the real Data Plane ([`AerisFilter`]),
/// real Storage Plane ([`ReferenceKvStore`] + [`RocksDbReverseMap`]), and
/// complete Deterministic Control Plane ([`SlidingWindowEstimator`],
/// [`DiscreteCostOptimizer`], [`ProactiveTrigger`], [`FalsePositiveLog`], and
/// [`AsyncRebuildCoordinator`]).
pub struct ControlledAerisEngine {
    root_dir: PathBuf,
    epoch: Epoch,
    filter: AerisFilter,
    kv_store: ReferenceKvStore,
    reverse_map: Arc<RocksDbReverseMap>,
    reverse_map_config: ReverseMapConfig,
    reverse_map_generation: u64,
    pending_dest_rm: Option<PendingReverseMap>,
    estimator: SlidingWindowEstimator,
    optimizer: DiscreteCostOptimizer,
    cost_model: CostModelParams,
    trigger: ProactiveTrigger,
    fp_log: Arc<FalsePositiveLog>,
    coordinator: AsyncRebuildCoordinator,
    throttle_config: IoThrottleConfig,
    dynamic_r_enabled: bool,
    bypass_active: bool,
    pending_contraction: bool,
    total_rebuilds: u64,
    total_contractions: u64,
    total_false_positives: u64,
    total_bypassed_queries: u64,
}

impl ControlledAerisEngine {
    /// Opens or initializes a [`ControlledAerisEngine`] rooted at `root_dir`.
    ///
    /// If `kv_store` already contains persisted keys from a previous session, the
    /// RAM [`AerisFilter`] and generational [`RocksDbReverseMap`] are reconstructed
    /// from the persisted key set and any logged false positives are replayed.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`], [`AerisError::StorageError`],
    /// [`AerisError::WalIoError`], or [`AerisError::WalCorruption`] on failure.
    pub fn open(root_dir: impl AsRef<Path>, config: ControlledEngineConfig) -> AerisResult<Self> {
        config.validate()?;
        let root_dir = root_dir.as_ref().to_path_buf();
        fs::create_dir_all(&root_dir).map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to create ControlledAerisEngine directory '{}': {err}",
                root_dir.display()
            ))
        })?;

        let kv_store = ReferenceKvStore::open_with_config(
            root_dir.join(ENGINE_KV_DIR_NAME),
            config.kv_store_config.clone(),
        )?;

        let initial_gen = 0u64;
        let rm_path = reverse_map_gen_path(&root_dir, initial_gen);
        let reverse_map = Arc::new(RocksDbReverseMap::open(
            &rm_path,
            config.reverse_map_config.clone(),
        )?);

        let fp_log = Arc::new(FalsePositiveLog::open(
            root_dir.join(ENGINE_FP_LOG_NAME),
            config.fp_log_config,
        )?);

        let existing_keys = kv_store.scan_all_keys()?;
        let mut filter = if existing_keys.is_empty() {
            AerisFilter::new(config.filter_config.clone())?
        } else {
            let mut cfg = config.filter_config.clone();
            let mut expansions = 0usize;
            let mut built = loop {
                match AerisFilter::bulk_load(cfg.clone(), &existing_keys, reverse_map.as_ref()) {
                    Ok(f) => break f,
                    Err(AerisError::FilterFull { .. })
                        if expansions < usize::from(MAX_QUOTIENT_BITS) =>
                    {
                        cfg = cfg.expanded()?;
                        expansions += 1;
                    }
                    Err(err) => return Err(err),
                }
            };
            if !fp_log.is_empty() {
                let _ = fp_log.replay_onto(&mut built, reverse_map.as_ref(), |r| {
                    Ok(!kv_store.range_non_empty(r)?)
                })?;
            }
            built
        };

        let mut estimator_config = config.estimator_config.clone();
        if !config.dynamic_r_enabled {
            estimator_config.enabled = false;
        }
        let estimator = SlidingWindowEstimator::with_initial_span(
            estimator_config,
            filter.config().max_range_r,
        )?;
        let trigger = ProactiveTrigger::new(config.trigger_config.clone())?;

        // Ensure filter is verified on initial load (no-op when empty).
        let _ = &mut filter;

        info!(
            root_dir = %root_dir.display(),
            q = filter.config().q,
            r = filter.config().r,
            m = filter.config().m,
            dynamic_r_enabled = config.dynamic_r_enabled,
            loaded_keys = existing_keys.len(),
            fp_log_entries = fp_log.len(),
            "opened ControlledAerisEngine"
        );

        Ok(Self {
            root_dir,
            epoch: 1,
            filter,
            kv_store,
            reverse_map,
            reverse_map_config: config.reverse_map_config,
            reverse_map_generation: initial_gen,
            pending_dest_rm: None,
            estimator,
            optimizer: DiscreteCostOptimizer::new(),
            cost_model: config.cost_model,
            trigger,
            fp_log,
            coordinator: AsyncRebuildCoordinator::new(),
            throttle_config: config.throttle_config,
            dynamic_r_enabled: config.dynamic_r_enabled,
            bypass_active: false,
            pending_contraction: false,
            total_rebuilds: 0,
            total_contractions: 0,
            total_false_positives: 0,
            total_bypassed_queries: 0,
        })
    }

    /// Inserts or updates `(key, value)` across [`AerisFilter`], [`ReferenceKvStore`],
    /// and [`RocksDbReverseMap`], mirroring the insert into [`DeltaBuffer`] if an
    /// asynchronous rebuild is in flight.
    ///
    /// Automatically expands `filter` (`q -> q + 1`) if inserting `key` would exceed
    /// `max_load_factor`.
    ///
    /// # Errors
    /// Propagates filter expansion or RocksDB storage errors.
    pub fn insert(&mut self, key: Key, value: &[u8]) -> AerisResult<()> {
        let already_exists = self.kv_store.get(key)?.is_some();
        self.kv_store.put(key, value)?;
        if already_exists {
            return Ok(());
        }

        let mut dummy_expansions = 0usize;
        let fp = insert_with_auto_expand(
            &mut self.filter,
            self.reverse_map.as_ref(),
            key,
            &mut dummy_expansions,
        )?;
        self.reverse_map.append_key(&fp, key)?;
        let _ = self.coordinator.delta_buffer().record_insert(key);
        Ok(())
    }

    /// Deletes `key` from [`ReferenceKvStore`], [`AerisFilter`], and
    /// [`RocksDbReverseMap`], mirroring the deletion into [`DeltaBuffer`] if an
    /// asynchronous rebuild is in flight.
    ///
    /// Returns `Ok(true)` if `key` existed and was deleted, or `Ok(false)` if absent.
    ///
    /// # Errors
    /// Propagates RocksDB or filter errors.
    pub fn delete(&mut self, key: Key) -> AerisResult<bool> {
        if self.kv_store.get(key)?.is_none() {
            return Ok(false);
        }
        self.kv_store.delete(key)?;
        if let Some(fp) = self.filter.delete(key)? {
            self.reverse_map.remove_key(&fp, key)?;
        }
        let _ = self.coordinator.delta_buffer().record_delete(key);
        Ok(true)
    }

    /// Performs a direct zero-overhead RAM [`AerisFilter::query_range`] probe when
    /// filter bypass is inactive, returning `true` immediately if bypass is active
    /// so callers preserve the zero-false-negative emptiness contract.
    ///
    /// Performs zero storage lookups, zero estimator mutations, and zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_range_filter_only(&self, range: QueryRange) -> bool {
        if self.bypass_active {
            return true;
        }
        self.filter.query_range(range)
    }

    /// Answers a closed range emptiness query `[a, b]` using the RAM [`AerisFilter`]
    /// (unless filter bypass is active), automatically confirming positives against
    /// [`ReferenceKvStore`], logging confirmed false positives to [`FalsePositiveLog`],
    /// and monotonically adapting the active filter.
    ///
    /// Records every query into [`SlidingWindowEstimator`] when `dynamic_r_enabled` is
    /// `true`; skips estimator recording entirely when `dynamic_r_enabled` is `false`.
    ///
    /// # Errors
    /// Propagates RocksDB, [`FalsePositiveLog`], or filter adaptation errors.
    pub fn query_range(&mut self, range: QueryRange) -> AerisResult<QueryOutcome> {
        if self.bypass_active {
            let non_empty = self.kv_store.range_non_empty(range)?;
            if self.dynamic_r_enabled {
                self.estimator.record_query(range, false);
            }
            self.total_bypassed_queries = self.total_bypassed_queries.saturating_add(1);
            return Ok(QueryOutcome {
                non_empty,
                filter_returned_positive: false,
                was_false_positive: false,
                bypassed_filter: true,
                adapted: false,
            });
        }

        let filter_hit = self.filter.query_range(range);
        if !filter_hit {
            if self.dynamic_r_enabled {
                self.estimator.record_query(range, false);
            }
            return Ok(QueryOutcome {
                non_empty: false,
                filter_returned_positive: false,
                was_false_positive: false,
                bypassed_filter: false,
                adapted: false,
            });
        }

        let non_empty = self.kv_store.range_non_empty(range)?;
        if non_empty {
            if self.dynamic_r_enabled {
                self.estimator.record_query(range, false);
            }
            return Ok(QueryOutcome {
                non_empty: true,
                filter_returned_positive: true,
                was_false_positive: false,
                bypassed_filter: false,
                adapted: false,
            });
        }

        // Confirmed false positive: record into estimator (if dynamic-R enabled) &
        // persistent FP Log, then adapt monotonically.
        self.total_false_positives = self.total_false_positives.saturating_add(1);
        if self.dynamic_r_enabled {
            self.estimator.record_query(range, true);
        }
        let _ = self.fp_log.record(range, current_unix_ms())?;
        adapt_with_auto_expand(&mut self.filter, range, self.reverse_map.as_ref())?;

        Ok(QueryOutcome {
            non_empty: false,
            filter_returned_positive: true,
            was_false_positive: true,
            bypassed_filter: false,
            adapted: true,
        })
    }

    /// Feeds an observed key ingestion rate sample `(count, elapsed_secs)` into the
    /// [`SlidingWindowEstimator`] EWMA tracker.
    pub fn record_ingest_sample(&mut self, count: u64, elapsed_secs: f64) {
        self.estimator.record_ingest(count, elapsed_secs);
    }

    /// Evaluates the full Deterministic Control Plane ([`SlidingWindowEstimator`],
    /// [`DiscreteCostOptimizer`], and [`ProactiveTrigger`]), updating the filter
    /// bypass state and returning a [`ControlPlaneDecision`].
    ///
    /// When `dynamic_r_enabled` is `false` (static-`R` mode), forces
    /// `optimization.should_rebuild = false` while leaving [`ProactiveTrigger`]
    /// capacity expansion and contraction evaluation completely intact.
    pub fn evaluate_control_plane(&mut self) -> ControlPlaneDecision {
        let workload = self.estimator.snapshot();
        let mut optimization =
            self.optimizer
                .optimize(&workload, self.filter.config(), &self.cost_model);
        if !self.dynamic_r_enabled {
            optimization.should_rebuild = false;
        }
        self.bypass_active = optimization.should_bypass_filter;

        let trigger = self.trigger.evaluate(&TriggerInput {
            capacity_slots: self.filter.capacity(),
            current_q: self.filter.config().q,
            current_alpha: self.filter.load_factor(),
            ingestion_rate_ewma: workload.ingestion_rate_ewma,
            reverse_map_entries: self.reverse_map.entry_count_estimate(),
            bulk_load_speed: workload.bulk_load_speed,
        });

        match trigger.action {
            TriggerAction::Contract => self.pending_contraction = true,
            TriggerAction::ProactiveExpand => self.pending_contraction = false,
            TriggerAction::None => {
                if self.filter.load_factor() > self.trigger.config().contraction_alpha_threshold {
                    self.pending_contraction = false;
                }
            }
        }

        ControlPlaneDecision {
            workload,
            optimization,
            trigger,
        }
    }

    /// Launches a non-blocking asynchronous background rebuild targeting `target_config`
    /// into a fresh generational [`RocksDbReverseMap`].
    ///
    /// While in flight, foreground `insert`, `delete`, and `query_range` calls continue
    /// serving off `F_old` while capturing mutations in [`DeltaBuffer`] and new false
    /// positives in [`FalsePositiveLog`].
    ///
    /// # Errors
    /// Returns [`AerisError::RebuildInProgress`] if a rebuild is already active,
    /// [`AerisError::InvalidConfig`] if `target_config` is invalid, or
    /// [`AerisError::StorageError`] if creating the destination Reverse Map fails.
    pub fn trigger_async_rebuild(
        &mut self,
        target_config: FilterConfig,
        reason: RebuildReason,
    ) -> AerisResult<()> {
        if self.coordinator.is_in_progress() {
            return Err(AerisError::RebuildInProgress);
        }
        target_config.validate()?;

        let next_gen = self.reverse_map_generation.saturating_add(1);
        let dest_path = reverse_map_gen_path(&self.root_dir, next_gen);
        if dest_path.exists() {
            let _ = fs::remove_dir_all(&dest_path);
        }
        let dest_rm = Arc::new(RocksDbReverseMap::open(
            &dest_path,
            self.reverse_map_config.clone(),
        )?);

        let source_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&self.reverse_map) as _;
        let dest_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&dest_rm) as _;

        self.coordinator.start_rebuild(
            source_dyn,
            dest_dyn,
            Arc::clone(&self.fp_log),
            target_config,
            reason,
            self.throttle_config,
        )?;

        self.pending_dest_rm = Some(PendingReverseMap {
            generation: next_gen,
            store: dest_rm,
        });
        Ok(())
    }

    /// Non-blocking poll that completes and swaps in `F_new` iff the background
    /// worker has finished, returning `Ok(None)` while the worker is still running
    /// or when no rebuild is active.
    ///
    /// # Errors
    /// Propagates any rebuild, delta-drain, or FP-replay error.
    pub fn complete_rebuild_if_ready(&mut self) -> AerisResult<Option<RebuildReport>> {
        if !self.coordinator.is_ready() {
            return Ok(None);
        }
        let report = self.finish_rebuild()?;
        Ok(Some(report))
    }

    /// Blocks until the active background rebuild completes, drains [`DeltaBuffer`],
    /// replays catch-up [`FalsePositiveLog`] entries, atomically swaps `F_old -> F_new`,
    /// updates bulk-load EWMA, and reclaims the obsolete Reverse Map generation.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if no rebuild is active or propagates
    /// worker/storage errors.
    pub fn finish_rebuild(&mut self) -> AerisResult<RebuildReport> {
        let kv_ref = &self.kv_store;
        let completed = match self
            .coordinator
            .finish_and_drain_deltas(|range| Ok(!kv_ref.range_non_empty(range)?))
        {
            Ok(done) => done,
            Err(err) => {
                self.pending_dest_rm = None;
                return Err(err);
            }
        };

        let pending = self.pending_dest_rm.take().ok_or_else(|| {
            AerisError::InternalInvariant(
                "missing pending destination ReverseMap during rebuild swap".to_string(),
            )
        })?;

        let old_rm_path = self.reverse_map.path().to_path_buf();
        let old_rm = std::mem::replace(&mut self.reverse_map, pending.store);
        drop(completed.reverse_map);
        drop(old_rm);

        // Best-effort physical cleanup of superseded Reverse Map generation directory.
        if old_rm_path != self.reverse_map.path() {
            let _ = fs::remove_dir_all(&old_rm_path);
        }

        self.filter = completed.filter;
        self.reverse_map_generation = pending.generation;
        self.epoch = self.epoch.saturating_add(1);
        self.total_rebuilds = self.total_rebuilds.saturating_add(1);
        if matches!(completed.report.reason, RebuildReason::Contraction { .. }) {
            self.total_contractions = self.total_contractions.saturating_add(1);
            self.pending_contraction = false;
        }
        self.estimator
            .record_bulk_load_speed(completed.report.observed_bulk_load_speed);

        info!(
            epoch = self.epoch,
            reason = ?completed.report.reason,
            new_q = self.filter.config().q,
            new_r = self.filter.config().r,
            new_m = self.filter.config().m,
            load_factor = self.filter.load_factor(),
            total_rebuilds = self.total_rebuilds,
            "swapped active AerisFilter to newly rebuilt generation"
        );
        Ok(completed.report)
    }

    /// Executes a `q -> q - 1` filter contraction if [`ProactiveTrigger`] has
    /// confirmed sustained underutilization (either during an earlier
    /// [`Self::evaluate_control_plane`] call or on this check) and no rebuild is
    /// currently in flight.
    ///
    /// Returns `Ok(Some(report))` if contraction ran, or `Ok(None)` otherwise.
    ///
    /// # Errors
    /// Propagates configuration, storage, or rebuild errors.
    pub fn contract_if_needed(&mut self) -> AerisResult<Option<RebuildReport>> {
        if self.coordinator.is_in_progress() {
            return Ok(None);
        }

        if !self.pending_contraction {
            let stats = self.estimator.snapshot();
            let eval = self.trigger.evaluate(&TriggerInput {
                capacity_slots: self.filter.capacity(),
                current_q: self.filter.config().q,
                current_alpha: self.filter.load_factor(),
                ingestion_rate_ewma: stats.ingestion_rate_ewma,
                reverse_map_entries: self.reverse_map.entry_count_estimate(),
                bulk_load_speed: stats.bulk_load_speed,
            });
            if eval.action == TriggerAction::Contract {
                self.pending_contraction = true;
            }
        }

        if !self.pending_contraction {
            return Ok(None);
        }

        self.pending_contraction = false;
        let old_q = self.filter.config().q;
        let target_config = self.filter.config().contracted()?;
        let new_q = target_config.q;

        self.trigger_async_rebuild(target_config, RebuildReason::Contraction { old_q, new_q })?;
        let report = self.finish_rebuild()?;
        Ok(Some(report))
    }

    /// Returns the monotonic configuration epoch (incremented on every filter swap).
    #[inline]
    #[must_use]
    pub const fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Returns a shared reference to the currently active [`AerisFilter`].
    #[inline]
    #[must_use]
    pub fn filter(&self) -> &AerisFilter {
        &self.filter
    }

    /// Returns a shared reference to the primary [`ReferenceKvStore`].
    #[inline]
    #[must_use]
    pub fn kv_store(&self) -> &ReferenceKvStore {
        &self.kv_store
    }

    /// Returns a shared handle to the currently active [`RocksDbReverseMap`].
    #[inline]
    #[must_use]
    pub fn reverse_map(&self) -> Arc<RocksDbReverseMap> {
        Arc::clone(&self.reverse_map)
    }

    /// Returns a shared reference to the [`SlidingWindowEstimator`].
    #[inline]
    #[must_use]
    pub fn estimator(&self) -> &SlidingWindowEstimator {
        &self.estimator
    }

    /// Returns a mutable reference to the [`SlidingWindowEstimator`].
    #[inline]
    pub fn estimator_mut(&mut self) -> &mut SlidingWindowEstimator {
        &mut self.estimator
    }

    /// Returns the active [`DiscreteCostOptimizer`].
    #[inline]
    #[must_use]
    pub const fn optimizer(&self) -> DiscreteCostOptimizer {
        self.optimizer
    }

    /// Returns a shared reference to the [`ProactiveTrigger`].
    #[inline]
    #[must_use]
    pub fn trigger(&self) -> &ProactiveTrigger {
        &self.trigger
    }

    /// Returns a mutable reference to the [`ProactiveTrigger`].
    #[inline]
    pub fn trigger_mut(&mut self) -> &mut ProactiveTrigger {
        &mut self.trigger
    }

    /// Returns a shared handle to the persistent [`FalsePositiveLog`].
    #[inline]
    #[must_use]
    pub fn fp_log(&self) -> Arc<FalsePositiveLog> {
        Arc::clone(&self.fp_log)
    }

    /// Returns a shared reference to the [`AsyncRebuildCoordinator`].
    #[inline]
    #[must_use]
    pub fn coordinator(&self) -> &AsyncRebuildCoordinator {
        &self.coordinator
    }

    /// Returns `true` if dynamic-`R` workload estimation and rebuild recommendations
    /// are enabled (`false` = lightweight static-`R` fast-path bypass).
    #[inline]
    #[must_use]
    pub const fn is_dynamic_r_enabled(&self) -> bool {
        self.dynamic_r_enabled
    }

    /// Enables or disables dynamic-`R` workload estimation and rebuild recommendations,
    /// synchronizing the underlying [`SlidingWindowEstimator`].
    #[inline]
    pub fn set_dynamic_r_enabled(&mut self, enabled: bool) {
        self.dynamic_r_enabled = enabled;
        self.estimator.set_enabled(enabled);
    }

    /// Returns `true` if filter bypass is currently active.
    #[inline]
    #[must_use]
    pub const fn bypass_active(&self) -> bool {
        self.bypass_active
    }

    /// Returns cumulative completed background rebuilds (including contractions).
    #[inline]
    #[must_use]
    pub const fn total_rebuilds(&self) -> u64 {
        self.total_rebuilds
    }

    /// Returns cumulative completed filter contractions (`q -> q - 1`).
    #[inline]
    #[must_use]
    pub const fn total_contractions(&self) -> u64 {
        self.total_contractions
    }

    /// Returns cumulative confirmed false positives observed through [`Self::query_range`].
    #[inline]
    #[must_use]
    pub const fn total_false_positives(&self) -> u64 {
        self.total_false_positives
    }

    /// Returns cumulative queries served via direct storage bypass.
    #[inline]
    #[must_use]
    pub const fn total_bypassed_queries(&self) -> u64 {
        self.total_bypassed_queries
    }
}

/// Computes the filesystem path for Reverse Map generation `gen`.
fn reverse_map_gen_path(root_dir: &Path, generation: u64) -> PathBuf {
    root_dir.join(format!("{ENGINE_RM_GEN_PREFIX}{generation}"))
}

/// Returns current Unix timestamp in milliseconds (clamped to `0` if system clock precedes epoch).
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_buffer_activation_fifo_and_atomic_drain() {
        let buf = DeltaBuffer::new();
        assert!(!buf.is_active());
        assert!(buf.is_empty());

        // Inactive buffer drops writes cleanly.
        assert!(!buf.record_insert(10));
        assert!(!buf.record_delete(20));
        assert_eq!(buf.len(), 0);

        buf.activate();
        assert!(buf.is_active());
        assert!(buf.record_insert(10));
        assert!(buf.record_delete(20));
        assert!(buf.record_insert(30));
        assert_eq!(buf.len(), 3);

        let drained = buf.drain_all();
        assert_eq!(
            drained,
            vec![
                DeltaMutation::Insert(10),
                DeltaMutation::Delete(20),
                DeltaMutation::Insert(30),
            ]
        );
        assert!(buf.is_active());
        assert!(buf.is_empty());

        assert!(buf.record_delete(99));
        let final_ops = buf.drain_and_deactivate();
        assert_eq!(final_ops, vec![DeltaMutation::Delete(99)]);
        assert!(!buf.is_active());
        assert!(!buf.record_insert(100));
    }

    #[test]
    fn io_throttler_virtual_clock_enforces_exact_token_bucket_rate() {
        assert!(IoThrottleConfig::limited(0, 100).validate().is_err());
        assert!(IoThrottleConfig::limited(1000, 0).validate().is_err());

        // 10_000 keys/sec, batch size 1_000 -> each 1_000-key batch requires exactly 100 ms.
        let mut throttler = IoThrottler::new(IoThrottleConfig::limited(10_000, 1_000)).unwrap();

        // Batch 1 at t = 0: needs 100 ms (100_000_000 ns).
        let w1 = throttler.consume_at(1_000, 0);
        assert_eq!(w1, 100_000_000);

        // Batch 2 arrives at t = 140 ms (40 ms of I/O work elapsed after w1 -> 400 tokens refilled):
        // deficit = 600 keys -> needs 60 ms.
        let w2 = throttler.consume_at(1_000, 140_000_000);
        assert_eq!(w2, 60_000_000);

        // Batch 3 arrives after a 150 ms pause (t = 350 ms -> bucket refills to cap = 1_000):
        // zero wait required!
        let w3 = throttler.consume_at(1_000, 350_000_000);
        assert_eq!(w3, 0);

        let metrics = throttler.metrics();
        assert_eq!(metrics.keys_paced, 3_000);
        assert_eq!(metrics.throttle_events, 2);
        assert_eq!(metrics.total_throttled_ns, 160_000_000);
    }

    #[test]
    fn io_throttler_unthrottled_never_waits() {
        let mut throttler = IoThrottler::new(IoThrottleConfig::default()).unwrap();
        assert_eq!(throttler.consume_at(50_000, 0), 0);
        assert_eq!(throttler.consume_at(50_000, 1_000), 0);
        let metrics = throttler.metrics();
        assert_eq!(metrics.keys_paced, 100_000);
        assert_eq!(metrics.throttle_events, 0);
        assert_eq!(metrics.total_throttled_ns, 0);
    }

    #[test]
    fn two_phase_incremental_delta_drain_reduces_tail_and_preserves_zero_fn() {
        let dir = tempfile::TempDir::new().unwrap();
        let src_rm = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_src"), ReverseMapConfig::default())
                .unwrap(),
        );
        let dst_rm = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_dst"), ReverseMapConfig::default())
                .unwrap(),
        );
        let fp_log = Arc::new(
            FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
        );

        let base_cfg = FilterConfig::new(10, 15, 32, 20, 1.0 / 256.0).unwrap();
        let seed_keys: Vec<Key> = (0..400u64).map(|i| i * 64 + 7).collect();
        let _seed_filter =
            AerisFilter::bulk_load(base_cfg.clone(), &seed_keys, src_rm.as_ref()).unwrap();

        let coord = AsyncRebuildCoordinator::new();
        let target_cfg = base_cfg.expanded().unwrap(); // q=10 -> q=11
        coord
            .start_rebuild(
                Arc::clone(&src_rm) as Arc<dyn ReverseMapStore>,
                Arc::clone(&dst_rm) as Arc<dyn ReverseMapStore>,
                Arc::clone(&fp_log),
                target_cfg,
                RebuildReason::ProactiveExpansion {
                    old_q: 10,
                    new_q: 11,
                },
                IoThrottleConfig::default(),
            )
            .unwrap();

        // Enqueue a substantial delta backlog (2,500 inserts + 300 deletes) while rebuild runs.
        let delta_buf = coord.delta_buffer();
        let mut expected_live: BTreeSet<Key> = seed_keys.iter().copied().collect();
        for i in 0..2_500u64 {
            let k = 100_000 + i * 32;
            assert!(delta_buf.record_insert(k));
            expected_live.insert(k);
        }
        for i in 0..300u64 {
            let k = 100_000 + i * 32;
            assert!(delta_buf.record_delete(k));
            expected_live.remove(&k);
        }

        // Wait for worker completion off-lock, then run Phase 1 incremental pre-drain.
        while !coord.is_ready() {
            thread::yield_now();
        }
        let mut staged = coord.prepare_finish().unwrap();
        assert!(
            coord.is_in_progress(),
            "Draining slot must block overlapping rebuilds"
        );
        assert!(matches!(
            coord.start_rebuild(
                Arc::clone(&src_rm) as Arc<dyn ReverseMapStore>,
                Arc::clone(&dst_rm) as Arc<dyn ReverseMapStore>,
                Arc::clone(&fp_log),
                base_cfg.clone(),
                RebuildReason::ProactiveExpansion {
                    old_q: 10,
                    new_q: 11
                },
                IoThrottleConfig::default(),
            ),
            Err(AerisError::RebuildInProgress)
        ));

        let drained_off_lock = staged.pre_drain_until_small(256, 512, 32).unwrap();
        assert!(
            drained_off_lock >= 2_500,
            "expected off-lock pre-drain to absorb bulk delta backlog, got {drained_off_lock}"
        );
        assert!(staged.pre_drain_passes() >= 5);
        assert!(
            staged.pending_delta_len() <= 256,
            "residual tail {} exceeded threshold 256",
            staged.pending_delta_len()
        );
        staged.verify_pre_drained_keys().unwrap();

        // Simulate a small tail arriving right before locked swap.
        for tail_i in 0..16u64 {
            let k = 9_000_000 + tail_i * 64;
            assert!(delta_buf.record_insert(k));
            expected_live.insert(k);
        }

        // Phase 2: drain tiny residual tail and finalize.
        let completed = staged.finish_tail_with_live_keys().unwrap();
        assert!(!coord.is_in_progress());
        assert!(!delta_buf.is_active());
        assert_eq!(completed.report.delta_inserts_applied, 2_516);
        assert_eq!(completed.report.delta_deletes_applied, 300);

        for &k in &expected_live {
            assert!(
                completed.filter.query_point(k),
                "false negative for live key {k} after two-phase delta drain"
            );
        }
        let rm_keys: BTreeSet<Key> = dst_rm.scan_all_keys().unwrap().into_iter().collect();
        assert_eq!(rm_keys, expected_live);
    }

    #[test]
    fn staged_rebuild_drop_guard_recovers_coordinator_to_idle() {
        let dir = tempfile::TempDir::new().unwrap();
        let src_rm = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_src"), ReverseMapConfig::default())
                .unwrap(),
        );
        let dst_rm = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_dst"), ReverseMapConfig::default())
                .unwrap(),
        );
        let fp_log = Arc::new(
            FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
        );
        let cfg = FilterConfig::new(8, 15, 32, 20, 1.0 / 256.0).unwrap();

        let coord = AsyncRebuildCoordinator::new();
        coord
            .start_rebuild(
                Arc::clone(&src_rm) as Arc<dyn ReverseMapStore>,
                Arc::clone(&dst_rm) as Arc<dyn ReverseMapStore>,
                Arc::clone(&fp_log),
                cfg,
                RebuildReason::ProactiveExpansion { old_q: 8, new_q: 9 },
                IoThrottleConfig::default(),
            )
            .unwrap();

        while !coord.is_ready() {
            thread::yield_now();
        }
        let staged = coord.prepare_finish().unwrap();
        assert!(coord.is_in_progress());
        assert!(coord.delta_buffer().is_active());

        // Dropping StagedRebuild without calling finish_tail must release Draining -> Idle.
        drop(staged);
        assert!(!coord.is_in_progress());
        assert!(!coord.delta_buffer().is_active());
    }

    #[test]
    fn static_r_bypass_skips_estimator_prevents_dynamic_rebuilds_and_preserves_monotonic_fp_adapt()
    {
        let dir = tempfile::TempDir::new().unwrap();
        // Low eps (1/16 -> r_min = 7) with C = 15, m = 5 (R = 32) yields natural FPs quickly.
        let cfg = FilterConfig::new(9, 10, 32, 15, 1.0 / 16.0).unwrap();
        let default_engine_cfg = ControlledEngineConfig::new(cfg.clone());
        assert!(default_engine_cfg.dynamic_r_enabled);
        assert!(default_engine_cfg.estimator_config.enabled);

        let mut engine_cfg = ControlledEngineConfig::static_r_only(cfg);
        assert!(!engine_cfg.dynamic_r_enabled);
        assert!(!engine_cfg.estimator_config.enabled);
        engine_cfg.estimator_config.window_size = 128;
        engine_cfg.estimator_config.cooldown_queries = 32;
        engine_cfg.estimator_config.required_sustained_evals = 2;

        let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();
        assert!(!engine.is_dynamic_r_enabled());
        assert!(!engine.estimator().is_enabled());

        // Populate 350 keys in [0, 350_000) so load factor is ~0.68 and FPs arise on empty ranges.
        let mut live_keys = Vec::with_capacity(350);
        for i in 0..350u64 {
            let k = i * 1_000 + 13;
            engine.insert(k, &k.to_le_bytes()).unwrap();
            live_keys.push(k);
        }

        // Verify zero false negatives via both `query_range` and `query_range_filter_only`.
        for &k in &live_keys {
            let r = QueryRange::point(k);
            assert!(engine.query_range_filter_only(r));
            let outcome = engine.query_range(r).unwrap();
            assert!(outcome.non_empty);
            assert!(!outcome.was_false_positive);
        }
        assert_eq!(
            engine.estimator().total_queries(),
            0,
            "static-R bypass must skip estimator recording on true positives"
        );

        // Probe large shifted empty ranges (R = 512) until we observe & adapt genuine FPs.
        let mut adapted_fps = Vec::new();
        for probe in 0..40_000u64 {
            let start = 10_000_000 + probe * 1_024;
            let range = QueryRange::with_len(start, 512).unwrap();
            let filter_only_before = engine.query_range_filter_only(range);
            let outcome = engine.query_range(range).unwrap();
            assert!(!outcome.non_empty);
            assert_eq!(outcome.filter_returned_positive, filter_only_before);
            if outcome.was_false_positive {
                assert!(outcome.adapted);
                assert!(
                    !engine.query_range_filter_only(range),
                    "monotonically adapted range must immediately return false"
                );
                adapted_fps.push(range);
                if adapted_fps.len() >= 10 {
                    break;
                }
            }
        }
        assert!(
            adapted_fps.len() >= 5,
            "expected >= 5 genuine FPs to confirm adaptation remains active in static-R mode"
        );
        assert_eq!(engine.fp_log().len(), adapted_fps.len());
        assert_eq!(engine.total_false_positives(), adapted_fps.len() as u64);
        assert_eq!(
            engine.estimator().total_queries(),
            0,
            "static-R bypass must skip estimator recording even across negatives and FPs"
        );

        // Evaluate control plane: dynamic-R rebuild MUST stay false, even after huge ingestion
        // rate triggers ProactiveExpand!
        engine.estimator_mut().record_bulk_load_speed(1_000.0);
        engine.record_ingest_sample(10_000, 1.0);
        let decision = engine.evaluate_control_plane();
        assert!(
            !decision.optimization.should_rebuild,
            "static-R mode must never trigger dynamic-R rebuilds"
        );
        assert_eq!(
            decision.trigger.action,
            TriggerAction::ProactiveExpand,
            "proactive capacity trigger must remain active in static-R mode"
        );

        // Re-enabling dynamic-R dynamically synchronizes estimator and restores recording.
        engine.set_dynamic_r_enabled(true);
        assert!(engine.is_dynamic_r_enabled());
        assert!(engine.estimator().is_enabled());
        for i in 0..256u64 {
            let r = QueryRange::with_len(500_000_000 + i * 2_048, 512).unwrap();
            let _ = engine.query_range(r).unwrap();
        }
        assert_eq!(engine.estimator().total_queries(), 256);

        // Toggling dynamic-R off again suppresses `should_rebuild` even if estimator span shifted.
        engine.set_dynamic_r_enabled(false);
        assert!(!engine.is_dynamic_r_enabled());
        assert!(!engine.estimator().is_enabled());
        let post_toggle = engine.evaluate_control_plane();
        assert!(!post_toggle.optimization.should_rebuild);

        // All live keys still report 0 FN and all adapted FPs stay strictly false.
        for &k in &live_keys {
            assert!(engine.query_range_filter_only(QueryRange::point(k)));
        }
        for &r in &adapted_fps {
            assert!(!engine.query_range_filter_only(r));
        }
    }
}
