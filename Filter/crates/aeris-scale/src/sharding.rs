//! Range Sharding over independent `2^(q_i)` shards ([`ShardedAerisSystem`])
//! with wait-free RCU routing, dynamic data-driven median splits, adjacent shard
//! merges, and 100% monotonic false-positive elimination across topology changes.
//!
//! # Concurrency & Race-Free Split/Merge Protocol
//! - **Lock-Free Routing:** Active shards live inside an immutable
//!   [`RoutingSnapshot`] published through [`EpochCell`] with a flat contiguous
//!   `max_keys` array so `query_point` and `query_range` route in L1 cache with
//!   zero `Arc` pointer chases, zero heap allocations, and zero mutexes.
//! - **Zero Lost Writes During Live Splits/Merges:** Every [`ShardInstance`]
//!   owns a per-shard `write_coordinator` reader-writer lock and a
//!   `retired: AtomicBool` flag.
//!   - Writers (`insert`, `delete`, `adapt_range`) route via RCU, acquire ONLY
//!     a shared read lock on the target shard's `write_coordinator` (allowing
//!     concurrent writers on distinct chunks of the same shard), and check
//!     `shard.is_retired()`. If `true`, they drop the lock and retry against the
//!     updated [`RoutingSnapshot`].
//!   - `split_shard` acquires `topology_lock` -> exclusive `target.write_coordinator.write()`,
//!     materializes balanced children with fresh `RocksDbReverseMap` +
//!     `AerisFilter` + replayed `FalsePositiveLog`, publishes the new
//!     [`RoutingSnapshot`] via [`EpochCell::swap`], marks `target` retired, and
//!     releases `target.write_coordinator`.
//!   - `merge_adjacent_shards` acquires `topology_lock` -> exclusive
//!     `left.write_coordinator.write()` -> exclusive `right.write_coordinator.write()`
//!     in **strict ascending `ShardBounds::min_key` order**, builds the merged shard,
//!     publishes the new [`RoutingSnapshot`], marks both predecessors retired, and
//!     releases locks.
//!
//! # Lock Ordering
//! 1. `ShardedAerisSystem::topology_lock` (split/merge/rebalance only; NEVER on normal reads/writes)
//! 2. `ShardInstance::write_coordinator` (strict ascending `ShardBounds::min_key` order)
//! 3. `RcuAerisFilter::swap_lock`
//! 4. `ChunkLockTable` stripes (strict ascending stripe index order)
//! 5. `FilterGeneration::filter` (`parking_lot::RwLock<AerisFilter>`)
//! 6. `ReverseMapStore` / `FalsePositiveLog` / `DeltaBuffer` / `SlidingWindowEstimator`

#![forbid(unsafe_code)]

use crate::chunk_locks::DEFAULT_MAX_LOCK_STRIPES;
use crate::epoch_cell::{
    current_thread_stripe_slot, try_reclaim_retired_epochs, EpochCell, COUNTER_STRIPES,
};
use crate::rcu_filter::RcuAerisFilter;
use aeris_control::{EstimatorConfig, FalsePositiveLog, FpLogConfig, SlidingWindowEstimator};
use aeris_core::AerisFilter;
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, Epoch, FilterConfig, FpLogEntry, Key, QueryRange,
    RangeFilter, ReverseMapStore, ShardBounds, ShardId, ShardTelemetrySnapshot, WorkloadEstimator,
    MAX_LOAD_FACTOR, MAX_QUOTIENT_BITS,
};
use crossbeam_epoch::Guard;
use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, MutexGuard, RwLock};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Default initial number of contiguous range shards (`+12.5%` incremental RAM growth per shard).
pub const DEFAULT_INITIAL_SHARD_COUNT: usize = 8;

/// Default load factor ceiling at which a shard qualifies for dynamic median splitting.
pub const DEFAULT_SPLIT_LOAD_THRESHOLD: f64 = 0.80;

/// Default minimum live key count required before splitting a shard instead of expanding.
pub const DEFAULT_SPLIT_MIN_KEYS: usize = 128;

/// Default load factor below which two adjacent shards are eligible for merging.
pub const DEFAULT_MERGE_LOAD_THRESHOLD: f64 = 0.20;

/// Maximum permitted post-merge load ratio (`<= 0.50`) ensuring merged shards remain well below split threshold.
pub const MAX_POST_MERGE_LOAD_FACTOR: f64 = 0.50;

/// Default minimum number of active shards retained across merges.
pub const DEFAULT_MIN_SHARDS: usize = 1;

/// Default maximum number of active shards permitted across splits.
pub const DEFAULT_MAX_SHARDS: usize = 256;

/// Subdirectory name for each shard's persistent [`RocksDbReverseMap`].
const SHARD_REVERSE_MAP_DIR: &str = "reverse_map";

/// Filename for each shard's persistent [`FalsePositiveLog`].
const SHARD_FP_LOG_FILE: &str = "fp.log";

/// Configuration parameters governing [`ShardedAerisSystem`] topology, split/merge
/// hysteresis thresholds, and per-shard storage/control engines.
#[derive(Clone, Debug)]
pub struct ShardingConfig {
    /// Number of uniform contiguous shards initialized across `[0, u64::MAX]` (default `8`).
    pub initial_shard_count: usize,
    /// Baseline validated [`FilterConfig`] applied to newly created shards.
    pub base_filter_config: FilterConfig,
    /// Load factor ceiling above which a shard with `>= split_min_keys` triggers a split (`0.80`).
    pub split_load_threshold: f64,
    /// Minimum live keys required before splitting a shard (`128`).
    pub split_min_keys: usize,
    /// Upper cap on a single shard's quotient width `q`; shards exceeding `max_shard_q`
    /// are split rather than growing monolithically.
    pub max_shard_q: u8,
    /// Optional explicit key-count ceiling per shard triggering a split during rebalance.
    pub split_key_threshold: Option<usize>,
    /// Load factor below which two adjacent shards may merge if combined occupancy is `<= 0.50` (`0.20`).
    pub merge_load_threshold: f64,
    /// Lower floor on total active shard count (`1`).
    pub min_shards: usize,
    /// Upper ceiling on total active shard count (`256`).
    pub max_shards: usize,
    /// Maximum cache-line-padded chunk lock stripes allocated per shard generation.
    pub max_lock_stripes_per_shard: usize,
    /// RocksDB options for each shard's [`RocksDbReverseMap`].
    pub reverse_map_config: ReverseMapConfig,
    /// Persistence options for each shard's [`FalsePositiveLog`].
    pub fp_log_config: FpLogConfig,
    /// Sliding-window workload estimator options for each shard.
    pub estimator_config: EstimatorConfig,
    /// When `true` (default) and the active topology contains a single shard,
    /// `query_point` / `query_range` (and their `_direct` variants) bypass
    /// `route_key_index`, `max_keys` comparisons, and range clamping completely.
    pub single_shard_bypass: bool,
}

impl ShardingConfig {
    /// Creates a [`ShardingConfig`] with production defaults centered on `base_filter_config`.
    #[must_use]
    pub fn new(base_filter_config: FilterConfig) -> Self {
        let max_shard_q = base_filter_config
            .q
            .saturating_add(2)
            .min(MAX_QUOTIENT_BITS);
        Self {
            initial_shard_count: DEFAULT_INITIAL_SHARD_COUNT,
            base_filter_config,
            split_load_threshold: DEFAULT_SPLIT_LOAD_THRESHOLD,
            split_min_keys: DEFAULT_SPLIT_MIN_KEYS,
            max_shard_q,
            split_key_threshold: None,
            merge_load_threshold: DEFAULT_MERGE_LOAD_THRESHOLD,
            min_shards: DEFAULT_MIN_SHARDS,
            max_shards: DEFAULT_MAX_SHARDS,
            max_lock_stripes_per_shard: DEFAULT_MAX_LOCK_STRIPES,
            reverse_map_config: ReverseMapConfig::default(),
            fp_log_config: FpLogConfig::default(),
            estimator_config: EstimatorConfig::default(),
            single_shard_bypass: true,
        }
    }

    /// Overrides `initial_shard_count`.
    #[must_use]
    pub const fn with_initial_shard_count(mut self, count: usize) -> Self {
        self.initial_shard_count = count;
        self
    }

    /// Overrides `split_load_threshold`.
    #[must_use]
    pub const fn with_split_load_threshold(mut self, threshold: f64) -> Self {
        self.split_load_threshold = threshold;
        self
    }

    /// Overrides `split_min_keys`.
    #[must_use]
    pub const fn with_split_min_keys(mut self, min_keys: usize) -> Self {
        self.split_min_keys = min_keys;
        self
    }

    /// Overrides `max_shard_q`.
    #[must_use]
    pub const fn with_max_shard_q(mut self, max_q: u8) -> Self {
        self.max_shard_q = max_q;
        self
    }

    /// Overrides `split_key_threshold`.
    #[must_use]
    pub const fn with_split_key_threshold(mut self, threshold: Option<usize>) -> Self {
        self.split_key_threshold = threshold;
        self
    }

    /// Overrides `merge_load_threshold`.
    #[must_use]
    pub const fn with_merge_load_threshold(mut self, threshold: f64) -> Self {
        self.merge_load_threshold = threshold;
        self
    }

    /// Overrides `min_shards`.
    #[must_use]
    pub const fn with_min_shards(mut self, min_shards: usize) -> Self {
        self.min_shards = min_shards;
        self
    }

    /// Overrides `max_shards`.
    #[must_use]
    pub const fn with_max_shards(mut self, max_shards: usize) -> Self {
        self.max_shards = max_shards;
        self
    }

    /// Overrides `reverse_map_config`.
    #[must_use]
    pub fn with_reverse_map_config(mut self, cfg: ReverseMapConfig) -> Self {
        self.reverse_map_config = cfg;
        self
    }

    /// Overrides `fp_log_config`.
    #[must_use]
    pub const fn with_fp_log_config(mut self, cfg: FpLogConfig) -> Self {
        self.fp_log_config = cfg;
        self
    }

    /// Overrides `estimator_config`.
    #[must_use]
    pub fn with_estimator_config(mut self, cfg: EstimatorConfig) -> Self {
        self.estimator_config = cfg;
        self
    }

    /// Overrides `single_shard_bypass` (default `true`).
    #[must_use]
    pub const fn with_single_shard_bypass(mut self, enabled: bool) -> Self {
        self.single_shard_bypass = enabled;
        self
    }

    /// Validates all structural, numerical, and sub-component invariants.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on the first violated invariant.
    pub fn validate(&self) -> AerisResult<()> {
        self.base_filter_config.validate()?;
        self.fp_log_config.validate()?;
        self.estimator_config.validate()?;

        if self.min_shards == 0 {
            return Err(AerisError::InvalidConfig(
                "min_shards must be at least 1".to_string(),
            ));
        }
        if self.max_shards < self.min_shards {
            return Err(AerisError::InvalidConfig(format!(
                "max_shards ({}) must be >= min_shards ({})",
                self.max_shards, self.min_shards
            )));
        }
        if self.initial_shard_count < self.min_shards || self.initial_shard_count > self.max_shards
        {
            return Err(AerisError::InvalidConfig(format!(
                "initial_shard_count ({}) must be in [min_shards={}, max_shards={}]",
                self.initial_shard_count, self.min_shards, self.max_shards
            )));
        }
        if self.max_shard_q < self.base_filter_config.q || self.max_shard_q > MAX_QUOTIENT_BITS {
            return Err(AerisError::InvalidConfig(format!(
                "max_shard_q ({}) must be in [base_q={}, {}]",
                self.max_shard_q, self.base_filter_config.q, MAX_QUOTIENT_BITS
            )));
        }
        if !self.split_load_threshold.is_finite()
            || self.split_load_threshold <= 0.0
            || self.split_load_threshold > MAX_LOAD_FACTOR
        {
            return Err(AerisError::InvalidConfig(format!(
                "split_load_threshold ({}) must be finite and in (0.0, {MAX_LOAD_FACTOR}]",
                self.split_load_threshold
            )));
        }
        if !self.merge_load_threshold.is_finite()
            || self.merge_load_threshold < 0.0
            || self.merge_load_threshold >= self.split_load_threshold
        {
            return Err(AerisError::InvalidConfig(format!(
                "merge_load_threshold ({}) must be in [0.0, split_load_threshold={})",
                self.merge_load_threshold, self.split_load_threshold
            )));
        }
        if let Some(key_cap) = self.split_key_threshold {
            if key_cap < 2 {
                return Err(AerisError::InvalidConfig(format!(
                    "split_key_threshold ({key_cap}) must be at least 2 when specified"
                )));
            }
        }
        if self.max_lock_stripes_per_shard == 0 {
            return Err(AerisError::InvalidConfig(
                "max_lock_stripes_per_shard must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Independent range shard owning a contiguous key interval `[min_key, max_key]`,
/// an epoch-RCU [`RcuAerisFilter`], a persistent [`RocksDbReverseMap`], a
/// persistent [`FalsePositiveLog`], and a local [`SlidingWindowEstimator`].
///
/// Cache-line layout isolates read-only routing/filter handles, striped per-thread
/// query counters, and write/topology coordination metadata onto distinct cache
/// lines so multi-core reads never suffer false sharing or atomic ping-pong.
pub struct ShardInstance {
    shard_id: ShardId,
    bounds: ShardBounds,
    rcu_filter: RcuAerisFilter,
    reverse_map: Arc<RocksDbReverseMap>,
    fp_log: Arc<FalsePositiveLog>,
    estimator: Mutex<SlidingWindowEstimator>,
    /// Lock Level 2: coordinates concurrent shard mutations (`read()`) with
    /// exclusive split/merge quiescence (`write()`, strict ascending `min_key`
    /// order). NEVER held by point or range query readers!
    write_coordinator: CachePadded<RwLock<()>>,
    /// Set to `true` before releasing `write_coordinator` after split/merge publishes
    /// the replacement [`RoutingSnapshot`], forcing waiting writers to re-route cleanly.
    retired: CachePadded<AtomicBool>,
    /// Cache-line-padded striped exact query counters indexed by thread-local
    /// stripe slot, eliminating cross-core `LOCK XADD` cache-line bouncing on reads.
    query_counts: [CachePadded<AtomicU64>; COUNTER_STRIPES],
    fp_count: CachePadded<AtomicU64>,
    write_count: CachePadded<AtomicU64>,
}

impl ShardInstance {
    /// Returns this shard's globally unique [`ShardId`].
    #[inline]
    #[must_use]
    pub const fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    /// Returns this shard's inclusive key-space boundaries `[min_key, max_key]`.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> ShardBounds {
        self.bounds
    }

    /// Returns a reference to this shard's [`RcuAerisFilter`].
    #[inline]
    #[must_use]
    pub const fn rcu_filter(&self) -> &RcuAerisFilter {
        &self.rcu_filter
    }

    /// Returns the initial persistent [`RocksDbReverseMap`] created for this shard.
    #[inline]
    #[must_use]
    pub fn reverse_map(&self) -> Arc<RocksDbReverseMap> {
        Arc::clone(&self.reverse_map)
    }

    /// Returns the currently active [`ReverseMapStore`] bound to the shard's latest
    /// [`RcuAerisFilter`] generation (identical to [`Self::reverse_map`] unless a
    /// dynamic-`R` rebuild swapped in a fresh Reverse Map).
    #[inline]
    #[must_use]
    pub fn active_reverse_map(&self) -> Arc<dyn ReverseMapStore> {
        self.rcu_filter.active_reverse_map()
    }

    /// Returns a shared handle to this shard's persistent [`FalsePositiveLog`].
    #[inline]
    #[must_use]
    pub fn fp_log(&self) -> Arc<FalsePositiveLog> {
        Arc::clone(&self.fp_log)
    }

    /// Returns `true` if this shard has been retired by a completed split or merge.
    #[inline]
    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// Marks this shard as retired after publishing its replacement topology.
    #[inline]
    fn mark_retired(&self) {
        self.retired.store(true, Ordering::Release);
    }

    /// Increments the thread-local striped query counter for this shard with zero
    /// cross-core cache-line contention.
    #[inline]
    fn record_query_hit_with_slot(&self, slot: usize) {
        self.query_counts[slot & (COUNTER_STRIPES - 1)].fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the current tracked live key count for this shard.
    #[inline]
    #[must_use]
    pub fn key_count(&self) -> u64 {
        self.rcu_filter.len() as u64
    }

    /// Returns the exact cumulative count of point and range queries routed to
    /// this shard by summing all cache-line-padded counter stripes.
    #[inline]
    #[must_use]
    pub fn query_count(&self) -> u64 {
        self.query_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }

    /// Returns the cumulative count of confirmed false positives adapted on this shard.
    #[inline]
    #[must_use]
    pub fn fp_count(&self) -> u64 {
        self.fp_count.load(Ordering::Relaxed)
    }

    /// Returns the cumulative count of write mutations (inserts + deletes) applied to this shard.
    #[inline]
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.write_count.load(Ordering::Relaxed)
    }

    /// Captures a [`ShardTelemetrySnapshot`] suitable for multi-tenant
    /// memory allocation and observability dashboards.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> ShardTelemetrySnapshot {
        let cfg = self.rcu_filter.config();
        let est = self.estimator.lock().snapshot();
        let observed_fpr = if est.window_query_count > 0 {
            est.observed_fpr
        } else {
            let queries = self.query_count();
            if queries == 0 {
                0.0
            } else {
                (self.fp_count.load(Ordering::Relaxed) as f64 / queries as f64).min(1.0)
            }
        };

        ShardTelemetrySnapshot {
            shard_id: self.shard_id,
            key_count: self.rcu_filter.len() as u64,
            current_memory_bytes: self.rcu_filter.memory_usage_bytes() as u64,
            query_rate: est.query_rate,
            observed_fpr,
            p95_query_len: est.p95_query_len as f64,
            load_factor: self.rcu_filter.load_factor(),
            current_q: cfg.q,
            current_r: cfg.r,
            current_m: cfg.m,
        }
    }
}

/// Threshold below which [`RoutingSnapshot::route_key_index`] executes a
/// branchless flat scan over 1–2 L1 cache lines (`<= 16` `u64` upper bounds),
/// avoiding branch mispredictions and `Arc` pointer chasing entirely.
const FLAT_LINEAR_SCAN_MAX_SHARDS: usize = 16;

/// Immutable, contiguous routing table snapshot published through [`EpochCell`].
///
/// Stores a flat `max_keys: Box<[Key]>` array alongside `shards: Box<[Arc<ShardInstance>]>`
/// so point and range routing inspect 1–4 contiguous L1 cache lines with ZERO
/// `Arc` dereferences until the winning shard is selected.
///
/// # Invariants
/// 1. `shards` is non-empty and sorted strictly ascending by `bounds.min_key`.
/// 2. `max_keys.len() == shards.len()` and `max_keys[i] == shards[i].bounds.max_key`.
/// 3. `shards[0].bounds.min_key == 0` and `shards.last().unwrap().bounds.max_key == u64::MAX`.
/// 4. For every adjacent pair `(prev, next)`, `prev.bounds.max_key + 1 == next.bounds.min_key`
///    (zero gaps, zero overlaps).
/// 5. All `shard_id`s in `shards` are unique.
pub struct RoutingSnapshot {
    max_keys: Box<[Key]>,
    shards: Box<[Arc<ShardInstance>]>,
}

impl RoutingSnapshot {
    /// Constructs and validates a contiguous [`RoutingSnapshot`] covering `[0, u64::MAX]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if any contiguity, ordering, or
    /// uniqueness invariant is violated.
    pub fn new(shards: Vec<Arc<ShardInstance>>) -> AerisResult<Self> {
        let max_keys: Box<[Key]> = shards.iter().map(|s| s.bounds.max_key).collect();
        let snap = Self {
            max_keys,
            shards: shards.into_boxed_slice(),
        };
        snap.verify_invariants()?;
        Ok(snap)
    }

    /// Returns all active shards in strict ascending `bounds.min_key` order.
    #[inline]
    #[must_use]
    pub fn shards(&self) -> &[Arc<ShardInstance>] {
        &self.shards
    }

    /// Wait-free index lookup on flat `max_keys` returning the unique shard index
    /// owning `key` with zero `Arc` pointer chasing.
    ///
    /// Because `max_keys` is strictly ascending and terminates with `u64::MAX`,
    /// counting entries `< key` is mathematically identical to
    /// `partition_point(|&mk| mk < key)` and executes branchlessly over 1–2 L1
    /// cache lines for `<= 16` shards.
    #[inline]
    #[must_use]
    fn route_key_index(&self, key: Key) -> usize {
        let max_keys = &*self.max_keys;
        if max_keys.len() <= FLAT_LINEAR_SCAN_MAX_SHARDS {
            let mut idx = 0usize;
            for &mk in max_keys {
                idx += usize::from(mk < key);
            }
            idx
        } else {
            max_keys.partition_point(|&mk| mk < key)
        }
    }

    /// Wait-free `O(log S)` lookup returning the unique shard owning `key`.
    ///
    /// # Complexity
    /// `O(1)` L1 cache-line scan (`<= 16` shards) or `O(log S)` flat binary search,
    /// zero heap allocations, zero intermediate `Arc` dereferences.
    #[inline]
    #[must_use]
    pub fn route_key(&self, key: Key) -> &Arc<ShardInstance> {
        let idx = self.route_key_index(key);
        debug_assert!(
            idx < self.shards.len() && self.shards[idx].bounds.contains(key),
            "routing invariant violated: key {key} failed to resolve to a valid shard"
        );
        &self.shards[idx]
    }

    /// Wait-free `O(log S)` lookup returning the exact contiguous slice of
    /// shards intersecting `range`.
    ///
    /// Fast-paths the common single-shard case (`range.end() <= max_keys[first]`)
    /// with a single comparison, skipping the second search entirely.
    ///
    /// # Complexity
    /// `O(log S)` time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn route_range(&self, range: QueryRange) -> &[Arc<ShardInstance>] {
        let first = self.route_key_index(range.start());
        if range.end() <= self.max_keys[first] {
            return &self.shards[first..=first];
        }
        let end_excl = self.route_range_end_excl(first, range.end());
        debug_assert!(
            first < end_excl && end_excl <= self.shards.len(),
            "routing invariant violated: range {range:?} yielded empty slice [{first}..{end_excl}]"
        );
        &self.shards[first..end_excl]
    }

    /// Computes the exclusive upper shard index for a multi-shard range whose
    /// `range_end > self.max_keys[first]`.
    #[inline]
    fn route_range_end_excl(&self, first: usize, range_end: Key) -> usize {
        let tail = &self.max_keys[(first + 1)..];
        let offset = if tail.len() <= FLAT_LINEAR_SCAN_MAX_SHARDS {
            let mut idx = 0usize;
            for &mk in tail {
                idx += usize::from(mk < range_end);
            }
            idx
        } else {
            tail.partition_point(|&mk| mk < range_end)
        };
        // `last = first + 1 + offset` is the smallest shard with `max_key >= range_end`.
        first + offset + 2
    }

    /// Finds the positional index of `shard_id` in this routing snapshot.
    #[must_use]
    pub fn find_shard_index(&self, shard_id: ShardId) -> Option<usize> {
        self.shards.iter().position(|s| s.shard_id == shard_id)
    }

    /// Verifies complete contiguous `[0, u64::MAX]` coverage, zero gaps, zero
    /// overlaps, active (non-retired) status, and unique [`ShardId`]s.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on the first violation detected.
    pub fn verify_invariants(&self) -> AerisResult<()> {
        if self.shards.is_empty() {
            return Err(AerisError::InternalInvariant(
                "RoutingSnapshot contains zero shards".to_string(),
            ));
        }

        let first = &self.shards[0];
        if first.bounds.min_key != 0 {
            return Err(AerisError::InternalInvariant(format!(
                "first shard {} starts at {} instead of 0",
                first.shard_id, first.bounds.min_key
            )));
        }

        let last = self.shards.last().expect("verified non-empty shards above");
        if last.bounds.max_key != u64::MAX {
            return Err(AerisError::InternalInvariant(format!(
                "last shard {} ends at {} instead of u64::MAX",
                last.shard_id, last.bounds.max_key
            )));
        }

        let mut seen_ids = HashSet::with_capacity(self.shards.len());
        for (idx, shard) in self.shards.iter().enumerate() {
            if !seen_ids.insert(shard.shard_id) {
                return Err(AerisError::InternalInvariant(format!(
                    "duplicate shard_id {} in RoutingSnapshot",
                    shard.shard_id
                )));
            }
            if shard.bounds.min_key > shard.bounds.max_key {
                return Err(AerisError::InternalInvariant(format!(
                    "shard {} has inverted bounds [{}, {}]",
                    shard.shard_id, shard.bounds.min_key, shard.bounds.max_key
                )));
            }
            if shard.is_retired() {
                return Err(AerisError::InternalInvariant(format!(
                    "active RoutingSnapshot references retired shard {}",
                    shard.shard_id
                )));
            }
            if idx > 0 {
                let prev = &self.shards[idx - 1];
                let expected_min = prev.bounds.max_key.checked_add(1).ok_or_else(|| {
                    AerisError::InternalInvariant(format!(
                        "shard {} ends at u64::MAX before final shard",
                        prev.shard_id
                    ))
                })?;
                if shard.bounds.min_key != expected_min {
                    return Err(AerisError::InternalInvariant(format!(
                        "routing discontinuity between shard {} [.., {}] and shard {} [{}, ..]",
                        prev.shard_id, prev.bounds.max_key, shard.shard_id, shard.bounds.min_key
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Diagnostic summary returned by [`ShardedAerisSystem::split_shard`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitReport {
    /// Identifier of the retired parent shard.
    pub old_shard_id: ShardId,
    /// Identifier of the newly created lower-half child shard `[min, split_key]`.
    pub left_shard_id: ShardId,
    /// Identifier of the newly created upper-half child shard `[split_key + 1, max]`.
    pub right_shard_id: ShardId,
    /// Data-driven median split boundary `split_key` (inclusive upper bound of left child).
    pub split_key: Key,
    /// Key bounds owned by `left_shard_id`.
    pub left_bounds: ShardBounds,
    /// Key bounds owned by `right_shard_id`.
    pub right_bounds: ShardBounds,
    /// Number of live keys migrated to `left_shard_id`.
    pub left_keys: usize,
    /// Number of live keys migrated to `right_shard_id`.
    pub right_keys: usize,
    /// Number of False-Positive Log entries replayed onto `left_shard_id`.
    pub fp_entries_replayed_left: usize,
    /// Number of False-Positive Log entries replayed onto `right_shard_id`.
    pub fp_entries_replayed_right: usize,
    /// Newly published routing topology [`Epoch`].
    pub new_routing_epoch: Epoch,
    /// Wall-clock duration of the split operation in nanoseconds.
    pub latency_ns: u64,
}

/// Diagnostic summary returned by [`ShardedAerisSystem::merge_adjacent_shards`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeReport {
    /// Identifier of the retired lower adjacent shard.
    pub left_shard_id: ShardId,
    /// Identifier of the retired upper adjacent shard.
    pub right_shard_id: ShardId,
    /// Identifier of the newly created merged shard `[left.min, right.max]`.
    pub merged_shard_id: ShardId,
    /// Key bounds owned by `merged_shard_id`.
    pub merged_bounds: ShardBounds,
    /// Total live keys consolidated into `merged_shard_id`.
    pub merged_keys: usize,
    /// Total combined False-Positive Log entries replayed onto `merged_shard_id`.
    pub fp_entries_replayed: usize,
    /// Newly published routing topology [`Epoch`].
    pub new_routing_epoch: Epoch,
    /// Wall-clock duration of the merge operation in nanoseconds.
    pub latency_ns: u64,
}

/// Diagnostic summary returned by [`ShardedAerisSystem::rebalance_if_needed`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RebalanceReport {
    /// Splits performed to relieve overloaded or hot-spot shards.
    pub splits: Vec<SplitReport>,
    /// Adjacent merges performed to reclaim RAM from underloaded shards.
    pub merges: Vec<MergeReport>,
    /// Active shard count after rebalancing completed.
    pub final_shard_count: usize,
    /// Active routing topology [`Epoch`] after rebalancing completed.
    pub final_routing_epoch: Epoch,
}

/// Production Scale Plane range-sharded Aeris Range Filtering System covering
/// the full 64-bit unsigned key space `[0, u64::MAX]` with independent `2^(q_i)`
/// shards, lock-free epoch-RCU routing, incremental RAM growth, and race-free
/// dynamic splits and merges.
pub struct ShardedAerisSystem {
    root_dir: PathBuf,
    config: ShardingConfig,
    routing: EpochCell<RoutingSnapshot>,
    /// Lock Level 1: serializes split/merge/rebalance topology transitions.
    /// NEVER acquired by `insert`, `delete`, `query_point`, `query_range`, or `adapt_range`.
    topology_lock: Mutex<()>,
    next_shard_id: AtomicU64,
}

impl ShardedAerisSystem {
    /// Initializes a [`ShardedAerisSystem`] rooted at `root_dir` with
    /// `config.initial_shard_count` uniform contiguous shards over `[0, u64::MAX]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`], [`AerisError::StorageError`],
    /// [`AerisError::WalIoError`], or [`AerisError::InternalInvariant`] on failure.
    pub fn open(root_dir: impl AsRef<Path>, config: ShardingConfig) -> AerisResult<Self> {
        config.validate()?;
        let bounds = compute_uniform_bounds(config.initial_shard_count)?;
        Self::open_with_exact_bounds(root_dir, config, &bounds)
    }

    /// Initializes a [`ShardedAerisSystem`] rooted at `root_dir` using explicit
    /// strictly ascending inclusive split boundaries `split_keys`.
    ///
    /// For `split_keys = [b_0, b_1]`, creates shards:
    /// `[0, b_0]`, `[b_0 + 1, b_1]`, `[b_1 + 1, u64::MAX]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `split_keys` is unsorted, has
    /// duplicates, contains `u64::MAX`, or yields a shard count outside
    /// `[min_shards, max_shards]`.
    pub fn open_with_boundaries(
        root_dir: impl AsRef<Path>,
        mut config: ShardingConfig,
        split_keys: &[Key],
    ) -> AerisResult<Self> {
        let bounds = compute_bounds_from_split_keys(split_keys)?;
        config.initial_shard_count = bounds.len();
        config.validate()?;
        Self::open_with_exact_bounds(root_dir, config, &bounds)
    }

    /// Internal constructor initializing one fresh [`ShardInstance`] per [`ShardBounds`].
    fn open_with_exact_bounds(
        root_dir: impl AsRef<Path>,
        config: ShardingConfig,
        bounds_list: &[ShardBounds],
    ) -> AerisResult<Self> {
        let root_dir = root_dir.as_ref().to_path_buf();
        fs::create_dir_all(&root_dir).map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to create ShardedAerisSystem root directory '{}': {err}",
                root_dir.display()
            ))
        })?;

        let mut shards = Vec::with_capacity(bounds_list.len());
        let mut next_id = 0u64;
        for &bounds in bounds_list {
            let shard_id = next_id;
            next_id += 1;
            let (shard, _) = build_shard_from_state(
                &root_dir,
                &config,
                shard_id,
                bounds,
                &config.base_filter_config,
                &[],
                &[],
            )?;
            shards.push(Arc::new(shard));
        }

        let snapshot = RoutingSnapshot::new(shards)?;
        info!(
            root_dir = %root_dir.display(),
            shard_count = bounds_list.len(),
            base_q = config.base_filter_config.q,
            base_r = config.base_filter_config.r,
            base_m = config.base_filter_config.m,
            "opened ShardedAerisSystem"
        );

        Ok(Self {
            root_dir,
            config,
            routing: EpochCell::new(snapshot),
            topology_lock: Mutex::new(()),
            next_shard_id: AtomicU64::new(next_id),
        })
    }

    /// Returns the active [`ShardingConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &ShardingConfig {
        &self.config
    }

    /// Returns the filesystem root directory hosting shard Reverse Maps and FP Logs.
    #[inline]
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Returns the current monotonic routing topology [`Epoch`].
    #[inline]
    #[must_use]
    pub fn routing_epoch(&self) -> Epoch {
        self.routing.epoch()
    }

    /// Returns the number of active contiguous range shards.
    #[inline]
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.routing.with_ref(|snap| snap.shards.len())
    }

    /// Returns `(ShardId, ShardBounds)` for all active shards in strict ascending
    /// `bounds.min_key` order.
    #[must_use]
    pub fn shard_bounds(&self) -> Vec<(ShardId, ShardBounds)> {
        self.routing
            .with_ref(|snap| snap.shards.iter().map(|s| (s.shard_id, s.bounds)).collect())
    }

    /// Returns the total RAM footprint (bytes) summed across all active shard filters.
    #[must_use]
    pub fn total_memory_bytes(&self) -> usize {
        self.routing.with_ref(|snap| {
            snap.shards
                .iter()
                .map(|s| s.rcu_filter.memory_usage_bytes())
                .sum()
        })
    }

    /// Returns the total live key count summed across all active shard filters.
    #[must_use]
    pub fn total_keys(&self) -> usize {
        self.routing
            .with_ref(|snap| snap.shards.iter().map(|s| s.rcu_filter.len()).sum())
    }

    /// Returns per-shard telemetry snapshots for all active shards in ascending key order.
    #[must_use]
    pub fn shard_telemetry_snapshots(&self) -> Vec<ShardTelemetrySnapshot> {
        self.routing
            .with_ref(|snap| snap.shards.iter().map(|s| s.telemetry_snapshot()).collect())
    }

    /// Returns the [`ShardId`] currently owning `key` via wait-free RCU routing.
    #[inline]
    #[must_use]
    pub fn route_shard_id(&self, key: Key) -> ShardId {
        self.routing.with_ref(|snap| snap.route_key(key).shard_id)
    }

    /// Returns a cloned [`Arc<ShardInstance>`] for `shard_id` if currently active.
    ///
    /// # Errors
    /// Returns [`AerisError::ShardNotFound`] if `shard_id` is not in the active routing table.
    pub fn shard_handle(&self, shard_id: ShardId) -> AerisResult<Arc<ShardInstance>> {
        self.routing.with_ref(|snap| {
            snap.find_shard_index(shard_id)
                .map(|idx| Arc::clone(&snap.shards[idx]))
                .ok_or(AerisError::ShardNotFound(shard_id))
        })
    }

    /// Verifies that the active [`RoutingSnapshot`] covers `[0, u64::MAX]`
    /// contiguously with zero gaps, zero overlaps, and no retired shards.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if any routing invariant is violated.
    pub fn verify_routing_invariants(&self) -> AerisResult<()> {
        self.routing.with_ref(RoutingSnapshot::verify_invariants)
    }

    /// Verifies routing invariants AND every active shard's Quotient Filter
    /// invariants AND that every persisted Reverse Map key lies inside its owning
    /// shard's [`ShardBounds`].
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] or [`AerisError::StorageError`] on failure.
    pub fn verify_all_invariants(&self) -> AerisResult<()> {
        let shards: Vec<Arc<ShardInstance>> = self.routing.with_ref(|snap| -> AerisResult<_> {
            snap.verify_invariants()?;
            Ok(snap.shards.to_vec())
        })?;

        for shard in shards {
            shard.rcu_filter.verify_invariants()?;
            let keys = shard.active_reverse_map().scan_all_keys()?;
            for k in keys {
                if !shard.bounds.contains(k) {
                    return Err(AerisError::InternalInvariant(format!(
                        "shard {} Reverse Map contains out-of-bounds key {k} outside [{}, {}]",
                        shard.shard_id, shard.bounds.min_key, shard.bounds.max_key
                    )));
                }
            }
        }
        Ok(())
    }

    /// Inserts `key` into its owning shard via lock-free RCU routing and per-shard
    /// `write_coordinator` shared locking, automatically retrying if the shard
    /// retires due to a concurrent split or merge.
    ///
    /// Never acquires `topology_lock`. Uses a single Crossbeam epoch pin with
    /// zero `Arc::clone` / `Arc::drop` traffic and zero redundant `len()` locks
    /// on the hot write path. If the owning shard reaches `max_load_factor`,
    /// only that single shard expands via RCU swap (`+1/S` incremental RAM growth).
    ///
    /// Returns the [`ShardId`] that durably accepted the insert.
    ///
    /// # Errors
    /// Propagates filter expansion or RocksDB storage errors.
    pub fn insert(&self, key: Key) -> AerisResult<ShardId> {
        loop {
            let guard = crossbeam_epoch::pin();
            let snap = self.routing.load(&guard);
            let shard = snap.route_key(key);
            let _write_guard = shard.write_coordinator.read();
            if shard.is_retired() {
                continue;
            }

            let _fp = shard.rcu_filter.insert_pinned(key, &guard)?;
            shard.write_count.fetch_add(1, Ordering::Relaxed);
            return Ok(shard.shard_id);
        }
    }

    /// Deletes `key` from its owning shard via lock-free RCU routing and per-shard
    /// `write_coordinator` shared locking, automatically retrying if the shard
    /// retires during a concurrent split or merge.
    ///
    /// Returns `Ok(true)` if a matching entry was deleted, or `Ok(false)` if absent.
    ///
    /// # Errors
    /// Propagates RocksDB storage errors.
    pub fn delete(&self, key: Key) -> AerisResult<bool> {
        loop {
            let guard = crossbeam_epoch::pin();
            let snap = self.routing.load(&guard);
            let shard = snap.route_key(key);
            let _write_guard = shard.write_coordinator.read();
            if shard.is_retired() {
                continue;
            }

            let removed = shard.rcu_filter.delete_pinned(key, &guard)?.is_some();
            if removed {
                shard.write_count.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(removed);
        }
    }

    /// Answers a point emptiness query for `key` on the lock-free RCU hot path
    /// (`true` = possibly present, `false` = definitely absent).
    ///
    /// Pins a Crossbeam epoch guard and reads the calling thread's stripe slot
    /// once across routing, striped exact query accounting, and the target
    /// shard's [`RcuAerisFilter`]. When `config.single_shard_bypass` is enabled
    /// and the active topology has a single shard, bypasses routing completely.
    ///
    /// # Complexity
    /// `O(1)` L1 cache-line scan (`<= 16` shards) or `O(log S)` flat search +
    /// `O(1)` Quotient Filter probe, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_point(&self, key: Key) -> bool {
        let guard = crossbeam_epoch::pin();
        let snap = self.routing.load(&guard);
        let shard = if self.config.single_shard_bypass && snap.shards.len() == 1 {
            &snap.shards[0]
        } else {
            snap.route_key(key)
        };
        let slot = current_thread_stripe_slot();
        shard.record_query_hit_with_slot(slot);
        shard
            .rcu_filter
            .query_point_pinned_with_slot(key, &guard, slot)
    }

    /// Answers a closed range emptiness query `[start, end]` across all
    /// intersecting shards on the lock-free RCU hot path, clamping each sub-probe
    /// strictly to the owning shard's [`ShardBounds`].
    ///
    /// When `config.single_shard_bypass` is enabled and the active topology has a
    /// single shard, dispatches directly to shard 0 with zero routing comparisons
    /// and zero clamping. Otherwise fast-paths single-shard ranges
    /// (`range.end() <= max_keys[first]`) with a single epoch pin and zero clamping.
    ///
    /// # Complexity
    /// `O(log S + intersecting_shards * probes_per_shard)`, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_range(&self, range: QueryRange) -> bool {
        let guard = crossbeam_epoch::pin();
        let snap = self.routing.load(&guard);
        let slot = current_thread_stripe_slot();
        if self.config.single_shard_bypass && snap.shards.len() == 1 {
            let shard = &snap.shards[0];
            shard.record_query_hit_with_slot(slot);
            return shard
                .rcu_filter
                .query_range_pinned_with_slot(range, &guard, slot);
        }
        let first = snap.route_key_index(range.start());
        if range.end() <= snap.max_keys[first] {
            let shard = &snap.shards[first];
            shard.record_query_hit_with_slot(slot);
            return shard
                .rcu_filter
                .query_range_pinned_with_slot(range, &guard, slot);
        }
        Self::query_range_multi_shard(snap, first, range, &guard, slot)
    }

    /// Zero-overhead exclusive single-thread point emptiness query (`&mut self`).
    ///
    /// Uses compile-time `&mut self` exclusivity to bypass Crossbeam epoch pinning,
    /// `StripedRwLock::read()`, and atomic query counter increments completely.
    /// When `config.single_shard_bypass` is active on a 1-shard topology, also
    /// bypasses routing and `max_keys` comparisons. Safely falls back to the
    /// pinned reader path only if an external [`Arc<ShardInstance>`] clone exists.
    ///
    /// # Complexity
    /// `O(1)` (`<= 16` shards) or `O(log S)` + `O(1)` filter probe, zero locks,
    /// zero epoch pinning, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_point_direct(&mut self, key: Key) -> bool {
        let snap = self.routing.get_mut();
        let idx = if self.config.single_shard_bypass && snap.shards.len() == 1 {
            0
        } else {
            snap.route_key_index(key)
        };
        if let Some(shard) = Arc::get_mut(&mut snap.shards[idx]) {
            shard.rcu_filter.query_point_direct(key)
        } else {
            let guard = crossbeam_epoch::pin();
            let slot = current_thread_stripe_slot();
            snap.shards[idx]
                .rcu_filter
                .query_point_pinned_with_slot(key, &guard, slot)
        }
    }

    /// Zero-overhead exclusive single-thread closed range emptiness query (`&mut self`).
    ///
    /// Uses compile-time `&mut self` exclusivity to bypass Crossbeam epoch pinning,
    /// `StripedRwLock::read()`, and atomic query counter increments completely.
    /// When `config.single_shard_bypass` is active on a 1-shard topology, also
    /// bypasses routing, `max_keys` comparisons, and range clamping. Safely falls
    /// back to the pinned reader path only if an external [`Arc<ShardInstance>`]
    /// clone exists.
    ///
    /// # Complexity
    /// `O(log S + intersecting_shards * probes_per_shard)`, zero locks, zero epoch
    /// pinning, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_range_direct(&mut self, range: QueryRange) -> bool {
        let snap = self.routing.get_mut();
        if self.config.single_shard_bypass && snap.shards.len() == 1 {
            return if let Some(shard) = Arc::get_mut(&mut snap.shards[0]) {
                shard.rcu_filter.query_range_direct(range)
            } else {
                let guard = crossbeam_epoch::pin();
                let slot = current_thread_stripe_slot();
                snap.shards[0]
                    .rcu_filter
                    .query_range_pinned_with_slot(range, &guard, slot)
            };
        }
        let first = snap.route_key_index(range.start());
        if range.end() <= snap.max_keys[first] {
            return if let Some(shard) = Arc::get_mut(&mut snap.shards[first]) {
                shard.rcu_filter.query_range_direct(range)
            } else {
                let guard = crossbeam_epoch::pin();
                let slot = current_thread_stripe_slot();
                snap.shards[first]
                    .rcu_filter
                    .query_range_pinned_with_slot(range, &guard, slot)
            };
        }
        Self::query_range_direct_multi_shard(snap, first, range)
    }

    /// Exclusive single-thread insert (`&mut self`), bypassing Crossbeam epoch
    /// pinning, `write_coordinator` locking, chunk-stripe locking, and
    /// `StripedRwLock` acquisition whenever no external [`Arc<ShardInstance>`]
    /// clone exists.
    ///
    /// Falls back transparently to [`Self::insert`] if an external shard handle
    /// clone is alive.
    ///
    /// # Errors
    /// Propagates filter expansion or RocksDB storage errors.
    pub fn insert_direct(&mut self, key: Key) -> AerisResult<ShardId> {
        let snap = self.routing.get_mut();
        let idx = if self.config.single_shard_bypass && snap.shards.len() == 1 {
            0
        } else {
            snap.route_key_index(key)
        };
        if let Some(shard) = Arc::get_mut(&mut snap.shards[idx]) {
            let _fp = shard.rcu_filter.insert_direct(key)?;
            *shard.write_count.get_mut() += 1;
            Ok(shard.shard_id)
        } else {
            self.insert(key)
        }
    }

    /// Cold multi-shard range probe fallback for [`Self::query_range_direct`] when
    /// `range` straddles at least one shard boundary.
    #[cold]
    #[inline(never)]
    fn query_range_direct_multi_shard(
        snap: &mut RoutingSnapshot,
        first: usize,
        range: QueryRange,
    ) -> bool {
        let end_excl = snap.route_range_end_excl(first, range.end());
        for shard_arc in &mut snap.shards[first..end_excl] {
            let clamped = clamp_range_to_bounds(range, shard_arc.bounds);
            let hit = if let Some(shard) = Arc::get_mut(shard_arc) {
                shard.rcu_filter.query_range_direct(clamped)
            } else {
                let guard = crossbeam_epoch::pin();
                let slot = current_thread_stripe_slot();
                shard_arc
                    .rcu_filter
                    .query_range_pinned_with_slot(clamped, &guard, slot)
            };
            if hit {
                return true;
            }
        }
        false
    }

    /// Cold multi-shard range probe fallback when `range` straddles at least
    /// one shard boundary.
    #[cold]
    #[inline(never)]
    fn query_range_multi_shard(
        snap: &RoutingSnapshot,
        first: usize,
        range: QueryRange,
        guard: &Guard,
        slot: usize,
    ) -> bool {
        let end_excl = snap.route_range_end_excl(first, range.end());
        for shard in &snap.shards[first..end_excl] {
            let clamped = clamp_range_to_bounds(range, shard.bounds);
            shard.record_query_hit_with_slot(slot);
            if shard
                .rcu_filter
                .query_range_pinned_with_slot(clamped, guard, slot)
            {
                return true;
            }
        }
        false
    }

    /// Records a verified false-positive empty interval `range` in every
    /// intersecting shard's [`FalsePositiveLog`] and monotonically adapts each
    /// shard on its clamped sub-interval.
    ///
    /// Uses a cursor-based loop acquiring at most one shard's `write_coordinator`
    /// at a time, automatically re-routing across concurrent live splits or merges
    /// without skipping any key in `range`.
    ///
    /// # Errors
    /// Propagates [`FalsePositiveLog`], filter expansion, or Reverse Map errors.
    pub fn adapt_range(&self, range: QueryRange) -> AerisResult<AdaptationReport> {
        let mut total = AdaptationReport::default();
        let mut cursor = range.start();

        loop {
            let guard = crossbeam_epoch::pin();
            let snap = self.routing.load(&guard);
            let shard = snap.route_key(cursor);
            let _write_guard = shard.write_coordinator.read();
            if shard.is_retired() {
                continue;
            }

            let sub_end = range.end().min(shard.bounds.max_key);
            let clamped = QueryRange::new(cursor, sub_end)?;

            let _ = shard.fp_log.record(clamped, current_unix_ms())?;
            let report = shard.rcu_filter.adapt(clamped)?;
            shard.fp_count.fetch_add(1, Ordering::Relaxed);
            {
                let mut est = shard.estimator.lock();
                est.record_query(clamped, true);
            }

            total.subqueries_checked += report.subqueries_checked;
            total.boxes_split += report.boxes_split;
            total.keys_rehashed += report.keys_rehashed;
            total.extension_slots_added += report.extension_slots_added;
            total.rejuvenated_boxes += report.rejuvenated_boxes;

            if sub_end == range.end() {
                return Ok(total);
            }
            cursor = sub_end + 1;
        }
    }

    /// Splits `shard_id` at the exact **data-driven median key** of its live key
    /// set into two balanced contiguous child shards (`[min, split_key]` and
    /// `[split_key + 1, max]`), migrating and replaying relevant [`FalsePositiveLog`]
    /// entries for 100% monotonic adaptivity and publishing the updated
    /// [`RoutingSnapshot`] atomically via [`EpochCell::swap`].
    ///
    /// Concurrent readers and writers on all other shards run completely unblocked;
    /// writers targeting `shard_id` observe `retired == true` and seamlessly re-route
    /// to the replacement children with zero lost writes.
    ///
    /// # Errors
    /// - Returns [`AerisError::ShardNotFound`] if `shard_id` is not active.
    /// - Returns [`AerisError::InvalidConfig`] if splitting would exceed `max_shards`
    ///   or if `shard_id` spans only a single key (`min_key == max_key`).
    /// - Propagates RocksDB, filter, or FP log errors.
    pub fn split_shard(&self, shard_id: ShardId) -> AerisResult<SplitReport> {
        let topo_guard = self.topology_lock.lock();
        self.split_shard_locked(&topo_guard, shard_id)
    }

    /// Merges two strictly adjacent shards (`left_id`, `right_id` where
    /// `left.max_key + 1 == right.min_key`) into a single consolidated shard
    /// `[left.min_key, right.max_key]`, combining both shards' live keys and
    /// [`FalsePositiveLog`]s with full monotonic replay.
    ///
    /// Acquires `left.write_coordinator` followed by `right.write_coordinator` in
    /// **strict ascending `ShardBounds::min_key` order** to prevent deadlocks.
    ///
    /// # Errors
    /// - Returns [`AerisError::ShardNotFound`] if either shard is not active.
    /// - Returns [`AerisError::InvalidConfig`] if the shards are not adjacent in
    ///   `(left, right)` order or if merging would drop below `min_shards`.
    /// - Propagates RocksDB, filter, or FP log errors.
    pub fn merge_adjacent_shards(
        &self,
        left_id: ShardId,
        right_id: ShardId,
    ) -> AerisResult<MergeReport> {
        let topo_guard = self.topology_lock.lock();
        self.merge_adjacent_shards_locked(&topo_guard, left_id, right_id)
    }

    /// Evaluates all active shards under `topology_lock`, automatically splitting
    /// overloaded/hot-spot shards (exceeding `split_load_threshold`, `max_shard_q`,
    /// or `split_key_threshold`) and merging adjacent underloaded shards (both
    /// `< merge_load_threshold` with combined occupancy `<= 0.50`).
    ///
    /// # Errors
    /// Propagates any storage, filter, or FP log error encountered during split or merge.
    pub fn rebalance_if_needed(&self) -> AerisResult<RebalanceReport> {
        let topo_guard = self.topology_lock.lock();
        let mut report = RebalanceReport::default();
        let mut unsplitable: HashSet<ShardId> = HashSet::new();

        // Phase 1: Split overloaded or hot-spot shards (bounded to guarantee termination).
        for _ in 0..self.config.max_shards {
            let candidate = self.routing.with_ref(|snap| {
                if snap.shards.len() >= self.config.max_shards {
                    return None;
                }
                snap.shards
                    .iter()
                    .find(|s| !unsplitable.contains(&s.shard_id) && self.should_split_shard(s))
                    .map(|s| s.shard_id)
            });

            let Some(victim_id) = candidate else {
                break;
            };

            match self.split_shard_locked(&topo_guard, victim_id) {
                Ok(split_rep) => report.splits.push(split_rep),
                Err(AerisError::InvalidConfig(reason)) => {
                    warn!(
                        shard_id = victim_id,
                        reason, "skipping unsplittable shard during rebalance"
                    );
                    unsplitable.insert(victim_id);
                }
                Err(err) => return Err(err),
            }
        }

        // Phase 2: Merge adjacent underloaded shards (bounded).
        for _ in 0..self.config.max_shards {
            let candidate_pair = self.routing.with_ref(|snap| {
                if snap.shards.len() <= self.config.min_shards {
                    return None;
                }
                snap.shards.windows(2).find_map(|pair| {
                    if self.should_merge_pair(&pair[0], &pair[1]) {
                        Some((pair[0].shard_id, pair[1].shard_id))
                    } else {
                        None
                    }
                })
            });

            let Some((left_id, right_id)) = candidate_pair else {
                break;
            };

            let merge_rep = self.merge_adjacent_shards_locked(&topo_guard, left_id, right_id)?;
            report.merges.push(merge_rep);
        }

        report.final_shard_count = self.shard_count();
        report.final_routing_epoch = self.routing_epoch();
        debug!(
            splits = report.splits.len(),
            merges = report.merges.len(),
            final_shard_count = report.final_shard_count,
            epoch = report.final_routing_epoch,
            "completed ShardedAerisSystem rebalance pass"
        );
        Ok(report)
    }

    /// Allocates a strictly unique monotonic [`ShardId`].
    #[inline]
    fn allocate_shard_id(&self) -> ShardId {
        self.next_shard_id.fetch_add(1, Ordering::AcqRel)
    }

    /// Evaluates whether `shard` qualifies for a dynamic split.
    fn should_split_shard(&self, shard: &ShardInstance) -> bool {
        if shard.bounds.min_key >= shard.bounds.max_key {
            return false;
        }
        let (key_count, current_q, load) = shard.rcu_filter.split_check_metrics();
        if key_count < self.config.split_min_keys || key_count < 2 {
            return false;
        }
        let exceeds_load = load >= self.config.split_load_threshold;
        let exceeds_max_q = current_q > self.config.max_shard_q;
        let exceeds_key_cap = self
            .config
            .split_key_threshold
            .is_some_and(|limit| key_count >= limit);
        exceeds_load || exceeds_max_q || exceeds_key_cap
    }

    /// Evaluates whether adjacent shards `(left, right)` qualify for merging.
    fn should_merge_pair(&self, left: &ShardInstance, right: &ShardInstance) -> bool {
        let left_load = left.rcu_filter.load_factor();
        let right_load = right.rcu_filter.load_factor();
        if left_load >= self.config.merge_load_threshold
            || right_load >= self.config.merge_load_threshold
        {
            return false;
        }
        let left_q = left.rcu_filter.config().q;
        let right_q = right.rcu_filter.config().q;
        let target_q = self.config.base_filter_config.q.max(left_q.min(right_q));
        let target_cap = (1usize << target_q) as f64;
        let combined_slots =
            (left.rcu_filter.occupied_slots() + right.rcu_filter.occupied_slots()) as f64;
        let combined_keys = (left.rcu_filter.len() + right.rcu_filter.len()) as f64;

        (combined_slots / target_cap) <= MAX_POST_MERGE_LOAD_FACTOR
            && (combined_keys / target_cap) <= MAX_POST_MERGE_LOAD_FACTOR
    }

    /// Derives the baseline [`FilterConfig`] for newly created split/merged children,
    /// resetting `q` to `self.config.base_filter_config.q` while preserving any
    /// dynamic `(r, m, max_range_r, target_epsilon)` tuning from `parent_cfg`.
    fn derived_child_filter_config(&self, parent_cfg: &FilterConfig) -> FilterConfig {
        let mut child_cfg = parent_cfg.clone();
        child_cfg.q = self.config.base_filter_config.q;
        child_cfg
    }

    /// Executes `split_shard` while the caller holds `self.topology_lock`.
    fn split_shard_locked(
        &self,
        _topo_guard: &MutexGuard<'_, ()>,
        shard_id: ShardId,
    ) -> AerisResult<SplitReport> {
        let start = Instant::now();
        let (shard_idx, old_shards, target) = self.routing.with_ref(|snap| {
            let idx = snap
                .find_shard_index(shard_id)
                .ok_or(AerisError::ShardNotFound(shard_id))?;
            Ok::<_, AerisError>((idx, snap.shards.to_vec(), Arc::clone(&snap.shards[idx])))
        })?;

        if old_shards.len() >= self.config.max_shards {
            return Err(AerisError::InvalidConfig(format!(
                "cannot split shard {shard_id}: shard count {} already at max_shards {}",
                old_shards.len(),
                self.config.max_shards
            )));
        }
        if target.bounds.min_key >= target.bounds.max_key {
            return Err(AerisError::InvalidConfig(format!(
                "cannot split single-key shard {shard_id} [{}, {}]",
                target.bounds.min_key, target.bounds.max_key
            )));
        }

        // Level 2 lock (exclusive): quiesce writers on `target` only; readers on
        // `target` and all traffic on every other shard proceed uninterrupted.
        let _write_guard = target.write_coordinator.write();
        if target.is_retired() {
            return Err(AerisError::ShardNotFound(shard_id));
        }

        let all_keys = target.active_reverse_map().scan_all_keys()?;
        let (_, fp_entries) = target.fp_log.snapshot_entries();

        let split_key = compute_median_split_key(target.bounds, &all_keys);
        let left_bounds = ShardBounds::new(target.bounds.min_key, split_key)?;
        let right_bounds = ShardBounds::new(split_key + 1, target.bounds.max_key)?;

        let split_pos = all_keys.partition_point(|&k| k <= split_key);
        let left_keys = &all_keys[..split_pos];
        let right_keys = &all_keys[split_pos..];

        let left_id = self.allocate_shard_id();
        let right_id = self.allocate_shard_id();
        let child_cfg = self.derived_child_filter_config(&target.rcu_filter.config());

        let (left_shard, fp_left) = build_shard_from_state(
            &self.root_dir,
            &self.config,
            left_id,
            left_bounds,
            &child_cfg,
            left_keys,
            &fp_entries,
        )?;
        let (right_shard, fp_right) = build_shard_from_state(
            &self.root_dir,
            &self.config,
            right_id,
            right_bounds,
            &child_cfg,
            right_keys,
            &fp_entries,
        )?;

        let mut new_shards = Vec::with_capacity(old_shards.len() + 1);
        new_shards.extend_from_slice(&old_shards[..shard_idx]);
        new_shards.push(Arc::new(left_shard));
        new_shards.push(Arc::new(right_shard));
        new_shards.extend_from_slice(&old_shards[shard_idx + 1..]);
        let new_snapshot = RoutingSnapshot::new(new_shards)?;

        // Publish new RoutingSnapshot FIRST, then mark `target` retired BEFORE releasing
        // `target.write_coordinator`, guaranteeing waiting writers immediately observe the new route.
        let new_epoch = self.routing.swap(new_snapshot);
        target.mark_retired();
        drop(_write_guard);
        drop(old_shards);
        drop(target);
        // Opportunistically reclaim retired snapshot/RocksDB on the split thread
        // when no older reader guards are pinned so hot-path inserts avoid DB close stalls.
        try_reclaim_retired_epochs();

        let latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            shard_id,
            left_id,
            right_id,
            split_key,
            left_keys = left_keys.len(),
            right_keys = right_keys.len(),
            fp_left,
            fp_right,
            epoch = new_epoch,
            latency_ns,
            "completed dynamic range shard split"
        );

        Ok(SplitReport {
            old_shard_id: shard_id,
            left_shard_id: left_id,
            right_shard_id: right_id,
            split_key,
            left_bounds,
            right_bounds,
            left_keys: left_keys.len(),
            right_keys: right_keys.len(),
            fp_entries_replayed_left: fp_left,
            fp_entries_replayed_right: fp_right,
            new_routing_epoch: new_epoch,
            latency_ns,
        })
    }

    /// Executes `merge_adjacent_shards` while the caller holds `self.topology_lock`.
    fn merge_adjacent_shards_locked(
        &self,
        _topo_guard: &MutexGuard<'_, ()>,
        left_id: ShardId,
        right_id: ShardId,
    ) -> AerisResult<MergeReport> {
        let start = Instant::now();
        let (left_idx, old_shards, left, right) = self.routing.with_ref(|snap| {
            let l_idx = snap
                .find_shard_index(left_id)
                .ok_or(AerisError::ShardNotFound(left_id))?;
            let r_idx = snap
                .find_shard_index(right_id)
                .ok_or(AerisError::ShardNotFound(right_id))?;
            if l_idx + 1 != r_idx {
                return Err(AerisError::InvalidConfig(format!(
                    "shards ({left_id}, {right_id}) are not adjacent in ascending key order"
                )));
            }
            Ok::<_, AerisError>((
                l_idx,
                snap.shards.to_vec(),
                Arc::clone(&snap.shards[l_idx]),
                Arc::clone(&snap.shards[r_idx]),
            ))
        })?;

        if old_shards.len() <= self.config.min_shards {
            return Err(AerisError::InvalidConfig(format!(
                "cannot merge shards ({left_id}, {right_id}): shard count {} already at min_shards {}",
                old_shards.len(),
                self.config.min_shards
            )));
        }

        debug_assert!(
            left.bounds.min_key < right.bounds.min_key
                && left.bounds.max_key.checked_add(1) == Some(right.bounds.min_key),
            "adjacent shards must be strictly ascending and contiguous"
        );

        // Level 2 lock (exclusive): strictly ascending `ShardBounds::min_key` order (`left` then `right`).
        let _left_write = left.write_coordinator.write();
        let _right_write = right.write_coordinator.write();
        if left.is_retired() {
            return Err(AerisError::ShardNotFound(left_id));
        }
        if right.is_retired() {
            return Err(AerisError::ShardNotFound(right_id));
        }

        let left_keys = left.active_reverse_map().scan_all_keys()?;
        let right_keys = right.active_reverse_map().scan_all_keys()?;
        let mut merged_keys = Vec::with_capacity(left_keys.len() + right_keys.len());
        merged_keys.extend_from_slice(&left_keys);
        merged_keys.extend_from_slice(&right_keys);
        debug_assert!(
            merged_keys.windows(2).all(|w| w[0] < w[1]),
            "disjoint adjacent shard keys must concatenate in strict ascending order"
        );

        let (_, mut combined_fps) = left.fp_log.snapshot_entries();
        let (_, right_fps) = right.fp_log.snapshot_entries();
        combined_fps.extend(right_fps);

        let merged_bounds = ShardBounds::new(left.bounds.min_key, right.bounds.max_key)?;
        let merged_id = self.allocate_shard_id();
        let child_cfg = self.derived_child_filter_config(&left.rcu_filter.config());

        let (merged_shard, fp_replayed) = build_shard_from_state(
            &self.root_dir,
            &self.config,
            merged_id,
            merged_bounds,
            &child_cfg,
            &merged_keys,
            &combined_fps,
        )?;

        let mut new_shards = Vec::with_capacity(old_shards.len() - 1);
        new_shards.extend_from_slice(&old_shards[..left_idx]);
        new_shards.push(Arc::new(merged_shard));
        new_shards.extend_from_slice(&old_shards[left_idx + 2..]);
        let new_snapshot = RoutingSnapshot::new(new_shards)?;

        // Publish new RoutingSnapshot FIRST, then mark both old shards retired BEFORE
        // releasing their `write_coordinator` locks.
        let new_epoch = self.routing.swap(new_snapshot);
        left.mark_retired();
        right.mark_retired();
        drop(_right_write);
        drop(_left_write);
        drop(old_shards);
        drop(right);
        drop(left);
        try_reclaim_retired_epochs();

        let latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            left_id,
            right_id,
            merged_id,
            merged_keys = merged_keys.len(),
            fp_replayed,
            epoch = new_epoch,
            latency_ns,
            "completed adjacent range shard merge"
        );

        Ok(MergeReport {
            left_shard_id: left_id,
            right_shard_id: right_id,
            merged_shard_id: merged_id,
            merged_bounds,
            merged_keys: merged_keys.len(),
            fp_entries_replayed: fp_replayed,
            new_routing_epoch: new_epoch,
            latency_ns,
        })
    }
}

/// Builds a fresh persistent [`ShardInstance`] at `<root_dir>/shard_<shard_id>/`,
/// bulk-loads `sorted_keys` with auto-expansion, migrates relevant empty FP log
/// entries clamped to `bounds`, and replays them for 100% monotonic adaptivity.
fn build_shard_from_state(
    root_dir: &Path,
    config: &ShardingConfig,
    shard_id: ShardId,
    bounds: ShardBounds,
    base_cfg: &FilterConfig,
    sorted_keys: &[Key],
    source_fp_entries: &[FpLogEntry],
) -> AerisResult<(ShardInstance, usize)> {
    let shard_dir = root_dir.join(format!("shard_{shard_id}"));
    fs::create_dir_all(&shard_dir).map_err(|err| {
        AerisError::WalIoError(format!(
            "failed to create shard directory '{}': {err}",
            shard_dir.display()
        ))
    })?;

    let reverse_map = Arc::new(RocksDbReverseMap::open(
        shard_dir.join(SHARD_REVERSE_MAP_DIR),
        config.reverse_map_config.clone(),
    )?);
    let fp_log = Arc::new(FalsePositiveLog::open(
        shard_dir.join(SHARD_FP_LOG_FILE),
        config.fp_log_config,
    )?);

    let mut filter = bulk_load_with_auto_expand(base_cfg, sorted_keys, reverse_map.as_ref())?;

    // Migrate overlapping still-empty FP intervals clamped to `bounds` into this shard's FP log.
    for entry in source_fp_entries {
        if bounds.overlaps_range(entry.range.start(), entry.range.end()) {
            let clamped = clamp_range_to_bounds(entry.range, bounds);
            if sorted_slice_range_is_empty(sorted_keys, clamped) {
                let _ = fp_log.record(clamped, entry.timestamp_ms)?;
            }
        }
    }

    let fp_replayed = if fp_log.is_empty() {
        0
    } else {
        let replay = fp_log.replay_onto(&mut filter, reverse_map.as_ref(), |r| {
            Ok(sorted_slice_range_is_empty(sorted_keys, r))
        })?;
        replay.replayed_entries
    };

    let active_max_range_r = filter.config().max_range_r;
    let estimator = SlidingWindowEstimator::with_initial_span(
        config.estimator_config.clone(),
        active_max_range_r,
    )?;
    let rcu_filter = RcuAerisFilter::with_max_stripes(
        shard_id,
        filter,
        Arc::clone(&reverse_map) as Arc<dyn ReverseMapStore>,
        config.max_lock_stripes_per_shard,
    );

    let instance = ShardInstance {
        shard_id,
        bounds,
        rcu_filter,
        reverse_map,
        fp_log,
        estimator: Mutex::new(estimator),
        write_coordinator: CachePadded::new(RwLock::new(())),
        retired: CachePadded::new(AtomicBool::new(false)),
        query_counts: std::array::from_fn(|_| CachePadded::new(AtomicU64::new(0))),
        fp_count: CachePadded::new(AtomicU64::new(0)),
        write_count: CachePadded::new(AtomicU64::new(0)),
    };
    Ok((instance, fp_replayed))
}

/// Bulk-loads `keys` into a new [`AerisFilter`], automatically expanding `q -> q + 1`
/// whenever [`AerisError::FilterFull`] occurs before writing to `reverse_map`.
fn bulk_load_with_auto_expand(
    base_cfg: &FilterConfig,
    keys: &[Key],
    reverse_map: &dyn ReverseMapStore,
) -> AerisResult<AerisFilter> {
    if keys.is_empty() {
        return AerisFilter::new(base_cfg.clone());
    }
    let mut cfg = base_cfg.clone();
    let mut expansions = 0u8;
    loop {
        match AerisFilter::bulk_load(cfg.clone(), keys, reverse_map) {
            Ok(filter) => return Ok(filter),
            Err(AerisError::FilterFull { .. }) if expansions < MAX_QUOTIENT_BITS => {
                cfg = cfg.expanded()?;
                expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Computes `shard_count` contiguous uniform [`ShardBounds`] partitioning `[0, u64::MAX]`
/// using overflow-free `u128` arithmetic.
///
/// # Errors
/// Returns [`AerisError::InvalidConfig`] if `shard_count == 0`.
pub fn compute_uniform_bounds(shard_count: usize) -> AerisResult<Vec<ShardBounds>> {
    if shard_count == 0 {
        return Err(AerisError::InvalidConfig(
            "shard_count must be at least 1".to_string(),
        ));
    }
    let universe: u128 = (u64::MAX as u128) + 1;
    let n = shard_count as u128;
    let mut out = Vec::with_capacity(shard_count);
    for i in 0..shard_count {
        let start = ((i as u128) * universe / n) as u64;
        let end_excl = (((i + 1) as u128) * universe) / n;
        let max_key = (end_excl - 1) as u64;
        out.push(ShardBounds::new(start, max_key)?);
    }
    Ok(out)
}

/// Builds contiguous [`ShardBounds`] covering `[0, u64::MAX]` from strictly
/// ascending inclusive split points `split_keys` (each `< u64::MAX`).
///
/// # Errors
/// Returns [`AerisError::InvalidConfig`] if `split_keys` is not strictly ascending
/// or contains `u64::MAX`.
pub fn compute_bounds_from_split_keys(split_keys: &[Key]) -> AerisResult<Vec<ShardBounds>> {
    let mut out = Vec::with_capacity(split_keys.len() + 1);
    let mut current_min = 0u64;
    for &split in split_keys {
        if split == u64::MAX {
            return Err(AerisError::InvalidConfig(
                "split boundary cannot equal u64::MAX".to_string(),
            ));
        }
        if split < current_min {
            return Err(AerisError::InvalidConfig(format!(
                "split boundaries must be strictly ascending: {split} < {current_min}"
            )));
        }
        out.push(ShardBounds::new(current_min, split)?);
        current_min = split + 1;
    }
    out.push(ShardBounds::new(current_min, u64::MAX)?);
    Ok(out)
}

/// Computes the data-driven median split boundary `split_key` for a shard spanning
/// `bounds` (with `bounds.min_key < bounds.max_key`) and holding strictly sorted
/// deduplicated `sorted_keys`.
///
/// Guarantees `bounds.min_key <= split_key < bounds.max_key` so both
/// `[bounds.min_key, split_key]` and `[split_key + 1, bounds.max_key]` are valid
/// non-empty key intervals without `u64` overflow.
#[inline]
#[must_use]
pub fn compute_median_split_key(bounds: ShardBounds, sorted_keys: &[Key]) -> Key {
    debug_assert!(
        bounds.min_key < bounds.max_key,
        "compute_median_split_key requires min_key < max_key"
    );
    if sorted_keys.len() >= 2 {
        let mid_idx = (sorted_keys.len() - 1) / 2;
        let candidate = sorted_keys[mid_idx];
        // Because `sorted_keys` is strictly increasing and lies inside `bounds`,
        // `candidate < sorted_keys[mid_idx + 1] <= bounds.max_key`.
        if candidate >= bounds.min_key && candidate < bounds.max_key {
            return candidate;
        }
    }
    bounds.min_key + (bounds.max_key - bounds.min_key) / 2
}

/// Clamps `range` to the non-empty intersection with `bounds`.
///
/// # Preconditions
/// `bounds.overlaps_range(range.start(), range.end())` MUST be `true`.
#[inline]
fn clamp_range_to_bounds(range: QueryRange, bounds: ShardBounds) -> QueryRange {
    let start = range.start().max(bounds.min_key);
    let end = range.end().min(bounds.max_key);
    QueryRange::new(start, end).expect("overlapping range clamp guarantees start <= end")
}

/// Exact `O(log K)` zero-allocation emptiness check against a sorted key slice.
#[inline]
fn sorted_slice_range_is_empty(sorted_keys: &[Key], range: QueryRange) -> bool {
    let pos = sorted_keys.partition_point(|&k| k < range.start());
    pos >= sorted_keys.len() || sorted_keys[pos] > range.end()
}

/// Returns current Unix timestamp in milliseconds for [`FalsePositiveLog`] entries.
#[inline]
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn base_filter_cfg(q: u8) -> FilterConfig {
        FilterConfig::new(q, 14, 64, 20, 1.0 / 256.0).unwrap()
    }

    #[test]
    fn uniform_bounds_cover_full_u64_universe_for_all_counts() {
        for count in [1usize, 2, 3, 4, 7, 8, 16, 64, 256] {
            let bounds = compute_uniform_bounds(count).unwrap();
            assert_eq!(bounds.len(), count);
            assert_eq!(bounds[0].min_key, 0);
            assert_eq!(bounds.last().unwrap().max_key, u64::MAX);
            for w in bounds.windows(2) {
                assert!(w[0].min_key <= w[0].max_key);
                assert_eq!(w[0].max_key.checked_add(1), Some(w[1].min_key));
            }
        }
        assert!(compute_uniform_bounds(0).is_err());
    }

    #[test]
    fn explicit_split_boundaries_validation() {
        let b = compute_bounds_from_split_keys(&[]).unwrap();
        assert_eq!(b, vec![ShardBounds::full_universe()]);

        let b2 = compute_bounds_from_split_keys(&[0, 100, u64::MAX - 1]).unwrap();
        assert_eq!(
            b2,
            vec![
                ShardBounds::new(0, 0).unwrap(),
                ShardBounds::new(1, 100).unwrap(),
                ShardBounds::new(101, u64::MAX - 1).unwrap(),
                ShardBounds::new(u64::MAX, u64::MAX).unwrap(),
            ]
        );

        assert!(compute_bounds_from_split_keys(&[10, 10]).is_err());
        assert!(compute_bounds_from_split_keys(&[20, 10]).is_err());
        assert!(compute_bounds_from_split_keys(&[u64::MAX]).is_err());
    }

    #[test]
    fn median_split_handles_skewed_hotspot_and_extremes() {
        let full = ShardBounds::full_universe();
        let skewed: Vec<Key> = (1_000_000..1_000_100).collect();
        let mid = compute_median_split_key(full, &skewed);
        assert_eq!(mid, 1_000_049);

        let two_at_top = vec![u64::MAX - 1, u64::MAX];
        assert_eq!(compute_median_split_key(full, &two_at_top), u64::MAX - 1);

        let empty_mid = compute_median_split_key(full, &[]);
        assert_eq!(empty_mid, u64::MAX / 2);
    }

    #[test]
    fn routing_binary_search_exact_on_all_shard_boundaries() {
        let dir = TempDir::new().unwrap();
        let cfg = ShardingConfig::new(base_filter_cfg(8)).with_initial_shard_count(5);
        let sys = ShardedAerisSystem::open(dir.path(), cfg).unwrap();
        sys.verify_routing_invariants().unwrap();

        let bounds = sys.shard_bounds();
        assert_eq!(bounds.len(), 5);
        for (sid, b) in &bounds {
            assert_eq!(sys.route_shard_id(b.min_key), *sid);
            assert_eq!(sys.route_shard_id(b.max_key), *sid);
            let mid = b.min_key + (b.max_key - b.min_key) / 2;
            assert_eq!(sys.route_shard_id(mid), *sid);
        }

        // Cross-shard insert and range query verification.
        let mut keys = Vec::new();
        for (_, b) in &bounds {
            keys.push(b.min_key);
            keys.push(b.max_key);
        }
        for &k in &keys {
            sys.insert(k).unwrap();
            assert!(sys.query_point(k));
        }

        // Full universe query returns true; every shard boundary pair returns true.
        assert!(sys.query_range(QueryRange::new(0, u64::MAX).unwrap()));
        for w in bounds.windows(2) {
            let span = QueryRange::new(w[0].1.max_key, w[1].1.min_key).unwrap();
            assert!(sys.query_range(span));
        }
        sys.verify_all_invariants().unwrap();
    }

    #[test]
    fn flat_routing_and_striped_query_counts_exact_above_and_below_linear_threshold() {
        for shard_count in [1usize, 8, 16, 17, 26] {
            let dir = TempDir::new().unwrap();
            let cfg = ShardingConfig::new(base_filter_cfg(8))
                .with_initial_shard_count(shard_count)
                .with_max_shards(64);
            let sys = Arc::new(ShardedAerisSystem::open(dir.path(), cfg).unwrap());
            let bounds = sys.shard_bounds();
            assert_eq!(bounds.len(), shard_count);

            for (sid, b) in &bounds {
                assert_eq!(sys.route_shard_id(b.min_key), *sid);
                assert_eq!(sys.route_shard_id(b.max_key), *sid);
                sys.insert(b.min_key).unwrap();
                let handle = sys.shard_handle(*sid).unwrap();
                assert_eq!(handle.key_count(), 1);
            }

            // Exercise multi-threaded striped query_count exactness across 6 threads.
            const THREADS: usize = 6;
            const QUERIES_PER_THREAD: usize = 50;
            let probe_key = bounds[0].1.min_key;
            std::thread::scope(|s| {
                for _ in 0..THREADS {
                    let sys_ref = Arc::clone(&sys);
                    s.spawn(move || {
                        for _ in 0..QUERIES_PER_THREAD {
                            assert!(sys_ref.query_point(probe_key));
                        }
                    });
                }
            });
            let shard0 = sys.shard_handle(bounds[0].0).unwrap();
            assert_eq!(shard0.query_count(), (THREADS * QUERIES_PER_THREAD) as u64);

            // Cross-boundary range queries spanning 2 shards and full universe.
            assert!(sys.query_range(QueryRange::new(0, u64::MAX).unwrap()));
            for w in bounds.windows(2) {
                let r = QueryRange::new(w[0].1.max_key, w[1].1.min_key).unwrap();
                assert!(sys.query_range(r));
            }
            sys.verify_all_invariants().unwrap();
        }
    }

    #[test]
    fn direct_bypass_matches_concurrent_queries_across_1_shard_and_8_shard_topologies() {
        for &(shard_count, bypass_flag) in &[(1usize, true), (1, false), (8, true), (8, false)] {
            let dir = TempDir::new().unwrap();
            let cfg = ShardingConfig::new(base_filter_cfg(7))
                .with_initial_shard_count(shard_count)
                .with_single_shard_bypass(bypass_flag);
            assert_eq!(cfg.single_shard_bypass, bypass_flag);
            let mut sys = ShardedAerisSystem::open(dir.path(), cfg).unwrap();

            let bounds = sys.shard_bounds();
            assert_eq!(bounds.len(), shard_count);

            // Insert keys across all shards via `insert_direct` (triggering auto-expansion too).
            let mut inserted = Vec::new();
            for (shard_idx, &(sid, b)) in bounds.iter().enumerate() {
                let step = if shard_count == 1 { 53u64 } else { 97u64 };
                let count = if shard_count == 1 { 200u64 } else { 40u64 };
                for i in 0..count {
                    let k = b.min_key.saturating_add(i * step + (shard_idx as u64));
                    if b.contains(k) {
                        let assigned = sys.insert_direct(k).unwrap();
                        assert_eq!(assigned, sid);
                        inserted.push(k);
                    }
                }
            }
            // Also include extreme u64 boundary keys.
            for &k in &[0u64, u64::MAX] {
                let _ = sys.insert_direct(k).unwrap();
                inserted.push(k);
            }

            // 1. Zero false negatives & bit-exact agreement on every inserted key.
            for &k in &inserted {
                assert!(sys.query_point_direct(k));
                assert!(sys.query_point(k));
                let r = QueryRange::new(k, k.saturating_add(15)).unwrap();
                assert!(sys.query_range_direct(r));
                assert!(sys.query_range(r));
            }

            // 2. Bit-exact agreement on mixed present/absent point & range probes,
            // including cross-shard boundary straddling ranges and full-universe range.
            for (_, b) in &bounds {
                for probe in 0..100u64 {
                    let k = b.min_key.saturating_add(1_000_000 + probe * 131);
                    if b.contains(k) {
                        assert_eq!(sys.query_point_direct(k), sys.query_point(k));
                        let r = QueryRange::new(k, k.saturating_add(63)).unwrap();
                        assert_eq!(sys.query_range_direct(r), sys.query_range(r));
                    }
                }
            }
            for w in bounds.windows(2) {
                let cross = QueryRange::new(
                    w[0].1.max_key.saturating_sub(32),
                    w[1].1.min_key.saturating_add(32),
                )
                .unwrap();
                assert_eq!(sys.query_range_direct(cross), sys.query_range(cross));
            }
            let full = QueryRange::new(0, u64::MAX).unwrap();
            assert_eq!(sys.query_range_direct(full), sys.query_range(full));

            // 3. Verify external `Arc<ShardInstance>` clone fallback path remains bit-exact.
            let held_handle = sys.shard_handle(bounds[0].0).unwrap();
            let extra_key = bounds[0].1.min_key.saturating_add(7_777_777);
            if bounds[0].1.contains(extra_key) {
                let sid = sys.insert_direct(extra_key).unwrap();
                assert_eq!(sid, bounds[0].0);
                assert!(sys.query_point_direct(extra_key));
                assert_eq!(
                    sys.query_range_direct(QueryRange::new(extra_key, extra_key + 7).unwrap()),
                    sys.query_range(QueryRange::new(extra_key, extra_key + 7).unwrap()),
                );
            }
            drop(held_handle);

            sys.verify_all_invariants().unwrap();
        }
    }
}
