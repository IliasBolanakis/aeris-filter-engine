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
//!     a shared read lock on the target shard's `write_coordinator`, and check
//!     `shard.is_retired()`. If `true`, they drop the lock and retry against the
//!     updated [`RoutingSnapshot`].
//!   - `split_shard` and `merge_adjacent_shards` acquire `topology_lock` ->
//!     exclusive `write_coordinator.write()` in ascending `min_key` order,
//!     publish the replacement [`RoutingSnapshot`] via [`EpochCell::swap`], and
//!     mark predecessor shards retired before releasing `write_coordinator`.
//!
//! # Lock Ordering
//! 1. `ShardedAerisSystem::topology_lock` (split/merge/rebalance only; NEVER on normal reads/writes)
//! 2. `ShardInstance::write_coordinator` (strict ascending `ShardBounds::min_key` order)
//! 3. `RcuAerisFilter::swap_lock`
//! 4. `ChunkLockTable` stripes (strict ascending stripe index order)
//! 5. `RcuGeneration::filter` (`StripedRwLock<AerisFilter>`)
//! 6. `ReverseMapStore` / `FalsePositiveLog` / `DeltaBuffer` / `SlidingWindowEstimator`

#![forbid(unsafe_code)]

mod config;
mod rebalance;
mod routing;
mod shard;
#[cfg(test)]
mod tests;

pub use config::{
    ShardingConfig, DEFAULT_INITIAL_SHARD_COUNT, DEFAULT_MAX_SHARDS, DEFAULT_MERGE_LOAD_THRESHOLD,
    DEFAULT_MIN_SHARDS, DEFAULT_SPLIT_LOAD_THRESHOLD, DEFAULT_SPLIT_MIN_KEYS,
    MAX_POST_MERGE_LOAD_FACTOR,
};
pub use rebalance::{MergeReport, RebalanceReport, SplitReport};
pub use routing::{
    compute_bounds_from_split_keys, compute_median_split_key, compute_uniform_bounds,
    RoutingSnapshot,
};
pub use shard::ShardInstance;

use crate::epoch_cell::{current_thread_stripe_slot, EpochCell};
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, Epoch, Key, QueryRange, ShardBounds, ShardId,
    ShardTelemetrySnapshot, WorkloadEstimator,
};
use crossbeam_epoch::Guard;
use parking_lot::Mutex;
use shard::{build_shard_from_state, clamp_range_to_bounds, current_unix_ms};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::info;

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

    /// Verifies routing invariants, every active shard's Quotient Filter
    /// invariants, and that every persisted Reverse Map key lies inside its owning
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
    /// # Complexity
    /// `O(log S + intersecting_shards * probes_per_shard)`, zero locks, zero epoch
    /// pinning, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_range_direct(&mut self, range: QueryRange) -> bool {
        let snap = self.routing.get_mut();
        if self.config.single_shard_bypass && snap.shards.len() == 1 {
            return query_range_shard_direct(&mut snap.shards[0], range);
        }
        let first = snap.route_key_index(range.start());
        if range.end() <= snap.max_keys[first] {
            return query_range_shard_direct(&mut snap.shards[first], range);
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
            if query_range_shard_direct(shard_arc, clamped) {
                return true;
            }
        }
        false
    }

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
    /// intersecting shard's [`aeris_control::FalsePositiveLog`] and monotonically adapts each
    /// shard on its clamped sub-interval.
    ///
    /// # Errors
    /// Propagates [`aeris_control::FalsePositiveLog`], filter expansion, or Reverse Map errors.
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
            shard.estimator.lock().record_query(clamped, true);

            accumulate_adaptation(&mut total, report);

            if sub_end == range.end() {
                return Ok(total);
            }
            cursor = sub_end + 1;
        }
    }

    /// Allocates a strictly unique monotonic [`ShardId`].
    #[inline]
    pub(super) fn allocate_shard_id(&self) -> ShardId {
        self.next_shard_id.fetch_add(1, Ordering::AcqRel)
    }
}

#[inline]
fn query_range_shard_direct(shard_arc: &mut Arc<ShardInstance>, range: QueryRange) -> bool {
    if let Some(shard) = Arc::get_mut(shard_arc) {
        shard.rcu_filter.query_range_direct(range)
    } else {
        let guard = crossbeam_epoch::pin();
        let slot = current_thread_stripe_slot();
        shard_arc
            .rcu_filter
            .query_range_pinned_with_slot(range, &guard, slot)
    }
}

#[inline]
fn accumulate_adaptation(total: &mut AdaptationReport, report: AdaptationReport) {
    total.subqueries_checked += report.subqueries_checked;
    total.boxes_split += report.boxes_split;
    total.keys_rehashed += report.keys_rehashed;
    total.extension_slots_added += report.extension_slots_added;
    total.rejuvenated_boxes += report.rejuvenated_boxes;
}
