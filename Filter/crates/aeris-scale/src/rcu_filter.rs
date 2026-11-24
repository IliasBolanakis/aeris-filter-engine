//! Crossbeam epoch-based RCU wrapper ([`RcuAerisFilter`]) coupling zero-stall
//! concurrent readers with fine-grained chunk-striped writers and atomic
//! `F_old -> F_new` swaps.
//!
//! # Concurrency & Zero-Stall Reader Guarantee
//! - **Readers (`query_point`, `query_range`):** Pin a Crossbeam epoch guard,
//!   load `&FilterGeneration` lock-free from [`EpochCell`], acquire a shared
//!   read lock on `FilterGeneration::filter`, and execute with **zero heap
//!   allocations**. Readers NEVER acquire `swap_lock` or `ChunkLockTable`
//!   stripes, so readers on `F_old` run completely uninterrupted throughout
//!   `2x` expansions, background dynamic-`R` rebuilds, and atomic swaps.
//! - **Writers (`insert`, `delete`, `adapt`):** Acquire the affected chunk/cluster
//!   stripes in strict ascending order via [`ChunkLockTable`], verify that the
//!   loaded generation has not been retired (`FilterGeneration::is_retired`),
//!   acquire `FilterGeneration::filter.write()`, persist to the generation's
//!   coupled [`ReverseMapStore`], and mirror mutations into [`DeltaBuffer`] when
//!   a background rebuild is active.
//! - **Swappers (`swap_filter`, `expand_rcu`, `rebuild_async_and_swap`):**
//!   Serialize via `swap_lock`, quiesce writers on the retiring generation via
//!   `ChunkLockTable::write_all()` (WITHOUT blocking readers on `filter.read()`!),
//!   drain pending [`DeltaBuffer`] mutations and catch-up [`FalsePositiveLog`]
//!   entries, mark `gen_old` retired, and atomically publish `F_new` through
//!   [`EpochCell::swap`].
//!
//! # Lock Ordering
//! Strictly obeys the crate-level hierarchy:
//! `swap_lock` -> `ChunkLockTable` stripes (strict ascending) ->
//! `FilterGeneration::filter` -> `ReverseMapStore` / `FalsePositiveLog` / `DeltaBuffer`.

#![forbid(unsafe_code)]

use crate::chunk_locks::{ChunkLockTable, ClusterWriteGuard, DEFAULT_MAX_LOCK_STRIPES};
use crate::epoch_cell::{EpochCell, StripedRwLock};
use aeris_control::{
    AsyncRebuildCoordinator, DeltaBuffer, DeltaMutation, FalsePositiveLog, IoThrottleConfig,
    RebuildReason, RebuildReport, DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
    DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES, DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
};
use aeris_core::AerisFilter;
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, Epoch, ExpansionReport, FilterConfig, Fingerprint,
    Key, PartitionHasher, QueryRange, RangeFilter, ReverseMapStore, ShardId, MAX_QUOTIENT_BITS,
    SLOTS_PER_CHUNK_LOG2,
};
use crossbeam_epoch::Guard;
use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tracing::{debug, info};

/// Immutable generation snapshot published atomically through [`EpochCell`].
///
/// Couples a single [`AerisFilter`] instance with its matching [`FilterConfig`],
/// aligned [`ChunkLockTable`], generation-bound [`ReverseMapStore`], monotonic
/// [`Epoch`], and retirement flag so writers never mutate a swapped-out filter.
pub struct FilterGeneration {
    /// Monotonic RCU epoch assigned to this generation.
    pub epoch: Epoch,
    /// Validated geometric and probabilistic configuration active in this generation.
    pub config: FilterConfig,
    /// Fine-grained cache-line-padded chunk lock table aligned to `config.num_chunks()`.
    pub chunk_locks: ChunkLockTable,
    /// Active RAM [`AerisFilter`] protected by a reader-writer lock.
    pub filter: RwLock<AerisFilter>,
    /// Disk-resident [`ReverseMapStore`] aligned with this generation's fingerprints.
    reverse_map: Arc<dyn ReverseMapStore>,
    /// Set to `true` under `chunk_locks.write_all()` immediately when a successor
    /// generation is published via [`EpochCell::swap`].
    retired: AtomicBool,
}

impl FilterGeneration {
    /// Constructs a live [`FilterGeneration`] for `filter` and `reverse_map` at `epoch`.
    ///
    /// # Complexity
    /// `O(num_stripes)` time and space.
    #[must_use]
    pub fn new(
        epoch: Epoch,
        filter: AerisFilter,
        reverse_map: Arc<dyn ReverseMapStore>,
        max_lock_stripes: usize,
    ) -> Self {
        let config = filter.config().clone();
        let chunk_locks = ChunkLockTable::with_max_stripes(config.num_chunks(), max_lock_stripes);
        Self {
            epoch,
            config,
            chunk_locks,
            filter: RwLock::new(filter),
            reverse_map,
            retired: AtomicBool::new(false),
        }
    }

    /// Returns `true` if a newer generation has already replaced this one.
    #[inline]
    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// Returns a cloned handle to the [`ReverseMapStore`] bound to this generation.
    #[inline]
    #[must_use]
    pub fn reverse_map(&self) -> Arc<dyn ReverseMapStore> {
        Arc::clone(&self.reverse_map)
    }
}

/// Internal cache-line-optimized generation published inside [`RcuAerisFilter`].
///
/// Uses [`StripedRwLock`] so concurrent readers on distinct threads never bounce
/// a single reader-count cache line, isolates `retired` and `live_keys` onto
/// dedicated cache lines, and enables lock-free `O(1)` live key count queries.
struct RcuGeneration {
    epoch: Epoch,
    config: FilterConfig,
    chunk_locks: ChunkLockTable,
    filter: StripedRwLock<AerisFilter>,
    reverse_map: Arc<dyn ReverseMapStore>,
    retired: CachePadded<AtomicBool>,
    live_keys: CachePadded<AtomicUsize>,
}

impl RcuGeneration {
    #[must_use]
    fn new(
        epoch: Epoch,
        filter: AerisFilter,
        reverse_map: Arc<dyn ReverseMapStore>,
        max_lock_stripes: usize,
    ) -> Self {
        let config = filter.config().clone();
        let initial_len = filter.len();
        let chunk_locks = ChunkLockTable::with_max_stripes(config.num_chunks(), max_lock_stripes);
        Self {
            epoch,
            config,
            chunk_locks,
            filter: StripedRwLock::new(filter),
            reverse_map,
            retired: CachePadded::new(AtomicBool::new(false)),
            live_keys: CachePadded::new(AtomicUsize::new(initial_len)),
        }
    }

    #[inline]
    #[must_use]
    fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    #[inline]
    fn mark_retired(&self) {
        self.retired.store(true, Ordering::Release);
    }

    #[inline]
    #[must_use]
    fn reverse_map(&self) -> Arc<dyn ReverseMapStore> {
        Arc::clone(&self.reverse_map)
    }
}

/// Diagnostic summary returned by [`RcuAerisFilter::expand_rcu`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RcuExpansionReport {
    /// Epoch before the `2x` RCU expansion.
    pub old_epoch: Epoch,
    /// Newly published epoch after the `2x` RCU swap.
    pub new_epoch: Epoch,
    /// Underlying InfiniFilter `2x` expansion telemetry.
    pub expansion: ExpansionReport,
}

/// Diagnostic summary returned by [`RcuAerisFilter::rebuild_async_and_swap`] and
/// [`RcuAerisFilter::finish_async_rebuild_and_swap`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RcuRebuildReport {
    /// Epoch before publishing the rebuilt filter `F_new`.
    pub old_epoch: Epoch,
    /// Newly published epoch after the atomic RCU swap.
    pub new_epoch: Epoch,
    /// Underlying asynchronous rebuild, delta-drain, and FP-replay telemetry.
    pub rebuild: RebuildReport,
}

/// Concurrent Scale Plane wrapper around [`AerisFilter`] providing wait-free
/// epoch-RCU reader access, striped chunk/cluster locking for writers, and
/// zero-stall atomic `F_old -> F_new` swaps for expansions and dynamic-`R` rebuilds.
pub struct RcuAerisFilter {
    shard_id: ShardId,
    current: EpochCell<RcuGeneration>,
    swap_lock: CachePadded<Mutex<()>>,
    coordinator: AsyncRebuildCoordinator,
    delta_buffer: Arc<DeltaBuffer>,
    max_lock_stripes: usize,
}

impl RcuAerisFilter {
    /// Creates a new [`RcuAerisFilter`] owning `filter` and its coupled
    /// `reverse_map` for `shard_id`, using [`DEFAULT_MAX_LOCK_STRIPES`].
    #[must_use]
    pub fn new(
        shard_id: ShardId,
        filter: AerisFilter,
        reverse_map: Arc<dyn ReverseMapStore>,
    ) -> Self {
        Self::with_max_stripes(shard_id, filter, reverse_map, DEFAULT_MAX_LOCK_STRIPES)
    }

    /// Creates a new [`RcuAerisFilter`] with an explicit cap on chunk lock stripes.
    #[must_use]
    pub fn with_max_stripes(
        shard_id: ShardId,
        filter: AerisFilter,
        reverse_map: Arc<dyn ReverseMapStore>,
        max_lock_stripes: usize,
    ) -> Self {
        let initial_epoch = crate::epoch_cell::INITIAL_EPOCH;
        let gen = RcuGeneration::new(initial_epoch, filter, reverse_map, max_lock_stripes);
        let coordinator = AsyncRebuildCoordinator::new();
        let delta_buffer = coordinator.delta_buffer();
        Self {
            shard_id,
            current: EpochCell::with_epoch(gen, initial_epoch),
            swap_lock: CachePadded::new(Mutex::new(())),
            coordinator,
            delta_buffer,
            max_lock_stripes,
        }
    }

    /// Returns the shard identifier associated with this filter instance.
    #[inline]
    #[must_use]
    pub const fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    /// Returns the most recently published RCU [`Epoch`].
    #[inline]
    #[must_use]
    pub fn epoch(&self) -> Epoch {
        self.current.epoch()
    }

    /// Returns a cloned copy of the currently active [`FilterConfig`].
    #[must_use]
    pub fn config(&self) -> FilterConfig {
        self.current.with_ref(|gen| gen.config.clone())
    }

    /// Returns a shared handle to the active [`DeltaBuffer`] mirroring live
    /// mutations whenever an asynchronous background rebuild is in flight.
    #[inline]
    #[must_use]
    pub fn delta_buffer(&self) -> Arc<DeltaBuffer> {
        Arc::clone(&self.delta_buffer)
    }

    /// Returns a shared handle to the [`ReverseMapStore`] bound to the currently
    /// published generation.
    #[must_use]
    pub fn active_reverse_map(&self) -> Arc<dyn ReverseMapStore> {
        self.current.with_ref(RcuGeneration::reverse_map)
    }

    /// Returns `true` if an asynchronous background rebuild is currently running
    /// or waiting to swap.
    #[must_use]
    pub fn is_rebuild_in_progress(&self) -> bool {
        self.coordinator.is_in_progress()
    }

    /// Returns `true` if the active background rebuild worker has completed its
    /// background scan/bulk-load/replay work and is ready for atomic swap.
    #[must_use]
    pub fn is_rebuild_ready(&self) -> bool {
        self.coordinator.is_ready()
    }

    /// Answers a point emptiness query for `key` (`true` = possibly present,
    /// `false` = definitely absent) on the zero-stall, zero-allocation RCU hot path.
    ///
    /// Readers NEVER acquire `swap_lock` or [`ChunkLockTable`] stripes and
    /// continue uninterrupted during expansions and background rebuild swaps.
    ///
    /// # Complexity
    /// `O(1)` expected time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_point(&self, key: Key) -> bool {
        let guard = crossbeam_epoch::pin();
        let slot = crate::epoch_cell::current_thread_stripe_slot();
        self.query_point_pinned_with_slot(key, &guard, slot)
    }

    /// Internal zero-double-pin point query helper reusing the caller's pinned
    /// Crossbeam [`Guard`] and pre-fetched reader stripe `slot`.
    #[inline]
    #[must_use]
    pub(crate) fn query_point_pinned_with_slot(
        &self,
        key: Key,
        guard: &Guard,
        slot: usize,
    ) -> bool {
        let gen = self.current.load(guard);
        let filter = gen.filter.read_with_slot(slot);
        filter.query_point(key)
    }

    /// Answers a closed range emptiness query `[start, end]` (`true` = possibly
    /// non-empty, `false` = definitely empty) on the zero-stall, zero-allocation
    /// RCU hot path.
    ///
    /// # Complexity
    /// `O(ceil(len / P) + 1)` expected time (`O(1)` for `len <= R`), zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_range(&self, range: QueryRange) -> bool {
        let guard = crossbeam_epoch::pin();
        let slot = crate::epoch_cell::current_thread_stripe_slot();
        self.query_range_pinned_with_slot(range, &guard, slot)
    }

    /// Internal zero-double-pin range query helper reusing the caller's pinned
    /// Crossbeam [`Guard`] and pre-fetched reader stripe `slot`.
    #[inline]
    #[must_use]
    pub(crate) fn query_range_pinned_with_slot(
        &self,
        range: QueryRange,
        guard: &Guard,
        slot: usize,
    ) -> bool {
        let gen = self.current.load(guard);
        let filter = gen.filter.read_with_slot(slot);
        filter.query_range(range)
    }

    /// Answers a point emptiness query for `key` with zero synchronization overhead
    /// using compile-time `&mut self` exclusivity (zero Crossbeam epoch pinning,
    /// zero reader-stripe locking, zero atomic operations).
    ///
    /// # Complexity
    /// `O(1)` expected time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_point_direct(&mut self, key: Key) -> bool {
        self.current.get_mut().filter.get_mut().query_point(key)
    }

    /// Answers a closed range emptiness query `[start, end]` with zero synchronization
    /// overhead using compile-time `&mut self` exclusivity (zero Crossbeam epoch
    /// pinning, zero reader-stripe locking, zero atomic operations).
    ///
    /// # Complexity
    /// `O(ceil(len / P) + 1)` expected time (`O(1)` for `len <= R`), zero heap allocations.
    #[inline]
    #[must_use]
    pub fn query_range_direct(&mut self, range: QueryRange) -> bool {
        self.current.get_mut().filter.get_mut().query_range(range)
    }

    /// Inserts `key` into the active generation with exclusive `&mut self` access,
    /// bypassing Crossbeam epoch pinning, [`ChunkLockTable`] locking, and
    /// [`StripedRwLock`] acquisition on the common non-expanding path while still
    /// persisting to the coupled [`ReverseMapStore`], mirroring into [`DeltaBuffer`],
    /// and auto-expanding via [`Self::expand_rcu`] on [`AerisError::FilterFull`].
    ///
    /// # Errors
    /// Propagates [`AerisError::InvalidConfig`] if `q` is already at
    /// `MAX_QUOTIENT_BITS`, or [`AerisError::StorageError`] on Reverse Map I/O failure.
    pub fn insert_direct(&mut self, key: Key) -> AerisResult<Fingerprint> {
        let mut expansion_attempts = 0u8;
        loop {
            let gen = self.current.get_mut();
            let filter = gen.filter.get_mut();
            match filter.insert(key) {
                Ok(fp) => {
                    *gen.live_keys.get_mut() = filter.len();
                    if let Err(err) = gen.reverse_map.append_key(&fp, key) {
                        let _ = filter.delete(key);
                        *gen.live_keys.get_mut() = filter.len();
                        return Err(err);
                    }
                    let _ = self.delta_buffer.record_insert(key);
                    return Ok(fp);
                }
                Err(AerisError::FilterFull { .. }) => {
                    if expansion_attempts >= MAX_QUOTIENT_BITS {
                        return Err(AerisError::InvalidConfig(
                            "exceeded maximum RCU auto-expansion attempts during insert_direct"
                                .to_string(),
                        ));
                    }
                    expansion_attempts += 1;
                    let _ = self.expand_rcu()?;
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Locates all canonical keepsake box [`Fingerprint`]s matching `range` in
    /// the currently published generation.
    #[must_use]
    pub fn locate_matching_fingerprints(&self, range: QueryRange) -> Vec<Fingerprint> {
        self.current
            .with_ref(|gen| gen.filter.read().locate_matching_fingerprints(range))
    }

    /// Returns the number of live keys currently represented by the active generation.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.current
            .with_ref(|gen| gen.live_keys.load(Ordering::Acquire))
    }

    /// Returns `true` if the active generation represents zero live keys.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the canonical slot capacity `N = 2^q` of the active generation.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.current.with_ref(|gen| gen.config.num_slots())
    }

    /// Returns the number of occupied physical Quotient Filter slots in the active generation.
    #[must_use]
    pub fn occupied_slots(&self) -> usize {
        self.current
            .with_ref(|gen| gen.filter.read().occupied_slots())
    }

    /// Returns the slot occupancy load factor `occupied_slots / N` of the active generation.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        self.current.with_ref(|gen| gen.filter.read().load_factor())
    }

    /// Returns `(live_keys, q, load_factor)` under a single epoch pin for shard
    /// rebalance evaluation.
    #[inline]
    pub(crate) fn split_check_metrics(&self) -> (usize, u8, f64) {
        self.current.with_ref(|gen| {
            let live = gen.live_keys.load(Ordering::Acquire);
            let q = gen.config.q;
            let load = gen.filter.read().load_factor();
            (live, q, load)
        })
    }

    /// Returns the heap RAM footprint in bytes of the active generation's [`AerisFilter`].
    #[must_use]
    pub fn memory_usage_bytes(&self) -> usize {
        self.current
            .with_ref(|gen| gen.filter.read().memory_usage_bytes())
    }

    /// Returns the cumulative number of `2x` expansion cycles completed on the active lineage.
    #[must_use]
    pub fn expansion_cycles(&self) -> u64 {
        self.current
            .with_ref(|gen| gen.filter.read().expansion_cycles())
    }

    /// Verifies all structural Quotient Filter and keepsake box invariants on
    /// the currently published generation.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if any invariant is violated.
    pub fn verify_invariants(&self) -> AerisResult<()> {
        self.current
            .with_ref(|gen| gen.filter.read().verify_invariants())
    }

    /// Inserts `key` into the active generation under fine-grained cluster-striped
    /// locking, persisting `(fp, key)` to the generation's [`ReverseMapStore`] and
    /// mirroring the insert into [`DeltaBuffer`] if an asynchronous rebuild is active.
    ///
    /// Releases the exclusive RAM filter lock immediately after the in-memory
    /// Quotient Filter mutation completes, BEFORE RocksDB Reverse Map I/O, while
    /// retaining [`ChunkLockTable`] cluster stripes so overlapping writers,
    /// adaptations, expansions, and swaps remain strictly serialized and
    /// concurrent readers never stall on disk/memtable writes.
    ///
    /// Automatically triggers a zero-stall RCU `2x` expansion ([`Self::expand_rcu`])
    /// and retries whenever [`AerisError::FilterFull`] is encountered.
    ///
    /// # Errors
    /// Propagates [`AerisError::InvalidConfig`] if `q` is already at
    /// `MAX_QUOTIENT_BITS`, or [`AerisError::StorageError`] on Reverse Map I/O failure.
    pub fn insert(&self, key: Key) -> AerisResult<Fingerprint> {
        let mut expansion_attempts = 0u8;
        loop {
            let full_epoch = {
                let guard = crossbeam_epoch::pin();
                match self.try_insert_once(key, &guard)? {
                    InsertStep::Done(fp) => return Ok(fp),
                    InsertStep::Retired => continue,
                    InsertStep::Full(ep) => ep,
                }
            };

            if expansion_attempts >= MAX_QUOTIENT_BITS {
                return Err(AerisError::InvalidConfig(
                    "exceeded maximum RCU auto-expansion attempts during insert".to_string(),
                ));
            }
            expansion_attempts += 1;
            let _ = self.expand_rcu_if_epoch(full_epoch)?;
        }
    }

    /// Internal single-pin insert helper used by [`crate::ShardedAerisSystem`]
    /// to avoid double-pinning on the common non-expanding write path.
    pub(crate) fn insert_pinned(&self, key: Key, guard: &Guard) -> AerisResult<Fingerprint> {
        let mut expansion_attempts = 0u8;
        loop {
            let full_epoch = match self.try_insert_once(key, guard)? {
                InsertStep::Done(fp) => return Ok(fp),
                InsertStep::Retired => continue,
                InsertStep::Full(ep) => ep,
            };

            if expansion_attempts >= MAX_QUOTIENT_BITS {
                return Err(AerisError::InvalidConfig(
                    "exceeded maximum RCU auto-expansion attempts during insert".to_string(),
                ));
            }
            expansion_attempts += 1;
            let _ = self.expand_rcu_if_epoch(full_epoch)?;
        }
    }

    /// Executes a single attempt to insert `key` under `guard`.
    #[inline]
    fn try_insert_once(&self, key: Key, guard: &Guard) -> AerisResult<InsertStep> {
        let gen = self.current.load(guard);
        let (start_chunk, end_chunk) = key_cluster_chunk_bounds(&gen.config, key);
        let _chunk_guard = gen.chunk_locks.write_cluster_range(start_chunk, end_chunk);
        if gen.is_retired() {
            return Ok(InsertStep::Retired);
        }

        // Hold `gen.filter.write()` strictly for the in-memory filter update,
        // releasing it BEFORE RocksDB `append_key` I/O so readers never stall
        // while `_chunk_guard` continues serializing overlapping writers & swaps.
        let insert_res = {
            let mut filter = gen.filter.write();
            match filter.insert(key) {
                Ok(fp) => {
                    gen.live_keys.store(filter.len(), Ordering::Release);
                    Ok(fp)
                }
                Err(err) => Err(err),
            }
        };

        match insert_res {
            Ok(fp) => {
                if let Err(err) = gen.reverse_map.append_key(&fp, key) {
                    // Cold rollback path: `_chunk_guard` (Level 4) is still held;
                    // re-acquiring `gen.filter.write()` (Level 5) obeys lock order.
                    let mut filter = gen.filter.write();
                    let _ = filter.delete(key);
                    gen.live_keys.store(filter.len(), Ordering::Release);
                    return Err(err);
                }
                let _ = self.delta_buffer.record_insert(key);
                Ok(InsertStep::Done(fp))
            }
            Err(AerisError::FilterFull { .. }) => Ok(InsertStep::Full(gen.epoch)),
            Err(err) => Err(err),
        }
    }

    /// Deletes `key` from the active generation under fine-grained cluster-striped
    /// locking, updating the generation's [`ReverseMapStore`] and mirroring the
    /// deletion into [`DeltaBuffer`] if an asynchronous rebuild is active.
    ///
    /// Releases `gen.filter.write()` immediately after the in-memory deletion
    /// completes (before RocksDB I/O) while keeping `_chunk_guard` held through
    /// persistence and delta mirroring.
    ///
    /// Returns `Ok(Some(fp))` if a matching entry was removed, or `Ok(None)` if absent.
    ///
    /// # Errors
    /// Propagates [`AerisError::StorageError`] on Reverse Map I/O failure.
    pub fn delete(&self, key: Key) -> AerisResult<Option<Fingerprint>> {
        let guard = crossbeam_epoch::pin();
        self.delete_pinned(key, &guard)
    }

    /// Internal single-pin delete helper used by [`crate::ShardedAerisSystem`].
    pub(crate) fn delete_pinned(
        &self,
        key: Key,
        guard: &Guard,
    ) -> AerisResult<Option<Fingerprint>> {
        loop {
            let gen = self.current.load(guard);
            let (start_chunk, end_chunk) = key_cluster_chunk_bounds(&gen.config, key);
            let _chunk_guard = gen.chunk_locks.write_cluster_range(start_chunk, end_chunk);
            if gen.is_retired() {
                continue;
            }

            let removed = {
                let mut filter = gen.filter.write();
                let res = filter.delete(key)?;
                if res.is_some() {
                    gen.live_keys.store(filter.len(), Ordering::Release);
                }
                res
            };

            if let Some(ref fp) = removed {
                if let Err(err) = gen.reverse_map.remove_key(fp, key) {
                    // Cold rollback path under still-held `_chunk_guard` (Level 4 -> Level 5).
                    let mut filter = gen.filter.write();
                    let _ = filter.insert(key);
                    gen.live_keys.store(filter.len(), Ordering::Release);
                    return Err(err);
                }
                let _ = self.delta_buffer.record_delete(key);
            }
            return Ok(removed);
        }
    }

    /// Monotonically adapts the active generation to a verified false-positive
    /// closed interval `range`, locking all affected chunk stripes in strict
    /// ascending order and auto-expanding via RCU swap if [`AerisError::FilterFull`]
    /// occurs.
    ///
    /// # Errors
    /// Propagates [`AerisError::InvalidConfig`] if expansion exceeds `MAX_QUOTIENT_BITS`,
    /// or [`AerisError::StorageError`] on Reverse Map I/O failure.
    pub fn adapt(&self, range: QueryRange) -> AerisResult<AdaptationReport> {
        let mut expansion_attempts = 0u8;
        loop {
            let full_epoch = {
                let guard = crossbeam_epoch::pin();
                let gen = self.current.load(&guard);
                let _chunk_guard = lock_range_chunks(gen, range);
                if gen.is_retired() {
                    continue;
                }

                let mut filter = gen.filter.write();
                match filter.adapt(range, gen.reverse_map.as_ref()) {
                    Ok(report) => return Ok(report),
                    Err(AerisError::FilterFull { .. }) => gen.epoch,
                    Err(err) => return Err(err),
                }
            };

            if expansion_attempts >= MAX_QUOTIENT_BITS {
                return Err(AerisError::InvalidConfig(
                    "exceeded maximum RCU auto-expansion attempts during adapt".to_string(),
                ));
            }
            expansion_attempts += 1;
            let _ = self.expand_rcu_if_epoch(full_epoch)?;
        }
    }

    /// Atomically publishes a prebuilt replacement `(new_filter, new_reverse_map)`
    /// via Crossbeam epoch RCU (`F_old -> F_new`), draining any active [`DeltaBuffer`]
    /// mutations onto `new_filter` under `swap_lock` + retiring-generation chunk
    /// quiescence without EVER stalling readers on `F_old`.
    ///
    /// Returns the newly published monotonic [`Epoch`].
    ///
    /// # Errors
    /// Propagates filter expansion or storage errors if draining buffered deltas fails.
    pub fn swap_filter(
        &self,
        mut new_filter: AerisFilter,
        new_reverse_map: Arc<dyn ReverseMapStore>,
    ) -> AerisResult<Epoch> {
        let start = Instant::now();
        let _swap_guard = self.swap_lock.lock();
        let guard = crossbeam_epoch::pin();
        let gen_old = self.current.load(&guard);
        let _all_chunks = gen_old.chunk_locks.write_all();

        if self.delta_buffer.is_active() {
            let deltas = self.delta_buffer.drain_and_deactivate();
            apply_delta_mutations(&mut new_filter, new_reverse_map.as_ref(), &deltas)?;
        }

        let old_epoch = gen_old.epoch;
        let next_epoch = old_epoch + 1;
        let alpha = new_filter.load_factor();
        let new_cfg = new_filter.config().clone();
        let new_gen = RcuGeneration::new(
            next_epoch,
            new_filter,
            new_reverse_map,
            self.max_lock_stripes,
        );

        gen_old.mark_retired();
        let published_epoch = self.current.swap(new_gen);
        debug_assert_eq!(published_epoch, next_epoch);

        let latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            shard_id = self.shard_id,
            old_epoch,
            new_epoch = published_epoch,
            alpha,
            q = new_cfg.q,
            r = new_cfg.r,
            m = new_cfg.m,
            latency_ns,
            "completed atomic RCU filter swap"
        );
        Ok(published_epoch)
    }

    /// Performs an in-memory `2x` InfiniFilter expansion (`q -> q + 1`) with
    /// selective rejuvenation completely off `F_old`'s reader lock, then
    /// atomically swaps `F_old -> F_new` via [`EpochCell::swap`].
    ///
    /// Readers executing `query_point` / `query_range` on `F_old` experience
    /// **zero stalls** throughout the expansion and swap.
    ///
    /// # Errors
    /// Propagates [`AerisError::InvalidConfig`] if `q == MAX_QUOTIENT_BITS` or
    /// [`AerisError::StorageError`] if selective rejuvenation fails against the Reverse Map.
    pub fn expand_rcu(&self) -> AerisResult<RcuExpansionReport> {
        let _swap_guard = self.swap_lock.lock();
        self.expand_rcu_locked()
    }

    /// Expands via RCU swap only if the currently published epoch still equals
    /// `expected_epoch`, avoiding redundant double expansions when multiple
    /// concurrent writers observe `FilterFull` on the same generation.
    fn expand_rcu_if_epoch(
        &self,
        expected_epoch: Epoch,
    ) -> AerisResult<Option<RcuExpansionReport>> {
        let _swap_guard = self.swap_lock.lock();
        if self.current.epoch() != expected_epoch {
            debug!(
                shard_id = self.shard_id,
                expected_epoch,
                current_epoch = self.current.epoch(),
                "skipping redundant RCU expansion: generation already advanced"
            );
            return Ok(None);
        }
        self.expand_rcu_locked().map(Some)
    }

    /// Executes `expand_rcu` while the caller holds `self.swap_lock`.
    fn expand_rcu_locked(&self) -> AerisResult<RcuExpansionReport> {
        let start = Instant::now();
        let guard = crossbeam_epoch::pin();
        let gen_old = self.current.load(&guard);

        // Quiesce writers on `gen_old` without acquiring `gen_old.filter.write()`.
        // Readers continue taking private-stripe `gen_old.filter.read()` with zero stalls!
        let _all_chunks = gen_old.chunk_locks.write_all();
        let mut f_new = {
            let reader = gen_old.filter.read();
            reader.clone()
        };

        let reverse_map = gen_old.reverse_map();
        let expansion = f_new.expand(reverse_map.as_ref())?;

        let old_epoch = gen_old.epoch;
        let next_epoch = old_epoch + 1;
        let alpha = f_new.load_factor();
        let new_gen = RcuGeneration::new(next_epoch, f_new, reverse_map, self.max_lock_stripes);

        gen_old.mark_retired();
        let published_epoch = self.current.swap(new_gen);
        debug_assert_eq!(published_epoch, next_epoch);

        let latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            shard_id = self.shard_id,
            old_epoch,
            new_epoch = published_epoch,
            old_q = expansion.old_q,
            new_q = expansion.new_q,
            alpha,
            boxes_transferred = expansion.boxes_transferred,
            boxes_rejuvenated = expansion.boxes_rejuvenated,
            latency_ns,
            "completed zero-stall RCU 2x expansion swap"
        );

        Ok(RcuExpansionReport {
            old_epoch,
            new_epoch: published_epoch,
            expansion,
        })
    }

    /// Launches a non-blocking background dynamic-`R` / capacity rebuild via
    /// [`AsyncRebuildCoordinator`], activating [`DeltaBuffer`] atomically with
    /// respect to writers on `F_old` while keeping readers completely unblocked.
    ///
    /// Foreground readers AND writers continue executing on `F_old` throughout
    /// the background thread's I/O-throttled scan, bulk-load, and initial FP replay.
    ///
    /// # Errors
    /// Propagates [`AerisError::RebuildInProgress`] or [`AerisError::InvalidConfig`].
    pub fn start_async_rebuild(
        &self,
        dest_reverse_map: Arc<dyn ReverseMapStore>,
        fp_log: Arc<FalsePositiveLog>,
        target_config: FilterConfig,
        reason: RebuildReason,
        throttle_config: IoThrottleConfig,
    ) -> AerisResult<()> {
        let _swap_guard = self.swap_lock.lock();
        let guard = crossbeam_epoch::pin();
        let gen_old = self.current.load(&guard);
        // Briefly quiesce writers so every prior write is visible in `source_rm`
        // and every subsequent write sees `delta_buffer` active. Readers NEVER stall.
        let _all_chunks = gen_old.chunk_locks.write_all();
        let source_rm = gen_old.reverse_map();

        self.coordinator.start_rebuild(
            source_rm,
            dest_reverse_map,
            fp_log,
            target_config,
            reason,
            throttle_config,
        )
    }

    /// Completes an active background rebuild started by [`Self::start_async_rebuild`],
    /// incrementally pre-draining [`DeltaBuffer`] mutations off-lock until the residual
    /// tail is small, then acquiring `swap_lock` + chunk quiescence strictly for the
    /// tiny tail drain, catch-up [`FalsePositiveLog`] replay, and atomic `F_old -> F_new`
    /// RCU swap.
    ///
    /// Waits for background worker completion and drains the bulk delta backlog off all
    /// locks so foreground readers and writers stay unblocked until the brief final swap.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if no rebuild is active, or
    /// propagates worker, delta-drain, or FP-replay errors.
    pub fn finish_async_rebuild_and_swap<F>(&self, verify_empty: F) -> AerisResult<RcuRebuildReport>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        if !self.coordinator.is_in_progress() {
            return Err(AerisError::InternalInvariant(
                "finish_async_rebuild_and_swap called with no active rebuild".to_string(),
            ));
        }
        while self.coordinator.is_in_progress() && !self.coordinator.is_ready() {
            thread::yield_now();
        }

        // Phase 1 (off-lock): join background worker, incrementally drain DeltaBuffer
        // in bounded batches until remaining tail <= threshold, and verify pre-drained
        // keys while foreground readers AND writers keep running on `F_old`.
        let mut staged = self.coordinator.prepare_finish()?;
        let _ = staged.pre_drain_until_small(
            DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
            DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
            DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
        )?;
        staged.verify_pre_drained_keys()?;

        // Phase 2 (locked): acquire `swap_lock` + chunk quiescence strictly for the tiny
        // residual tail drain, catch-up FP replay, and atomic `EpochCell::swap`.
        let swap_start = Instant::now();
        let _swap_guard = self.swap_lock.lock();
        let guard = crossbeam_epoch::pin();
        let gen_old = self.current.load(&guard);
        let _all_chunks = gen_old.chunk_locks.write_all();

        let completed = staged.finish_tail(verify_empty)?;
        self.publish_completed_rebuild(gen_old, completed, swap_start)
    }

    /// Completes an active background rebuild using two-phase incremental delta
    /// draining, deriving exact range emptiness directly from the reconciled live
    /// key set at the swap boundary (requiring no external KV store handle or locked
    /// Reverse Map scan), and atomically swaps `F_old -> F_new`.
    ///
    /// # Errors
    /// Propagates rebuild, delta-drain, or FP-replay errors.
    pub fn finish_async_rebuild_and_swap_from_reverse_map(&self) -> AerisResult<RcuRebuildReport> {
        if !self.coordinator.is_in_progress() {
            return Err(AerisError::InternalInvariant(
                "finish_async_rebuild_and_swap_from_reverse_map called with no active rebuild"
                    .to_string(),
            ));
        }
        while self.coordinator.is_in_progress() && !self.coordinator.is_ready() {
            thread::yield_now();
        }

        // Phase 1 (off-lock): join worker and pre-drain buffered deltas in bounded batches.
        let mut staged = self.coordinator.prepare_finish()?;
        let _ = staged.pre_drain_until_small(
            DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
            DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
            DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
        )?;
        staged.verify_pre_drained_keys()?;

        // Phase 2 (locked): quiesce writers, drain residual tail, replay catch-up FPs
        // against the exact reconciled live key set, and publish `F_new` via RCU.
        let swap_start = Instant::now();
        let _swap_guard = self.swap_lock.lock();
        let guard = crossbeam_epoch::pin();
        let gen_old = self.current.load(&guard);
        let _all_chunks = gen_old.chunk_locks.write_all();

        let completed = staged.finish_tail_with_live_keys()?;
        self.publish_completed_rebuild(gen_old, completed, swap_start)
    }

    /// Runs a full asynchronous background rebuild (`F_old -> F_new`) via
    /// [`AsyncRebuildCoordinator`] with [`FalsePositiveLog`] replay and atomic
    /// delta draining, publishing `F_new` via Crossbeam epoch RCU while keeping
    /// concurrent readers and writers completely unblocked during the rebuild.
    ///
    /// # Errors
    /// Propagates configuration, rebuild, storage, or FP-replay errors.
    pub fn rebuild_async_and_swap(
        &self,
        dest_reverse_map: Arc<dyn ReverseMapStore>,
        fp_log: Arc<FalsePositiveLog>,
        target_config: FilterConfig,
        reason: RebuildReason,
        throttle_config: IoThrottleConfig,
    ) -> AerisResult<RcuRebuildReport> {
        self.start_async_rebuild(
            dest_reverse_map,
            fp_log,
            target_config,
            reason,
            throttle_config,
        )?;
        self.finish_async_rebuild_and_swap_from_reverse_map()
    }

    /// Variant of [`Self::rebuild_async_and_swap`] accepting an explicit
    /// caller-supplied `verify_empty` oracle (for example backed by `ReferenceKvStore`).
    ///
    /// # Errors
    /// Propagates configuration, rebuild, storage, or FP-replay errors.
    pub fn rebuild_async_and_swap_with_verifier<F>(
        &self,
        dest_reverse_map: Arc<dyn ReverseMapStore>,
        fp_log: Arc<FalsePositiveLog>,
        target_config: FilterConfig,
        reason: RebuildReason,
        throttle_config: IoThrottleConfig,
        verify_empty: F,
    ) -> AerisResult<RcuRebuildReport>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        self.start_async_rebuild(
            dest_reverse_map,
            fp_log,
            target_config,
            reason,
            throttle_config,
        )?;
        self.finish_async_rebuild_and_swap(verify_empty)
    }

    /// Publishes a [`aeris_control::CompletedRebuild`] atomically while the caller
    /// holds `swap_lock` and `gen_old.chunk_locks.write_all()`.
    fn publish_completed_rebuild(
        &self,
        gen_old: &RcuGeneration,
        completed: aeris_control::CompletedRebuild,
        swap_start: Instant,
    ) -> AerisResult<RcuRebuildReport> {
        let old_epoch = gen_old.epoch;
        let next_epoch = old_epoch + 1;
        let alpha = completed.filter.load_factor();
        let new_cfg = completed.filter.config().clone();
        let new_gen = RcuGeneration::new(
            next_epoch,
            completed.filter,
            completed.reverse_map,
            self.max_lock_stripes,
        );

        gen_old.mark_retired();
        let published_epoch = self.current.swap(new_gen);
        debug_assert_eq!(published_epoch, next_epoch);

        let swap_latency_ns = swap_start.elapsed().as_nanos() as u64;
        info!(
            shard_id = self.shard_id,
            old_epoch,
            new_epoch = published_epoch,
            alpha,
            q = new_cfg.q,
            r = new_cfg.r,
            m = new_cfg.m,
            keys_loaded = completed.report.keys_loaded,
            delta_inserts = completed.report.delta_inserts_applied,
            delta_deletes = completed.report.delta_deletes_applied,
            latency_ns = completed.report.duration_ns,
            swap_latency_ns,
            "completed zero-stall RCU async rebuild swap"
        );

        Ok(RcuRebuildReport {
            old_epoch,
            new_epoch: published_epoch,
            rebuild: completed.report,
        })
    }
}

/// Outcome of a single insert attempt against the currently loaded [`RcuGeneration`].
enum InsertStep {
    Done(Fingerprint),
    Retired,
    Full(Epoch),
}

/// Computes the inclusive `[home_chunk, home_chunk + 1]` cluster chunk bounds
/// for a single `key` under `config`, covering potential Robin Hood displacement
/// across the 64-slot chunk boundary.
#[inline]
fn key_cluster_chunk_bounds(config: &FilterConfig, key: Key) -> (usize, usize) {
    let hasher = PartitionHasher::new(config.hash_seed);
    let partition = config.partition_of(key);
    let (home_u64, _) = hasher.quotient_and_remainder(partition, config.q, config.r);
    let home_chunk = (home_u64 as usize) >> SLOTS_PER_CHUNK_LOG2;
    (home_chunk, home_chunk.saturating_add(1))
}

/// Acquires ascending write locks covering every chunk touched by the partition
/// subqueries of `range` (plus each home chunk's immediate rightward neighbor).
///
/// Uses a zero-allocation stack buffer for ranges touching `<= 2` partitions
/// (`<= 4` chunk locks, covering 99%+ of `adapt` calls).
fn lock_range_chunks(gen: &RcuGeneration, range: QueryRange) -> ClusterWriteGuard<'_> {
    let num_stripes = gen.chunk_locks.num_stripes();
    let max_subqueries = num_stripes;
    let hasher = PartitionHasher::new(gen.config.hash_seed);

    let mut inline_chunks = [0usize; 4];
    let mut inline_len = 0usize;
    let mut heap_chunks: Option<Vec<usize>> = None;

    for (idx, (partition, _, _)) in range.partition_subqueries(gen.config.m).enumerate() {
        if idx >= max_subqueries {
            return gen.chunk_locks.write_all();
        }
        let (home_u64, _) = hasher.quotient_and_remainder(partition, gen.config.q, gen.config.r);
        let home_chunk = (home_u64 as usize) >> SLOTS_PER_CHUNK_LOG2;
        let next_chunk = home_chunk.saturating_add(1);

        if let Some(ref mut vec) = heap_chunks {
            vec.push(home_chunk);
            vec.push(next_chunk);
        } else if inline_len + 2 <= inline_chunks.len() {
            inline_chunks[inline_len] = home_chunk;
            inline_chunks[inline_len + 1] = next_chunk;
            inline_len += 2;
        } else {
            let mut vec = Vec::with_capacity(8);
            vec.extend_from_slice(&inline_chunks[..inline_len]);
            vec.push(home_chunk);
            vec.push(next_chunk);
            heap_chunks = Some(vec);
        }
    }

    match heap_chunks {
        Some(ref vec) => gen.chunk_locks.write_chunks(vec),
        None => gen.chunk_locks.write_chunks(&inline_chunks[..inline_len]),
    }
}

/// Applies drained FIFO [`DeltaMutation`]s onto `filter` and `reverse_map`,
/// auto-expanding `filter` whenever [`AerisError::FilterFull`] is encountered.
fn apply_delta_mutations(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    deltas: &[DeltaMutation],
) -> AerisResult<()> {
    for &mutation in deltas {
        match mutation {
            DeltaMutation::Insert(key) => {
                let mut expansions = 0u8;
                let fp = loop {
                    match filter.insert(key) {
                        Ok(fp) => break fp,
                        Err(AerisError::FilterFull { .. }) if expansions < MAX_QUOTIENT_BITS => {
                            filter.expand(reverse_map)?;
                            expansions += 1;
                        }
                        Err(err) => return Err(err),
                    }
                };
                reverse_map.append_key(&fp, key)?;
            }
            DeltaMutation::Delete(key) => {
                if let Some(fp) = filter.delete(key)? {
                    reverse_map.remove_key(&fp, key)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeris_control::FpLogConfig;
    use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
    use tempfile::TempDir;

    fn test_config(q: u8, m: u8) -> FilterConfig {
        let max_range_r = 1u64 << m;
        let payload_budget_c = 20;
        let r = payload_budget_c - m;
        FilterConfig::new(q, r, max_range_r, payload_budget_c, 1.0 / 256.0).unwrap()
    }

    #[test]
    fn rcu_insert_query_delete_and_auto_expand() {
        let dir = TempDir::new().unwrap();
        let rm = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm"), ReverseMapConfig::default()).unwrap(),
        );
        let cfg = test_config(7, 4); // 128 slots -> triggers auto-expand quickly
        let filter = AerisFilter::new(cfg).unwrap();
        let rcu = RcuAerisFilter::new(0, filter, rm);

        let initial_epoch = rcu.epoch();
        for k in 0..300u64 {
            let key = k * 17;
            rcu.insert(key).unwrap();
            assert!(rcu.query_point(key));
        }

        assert!(
            rcu.epoch() > initial_epoch,
            "auto-expansion must advance RCU epoch"
        );
        assert!(rcu.capacity() > 128);
        assert_eq!(rcu.len(), 300);
        rcu.verify_invariants().unwrap();

        for k in 0..300u64 {
            assert!(rcu.query_point(k * 17));
        }
        for k in 0..100u64 {
            assert!(rcu.delete(k * 17).unwrap().is_some());
        }
        assert_eq!(rcu.len(), 200);
        for k in 100..300u64 {
            assert!(rcu.query_point(k * 17));
        }
        rcu.verify_invariants().unwrap();
    }

    #[test]
    fn rcu_async_rebuild_changes_m_and_preserves_monotonicity() {
        let dir = TempDir::new().unwrap();
        let rm0 = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_0"), ReverseMapConfig::default()).unwrap(),
        );
        let rm1 = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_1"), ReverseMapConfig::default()).unwrap(),
        );
        let fp_log = Arc::new(
            FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
        );

        let cfg_old = test_config(9, 4); // m = 4 (R = 16)
        let filter = AerisFilter::new(cfg_old.clone()).unwrap();
        let rcu = RcuAerisFilter::new(7, filter, rm0);

        let mut inserted = Vec::new();
        for i in 0..200u64 {
            let k = i * 1000;
            rcu.insert(k).unwrap();
            inserted.push(k);
        }

        // Find real empty-range false positives and adapt before rebuild.
        let mut adapted_ranges = Vec::new();
        for probe in 0..20_000u64 {
            let start = probe * 1000 + 100;
            let r = QueryRange::new(start, start + 15).unwrap();
            if rcu.query_range(r) {
                fp_log.record(r, 1).unwrap();
                rcu.adapt(r).unwrap();
                assert!(!rcu.query_range(r));
                adapted_ranges.push(r);
                if adapted_ranges.len() >= 10 {
                    break;
                }
            }
        }

        let cfg_new = cfg_old.reconfigure_for_m(6).unwrap(); // m = 6 (R = 64)
        let report = rcu
            .rebuild_async_and_swap(
                rm1,
                fp_log,
                cfg_new.clone(),
                RebuildReason::DynamicRangeChange { old_m: 4, new_m: 6 },
                IoThrottleConfig::default(),
            )
            .unwrap();

        assert!(report.new_epoch > report.old_epoch);
        assert_eq!(rcu.config().m, 6);
        assert_eq!(rcu.config().r, 14);
        rcu.verify_invariants().unwrap();

        // Zero false negatives across all inserted keys.
        for k in inserted {
            assert!(rcu.query_point(k), "false negative for key {k}");
        }
        // 100% monotonic adaptivity across dynamic-R RCU swap.
        for r in adapted_ranges {
            assert!(
                !rcu.query_range(r),
                "previously adapted empty range {r:?} regressed to FP after RCU rebuild"
            );
        }
    }

    #[test]
    fn rcu_swap_filter_drains_delta_buffer_and_gates_concurrent_rebuild() {
        let dir = TempDir::new().unwrap();
        let rm0 = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm0"), ReverseMapConfig::default()).unwrap(),
        );
        let rm1 = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm1"), ReverseMapConfig::default()).unwrap(),
        );
        let rm2 = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm2"), ReverseMapConfig::default()).unwrap(),
        );
        let fp_log = Arc::new(
            FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
        );

        let cfg = test_config(8, 4);
        let rcu = RcuAerisFilter::new(3, AerisFilter::new(cfg.clone()).unwrap(), rm0);
        for k in 0..40u64 {
            rcu.insert(k * 10).unwrap();
        }

        // Manual swap_filter drains active DeltaBuffer inserts and deletes atomically.
        rcu.delta_buffer().activate();
        let offline_keys: Vec<Key> = (0..40u64).map(|k| k * 10).collect();
        let f_new = AerisFilter::bulk_load(cfg.clone(), &offline_keys, rm1.as_ref()).unwrap();

        // Concurrent mutations arriving after offline snapshot are captured in DeltaBuffer.
        rcu.insert(9999).unwrap();
        rcu.insert(8888).unwrap();
        assert!(rcu.delete(10).unwrap().is_some());

        let new_ep = rcu
            .swap_filter(f_new, Arc::clone(&rm1) as Arc<dyn ReverseMapStore>)
            .unwrap();
        assert_eq!(new_ep, 2);
        assert!(!rcu.delta_buffer().is_active());
        assert!(rcu.query_point(9999));
        assert!(rcu.query_point(8888));
        assert!(rcu.query_point(0));
        rcu.verify_invariants().unwrap();

        // Two-phase async rebuild rejects overlapping start_async_rebuild with RebuildInProgress.
        let cfg_m5 = cfg.reconfigure_for_m(5).unwrap();
        rcu.start_async_rebuild(
            Arc::clone(&rm2) as Arc<dyn ReverseMapStore>,
            Arc::clone(&fp_log),
            cfg_m5.clone(),
            RebuildReason::DynamicRangeChange { old_m: 4, new_m: 5 },
            IoThrottleConfig::default(),
        )
        .unwrap();
        assert!(rcu.is_rebuild_in_progress());

        let dup = rcu.start_async_rebuild(
            rm2,
            fp_log,
            cfg_m5,
            RebuildReason::DynamicRangeChange { old_m: 4, new_m: 5 },
            IoThrottleConfig::default(),
        );
        assert!(matches!(dup, Err(AerisError::RebuildInProgress)));

        // Foreground writes proceed cleanly while rebuild is active and survive swap.
        rcu.insert(77_777).unwrap();
        let rep = rcu
            .finish_async_rebuild_and_swap_from_reverse_map()
            .unwrap();
        assert_eq!(rep.new_epoch, 3);
        assert!(!rcu.is_rebuild_in_progress());
        assert!(rcu.query_point(77_777));
        assert!(rcu.query_point(9999));
        assert_eq!(rcu.config().m, 5);
        rcu.verify_invariants().unwrap();
    }

    #[test]
    fn rcu_insert_releases_filter_lock_before_reverse_map_io_and_rolls_back_on_error() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Barrier;

        struct ControlledReverseMap {
            inner: RocksDbReverseMap,
            fail_append: AtomicBool,
            fail_remove: AtomicBool,
            io_entered: Option<Arc<Barrier>>,
            io_release: Option<Arc<Barrier>>,
        }

        impl ReverseMapStore for ControlledReverseMap {
            fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
                if let Some(ref b) = self.io_entered {
                    b.wait();
                }
                if let Some(ref b) = self.io_release {
                    b.wait();
                }
                if self.fail_append.load(Ordering::Acquire) {
                    return Err(AerisError::StorageError("injected append failure".into()));
                }
                self.inner.append_key(fp, key)
            }
            fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
                self.inner.append_batch(entries)
            }
            fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
                if self.fail_remove.load(Ordering::Acquire) {
                    return Err(AerisError::StorageError("injected remove failure".into()));
                }
                self.inner.remove_key(fp, key)
            }
            fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
                self.inner.get_keys(fp)
            }
            fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
                self.inner.tombstone(fp)
            }
            fn replace_fingerprint(
                &self,
                old_fp: &Fingerprint,
                new_entries: &[(Fingerprint, Key)],
            ) -> AerisResult<()> {
                self.inner.replace_fingerprint(old_fp, new_entries)
            }
            fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
                self.inner.scan_all_keys()
            }
            fn iter_all_keys(
                &self,
            ) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
                self.inner.iter_all_keys()
            }
            fn flush(&self) -> AerisResult<()> {
                self.inner.flush()
            }
            fn compact(&self) -> AerisResult<()> {
                self.inner.compact()
            }
            fn entry_count_estimate(&self) -> u64 {
                self.inner.entry_count_estimate()
            }
        }

        // Part 1: Verify rollback restores filter state and `len()` on storage errors.
        let dir = TempDir::new().unwrap();
        let inner = RocksDbReverseMap::open(dir.path().join("rm_err"), ReverseMapConfig::default())
            .unwrap();
        let rm = Arc::new(ControlledReverseMap {
            inner,
            fail_append: AtomicBool::new(false),
            fail_remove: AtomicBool::new(false),
            io_entered: None,
            io_release: None,
        });
        let rcu = RcuAerisFilter::new(
            1,
            AerisFilter::new(test_config(8, 4)).unwrap(),
            Arc::clone(&rm) as Arc<dyn ReverseMapStore>,
        );

        rcu.insert(100).unwrap();
        assert_eq!(rcu.len(), 1);

        rm.fail_append.store(true, Ordering::Release);
        assert!(matches!(rcu.insert(200), Err(AerisError::StorageError(_))));
        assert_eq!(rcu.len(), 1);
        assert!(!rcu.query_point(200));
        rcu.verify_invariants().unwrap();

        rm.fail_append.store(false, Ordering::Release);
        rm.fail_remove.store(true, Ordering::Release);
        assert!(matches!(rcu.delete(100), Err(AerisError::StorageError(_))));
        assert_eq!(rcu.len(), 1);
        assert!(rcu.query_point(100));
        rcu.verify_invariants().unwrap();

        // Part 2: Prove concurrent readers complete while a writer is parked inside
        // `reverse_map.append_key` (would deadlock if `gen.filter.write()` were still held!).
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let inner2 =
            RocksDbReverseMap::open(dir.path().join("rm_block"), ReverseMapConfig::default())
                .unwrap();
        let rm2 = Arc::new(ControlledReverseMap {
            inner: inner2,
            fail_append: AtomicBool::new(false),
            fail_remove: AtomicBool::new(false),
            io_entered: Some(Arc::clone(&entered)),
            io_release: Some(Arc::clone(&release)),
        });
        let rcu2 = Arc::new(RcuAerisFilter::new(
            2,
            AerisFilter::new(test_config(8, 4)).unwrap(),
            rm2 as Arc<dyn ReverseMapStore>,
        ));

        let writer_rcu = Arc::clone(&rcu2);
        let writer = thread::spawn(move || {
            writer_rcu.insert(42_000).unwrap();
        });

        // Wait until writer is inside `append_key` (after `filter.insert`!).
        entered.wait();
        // If `gen.filter.write()` were still held across `append_key`, this read would deadlock!
        assert!(rcu2.query_point(42_000));
        assert_eq!(rcu2.len(), 1);
        release.wait();

        writer.join().expect("writer finished cleanly");
        rcu2.verify_invariants().unwrap();
    }

    #[test]
    fn rcu_direct_methods_match_concurrent_api_across_auto_expansions() {
        let dir = TempDir::new().unwrap();
        let rm = Arc::new(
            RocksDbReverseMap::open(dir.path().join("rm_direct"), ReverseMapConfig::default())
                .unwrap(),
        );
        let cfg = test_config(7, 4); // 128 slots -> triggers multiple auto-expansions
        let mut rcu = RcuAerisFilter::new(5, AerisFilter::new(cfg).unwrap(), rm);

        let initial_epoch = rcu.epoch();
        for k in 0..320u64 {
            let key = k * 37 + 5;
            rcu.insert_direct(key).unwrap();
            assert!(rcu.query_point_direct(key));
            assert!(rcu.query_point(key));
        }

        assert!(rcu.epoch() > initial_epoch);
        assert_eq!(rcu.len(), 320);
        rcu.verify_invariants().unwrap();

        // Bit-exact agreement on present, absent, point, and range queries.
        for probe in 0..500u64 {
            let key = probe * 19;
            assert_eq!(rcu.query_point_direct(key), rcu.query_point(key));
            let range = QueryRange::new(key, key + 31).unwrap();
            assert_eq!(rcu.query_range_direct(range), rcu.query_range(range));
        }
    }
}
