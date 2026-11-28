//! Crossbeam epoch-based RCU wrapper ([`RcuAerisFilter`]) coupling zero-stall
//! concurrent readers with fine-grained chunk-striped writers and atomic
//! `F_old -> F_new` swaps.
//!
//! # Concurrency & Zero-Stall Reader Guarantee
//! - **Readers (`query_point`, `query_range`):** Pin a Crossbeam epoch guard,
//!   load `&RcuGeneration` lock-free from [`EpochCell`], acquire a shared
//!   read lock on `RcuGeneration::filter`, and execute with **zero heap
//!   allocations**. Readers NEVER acquire `swap_lock` or [`ChunkLockTable`]
//!   stripes, so readers on `F_old` run completely uninterrupted throughout
//!   `2x` expansions, background dynamic-`R` rebuilds, and atomic swaps.
//! - **Writers (`insert`, `delete`, `adapt`):** Acquire the affected chunk/cluster
//!   stripes in strict ascending order via [`ChunkLockTable`], verify that the
//!   loaded generation has not been retired (`RcuGeneration::is_retired`),
//!   acquire `RcuGeneration::filter.write()`, persist to the generation's
//!   coupled [`ReverseMapStore`], and mirror mutations into [`DeltaBuffer`] when
//!   a background rebuild is active.
//! - **Swappers (`swap_filter`, `expand_rcu`, `rebuild_async_and_swap`):**
//!   Serialize via `swap_lock`, quiesce writers on the retiring generation via
//!   `ChunkLockTable::write_all()` (WITHOUT blocking readers on `filter.read()`),
//!   drain pending [`DeltaBuffer`] mutations and catch-up [`aeris_control::FalsePositiveLog`]
//!   entries, mark `gen_old` retired, and atomically publish `F_new` through
//!   [`EpochCell::swap`].
//!
//! # Lock Ordering
//! Strictly obeys the crate-level hierarchy:
//! `swap_lock` -> `ChunkLockTable` stripes (strict ascending) ->
//! `RcuGeneration::filter` -> `ReverseMapStore` / `FalsePositiveLog` / `DeltaBuffer`.

#![forbid(unsafe_code)]

mod generation;
mod rebuild;
#[cfg(test)]
mod tests;

pub use generation::FilterGeneration;
pub use rebuild::{RcuExpansionReport, RcuRebuildReport};

use crate::chunk_locks::{ClusterWriteGuard, DEFAULT_MAX_LOCK_STRIPES};
use crate::epoch_cell::EpochCell;
use aeris_control::{AsyncRebuildCoordinator, DeltaBuffer};
use aeris_core::AerisFilter;
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, Epoch, FilterConfig, Fingerprint, Key,
    PartitionHasher, QueryRange, RangeFilter, ReverseMapStore, ShardId, MAX_QUOTIENT_BITS,
    SLOTS_PER_CHUNK_LOG2,
};
use crossbeam_epoch::Guard;
use crossbeam_utils::CachePadded;
use generation::RcuGeneration;
use parking_lot::Mutex;
use std::sync::atomic::Ordering;
use std::sync::Arc;

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
    /// bypassing Crossbeam epoch pinning, chunk-stripe locking, and reader-stripe
    /// acquisition on the common non-expanding path while persisting to the coupled
    /// [`ReverseMapStore`], mirroring into [`DeltaBuffer`], and auto-expanding via
    /// [`Self::expand_rcu`] on [`AerisError::FilterFull`].
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
    /// Automatically triggers a zero-stall RCU `2x` expansion ([`Self::expand_rcu`])
    /// and retries whenever [`AerisError::FilterFull`] is encountered.
    ///
    /// # Errors
    /// Propagates [`AerisError::InvalidConfig`] if `q` is already at
    /// `MAX_QUOTIENT_BITS`, or [`AerisError::StorageError`] on Reverse Map I/O failure.
    pub fn insert(&self, key: Key) -> AerisResult<Fingerprint> {
        let guard = crossbeam_epoch::pin();
        self.insert_pinned(key, &guard)
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
    let max_subqueries = gen.chunk_locks.num_stripes();
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
