//! # `aeris-scale`
//!
//! Scale Plane for the **Aeris-Based Optimized Range
//! Filtering System**, providing:
//! - [`EpochCell`]: minimal, safe Crossbeam epoch-based RCU cell isolating all
//!   raw pointer dereferencing and deferred reclamation (`unsafe` confined
//!   exclusively to [`epoch_cell`]).
//! - [`ChunkLockTable`]: cache-line-padded (`CachePadded<RwLock<()>>`) striped
//!   locking aligned to 64-slot Quotient Filter chunks with strict ascending
//!   multi-chunk acquisition for deadlock-free Robin Hood cluster updates.
//! - [`RcuAerisFilter`]: wait-free epoch-RCU reader hot path (`query_point`,
//!   `query_range`) with zero heap allocations and zero stalls across `2x`
//!   InfiniFilter expansions and asynchronous dynamic-`R` background rebuilds
//!   (`F_old -> F_new`).
//! - [`ShardedAerisSystem`]: range sharding over independent `2^(q_i)` shards
//!   covering `[0, u64::MAX]` contiguously with wait-free `O(log S)` RCU routing,
//!   incremental RAM growth (`+1/S` per hot-shard expansion/split), data-driven
//!   median splits, adjacent underloaded merges, race-free `retired` retry
//!   semantics, and 100% monotonic false-positive elimination across topology
//!   changes.
//!
//! # Global Lock-Ordering Hierarchy
//!
//! To statically prevent deadlocks across every concurrent read, write,
//! expansion, background rebuild, split, and merge path, all locks in
//! `aeris-scale` MUST be acquired strictly in ascending rank order:
//!
//! 1. **`ShardedAerisSystem::topology_lock`** — acquired ONLY by topology
//!    transitions (`split_shard`, `merge_adjacent_shards`, `rebalance_if_needed`);
//!    **NEVER** held on normal reads (`query_point`, `query_range`) or writes
//!    (`insert`, `delete`, `adapt_range`).
//! 2. **`ShardInstance::write_coordinator`** — acquired per shard for mutations
//!    and split/merge quiescence; when locking two adjacent shards for a merge,
//!    ALWAYS acquired in **strict ascending `ShardBounds::min_key` order**
//!    (`left` then `right`). **NEVER** held by readers.
//! 3. **`RcuAerisFilter::swap_lock`** — serializes RCU swaps, expansions, and
//!    background rebuild start/finish barriers on a single filter without EVER
//!    blocking readers.
//! 4. **`ChunkLockTable` stripes** — acquired in **strict ascending stripe index
//!    order** (`s_0 < s_1 < ... < s_{k-1}`) for single-chunk, cluster-range,
//!    multi-chunk, and `write_all` barriers.
//! 5. **`FilterGeneration::filter`** (`parking_lot::RwLock<AerisFilter>`) — shared
//!    read lock on the zero-stall query hot path; exclusive write lock on
//!    chunk-protected mutations.
//! 6. **Leaf Storage / Control Plane Locks** — [`aeris_types::ReverseMapStore`]
//!    (RocksDB), [`aeris_control::FalsePositiveLog`], [`aeris_control::DeltaBuffer`],
//!    and [`aeris_control::SlidingWindowEstimator`].
//!
//! # Safety Policy
//! `unsafe` code is strictly confined to [`epoch_cell`] for Crossbeam epoch
//! pointer dereference, `defer_destroy`, and final `Drop` reclamation, backed
//! by explicit `// SAFETY:` proofs. Every other module enforces
//! `#![forbid(unsafe_code)]`.

#![deny(missing_docs)]

pub mod chunk_locks;
pub mod epoch_cell;
pub mod rcu_filter;
pub mod sharding;

pub use aeris_control;
pub use aeris_core;
pub use aeris_storage;
pub use aeris_types;

pub use chunk_locks::{
    ChunkLockTable, ChunkReadGuard, ChunkWriteGuard, ClusterReadGuard, ClusterWriteGuard,
    DEFAULT_MAX_LOCK_STRIPES,
};
pub use epoch_cell::{EpochCell, INITIAL_EPOCH};
pub use rcu_filter::{FilterGeneration, RcuAerisFilter, RcuExpansionReport, RcuRebuildReport};
pub use sharding::{
    compute_bounds_from_split_keys, compute_median_split_key, compute_uniform_bounds, MergeReport,
    RebalanceReport, RoutingSnapshot, ShardInstance, ShardedAerisSystem, ShardingConfig,
    SplitReport, DEFAULT_INITIAL_SHARD_COUNT, DEFAULT_MAX_SHARDS, DEFAULT_MERGE_LOAD_THRESHOLD,
    DEFAULT_MIN_SHARDS, DEFAULT_SPLIT_LOAD_THRESHOLD, DEFAULT_SPLIT_MIN_KEYS,
    MAX_POST_MERGE_LOAD_FACTOR,
};
