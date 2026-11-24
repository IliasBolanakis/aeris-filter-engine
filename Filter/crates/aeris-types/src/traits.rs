//! Core cross-plane trait contracts locked at Milestone M0.
//!
//! All architectural planes program strictly against these traits so pure
//! domain logic remains isolated from disk I/O, clocks, and IPC transport.

use crate::config::{CostModelParams, FilterConfig};
use crate::error::AerisResult;
use crate::fingerprint::Fingerprint;
use crate::range::{Epoch, Key, Lsn, QueryRange};
use crate::telemetry::{
    AdaptationReport, ExpansionReport, MemoryAllocationPlan, OptimizationDecision,
    ShardTelemetrySnapshot, TelemetrySnapshot, TuningProposal, WorkloadStats,
};
use crate::wal::{WalOp, WalRecord};

/// Data Plane contract for an adaptive, expandable Aeris Range Filter in RAM.
///
/// # Invariants
/// - **Zero False Negatives:** If a key `k` has been inserted and not deleted,
///   `query_point(k)` and `query_range(r)` for any `r` containing `k` MUST
///   return `true` across all adaptations, expansions, rebuilds, and recoveries.
/// - **Monotonic Adaptivity:** Once `adapt(range, reverse_map)` completes for an
///   empty range `[a, b]` (no inserted keys in `[a, b]`), subsequent calls to
///   `query_range(range)` MUST return `false` unless a key inside `[a, b]` is
///   subsequently inserted.
pub trait RangeFilter {
    /// Returns the active [`FilterConfig`] of this filter instance.
    fn config(&self) -> &FilterConfig;

    /// Inserts a 64-bit unsigned integer `key` into the filter.
    ///
    /// Returns the canonical [`Fingerprint`] of the keepsake box receiving `key`,
    /// which the caller persists into the [`ReverseMapStore`].
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::FilterFull`] if the filter cannot
    /// accommodate additional slots without exceeding [`FilterConfig::max_load_factor`].
    fn insert(&mut self, key: Key) -> AerisResult<Fingerprint>;

    /// Deletes a 64-bit unsigned integer `key` from the filter.
    ///
    /// Returns `Ok(Some(fp))` with the keepsake box fingerprint if a matching
    /// memento was removed, or `Ok(None)` if no matching entry existed.
    ///
    /// # Errors
    /// Returns an error if an internal storage invariant is violated.
    fn delete(&mut self, key: Key) -> AerisResult<Option<Fingerprint>>;

    /// Answers a point emptiness query for `key` (`true` = possibly present, `false` = definitely absent).
    ///
    /// # Complexity
    /// `O(1)` expected time, zero heap allocations.
    fn query_point(&self, key: Key) -> bool;

    /// Answers a closed range emptiness query `[a, b]` (`true` = possibly non-empty, `false` = definitely empty).
    ///
    /// # Complexity
    /// `O(ceil(len / P) + 1)` expected time (`O(1)` for length `<= R`), zero heap allocations.
    fn query_range(&self, range: QueryRange) -> bool;

    /// Locates all keepsake box fingerprints matching any subquery of `range`.
    ///
    /// Used off the hot path when a false positive has been confirmed against the
    /// backing store.
    fn locate_matching_fingerprints(&self, range: QueryRange) -> Vec<Fingerprint>;

    /// Monotonically adapts the filter to a verified false-positive range query `[a, b]`
    /// by retrieving colliding keys from `reverse_map`, extending colliding fingerprints
    /// (handling the query-partition edge case by extending colliding partitions first
    /// and the query partition last), opportunistically rejuvenating aged boxes, and
    /// atomically updating `reverse_map`.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::FilterFull`] if extension slots require
    /// expansion first, or [`crate::error::AerisError::StorageError`] on Reverse Map I/O failure.
    fn adapt(
        &mut self,
        range: QueryRange,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<AdaptationReport>;

    /// Doubles the slot table (`q -> q + 1`) using InfiniFilter bit-splitting with
    /// unary age counters and selective rejuvenation of oldest fingerprints via
    /// `reverse_map`.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::InvalidConfig`] if `q` is already at
    /// `MAX_QUOTIENT_BITS`, or [`crate::error::AerisError::StorageError`] on Reverse Map I/O failure.
    fn expand(&mut self, reverse_map: &dyn ReverseMapStore) -> AerisResult<ExpansionReport>;

    /// Returns the number of live keys currently represented by the filter.
    fn len(&self) -> usize;

    /// Returns `true` if no keys are currently stored in the filter.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total Quotient Filter slot capacity `N = 2^q`.
    fn capacity(&self) -> usize;

    /// Returns the current slot occupancy load factor `occupied_slots / N`.
    fn load_factor(&self) -> f64;

    /// Returns the cumulative number of `2x` expansion cycles completed.
    fn expansion_cycles(&self) -> u64;
}

/// Storage Plane contract for the write-optimized disk-resident Reverse Map
/// mapping canonical keepsake box [`Fingerprint`]s to full 64-bit [`Key`] lists.
///
/// Implementations (e.g., RocksDB with a custom key-list concatenation merge
/// operator) MUST support write-optimized out-of-place appends, tombstones,
/// compaction, and concurrent `&self` access.
pub trait ReverseMapStore: Send + Sync {
    /// Appends `key` to the key list associated with `fp` without read-modify-write I/O.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if persistence fails.
    fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()>;

    /// Atomically appends a batch of `(fingerprint, key)` mappings.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if the batch write fails.
    fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()>;

    /// Records a write-optimized key removal operand for `(fp, key)`.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if persistence fails.
    fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()>;

    /// Retrieves all live keys currently mapped to `fp` (folding pending appends,
    /// removals, and respecting prior tombstones).
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if storage read fails.
    fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>>;

    /// Writes a tombstone for `fp`, logically removing all prior key mappings for
    /// `fp` and allowing physical reclamation during compaction.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if persistence fails.
    fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()>;

    /// Atomically tombstones `old_fp` and appends all replacement `(new_fp, key)` entries.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if the atomic batch write fails.
    fn replace_fingerprint(
        &self,
        old_fp: &Fingerprint,
        new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()>;

    /// Scans all live keys currently stored in the Reverse Map (deduplicated).
    ///
    /// Used by background rebuilds when dynamic `R` reconfiguration rehashes the key set.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if iteration fails.
    fn scan_all_keys(&self) -> AerisResult<Vec<Key>>;

    /// Returns a streaming iterator over live keys stored in the Reverse Map
    /// for I/O-throttled background rebuilds.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if iterator creation fails.
    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>>;

    /// Returns an approximate count of live keys stored in the Reverse Map
    /// (used for `T_rebuild = |ReverseMap| / BulkLoadSpeed`).
    fn entry_count_estimate(&self) -> u64;

    /// Flushes in-memory write buffers (memtables) to durable storage.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if flush fails.
    fn flush(&self) -> AerisResult<()>;

    /// Triggers compaction to collapse concatenated key lists and reclaim tombstoned entries.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::StorageError`] if compaction fails.
    fn compact(&self) -> AerisResult<()>;

    /// Returns `true` if this store is an intentional no-op sink (e.g., pure-filter
    /// standalone `bulk_load` benchmarking without Reverse Map persistence), allowing
    /// bulk builders to skip `(Fingerprint, Key)` batch construction entirely while
    /// preserving `dyn ReverseMapStore` object safety.
    #[inline]
    fn is_noop(&self) -> bool {
        false
    }
}

/// Storage Plane contract for the crash-consistent Write-Ahead Log (WAL).
///
/// # Durability Invariant
/// Every acknowledged mutation MUST be appended and `fsync`-ed through [`WalEngine::sync`]
/// before becoming externally durable. Replay MUST be idempotent and MUST validate
/// per-record checksums.
pub trait WalEngine: Send + Sync {
    /// Appends a mutation operation to the WAL and returns its assigned monotonic [`Lsn`].
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::WalIoError`] or [`crate::error::AerisError::EncodingError`].
    fn append(&self, op: WalOp) -> AerisResult<Lsn>;

    /// Flushes OS page caches to stable storage (`fsync` / `sync_all`).
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::WalIoError`] on OS sync failure.
    fn sync(&self) -> AerisResult<()>;

    /// Returns the highest LSN appended so far (`0` if empty).
    fn last_lsn(&self) -> Lsn;

    /// Replays all intact, checksum-verified WAL records with `lsn >= start_lsn`
    /// in strict ascending LSN order.
    ///
    /// Torn tail records left by a sudden crash (`SIGKILL`) are detected via CRC32
    /// and cleanly truncated rather than corrupting recovered state.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::WalCorruption`] on mid-log corruption
    /// or [`crate::error::AerisError::WalIoError`] on filesystem failure.
    fn replay_from(&self, start_lsn: Lsn) -> AerisResult<Vec<WalRecord>>;

    /// Records a durable checkpoint at `up_to_lsn`, allowing log truncation of
    /// earlier records once both primary store and Reverse Map are flushed.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::WalIoError`] on filesystem failure.
    fn checkpoint(&self, up_to_lsn: Lsn) -> AerisResult<()>;
}

/// Control Plane contract for estimating query range quantiles, ingestion rate EWMA,
/// and bulk-load throughput with hysteresis, cooldown, and outlier clipping.
pub trait WorkloadEstimator: Send + Sync {
    /// Records an observed query range and whether it resulted in a false positive.
    fn record_query(&mut self, range: QueryRange, is_false_positive: bool);

    /// Records a batch of `count` ingested keys arriving over `elapsed_secs` seconds.
    fn record_ingest(&mut self, count: u64, elapsed_secs: f64);

    /// Updates the EWMA estimate of background bulk-load throughput (keys/sec).
    fn record_bulk_load_speed(&mut self, keys_per_sec: f64);

    /// Captures an immutable [`WorkloadStats`] snapshot for cost optimization and telemetry.
    fn snapshot(&self) -> WorkloadStats;

    /// Returns the current hysteresis-stabilized effective query span `R_eff`.
    fn effective_query_span(&self) -> u64;

    /// Returns the current EWMA key ingestion rate (keys/sec).
    fn ingestion_rate_ewma(&self) -> f64;
}

/// Control Plane contract for the deterministic analytical Discrete Cost Optimizer.
///
/// Exhaustively evaluates discrete `m = log2(P)` candidates (typically `~20` values)
/// in `< 5 us` while respecting the fixed payload budget `r + m = C` and minimum
/// remainder floor `r >= r_min(epsilon)`.
pub trait CostOptimizer: Send + Sync {
    /// Evaluates all valid discrete `m` candidates and returns an [`OptimizationDecision`].
    ///
    /// # Complexity
    /// `O(max_m)` time (`<= 21` iterations, `< 5 us`), `O(1)` space.
    fn optimize(
        &self,
        stats: &WorkloadStats,
        current: &FilterConfig,
        cost_model: &CostModelParams,
    ) -> OptimizationDecision;
}

/// Phase 5 Control Plane contract for optional asynchronous ML/RL workload
/// forecasting and multi-tenant RAM allocation.
///
/// Implementations MUST execute strictly off the Data Plane query hot path and
/// MUST fall back cleanly to the deterministic optimizer on timeout, crash, or
/// guardrail rejection.
pub trait TuningAdvisor: Send + Sync {
    /// Requests a macro-level tuning proposal for the supplied [`TelemetrySnapshot`].
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::SidecarUnavailable`],
    /// [`crate::error::AerisError::SidecarTimeout`], or
    /// [`crate::error::AerisError::GuardrailRejected`].
    fn advise(&self, telemetry: &TelemetrySnapshot) -> AerisResult<TuningProposal>;

    /// Requests a multi-tenant RAM allocation plan across active range shards.
    ///
    /// # Errors
    /// Returns [`crate::error::AerisError::SidecarUnavailable`] or
    /// [`crate::error::AerisError::SidecarTimeout`].
    fn allocate_memory(
        &self,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan>;

    /// Returns `true` if the advisor backend is currently healthy and reachable.
    fn is_healthy(&self) -> bool;
}
