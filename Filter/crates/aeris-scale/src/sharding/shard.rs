//! Independent shard state ([`ShardInstance`]) and persistent shard construction
//! helpers for [`super::ShardedAerisSystem`].

use super::config::ShardingConfig;
use crate::epoch_cell::COUNTER_STRIPES;
use crate::rcu_filter::RcuAerisFilter;
use aeris_control::{FalsePositiveLog, SlidingWindowEstimator};
use aeris_core::AerisFilter;
use aeris_storage::RocksDbReverseMap;
use aeris_types::{
    AerisError, AerisResult, FilterConfig, FpLogEntry, Key, QueryRange, RangeFilter,
    ReverseMapStore, ShardBounds, ShardId, ShardTelemetrySnapshot, WorkloadEstimator,
    MAX_QUOTIENT_BITS,
};
use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, RwLock};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Subdirectory name for each shard's persistent [`RocksDbReverseMap`].
const SHARD_REVERSE_MAP_DIR: &str = "reverse_map";

/// Filename for each shard's persistent [`FalsePositiveLog`].
const SHARD_FP_LOG_FILE: &str = "fp.log";

/// Independent range shard owning a contiguous key interval `[min_key, max_key]`,
/// an epoch-RCU [`RcuAerisFilter`], a persistent [`RocksDbReverseMap`], a
/// persistent [`FalsePositiveLog`], and a local [`SlidingWindowEstimator`].
///
/// Cache-line layout isolates read-only routing/filter handles, striped per-thread
/// query counters, and write/topology coordination metadata onto distinct cache
/// lines so multi-core reads never suffer false sharing or atomic ping-pong.
pub struct ShardInstance {
    pub(super) shard_id: ShardId,
    pub(super) bounds: ShardBounds,
    pub(super) rcu_filter: RcuAerisFilter,
    reverse_map: Arc<RocksDbReverseMap>,
    pub(super) fp_log: Arc<FalsePositiveLog>,
    pub(super) estimator: Mutex<SlidingWindowEstimator>,
    /// Lock Level 2: coordinates concurrent shard mutations (`read()`) with
    /// exclusive split/merge quiescence (`write()`, strict ascending `min_key`
    /// order). NEVER held by point or range query readers.
    pub(super) write_coordinator: CachePadded<RwLock<()>>,
    /// Set to `true` before releasing `write_coordinator` after split/merge publishes
    /// the replacement [`super::RoutingSnapshot`], forcing waiting writers to re-route cleanly.
    retired: CachePadded<AtomicBool>,
    /// Cache-line-padded striped exact query counters indexed by thread-local
    /// stripe slot, eliminating cross-core `LOCK XADD` cache-line bouncing on reads.
    query_counts: [CachePadded<AtomicU64>; COUNTER_STRIPES],
    pub(super) fp_count: CachePadded<AtomicU64>,
    pub(super) write_count: CachePadded<AtomicU64>,
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
    pub(super) fn mark_retired(&self) {
        self.retired.store(true, Ordering::Release);
    }

    /// Increments the thread-local striped query counter for this shard with zero
    /// cross-core cache-line contention.
    #[inline]
    pub(super) fn record_query_hit_with_slot(&self, slot: usize) {
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

/// Builds a fresh persistent [`ShardInstance`] at `<root_dir>/shard_<shard_id>/`,
/// bulk-loads `sorted_keys` with auto-expansion, migrates relevant empty FP log
/// entries clamped to `bounds`, and replays them for 100% monotonic adaptivity.
pub(super) fn build_shard_from_state(
    root_dir: &Path,
    config: &ShardingConfig,
    shard_id: ShardId,
    bounds: ShardBounds,
    base_cfg: &FilterConfig,
    sorted_keys: &[Key],
    source_fp_entries: &[FpLogEntry],
) -> AerisResult<(ShardInstance, usize)> {
    let (reverse_map, fp_log) = open_shard_stores(root_dir, config, shard_id)?;
    let mut filter = bulk_load_with_auto_expand(base_cfg, sorted_keys, reverse_map.as_ref())?;
    let fp_replayed = migrate_and_replay_fp_log(
        &fp_log,
        &mut filter,
        reverse_map.as_ref(),
        bounds,
        sorted_keys,
        source_fp_entries,
    )?;

    let estimator = SlidingWindowEstimator::with_initial_span(
        config.estimator_config.clone(),
        filter.config().max_range_r,
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

fn open_shard_stores(
    root_dir: &Path,
    config: &ShardingConfig,
    shard_id: ShardId,
) -> AerisResult<(Arc<RocksDbReverseMap>, Arc<FalsePositiveLog>)> {
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
    Ok((reverse_map, fp_log))
}

fn migrate_and_replay_fp_log(
    fp_log: &FalsePositiveLog,
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    bounds: ShardBounds,
    sorted_keys: &[Key],
    source_fp_entries: &[FpLogEntry],
) -> AerisResult<usize> {
    for entry in source_fp_entries {
        if bounds.overlaps_range(entry.range.start(), entry.range.end()) {
            let clamped = clamp_range_to_bounds(entry.range, bounds);
            if sorted_slice_range_is_empty(sorted_keys, clamped) {
                let _ = fp_log.record(clamped, entry.timestamp_ms)?;
            }
        }
    }

    if fp_log.is_empty() {
        return Ok(0);
    }
    let replay = fp_log.replay_onto(filter, reverse_map, |r| {
        Ok(sorted_slice_range_is_empty(sorted_keys, r))
    })?;
    Ok(replay.replayed_entries)
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

/// Clamps `range` to the non-empty intersection with `bounds`.
///
/// # Preconditions
/// `bounds.overlaps_range(range.start(), range.end())` MUST be `true`.
#[inline]
pub(super) fn clamp_range_to_bounds(range: QueryRange, bounds: ShardBounds) -> QueryRange {
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
pub(super) fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
