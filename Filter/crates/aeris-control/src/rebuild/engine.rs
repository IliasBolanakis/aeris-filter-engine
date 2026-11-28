//! Unified [`ControlledAerisEngine`] facade coupling the Data Plane,
//! Storage Plane, and Deterministic Control Plane.

use super::config::{ControlledEngineConfig, QueryOutcome};
use super::coordinator::AsyncRebuildCoordinator;
use super::engine_lifecycle::open_or_build_initial_filter;
use super::staged::{adapt_with_auto_expand, insert_with_auto_expand};
use super::throttle::IoThrottleConfig;
use crate::estimator::SlidingWindowEstimator;
use crate::fp_log::FalsePositiveLog;
use crate::optimizer::DiscreteCostOptimizer;
use crate::trigger::ProactiveTrigger;
use aeris_core::AerisFilter;
use aeris_storage::{ReferenceKvStore, ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, AerisResult, CostModelParams, Epoch, Key, QueryRange, RangeFilter, ReverseMapStore,
    WorkloadEstimator,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

/// Subdirectory name for the primary [`ReferenceKvStore`] under a [`ControlledAerisEngine`] root.
const ENGINE_KV_DIR_NAME: &str = "kv_db";

/// Filename for the persistent [`FalsePositiveLog`] under a [`ControlledAerisEngine`] root.
const ENGINE_FP_LOG_NAME: &str = "fp.log";

/// Prefix for generational [`RocksDbReverseMap`] subdirectories swapped across rebuilds.
const ENGINE_RM_GEN_PREFIX: &str = "reverse_map_gen_";

/// Pending destination RocksDB Reverse Map awaiting atomic swap at rebuild finish.
pub(super) struct PendingReverseMap {
    pub(super) generation: u64,
    pub(super) store: Arc<RocksDbReverseMap>,
}

/// Unified high-level facade coupling the real Data Plane ([`AerisFilter`]),
/// real Storage Plane ([`ReferenceKvStore`] + [`RocksDbReverseMap`]), and
/// complete Deterministic Control Plane ([`SlidingWindowEstimator`],
/// [`DiscreteCostOptimizer`], [`ProactiveTrigger`], [`FalsePositiveLog`], and
/// [`AsyncRebuildCoordinator`]).
pub struct ControlledAerisEngine {
    pub(super) root_dir: PathBuf,
    pub(super) epoch: Epoch,
    pub(super) filter: AerisFilter,
    pub(super) kv_store: ReferenceKvStore,
    pub(super) reverse_map: Arc<RocksDbReverseMap>,
    pub(super) reverse_map_config: ReverseMapConfig,
    pub(super) reverse_map_generation: u64,
    pub(super) pending_dest_rm: Option<PendingReverseMap>,
    pub(super) estimator: SlidingWindowEstimator,
    pub(super) optimizer: DiscreteCostOptimizer,
    pub(super) cost_model: CostModelParams,
    pub(super) trigger: ProactiveTrigger,
    pub(super) fp_log: Arc<FalsePositiveLog>,
    pub(super) coordinator: AsyncRebuildCoordinator,
    pub(super) throttle_config: IoThrottleConfig,
    pub(super) dynamic_r_enabled: bool,
    pub(super) bypass_active: bool,
    pub(super) pending_contraction: bool,
    pub(super) total_rebuilds: u64,
    pub(super) total_contractions: u64,
    pub(super) total_false_positives: u64,
    pub(super) total_bypassed_queries: u64,
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
        let reverse_map = Arc::new(RocksDbReverseMap::open(
            reverse_map_gen_path(&root_dir, initial_gen),
            config.reverse_map_config.clone(),
        )?);
        let fp_log = Arc::new(FalsePositiveLog::open(
            root_dir.join(ENGINE_FP_LOG_NAME),
            config.fp_log_config,
        )?);

        let (filter, loaded_keys) = open_or_build_initial_filter(
            &config.filter_config,
            &kv_store,
            reverse_map.as_ref(),
            fp_log.as_ref(),
        )?;

        let mut estimator_config = config.estimator_config.clone();
        if !config.dynamic_r_enabled {
            estimator_config.enabled = false;
        }
        let estimator = SlidingWindowEstimator::with_initial_span(
            estimator_config,
            filter.config().max_range_r,
        )?;
        let trigger = ProactiveTrigger::new(config.trigger_config.clone())?;

        info!(
            root_dir = %root_dir.display(),
            q = filter.config().q,
            r = filter.config().r,
            m = filter.config().m,
            dynamic_r_enabled = config.dynamic_r_enabled,
            loaded_keys,
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
    /// and [`RocksDbReverseMap`], mirroring the insert into [`super::DeltaBuffer`] if an
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
    /// [`RocksDbReverseMap`], mirroring the deletion into [`super::DeltaBuffer`] if an
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
            return self.handle_bypass_query(range);
        }

        if !self.filter.query_range(range) {
            self.record_estimator_if_enabled(range, false);
            return Ok(QueryOutcome {
                non_empty: false,
                filter_returned_positive: false,
                was_false_positive: false,
                bypassed_filter: false,
                adapted: false,
            });
        }

        if self.kv_store.range_non_empty(range)? {
            self.record_estimator_if_enabled(range, false);
            return Ok(QueryOutcome {
                non_empty: true,
                filter_returned_positive: true,
                was_false_positive: false,
                bypassed_filter: false,
                adapted: false,
            });
        }

        self.handle_false_positive(range)
    }

    #[inline]
    fn record_estimator_if_enabled(&mut self, range: QueryRange, is_fp: bool) {
        if self.dynamic_r_enabled {
            self.estimator.record_query(range, is_fp);
        }
    }

    fn handle_bypass_query(&mut self, range: QueryRange) -> AerisResult<QueryOutcome> {
        let non_empty = self.kv_store.range_non_empty(range)?;
        self.record_estimator_if_enabled(range, false);
        self.total_bypassed_queries = self.total_bypassed_queries.saturating_add(1);
        Ok(QueryOutcome {
            non_empty,
            filter_returned_positive: false,
            was_false_positive: false,
            bypassed_filter: true,
            adapted: false,
        })
    }

    fn handle_false_positive(&mut self, range: QueryRange) -> AerisResult<QueryOutcome> {
        self.total_false_positives = self.total_false_positives.saturating_add(1);
        self.record_estimator_if_enabled(range, true);
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

pub(super) fn reverse_map_gen_path(root_dir: &Path, generation: u64) -> PathBuf {
    root_dir.join(format!("{ENGINE_RM_GEN_PREFIX}{generation}"))
}

fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
