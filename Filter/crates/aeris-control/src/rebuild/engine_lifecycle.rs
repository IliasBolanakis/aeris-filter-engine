//! Control-plane evaluation and generational background rebuild lifecycle
//! methods for [`super::ControlledAerisEngine`].

use super::config::ControlPlaneDecision;
use super::coordinator::{bulk_load_with_auto_expand, RebuildReason, RebuildReport};
use super::engine::{reverse_map_gen_path, ControlledAerisEngine, PendingReverseMap};
use crate::fp_log::FalsePositiveLog;
use crate::trigger::{TriggerAction, TriggerInput};
use aeris_core::AerisFilter;
use aeris_storage::{ReferenceKvStore, RocksDbReverseMap};
use aeris_types::{
    AerisError, AerisResult, CostOptimizer, FilterConfig, RangeFilter, ReverseMapStore,
    WorkloadEstimator,
};
use std::fs;
use std::sync::Arc;
use tracing::info;

impl ControlledAerisEngine {
    /// Evaluates the full Deterministic Control Plane ([`crate::estimator::SlidingWindowEstimator`],
    /// [`crate::optimizer::DiscreteCostOptimizer`], and [`crate::trigger::ProactiveTrigger`]),
    /// updating the filter bypass state and returning a [`ControlPlaneDecision`].
    ///
    /// When `dynamic_r_enabled` is `false` (static-`R` mode), forces
    /// `optimization.should_rebuild = false` while leaving [`crate::trigger::ProactiveTrigger`]
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
    /// serving off `F_old` while capturing mutations in [`super::DeltaBuffer`] and new false
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

    /// Blocks until the active background rebuild completes, drains [`super::DeltaBuffer`],
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

    /// Executes a `q -> q - 1` filter contraction if [`crate::trigger::ProactiveTrigger`] has
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
}

pub(super) fn open_or_build_initial_filter(
    filter_config: &FilterConfig,
    kv_store: &ReferenceKvStore,
    reverse_map: &dyn ReverseMapStore,
    fp_log: &FalsePositiveLog,
) -> AerisResult<(AerisFilter, usize)> {
    let existing_keys = kv_store.scan_all_keys()?;
    if existing_keys.is_empty() {
        return Ok((AerisFilter::new(filter_config.clone())?, 0));
    }

    let (mut built, _, _) =
        bulk_load_with_auto_expand(filter_config.clone(), &existing_keys, reverse_map)?;
    if !fp_log.is_empty() {
        let _ = fp_log.replay_onto(&mut built, reverse_map, |r| {
            Ok(!kv_store.range_non_empty(r)?)
        })?;
    }
    Ok((built, existing_keys.len()))
}
