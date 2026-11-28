//! Zero-stall RCU swap, `2x` InfiniFilter expansion, and asynchronous
//! background dynamic-`R` rebuild orchestration for [`super::RcuAerisFilter`].

use super::generation::RcuGeneration;
use super::RcuAerisFilter;
use aeris_control::{
    CompletedRebuild, DeltaMutation, FalsePositiveLog, IoThrottleConfig, RebuildReason,
    RebuildReport, StagedRebuild, DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
    DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES, DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
};
use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, Epoch, ExpansionReport, FilterConfig, QueryRange, RangeFilter,
    ReverseMapStore, MAX_QUOTIENT_BITS,
};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tracing::{debug, info};

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

impl RcuAerisFilter {
    /// Atomically publishes a prebuilt replacement `(new_filter, new_reverse_map)`
    /// via Crossbeam epoch RCU (`F_old -> F_new`), draining any active [`aeris_control::DeltaBuffer`]
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
        let alpha = new_filter.load_factor();
        let new_cfg = new_filter.config().clone();
        let published_epoch = self.publish_new_generation(gen_old, new_filter, new_reverse_map);

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
    /// atomically swaps `F_old -> F_new` via [`crate::EpochCell::swap`].
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
    pub(super) fn expand_rcu_if_epoch(
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
        // Readers continue taking private-stripe `gen_old.filter.read()` with zero stalls.
        let _all_chunks = gen_old.chunk_locks.write_all();
        let mut f_new = {
            let reader = gen_old.filter.read();
            reader.clone()
        };

        let reverse_map = gen_old.reverse_map();
        let expansion = f_new.expand(reverse_map.as_ref())?;

        let old_epoch = gen_old.epoch;
        let alpha = f_new.load_factor();
        let published_epoch = self.publish_new_generation(gen_old, f_new, reverse_map);

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
    /// [`aeris_control::AsyncRebuildCoordinator`], activating [`aeris_control::DeltaBuffer`]
    /// atomically with respect to writers on `F_old` while keeping readers
    /// completely unblocked.
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
    /// incrementally pre-draining [`aeris_control::DeltaBuffer`] mutations off-lock
    /// until the residual tail is small, then acquiring `swap_lock` + chunk quiescence
    /// strictly for the tiny tail drain, catch-up [`FalsePositiveLog`] replay, and
    /// atomic `F_old -> F_new` RCU swap.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if no rebuild is active, or
    /// propagates worker, delta-drain, or FP-replay errors.
    pub fn finish_async_rebuild_and_swap<F>(&self, verify_empty: F) -> AerisResult<RcuRebuildReport>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        self.finish_async_rebuild_core("finish_async_rebuild_and_swap", move |staged| {
            staged.finish_tail(verify_empty)
        })
    }

    /// Completes an active background rebuild using two-phase incremental delta
    /// draining, deriving exact range emptiness directly from the reconciled live
    /// key set at the swap boundary (requiring no external KV store handle or locked
    /// Reverse Map scan), and atomically swaps `F_old -> F_new`.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if no rebuild is active, or
    /// propagates rebuild, delta-drain, or FP-replay errors.
    pub fn finish_async_rebuild_and_swap_from_reverse_map(&self) -> AerisResult<RcuRebuildReport> {
        self.finish_async_rebuild_core(
            "finish_async_rebuild_and_swap_from_reverse_map",
            StagedRebuild::finish_tail_with_live_keys,
        )
    }

    /// Runs a full asynchronous background rebuild (`F_old -> F_new`) via
    /// [`aeris_control::AsyncRebuildCoordinator`] with [`FalsePositiveLog`] replay
    /// and atomic delta draining, publishing `F_new` via Crossbeam epoch RCU while
    /// keeping concurrent readers and writers completely unblocked during the rebuild.
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

    /// Shared two-phase completion pipeline: off-lock worker join + pre-drain,
    /// followed by locked tail reconciliation and atomic RCU generation swap.
    fn finish_async_rebuild_core<F>(
        &self,
        caller: &str,
        finish_tail: F,
    ) -> AerisResult<RcuRebuildReport>
    where
        F: FnOnce(StagedRebuild) -> AerisResult<CompletedRebuild>,
    {
        if !self.coordinator.is_in_progress() {
            return Err(AerisError::InternalInvariant(format!(
                "{caller} called with no active rebuild"
            )));
        }
        while self.coordinator.is_in_progress() && !self.coordinator.is_ready() {
            thread::yield_now();
        }

        // Phase 1 (off-lock): join worker and pre-drain buffered deltas in bounded
        // batches while foreground readers AND writers keep running on `F_old`.
        let mut staged = self.coordinator.prepare_finish()?;
        let _ = staged.pre_drain_until_small(
            DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
            DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
            DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
        )?;
        staged.verify_pre_drained_keys()?;

        // Phase 2 (locked): acquire `swap_lock` + chunk quiescence strictly for
        // residual tail drain, catch-up FP replay, and atomic `EpochCell::swap`.
        let swap_start = Instant::now();
        let _swap_guard = self.swap_lock.lock();
        let guard = crossbeam_epoch::pin();
        let gen_old = self.current.load(&guard);
        let _all_chunks = gen_old.chunk_locks.write_all();

        let completed = finish_tail(staged)?;
        self.publish_completed_rebuild(gen_old, completed, swap_start)
    }

    /// Publishes a [`CompletedRebuild`] atomically while the caller holds
    /// `swap_lock` and `gen_old.chunk_locks.write_all()`.
    fn publish_completed_rebuild(
        &self,
        gen_old: &RcuGeneration,
        completed: CompletedRebuild,
        swap_start: Instant,
    ) -> AerisResult<RcuRebuildReport> {
        let old_epoch = gen_old.epoch;
        let alpha = completed.filter.load_factor();
        let new_cfg = completed.filter.config().clone();
        let published_epoch =
            self.publish_new_generation(gen_old, completed.filter, completed.reverse_map);

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

    /// Retires `gen_old` and atomically publishes `(new_filter, new_reverse_map)`
    /// at `gen_old.epoch + 1` while the caller holds `swap_lock` and `write_all()`.
    fn publish_new_generation(
        &self,
        gen_old: &RcuGeneration,
        new_filter: AerisFilter,
        new_reverse_map: Arc<dyn ReverseMapStore>,
    ) -> Epoch {
        let next_epoch = gen_old.epoch + 1;
        let new_gen = RcuGeneration::new(
            next_epoch,
            new_filter,
            new_reverse_map,
            self.max_lock_stripes,
        );
        // Mark `gen_old` retired under `write_all()` so waiting writers immediately
        // re-load the newly published generation upon guard release.
        gen_old.mark_retired();
        let published_epoch = self.current.swap(new_gen);
        debug_assert_eq!(published_epoch, next_epoch);
        published_epoch
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
