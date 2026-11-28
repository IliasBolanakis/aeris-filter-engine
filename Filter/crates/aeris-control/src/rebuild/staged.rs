//! Two-phase incremental [`DeltaBuffer`] drain and post-tail verification
//! stage (`StagedRebuild`).

use super::coordinator::{BackgroundRebuildOutput, CompletedRebuild, RebuildReport, RebuildSlot};
use super::delta_buffer::{DeltaBuffer, DeltaMutation, DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE};
use crate::fp_log::FpReplayReport;
use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, Key, QueryRange, RangeFilter, ReverseMapStore, MAX_QUOTIENT_BITS,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// RAII drop guard ensuring [`RebuildSlot::Draining`] always transitions back to
/// [`RebuildSlot::Idle`] and deactivates [`DeltaBuffer`] if a [`StagedRebuild`]
/// is dropped early due to error or panic.
pub(super) struct DrainGuard {
    in_flight: Arc<Mutex<RebuildSlot>>,
    delta_buffer: Arc<DeltaBuffer>,
    active: bool,
}

impl DrainGuard {
    pub(super) fn new(in_flight: Arc<Mutex<RebuildSlot>>, delta_buffer: Arc<DeltaBuffer>) -> Self {
        Self {
            in_flight,
            delta_buffer,
            active: true,
        }
    }

    fn release(mut self) {
        self.active = false;
        *self.in_flight.lock() = RebuildSlot::Idle;
    }
}

impl Drop for DrainGuard {
    fn drop(&mut self) {
        if self.active {
            self.delta_buffer.deactivate();
            *self.in_flight.lock() = RebuildSlot::Idle;
        }
    }
}

/// Joined background rebuild handle supporting two-phase incremental [`DeltaBuffer`]
/// draining so large concurrent mutation backlogs are applied to `F_new` *before*
/// acquiring exclusive swap / chunk-quiescence locks.
///
/// # Two-Phase Protocol
/// 1. **Off-Lock Pre-Drain:** Call [`Self::pre_drain_until_small`] (or repeated
///    [`Self::pre_drain_batch`] calls) followed by [`Self::verify_pre_drained_keys`]
///    while foreground writers continue executing on `F_old`.
/// 2. **Locked Tail Swap:** Acquire swap/quiescence locks and invoke
///    [`Self::finish_tail`] or [`Self::finish_tail_with_live_keys`] to drain only
///    the tiny residual delta tail, replay catch-up [`crate::fp_log::FalsePositiveLog`]
///    entries, and return [`CompletedRebuild`].
pub struct StagedRebuild {
    pub(super) bg: BackgroundRebuildOutput,
    pub(super) started_at: Instant,
    pub(super) delta_buffer: Arc<DeltaBuffer>,
    pub(super) drain_guard: DrainGuard,
    pub(super) delta_inserts_applied: usize,
    pub(super) delta_deletes_applied: usize,
    pub(super) auto_expansions: usize,
    pub(super) all_inserted_delta_keys: Vec<Key>,
    pub(super) unverified_inserted_keys: Vec<Key>,
    pub(super) pre_drain_passes: usize,
    pub(super) pre_drained_mutations: usize,
}

impl StagedRebuild {
    /// Returns the number of mutations currently waiting in [`DeltaBuffer`].
    #[inline]
    #[must_use]
    pub fn pending_delta_len(&self) -> usize {
        self.delta_buffer.len()
    }

    /// Returns the number of off-lock pre-drain batch passes executed so far.
    #[inline]
    #[must_use]
    pub const fn pre_drain_passes(&self) -> usize {
        self.pre_drain_passes
    }

    /// Returns the cumulative number of delta mutations drained off-lock so far.
    #[inline]
    #[must_use]
    pub const fn pre_drained_mutations(&self) -> usize {
        self.pre_drained_mutations
    }

    /// Drains up to `max_items` mutations from [`DeltaBuffer`] and applies them
    /// idempotently onto `F_new` and `dest_reverse_map` (using batched Reverse Map
    /// inserts) while leaving [`DeltaBuffer`] active for concurrent writers.
    ///
    /// Returns the number of mutations popped and processed in this batch.
    ///
    /// # Errors
    /// Propagates any filter expansion or Reverse Map storage error.
    pub fn pre_drain_batch(&mut self, max_items: usize) -> AerisResult<usize> {
        let batch = self.delta_buffer.drain_batch(max_items);
        let count = batch.len();
        if count == 0 {
            return Ok(0);
        }
        self.apply_mutations(batch)?;
        self.pre_drain_passes = self.pre_drain_passes.saturating_add(1);
        self.pre_drained_mutations = self.pre_drained_mutations.saturating_add(count);
        Ok(count)
    }

    /// Incrementally drains [`DeltaBuffer`] in bounded `batch_size` passes *off-lock*
    /// until `delta_buffer.len() <= tail_threshold` or `max_passes` passes complete.
    ///
    /// Returns the total number of mutations drained off-lock.
    ///
    /// # Errors
    /// Propagates any filter expansion or Reverse Map storage error.
    pub fn pre_drain_until_small(
        &mut self,
        tail_threshold: usize,
        batch_size: usize,
        max_passes: usize,
    ) -> AerisResult<usize> {
        let effective_batch = batch_size.max(1);
        let mut drained_total = 0usize;
        let mut passes = 0usize;

        while passes < max_passes && self.delta_buffer.len() > tail_threshold {
            let drained = self.pre_drain_batch(effective_batch)?;
            if drained == 0 {
                break;
            }
            drained_total = drained_total.saturating_add(drained);
            passes += 1;
        }

        debug!(
            reason = ?self.bg.reason,
            passes,
            drained_total,
            remaining_tail = self.delta_buffer.len(),
            tail_threshold,
            "completed off-lock incremental DeltaBuffer pre-drain"
        );
        Ok(drained_total)
    }

    /// Verifies zero false negatives across all still-live delta-inserted keys drained
    /// so far, allowing verification of large pre-drained backlogs to run *off-lock*
    /// prior to acquiring swap/quiescence locks.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if any live delta-inserted key
    /// returns `false` on `query_point`.
    pub fn verify_pre_drained_keys(&mut self) -> AerisResult<()> {
        for key in self.unverified_inserted_keys.drain(..) {
            if self.bg.loaded_keys.contains(&key) && !self.bg.filter.query_point(key) {
                return Err(AerisError::InternalInvariant(format!(
                    "rebuilt filter produced false negative for pre-drained live key {key}"
                )));
            }
        }
        Ok(())
    }

    /// Completes the final locked phase: atomically drains and deactivates any
    /// residual [`DeltaBuffer`] tail, replays catch-up [`crate::fp_log::FalsePositiveLog`]
    /// entries using `verify_empty`, verifies zero false negatives, and returns [`CompletedRebuild`].
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on false-negative detection, or
    /// propagates storage, filter, or FP-log errors.
    pub fn finish_tail<F>(mut self, verify_empty: F) -> AerisResult<CompletedRebuild>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let expansions_before_tail = self.auto_expansions;
        let tail = self.delta_buffer.drain_and_deactivate();
        let tail_len = tail.len();
        if !tail.is_empty() {
            self.apply_mutations(tail)?;
        }

        self.finalize_with_verifier(expansions_before_tail, tail_len, verify_empty)
    }

    /// Variant of [`Self::finish_tail`] that uses the exact reconciled `live_keys`
    /// set (background snapshot plus all drained delta mutations) as the ground-truth
    /// emptiness oracle without requiring an extra Reverse Map full scan under lock.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on false-negative detection, or
    /// propagates storage, filter, or FP-log errors.
    pub fn finish_tail_with_live_keys(mut self) -> AerisResult<CompletedRebuild> {
        let expansions_before_tail = self.auto_expansions;
        let tail = self.delta_buffer.drain_and_deactivate();
        let tail_len = tail.len();
        if !tail.is_empty() {
            self.apply_mutations(tail)?;
        }

        let catchup_min_seq = if self.delta_inserts_applied > 0 || self.delta_deletes_applied > 0 {
            0
        } else {
            self.bg.fp_snapshot_seq
        };

        let live_keys_ref = &self.bg.loaded_keys;
        let fp_replay_catchup = self.bg.fp_log.replay_since(
            catchup_min_seq,
            &mut self.bg.filter,
            self.bg.dest_reverse_map.as_ref(),
            |range| {
                Ok(live_keys_ref
                    .range(range.start()..=range.end())
                    .next()
                    .is_none())
            },
        )?;
        self.auto_expansions += fp_replay_catchup.expansions_during_replay;

        self.verify_post_tail_and_build(expansions_before_tail, tail_len, fp_replay_catchup)
    }

    /// Applies a batch of FIFO [`DeltaMutation`]s idempotently onto `F_new` and
    /// `dest_reverse_map`, coalescing consecutive inserts into [`ReverseMapStore::append_batch`]
    /// and flushing pending Reverse Map inserts before any `expand()` or `Delete`.
    fn apply_mutations(&mut self, mutations: Vec<DeltaMutation>) -> AerisResult<()> {
        let mut rm_insert_batch: Vec<(aeris_types::Fingerprint, Key)> =
            Vec::with_capacity(mutations.len().min(DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE));

        for mutation in mutations {
            match mutation {
                DeltaMutation::Insert(key) => {
                    if self.bg.loaded_keys.insert(key) {
                        let fp = insert_with_auto_expand_flushing_rm(
                            &mut self.bg.filter,
                            self.bg.dest_reverse_map.as_ref(),
                            key,
                            &mut self.auto_expansions,
                            &mut rm_insert_batch,
                        )?;
                        rm_insert_batch.push((fp, key));
                        self.all_inserted_delta_keys.push(key);
                        self.unverified_inserted_keys.push(key);
                        self.delta_inserts_applied += 1;
                    }
                }
                DeltaMutation::Delete(key) => {
                    if self.bg.loaded_keys.remove(&key) {
                        if !rm_insert_batch.is_empty() {
                            self.bg.dest_reverse_map.append_batch(&rm_insert_batch)?;
                            rm_insert_batch.clear();
                        }
                        if let Some(fp) = self.bg.filter.delete(key)? {
                            self.bg.dest_reverse_map.remove_key(&fp, key)?;
                        }
                        self.delta_deletes_applied += 1;
                    }
                }
            }
        }

        if !rm_insert_batch.is_empty() {
            self.bg.dest_reverse_map.append_batch(&rm_insert_batch)?;
        }
        Ok(())
    }

    fn finalize_with_verifier<F>(
        mut self,
        expansions_before_tail: usize,
        tail_len: usize,
        verify_empty: F,
    ) -> AerisResult<CompletedRebuild>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let catchup_min_seq = if self.delta_inserts_applied > 0 || self.delta_deletes_applied > 0 {
            0
        } else {
            self.bg.fp_snapshot_seq
        };

        let fp_replay_catchup = self.bg.fp_log.replay_since(
            catchup_min_seq,
            &mut self.bg.filter,
            self.bg.dest_reverse_map.as_ref(),
            &verify_empty,
        )?;
        self.auto_expansions += fp_replay_catchup.expansions_during_replay;

        self.verify_post_tail_and_build(expansions_before_tail, tail_len, fp_replay_catchup)
    }

    fn verify_post_tail_and_build(
        self,
        expansions_before_tail: usize,
        tail_len: usize,
        fp_replay_catchup: FpReplayReport,
    ) -> AerisResult<CompletedRebuild> {
        let keys_to_verify: &[Key] = if self.auto_expansions > expansions_before_tail {
            &self.all_inserted_delta_keys
        } else {
            &self.unverified_inserted_keys
        };
        for &key in keys_to_verify {
            if self.bg.loaded_keys.contains(&key) && !self.bg.filter.query_point(key) {
                return Err(AerisError::InternalInvariant(format!(
                    "rebuilt filter produced false negative for live delta-inserted key {key}"
                )));
            }
        }

        let duration_ns = self.started_at.elapsed().as_nanos() as u64;
        let report = RebuildReport {
            reason: self.bg.reason,
            keys_loaded: self.bg.throttle_metrics.keys_paced as usize,
            delta_inserts_applied: self.delta_inserts_applied,
            delta_deletes_applied: self.delta_deletes_applied,
            fp_replay_initial: self.bg.fp_replay_initial,
            fp_replay_catchup,
            auto_expansions: self.auto_expansions,
            observed_bulk_load_speed: self.bg.observed_bulk_load_speed,
            throttle_metrics: self.bg.throttle_metrics,
            duration_ns,
        };

        info!(
            reason = ?report.reason,
            keys_loaded = report.keys_loaded,
            delta_inserts_applied = report.delta_inserts_applied,
            delta_deletes_applied = report.delta_deletes_applied,
            pre_drain_passes = self.pre_drain_passes,
            pre_drained_mutations = self.pre_drained_mutations,
            final_tail_mutations = tail_len,
            fp_collisions_resolved =
                report.fp_replay_initial.collisions_resolved
                    + report.fp_replay_catchup.collisions_resolved,
            auto_expansions = report.auto_expansions,
            final_q = self.bg.filter.config().q,
            final_m = self.bg.filter.config().m,
            duration_ns = report.duration_ns,
            "completed asynchronous rebuild and delta drain"
        );

        self.drain_guard.release();

        Ok(CompletedRebuild {
            filter: self.bg.filter,
            reverse_map: self.bg.dest_reverse_map,
            report,
        })
    }
}

/// Inserts `key` into `filter`, flushing `pending_rm_batch` into `reverse_map`
/// before invoking `filter.expand(reverse_map)` whenever [`AerisError::FilterFull`]
/// occurs so selective rejuvenation always observes every previously inserted key.
fn insert_with_auto_expand_flushing_rm(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    key: Key,
    expansions_counter: &mut usize,
    pending_rm_batch: &mut Vec<(aeris_types::Fingerprint, Key)>,
) -> AerisResult<aeris_types::Fingerprint> {
    let mut local_expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                if !pending_rm_batch.is_empty() {
                    reverse_map.append_batch(pending_rm_batch)?;
                    pending_rm_batch.clear();
                }
                filter.expand(reverse_map)?;
                local_expansions += 1;
                *expansions_counter += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Inserts `key` into `filter`, automatically invoking `filter.expand(reverse_map)`
/// whenever [`AerisError::FilterFull`] occurs.
pub(super) fn insert_with_auto_expand(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    key: Key,
    expansions_counter: &mut usize,
) -> AerisResult<aeris_types::Fingerprint> {
    let mut local_expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                filter.expand(reverse_map)?;
                local_expansions += 1;
                *expansions_counter += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Adapts `filter` to `range`, automatically invoking `filter.expand(reverse_map)`
/// whenever [`AerisError::FilterFull`] occurs.
pub(super) fn adapt_with_auto_expand(
    filter: &mut AerisFilter,
    range: QueryRange,
    reverse_map: &dyn ReverseMapStore,
) -> AerisResult<()> {
    let mut local_expansions = 0u8;
    loop {
        match filter.adapt(range, reverse_map) {
            Ok(_) => return Ok(()),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                filter.expand(reverse_map)?;
                local_expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}
