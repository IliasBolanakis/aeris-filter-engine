//! Asynchronous background rebuild coordinator and worker thread lifecycle.

use super::delta_buffer::{
    DeltaBuffer, DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE, DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
    DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
};
use super::staged::{DrainGuard, StagedRebuild};
use super::throttle::{IoThrottleConfig, IoThrottler, ThrottleMetrics};
use crate::fp_log::{FalsePositiveLog, FpReplayReport};
use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, FilterConfig, Key, QueryRange, ReverseMapStore, MAX_QUOTIENT_BITS,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tracing::{debug, error, info};

/// Dedicated OS thread name assigned to background rebuild workers.
pub const BG_REBUILD_THREAD_NAME: &str = "aeris-bg-rebuild";

/// Minimum elapsed seconds floor when computing observed bulk-load throughput.
const MIN_BULK_LOAD_ELAPSED_SECS: f64 = 1e-6;

/// Lifecycle cause for launching an asynchronous background rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebuildReason {
    /// Dynamic partition size reconfiguration (`m_old -> m_new`) under fixed budget `C`.
    DynamicRangeChange {
        /// Previous memento bit width `m = log2(P)`.
        old_m: u8,
        /// Target memento bit width `m* = log2(P*)`.
        new_m: u8,
    },
    /// Proactive EWMA-triggered capacity expansion (`q_old -> q_new`).
    ProactiveExpansion {
        /// Previous quotient bit width `q`.
        old_q: u8,
        /// Target quotient bit width `q_new`.
        new_q: u8,
    },
    /// Hysteresis-guarded capacity contraction (`q_old -> q_new`) after bulk deletes.
    Contraction {
        /// Previous quotient bit width `q`.
        old_q: u8,
        /// Target quotient bit width `q - 1`.
        new_q: u8,
    },
}

/// Comprehensive diagnostic report returned upon completing a background rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RebuildReport {
    /// Lifecycle trigger that initiated the rebuild.
    pub reason: RebuildReason,
    /// Number of live keys streamed from the source Reverse Map during bulk load.
    pub keys_loaded: usize,
    /// Number of net new delta inserts applied during final drain.
    pub delta_inserts_applied: usize,
    /// Number of net delta deletes applied during final drain.
    pub delta_deletes_applied: usize,
    /// False-Positive Log replay report from the background worker phase.
    pub fp_replay_initial: FpReplayReport,
    /// False-Positive Log replay report from the post-delta catch-up phase.
    pub fp_replay_catchup: FpReplayReport,
    /// Number of automatic `2x` expansions executed during bulk load or delta drain.
    pub auto_expansions: usize,
    /// Observed background bulk-load throughput (keys/sec) for feeding `T_rebuild` EWMA.
    pub observed_bulk_load_speed: f64,
    /// I/O throttler metrics collected during the Reverse Map streaming scan.
    pub throttle_metrics: ThrottleMetrics,
    /// End-to-end duration from rebuild start through delta drain completion in nanoseconds.
    pub duration_ns: u64,
}

/// Ready-to-swap rebuilt filter, populated destination Reverse Map, and report
/// returned by [`AsyncRebuildCoordinator::finish_and_drain_deltas`].
pub struct CompletedRebuild {
    /// Fully built, delta-reconciled, FP-replayed replacement [`AerisFilter`] (`F_new`).
    pub filter: AerisFilter,
    /// Destination [`ReverseMapStore`] aligned with `filter`'s canonical fingerprints.
    pub reverse_map: Arc<dyn ReverseMapStore>,
    /// Telemetry and correctness summary for the completed rebuild.
    pub report: RebuildReport,
}

/// Internal payload produced by the background rebuild worker thread.
pub(super) struct BackgroundRebuildOutput {
    pub(super) filter: AerisFilter,
    pub(super) dest_reverse_map: Arc<dyn ReverseMapStore>,
    pub(super) fp_log: Arc<FalsePositiveLog>,
    pub(super) loaded_keys: BTreeSet<Key>,
    pub(super) fp_snapshot_seq: u64,
    pub(super) fp_replay_initial: FpReplayReport,
    pub(super) auto_expansions: usize,
    pub(super) observed_bulk_load_speed: f64,
    pub(super) throttle_metrics: ThrottleMetrics,
    pub(super) reason: RebuildReason,
}

/// Active background rebuild worker handle tracked by [`AsyncRebuildCoordinator`].
pub(super) struct InFlightRebuild {
    reason: RebuildReason,
    done: Arc<AtomicBool>,
    handle: JoinHandle<AerisResult<BackgroundRebuildOutput>>,
    started_at: Instant,
}

/// RAII drop guard ensuring `done` flips to `true` even if the background thread
/// exits early via error or unwind.
struct CompletionSignal(Arc<AtomicBool>);

impl Drop for CompletionSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Lifecycle state machine stored inside [`AsyncRebuildCoordinator::in_flight`].
pub(super) enum RebuildSlot {
    /// No background rebuild or staged delta drain is active.
    Idle,
    /// Background thread `"aeris-bg-rebuild"` is executing or waiting to be joined.
    Running(InFlightRebuild),
    /// Background thread has been joined into a [`StagedRebuild`] performing
    /// two-phase incremental delta draining before atomic swap.
    Draining,
}

/// Coordinates non-overlapping asynchronous rebuilds on a dedicated worker thread
/// (`"aeris-bg-rebuild"`), pairing throttled Reverse Map scans with [`DeltaBuffer`]
/// mutation capture and monotonic [`FalsePositiveLog`] replay.
pub struct AsyncRebuildCoordinator {
    delta_buffer: Arc<DeltaBuffer>,
    in_flight: Arc<Mutex<RebuildSlot>>,
}

impl Default for AsyncRebuildCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncRebuildCoordinator {
    /// Creates an idle [`AsyncRebuildCoordinator`] with a fresh inactive [`DeltaBuffer`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            delta_buffer: Arc::new(DeltaBuffer::new()),
            in_flight: Arc::new(Mutex::new(RebuildSlot::Idle)),
        }
    }

    /// Returns a shared handle to the coordinator's [`DeltaBuffer`].
    #[must_use]
    pub fn delta_buffer(&self) -> Arc<DeltaBuffer> {
        Arc::clone(&self.delta_buffer)
    }

    /// Returns `true` if a background rebuild is currently running or draining deltas.
    #[must_use]
    pub fn is_in_progress(&self) -> bool {
        !matches!(*self.in_flight.lock(), RebuildSlot::Idle)
    }

    /// Returns `true` if an in-flight background worker has completed its background
    /// phase and is ready for [`Self::prepare_finish`] / [`Self::finish_and_drain_deltas`]
    /// without long blocking.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(
            *self.in_flight.lock(),
            RebuildSlot::Running(ref job) if job.done.load(Ordering::Acquire)
        )
    }

    /// Launches a background rebuild constructing `F_new` under `target_config`
    /// from `source_reverse_map` into `dest_reverse_map`.
    ///
    /// Immediately activates [`DeltaBuffer`] before spawning `"aeris-bg-rebuild"` so
    /// concurrent foreground mutations are captured without loss.
    ///
    /// # Errors
    /// - Returns [`AerisError::RebuildInProgress`] if a rebuild is already in flight.
    /// - Returns [`AerisError::InvalidConfig`] if `target_config` or `throttle_config` is invalid.
    /// - Returns [`AerisError::InternalInvariant`] if the OS refuses to spawn the worker thread.
    pub fn start_rebuild(
        &self,
        source_reverse_map: Arc<dyn ReverseMapStore>,
        dest_reverse_map: Arc<dyn ReverseMapStore>,
        fp_log: Arc<FalsePositiveLog>,
        target_config: FilterConfig,
        reason: RebuildReason,
        throttle_config: IoThrottleConfig,
    ) -> AerisResult<()> {
        target_config.validate()?;
        throttle_config.validate()?;

        let mut slot = self.in_flight.lock();
        if !matches!(*slot, RebuildSlot::Idle) {
            return Err(AerisError::RebuildInProgress);
        }

        // Activate DeltaBuffer BEFORE spawning the worker so every mutation arriving
        // concurrently with or after the Reverse Map snapshot is captured.
        self.delta_buffer.activate();
        let started_at = Instant::now();
        info!(
            ?reason,
            target_q = target_config.q,
            target_r = target_config.r,
            target_m = target_config.m,
            max_keys_per_sec = throttle_config.max_keys_per_sec,
            batch_size = throttle_config.batch_size,
            "starting asynchronous background rebuild"
        );

        let done = Arc::new(AtomicBool::new(false));
        match spawn_rebuild_worker(
            Arc::clone(&done),
            source_reverse_map,
            dest_reverse_map,
            fp_log,
            target_config,
            reason,
            throttle_config,
        ) {
            Ok(handle) => {
                *slot = RebuildSlot::Running(InFlightRebuild {
                    reason,
                    done,
                    handle,
                    started_at,
                });
                Ok(())
            }
            Err(err) => {
                self.delta_buffer.deactivate();
                Err(err)
            }
        }
    }

    /// Joins the active background rebuild worker off-lock and transitions the
    /// coordinator into the [`StagedRebuild`] two-phase draining stage while keeping
    /// [`DeltaBuffer`] active for concurrent writers.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if no rebuild worker is running or
    /// if the worker thread panicked; propagates any background worker error.
    pub fn prepare_finish(&self) -> AerisResult<StagedRebuild> {
        let job = {
            let mut slot = self.in_flight.lock();
            match std::mem::replace(&mut *slot, RebuildSlot::Draining) {
                RebuildSlot::Running(job) => job,
                prev => {
                    *slot = prev;
                    return Err(AerisError::InternalInvariant(
                        "prepare_finish called with no active rebuild worker".to_string(),
                    ));
                }
            }
        };

        let drain_guard =
            DrainGuard::new(Arc::clone(&self.in_flight), Arc::clone(&self.delta_buffer));

        let bg_result = match job.handle.join() {
            Ok(res) => res,
            Err(_) => {
                error!(reason = ?job.reason, "background rebuild thread panicked");
                return Err(AerisError::InternalInvariant(
                    "background rebuild worker thread panicked".to_string(),
                ));
            }
        };

        let bg = bg_result.inspect_err(|err| {
            error!(reason = ?job.reason, error = %err, "background rebuild worker failed");
        })?;

        let auto_expansions = bg.auto_expansions;
        Ok(StagedRebuild {
            bg,
            started_at: job.started_at,
            delta_buffer: Arc::clone(&self.delta_buffer),
            drain_guard,
            delta_inserts_applied: 0,
            delta_deletes_applied: 0,
            auto_expansions,
            all_inserted_delta_keys: Vec::new(),
            unverified_inserted_keys: Vec::new(),
            pre_drain_passes: 0,
            pre_drained_mutations: 0,
        })
    }

    /// Joins the active background rebuild worker, runs incremental off-lock delta
    /// pre-draining, atomically drains and deactivates [`DeltaBuffer`], replays
    /// catch-up [`FalsePositiveLog`] entries, and verifies zero false negatives for
    /// all live delta-inserted keys.
    ///
    /// `verify_empty` MUST return `Ok(true)` iff `range` currently contains zero live
    /// keys in the authoritative store (including all drained delta mutations).
    ///
    /// # Errors
    /// - Returns [`AerisError::InternalInvariant`] if no rebuild is in progress, if the
    ///   worker panicked, or if a post-drain false negative is detected.
    /// - Propagates any storage, filter, or FP log error encountered during rebuild or drain.
    pub fn finish_and_drain_deltas<F>(&self, verify_empty: F) -> AerisResult<CompletedRebuild>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let mut staged = self.prepare_finish()?;
        staged.pre_drain_until_small(
            DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
            DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
            DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
        )?;
        staged.verify_pre_drained_keys()?;
        staged.finish_tail(verify_empty)
    }
}

fn spawn_rebuild_worker(
    done: Arc<AtomicBool>,
    source_reverse_map: Arc<dyn ReverseMapStore>,
    dest_reverse_map: Arc<dyn ReverseMapStore>,
    fp_log: Arc<FalsePositiveLog>,
    target_config: FilterConfig,
    reason: RebuildReason,
    throttle_config: IoThrottleConfig,
) -> AerisResult<JoinHandle<AerisResult<BackgroundRebuildOutput>>> {
    thread::Builder::new()
        .name(BG_REBUILD_THREAD_NAME.to_string())
        .spawn(move || {
            let _signal = CompletionSignal(done);
            execute_background_rebuild(
                source_reverse_map,
                dest_reverse_map,
                fp_log,
                target_config,
                reason,
                throttle_config,
            )
        })
        .map_err(|err| {
            error!(error = %err, "failed to spawn background rebuild thread");
            AerisError::InternalInvariant(format!(
                "failed to spawn '{BG_REBUILD_THREAD_NAME}' thread: {err}"
            ))
        })
}

/// Core execution body for the `"aeris-bg-rebuild"` background thread.
fn execute_background_rebuild(
    source_reverse_map: Arc<dyn ReverseMapStore>,
    dest_reverse_map: Arc<dyn ReverseMapStore>,
    fp_log: Arc<FalsePositiveLog>,
    target_config: FilterConfig,
    reason: RebuildReason,
    throttle_config: IoThrottleConfig,
) -> AerisResult<BackgroundRebuildOutput> {
    let phase_start = Instant::now();
    let (loaded_vec, throttle_metrics) =
        stream_source_keys_throttled(source_reverse_map.as_ref(), throttle_config)?;
    let loaded_keys: BTreeSet<Key> = loaded_vec.iter().copied().collect();

    let (mut new_filter, mut auto_expansions, observed_bulk_load_speed) =
        bulk_load_with_auto_expand(target_config, &loaded_vec, dest_reverse_map.as_ref())?;

    let fp_replay_initial =
        fp_log.replay_onto(&mut new_filter, dest_reverse_map.as_ref(), |r| {
            Ok(loaded_keys.range(r.start()..=r.end()).next().is_none())
        })?;
    auto_expansions += fp_replay_initial.expansions_during_replay;
    let fp_snapshot_seq = fp_replay_initial.last_replayed_seq;

    let duration_ns = phase_start.elapsed().as_nanos() as u64;
    info!(
        ?reason,
        keys_loaded = loaded_vec.len(),
        fp_collisions_resolved = fp_replay_initial.collisions_resolved,
        auto_expansions,
        observed_bulk_load_speed,
        throttle_events = throttle_metrics.throttle_events,
        duration_ns,
        "finished background rebuild worker phase"
    );

    Ok(BackgroundRebuildOutput {
        filter: new_filter,
        dest_reverse_map,
        fp_log,
        loaded_keys,
        fp_snapshot_seq,
        fp_replay_initial,
        auto_expansions,
        observed_bulk_load_speed,
        throttle_metrics,
        reason,
    })
}

fn stream_source_keys_throttled(
    source_reverse_map: &dyn ReverseMapStore,
    throttle_config: IoThrottleConfig,
) -> AerisResult<(Vec<Key>, ThrottleMetrics)> {
    let mut throttler = IoThrottler::new(throttle_config)?;
    let batch_size = throttler.batch_size() as u64;
    let mut loaded_vec = Vec::new();
    let mut batch_count = 0u64;

    for key_res in source_reverse_map.iter_all_keys()? {
        loaded_vec.push(key_res?);
        batch_count += 1;
        if batch_count >= batch_size {
            throttler.pace(batch_count);
            batch_count = 0;
        }
    }
    if batch_count > 0 {
        throttler.pace(batch_count);
    }

    Ok((loaded_vec, throttler.metrics()))
}

pub(super) fn bulk_load_with_auto_expand(
    mut cfg: FilterConfig,
    keys: &[Key],
    dest_reverse_map: &dyn ReverseMapStore,
) -> AerisResult<(AerisFilter, usize, f64)> {
    let bulk_start = Instant::now();
    let mut auto_expansions = 0usize;
    let filter = loop {
        match AerisFilter::bulk_load(cfg.clone(), keys, dest_reverse_map) {
            Ok(f) => break f,
            Err(AerisError::FilterFull { .. })
                if auto_expansions < usize::from(MAX_QUOTIENT_BITS) =>
            {
                cfg = cfg.expanded()?;
                auto_expansions += 1;
                debug!(
                    new_q = cfg.q,
                    keys = keys.len(),
                    "auto-expanded target_config during bulk_load"
                );
            }
            Err(err) => return Err(err),
        }
    };
    let bulk_elapsed_secs = bulk_start
        .elapsed()
        .as_secs_f64()
        .max(MIN_BULK_LOAD_ELAPSED_SECS);
    let observed_speed = (keys.len().max(1) as f64) / bulk_elapsed_secs;
    Ok((filter, auto_expansions, observed_speed))
}
