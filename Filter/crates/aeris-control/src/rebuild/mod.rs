//! Asynchronous background rebuild coordinator, concurrency delta buffer,
//! token-bucket I/O throttler, and unified [`ControlledAerisEngine`] facade.
//!
//! # Zero-Downtime Dynamic Rebuild Protocol (`F_old -> F_new`)
//! 1. Activate [`DeltaBuffer`] so concurrent `Insert`/`Delete` mutations arriving
//!    on `F_old` are durably buffered in FIFO arrival order.
//! 2. Spawn dedicated background thread (`"aeris-bg-rebuild"`) that streams all
//!    live keys from `source_reverse_map.iter_all_keys()` in batches paced by
//!    [`IoThrottler`].
//! 3. Bulk-load `F_new` (`AerisFilter::bulk_load`) under `target_config` into a
//!    fresh destination [`aeris_types::ReverseMapStore`], automatically expanding
//!    `q -> q + 1` if the live key set exceeds `target_config.max_load_factor`.
//! 4. Replay [`crate::fp_log::FalsePositiveLog`] onto `F_new` so every previously
//!    observed false positive is monotonically eliminated off the foreground critical path.
//! 5. On [`AsyncRebuildCoordinator::finish_and_drain_deltas`], join the worker,
//!    atomically drain + deactivate [`DeltaBuffer`], apply deltas idempotently
//!    against the loaded snapshot key set (preventing duplicate mementos across
//!    snapshot/delta overlap), execute catch-up [`crate::fp_log::FalsePositiveLog`]
//!    replay, and verify zero false negatives for all live delta-inserted keys before
//!    returning [`CompletedRebuild`] for atomic `F_old -> F_new` swap.
//!
//! # Lock Ordering
//! `AsyncRebuildCoordinator::in_flight` -> `DeltaBuffer::inner` (leaf) and
//! `FalsePositiveLog::inner` (leaf). Neither leaf lock is ever held while
//! invoking [`aeris_core::AerisFilter`] or [`aeris_types::ReverseMapStore`] operations.

mod config;
mod coordinator;
mod delta_buffer;
mod engine;
mod engine_lifecycle;
mod staged;
mod throttle;

#[cfg(test)]
mod tests;

pub use config::{ControlPlaneDecision, ControlledEngineConfig, QueryOutcome};
pub use coordinator::{
    AsyncRebuildCoordinator, CompletedRebuild, RebuildReason, RebuildReport, BG_REBUILD_THREAD_NAME,
};
pub use delta_buffer::{
    DeltaBuffer, DeltaMutation, DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE,
    DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES, DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD,
};
pub use engine::ControlledAerisEngine;
pub use staged::StagedRebuild;
pub use throttle::{
    IoThrottleConfig, IoThrottler, ThrottleMetrics, DEFAULT_IO_THROTTLE_BATCH_SIZE,
};
