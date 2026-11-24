//! # `aeris-control`
//!
//! Deterministic Control Plane (Phase 3 / Milestone M3) for the **Aeris-Based
//! Optimized Range Filtering System**, providing:
//! - [`SlidingWindowEstimator`] (`WorkloadEstimator`): bounded sliding-window
//!   quantile estimator with breakdown-50% median-anchored outlier clipping,
//!   hysteresis + cooldown anti-thrashing, sliding-window FPR, and EWMA tracking
//!   for ingestion rate, bulk-load speed, and query arrival rate.
//! - [`DiscreteCostOptimizer`] (`CostOptimizer`): zero-allocation `O(max_m)`
//!   analytical optimizer evaluating all valid discrete `m = log2(P)` candidates
//!   under fixed RAM payload budget `r + m = C`, monotonic remainder floor
//!   `r >= r_min(epsilon)`, hysteresis rebuild gating, and direct-scan filter
//!   bypass thresholding.
//! - [`ProactiveTrigger`]: pure deterministic EWMA proactive expansion trigger
//!   (`T_full <= T_rebuild * margin` or `alpha >= hard_expand_alpha`) and
//!   sustained low-load contraction state machine (`q -> q - 1`).
//! - [`FalsePositiveLog`]: persistent 40-byte CRC32-framed, deduplicating,
//!   bounded append-only False-Positive Log with torn-tail recovery, atomic
//!   rotation, and monotonic replay (`replay_onto` / `replay_since`).
//! - [`AsyncRebuildCoordinator`], [`DeltaBuffer`], [`IoThrottler`], and
//!   [`ControlledAerisEngine`]: zero-downtime asynchronous background rebuilds
//!   pairing token-bucket I/O throttling, FIFO delta buffering, idempotent
//!   delta drain, and 100% monotonic FP Log replay over real `AerisFilter` +
//!   `RocksDbReverseMap` + `ReferenceKvStore`.
//!
//! # Safety Policy
//! This crate enforces `#![forbid(unsafe_code)]` and `#![deny(missing_docs)]`.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod estimator;
pub mod fp_log;
pub mod optimizer;
pub mod rebuild;
pub mod trigger;

pub use aeris_core;
pub use aeris_storage;
pub use aeris_types;

pub use estimator::{
    EstimatorConfig, SlidingWindowEstimator, DEFAULT_COOLDOWN_QUERIES, DEFAULT_EWMA_ALPHA,
    DEFAULT_HYSTERESIS_RATIO, DEFAULT_MAX_CLIP_SPAN, DEFAULT_MIN_SAMPLES_FOR_CLIPPING,
    DEFAULT_OUTLIER_FLOOR_SPAN, DEFAULT_OUTLIER_MEDIAN_MULTIPLIER,
    DEFAULT_REQUIRED_SUSTAINED_EVALS, DEFAULT_SAMPLE_STRIDE, DEFAULT_WINDOW_SIZE,
};
pub use fp_log::{
    decode_fp_frame, encode_fp_frame, FalsePositiveLog, FpLogConfig, FpReplayReport,
    DEFAULT_FP_LOG_MAX_ENTRIES, FP_LOG_FRAME_LEN, FP_LOG_MAGIC,
};
pub use optimizer::{CandidateCost, DiscreteCostOptimizer};
pub use rebuild::{
    AsyncRebuildCoordinator, CompletedRebuild, ControlPlaneDecision, ControlledAerisEngine,
    ControlledEngineConfig, DeltaBuffer, DeltaMutation, IoThrottleConfig, IoThrottler,
    QueryOutcome, RebuildReason, RebuildReport, StagedRebuild, ThrottleMetrics,
    BG_REBUILD_THREAD_NAME, DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE, DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES,
    DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD, DEFAULT_IO_THROTTLE_BATCH_SIZE,
};
pub use trigger::{
    ProactiveTrigger, ProactiveTriggerConfig, TriggerAction, TriggerEvaluation, TriggerInput,
    DEFAULT_CONTRACTION_ALPHA_THRESHOLD, DEFAULT_CONTRACTION_SUSTAINED_CHECKS,
    DEFAULT_HARD_EXPAND_ALPHA, DEFAULT_MAX_POST_CONTRACTION_ALPHA, DEFAULT_PROACTIVE_MARGIN,
};
