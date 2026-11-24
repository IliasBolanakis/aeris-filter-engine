//! # `aeris-system`
//!
//! Umbrella crate unifying all architectural planes of the **Aeris-Based
//! Optimized Range Filtering System**:
//! - [`aeris_types`]: Shared domain types, mathematical invariants, and traits.
//! - [`aeris_core`]: Data Plane Aeris Quotient Filter in RAM.
//! - [`aeris_storage`]: Storage Plane RocksDB Reverse Map and coupled WAL.
//! - [`aeris_control`]: Deterministic Control Plane, async rebuild, and FP Log replay.
//! - [`aeris_scale`]: Multi-core chunk locking, epoch RCU swap, and Range Sharding.
//! - [`aeris_sidecar`]: Optional Phase 5 gRPC sidecar, `SafetyGuardrail`, and fallback.

#![deny(missing_docs)]

pub use aeris_control;
pub use aeris_core;
pub use aeris_scale;
pub use aeris_sidecar;
pub use aeris_storage;
pub use aeris_types;
