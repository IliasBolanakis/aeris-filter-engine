//! # `aeris-sidecar`
//!
//! Optional Phase 5 Control Plane extension providing an out-of-process gRPC
//! client/server over Unix Domain Sockets (UDS on Unix; loopback TCP on Windows),
//! a deterministic Rust [`SafetyGuardrail`], and automatic fallback to a native
//! analytical [`aeris_types::CostOptimizer`] with zero Data Plane hot-path overhead.
//!
//! # Architectural Boundaries
//! - [`guardrail::SafetyGuardrail`] is pure, deterministic, and zero-I/O.
//! - [`transport::SidecarEndpoint`] and [`transport::ClientConfig`] isolate local
//!   IPC addressing, timeouts, and deterministic jittered retry backoff.
//! - [`client::SidecarClient`] handles async/sync gRPC transport, channel
//!   lifecycle, idempotent retries, and protobuf DTO translation.
//! - [`advisor::GuardedTuningAdvisor`] implements [`aeris_types::TuningAdvisor`]
//!   and guarantees every emitted proposal or memory plan satisfies
//!   [`aeris_types::SafetyBounds`], transparently falling back to the native
//!   optimizer on any transport, timeout, crash, or guardrail failure.
//! - [`server::ReferenceSidecarServer`] provides a fault-injectable Rust gRPC
//!   server for deterministic testing.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

/// Generated Protocol Buffers and `tonic` gRPC definitions for `aeris.control.v1`.
#[allow(missing_docs, clippy::result_large_err)]
pub mod proto {
    tonic::include_proto!("aeris.control.v1");
}

pub mod advisor;
pub mod client;
pub mod guardrail;
pub mod server;
pub mod transport;

pub use advisor::{
    GuardedTuningAdvisor, NativeFallbackOptimizer, SidecarMetrics, SidecarMetricsSnapshot,
    NATIVE_FALLBACK_MODEL_VERSION,
};
pub use aeris_types;
pub use client::{CircuitBreakerState, CircuitBreakerStats, HealthStatus, SidecarClient};
pub use guardrail::SafetyGuardrail;
pub use server::{
    ReferenceSidecarServer, RunningServer, ServerBehavior, REFERENCE_SERVER_MODEL_VERSION,
};
pub use transport::{
    compute_retry_delay, ClientConfig, SidecarEndpoint, DEFAULT_BASE_BACKOFF_MS,
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN, DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_MAX_RETRIES, DEFAULT_RPC_TIMEOUT, MAX_RETRY_BACKOFF_MS,
};
