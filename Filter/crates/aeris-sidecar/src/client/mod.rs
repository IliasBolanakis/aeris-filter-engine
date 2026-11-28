//! Out-of-process gRPC client for the `ControlPlaneAdvisor` service.
//!
//! Enforces explicit per-attempt timeouts (`tokio::time::timeout`), bounded
//! retries with deterministic jitter on transient failures, preserved
//! `request_id` idempotency keys across retries, strict narrowing bounds checks
//! during protobuf-to-DTO conversion, and structured `tracing` observability.

mod circuit_breaker;
pub(crate) mod codec;
mod rpc;
#[cfg(test)]
mod tests;

pub use circuit_breaker::{CircuitBreakerState, CircuitBreakerStats};

use std::sync::atomic::{AtomicU64, Ordering};

use aeris_types::{
    AerisResult, Epoch, MemoryAllocationPlan, ShardTelemetrySnapshot, TelemetrySnapshot,
    TuningProposal,
};
use circuit_breaker::CircuitBreaker;
use codec::{proto_to_memory_plan, proto_to_tuning_proposal, shards_to_proto, telemetry_to_proto};
use tokio::runtime::Handle;
use tokio::sync::RwLock;
use tonic::transport::Channel;

use crate::proto;
use crate::proto::control_plane_advisor_client::ControlPlaneAdvisorClient;
use crate::transport::ClientConfig;

/// Health probe response returned by [`SidecarClient::health_check_async`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthStatus {
    /// `true` if the sidecar reported healthy status.
    pub healthy: bool,
    /// Model version string reported by the sidecar process.
    pub model_version: String,
    /// Sidecar process uptime in milliseconds.
    pub uptime_ms: u64,
}

/// Resilient async gRPC client (with explicit blocking bridges) communicating
/// with the out-of-process Control Plane advisory sidecar.
///
/// # Lock Ordering
/// `channel` (`tokio::sync::RwLock`) and `circuit_breaker` (`std::sync::Mutex`)
/// are independent leaf locks; neither is ever held while acquiring the other or
/// across `.await` points.
#[derive(Debug)]
pub struct SidecarClient {
    config: ClientConfig,
    channel: RwLock<Option<Channel>>,
    circuit_breaker: CircuitBreaker,
    request_counter: AtomicU64,
}

impl SidecarClient {
    /// Constructs a new [`SidecarClient`] after validating `config`.
    ///
    /// Connection establishment is deferred lazily to the first RPC so client
    /// construction never blocks and requires no active Tokio runtime.
    ///
    /// # Errors
    /// Returns [`aeris_types::AerisError::InvalidConfig`] if `config` fails validation.
    pub fn new(config: ClientConfig) -> AerisResult<Self> {
        config.validate()?;
        let circuit_breaker = CircuitBreaker::new(
            config.circuit_breaker_failure_threshold,
            config.circuit_breaker_cooldown,
        );
        Ok(Self {
            config,
            channel: RwLock::new(None),
            circuit_breaker,
            request_counter: AtomicU64::new(1),
        })
    }

    /// Returns the immutable transport and retry configuration of this client.
    #[inline]
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Fast, synchronous, non-blocking check returning `true` when the circuit
    /// breaker is currently `Open` (within cooldown) or already executing a
    /// `HalfOpen` probe, allowing callers to fall back in nanoseconds without
    /// crossing an async runtime boundary or touching any socket.
    #[inline]
    pub fn try_fast_short_circuit(&self) -> bool {
        self.circuit_breaker.try_fast_short_circuit()
    }

    /// Returns the current [`CircuitBreakerState`] of this client.
    #[inline]
    pub fn circuit_breaker_state(&self) -> CircuitBreakerState {
        self.circuit_breaker.state()
    }

    /// Returns a point-in-time [`CircuitBreakerStats`] snapshot for observability.
    #[inline]
    pub fn circuit_breaker_stats(&self) -> CircuitBreakerStats {
        self.circuit_breaker.stats()
    }

    /// Sends a macro-level `ProposeTuning` RPC with explicit per-attempt timeout,
    /// circuit-breaker protection, and bounded retries using a preserved
    /// `request_id` idempotency key.
    ///
    /// # Errors
    /// - [`aeris_types::AerisError::SidecarTimeout`] if all attempts exceed `rpc_timeout`.
    /// - [`aeris_types::AerisError::SidecarUnavailable`] if the circuit breaker is `Open`,
    ///   the sidecar is unreachable, or it returns an unrecoverable gRPC status.
    /// - [`aeris_types::AerisError::GuardrailRejected`] if returned `uint32` bit widths
    ///   exceed `u8::MAX`.
    pub async fn propose_tuning_async(
        &self,
        telemetry: &TelemetrySnapshot,
    ) -> AerisResult<TuningProposal> {
        let seq = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let request_id = if telemetry.request_id.is_empty() {
            format!("tune-s{}-e{}-{seq}", telemetry.shard_id, telemetry.epoch)
        } else {
            telemetry.request_id.clone()
        };
        let proto_req = telemetry_to_proto(&request_id, telemetry);

        let (resp, attempt, latency_ns) = self
            .execute_rpc_with_retries(
                "ProposeTuning",
                &request_id,
                self.config.max_retries,
                seq,
                |channel| {
                    let req = proto_req.clone();
                    async move {
                        let mut client = ControlPlaneAdvisorClient::new(channel);
                        client.propose_tuning(req).await
                    }
                },
            )
            .await?;

        let proposal = proto_to_tuning_proposal(&request_id, resp.into_inner())?;
        tracing::debug!(
            shard_id = telemetry.shard_id,
            epoch = telemetry.epoch,
            request_id = %request_id,
            attempt,
            latency_ns,
            model_version = %proposal.model_version,
            "ProposeTuning RPC succeeded"
        );
        Ok(proposal)
    }

    /// Sends a multi-tenant `AllocateMemory` RPC with explicit per-attempt timeout,
    /// circuit-breaker protection, and bounded retries preserving the same
    /// `request_id` idempotency key.
    ///
    /// # Errors
    /// - [`aeris_types::AerisError::SidecarTimeout`] if all attempts exceed `rpc_timeout`.
    /// - [`aeris_types::AerisError::SidecarUnavailable`] if the circuit breaker is `Open`,
    ///   the sidecar is unreachable, or fails.
    /// - [`aeris_types::AerisError::GuardrailRejected`] if returned bit widths exceed `u8::MAX`.
    pub async fn allocate_memory_async(
        &self,
        request_id: String,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan> {
        let seq = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let effective_id = if request_id.is_empty() {
            format!("mem-e{epoch}-{seq}")
        } else {
            request_id
        };
        let proto_req = shards_to_proto(&effective_id, epoch, total_budget_bytes, shards);

        let (resp, attempt, latency_ns) = self
            .execute_rpc_with_retries(
                "AllocateMemory",
                &effective_id,
                self.config.max_retries,
                seq,
                |channel| {
                    let req = proto_req.clone();
                    async move {
                        let mut client = ControlPlaneAdvisorClient::new(channel);
                        client.allocate_memory(req).await
                    }
                },
            )
            .await?;

        let plan = proto_to_memory_plan(&effective_id, resp.into_inner())?;
        tracing::debug!(
            epoch,
            request_id = %effective_id,
            shard_count = shards.len(),
            attempt,
            latency_ns,
            model_version = %plan.model_version,
            "AllocateMemory RPC succeeded"
        );
        Ok(plan)
    }

    /// Probes sidecar liveness and model version with a single bounded timeout
    /// and circuit-breaker protection (no retries so liveness probes return promptly).
    ///
    /// # Errors
    /// Returns [`aeris_types::AerisError::SidecarTimeout`] or
    /// [`aeris_types::AerisError::SidecarUnavailable`].
    pub async fn health_check_async(&self) -> AerisResult<HealthStatus> {
        let (resp, _attempt, latency_ns) = self
            .execute_rpc_with_retries("HealthCheck", "health", 0, 0, |channel| async move {
                let mut client = ControlPlaneAdvisorClient::new(channel);
                client.health_check(proto::HealthCheckRequest {}).await
            })
            .await?;

        let inner = resp.into_inner();
        tracing::debug!(
            healthy = inner.healthy,
            model_version = %inner.model_version,
            uptime_ms = inner.uptime_ms,
            latency_ns,
            "HealthCheck RPC succeeded"
        );
        Ok(HealthStatus {
            healthy: inner.healthy,
            model_version: inner.model_version,
            uptime_ms: inner.uptime_ms,
        })
    }

    /// Synchronous bridge for [`Self::propose_tuning_async`] executing on `handle`.
    ///
    /// # Panics
    /// Panics if invoked directly from within the same Tokio runtime owned by `handle`.
    /// Prefer [`crate::advisor::GuardedTuningAdvisor`] for safe sync/async bridging.
    ///
    /// # Errors
    /// Propagates any error from [`Self::propose_tuning_async`].
    pub fn propose_tuning_blocking(
        &self,
        handle: &Handle,
        telemetry: &TelemetrySnapshot,
    ) -> AerisResult<TuningProposal> {
        handle.block_on(self.propose_tuning_async(telemetry))
    }

    /// Synchronous bridge for [`Self::allocate_memory_async`] executing on `handle`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::allocate_memory_async`].
    pub fn allocate_memory_blocking(
        &self,
        handle: &Handle,
        request_id: String,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan> {
        handle.block_on(self.allocate_memory_async(request_id, epoch, total_budget_bytes, shards))
    }

    /// Synchronous bridge for [`Self::health_check_async`] executing on `handle`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::health_check_async`].
    pub fn health_check_blocking(&self, handle: &Handle) -> AerisResult<HealthStatus> {
        handle.block_on(self.health_check_async())
    }

    /// Returns a cached gRPC [`Channel`] or establishes a fresh connection.
    async fn get_or_connect_channel(&self) -> AerisResult<Channel> {
        {
            let guard = self.channel.read().await;
            if let Some(ch) = guard.as_ref() {
                return Ok(ch.clone());
            }
        }

        let connected = self
            .config
            .endpoint
            .connect(self.config.rpc_timeout)
            .await?;
        let mut guard = self.channel.write().await;
        *guard = Some(connected.clone());
        Ok(connected)
    }

    /// Drops any cached gRPC channel so the subsequent attempt reconnects cleanly.
    async fn invalidate_channel(&self) {
        let mut guard = self.channel.write().await;
        *guard = None;
    }
}
