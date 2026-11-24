//! Out-of-process gRPC client for the `ControlPlaneAdvisor` service.
//!
//! Enforces explicit per-attempt timeouts (`tokio::time::timeout`), bounded
//! retries with deterministic jitter on transient failures, preserved
//! `request_id` idempotency keys across retries, strict narrowing bounds checks
//! during protobuf-to-DTO conversion, and structured `tracing` observability.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aeris_types::{
    AerisError, AerisResult, Epoch, MemoryAllocationPlan, ShardAllocationDecision,
    ShardTelemetrySnapshot, TelemetrySnapshot, TuningProposal,
};
use tokio::runtime::Handle;
use tokio::sync::RwLock;
use tonic::transport::Channel;
use tonic::Code;

use crate::proto;
use crate::proto::control_plane_advisor_client::ControlPlaneAdvisorClient;
use crate::transport::{compute_retry_delay, ClientConfig};

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

/// Lifecycle state of the [`SidecarClient`] transport circuit breaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Normal operation: RPCs are permitted and consecutive failures are counted.
    Closed,
    /// Sidecar recently failed repeatedly: RPCs short-circuit immediately to
    /// native fallback without touching any socket until `cooldown` elapses.
    Open,
    /// Cooldown elapsed: a single probe RPC is permitted to verify recovery.
    /// Success transitions back to [`Self::Closed`]; failure re-opens [`Self::Open`].
    HalfOpen,
}

/// Immutable observability snapshot of the [`SidecarClient`] circuit breaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitBreakerStats {
    /// Current circuit breaker state (`Closed`, `Open`, or `HalfOpen`).
    pub state: CircuitBreakerState,
    /// Current streak of consecutive transport/timeout RPC failures.
    pub consecutive_failures: u32,
    /// Cumulative number of transitions into [`CircuitBreakerState::Open`].
    pub trip_count: u64,
    /// Cumulative number of calls short-circuited without attempting an RPC.
    pub short_circuit_count: u64,
}

/// Pure inner state machine for the sidecar transport circuit breaker.
#[derive(Debug)]
struct CircuitBreakerInner {
    state: CircuitBreakerState,
    consecutive_failures: u32,
    opened_at: Option<Instant>,
    half_open_probe_in_flight: bool,
    trip_count: u64,
    short_circuit_count: u64,
}

impl CircuitBreakerInner {
    const fn new() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            consecutive_failures: 0,
            opened_at: None,
            half_open_probe_in_flight: false,
            trip_count: 0,
            short_circuit_count: 0,
        }
    }

    /// Returns `true` if a caller should immediately short-circuit to native fallback
    /// without starting an RPC or consuming a `HalfOpen` probe.
    fn try_fast_short_circuit_at(&mut self, now: Instant, cooldown: Duration) -> bool {
        match self.state {
            CircuitBreakerState::Closed => false,
            CircuitBreakerState::Open => {
                let elapsed = self
                    .opened_at
                    .map_or(cooldown, |t| now.saturating_duration_since(t));
                if elapsed < cooldown {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                if self.half_open_probe_in_flight {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Attempts to acquire permission to execute an RPC at `now`.
    ///
    /// Returns `Some(is_half_open_probe)` if permitted, or `None` if short-circuited.
    fn try_acquire_at(&mut self, now: Instant, cooldown: Duration) -> Option<bool> {
        match self.state {
            CircuitBreakerState::Closed => Some(false),
            CircuitBreakerState::Open => {
                let elapsed = self
                    .opened_at
                    .map_or(cooldown, |t| now.saturating_duration_since(t));
                if elapsed >= cooldown {
                    self.state = CircuitBreakerState::HalfOpen;
                    self.half_open_probe_in_flight = true;
                    tracing::info!(
                        cooldown_ms = cooldown.as_millis() as u64,
                        trip_count = self.trip_count,
                        "sidecar circuit breaker transitioning Open -> HalfOpen for recovery probe"
                    );
                    Some(true)
                } else {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    None
                }
            }
            CircuitBreakerState::HalfOpen => {
                if !self.half_open_probe_in_flight {
                    self.half_open_probe_in_flight = true;
                    Some(true)
                } else {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    None
                }
            }
        }
    }

    /// Records a successful transport round-trip at `now`, closing the breaker.
    fn record_success_at(&mut self, _now: Instant) {
        let prev = self.state;
        self.state = CircuitBreakerState::Closed;
        self.consecutive_failures = 0;
        self.opened_at = None;
        self.half_open_probe_in_flight = false;
        if prev != CircuitBreakerState::Closed {
            tracing::info!(
                previous_state = ?prev,
                trip_count = self.trip_count,
                "sidecar circuit breaker recovered and closed"
            );
        }
    }

    /// Records an RPC-level transport or timeout failure at `now`, tripping to `Open`
    /// once `consecutive_failures >= failure_threshold` or immediately on `HalfOpen` failure.
    fn record_transport_failure_at(&mut self, now: Instant, failure_threshold: u32) {
        self.half_open_probe_in_flight = false;
        match self.state {
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.opened_at = Some(now);
                self.trip_count = self.trip_count.saturating_add(1);
                tracing::warn!(
                    trip_count = self.trip_count,
                    "sidecar HalfOpen probe failed; re-opening circuit breaker"
                );
            }
            CircuitBreakerState::Closed => {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                if self.consecutive_failures >= failure_threshold {
                    self.state = CircuitBreakerState::Open;
                    self.opened_at = Some(now);
                    self.trip_count = self.trip_count.saturating_add(1);
                    tracing::warn!(
                        consecutive_failures = self.consecutive_failures,
                        failure_threshold,
                        trip_count = self.trip_count,
                        "sidecar circuit breaker tripped Closed -> Open"
                    );
                }
            }
            CircuitBreakerState::Open => {
                self.opened_at = Some(now);
            }
        }
    }

    /// Releases an in-flight `HalfOpen` probe slot if the caller was cancelled before completion.
    fn release_aborted_probe(&mut self) {
        if self.state == CircuitBreakerState::HalfOpen {
            self.half_open_probe_in_flight = false;
        }
    }

    /// Computes effective state at `now` without mutating internal counters.
    fn effective_state_at(&self, now: Instant, cooldown: Duration) -> CircuitBreakerState {
        match self.state {
            CircuitBreakerState::Open => {
                let elapsed = self
                    .opened_at
                    .map_or(cooldown, |t| now.saturating_duration_since(t));
                if elapsed >= cooldown {
                    CircuitBreakerState::HalfOpen
                } else {
                    CircuitBreakerState::Open
                }
            }
            other => other,
        }
    }

    fn stats_at(&self, now: Instant, cooldown: Duration) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.effective_state_at(now, cooldown),
            consecutive_failures: self.consecutive_failures,
            trip_count: self.trip_count,
            short_circuit_count: self.short_circuit_count,
        }
    }
}

/// Thread-safe, lock-brief circuit breaker guarding [`SidecarClient`] RPC paths.
///
/// # Lock Ordering
/// `inner` is a synchronous leaf `std::sync::Mutex` never held across `.await` points.
#[derive(Debug)]
struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    inner: Mutex<CircuitBreakerInner>,
}

impl CircuitBreaker {
    fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            inner: Mutex::new(CircuitBreakerInner::new()),
        }
    }

    fn try_fast_short_circuit(&self) -> bool {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.try_fast_short_circuit_at(Instant::now(), self.cooldown)
    }

    fn try_acquire(&self) -> Option<CircuitBreakerPermit<'_>> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner
            .try_acquire_at(Instant::now(), self.cooldown)
            .map(|is_half_open_probe| CircuitBreakerPermit {
                breaker: self,
                is_half_open_probe,
                completed: false,
            })
    }

    fn state(&self) -> CircuitBreakerState {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.effective_state_at(Instant::now(), self.cooldown)
    }

    fn stats(&self) -> CircuitBreakerStats {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.stats_at(Instant::now(), self.cooldown)
    }
}

/// RAII permit guaranteeing that a cancelled `HalfOpen` probe never strands the
/// circuit breaker in an un-probeable state.
struct CircuitBreakerPermit<'a> {
    breaker: &'a CircuitBreaker,
    is_half_open_probe: bool,
    completed: bool,
}

impl CircuitBreakerPermit<'_> {
    fn record_success(mut self) {
        self.completed = true;
        let mut inner = self
            .breaker
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.record_success_at(Instant::now());
    }

    fn record_transport_failure(mut self) {
        self.completed = true;
        let mut inner = self
            .breaker
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.record_transport_failure_at(Instant::now(), self.breaker.failure_threshold);
    }
}

impl Drop for CircuitBreakerPermit<'_> {
    fn drop(&mut self) {
        if !self.completed && self.is_half_open_probe {
            let mut inner = self
                .breaker
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.release_aborted_probe();
        }
    }
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
    /// Returns [`AerisError::InvalidConfig`] if `config` fails validation.
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
    /// - [`AerisError::SidecarTimeout`] if all attempts exceed `rpc_timeout`.
    /// - [`AerisError::SidecarUnavailable`] if the circuit breaker is `Open`, the
    ///   sidecar is unreachable, or it returns an unrecoverable gRPC status.
    /// - [`AerisError::GuardrailRejected`] if returned `uint32` bit widths exceed
    ///   `u8::MAX` (preventing silent integer truncation).
    pub async fn propose_tuning_async(
        &self,
        telemetry: &TelemetrySnapshot,
    ) -> AerisResult<TuningProposal> {
        let Some(permit) = self.circuit_breaker.try_acquire() else {
            return Err(AerisError::SidecarUnavailable(
                "sidecar circuit breaker open; short-circuiting ProposeTuning".to_string(),
            ));
        };

        let seq = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let request_id = if telemetry.request_id.is_empty() {
            format!("tune-s{}-e{}-{seq}", telemetry.shard_id, telemetry.epoch)
        } else {
            telemetry.request_id.clone()
        };
        let proto_req = telemetry_to_proto(&request_id, telemetry);
        let timeout_ms = self.config.rpc_timeout.as_millis() as u64;

        let mut last_err: Option<AerisError> = None;

        for attempt in 0..=self.config.max_retries {
            let start = Instant::now();
            let req = proto_req.clone();
            let rpc_future = async {
                let channel = self.get_or_connect_channel().await?;
                let mut client = ControlPlaneAdvisorClient::new(channel);
                let response = client.propose_tuning(req).await;
                Ok::<_, AerisError>(response)
            };

            let outcome = tokio::time::timeout(self.config.rpc_timeout, rpc_future).await;
            let latency_ns = start.elapsed().as_nanos() as u64;

            match outcome {
                Ok(Ok(Ok(resp))) => {
                    permit.record_success();
                    let inner = resp.into_inner();
                    let proposal = proto_to_tuning_proposal(&request_id, inner)?;
                    tracing::debug!(
                        shard_id = telemetry.shard_id,
                        epoch = telemetry.epoch,
                        request_id = %request_id,
                        attempt,
                        latency_ns,
                        model_version = %proposal.model_version,
                        "ProposeTuning RPC succeeded"
                    );
                    return Ok(proposal);
                }
                Ok(Ok(Err(status))) => {
                    let retryable = is_retryable_status(&status);
                    if should_reset_channel(&status) {
                        self.invalidate_channel().await;
                    }
                    tracing::warn!(
                        shard_id = telemetry.shard_id,
                        epoch = telemetry.epoch,
                        request_id = %request_id,
                        attempt,
                        latency_ns,
                        grpc_code = ?status.code(),
                        retryable,
                        error = %status.message(),
                        "ProposeTuning RPC returned gRPC error"
                    );
                    let err = map_grpc_status(status, timeout_ms);
                    if !retryable || attempt == self.config.max_retries {
                        permit.record_transport_failure();
                        return Err(err);
                    }
                    last_err = Some(err);
                }
                Ok(Err(connect_err)) => {
                    self.invalidate_channel().await;
                    tracing::warn!(
                        shard_id = telemetry.shard_id,
                        epoch = telemetry.epoch,
                        request_id = %request_id,
                        attempt,
                        latency_ns,
                        error = %connect_err,
                        "ProposeTuning failed to establish sidecar channel"
                    );
                    if attempt == self.config.max_retries {
                        permit.record_transport_failure();
                        return Err(connect_err);
                    }
                    last_err = Some(connect_err);
                }
                Err(_elapsed) => {
                    self.invalidate_channel().await;
                    tracing::warn!(
                        shard_id = telemetry.shard_id,
                        epoch = telemetry.epoch,
                        request_id = %request_id,
                        attempt,
                        latency_ns,
                        timeout_ms,
                        "ProposeTuning RPC timed out"
                    );
                    let err = AerisError::SidecarTimeout { timeout_ms };
                    if attempt == self.config.max_retries {
                        permit.record_transport_failure();
                        return Err(err);
                    }
                    last_err = Some(err);
                }
            }

            let delay = compute_retry_delay(
                self.config.base_backoff_ms,
                attempt,
                self.config.jitter_seed,
                seq,
            );
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
        }

        permit.record_transport_failure();
        Err(last_err.unwrap_or_else(|| {
            AerisError::SidecarUnavailable("ProposeTuning exhausted retry budget".to_string())
        }))
    }

    /// Sends a multi-tenant `AllocateMemory` RPC with explicit per-attempt timeout,
    /// circuit-breaker protection, and bounded retries preserving the same
    /// `request_id` idempotency key.
    ///
    /// # Errors
    /// - [`AerisError::SidecarTimeout`] if all attempts exceed `rpc_timeout`.
    /// - [`AerisError::SidecarUnavailable`] if the circuit breaker is `Open`, the
    ///   sidecar is unreachable, or fails.
    /// - [`AerisError::GuardrailRejected`] if returned bit widths exceed `u8::MAX`.
    pub async fn allocate_memory_async(
        &self,
        request_id: String,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan> {
        let Some(permit) = self.circuit_breaker.try_acquire() else {
            return Err(AerisError::SidecarUnavailable(
                "sidecar circuit breaker open; short-circuiting AllocateMemory".to_string(),
            ));
        };

        let seq = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let effective_id = if request_id.is_empty() {
            format!("mem-e{epoch}-{seq}")
        } else {
            request_id
        };
        let proto_req = shards_to_proto(&effective_id, epoch, total_budget_bytes, shards);
        let timeout_ms = self.config.rpc_timeout.as_millis() as u64;

        let mut last_err: Option<AerisError> = None;

        for attempt in 0..=self.config.max_retries {
            let start = Instant::now();
            let req = proto_req.clone();
            let rpc_future = async {
                let channel = self.get_or_connect_channel().await?;
                let mut client = ControlPlaneAdvisorClient::new(channel);
                let response = client.allocate_memory(req).await;
                Ok::<_, AerisError>(response)
            };

            let outcome = tokio::time::timeout(self.config.rpc_timeout, rpc_future).await;
            let latency_ns = start.elapsed().as_nanos() as u64;

            match outcome {
                Ok(Ok(Ok(resp))) => {
                    permit.record_success();
                    let inner = resp.into_inner();
                    let plan = proto_to_memory_plan(&effective_id, inner)?;
                    tracing::debug!(
                        epoch,
                        request_id = %effective_id,
                        shard_count = shards.len(),
                        attempt,
                        latency_ns,
                        model_version = %plan.model_version,
                        "AllocateMemory RPC succeeded"
                    );
                    return Ok(plan);
                }
                Ok(Ok(Err(status))) => {
                    let retryable = is_retryable_status(&status);
                    if should_reset_channel(&status) {
                        self.invalidate_channel().await;
                    }
                    tracing::warn!(
                        epoch,
                        request_id = %effective_id,
                        attempt,
                        latency_ns,
                        grpc_code = ?status.code(),
                        retryable,
                        error = %status.message(),
                        "AllocateMemory RPC returned gRPC error"
                    );
                    let err = map_grpc_status(status, timeout_ms);
                    if !retryable || attempt == self.config.max_retries {
                        permit.record_transport_failure();
                        return Err(err);
                    }
                    last_err = Some(err);
                }
                Ok(Err(connect_err)) => {
                    self.invalidate_channel().await;
                    tracing::warn!(
                        epoch,
                        request_id = %effective_id,
                        attempt,
                        latency_ns,
                        error = %connect_err,
                        "AllocateMemory failed to establish sidecar channel"
                    );
                    if attempt == self.config.max_retries {
                        permit.record_transport_failure();
                        return Err(connect_err);
                    }
                    last_err = Some(connect_err);
                }
                Err(_elapsed) => {
                    self.invalidate_channel().await;
                    tracing::warn!(
                        epoch,
                        request_id = %effective_id,
                        attempt,
                        latency_ns,
                        timeout_ms,
                        "AllocateMemory RPC timed out"
                    );
                    let err = AerisError::SidecarTimeout { timeout_ms };
                    if attempt == self.config.max_retries {
                        permit.record_transport_failure();
                        return Err(err);
                    }
                    last_err = Some(err);
                }
            }

            let delay = compute_retry_delay(
                self.config.base_backoff_ms,
                attempt,
                self.config.jitter_seed,
                seq,
            );
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
        }

        permit.record_transport_failure();
        Err(last_err.unwrap_or_else(|| {
            AerisError::SidecarUnavailable("AllocateMemory exhausted retry budget".to_string())
        }))
    }

    /// Probes sidecar liveness and model version with a single bounded timeout
    /// and circuit-breaker protection (no retries so liveness probes return promptly).
    ///
    /// # Errors
    /// Returns [`AerisError::SidecarTimeout`] or [`AerisError::SidecarUnavailable`].
    pub async fn health_check_async(&self) -> AerisResult<HealthStatus> {
        let Some(permit) = self.circuit_breaker.try_acquire() else {
            return Err(AerisError::SidecarUnavailable(
                "sidecar circuit breaker open; short-circuiting HealthCheck".to_string(),
            ));
        };

        let start = Instant::now();
        let timeout_ms = self.config.rpc_timeout.as_millis() as u64;
        let rpc_future = async {
            let channel = self.get_or_connect_channel().await?;
            let mut client = ControlPlaneAdvisorClient::new(channel);
            let response = client.health_check(proto::HealthCheckRequest {}).await;
            Ok::<_, AerisError>(response)
        };

        let outcome = tokio::time::timeout(self.config.rpc_timeout, rpc_future).await;
        let latency_ns = start.elapsed().as_nanos() as u64;

        match outcome {
            Ok(Ok(Ok(resp))) => {
                permit.record_success();
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
            Ok(Ok(Err(status))) => {
                if should_reset_channel(&status) {
                    self.invalidate_channel().await;
                }
                permit.record_transport_failure();
                tracing::warn!(
                    grpc_code = ?status.code(),
                    latency_ns,
                    error = %status.message(),
                    "HealthCheck RPC failed"
                );
                Err(map_grpc_status(status, timeout_ms))
            }
            Ok(Err(connect_err)) => {
                self.invalidate_channel().await;
                permit.record_transport_failure();
                Err(connect_err)
            }
            Err(_elapsed) => {
                self.invalidate_channel().await;
                permit.record_transport_failure();
                Err(AerisError::SidecarTimeout { timeout_ms })
            }
        }
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

/// Converts a domain [`TelemetrySnapshot`] into a protobuf [`proto::TuningRequest`].
pub(crate) fn telemetry_to_proto(
    request_id: &str,
    telemetry: &TelemetrySnapshot,
) -> proto::TuningRequest {
    proto::TuningRequest {
        request_id: request_id.to_string(),
        shard_id: telemetry.shard_id,
        epoch: telemetry.epoch,
        timestamp_ms: telemetry.timestamp_ms,
        current_q: u32::from(telemetry.current_q),
        current_r: u32::from(telemetry.current_r),
        current_m: u32::from(telemetry.current_m),
        payload_budget_c: u32::from(telemetry.payload_budget_c),
        target_epsilon: telemetry.target_epsilon,
        load_factor: telemetry.load_factor,
        ingestion_rate_ewma: telemetry.ingestion_rate_ewma,
        query_rate: telemetry.query_rate,
        observed_fpr: telemetry.observed_fpr,
        p95_query_len: telemetry.p95_query_len,
        p99_query_len: telemetry.p99_query_len,
        reverse_map_entries: telemetry.reverse_map_entries,
        bulk_load_speed: telemetry.bulk_load_speed,
        recent_query_len_series: telemetry.recent_query_len_series.clone(),
        recent_ingest_rate_series: telemetry.recent_ingest_rate_series.clone(),
    }
}

/// Converts a protobuf [`proto::TuningResponse`] into a domain [`TuningProposal`]
/// with strict `u32 -> u8` narrowing checks.
pub(crate) fn proto_to_tuning_proposal(
    expected_request_id: &str,
    resp: proto::TuningResponse,
) -> AerisResult<TuningProposal> {
    if !resp.request_id.is_empty() && resp.request_id != expected_request_id {
        return Err(AerisError::SidecarUnavailable(format!(
            "sidecar returned mismatched request_id '{}' (expected '{}')",
            resp.request_id, expected_request_id
        )));
    }
    let proposed_r = u8::try_from(resp.proposed_r).map_err(|_| AerisError::GuardrailRejected {
        reason: format!("proposed_r ({}) exceeds u8::MAX", resp.proposed_r),
    })?;
    let proposed_m = u8::try_from(resp.proposed_m).map_err(|_| AerisError::GuardrailRejected {
        reason: format!("proposed_m ({}) exceeds u8::MAX", resp.proposed_m),
    })?;

    Ok(TuningProposal {
        request_id: if resp.request_id.is_empty() {
            expected_request_id.to_string()
        } else {
            resp.request_id
        },
        proposed_r,
        proposed_m,
        proposed_target_alpha: resp.proposed_target_alpha,
        proposed_proactive_margin_k: resp.proposed_proactive_margin_k,
        confidence: resp.confidence,
        trigger_proactive_rebuild: resp.trigger_proactive_rebuild,
        model_version: resp.model_version,
        rationale: resp.rationale,
        forecast_peak_ingest_rate: resp.forecast_peak_ingest_rate,
    })
}

/// Converts shard telemetry slices into a protobuf [`proto::MemoryAllocationRequest`].
pub(crate) fn shards_to_proto(
    request_id: &str,
    epoch: Epoch,
    total_budget_bytes: u64,
    shards: &[ShardTelemetrySnapshot],
) -> proto::MemoryAllocationRequest {
    proto::MemoryAllocationRequest {
        request_id: request_id.to_string(),
        epoch,
        total_memory_budget_bytes: total_budget_bytes,
        shards: shards
            .iter()
            .map(|s| proto::ShardTelemetry {
                shard_id: s.shard_id,
                key_count: s.key_count,
                current_memory_bytes: s.current_memory_bytes,
                query_rate: s.query_rate,
                observed_fpr: s.observed_fpr,
                p95_query_len: s.p95_query_len,
                load_factor: s.load_factor,
                current_q: u32::from(s.current_q),
                current_r: u32::from(s.current_r),
                current_m: u32::from(s.current_m),
            })
            .collect(),
    }
}

/// Converts a protobuf [`proto::MemoryAllocationResponse`] into a domain
/// [`MemoryAllocationPlan`] with strict `u32 -> u8` narrowing checks.
pub(crate) fn proto_to_memory_plan(
    expected_request_id: &str,
    resp: proto::MemoryAllocationResponse,
) -> AerisResult<MemoryAllocationPlan> {
    if !resp.request_id.is_empty() && resp.request_id != expected_request_id {
        return Err(AerisError::SidecarUnavailable(format!(
            "sidecar returned mismatched request_id '{}' (expected '{}')",
            resp.request_id, expected_request_id
        )));
    }
    let mut allocations = Vec::with_capacity(resp.allocations.len());
    for alloc in resp.allocations {
        let recommended_q =
            u8::try_from(alloc.recommended_q).map_err(|_| AerisError::GuardrailRejected {
                reason: format!(
                    "shard {}: recommended_q ({}) exceeds u8::MAX",
                    alloc.shard_id, alloc.recommended_q
                ),
            })?;
        let recommended_r =
            u8::try_from(alloc.recommended_r).map_err(|_| AerisError::GuardrailRejected {
                reason: format!(
                    "shard {}: recommended_r ({}) exceeds u8::MAX",
                    alloc.shard_id, alloc.recommended_r
                ),
            })?;
        let recommended_m =
            u8::try_from(alloc.recommended_m).map_err(|_| AerisError::GuardrailRejected {
                reason: format!(
                    "shard {}: recommended_m ({}) exceeds u8::MAX",
                    alloc.shard_id, alloc.recommended_m
                ),
            })?;

        allocations.push(ShardAllocationDecision {
            shard_id: alloc.shard_id,
            allocated_memory_bytes: alloc.allocated_memory_bytes,
            recommended_q,
            recommended_r,
            recommended_m,
        });
    }

    Ok(MemoryAllocationPlan {
        request_id: if resp.request_id.is_empty() {
            expected_request_id.to_string()
        } else {
            resp.request_id
        },
        allocations,
        confidence: resp.confidence,
        model_version: resp.model_version,
    })
}

/// Classifies whether a gRPC [`tonic::Status`] is transient and eligible for bounded retry.
#[inline]
fn is_retryable_status(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        Code::Unavailable
            | Code::DeadlineExceeded
            | Code::Aborted
            | Code::ResourceExhausted
            | Code::Unknown
    )
}

/// Determines whether the cached transport channel should be discarded before reconnecting.
#[inline]
fn should_reset_channel(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        Code::Unavailable | Code::Unknown | Code::Cancelled | Code::DeadlineExceeded
    )
}

/// Maps a [`tonic::Status`] onto the unified [`AerisError`] hierarchy.
fn map_grpc_status(status: tonic::Status, timeout_ms: u64) -> AerisError {
    if status.code() == Code::DeadlineExceeded {
        AerisError::SidecarTimeout { timeout_ms }
    } else {
        AerisError::SidecarUnavailable(format!("gRPC {:?}: {}", status.code(), status.message()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proto_to_tuning_proposal_rejects_u8_overflow_and_request_id_mismatch() {
        let overflow_resp = proto::TuningResponse {
            request_id: "req-1".to_string(),
            proposed_r: 300,
            proposed_m: 4,
            proposed_target_alpha: 0.85,
            proposed_proactive_margin_k: 1.5,
            confidence: 0.9,
            trigger_proactive_rebuild: false,
            model_version: "m".to_string(),
            rationale: "r".to_string(),
            forecast_peak_ingest_rate: 100.0,
        };
        assert!(matches!(
            proto_to_tuning_proposal("req-1", overflow_resp),
            Err(AerisError::GuardrailRejected { .. })
        ));

        let mismatch_resp = proto::TuningResponse {
            request_id: "wrong-id".to_string(),
            proposed_r: 14,
            proposed_m: 6,
            proposed_target_alpha: 0.85,
            proposed_proactive_margin_k: 1.5,
            confidence: 0.9,
            trigger_proactive_rebuild: false,
            model_version: "m".to_string(),
            rationale: "r".to_string(),
            forecast_peak_ingest_rate: 100.0,
        };
        assert!(matches!(
            proto_to_tuning_proposal("req-1", mismatch_resp),
            Err(AerisError::SidecarUnavailable(_))
        ));
    }

    #[test]
    fn proto_to_memory_plan_rejects_u8_overflow_and_request_id_mismatch() {
        let overflow_plan = proto::MemoryAllocationResponse {
            request_id: "mem-1".to_string(),
            allocations: vec![proto::ShardAllocation {
                shard_id: 1,
                allocated_memory_bytes: 1024,
                recommended_q: 300,
                recommended_r: 14,
                recommended_m: 6,
            }],
            confidence: 0.9,
            model_version: "m".to_string(),
        };
        assert!(matches!(
            proto_to_memory_plan("mem-1", overflow_plan),
            Err(AerisError::GuardrailRejected { .. })
        ));

        let mismatch_plan = proto::MemoryAllocationResponse {
            request_id: "other-mem".to_string(),
            allocations: vec![],
            confidence: 0.9,
            model_version: "m".to_string(),
        };
        assert!(matches!(
            proto_to_memory_plan("mem-1", mismatch_plan),
            Err(AerisError::SidecarUnavailable(_))
        ));
    }

    #[test]
    fn circuit_breaker_closed_open_half_open_closed_lifecycle() {
        let threshold = 2;
        let cooldown = Duration::from_secs(10);
        let t0 = Instant::now();
        let mut cb = CircuitBreakerInner::new();

        assert_eq!(
            cb.effective_state_at(t0, cooldown),
            CircuitBreakerState::Closed
        );
        assert!(!cb.try_fast_short_circuit_at(t0, cooldown));
        assert_eq!(cb.try_acquire_at(t0, cooldown), Some(false));

        // Failure #1 keeps circuit Closed (< threshold=2).
        cb.record_transport_failure_at(t0, threshold);
        assert_eq!(
            cb.effective_state_at(t0, cooldown),
            CircuitBreakerState::Closed
        );
        assert_eq!(cb.consecutive_failures, 1);
        assert_eq!(cb.trip_count, 0);

        // Failure #2 trips Closed -> Open.
        let t1 = t0 + Duration::from_millis(50);
        cb.record_transport_failure_at(t1, threshold);
        assert_eq!(
            cb.effective_state_at(t1, cooldown),
            CircuitBreakerState::Open
        );
        assert_eq!(cb.trip_count, 1);

        // Before cooldown elapses: fast short-circuit and try_acquire both reject immediately.
        let t_mid = t1 + Duration::from_secs(5);
        assert!(cb.try_fast_short_circuit_at(t_mid, cooldown));
        assert_eq!(cb.try_acquire_at(t_mid, cooldown), None);
        assert_eq!(cb.short_circuit_count, 2);

        // After cooldown elapses: effective state is HalfOpen, fast-short-circuit yields false,
        // and a single HalfOpen probe is granted.
        let t_after = t1 + cooldown;
        assert_eq!(
            cb.effective_state_at(t_after, cooldown),
            CircuitBreakerState::HalfOpen
        );
        assert!(!cb.try_fast_short_circuit_at(t_after, cooldown));
        assert_eq!(cb.try_acquire_at(t_after, cooldown), Some(true));
        assert_eq!(cb.state, CircuitBreakerState::HalfOpen);

        // Probe succeeds -> transitions HalfOpen -> Closed with reset failure counter.
        cb.record_success_at(t_after + Duration::from_millis(5));
        assert_eq!(cb.state, CircuitBreakerState::Closed);
        assert_eq!(cb.consecutive_failures, 0);
        assert!(!cb.try_fast_short_circuit_at(t_after + Duration::from_millis(10), cooldown));
    }

    #[test]
    fn circuit_breaker_half_open_probe_failure_and_single_probe_gate() {
        let threshold = 2;
        let cooldown = Duration::from_secs(5);
        let t0 = Instant::now();
        let mut cb = CircuitBreakerInner::new();

        cb.record_transport_failure_at(t0, threshold);
        cb.record_transport_failure_at(t0, threshold);
        assert_eq!(cb.state, CircuitBreakerState::Open);
        assert_eq!(cb.trip_count, 1);

        // Cooldown expires: first caller acquires the sole HalfOpen probe.
        let t1 = t0 + cooldown;
        assert_eq!(cb.try_acquire_at(t1, cooldown), Some(true));

        // Concurrent second caller is short-circuited while the probe is in flight.
        assert!(cb.try_fast_short_circuit_at(t1, cooldown));
        assert_eq!(cb.try_acquire_at(t1, cooldown), None);

        // HalfOpen probe fails -> immediately re-opens for another full cooldown window.
        let t2 = t1 + Duration::from_millis(20);
        cb.record_transport_failure_at(t2, threshold);
        assert_eq!(cb.state, CircuitBreakerState::Open);
        assert_eq!(cb.trip_count, 2);
        assert!(cb.try_fast_short_circuit_at(t2 + Duration::from_secs(1), cooldown));
    }

    #[test]
    fn circuit_breaker_aborted_probe_releases_slot() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(1));
        {
            let permit = cb.try_acquire().expect("initial Closed permit");
            permit.record_transport_failure();
        }
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Verify pure abort-release semantics directly without wall-clock sleeping.
        let t0 = Instant::now();
        let mut inner = CircuitBreakerInner::new();
        inner.record_transport_failure_at(t0, 1);
        let t1 = t0 + Duration::from_secs(1);
        assert_eq!(inner.try_acquire_at(t1, Duration::from_secs(1)), Some(true));
        assert_eq!(inner.try_acquire_at(t1, Duration::from_secs(1)), None);
        inner.release_aborted_probe();
        assert_eq!(inner.try_acquire_at(t1, Duration::from_secs(1)), Some(true));
    }

    #[test]
    fn client_config_validates_circuit_breaker_parameters() {
        let ep = crate::transport::SidecarEndpoint::loopback_v4(50051);
        assert!(ClientConfig::new(ep.clone()).validate().is_ok());
        assert!(ClientConfig::new(ep.clone())
            .with_circuit_breaker_failure_threshold(0)
            .validate()
            .is_err());
        assert!(ClientConfig::new(ep)
            .with_circuit_breaker_cooldown(Duration::ZERO)
            .validate()
            .is_err());
    }
}
