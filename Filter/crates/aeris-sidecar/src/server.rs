//! Reference Rust gRPC `ControlPlaneAdvisor` server supporting deterministic
//! fault injection over `SidecarEndpoint` (UDS on Unix, loopback TCP on Windows).
//!
//! Designed for exhaustive integration, guardrail, retry, timeout, and crash
//! fallback verification without requiring external infrastructure.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aeris_types::{
    AerisError, AerisResult, FilterConfig, DEFAULT_TARGET_LOAD_FACTOR, MAX_MEMENTO_BITS,
    MAX_QUOTIENT_BITS, MIN_QUOTIENT_BITS,
};
use parking_lot::{Mutex, RwLock};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::proto;
use crate::proto::control_plane_advisor_server::{ControlPlaneAdvisor, ControlPlaneAdvisorServer};
use crate::transport::SidecarEndpoint;

/// Model version reported by the reference Rust gRPC sidecar server.
pub const REFERENCE_SERVER_MODEL_VERSION: &str = "rust-reference-sidecar-v1";

/// Maximum entries retained in the bounded `request_id` idempotency cache.
const IDEMPOTENCY_CACHE_CAPACITY: usize = 1024;

/// Configurable behavior mode for [`ReferenceSidecarServer`] enabling
/// deterministic fault-injection across every Control Plane failure path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerBehavior {
    /// Returns mathematically valid, guardrail-compliant proposals and
    /// budget-respecting memory allocation plans.
    Normal,
    /// Returns proposals that deliberately violate [`crate::guardrail::SafetyGuardrail`]
    /// (`r < r_min`, `alpha > 0.90`, `k > 3.0`, and memory allocations exceeding budget).
    OutOfBoundsProposal,
    /// Returns geometrically valid proposals with confidence (`0.25`) below the
    /// default guardrail minimum (`0.80`).
    LowConfidence,
    /// Delays every RPC response by the specified [`Duration`] to exercise
    /// client timeout and fallback paths.
    SlowTimeout(Duration),
    /// Rejects every RPC with `tonic::Status::internal`.
    FailInternal,
    /// Rejects every RPC with `tonic::Status::unavailable` (retryable).
    FailUnavailable,
    /// Returns `Unavailable` for the first `fail_count` requests, then serves
    /// [`Self::Normal`] responses to exercise bounded retries with idempotency.
    FlakyThenNormal {
        /// Number of initial requests that fail with `Unavailable`.
        fail_count: u32,
    },
}

/// Simple bounded FIFO/LRU idempotency cache keyed by `request_id`.
#[derive(Debug)]
struct BoundedCache<V> {
    capacity: usize,
    order: VecDeque<String>,
    map: HashMap<String, V>,
}

impl<V: Clone> BoundedCache<V> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::with_capacity(capacity),
            map: HashMap::with_capacity(capacity),
        }
    }

    fn get(&self, key: &str) -> Option<V> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: V) {
        if let Some(existing) = self.map.get_mut(&key) {
            *existing = value;
            return;
        }
        if self.order.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }
}

/// Shared internal state accessed by gRPC service handlers and [`RunningServer`].
struct ServerState {
    behavior: RwLock<ServerBehavior>,
    started_at: Instant,
    request_count: AtomicU64,
    tuning_cache_hits: AtomicU64,
    flaky_failures_emitted: AtomicU32,
    tuning_cache: Mutex<BoundedCache<proto::TuningResponse>>,
    memory_cache: Mutex<BoundedCache<proto::MemoryAllocationResponse>>,
}

impl ServerState {
    fn new(behavior: ServerBehavior) -> Self {
        Self {
            behavior: RwLock::new(behavior),
            started_at: Instant::now(),
            request_count: AtomicU64::new(0),
            tuning_cache_hits: AtomicU64::new(0),
            flaky_failures_emitted: AtomicU32::new(0),
            tuning_cache: Mutex::new(BoundedCache::new(IDEMPOTENCY_CACHE_CAPACITY)),
            memory_cache: Mutex::new(BoundedCache::new(IDEMPOTENCY_CACHE_CAPACITY)),
        }
    }

    fn current_behavior(&self) -> ServerBehavior {
        self.behavior.read().clone()
    }
}

/// Tonic gRPC service implementation backed by [`ServerState`].
struct ReferenceAdvisorService {
    state: Arc<ServerState>,
}

#[tonic::async_trait]
impl ControlPlaneAdvisor for ReferenceAdvisorService {
    async fn propose_tuning(
        &self,
        request: Request<proto::TuningRequest>,
    ) -> Result<Response<proto::TuningResponse>, Status> {
        let req = request.into_inner();
        self.state.request_count.fetch_add(1, Ordering::Relaxed);

        let behavior = self.state.current_behavior();
        match behavior {
            ServerBehavior::SlowTimeout(delay) => {
                tokio::time::sleep(delay).await;
                return Ok(Response::new(build_normal_tuning_response(&req)));
            }
            ServerBehavior::FailInternal => {
                return Err(Status::internal("injected internal sidecar failure"));
            }
            ServerBehavior::FailUnavailable => {
                return Err(Status::unavailable("injected unavailable sidecar failure"));
            }
            ServerBehavior::FlakyThenNormal { fail_count } => {
                let emitted = self
                    .state
                    .flaky_failures_emitted
                    .fetch_add(1, Ordering::Relaxed);
                if emitted < fail_count {
                    return Err(Status::unavailable(format!(
                        "injected transient failure {}/{}",
                        emitted + 1,
                        fail_count
                    )));
                }
            }
            ServerBehavior::OutOfBoundsProposal => {
                return Ok(Response::new(proto::TuningResponse {
                    request_id: req.request_id,
                    proposed_r: 1,
                    proposed_m: req.payload_budget_c.saturating_add(5),
                    proposed_target_alpha: 0.99,
                    proposed_proactive_margin_k: 5.5,
                    confidence: 0.95,
                    trigger_proactive_rebuild: true,
                    model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
                    rationale: "injected out-of-bounds proposal".to_string(),
                    forecast_peak_ingest_rate: -10.0,
                }));
            }
            ServerBehavior::LowConfidence => {
                let mut resp = build_normal_tuning_response(&req);
                resp.confidence = 0.25;
                resp.rationale = "injected low-confidence proposal".to_string();
                return Ok(Response::new(resp));
            }
            ServerBehavior::Normal => {}
        }

        if !req.request_id.is_empty() {
            let cache = self.state.tuning_cache.lock();
            if let Some(cached) = cache.get(&req.request_id) {
                self.state.tuning_cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Response::new(cached));
            }
        }

        let resp = build_normal_tuning_response(&req);
        if !req.request_id.is_empty() {
            self.state
                .tuning_cache
                .lock()
                .insert(req.request_id.clone(), resp.clone());
        }
        Ok(Response::new(resp))
    }

    async fn allocate_memory(
        &self,
        request: Request<proto::MemoryAllocationRequest>,
    ) -> Result<Response<proto::MemoryAllocationResponse>, Status> {
        let req = request.into_inner();
        self.state.request_count.fetch_add(1, Ordering::Relaxed);

        let behavior = self.state.current_behavior();
        match behavior {
            ServerBehavior::SlowTimeout(delay) => {
                tokio::time::sleep(delay).await;
                return Ok(Response::new(build_normal_memory_response(&req)));
            }
            ServerBehavior::FailInternal => {
                return Err(Status::internal("injected internal sidecar failure"));
            }
            ServerBehavior::FailUnavailable => {
                return Err(Status::unavailable("injected unavailable sidecar failure"));
            }
            ServerBehavior::FlakyThenNormal { fail_count } => {
                let emitted = self
                    .state
                    .flaky_failures_emitted
                    .fetch_add(1, Ordering::Relaxed);
                if emitted < fail_count {
                    return Err(Status::unavailable("injected transient flaky failure"));
                }
            }
            ServerBehavior::OutOfBoundsProposal => {
                let over_budget = req.total_memory_budget_bytes.saturating_add(1024 * 1024);
                let allocations = req
                    .shards
                    .iter()
                    .map(|s| proto::ShardAllocation {
                        shard_id: s.shard_id,
                        allocated_memory_bytes: over_budget,
                        recommended_q: 60,
                        recommended_r: 2,
                        recommended_m: 30,
                    })
                    .collect();
                return Ok(Response::new(proto::MemoryAllocationResponse {
                    request_id: req.request_id,
                    allocations,
                    confidence: 0.95,
                    model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
                }));
            }
            ServerBehavior::LowConfidence => {
                let mut resp = build_normal_memory_response(&req);
                resp.confidence = 0.25;
                return Ok(Response::new(resp));
            }
            ServerBehavior::Normal => {}
        }

        if !req.request_id.is_empty() {
            let cache = self.state.memory_cache.lock();
            if let Some(cached) = cache.get(&req.request_id) {
                return Ok(Response::new(cached));
            }
        }

        let resp = build_normal_memory_response(&req);
        if !req.request_id.is_empty() {
            self.state
                .memory_cache
                .lock()
                .insert(req.request_id.clone(), resp.clone());
        }
        Ok(Response::new(resp))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        let behavior = self.state.current_behavior();
        match behavior {
            ServerBehavior::SlowTimeout(delay) => {
                tokio::time::sleep(delay).await;
            }
            ServerBehavior::FailInternal => {
                return Err(Status::internal("injected internal health failure"));
            }
            ServerBehavior::FailUnavailable => {
                return Err(Status::unavailable("injected unavailable health failure"));
            }
            _ => {}
        }
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
            uptime_ms: self.state.started_at.elapsed().as_millis() as u64,
        }))
    }
}

/// Handle to a running [`ReferenceSidecarServer`] instance allowing dynamic
/// behavior mutation, call count inspection, graceful shutdown, or abrupt crash
/// simulation.
pub struct RunningServer {
    endpoint: SidecarEndpoint,
    state: Arc<ServerState>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl RunningServer {
    /// Returns the actual bound [`SidecarEndpoint`] (with ephemeral port `0`
    /// resolved to the OS-assigned loopback port).
    #[inline]
    pub fn endpoint(&self) -> &SidecarEndpoint {
        &self.endpoint
    }

    /// Dynamically updates the server's fault-injection [`ServerBehavior`]
    /// mid-session.
    pub fn set_behavior(&self, behavior: ServerBehavior) {
        *self.state.behavior.write() = behavior;
    }

    /// Returns the total number of `ProposeTuning` and `AllocateMemory` RPCs
    /// received by this server.
    #[inline]
    pub fn request_count(&self) -> u64 {
        self.state.request_count.load(Ordering::Relaxed)
    }

    /// Returns the number of `ProposeTuning` requests served from the bounded
    /// `request_id` idempotency cache.
    #[inline]
    pub fn tuning_cache_hits(&self) -> u64 {
        self.state.tuning_cache_hits.load(Ordering::Relaxed)
    }

    /// Gracefully stops the gRPC server and awaits listener teardown.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.await;
        }
        self.cleanup_unix_socket();
    }

    /// Abruptly terminates the server task without graceful drain to simulate
    /// an out-of-process crash mid-session.
    pub async fn abort(mut self) {
        let _ = self.shutdown_tx.take();
        if let Some(handle) = self.join_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        self.cleanup_unix_socket();
    }

    fn cleanup_unix_socket(&self) {
        #[cfg(unix)]
        if let SidecarEndpoint::UnixSocket(path) = &self.endpoint {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            handle.abort();
        }
        self.cleanup_unix_socket();
    }
}

/// Factory for starting a [`ReferenceSidecarServer`] bound to a local IPC
/// [`SidecarEndpoint`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceSidecarServer;

impl ReferenceSidecarServer {
    /// Starts the reference gRPC server on `endpoint` with the initial
    /// `behavior`.
    ///
    /// For `SidecarEndpoint::loopback_v4(0)`, the OS assigns an available
    /// ephemeral loopback port accessible via [`RunningServer::endpoint`].
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `endpoint` fails validation, or
    /// [`AerisError::SidecarUnavailable`] if binding the listener fails.
    pub async fn start(
        endpoint: SidecarEndpoint,
        behavior: ServerBehavior,
    ) -> AerisResult<RunningServer> {
        endpoint.validate()?;
        let state = Arc::new(ServerState::new(behavior));
        let svc = ReferenceAdvisorService {
            state: Arc::clone(&state),
        };
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        match endpoint {
            SidecarEndpoint::LoopbackTcp(addr) => {
                let listener = TcpListener::bind(addr).await.map_err(|err| {
                    AerisError::SidecarUnavailable(format!(
                        "failed to bind loopback TCP sidecar on {addr}: {err}"
                    ))
                })?;
                let bound_addr = listener.local_addr().map_err(|err| {
                    AerisError::SidecarUnavailable(format!(
                        "failed to inspect bound sidecar address: {err}"
                    ))
                })?;
                let incoming = TcpListenerStream::new(listener);
                let bound_endpoint = SidecarEndpoint::LoopbackTcp(bound_addr);

                tracing::info!(
                    endpoint = %bound_endpoint,
                    "starting reference Rust gRPC sidecar server"
                );

                let handle = tokio::spawn(async move {
                    let _ = tonic::transport::Server::builder()
                        .add_service(ControlPlaneAdvisorServer::new(svc))
                        .serve_with_incoming_shutdown(incoming, async {
                            let _ = shutdown_rx.await;
                        })
                        .await;
                });

                Ok(RunningServer {
                    endpoint: bound_endpoint,
                    state,
                    shutdown_tx: Some(shutdown_tx),
                    join_handle: Some(handle),
                })
            }
            #[cfg(unix)]
            SidecarEndpoint::UnixSocket(path) => {
                if path.exists() {
                    let _ = std::fs::remove_file(&path);
                }
                let listener = tokio::net::UnixListener::bind(&path).map_err(|err| {
                    AerisError::SidecarUnavailable(format!(
                        "failed to bind UDS sidecar at {}: {err}",
                        path.display()
                    ))
                })?;
                let incoming = tokio_stream::wrappers::UnixListenerStream::new(listener);
                let bound_endpoint = SidecarEndpoint::UnixSocket(path);

                tracing::info!(
                    endpoint = %bound_endpoint,
                    "starting reference Rust UDS gRPC sidecar server"
                );

                let handle = tokio::spawn(async move {
                    let _ = tonic::transport::Server::builder()
                        .add_service(ControlPlaneAdvisorServer::new(svc))
                        .serve_with_incoming_shutdown(incoming, async {
                            let _ = shutdown_rx.await;
                        })
                        .await;
                });

                Ok(RunningServer {
                    endpoint: bound_endpoint,
                    state,
                    shutdown_tx: Some(shutdown_tx),
                    join_handle: Some(handle),
                })
            }
        }
    }
}

/// Builds a guardrail-compliant [`proto::TuningResponse`] from `req`.
fn build_normal_tuning_response(req: &proto::TuningRequest) -> proto::TuningResponse {
    let eps =
        if req.target_epsilon.is_finite() && req.target_epsilon > 0.0 && req.target_epsilon < 0.5 {
            req.target_epsilon
        } else {
            1.0 / 256.0
        };
    let r_min = FilterConfig::min_remainder_bits(eps).unwrap_or(12);
    let budget_c = (req.payload_budget_c as u8).max(r_min);
    let max_m = budget_c.saturating_sub(r_min).min(MAX_MEMENTO_BITS);

    let span = if req.p95_query_len.is_finite() && req.p95_query_len >= 1.0 {
        req.p95_query_len as u64
    } else {
        1
    };
    let desired_m = FilterConfig::ceil_log2_u64(span).unwrap_or(0).min(max_m);
    let proposed_r = budget_c - desired_m;

    let series_peak = req
        .recent_ingest_rate_series
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .fold(0.0_f64, f64::max);
    let base_ingest = if req.ingestion_rate_ewma.is_finite() && req.ingestion_rate_ewma >= 0.0 {
        req.ingestion_rate_ewma
    } else {
        0.0
    };
    let forecast_peak_ingest_rate = series_peak.max(base_ingest);

    proto::TuningResponse {
        request_id: req.request_id.clone(),
        proposed_r: u32::from(proposed_r),
        proposed_m: u32::from(desired_m),
        proposed_target_alpha: DEFAULT_TARGET_LOAD_FACTOR.clamp(0.50, 0.90),
        proposed_proactive_margin_k: 1.75,
        confidence: 0.92,
        trigger_proactive_rebuild: req.load_factor >= 0.80,
        model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
        rationale: "reference server optimal discrete allocation".to_string(),
        forecast_peak_ingest_rate,
    }
}

/// Builds a guardrail-compliant [`proto::MemoryAllocationResponse`] from `req`.
fn build_normal_memory_response(
    req: &proto::MemoryAllocationRequest,
) -> proto::MemoryAllocationResponse {
    if req.shards.is_empty() {
        return proto::MemoryAllocationResponse {
            request_id: req.request_id.clone(),
            allocations: Vec::new(),
            confidence: 0.92,
            model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
        };
    }

    let weights: Vec<f64> = req
        .shards
        .iter()
        .map(|s| {
            let keys = (s.key_count as f64).max(1.0);
            let qrate = if s.query_rate.is_finite() && s.query_rate >= 0.0 {
                s.query_rate
            } else {
                0.0
            };
            keys * (1.0 + qrate.ln_1p())
        })
        .collect();
    let total_w: f64 = weights.iter().sum::<f64>().max(1.0);

    let mut remaining = req.total_memory_budget_bytes;
    let mut allocations = Vec::with_capacity(req.shards.len());
    for (shard, w) in req.shards.iter().zip(weights.iter()) {
        let share = ((w / total_w) * (req.total_memory_budget_bytes as f64)).floor() as u64;
        let alloc = share.min(remaining);
        remaining = remaining.saturating_sub(alloc);

        let r = (shard.current_r as u8).max(12);
        let m = (shard.current_m as u8).min(MAX_MEMENTO_BITS);
        let q = (shard.current_q as u8).clamp(MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS);

        allocations.push(proto::ShardAllocation {
            shard_id: shard.shard_id,
            allocated_memory_bytes: alloc,
            recommended_q: u32::from(q),
            recommended_r: u32::from(r),
            recommended_m: u32::from(m),
        });
    }

    proto::MemoryAllocationResponse {
        request_id: req.request_id.clone(),
        allocations,
        confidence: 0.92,
        model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
    }
}
