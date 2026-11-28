//! Tonic gRPC service implementation, bounded idempotency cache, and
//! deterministic fault-injection state for [`super::ReferenceSidecarServer`].

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use tonic::{Request, Response, Status};

use super::planner::{
    build_normal_memory_response, build_normal_tuning_response, build_oob_memory_response,
    build_oob_tuning_response, REFERENCE_SERVER_MODEL_VERSION,
};
use super::ServerBehavior;
use crate::proto;
use crate::proto::control_plane_advisor_server::ControlPlaneAdvisor;

/// Maximum entries retained in the bounded `request_id` idempotency cache.
const IDEMPOTENCY_CACHE_CAPACITY: usize = 1024;

/// Simple bounded FIFO idempotency cache keyed by `request_id`.
#[derive(Debug)]
pub(super) struct BoundedCache<V> {
    capacity: usize,
    order: VecDeque<String>,
    map: HashMap<String, V>,
}

impl<V: Clone> BoundedCache<V> {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::with_capacity(capacity),
            map: HashMap::with_capacity(capacity),
        }
    }

    pub(super) fn get(&self, key: &str) -> Option<V> {
        self.map.get(key).cloned()
    }

    pub(super) fn insert(&mut self, key: String, value: V) {
        if let Some(existing) = self.map.get_mut(&key) {
            *existing = value;
            return;
        }
        if self.capacity == 0 {
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

/// Shared internal state accessed by gRPC service handlers and [`super::RunningServer`].
pub(super) struct ServerState {
    pub(super) behavior: RwLock<ServerBehavior>,
    pub(super) started_at: Instant,
    pub(super) request_count: AtomicU64,
    pub(super) tuning_cache_hits: AtomicU64,
    pub(super) flaky_failures_emitted: AtomicU32,
    tuning_cache: Mutex<BoundedCache<proto::TuningResponse>>,
    memory_cache: Mutex<BoundedCache<proto::MemoryAllocationResponse>>,
}

impl ServerState {
    pub(super) fn new(behavior: ServerBehavior) -> Self {
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

    /// Evaluates configured transport faults before normal/cache handling.
    #[allow(clippy::result_large_err)]
    async fn apply_fault_behavior(&self) -> Result<ServerBehavior, Status> {
        let behavior = self.current_behavior();
        match &behavior {
            ServerBehavior::SlowTimeout(delay) => {
                tokio::time::sleep(*delay).await;
            }
            ServerBehavior::FailInternal => {
                return Err(Status::internal("injected internal sidecar failure"));
            }
            ServerBehavior::FailUnavailable => {
                return Err(Status::unavailable("injected unavailable sidecar failure"));
            }
            ServerBehavior::FlakyThenNormal { fail_count } => {
                let emitted = self.flaky_failures_emitted.fetch_add(1, Ordering::Relaxed);
                if emitted < *fail_count {
                    return Err(Status::unavailable(format!(
                        "injected transient failure {}/{fail_count}",
                        emitted + 1
                    )));
                }
            }
            ServerBehavior::Normal
            | ServerBehavior::OutOfBoundsProposal
            | ServerBehavior::LowConfidence => {}
        }
        Ok(behavior)
    }
}

/// Tonic gRPC service implementation backed by [`ServerState`].
pub(super) struct ReferenceAdvisorService {
    pub(super) state: Arc<ServerState>,
}

#[tonic::async_trait]
impl ControlPlaneAdvisor for ReferenceAdvisorService {
    async fn propose_tuning(
        &self,
        request: Request<proto::TuningRequest>,
    ) -> Result<Response<proto::TuningResponse>, Status> {
        let req = request.into_inner();
        self.state.request_count.fetch_add(1, Ordering::Relaxed);

        let behavior = self.state.apply_fault_behavior().await?;
        match behavior {
            ServerBehavior::SlowTimeout(_) => {
                return Ok(Response::new(build_normal_tuning_response(&req)));
            }
            ServerBehavior::OutOfBoundsProposal => {
                return Ok(Response::new(build_oob_tuning_response(&req)));
            }
            ServerBehavior::LowConfidence => {
                let mut resp = build_normal_tuning_response(&req);
                resp.confidence = 0.25;
                resp.rationale = "injected low-confidence proposal".to_string();
                return Ok(Response::new(resp));
            }
            _ => {}
        }

        let resp = serve_cached_or_build(
            &self.state.tuning_cache,
            Some(&self.state.tuning_cache_hits),
            &req.request_id,
            || build_normal_tuning_response(&req),
        );
        Ok(Response::new(resp))
    }

    async fn allocate_memory(
        &self,
        request: Request<proto::MemoryAllocationRequest>,
    ) -> Result<Response<proto::MemoryAllocationResponse>, Status> {
        let req = request.into_inner();
        self.state.request_count.fetch_add(1, Ordering::Relaxed);

        let behavior = self.state.apply_fault_behavior().await?;
        match behavior {
            ServerBehavior::SlowTimeout(_) => {
                return Ok(Response::new(build_normal_memory_response(&req)));
            }
            ServerBehavior::OutOfBoundsProposal => {
                return Ok(Response::new(build_oob_memory_response(&req)));
            }
            ServerBehavior::LowConfidence => {
                let mut resp = build_normal_memory_response(&req);
                resp.confidence = 0.25;
                return Ok(Response::new(resp));
            }
            _ => {}
        }

        let resp = serve_cached_or_build(&self.state.memory_cache, None, &req.request_id, || {
            build_normal_memory_response(&req)
        });
        Ok(Response::new(resp))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        match self.state.current_behavior() {
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

fn serve_cached_or_build<V: Clone>(
    cache: &Mutex<BoundedCache<V>>,
    hit_counter: Option<&AtomicU64>,
    request_id: &str,
    build: impl FnOnce() -> V,
) -> V {
    if !request_id.is_empty() {
        let guard = cache.lock();
        if let Some(cached) = guard.get(request_id) {
            if let Some(counter) = hit_counter {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            return cached;
        }
    }
    let resp = build();
    if !request_id.is_empty() {
        cache.lock().insert(request_id.to_string(), resp.clone());
    }
    resp
}
