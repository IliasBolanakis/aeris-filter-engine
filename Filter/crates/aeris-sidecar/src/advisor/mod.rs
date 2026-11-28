//! Guarded [`TuningAdvisor`] wrapping an optional out-of-process gRPC
//! [`SidecarClient`], deterministic [`SafetyGuardrail`], and guaranteed-safe
//! native analytical [`CostOptimizer`] fallback.
//!
//! # Architectural Isolation
//! [`GuardedTuningAdvisor`] executes all sidecar RPCs asynchronously on a
//! dedicated Control Plane Tokio runtime isolated from the Data Plane hot path.
//! Whenever the sidecar is disabled, unreachable, crashes mid-RPC, times out,
//! or emits ANY proposal that violates [`SafetyGuardrail`], the advisor logs a
//! structured `tracing::warn!` event, increments fallback metrics, and returns
//! a mathematically safe proposal computed by the native deterministic
//! [`CostOptimizer`] (`model_version = "native-deterministic-fallback"`).

mod fallback;
mod metrics;
#[cfg(test)]
mod tests;

pub use fallback::{NativeFallbackOptimizer, NATIVE_FALLBACK_MODEL_VERSION};
pub use metrics::{SidecarMetrics, SidecarMetricsSnapshot};

use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};

use aeris_types::{
    AerisError, AerisResult, CostModelParams, CostOptimizer, Epoch, FilterConfig, GuardrailVerdict,
    MemoryAllocationPlan, SafetyBounds, ShardId, ShardTelemetrySnapshot, TelemetrySnapshot,
    TuningAdvisor, TuningProposal,
};
use fallback::{compute_native_memory_fallback, compute_native_tuning_fallback};
use tokio::runtime::{Builder, Handle, Runtime};

use crate::client::SidecarClient;
use crate::guardrail::SafetyGuardrail;

/// Runtime ownership strategy isolating async sidecar I/O from the Data Plane.
enum AdvisorRuntime {
    /// Dedicated lightweight 1-worker Tokio runtime owned by the advisor.
    ///
    /// Wrapped in `Option` so [`Drop`] can invoke [`Runtime::shutdown_background`]
    /// if the advisor is ever dropped from within another Tokio async context.
    Owned(Option<Runtime>),
    /// Borrowed handle to an external Control Plane Tokio runtime.
    External(Handle),
}

impl AdvisorRuntime {
    #[inline]
    fn handle(&self) -> &Handle {
        match self {
            Self::Owned(Some(rt)) => rt.handle(),
            Self::Owned(None) => {
                unreachable!("AdvisorRuntime::Owned runtime is always Some until Drop")
            }
            Self::External(handle) => handle,
        }
    }
}

impl Drop for AdvisorRuntime {
    fn drop(&mut self) {
        if let Self::Owned(slot) = self {
            if let Some(rt) = slot.take() {
                if Handle::try_current().is_ok() {
                    rt.shutdown_background();
                } else {
                    drop(rt);
                }
            }
        }
    }
}

/// Guarded [`TuningAdvisor`] combining an optional out-of-process gRPC sidecar,
/// deterministic [`SafetyGuardrail`], and native [`CostOptimizer`] fallback.
///
/// # Safety Guarantee
/// Every [`TuningProposal`] and [`MemoryAllocationPlan`] returned by this
/// advisor is guaranteed to pass [`SafetyGuardrail`] against the configured
/// [`SafetyBounds`] — even when the sidecar returns adversarial parameters,
/// hangs indefinitely, or crashes mid-RPC.
pub struct GuardedTuningAdvisor<O: CostOptimizer> {
    client: Option<SidecarClient>,
    guardrail: SafetyGuardrail,
    fallback_optimizer: O,
    bounds: SafetyBounds,
    cost_model: CostModelParams,
    runtime: AdvisorRuntime,
    metrics: SidecarMetrics,
    memory_req_seq: AtomicU64,
}

impl<O: CostOptimizer> GuardedTuningAdvisor<O> {
    /// Constructs a [`GuardedTuningAdvisor`] backed by its own lightweight
    /// 1-worker Tokio runtime strictly isolated from the Data Plane.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `bounds.target_epsilon` is invalid,
    /// or [`AerisError::InternalInvariant`] if the isolated Tokio runtime fails
    /// to initialize.
    pub fn with_owned_runtime(
        client: Option<SidecarClient>,
        fallback_optimizer: O,
        bounds: SafetyBounds,
        cost_model: CostModelParams,
    ) -> AerisResult<Self> {
        validate_advisor_bounds(&bounds)?;
        let rt = Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("aeris-sidecar-rt")
            .enable_all()
            .build()
            .map_err(|err| {
                AerisError::InternalInvariant(format!(
                    "failed to initialize isolated sidecar Tokio runtime: {err}"
                ))
            })?;

        Ok(Self::from_runtime(
            client,
            fallback_optimizer,
            bounds,
            cost_model,
            AdvisorRuntime::Owned(Some(rt)),
        ))
    }

    /// Constructs a [`GuardedTuningAdvisor`] using an externally managed Control
    /// Plane Tokio runtime [`Handle`].
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `bounds.target_epsilon` is invalid.
    pub fn with_runtime_handle(
        client: Option<SidecarClient>,
        fallback_optimizer: O,
        bounds: SafetyBounds,
        cost_model: CostModelParams,
        handle: Handle,
    ) -> AerisResult<Self> {
        validate_advisor_bounds(&bounds)?;
        Ok(Self::from_runtime(
            client,
            fallback_optimizer,
            bounds,
            cost_model,
            AdvisorRuntime::External(handle),
        ))
    }

    /// Constructs a native-only [`GuardedTuningAdvisor`] with no out-of-process
    /// sidecar configured (always runs deterministic analytical optimization).
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `bounds` are invalid, or
    /// [`AerisError::InternalInvariant`] if runtime creation fails.
    pub fn native_only(
        fallback_optimizer: O,
        bounds: SafetyBounds,
        cost_model: CostModelParams,
    ) -> AerisResult<Self> {
        Self::with_owned_runtime(None, fallback_optimizer, bounds, cost_model)
    }

    fn from_runtime(
        client: Option<SidecarClient>,
        fallback_optimizer: O,
        bounds: SafetyBounds,
        cost_model: CostModelParams,
        runtime: AdvisorRuntime,
    ) -> Self {
        Self {
            client,
            guardrail: SafetyGuardrail::new(),
            fallback_optimizer,
            bounds,
            cost_model,
            runtime,
            metrics: SidecarMetrics::default(),
            memory_req_seq: AtomicU64::new(1),
        }
    }

    /// Returns a point-in-time snapshot of sidecar acceptance, guardrail
    /// rejection, and transport fallback counters.
    #[inline]
    pub fn metrics(&self) -> SidecarMetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Returns a reference to the underlying atomic [`SidecarMetrics`].
    #[inline]
    pub fn metrics_ref(&self) -> &SidecarMetrics {
        &self.metrics
    }

    /// Returns the active [`SafetyBounds`] enforced on every proposal.
    #[inline]
    pub fn bounds(&self) -> &SafetyBounds {
        &self.bounds
    }

    /// Returns the active [`CostModelParams`] used by the native fallback.
    #[inline]
    pub fn cost_model(&self) -> &CostModelParams {
        &self.cost_model
    }

    /// Returns the embedded deterministic [`SafetyGuardrail`].
    #[inline]
    pub const fn guardrail(&self) -> SafetyGuardrail {
        self.guardrail
    }

    /// Returns a reference to the configured [`SidecarClient`], if any.
    #[inline]
    pub fn client(&self) -> Option<&SidecarClient> {
        self.client.as_ref()
    }

    /// Async counterpart to [`TuningAdvisor::advise`] for async Control Plane
    /// tasks, avoiding any blocking bridge overhead.
    ///
    /// Always returns a guardrail-verified [`TuningProposal`]; transparently
    /// falls back to the native deterministic [`CostOptimizer`] on any transport
    /// error, timeout, or guardrail violation.
    pub async fn advise_async(&self, telemetry: &TelemetrySnapshot) -> AerisResult<TuningProposal> {
        if let Some(fallback) = self.try_short_circuit_tuning(telemetry) {
            return Ok(fallback);
        }
        let client = self
            .client
            .as_ref()
            .expect("try_short_circuit_tuning handles None");
        match client.propose_tuning_async(telemetry).await {
            Ok(proposal) => Ok(self.evaluate_tuning_proposal(telemetry, proposal)),
            Err(err) => {
                self.metrics.record_rpc_error(&err);
                tracing::warn!(
                    shard_id = telemetry.shard_id,
                    epoch = telemetry.epoch,
                    request_id = %telemetry.request_id,
                    error = %err,
                    "falling back to native optimizer after sidecar RPC failure"
                );
                Ok(self.native_tuning(telemetry, &format!("sidecar RPC fallback: {err}")))
            }
        }
    }

    /// Async counterpart to [`TuningAdvisor::allocate_memory`].
    ///
    /// Always returns a guardrail-verified [`MemoryAllocationPlan`] covering
    /// every requested shard and respecting `total_budget_bytes`.
    pub async fn allocate_memory_async(
        &self,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan> {
        let request_id = self.next_memory_request_id(epoch);
        if let Some(fallback) =
            self.try_short_circuit_memory(&request_id, epoch, total_budget_bytes, shards)
        {
            return Ok(fallback);
        }
        let client = self
            .client
            .as_ref()
            .expect("try_short_circuit_memory handles None");
        match client
            .allocate_memory_async(request_id.clone(), epoch, total_budget_bytes, shards)
            .await
        {
            Ok(plan) => {
                Ok(self.evaluate_memory_plan(&request_id, epoch, total_budget_bytes, shards, plan))
            }
            Err(err) => {
                self.metrics.record_rpc_error(&err);
                tracing::warn!(
                    epoch,
                    request_id = %request_id,
                    error = %err,
                    "falling back to native memory allocator after sidecar RPC failure"
                );
                Ok(self.native_memory(&request_id, total_budget_bytes, shards))
            }
        }
    }

    /// Async liveness probe returning `true` iff a sidecar client is configured
    /// and responds `healthy == true` within the configured timeout.
    pub async fn is_healthy_async(&self) -> bool {
        let Some(client) = self.client.as_ref() else {
            return false;
        };
        if client.try_fast_short_circuit() {
            return false;
        }
        matches!(client.health_check_async().await, Ok(status) if status.healthy)
    }

    fn try_short_circuit_tuning(&self, telemetry: &TelemetrySnapshot) -> Option<TuningProposal> {
        self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
        match self.client.as_ref() {
            None => {
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                Some(self.native_tuning(
                    telemetry,
                    "sidecar disabled; using native deterministic optimizer",
                ))
            }
            Some(client) if client.try_fast_short_circuit() => {
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    shard_id = telemetry.shard_id,
                    epoch = telemetry.epoch,
                    request_id = %telemetry.request_id,
                    "sidecar circuit breaker open; fast-path native tuning fallback"
                );
                Some(self.native_tuning(
                    telemetry,
                    "sidecar circuit breaker open; immediate native deterministic fallback",
                ))
            }
            Some(_) => None,
        }
    }

    fn try_short_circuit_memory(
        &self,
        request_id: &str,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> Option<MemoryAllocationPlan> {
        self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
        match self.client.as_ref() {
            None => {
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                Some(self.native_memory(request_id, total_budget_bytes, shards))
            }
            Some(client) if client.try_fast_short_circuit() => {
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    epoch,
                    request_id = %request_id,
                    "sidecar circuit breaker open; fast-path native memory allocation fallback"
                );
                Some(self.native_memory(request_id, total_budget_bytes, shards))
            }
            Some(_) => None,
        }
    }

    fn evaluate_tuning_proposal(
        &self,
        telemetry: &TelemetrySnapshot,
        proposal: TuningProposal,
    ) -> TuningProposal {
        match self.guardrail.validate_tuning(&proposal, &self.bounds) {
            GuardrailVerdict::Approved => {
                self.metrics
                    .sidecar_accepted
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    shard_id = telemetry.shard_id,
                    epoch = telemetry.epoch,
                    request_id = %proposal.request_id,
                    proposed_r = proposal.proposed_r,
                    proposed_m = proposal.proposed_m,
                    alpha = proposal.proposed_target_alpha,
                    proactive_margin_k = proposal.proposed_proactive_margin_k,
                    confidence = proposal.confidence,
                    model_version = %proposal.model_version,
                    "accepted sidecar tuning proposal"
                );
                proposal
            }
            GuardrailVerdict::Rejected { reasons } => {
                self.metrics
                    .guardrail_rejections
                    .fetch_add(1, Ordering::Relaxed);
                let joined = reasons.join("; ");
                tracing::warn!(
                    shard_id = telemetry.shard_id,
                    epoch = telemetry.epoch,
                    request_id = %telemetry.request_id,
                    reasons = %joined,
                    "falling back to native optimizer after guardrail rejection"
                );
                self.native_tuning(
                    telemetry,
                    &format!("guardrail rejected sidecar proposal: {joined}"),
                )
            }
        }
    }

    fn evaluate_memory_plan(
        &self,
        request_id: &str,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
        plan: MemoryAllocationPlan,
    ) -> MemoryAllocationPlan {
        let verdict = self
            .guardrail
            .validate_memory_plan(&plan, total_budget_bytes, &self.bounds);
        let coverage_ok = verify_shard_coverage(&plan, shards);

        match (verdict, coverage_ok) {
            (GuardrailVerdict::Approved, true) => {
                self.metrics
                    .sidecar_accepted
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    epoch,
                    request_id = %plan.request_id,
                    shard_count = plan.allocations.len(),
                    total_budget_bytes,
                    confidence = plan.confidence,
                    model_version = %plan.model_version,
                    "accepted sidecar memory allocation plan"
                );
                plan
            }
            (GuardrailVerdict::Approved, false) => {
                self.metrics
                    .guardrail_rejections
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    epoch,
                    request_id = %request_id,
                    expected_shards = shards.len(),
                    returned_shards = plan.allocations.len(),
                    "falling back to native memory allocator due to shard set mismatch"
                );
                self.native_memory(request_id, total_budget_bytes, shards)
            }
            (GuardrailVerdict::Rejected { reasons }, _) => {
                self.metrics
                    .guardrail_rejections
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    epoch,
                    request_id = %request_id,
                    reasons = ?reasons,
                    "falling back to native memory allocator after guardrail rejection"
                );
                self.native_memory(request_id, total_budget_bytes, shards)
            }
        }
    }

    #[inline]
    fn next_memory_request_id(&self, epoch: Epoch) -> String {
        let seq = self.memory_req_seq.fetch_add(1, Ordering::Relaxed);
        format!("mem-e{epoch}-{seq}")
    }

    #[inline]
    fn native_tuning(&self, telemetry: &TelemetrySnapshot, reason: &str) -> TuningProposal {
        compute_native_tuning_fallback(
            &self.fallback_optimizer,
            self.guardrail,
            &self.bounds,
            &self.cost_model,
            telemetry,
            reason,
        )
    }

    #[inline]
    fn native_memory(
        &self,
        request_id: &str,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> MemoryAllocationPlan {
        compute_native_memory_fallback(
            self.guardrail,
            &self.bounds,
            request_id,
            total_budget_bytes,
            shards,
        )
    }

    /// Executes an async sidecar future from synchronous [`TuningAdvisor`]
    /// trait methods without panicking if the caller happens to reside inside
    /// another Tokio runtime.
    fn run_blocking<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T> + Send,
        T: Send,
    {
        let handle = self.runtime.handle();
        if Handle::try_current().is_ok() {
            // Bounce off Tokio worker threads so `handle.block_on` never triggers
            // nested-runtime panics.
            std::thread::scope(|scope| {
                scope
                    .spawn(|| handle.block_on(future))
                    .join()
                    .expect("sidecar bridge thread must not panic")
            })
        } else {
            handle.block_on(future)
        }
    }
}

impl<O: CostOptimizer> TuningAdvisor for GuardedTuningAdvisor<O> {
    fn advise(&self, telemetry: &TelemetrySnapshot) -> AerisResult<TuningProposal> {
        let sync_fast_path = match self.client.as_ref() {
            None => true,
            Some(client) => client.try_fast_short_circuit(),
        };
        if sync_fast_path {
            if let Some(fallback) = self.try_short_circuit_tuning(telemetry) {
                return Ok(fallback);
            }
        }
        self.run_blocking(self.advise_async(telemetry))
    }

    fn allocate_memory(
        &self,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan> {
        let sync_fast_path = match self.client.as_ref() {
            None => true,
            Some(client) => client.try_fast_short_circuit(),
        };
        if sync_fast_path {
            let request_id = self.next_memory_request_id(epoch);
            if let Some(fallback) =
                self.try_short_circuit_memory(&request_id, epoch, total_budget_bytes, shards)
            {
                return Ok(fallback);
            }
        }
        self.run_blocking(self.allocate_memory_async(epoch, total_budget_bytes, shards))
    }

    fn is_healthy(&self) -> bool {
        let Some(client) = self.client.as_ref() else {
            return false;
        };
        if client.try_fast_short_circuit() {
            return false;
        }
        self.run_blocking(self.is_healthy_async())
    }
}

/// Validates that [`SafetyBounds`] can produce a valid fallback configuration.
pub(super) fn validate_advisor_bounds(bounds: &SafetyBounds) -> AerisResult<()> {
    let r_min = FilterConfig::min_remainder_bits(bounds.target_epsilon)?;
    if bounds.payload_budget_c < r_min {
        return Err(AerisError::InvalidConfig(format!(
            "SafetyBounds payload_budget_c ({}) < r_min(epsilon={}) ({r_min})",
            bounds.payload_budget_c, bounds.target_epsilon
        )));
    }
    if !bounds.min_alpha.is_finite()
        || !bounds.max_alpha.is_finite()
        || bounds.min_alpha > bounds.max_alpha
    {
        return Err(AerisError::InvalidConfig(format!(
            "SafetyBounds alpha range [{}, {}] is invalid",
            bounds.min_alpha, bounds.max_alpha
        )));
    }
    if !bounds.min_proactive_margin_k.is_finite()
        || !bounds.max_proactive_margin_k.is_finite()
        || bounds.min_proactive_margin_k > bounds.max_proactive_margin_k
    {
        return Err(AerisError::InvalidConfig(format!(
            "SafetyBounds proactive margin k range [{}, {}] is invalid",
            bounds.min_proactive_margin_k, bounds.max_proactive_margin_k
        )));
    }
    Ok(())
}

/// Verifies that `plan` assigns an allocation to every requested shard and no
/// spurious shards.
pub(super) fn verify_shard_coverage(
    plan: &MemoryAllocationPlan,
    shards: &[ShardTelemetrySnapshot],
) -> bool {
    if plan.allocations.len() != shards.len() {
        return false;
    }
    let expected: HashSet<ShardId> = shards.iter().map(|s| s.shard_id).collect();
    plan.allocations
        .iter()
        .all(|alloc| expected.contains(&alloc.shard_id))
}
