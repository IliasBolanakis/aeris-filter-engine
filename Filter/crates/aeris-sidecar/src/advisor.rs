//! Guarded Phase 5 [`TuningAdvisor`] wrapping an optional out-of-process gRPC
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

use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};

use aeris_types::{
    AerisError, AerisResult, CostModelParams, CostOptimizer, Epoch, FilterConfig, GuardrailVerdict,
    MemoryAllocationPlan, OptimizationDecision, SafetyBounds, ShardAllocationDecision, ShardId,
    ShardTelemetrySnapshot, TelemetrySnapshot, TuningAdvisor, TuningProposal, WorkloadStats,
    DEFAULT_TARGET_LOAD_FACTOR, MAX_QUOTIENT_BITS, MIN_QUOTIENT_BITS,
};
use tokio::runtime::{Builder, Handle, Runtime};

use crate::client::SidecarClient;
use crate::guardrail::SafetyGuardrail;

/// Canonical `model_version` tag emitted whenever the deterministic native
/// analytical fallback computes a tuning proposal or memory allocation plan.
pub const NATIVE_FALLBACK_MODEL_VERSION: &str = "native-deterministic-fallback";

/// Default conservative proactive rebuild safety margin `k` used by the native
/// fallback when the sidecar is unavailable or rejected.
const NATIVE_DEFAULT_PROACTIVE_MARGIN_K: f64 = 1.5;

/// Default bulk-load speed (keys/sec) assumed if telemetry reports `<= 0.0`.
const DEFAULT_BULK_LOAD_SPEED_FALLBACK: f64 = 500_000.0;

/// Self-contained deterministic analytical Discrete Cost Optimizer implementing
/// [`CostOptimizer`].
///
/// Exhaustively evaluates discrete memento bit widths `m = log2(P)` in
/// `0..=min(cost_model.max_m, current.payload_budget_c)` while strictly enforcing
/// `r = C - m >= r_min(target_epsilon)`. Allows `aeris-sidecar` to operate and
/// test standalone while also permitting any external [`CostOptimizer`] (such as
/// `aeris-control`'s `DiscreteCostOptimizer`) to be injected into
/// [`GuardedTuningAdvisor`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeFallbackOptimizer;

impl NativeFallbackOptimizer {
    /// Constructs a new [`NativeFallbackOptimizer`].
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluates the expected query cost (nanoseconds) for a candidate `(r, m)`
    /// pair under `stats` and `cost_model`.
    ///
    /// Models:
    /// - Partition size `P = 2^m`.
    /// - Expected partition probes `probes = 1 + (L - 1) / P` for effective span `L >= 1`.
    /// - Single-probe remainder collision rate `p_probe = 2^-r`.
    /// - Union false-positive probability across probes:
    ///   `p_fp = 1 - (1 - p_probe)^probes` (numerically stable via `ln_1p` / `exp_m1`).
    /// - Total expected cost: `probes * cpu_probe_ns + p_fp * disk_io_penalty_ns`.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    pub fn expected_cost_ns(
        r: u8,
        m: u8,
        effective_span: u64,
        cost_model: &CostModelParams,
    ) -> f64 {
        let span = effective_span.max(1) as f64;
        let partition_size = (1u64 << m.min(63)) as f64;
        let probes = 1.0 + (span - 1.0) / partition_size;
        let p_probe = 2.0_f64.powi(-(i32::from(r)));
        let p_fp = if p_probe >= 1.0 {
            1.0
        } else {
            (-((-p_probe).ln_1p() * probes).exp_m1()).clamp(0.0, 1.0)
        };
        probes * cost_model.cpu_probe_ns + p_fp * cost_model.disk_io_penalty_ns
    }
}

impl CostOptimizer for NativeFallbackOptimizer {
    /// Exhaustively evaluates all valid discrete `m` candidates respecting
    /// `r + m = C` and `r >= r_min(epsilon)` and returns the minimum-cost
    /// [`OptimizationDecision`].
    ///
    /// # Complexity
    /// `O(max_m)` time (`<= 21` iterations), `O(1)` space, zero heap allocations.
    fn optimize(
        &self,
        stats: &WorkloadStats,
        current: &FilterConfig,
        cost_model: &CostModelParams,
    ) -> OptimizationDecision {
        let effective_span = stats.effective_query_span.max(stats.p95_query_len).max(1);
        let r_min = FilterConfig::min_remainder_bits(current.target_epsilon).unwrap_or(current.r);
        let max_valid_m = current
            .payload_budget_c
            .saturating_sub(r_min)
            .min(cost_model.max_m);

        let current_cost_ns =
            Self::expected_cost_ns(current.r, current.m, effective_span, cost_model);

        let mut best_m = current.m.min(max_valid_m);
        let mut best_r = current.payload_budget_c.saturating_sub(best_m);
        let mut best_cost_ns = Self::expected_cost_ns(best_r, best_m, effective_span, cost_model);

        for candidate_m in 0..=max_valid_m {
            let candidate_r = current.payload_budget_c - candidate_m;
            if candidate_r < r_min {
                continue;
            }
            let cost = Self::expected_cost_ns(candidate_r, candidate_m, effective_span, cost_model);
            if cost < best_cost_ns {
                best_cost_ns = cost;
                best_m = candidate_m;
                best_r = candidate_r;
            }
        }

        let relative_gain = if current_cost_ns > 0.0 {
            (current_cost_ns - best_cost_ns) / current_cost_ns
        } else {
            0.0
        };
        let should_rebuild =
            best_m != current.m && relative_gain >= cost_model.rebuild_benefit_threshold.max(0.0);
        let should_bypass_filter = best_cost_ns > cost_model.direct_scan_cost_ns;

        OptimizationDecision {
            recommended_m: best_m,
            recommended_r: best_r,
            estimated_cost_ns: best_cost_ns,
            current_cost_ns,
            should_bypass_filter,
            should_rebuild,
        }
    }
}

/// Lock-free atomic observability counters for [`GuardedTuningAdvisor`].
///
/// # Conservation Invariant
/// After every completed `advise` or `allocate_memory` call:
/// `total_calls == sidecar_accepted + guardrail_rejections + transport_fallbacks`.
#[derive(Debug, Default)]
pub struct SidecarMetrics {
    /// Total `advise` and `allocate_memory` calls served by the advisor.
    pub total_calls: AtomicU64,
    /// Proposals returned by the sidecar and approved by [`SafetyGuardrail`].
    pub sidecar_accepted: AtomicU64,
    /// Proposals returned by the sidecar that failed [`SafetyGuardrail`] and
    /// triggered native fallback.
    pub guardrail_rejections: AtomicU64,
    /// Calls that fell back to the native optimizer because the sidecar was
    /// unconfigured, unreachable, crashed, errored, or timed out.
    pub transport_fallbacks: AtomicU64,
}

impl SidecarMetrics {
    /// Captures a plain value snapshot of all atomic counters.
    #[inline]
    pub fn snapshot(&self) -> SidecarMetricsSnapshot {
        SidecarMetricsSnapshot {
            total_calls: self.total_calls.load(Ordering::Relaxed),
            sidecar_accepted: self.sidecar_accepted.load(Ordering::Relaxed),
            guardrail_rejections: self.guardrail_rejections.load(Ordering::Relaxed),
            transport_fallbacks: self.transport_fallbacks.load(Ordering::Relaxed),
        }
    }
}

/// Immutable point-in-time snapshot of [`SidecarMetrics`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SidecarMetricsSnapshot {
    /// Total `advise` and `allocate_memory` calls served.
    pub total_calls: u64,
    /// Sidecar responses approved by [`SafetyGuardrail`].
    pub sidecar_accepted: u64,
    /// Sidecar responses rejected by [`SafetyGuardrail`] (fell back to native).
    pub guardrail_rejections: u64,
    /// Sidecar transport/timeout/disabled fallbacks to native optimizer.
    pub transport_fallbacks: u64,
}

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

/// Phase 5 [`TuningAdvisor`] combining an optional out-of-process gRPC sidecar,
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

        Ok(Self {
            client,
            guardrail: SafetyGuardrail::new(),
            fallback_optimizer,
            bounds,
            cost_model,
            runtime: AdvisorRuntime::Owned(Some(rt)),
            metrics: SidecarMetrics::default(),
            memory_req_seq: AtomicU64::new(1),
        })
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
        Ok(Self {
            client,
            guardrail: SafetyGuardrail::new(),
            fallback_optimizer,
            bounds,
            cost_model,
            runtime: AdvisorRuntime::External(handle),
            metrics: SidecarMetrics::default(),
            memory_req_seq: AtomicU64::new(1),
        })
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
        self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);

        let Some(client) = self.client.as_ref() else {
            self.metrics
                .transport_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return Ok(self.compute_native_tuning_fallback(
                telemetry,
                "sidecar disabled; using native deterministic optimizer",
            ));
        };

        if client.try_fast_short_circuit() {
            self.metrics
                .transport_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                shard_id = telemetry.shard_id,
                epoch = telemetry.epoch,
                request_id = %telemetry.request_id,
                "sidecar circuit breaker open; fast-path native tuning fallback"
            );
            return Ok(self.compute_native_tuning_fallback(
                telemetry,
                "sidecar circuit breaker open; immediate native deterministic fallback",
            ));
        }

        match client.propose_tuning_async(telemetry).await {
            Ok(proposal) => match self.guardrail.validate_tuning(&proposal, &self.bounds) {
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
                    Ok(proposal)
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
                    Ok(self.compute_native_tuning_fallback(
                        telemetry,
                        &format!("guardrail rejected sidecar proposal: {joined}"),
                    ))
                }
            },
            Err(err) => {
                match &err {
                    AerisError::GuardrailRejected { .. } => {
                        self.metrics
                            .guardrail_rejections
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        self.metrics
                            .transport_fallbacks
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                tracing::warn!(
                    shard_id = telemetry.shard_id,
                    epoch = telemetry.epoch,
                    request_id = %telemetry.request_id,
                    error = %err,
                    "falling back to native optimizer after sidecar RPC failure"
                );
                Ok(self.compute_native_tuning_fallback(
                    telemetry,
                    &format!("sidecar RPC fallback: {err}"),
                ))
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
        self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
        let seq = self.memory_req_seq.fetch_add(1, Ordering::Relaxed);
        let request_id = format!("mem-e{epoch}-{seq}");

        let Some(client) = self.client.as_ref() else {
            self.metrics
                .transport_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return Ok(self.compute_native_memory_fallback(
                &request_id,
                total_budget_bytes,
                shards,
            ));
        };

        if client.try_fast_short_circuit() {
            self.metrics
                .transport_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                epoch,
                request_id = %request_id,
                "sidecar circuit breaker open; fast-path native memory allocation fallback"
            );
            return Ok(self.compute_native_memory_fallback(
                &request_id,
                total_budget_bytes,
                shards,
            ));
        }

        match client
            .allocate_memory_async(request_id.clone(), epoch, total_budget_bytes, shards)
            .await
        {
            Ok(plan) => {
                let verdict =
                    self.guardrail
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
                        Ok(plan)
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
                        Ok(self.compute_native_memory_fallback(
                            &request_id,
                            total_budget_bytes,
                            shards,
                        ))
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
                        Ok(self.compute_native_memory_fallback(
                            &request_id,
                            total_budget_bytes,
                            shards,
                        ))
                    }
                }
            }
            Err(err) => {
                match &err {
                    AerisError::GuardrailRejected { .. } => {
                        self.metrics
                            .guardrail_rejections
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        self.metrics
                            .transport_fallbacks
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                tracing::warn!(
                    epoch,
                    request_id = %request_id,
                    error = %err,
                    "falling back to native memory allocator after sidecar RPC failure"
                );
                Ok(self.compute_native_memory_fallback(&request_id, total_budget_bytes, shards))
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
            // Caller is currently executing inside a Tokio worker thread; bounce
            // onto a short-lived scoped OS thread so `handle.block_on` never
            // triggers Tokio's nested-runtime panic.
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

    /// Computes a deterministic, guaranteed-safe [`TuningProposal`] using the
    /// native [`CostOptimizer`].
    fn compute_native_tuning_fallback(
        &self,
        telemetry: &TelemetrySnapshot,
        reason: &str,
    ) -> TuningProposal {
        let r_min = FilterConfig::min_remainder_bits(self.bounds.target_epsilon).unwrap_or(12);
        let budget_c = self.bounds.payload_budget_c.max(r_min);
        let max_valid_m = budget_c.saturating_sub(r_min).min(self.bounds.m_max);

        let safe_q = telemetry
            .current_q
            .clamp(MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS);
        let safe_m = telemetry.current_m.min(max_valid_m);
        let safe_r = budget_c - safe_m;
        let max_range_r = 1u64 << safe_m;

        let current_cfg = FilterConfig::new(
            safe_q,
            safe_r,
            max_range_r,
            budget_c,
            self.bounds.target_epsilon,
        )
        .unwrap_or_else(|_| {
            FilterConfig::with_minimum_bits(safe_q, 1, self.bounds.target_epsilon)
                .expect("validated SafetyBounds guarantee a valid minimum FilterConfig")
        });

        let stats = telemetry_to_workload_stats(telemetry);
        let decision = self
            .fallback_optimizer
            .optimize(&stats, &current_cfg, &self.cost_model);

        let proposed_m = decision.recommended_m.min(max_valid_m);
        let proposed_r = decision
            .recommended_r
            .max(r_min)
            .min(budget_c.saturating_sub(proposed_m));

        let proposed_target_alpha =
            DEFAULT_TARGET_LOAD_FACTOR.clamp(self.bounds.min_alpha, self.bounds.max_alpha);
        let proposed_proactive_margin_k = NATIVE_DEFAULT_PROACTIVE_MARGIN_K.clamp(
            self.bounds.min_proactive_margin_k,
            self.bounds.max_proactive_margin_k,
        );

        let forecast_peak_ingest_rate = sanitize_non_negative(telemetry.ingestion_rate_ewma);
        let capacity = (1u64 << safe_q) as f64;
        let current_alpha = if telemetry.load_factor.is_finite() {
            telemetry.load_factor.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let remaining_slots = (proposed_target_alpha - current_alpha).max(0.0) * capacity;
        let t_full = if forecast_peak_ingest_rate > 0.0 {
            remaining_slots / forecast_peak_ingest_rate
        } else {
            f64::INFINITY
        };
        let t_rebuild = (telemetry.reverse_map_entries as f64) / stats.bulk_load_speed.max(1.0);
        let trigger_proactive_rebuild =
            decision.should_rebuild || (t_full <= t_rebuild * proposed_proactive_margin_k);

        let proposal = TuningProposal {
            request_id: telemetry.request_id.clone(),
            proposed_r,
            proposed_m,
            proposed_target_alpha,
            proposed_proactive_margin_k,
            confidence: 1.0,
            trigger_proactive_rebuild,
            model_version: NATIVE_FALLBACK_MODEL_VERSION.to_string(),
            rationale: reason.to_string(),
            forecast_peak_ingest_rate,
        };

        debug_assert_eq!(
            self.guardrail.validate_tuning(&proposal, &self.bounds),
            GuardrailVerdict::Approved,
            "native fallback proposal must always satisfy SafetyGuardrail"
        );
        proposal
    }

    /// Computes a deterministic proportional multi-tenant RAM allocation plan
    /// weighted by key count, query rate, and observed FPR pressure.
    fn compute_native_memory_fallback(
        &self,
        request_id: &str,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> MemoryAllocationPlan {
        if shards.is_empty() {
            return MemoryAllocationPlan {
                request_id: request_id.to_string(),
                allocations: Vec::new(),
                confidence: 1.0,
                model_version: NATIVE_FALLBACK_MODEL_VERSION.to_string(),
            };
        }

        let r_min = FilterConfig::min_remainder_bits(self.bounds.target_epsilon).unwrap_or(12);
        let budget_c = self.bounds.payload_budget_c.max(r_min);
        let max_valid_m = budget_c.saturating_sub(r_min).min(self.bounds.m_max);
        let bytes_per_slot = ((f64::from(budget_c) + 4.0) / 8.0).max(1.0);

        let weights: Vec<f64> = shards
            .iter()
            .map(|s| {
                let keys = (s.key_count as f64).max(1.0);
                let qrate = sanitize_non_negative(s.query_rate);
                let fpr = if s.observed_fpr.is_finite() {
                    s.observed_fpr.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                keys * (1.0 + qrate.ln_1p()) * (1.0 + fpr * 10.0)
            })
            .collect();
        let total_weight: f64 = weights.iter().sum::<f64>().max(1.0);

        let mut remaining_budget = total_budget_bytes;
        let mut allocations = Vec::with_capacity(shards.len());

        for (shard, weight) in shards.iter().zip(weights.iter()) {
            let share = ((weight / total_weight) * (total_budget_bytes as f64)).floor() as u64;
            let allocated = share.min(remaining_budget);
            remaining_budget = remaining_budget.saturating_sub(allocated);

            let max_slots = ((allocated as f64) / bytes_per_slot)
                .floor()
                .max((1u64 << MIN_QUOTIENT_BITS) as f64);
            let recommended_q =
                (max_slots.log2().floor() as u8).clamp(MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS);

            let span = sanitize_non_negative(shard.p95_query_len).max(1.0) as u64;
            let desired_m = FilterConfig::ceil_log2_u64(span).unwrap_or(0);
            let recommended_m = desired_m.min(max_valid_m);
            let recommended_r = budget_c - recommended_m;

            allocations.push(ShardAllocationDecision {
                shard_id: shard.shard_id,
                allocated_memory_bytes: allocated,
                recommended_q,
                recommended_r,
                recommended_m,
            });
        }

        let plan = MemoryAllocationPlan {
            request_id: request_id.to_string(),
            allocations,
            confidence: 1.0,
            model_version: NATIVE_FALLBACK_MODEL_VERSION.to_string(),
        };

        debug_assert_eq!(
            self.guardrail
                .validate_memory_plan(&plan, total_budget_bytes, &self.bounds),
            GuardrailVerdict::Approved,
            "native fallback memory plan must always satisfy SafetyGuardrail"
        );
        plan
    }
}

impl<O: CostOptimizer> TuningAdvisor for GuardedTuningAdvisor<O> {
    fn advise(&self, telemetry: &TelemetrySnapshot) -> AerisResult<TuningProposal> {
        // Fast-path synchronous short-circuit when no sidecar is configured or the
        // circuit breaker is Open, avoiding Tokio runtime hop overhead completely.
        match self.client.as_ref() {
            None => {
                self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(self.compute_native_tuning_fallback(
                    telemetry,
                    "sidecar disabled; using native deterministic optimizer",
                ));
            }
            Some(client) if client.try_fast_short_circuit() => {
                self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    shard_id = telemetry.shard_id,
                    epoch = telemetry.epoch,
                    request_id = %telemetry.request_id,
                    "sidecar circuit breaker open; synchronous native tuning fallback"
                );
                return Ok(self.compute_native_tuning_fallback(
                    telemetry,
                    "sidecar circuit breaker open; immediate native deterministic fallback",
                ));
            }
            Some(_) => {}
        }
        self.run_blocking(self.advise_async(telemetry))
    }

    fn allocate_memory(
        &self,
        epoch: Epoch,
        total_budget_bytes: u64,
        shards: &[ShardTelemetrySnapshot],
    ) -> AerisResult<MemoryAllocationPlan> {
        match self.client.as_ref() {
            None => {
                self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                let seq = self.memory_req_seq.fetch_add(1, Ordering::Relaxed);
                let request_id = format!("mem-e{epoch}-{seq}");
                return Ok(self.compute_native_memory_fallback(
                    &request_id,
                    total_budget_bytes,
                    shards,
                ));
            }
            Some(client) if client.try_fast_short_circuit() => {
                self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .transport_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                let seq = self.memory_req_seq.fetch_add(1, Ordering::Relaxed);
                let request_id = format!("mem-e{epoch}-{seq}");
                tracing::debug!(
                    epoch,
                    request_id = %request_id,
                    "sidecar circuit breaker open; synchronous native memory allocation fallback"
                );
                return Ok(self.compute_native_memory_fallback(
                    &request_id,
                    total_budget_bytes,
                    shards,
                ));
            }
            Some(_) => {}
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
fn validate_advisor_bounds(bounds: &SafetyBounds) -> AerisResult<()> {
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
fn verify_shard_coverage(plan: &MemoryAllocationPlan, shards: &[ShardTelemetrySnapshot]) -> bool {
    if plan.allocations.len() != shards.len() {
        return false;
    }
    let expected: HashSet<ShardId> = shards.iter().map(|s| s.shard_id).collect();
    plan.allocations
        .iter()
        .all(|alloc| expected.contains(&alloc.shard_id))
}

/// Converts a [`TelemetrySnapshot`] into [`WorkloadStats`] for deterministic fallback.
fn telemetry_to_workload_stats(telemetry: &TelemetrySnapshot) -> WorkloadStats {
    let p95 = sanitize_non_negative(telemetry.p95_query_len).max(1.0) as u64;
    let p99 = sanitize_non_negative(telemetry.p99_query_len)
        .max(p95 as f64)
        .max(1.0) as u64;
    let p50 = ((p95 as f64) * 0.5).max(1.0) as u64;
    let bulk_load_speed =
        if telemetry.bulk_load_speed.is_finite() && telemetry.bulk_load_speed > 0.0 {
            telemetry.bulk_load_speed
        } else {
            DEFAULT_BULK_LOAD_SPEED_FALLBACK
        };

    WorkloadStats {
        window_query_count: 1,
        observed_fpr: if telemetry.observed_fpr.is_finite() {
            telemetry.observed_fpr.clamp(0.0, 1.0)
        } else {
            0.0
        },
        p50_query_len: p50,
        p95_query_len: p95,
        p99_query_len: p99,
        effective_query_span: p95,
        clipped_outliers: 0,
        ingestion_rate_ewma: sanitize_non_negative(telemetry.ingestion_rate_ewma),
        bulk_load_speed,
        query_rate: sanitize_non_negative(telemetry.query_rate),
        avg_keys_per_partition: 1.0,
    }
}

#[inline]
fn sanitize_non_negative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}
