//! Deterministic analytical fallback optimizer and safe tuning/memory plan
//! synthesis for [`super::GuardedTuningAdvisor`].

use aeris_types::{
    CostModelParams, CostOptimizer, FilterConfig, GuardrailVerdict, MemoryAllocationPlan,
    SafetyBounds, ShardAllocationDecision, ShardTelemetrySnapshot, TelemetrySnapshot,
    TuningProposal, WorkloadStats, DEFAULT_TARGET_LOAD_FACTOR, MAX_QUOTIENT_BITS,
    MIN_QUOTIENT_BITS,
};

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
/// `r = C - m >= r_min(target_epsilon)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeFallbackOptimizer;

impl NativeFallbackOptimizer {
    /// Constructs a new [`NativeFallbackOptimizer`].
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluates the expected query cost (nanoseconds) for a candidate `(r, m)`
    /// pair under `effective_span` and `cost_model`.
    ///
    /// Models partition size `P = 2^m`, expected probes `1 + (L - 1) / P`,
    /// remainder collision rate `2^-r`, and union FP probability via `ln_1p`/`exp_m1`.
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
    /// [`OptimizationDecision`](aeris_types::OptimizationDecision).
    ///
    /// # Complexity
    /// `O(max_m)` time (`<= 21` iterations), `O(1)` space, zero heap allocations.
    fn optimize(
        &self,
        stats: &WorkloadStats,
        current: &FilterConfig,
        cost_model: &CostModelParams,
    ) -> aeris_types::OptimizationDecision {
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

        aeris_types::OptimizationDecision {
            recommended_m: best_m,
            recommended_r: best_r,
            estimated_cost_ns: best_cost_ns,
            current_cost_ns,
            should_bypass_filter,
            should_rebuild,
        }
    }
}

/// Computes a deterministic, guardrail-compliant [`TuningProposal`] using
/// `fallback_optimizer`.
pub(super) fn compute_native_tuning_fallback<O: CostOptimizer>(
    fallback_optimizer: &O,
    guardrail: SafetyGuardrail,
    bounds: &SafetyBounds,
    cost_model: &CostModelParams,
    telemetry: &TelemetrySnapshot,
    reason: &str,
) -> TuningProposal {
    let r_min = FilterConfig::min_remainder_bits(bounds.target_epsilon).unwrap_or(12);
    let budget_c = bounds.payload_budget_c.max(r_min);
    let max_valid_m = budget_c.saturating_sub(r_min).min(bounds.m_max);

    let safe_q = telemetry
        .current_q
        .clamp(MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS);
    let current_cfg =
        build_fallback_filter_config(telemetry, bounds, safe_q, budget_c, max_valid_m);

    let stats = telemetry_to_workload_stats(telemetry);
    let decision = fallback_optimizer.optimize(&stats, &current_cfg, cost_model);

    let proposed_m = decision.recommended_m.min(max_valid_m);
    let proposed_r = decision
        .recommended_r
        .max(r_min)
        .min(budget_c.saturating_sub(proposed_m));

    let proposed_target_alpha =
        DEFAULT_TARGET_LOAD_FACTOR.clamp(bounds.min_alpha, bounds.max_alpha);
    let proposed_proactive_margin_k = NATIVE_DEFAULT_PROACTIVE_MARGIN_K
        .clamp(bounds.min_proactive_margin_k, bounds.max_proactive_margin_k);

    let forecast_peak_ingest_rate = sanitize_non_negative(telemetry.ingestion_rate_ewma);
    let trigger_proactive_rebuild = decision.should_rebuild
        || should_trigger_proactive_rebuild(
            telemetry,
            &stats,
            safe_q,
            proposed_target_alpha,
            proposed_proactive_margin_k,
            forecast_peak_ingest_rate,
        );

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
        guardrail.validate_tuning(&proposal, bounds),
        GuardrailVerdict::Approved,
        "native fallback proposal must always satisfy SafetyGuardrail"
    );
    proposal
}

fn build_fallback_filter_config(
    telemetry: &TelemetrySnapshot,
    bounds: &SafetyBounds,
    safe_q: u8,
    budget_c: u8,
    max_valid_m: u8,
) -> FilterConfig {
    let safe_m = telemetry.current_m.min(max_valid_m);
    let safe_r = budget_c - safe_m;
    let max_range_r = 1u64 << safe_m;
    FilterConfig::new(safe_q, safe_r, max_range_r, budget_c, bounds.target_epsilon).unwrap_or_else(
        |_| {
            FilterConfig::with_minimum_bits(safe_q, 1, bounds.target_epsilon)
                .expect("validated SafetyBounds guarantee a valid minimum FilterConfig")
        },
    )
}

fn should_trigger_proactive_rebuild(
    telemetry: &TelemetrySnapshot,
    stats: &WorkloadStats,
    safe_q: u8,
    target_alpha: f64,
    proactive_margin_k: f64,
    forecast_peak_ingest_rate: f64,
) -> bool {
    let capacity = (1u64 << safe_q) as f64;
    let current_alpha = if telemetry.load_factor.is_finite() {
        telemetry.load_factor.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let remaining_slots = (target_alpha - current_alpha).max(0.0) * capacity;
    let t_full = if forecast_peak_ingest_rate > 0.0 {
        remaining_slots / forecast_peak_ingest_rate
    } else {
        f64::INFINITY
    };
    let t_rebuild = (telemetry.reverse_map_entries as f64) / stats.bulk_load_speed.max(1.0);
    t_full <= t_rebuild * proactive_margin_k
}

/// Computes a deterministic proportional multi-tenant RAM allocation plan
/// weighted by key count, query rate, and observed FPR pressure.
pub(super) fn compute_native_memory_fallback(
    guardrail: SafetyGuardrail,
    bounds: &SafetyBounds,
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

    let r_min = FilterConfig::min_remainder_bits(bounds.target_epsilon).unwrap_or(12);
    let budget_c = bounds.payload_budget_c.max(r_min);
    let max_valid_m = budget_c.saturating_sub(r_min).min(bounds.m_max);
    let bytes_per_slot = ((f64::from(budget_c) + 4.0) / 8.0).max(1.0);

    let weights = compute_shard_weights(shards);
    let total_weight: f64 = weights.iter().sum::<f64>().max(1.0);

    let mut remaining_budget = total_budget_bytes;
    let mut allocations = Vec::with_capacity(shards.len());
    for (shard, weight) in shards.iter().zip(weights.iter()) {
        let share = ((weight / total_weight) * (total_budget_bytes as f64)).floor() as u64;
        let allocated = share.min(remaining_budget);
        remaining_budget = remaining_budget.saturating_sub(allocated);
        allocations.push(build_shard_allocation(
            shard,
            allocated,
            bytes_per_slot,
            budget_c,
            max_valid_m,
        ));
    }

    let plan = MemoryAllocationPlan {
        request_id: request_id.to_string(),
        allocations,
        confidence: 1.0,
        model_version: NATIVE_FALLBACK_MODEL_VERSION.to_string(),
    };

    debug_assert_eq!(
        guardrail.validate_memory_plan(&plan, total_budget_bytes, bounds),
        GuardrailVerdict::Approved,
        "native fallback memory plan must always satisfy SafetyGuardrail"
    );
    plan
}

fn compute_shard_weights(shards: &[ShardTelemetrySnapshot]) -> Vec<f64> {
    shards
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
        .collect()
}

fn build_shard_allocation(
    shard: &ShardTelemetrySnapshot,
    allocated: u64,
    bytes_per_slot: f64,
    budget_c: u8,
    max_valid_m: u8,
) -> ShardAllocationDecision {
    let max_slots = ((allocated as f64) / bytes_per_slot)
        .floor()
        .max((1u64 << MIN_QUOTIENT_BITS) as f64);
    let recommended_q =
        (max_slots.log2().floor() as u8).clamp(MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS);

    let span = sanitize_non_negative(shard.p95_query_len).max(1.0) as u64;
    let desired_m = FilterConfig::ceil_log2_u64(span).unwrap_or(0);
    let recommended_m = desired_m.min(max_valid_m);
    let recommended_r = budget_c - recommended_m;

    ShardAllocationDecision {
        shard_id: shard.shard_id,
        allocated_memory_bytes: allocated,
        recommended_q,
        recommended_r,
        recommended_m,
    }
}

/// Converts a [`TelemetrySnapshot`] into [`WorkloadStats`] for deterministic fallback.
pub(super) fn telemetry_to_workload_stats(telemetry: &TelemetrySnapshot) -> WorkloadStats {
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
pub(super) fn sanitize_non_negative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}
