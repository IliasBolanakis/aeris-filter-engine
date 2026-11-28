//! Deterministic analytical Discrete Cost Optimizer over `m = log2(P)`.
//!
//! Exhaustively evaluates all discrete memento bit widths `m in 0..=max_m`
//! satisfying the fixed per-slot RAM payload budget `r + m = C` and the
//! monotonic adaptivity remainder floor `r >= r_min(target_epsilon)`.
//!
//! # Performance Guarantee
//! Because `max_m <= 20` (`<= 21` candidate evaluations), [`DiscreteCostOptimizer::optimize`]
//! executes in strictly `O(max_m)` time (`< 5 us`, typically `~150 ns`) with
//! **zero heap allocations**.

#[cfg(test)]
mod tests;

use aeris_types::{
    CostModelParams, CostOptimizer, FilterConfig, OptimizationDecision, WorkloadStats,
    DEFAULT_TARGET_LOAD_FACTOR,
};
use tracing::{trace, warn};

/// Minimum positive denominator used when normalizing relative cost benefit to
/// guard against division by zero.
const MIN_COST_DENOMINATOR_NS: f64 = 1e-9;

/// Maximum shift width for `1u64 << m` without overflowing `u64`.
const MAX_U64_SHIFT_BITS: u8 = 63;

/// Analytical cost breakdown for a single candidate `(r, m)` configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateCost {
    /// Remainder bit width `r = C - m`.
    pub r: u8,
    /// Memento bit width `m = log2(P)`.
    pub m: u8,
    /// Expected number of Quotient Filter partition subqueries for `L = R_eff`.
    pub expected_subqueries: f64,
    /// Per-partition false positive probability bound `p_part(r, alpha)`.
    pub per_partition_fp_rate: f64,
    /// Combined range query false positive probability `1 - (1 - p_part)^subqueries`.
    pub combined_fp_rate: f64,
    /// Total expected query latency in nanoseconds (`CPU probes + FP disk I/O penalty`).
    pub total_cost_ns: f64,
}

/// Deterministic analytical optimizer implementing [`CostOptimizer`].
///
/// Stateless, pure, `Copy`, and zero-allocation; evaluates every valid discrete
/// `(r, m)` partition under fixed payload budget `C = r + m`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiscreteCostOptimizer;

impl DiscreteCostOptimizer {
    /// Creates a new [`DiscreteCostOptimizer`].
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluates the exact analytical cost model for a single candidate `(r, m)`.
    ///
    /// # Equations
    /// Given `P = 2^m` and `L = effective_query_span.max(1)`:
    /// - `subqueries = 1 + (L - 1) / P` if `L <= P`, else `ceil(L / P) + 1`.
    /// - `p_part = clamp((r + 2) * 2^(-(r + 1)) * alpha_est, 0, 1)`.
    /// - `p_fp = 1 - (1 - p_part)^subqueries` (computed via `ln_1p`/`exp_m1` for
    ///   numerical stability at tiny `p_part`).
    /// - `total_cost_ns = subqueries * cpu_probe_ns + p_fp * disk_io_penalty_ns`.
    ///
    /// # Complexity
    /// `O(1)` time, `O(1)` space, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn evaluate_candidate(
        r: u8,
        m: u8,
        effective_query_span: u64,
        alpha_est: f64,
        cost_model: &CostModelParams,
    ) -> CandidateCost {
        let l = effective_query_span.max(1);
        let m_clamped = m.min(MAX_U64_SHIFT_BITS);
        let p = 1u64 << m_clamped;

        // Exact power-of-two arithmetic avoids f64 division and `ceil()` calls:
        // - For `l <= p`: multiplying by `2^-m` is bit-exact in IEEE-754 binary64.
        // - For `l > p`: `ceil(l / 2^m) + 1 == ((l - 1) >> m) + 2`.
        let expected_subqueries = if l <= p {
            1.0 + ((l - 1) as f64) * neg_pow2_exact(m_clamped)
        } else {
            (((l - 1) >> m_clamped) as f64) + 2.0
        };

        let clamped_alpha = alpha_est.clamp(0.0, DEFAULT_TARGET_LOAD_FACTOR);
        let per_partition_fp_rate =
            (((r as f64) + 2.0) * neg_pow2_r_plus_1_exact(r) * clamped_alpha).clamp(0.0, 1.0);

        let combined_fp_rate = combined_fp_probability(per_partition_fp_rate, expected_subqueries);

        let total_cost_ns = expected_subqueries * cost_model.cpu_probe_ns
            + combined_fp_rate * cost_model.disk_io_penalty_ns;

        CandidateCost {
            r,
            m: m_clamped,
            expected_subqueries,
            per_partition_fp_rate,
            combined_fp_rate,
            total_cost_ns,
        }
    }
}

impl CostOptimizer for DiscreteCostOptimizer {
    /// Exhaustively evaluates all discrete `m in 0..=max_m` satisfying
    /// `r = C - m >= r_min(target_epsilon)` and returns an [`OptimizationDecision`].
    ///
    /// # Complexity
    /// `O(max_m)` time (`<= 21` iterations, `< 5 us`), `O(1)` space, zero allocations.
    fn optimize(
        &self,
        stats: &WorkloadStats,
        current: &FilterConfig,
        cost_model: &CostModelParams,
    ) -> OptimizationDecision {
        let span_l = stats.effective_query_span.max(1);
        let alpha_est = current
            .max_load_factor
            .clamp(0.0, DEFAULT_TARGET_LOAD_FACTOR);

        let current_eval =
            Self::evaluate_candidate(current.r, current.m, span_l, alpha_est, cost_model);
        let r_min = resolve_r_min(current);
        let best = find_best_candidate(current, span_l, alpha_est, cost_model, current_eval, r_min);
        build_decision(span_l, current, cost_model, current_eval, best)
    }
}

/// Resolves the monotonic remainder floor `r_min(target_epsilon)`, falling back
/// to `current.r.min(payload_budget_c)` if `target_epsilon` is invalid.
#[inline]
fn resolve_r_min(current: &FilterConfig) -> u8 {
    match FilterConfig::min_remainder_bits(current.target_epsilon) {
        Ok(min_bits) => min_bits,
        Err(err) => {
            warn!(
                target_epsilon = current.target_epsilon,
                error = %err,
                "fallback to current.r as r_min during optimization"
            );
            current.r.min(current.payload_budget_c)
        }
    }
}

/// Evaluates all valid discrete `m` candidates under fixed RAM payload budget
/// `r + m = C` and `r >= r_min`, breaking exact cost ties in favor of `current.m`.
#[inline]
fn find_best_candidate(
    current: &FilterConfig,
    span_l: u64,
    alpha_est: f64,
    cost_model: &CostModelParams,
    current_eval: CandidateCost,
    r_min: u8,
) -> CandidateCost {
    let max_m = cost_model
        .max_m
        .min(current.payload_budget_c)
        .min(MAX_U64_SHIFT_BITS);
    let mut best: Option<CandidateCost> = None;

    if current.payload_budget_c >= r_min {
        let effective_max_m = max_m.min(current.payload_budget_c - r_min);
        for m in 0..=effective_max_m {
            let r = current.payload_budget_c - m;
            let cand = if m == current.m && r == current.r {
                current_eval
            } else {
                DiscreteCostOptimizer::evaluate_candidate(r, m, span_l, alpha_est, cost_model)
            };
            let is_better = match best {
                None => true,
                Some(prev) => {
                    cand.total_cost_ns < prev.total_cost_ns
                        || (cand.total_cost_ns == prev.total_cost_ns && cand.m == current.m)
                }
            };
            if is_better {
                best = Some(cand);
            }
        }
    }

    best.unwrap_or(current_eval)
}

/// Translates the winning candidate into an [`OptimizationDecision`] with direct-scan
/// bypass gating and hysteresis rebuild suppression.
#[inline]
fn build_decision(
    span_l: u64,
    current: &FilterConfig,
    cost_model: &CostModelParams,
    current_eval: CandidateCost,
    best: CandidateCost,
) -> OptimizationDecision {
    let should_bypass_filter = best.total_cost_ns >= cost_model.direct_scan_cost_ns;
    let relative_benefit = (current_eval.total_cost_ns - best.total_cost_ns)
        / current_eval.total_cost_ns.max(MIN_COST_DENOMINATOR_NS);
    let should_rebuild = !should_bypass_filter
        && best.m != current.m
        && relative_benefit >= cost_model.rebuild_benefit_threshold;

    trace!(
        effective_query_span = span_l,
        current_m = current.m,
        current_r = current.r,
        recommended_m = best.m,
        recommended_r = best.r,
        current_cost_ns = current_eval.total_cost_ns,
        estimated_cost_ns = best.total_cost_ns,
        should_bypass_filter,
        should_rebuild,
        "completed discrete cost optimization"
    );

    OptimizationDecision {
        recommended_m: best.m,
        recommended_r: best.r,
        estimated_cost_ns: best.total_cost_ns,
        current_cost_ns: current_eval.total_cost_ns,
        should_bypass_filter,
        should_rebuild,
    }
}

/// Computes exact `2.0_f64.powi(-(m as i32))` for `m <= 63` via IEEE-754 exponent
/// encoding, avoiding libm calls with zero ULP error.
#[inline]
const fn neg_pow2_exact(m: u8) -> f64 {
    debug_assert!(m <= MAX_U64_SHIFT_BITS);
    let biased_exp = 1023_u64 - (m as u64);
    f64::from_bits(biased_exp << 52)
}

/// Computes exact `2.0_f64.powi(-(r as i32) - 1)` for any `r: u8` (`0..=255`) via
/// IEEE-754 exponent encoding (`biased_exp in 767..=1022`, strictly normal).
#[inline]
const fn neg_pow2_r_plus_1_exact(r: u8) -> f64 {
    let biased_exp = 1022_u64 - (r as u64);
    f64::from_bits(biased_exp << 52)
}

/// Computes `1 - (1 - p_part)^subqueries` without catastrophic cancellation when
/// `p_part` is very small.
#[inline]
fn combined_fp_probability(p_part: f64, subqueries: f64) -> f64 {
    if p_part <= 0.0 || subqueries <= 0.0 {
        return 0.0;
    }
    if p_part >= 1.0 {
        return 1.0;
    }
    if subqueries == 1.0 {
        return p_part;
    }
    (-((-p_part).ln_1p() * subqueries).exp_m1()).clamp(0.0, 1.0)
}
