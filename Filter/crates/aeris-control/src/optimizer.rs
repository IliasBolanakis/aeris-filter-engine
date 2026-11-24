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

        let r_min = match FilterConfig::min_remainder_bits(current.target_epsilon) {
            Ok(min_bits) => min_bits,
            Err(err) => {
                warn!(
                    target_epsilon = current.target_epsilon,
                    error = %err,
                    "fallback to current.r as r_min during optimization"
                );
                current.r.min(current.payload_budget_c)
            }
        };

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
                    Self::evaluate_candidate(r, m, span_l, alpha_est, cost_model)
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

        let best = best.unwrap_or(current_eval);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn analytical_subquery_and_fp_formulas_match_exact_equations() {
        let model = CostModelParams::default();

        // L <= P: L = 16, m = 5 (P = 32) -> subqueries = 1 + 15/32 = 1.46875
        let c1 = DiscreteCostOptimizer::evaluate_candidate(15, 5, 16, 0.90, &model);
        assert!((c1.expected_subqueries - (1.0 + 15.0 / 32.0)).abs() < 1e-12);

        let expected_p_part = (15.0 + 2.0) * 2.0_f64.powi(-16) * 0.90;
        assert!((c1.per_partition_fp_rate - expected_p_part).abs() < 1e-15);

        let expected_p_fp = 1.0 - (1.0 - expected_p_part).powf(c1.expected_subqueries);
        assert!((c1.combined_fp_rate - expected_p_fp).abs() < 1e-12);

        let expected_cost =
            c1.expected_subqueries * model.cpu_probe_ns + expected_p_fp * model.disk_io_penalty_ns;
        assert!((c1.total_cost_ns - expected_cost).abs() < 1e-9);

        // L > P: L = 100, m = 5 (P = 32) -> ceil(100/32) + 1 = 4 + 1 = 5
        let c2 = DiscreteCostOptimizer::evaluate_candidate(15, 5, 100, 0.90, &model);
        assert!((c2.expected_subqueries - 5.0).abs() < 1e-12);

        // Point query L = 1 -> subqueries = 1.0 for any m >= 0
        let c3 = DiscreteCostOptimizer::evaluate_candidate(20, 0, 1, 0.90, &model);
        assert!((c3.expected_subqueries - 1.0).abs() < 1e-12);
    }

    #[test]
    fn optimizer_respects_budget_and_monotonic_remainder_floor() {
        let opt = DiscreteCostOptimizer::new();
        let model = CostModelParams::default();
        // eps = 2^-8 -> r_min = 12. C = 20 -> max valid m = 8.
        let current = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();

        // Large effective query span R = 256 -> optimal m should increase to 8 (r = 12), never exceeding C - r_min.
        let stats = WorkloadStats {
            effective_query_span: 256,
            ..WorkloadStats::default()
        };
        let decision = opt.optimize(&stats, &current, &model);
        assert_eq!(
            decision.recommended_m + decision.recommended_r,
            current.payload_budget_c
        );
        assert!(
            decision.recommended_r
                >= FilterConfig::min_remainder_bits(current.target_epsilon).unwrap()
        );
        assert_eq!(decision.recommended_m, 8);
        assert!(!decision.should_bypass_filter);
        assert!(decision.should_rebuild);
        assert!(decision.estimated_cost_ns < decision.current_cost_ns);
    }

    #[test]
    fn optimizer_prefers_smaller_m_for_point_queries() {
        let opt = DiscreteCostOptimizer::new();
        let model = CostModelParams::default();
        // Currently configured for R = 256 (m = 8, r = 12, C = 20).
        let current = FilterConfig::new(12, 12, 256, 20, 1.0 / 256.0).unwrap();
        let stats = WorkloadStats {
            effective_query_span: 1,
            ..WorkloadStats::default()
        };
        let decision = opt.optimize(&stats, &current, &model);
        // Point queries always take 1 subquery regardless of m, so m = 0 maximizes r = 20 (minimizing FP penalty).
        assert_eq!(decision.recommended_m, 0);
        assert_eq!(decision.recommended_r, 20);
        assert!(decision.should_rebuild);
        assert!(!decision.should_bypass_filter);
    }

    #[test]
    fn optimizer_triggers_bypass_for_massive_range_scans() {
        let opt = DiscreteCostOptimizer::new();
        let model = CostModelParams::default();
        let current = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();
        let stats = WorkloadStats {
            effective_query_span: 1_000_000,
            ..WorkloadStats::default()
        };
        let decision = opt.optimize(&stats, &current, &model);
        assert!(decision.should_bypass_filter);
        assert!(
            !decision.should_rebuild,
            "should_rebuild must be false when filter is bypassed"
        );
        assert!(decision.estimated_cost_ns >= model.direct_scan_cost_ns);
    }

    #[test]
    fn optimizer_hysteresis_blocks_marginal_rebuild() {
        let opt = DiscreteCostOptimizer::new();
        let model = CostModelParams {
            rebuild_benefit_threshold: 0.99,
            ..CostModelParams::default()
        };
        let current = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();
        let stats = WorkloadStats {
            effective_query_span: 48,
            ..WorkloadStats::default()
        };
        let decision = opt.optimize(&stats, &current, &model);
        assert!(!decision.should_rebuild);
    }

    #[test]
    fn optimizer_executes_under_five_microseconds_per_call() {
        let opt = DiscreteCostOptimizer::new();
        let model = CostModelParams::default();
        let current = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();
        let stats = WorkloadStats {
            effective_query_span: 128,
            ..WorkloadStats::default()
        };

        // Warmup
        for _ in 0..100 {
            let _ = opt.optimize(&stats, &current, &model);
        }

        let iters = 10_000u32;
        let start = Instant::now();
        let mut sink = 0u64;
        for _ in 0..iters {
            let d = opt.optimize(&stats, &current, &model);
            sink = sink.wrapping_add(u64::from(d.recommended_m));
        }
        let avg_ns = start.elapsed().as_nanos() / u128::from(iters);
        assert!(sink > 0);
        // Even in unoptimized debug builds, 21 iterations execute well under 50 us;
        // check generous bound in debug and strict < 5 us bound in release.
        #[cfg(not(debug_assertions))]
        assert!(
            avg_ns < 5_000,
            "DiscreteCostOptimizer exceeded 5 us target: {avg_ns} ns"
        );
        #[cfg(debug_assertions)]
        assert!(
            avg_ns < 50_000,
            "DiscreteCostOptimizer unexpectedly slow in debug build: {avg_ns} ns"
        );
    }

    #[test]
    fn ieee754_neg_pow2_helpers_match_powi_bit_for_bit() {
        for m in 0..=MAX_U64_SHIFT_BITS {
            assert_eq!(
                neg_pow2_exact(m).to_bits(),
                2.0_f64.powi(-(m as i32)).to_bits(),
                "neg_pow2_exact mismatch at m={m}"
            );
        }
        for r in 0..=u8::MAX {
            assert_eq!(
                neg_pow2_r_plus_1_exact(r).to_bits(),
                2.0_f64.powi(-(r as i32) - 1).to_bits(),
                "neg_pow2_r_plus_1_exact mismatch at r={r}"
            );
        }
    }

    #[test]
    fn evaluate_candidate_matches_reference_across_boundary_spans_and_m() {
        let model = CostModelParams::default();
        for m in 0..=20u8 {
            let p = 1u64 << m;
            let spans = [
                1u64,
                p.saturating_sub(1).max(1),
                p,
                p.saturating_add(1),
                p.saturating_mul(2),
                p.saturating_mul(2).saturating_add(1),
                u64::MAX,
            ];
            for &span in &spans {
                let r = 20 - m;
                let c = DiscreteCostOptimizer::evaluate_candidate(r, m, span, 0.90, &model);
                let ref_sub = if span <= p {
                    1.0 + ((span - 1) as f64) / (p as f64)
                } else {
                    ((span as f64) / (p as f64)).ceil() + 1.0
                };
                let rel_err = (c.expected_subqueries - ref_sub).abs() / ref_sub.max(1.0);
                assert!(
                    rel_err < 1e-12,
                    "subquery mismatch at m={m}, span={span}: got {}, ref {ref_sub}",
                    c.expected_subqueries
                );
                assert!(c.total_cost_ns.is_finite() && c.total_cost_ns >= 0.0);
            }
        }
    }

    #[test]
    fn ab1_shifting_r_transition_sequence_matches_4_6_5_8() {
        use crate::estimator::{EstimatorConfig, SlidingWindowEstimator};
        use aeris_types::{QueryRange, WorkloadEstimator};

        let mut cfg = FilterConfig::new(21, 13, 16, 17, 1.0 / 16.0).unwrap();
        let mut est =
            SlidingWindowEstimator::with_initial_span(EstimatorConfig::default(), cfg.max_range_r)
                .unwrap();
        let opt = DiscreteCostOptimizer::new();
        let model = CostModelParams::default();

        let mut transitions: Vec<(u8, u8, u64)> = Vec::new();
        for &phase_r in &[16u64, 256, 32, 1024] {
            for i in 0..10_000u64 {
                let q = QueryRange::with_len(i * 4096, phase_r).unwrap();
                est.record_query(q, false);
                if (i + 1) % 256 == 0 {
                    let stats = est.snapshot();
                    let dec = opt.optimize(&stats, &cfg, &model);
                    if dec.should_rebuild {
                        transitions.push((cfg.m, dec.recommended_m, phase_r));
                        cfg = cfg.reconfigure_for_m(dec.recommended_m).unwrap();
                    }
                }
            }
        }

        assert_eq!(
            transitions,
            vec![(4, 6, 256), (6, 5, 32), (5, 8, 1024)],
            "unexpected Dynamic-R transition trajectory: {transitions:?}"
        );
    }
}
