//! Unit tests for [`super::DiscreteCostOptimizer`].

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
    let current = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();

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
        decision.recommended_r >= FilterConfig::min_remainder_bits(current.target_epsilon).unwrap()
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
    let current = FilterConfig::new(12, 12, 256, 20, 1.0 / 256.0).unwrap();
    let stats = WorkloadStats {
        effective_query_span: 1,
        ..WorkloadStats::default()
    };
    let decision = opt.optimize(&stats, &current, &model);
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

#[test]
fn resolve_r_min_fallback_and_combined_fp_edge_cases() {
    let opt = DiscreteCostOptimizer::new();
    let model = CostModelParams::default();
    let mut cfg = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();
    cfg.target_epsilon = 0.0;

    let stats = WorkloadStats {
        effective_query_span: 1,
        ..WorkloadStats::default()
    };
    let decision = opt.optimize(&stats, &cfg, &model);
    assert_eq!(decision.recommended_m + decision.recommended_r, 20);
    assert!(decision.recommended_r >= cfg.r);

    assert_eq!(combined_fp_probability(0.0, 4.0), 0.0);
    assert_eq!(combined_fp_probability(1.0, 4.0), 1.0);
    assert_eq!(combined_fp_probability(0.25, 1.0), 0.25);
}
