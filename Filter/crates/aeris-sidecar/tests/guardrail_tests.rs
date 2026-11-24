//! Exhaustive unit tests for [`SafetyGuardrail`], [`SidecarEndpoint`],
//! [`ClientConfig`], deterministic retry jitter, and [`NativeFallbackOptimizer`].

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use aeris_sidecar::{
    compute_retry_delay, ClientConfig, NativeFallbackOptimizer, SafetyGuardrail, SidecarEndpoint,
    MAX_RETRY_BACKOFF_MS,
};
use aeris_types::{
    CostModelParams, CostOptimizer, FilterConfig, GuardrailVerdict, MemoryAllocationPlan,
    SafetyBounds, ShardAllocationDecision, TuningProposal, WorkloadStats,
};

fn default_bounds() -> SafetyBounds {
    // eps = 2^-8 => r_min = 12, C = 20, m_max = 20, alpha in [0.50, 0.90], k in [1.0, 3.0], conf >= 0.80
    let cfg = FilterConfig::new(14, 15, 32, 20, 1.0 / 256.0).expect("valid FilterConfig");
    SafetyBounds::from_filter_config(&cfg)
}

fn valid_proposal() -> TuningProposal {
    TuningProposal {
        request_id: "req-1".to_string(),
        proposed_r: 14,
        proposed_m: 6,
        proposed_target_alpha: 0.85,
        proposed_proactive_margin_k: 1.5,
        confidence: 0.90,
        trigger_proactive_rebuild: false,
        model_version: "test-model-v1".to_string(),
        rationale: "valid test proposal".to_string(),
        forecast_peak_ingest_rate: 12_500.0,
    }
}

fn valid_memory_plan() -> MemoryAllocationPlan {
    MemoryAllocationPlan {
        request_id: "mem-1".to_string(),
        allocations: vec![
            ShardAllocationDecision {
                shard_id: 1,
                allocated_memory_bytes: 6_000_000,
                recommended_q: 16,
                recommended_r: 14,
                recommended_m: 6,
            },
            ShardAllocationDecision {
                shard_id: 2,
                allocated_memory_bytes: 4_000_000,
                recommended_q: 15,
                recommended_r: 15,
                recommended_m: 5,
            },
        ],
        confidence: 0.88,
        model_version: "test-bandit-v1".to_string(),
    }
}

fn assert_rejected_contains(verdict: GuardrailVerdict, needle: &str) {
    match verdict {
        GuardrailVerdict::Approved => {
            panic!("expected GuardrailVerdict::Rejected containing '{needle}', got Approved");
        }
        GuardrailVerdict::Rejected { reasons } => {
            assert!(
                reasons.iter().any(|r| r.contains(needle)),
                "expected rejection reasons to contain '{needle}', got {reasons:?}"
            );
        }
    }
}

#[test]
fn guardrail_approves_valid_and_exact_boundary_proposals() {
    let guardrail = SafetyGuardrail::new();
    let bounds = default_bounds();

    assert_eq!(
        guardrail.validate_tuning(&valid_proposal(), &bounds),
        GuardrailVerdict::Approved
    );

    // Exact lower boundaries: r == r_min (12), m == 0, alpha == 0.50, k == 1.0, conf == 0.80, rate == 0.0
    let lower_boundary = TuningProposal {
        proposed_r: 12,
        proposed_m: 0,
        proposed_target_alpha: 0.50,
        proposed_proactive_margin_k: 1.0,
        confidence: 0.80,
        forecast_peak_ingest_rate: 0.0,
        ..valid_proposal()
    };
    assert_eq!(
        guardrail.validate_tuning(&lower_boundary, &bounds),
        GuardrailVerdict::Approved
    );

    // Exact upper boundaries: r + m == C (12 + 8 == 20), alpha == 0.90, k == 3.0, conf == 1.0
    let upper_boundary = TuningProposal {
        proposed_r: 12,
        proposed_m: 8,
        proposed_target_alpha: 0.90,
        proposed_proactive_margin_k: 3.0,
        confidence: 1.0,
        forecast_peak_ingest_rate: 1_000_000.0,
        ..valid_proposal()
    };
    assert_eq!(
        guardrail.validate_tuning(&upper_boundary, &bounds),
        GuardrailVerdict::Approved
    );
}

#[test]
fn guardrail_rejects_every_single_out_of_bounds_tuning_parameter() {
    let guardrail = SafetyGuardrail::new();
    let mut bounds = default_bounds();
    bounds.m_max = 8;

    // 1. r < r_min(eps) (11 < 12)
    let mut p = valid_proposal();
    p.proposed_r = 11;
    assert_rejected_contains(guardrail.validate_tuning(&p, &bounds), "r_min");

    // 2. m > m_max (9 > 8)
    let mut p = valid_proposal();
    p.proposed_r = 12;
    p.proposed_m = 9;
    assert_rejected_contains(guardrail.validate_tuning(&p, &bounds), "m_max");

    // 3. r + m > C (15 + 6 = 21 > 20)
    let mut p = valid_proposal();
    p.proposed_r = 15;
    p.proposed_m = 6;
    assert_rejected_contains(guardrail.validate_tuning(&p, &bounds), "payload_budget_c");

    // 3b. Overflow-safe u8 + u8 > C (200 + 200 = 400 > 20, must NOT wrap u8!)
    let mut p = valid_proposal();
    p.proposed_r = 200;
    p.proposed_m = 200;
    assert_rejected_contains(guardrail.validate_tuning(&p, &bounds), "payload_budget_c");

    // 4. alpha < 0.50
    let mut p = valid_proposal();
    p.proposed_target_alpha = 0.4999;
    assert_rejected_contains(
        guardrail.validate_tuning(&p, &bounds),
        "proposed_target_alpha",
    );

    // 5. alpha > 0.90
    let mut p = valid_proposal();
    p.proposed_target_alpha = 0.9001;
    assert_rejected_contains(
        guardrail.validate_tuning(&p, &bounds),
        "proposed_target_alpha",
    );

    // 6. alpha = NaN / +Inf / -Inf
    for bad_alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = valid_proposal();
        p.proposed_target_alpha = bad_alpha;
        assert_rejected_contains(
            guardrail.validate_tuning(&p, &bounds),
            "proposed_target_alpha",
        );
    }

    // 7. k < 1.0
    let mut p = valid_proposal();
    p.proposed_proactive_margin_k = 0.99;
    assert_rejected_contains(
        guardrail.validate_tuning(&p, &bounds),
        "proposed_proactive_margin_k",
    );

    // 8. k > 3.0
    let mut p = valid_proposal();
    p.proposed_proactive_margin_k = 3.01;
    assert_rejected_contains(
        guardrail.validate_tuning(&p, &bounds),
        "proposed_proactive_margin_k",
    );

    // 9. k = NaN / Inf
    for bad_k in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = valid_proposal();
        p.proposed_proactive_margin_k = bad_k;
        assert_rejected_contains(
            guardrail.validate_tuning(&p, &bounds),
            "proposed_proactive_margin_k",
        );
    }

    // 10. confidence < min_confidence, > 1.0, NaN
    for bad_conf in [0.799, -0.1, 1.01, f64::NAN, f64::INFINITY] {
        let mut p = valid_proposal();
        p.confidence = bad_conf;
        assert_rejected_contains(guardrail.validate_tuning(&p, &bounds), "confidence");
    }

    // 11. forecast_peak_ingest_rate < 0.0, NaN, Inf
    for bad_rate in [-0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = valid_proposal();
        p.forecast_peak_ingest_rate = bad_rate;
        assert_rejected_contains(
            guardrail.validate_tuning(&p, &bounds),
            "forecast_peak_ingest_rate",
        );
    }
}

#[test]
fn guardrail_collects_all_simultaneous_violations_without_short_circuiting() {
    let guardrail = SafetyGuardrail::new();
    let mut bounds = default_bounds();
    bounds.m_max = 8;

    let bad = TuningProposal {
        request_id: "multi-bad".to_string(),
        proposed_r: 5,
        proposed_m: 18,
        proposed_target_alpha: f64::NAN,
        proposed_proactive_margin_k: 4.5,
        confidence: 0.20,
        trigger_proactive_rebuild: true,
        model_version: "bad".to_string(),
        rationale: "all violated".to_string(),
        forecast_peak_ingest_rate: -100.0,
    };

    match guardrail.validate_tuning(&bad, &bounds) {
        GuardrailVerdict::Approved => panic!("must reject multi-violation proposal"),
        GuardrailVerdict::Rejected { reasons } => {
            // r < r_min, m > m_max, r + m > C, alpha NaN, k > 3.0, conf < 0.80, rate < 0
            assert_eq!(reasons.len(), 7, "got reasons: {reasons:?}");
        }
    }
}

#[test]
fn guardrail_validates_memory_allocation_plans_exhaustively() {
    let guardrail = SafetyGuardrail::new();
    let bounds = default_bounds();

    // Valid exact-budget plan
    let plan = valid_memory_plan();
    assert_eq!(
        guardrail.validate_memory_plan(&plan, 10_000_000, &bounds),
        GuardrailVerdict::Approved
    );

    // Over budget by 1 byte
    assert_rejected_contains(
        guardrail.validate_memory_plan(&plan, 9_999_999, &bounds),
        "exceeds total_budget_bytes",
    );

    // u64 wraparound attempt: u64::MAX + 2 must NOT wrap to 1 and pass!
    let mut wrap_plan = valid_memory_plan();
    wrap_plan.allocations[0].allocated_memory_bytes = u64::MAX;
    wrap_plan.allocations[1].allocated_memory_bytes = 2;
    assert_rejected_contains(
        guardrail.validate_memory_plan(&wrap_plan, 100, &bounds),
        "exceeds total_budget_bytes",
    );

    // Duplicate shard_id
    let mut dup_plan = valid_memory_plan();
    dup_plan.allocations[1].shard_id = 1;
    assert_rejected_contains(
        guardrail.validate_memory_plan(&dup_plan, 20_000_000, &bounds),
        "duplicate shard_id",
    );

    // q below 6 and above 40
    for bad_q in [0, 5, 41, 64] {
        let mut q_plan = valid_memory_plan();
        q_plan.allocations[0].recommended_q = bad_q;
        assert_rejected_contains(
            guardrail.validate_memory_plan(&q_plan, 20_000_000, &bounds),
            "recommended_q",
        );
    }

    // Shard r < r_min
    let mut r_plan = valid_memory_plan();
    r_plan.allocations[0].recommended_r = 10;
    assert_rejected_contains(
        guardrail.validate_memory_plan(&r_plan, 20_000_000, &bounds),
        "recommended_r",
    );

    // Shard r + m > C
    let mut sum_plan = valid_memory_plan();
    sum_plan.allocations[1].recommended_r = 16;
    sum_plan.allocations[1].recommended_m = 6;
    assert_rejected_contains(
        guardrail.validate_memory_plan(&sum_plan, 20_000_000, &bounds),
        "payload_budget_c",
    );

    // Low confidence
    let mut conf_plan = valid_memory_plan();
    conf_plan.confidence = 0.50;
    assert_rejected_contains(
        guardrail.validate_memory_plan(&conf_plan, 20_000_000, &bounds),
        "confidence",
    );
}

#[test]
fn transport_validation_and_deterministic_retry_jitter() {
    // Loopback v4 and v6 pass
    assert!(SidecarEndpoint::loopback_v4(50051).validate().is_ok());
    assert!(
        SidecarEndpoint::LoopbackTcp(SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 50051))
            .validate()
            .is_ok()
    );

    // Non-loopback IPv4 rejected
    assert!(SidecarEndpoint::LoopbackTcp(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
        50051
    ))
    .validate()
    .is_err());
    assert!(SidecarEndpoint::LoopbackTcp(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10)),
        50051
    ))
    .validate()
    .is_err());

    // Zero timeout rejected
    let bad_cfg =
        ClientConfig::new(SidecarEndpoint::loopback_v4(50051)).with_rpc_timeout(Duration::ZERO);
    assert!(bad_cfg.validate().is_err());

    // Deterministic jitter is reproducible and strictly bounded
    let d1 = compute_retry_delay(25, 0, 42, 7);
    let d2 = compute_retry_delay(25, 0, 42, 7);
    assert_eq!(d1, d2);
    assert!(d1 >= Duration::from_millis(12) && d1 <= Duration::from_millis(25));

    let capped = compute_retry_delay(1_000, 10, 42, 1);
    assert!(capped <= Duration::from_millis(MAX_RETRY_BACKOFF_MS));
    assert_eq!(compute_retry_delay(0, 3, 42, 1), Duration::ZERO);
}

#[test]
fn native_fallback_optimizer_balances_probes_and_remainder_within_budget() {
    let opt = NativeFallbackOptimizer::new();
    let cfg = FilterConfig::new(14, 15, 32, 20, 1.0 / 256.0).expect("valid config");
    let cost_model = CostModelParams::default();

    // Point workload (span = 1): should prefer small m (0) and max r (20) to minimize FPR.
    let point_stats = WorkloadStats {
        effective_query_span: 1,
        p95_query_len: 1,
        ..WorkloadStats::default()
    };
    let d_point = opt.optimize(&point_stats, &cfg, &cost_model);
    assert_eq!(d_point.recommended_m, 0);
    assert_eq!(d_point.recommended_r, 20);
    assert_eq!(
        d_point.recommended_r + d_point.recommended_m,
        cfg.payload_budget_c
    );

    // Wide range workload (span = 1024): should increase m toward max valid m (20 - 12 = 8).
    let wide_stats = WorkloadStats {
        effective_query_span: 1024,
        p95_query_len: 1024,
        ..WorkloadStats::default()
    };
    let d_wide = opt.optimize(&wide_stats, &cfg, &cost_model);
    assert!(d_wide.recommended_m > d_point.recommended_m);
    assert!(d_wide.recommended_r >= FilterConfig::min_remainder_bits(cfg.target_epsilon).unwrap());
    assert_eq!(
        d_wide.recommended_r + d_wide.recommended_m,
        cfg.payload_budget_c
    );
}
