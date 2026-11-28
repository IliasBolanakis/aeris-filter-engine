//! Targeted unit tests for [`super::GuardedTuningAdvisor`],
//! [`super::NativeFallbackOptimizer`], and bounds/coverage validation.

use aeris_types::{
    CostModelParams, FilterConfig, GuardrailVerdict, MemoryAllocationPlan, SafetyBounds,
    ShardAllocationDecision, ShardTelemetrySnapshot, TelemetrySnapshot, TuningAdvisor,
};

use super::fallback::{compute_native_memory_fallback, compute_native_tuning_fallback};
use super::{
    validate_advisor_bounds, verify_shard_coverage, GuardedTuningAdvisor, NativeFallbackOptimizer,
    NATIVE_FALLBACK_MODEL_VERSION,
};
use crate::guardrail::SafetyGuardrail;

fn sample_bounds() -> SafetyBounds {
    let cfg = FilterConfig::new(14, 15, 32, 20, 1.0 / 256.0).unwrap();
    SafetyBounds::from_filter_config(&cfg)
}

fn sample_telemetry() -> TelemetrySnapshot {
    TelemetrySnapshot {
        request_id: "adv-unit-1".to_string(),
        shard_id: 2,
        epoch: 5,
        timestamp_ms: 1_700_000_000_000,
        current_q: 14,
        current_r: 15,
        current_m: 5,
        payload_budget_c: 20,
        target_epsilon: 1.0 / 256.0,
        load_factor: 0.75,
        ingestion_rate_ewma: 10_000.0,
        query_rate: 5_000.0,
        observed_fpr: 0.001,
        p95_query_len: 32.0,
        p99_query_len: 64.0,
        reverse_map_entries: 10_000,
        bulk_load_speed: 500_000.0,
        recent_query_len_series: vec![16.0, 32.0],
        recent_ingest_rate_series: vec![8_000.0, 10_000.0],
    }
}

#[test]
fn validate_advisor_bounds_rejects_invalid_budget_alpha_and_k() {
    let valid = sample_bounds();
    assert!(validate_advisor_bounds(&valid).is_ok());

    let mut bad_budget = valid.clone();
    bad_budget.payload_budget_c = 4;
    assert!(validate_advisor_bounds(&bad_budget).is_err());

    let mut bad_alpha = valid.clone();
    bad_alpha.min_alpha = 0.9;
    bad_alpha.max_alpha = 0.5;
    assert!(validate_advisor_bounds(&bad_alpha).is_err());

    let mut bad_k = valid;
    bad_k.min_proactive_margin_k = f64::NAN;
    assert!(validate_advisor_bounds(&bad_k).is_err());
}

#[test]
fn verify_shard_coverage_detects_missing_or_extra_shards() {
    let shards = vec![
        ShardTelemetrySnapshot {
            shard_id: 1,
            key_count: 100,
            current_memory_bytes: 1024,
            query_rate: 10.0,
            observed_fpr: 0.001,
            p95_query_len: 16.0,
            load_factor: 0.5,
            current_q: 14,
            current_r: 15,
            current_m: 5,
        },
        ShardTelemetrySnapshot {
            shard_id: 2,
            key_count: 100,
            current_memory_bytes: 1024,
            query_rate: 10.0,
            observed_fpr: 0.001,
            p95_query_len: 16.0,
            load_factor: 0.5,
            current_q: 14,
            current_r: 15,
            current_m: 5,
        },
    ];
    let ok_plan = MemoryAllocationPlan {
        request_id: "p".to_string(),
        allocations: vec![
            ShardAllocationDecision {
                shard_id: 2,
                allocated_memory_bytes: 512,
                recommended_q: 14,
                recommended_r: 15,
                recommended_m: 5,
            },
            ShardAllocationDecision {
                shard_id: 1,
                allocated_memory_bytes: 512,
                recommended_q: 14,
                recommended_r: 15,
                recommended_m: 5,
            },
        ],
        confidence: 0.9,
        model_version: "m".to_string(),
    };
    assert!(verify_shard_coverage(&ok_plan, &shards));

    let wrong_shard_plan = MemoryAllocationPlan {
        allocations: vec![
            ShardAllocationDecision {
                shard_id: 1,
                allocated_memory_bytes: 512,
                recommended_q: 14,
                recommended_r: 15,
                recommended_m: 5,
            },
            ShardAllocationDecision {
                shard_id: 99,
                allocated_memory_bytes: 512,
                recommended_q: 14,
                recommended_r: 15,
                recommended_m: 5,
            },
        ],
        ..ok_plan
    };
    assert!(!verify_shard_coverage(&wrong_shard_plan, &shards));
}

#[test]
fn native_fallback_handles_nan_telemetry_and_empty_shards_safely() {
    let guardrail = SafetyGuardrail::new();
    let bounds = sample_bounds();
    let cost_model = CostModelParams::default();

    let mut telemetry = sample_telemetry();
    telemetry.load_factor = f64::NAN;
    telemetry.ingestion_rate_ewma = f64::NEG_INFINITY;
    telemetry.query_rate = f64::NAN;
    telemetry.observed_fpr = f64::INFINITY;
    telemetry.p95_query_len = -5.0;
    telemetry.p99_query_len = f64::NAN;
    telemetry.bulk_load_speed = 0.0;

    let proposal = compute_native_tuning_fallback(
        &NativeFallbackOptimizer::new(),
        guardrail,
        &bounds,
        &cost_model,
        &telemetry,
        "nan-test",
    );
    assert_eq!(proposal.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        guardrail.validate_tuning(&proposal, &bounds),
        GuardrailVerdict::Approved
    );

    let empty_plan =
        compute_native_memory_fallback(guardrail, &bounds, "empty-req", 1_000_000, &[]);
    assert!(empty_plan.allocations.is_empty());
    assert_eq!(empty_plan.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        guardrail.validate_memory_plan(&empty_plan, 1_000_000, &bounds),
        GuardrailVerdict::Approved
    );
}

#[test]
fn native_only_advisor_satisfies_metrics_conservation_and_health() {
    let advisor = GuardedTuningAdvisor::native_only(
        NativeFallbackOptimizer::new(),
        sample_bounds(),
        CostModelParams::default(),
    )
    .unwrap();
    assert!(!advisor.is_healthy());
    assert!(advisor.client().is_none());

    let p = advisor.advise(&sample_telemetry()).unwrap();
    assert_eq!(p.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    let m = advisor.allocate_memory(1, 4_096_000, &[]).unwrap();
    assert!(m.allocations.is_empty());

    let snap = advisor.metrics();
    assert_eq!(snap.total_calls, 2);
    assert_eq!(snap.transport_fallbacks, 2);
    assert_eq!(
        snap.total_calls,
        snap.sidecar_accepted + snap.guardrail_rejections + snap.transport_fallbacks
    );
}
