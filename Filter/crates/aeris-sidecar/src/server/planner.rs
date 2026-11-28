//! Deterministic tuning and multi-tenant memory response builders for
//! [`super::ReferenceSidecarServer`].

use aeris_types::{
    FilterConfig, DEFAULT_TARGET_LOAD_FACTOR, MAX_MEMENTO_BITS, MAX_QUOTIENT_BITS,
    MIN_QUOTIENT_BITS,
};

use crate::proto;

/// Model version reported by the reference Rust gRPC sidecar server.
pub const REFERENCE_SERVER_MODEL_VERSION: &str = "rust-reference-sidecar-v1";

/// Builds a guardrail-compliant [`proto::TuningResponse`] from `req`.
pub(super) fn build_normal_tuning_response(req: &proto::TuningRequest) -> proto::TuningResponse {
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

/// Builds an intentionally guardrail-violating [`proto::TuningResponse`] for fault injection.
pub(super) fn build_oob_tuning_response(req: &proto::TuningRequest) -> proto::TuningResponse {
    proto::TuningResponse {
        request_id: req.request_id.clone(),
        proposed_r: 1,
        proposed_m: req.payload_budget_c.saturating_add(5),
        proposed_target_alpha: 0.99,
        proposed_proactive_margin_k: 5.5,
        confidence: 0.95,
        trigger_proactive_rebuild: true,
        model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
        rationale: "injected out-of-bounds proposal".to_string(),
        forecast_peak_ingest_rate: -10.0,
    }
}

/// Builds a guardrail-compliant [`proto::MemoryAllocationResponse`] from `req`.
pub(super) fn build_normal_memory_response(
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

/// Builds an intentionally guardrail-violating [`proto::MemoryAllocationResponse`].
pub(super) fn build_oob_memory_response(
    req: &proto::MemoryAllocationRequest,
) -> proto::MemoryAllocationResponse {
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
    proto::MemoryAllocationResponse {
        request_id: req.request_id.clone(),
        allocations,
        confidence: 0.95,
        model_version: REFERENCE_SERVER_MODEL_VERSION.to_string(),
    }
}
