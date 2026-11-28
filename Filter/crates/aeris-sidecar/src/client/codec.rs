//! Protobuf <-> domain DTO conversions with strict narrowing and idempotency checks.

use aeris_types::{
    AerisError, AerisResult, Epoch, MemoryAllocationPlan, ShardAllocationDecision,
    ShardTelemetrySnapshot, TelemetrySnapshot, TuningProposal,
};

use crate::proto;

/// Converts a domain [`TelemetrySnapshot`] into a protobuf [`proto::TuningRequest`].
pub(crate) fn telemetry_to_proto(
    request_id: &str,
    telemetry: &TelemetrySnapshot,
) -> proto::TuningRequest {
    proto::TuningRequest {
        request_id: request_id.to_string(),
        shard_id: telemetry.shard_id,
        epoch: telemetry.epoch,
        timestamp_ms: telemetry.timestamp_ms,
        current_q: u32::from(telemetry.current_q),
        current_r: u32::from(telemetry.current_r),
        current_m: u32::from(telemetry.current_m),
        payload_budget_c: u32::from(telemetry.payload_budget_c),
        target_epsilon: telemetry.target_epsilon,
        load_factor: telemetry.load_factor,
        ingestion_rate_ewma: telemetry.ingestion_rate_ewma,
        query_rate: telemetry.query_rate,
        observed_fpr: telemetry.observed_fpr,
        p95_query_len: telemetry.p95_query_len,
        p99_query_len: telemetry.p99_query_len,
        reverse_map_entries: telemetry.reverse_map_entries,
        bulk_load_speed: telemetry.bulk_load_speed,
        recent_query_len_series: telemetry.recent_query_len_series.clone(),
        recent_ingest_rate_series: telemetry.recent_ingest_rate_series.clone(),
    }
}

/// Converts a protobuf [`proto::TuningResponse`] into a domain [`TuningProposal`]
/// with strict `u32 -> u8` narrowing checks.
pub(crate) fn proto_to_tuning_proposal(
    expected_request_id: &str,
    resp: proto::TuningResponse,
) -> AerisResult<TuningProposal> {
    let request_id = resolve_response_request_id(resp.request_id, expected_request_id)?;
    let proposed_r = narrow_u8("proposed_r", resp.proposed_r, None)?;
    let proposed_m = narrow_u8("proposed_m", resp.proposed_m, None)?;

    Ok(TuningProposal {
        request_id,
        proposed_r,
        proposed_m,
        proposed_target_alpha: resp.proposed_target_alpha,
        proposed_proactive_margin_k: resp.proposed_proactive_margin_k,
        confidence: resp.confidence,
        trigger_proactive_rebuild: resp.trigger_proactive_rebuild,
        model_version: resp.model_version,
        rationale: resp.rationale,
        forecast_peak_ingest_rate: resp.forecast_peak_ingest_rate,
    })
}

/// Converts shard telemetry slices into a protobuf [`proto::MemoryAllocationRequest`].
pub(crate) fn shards_to_proto(
    request_id: &str,
    epoch: Epoch,
    total_budget_bytes: u64,
    shards: &[ShardTelemetrySnapshot],
) -> proto::MemoryAllocationRequest {
    proto::MemoryAllocationRequest {
        request_id: request_id.to_string(),
        epoch,
        total_memory_budget_bytes: total_budget_bytes,
        shards: shards
            .iter()
            .map(|s| proto::ShardTelemetry {
                shard_id: s.shard_id,
                key_count: s.key_count,
                current_memory_bytes: s.current_memory_bytes,
                query_rate: s.query_rate,
                observed_fpr: s.observed_fpr,
                p95_query_len: s.p95_query_len,
                load_factor: s.load_factor,
                current_q: u32::from(s.current_q),
                current_r: u32::from(s.current_r),
                current_m: u32::from(s.current_m),
            })
            .collect(),
    }
}

/// Converts a protobuf [`proto::MemoryAllocationResponse`] into a domain
/// [`MemoryAllocationPlan`] with strict `u32 -> u8` narrowing checks.
pub(crate) fn proto_to_memory_plan(
    expected_request_id: &str,
    resp: proto::MemoryAllocationResponse,
) -> AerisResult<MemoryAllocationPlan> {
    let request_id = resolve_response_request_id(resp.request_id, expected_request_id)?;
    let mut allocations = Vec::with_capacity(resp.allocations.len());
    for alloc in resp.allocations {
        let shard = Some(alloc.shard_id);
        let recommended_q = narrow_u8("recommended_q", alloc.recommended_q, shard)?;
        let recommended_r = narrow_u8("recommended_r", alloc.recommended_r, shard)?;
        let recommended_m = narrow_u8("recommended_m", alloc.recommended_m, shard)?;

        allocations.push(ShardAllocationDecision {
            shard_id: alloc.shard_id,
            allocated_memory_bytes: alloc.allocated_memory_bytes,
            recommended_q,
            recommended_r,
            recommended_m,
        });
    }

    Ok(MemoryAllocationPlan {
        request_id,
        allocations,
        confidence: resp.confidence,
        model_version: resp.model_version,
    })
}

fn resolve_response_request_id(actual: String, expected: &str) -> AerisResult<String> {
    if !actual.is_empty() && actual != expected {
        return Err(AerisError::SidecarUnavailable(format!(
            "sidecar returned mismatched request_id '{actual}' (expected '{expected}')"
        )));
    }
    Ok(if actual.is_empty() {
        expected.to_string()
    } else {
        actual
    })
}

fn narrow_u8(field: &str, value: u32, shard_id: Option<u64>) -> AerisResult<u8> {
    u8::try_from(value).map_err(|_| {
        let reason = match shard_id {
            Some(id) => format!("shard {id}: {field} ({value}) exceeds u8::MAX"),
            None => format!("{field} ({value}) exceeds u8::MAX"),
        };
        AerisError::GuardrailRejected { reason }
    })
}
