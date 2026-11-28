"""Request/response DTO mapping and control-policy helpers for the gRPC server."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bandit import (
    DEFAULT_M_MAX,
    ShardAllocationPlanItem,
    ShardContext,
    min_remainder_bits,
)
from .forecaster import ForecastResult
from .proto import control_plane_pb2

DEFAULT_CPU_PROBE_NS: float = 35.0
DEFAULT_DISK_IO_PENALTY_NS: float = 50_000.0
MIN_ALPHA: float = 0.50
MAX_ALPHA: float = 0.90
MIN_PROACTIVE_K: float = 1.0
MAX_PROACTIVE_K: float = 3.0


@dataclass(frozen=True)
class TuningPolicyProposal:
    """Computed tuning policy parameters derived from forecast and request state."""

    proposed_r: int
    proposed_m: int
    proposed_alpha: float
    proposed_k: float
    trigger_rebuild: bool
    peak_rate: float
    spike_ratio: float


def select_optimal_rm(
    forecast_p95_len: float,
    payload_budget_c: int,
    target_epsilon: float,
    m_max: int = DEFAULT_M_MAX,
) -> tuple[int, int]:
    """Selects discrete ``(r, m)`` minimizing expected probe and FPR disk-I/O cost.

    Args:
        forecast_p95_len: Forecasted 95th-percentile range query span.
        payload_budget_c: Shared payload bit budget ``C = r + m``.
        target_epsilon: Target false-positive rate floor ``epsilon in (0, 0.5)``.
        m_max: Hard upper bound on range-extension bits ``m``.

    Returns:
        Tuple ``(best_r, best_m)`` satisfying ``best_r >= r_min(epsilon)``,
        ``0 <= best_m <= m_max``, and ``best_r + best_m == max(payload_budget_c, r_min)``.
    """
    eps = (
        target_epsilon
        if (math.isfinite(target_epsilon) and 0.0 < target_epsilon < 0.5)
        else (1.0 / 256.0)
    )
    r_min = min_remainder_bits(eps)
    budget_c = max(int(payload_budget_c), r_min)
    max_valid_m = min(int(m_max), budget_c - r_min)
    span = max(float(forecast_p95_len) if math.isfinite(forecast_p95_len) else 1.0, 1.0)

    best_m = 0
    best_r = budget_c
    best_cost = math.inf
    for cand_m in range(0, max_valid_m + 1):
        cand_r = budget_c - cand_m
        if cand_r < r_min:
            continue
        partition_size = float(1 << cand_m)
        probes = 1.0 + (span - 1.0) / partition_size
        p_probe = 2.0 ** (-cand_r)
        p_fp = 1.0 - math.exp(probes * math.log1p(-p_probe)) if p_probe < 1.0 else 1.0
        cost = probes * DEFAULT_CPU_PROBE_NS + p_fp * DEFAULT_DISK_IO_PENALTY_NS
        if cost < best_cost:
            best_cost = cost
            best_m = cand_m
            best_r = cand_r

    return best_r, best_m


def compute_tuning_policy(
    request: control_plane_pb2.TuningRequest,
    forecast: ForecastResult,
) -> TuningPolicyProposal:
    """Computes geometry, load-factor target, and proactive rebuild decision.

    Args:
        request: Incoming gRPC ``TuningRequest`` telemetry snapshot.
        forecast: Horizon forecast produced by :class:`SeasonalWorkloadForecaster`.

    Returns:
        A :class:`TuningPolicyProposal` ready for protobuf serialization.
    """
    proposed_r, proposed_m = select_optimal_rm(
        forecast_p95_len=forecast.forecast_p95_query_len,
        payload_budget_c=int(request.payload_budget_c),
        target_epsilon=float(request.target_epsilon),
    )

    current_ingest = max(float(request.ingestion_rate_ewma), 1.0)
    spike_ratio = forecast.forecast_peak_ingest_rate / current_ingest
    # Leave extra load-factor headroom and widen proactive margin k when
    # seasonal forecasting predicts an incoming ingestion surge.
    proposed_alpha = float(
        min(max(0.90 - 0.06 * max(spike_ratio - 1.0, 0.0), MIN_ALPHA), MAX_ALPHA)
    )
    proposed_k = float(
        min(
            max(1.25 + 0.50 * max(spike_ratio - 1.0, 0.0), MIN_PROACTIVE_K),
            MAX_PROACTIVE_K,
        )
    )

    safe_q = min(max(int(request.current_q), 6), 40)
    capacity = float(1 << safe_q)
    load_factor = min(max(float(request.load_factor), 0.0), 1.0)
    remaining_slots = max(proposed_alpha - load_factor, 0.0) * capacity
    peak_rate = max(forecast.forecast_peak_ingest_rate, 0.0)
    t_full = remaining_slots / peak_rate if peak_rate > 0.0 else math.inf
    bulk_speed = max(float(request.bulk_load_speed), 1.0)
    t_rebuild = float(request.reverse_map_entries) / bulk_speed
    trigger_rebuild = (proposed_m != int(request.current_m)) or (
        t_full <= t_rebuild * proposed_k
    )

    return TuningPolicyProposal(
        proposed_r=proposed_r,
        proposed_m=proposed_m,
        proposed_alpha=proposed_alpha,
        proposed_k=proposed_k,
        trigger_rebuild=trigger_rebuild,
        peak_rate=peak_rate,
        spike_ratio=spike_ratio,
    )


def build_oob_tuning_response(
    request: control_plane_pb2.TuningRequest,
    model_version: str,
) -> control_plane_pb2.TuningResponse:
    """Constructs a deliberately invalid ``TuningResponse`` for fault-injection tests."""
    return control_plane_pb2.TuningResponse(
        request_id=request.request_id,
        proposed_r=1,
        proposed_m=int(request.payload_budget_c) + 4,
        proposed_target_alpha=0.98,
        proposed_proactive_margin_k=4.5,
        confidence=0.95,
        trigger_proactive_rebuild=True,
        model_version=model_version,
        rationale="injected out-of-bounds proposal",
        forecast_peak_ingest_rate=-5.0,
    )


def build_tuning_response(
    request_id: str,
    proposal: TuningPolicyProposal,
    forecast: ForecastResult,
    confidence: float,
    model_version: str,
) -> control_plane_pb2.TuningResponse:
    """Serializes a :class:`TuningPolicyProposal` into a protobuf ``TuningResponse``."""
    return control_plane_pb2.TuningResponse(
        request_id=request_id,
        proposed_r=proposal.proposed_r,
        proposed_m=proposal.proposed_m,
        proposed_target_alpha=proposal.proposed_alpha,
        proposed_proactive_margin_k=proposal.proposed_k,
        confidence=confidence,
        trigger_proactive_rebuild=proposal.trigger_rebuild,
        model_version=model_version,
        rationale=(
            f"seasonal forecast peak_ingest={proposal.peak_rate:.1f}/s "
            f"p95_len={forecast.forecast_p95_query_len:.1f} "
            f"spike_ratio={proposal.spike_ratio:.2f}"
        ),
        forecast_peak_ingest_rate=proposal.peak_rate,
    )


def build_oob_memory_response(
    request: control_plane_pb2.MemoryAllocationRequest,
    model_version: str,
) -> control_plane_pb2.MemoryAllocationResponse:
    """Constructs an over-budget ``MemoryAllocationResponse`` for fault-injection tests."""
    over = int(request.total_memory_budget_bytes) + 1024 * 1024
    allocs = [
        control_plane_pb2.ShardAllocation(
            shard_id=s.shard_id,
            allocated_memory_bytes=over,
            recommended_q=55,
            recommended_r=2,
            recommended_m=28,
        )
        for s in request.shards
    ]
    return control_plane_pb2.MemoryAllocationResponse(
        request_id=request.request_id,
        allocations=allocs,
        confidence=0.95,
        model_version=model_version,
    )


def telemetry_to_shard_contexts(
    request: control_plane_pb2.MemoryAllocationRequest,
) -> list[ShardContext]:
    """Converts incoming protobuf ``ShardTelemetry`` items into :class:`ShardContext` DTOs."""
    return [
        ShardContext(
            shard_id=int(s.shard_id),
            key_count=int(s.key_count),
            current_memory_bytes=int(s.current_memory_bytes),
            query_rate=float(s.query_rate),
            observed_fpr=float(s.observed_fpr),
            p95_query_len=float(s.p95_query_len),
            load_factor=float(s.load_factor),
            current_q=int(s.current_q),
            current_r=int(s.current_r),
            current_m=int(s.current_m),
        )
        for s in request.shards
    ]


def infer_payload_budget_c(shard_contexts: list[ShardContext]) -> int:
    """Infers the shared payload bit budget ``C = r + m`` from active shard telemetry."""
    if not shard_contexts:
        return 20
    return max(max(s.current_r + s.current_m for s in shard_contexts), 16)


def build_memory_response(
    request_id: str,
    plan_items: list[ShardAllocationPlanItem],
    confidence: float,
    model_version: str,
) -> control_plane_pb2.MemoryAllocationResponse:
    """Serializes bandit allocation items into a protobuf ``MemoryAllocationResponse``."""
    return control_plane_pb2.MemoryAllocationResponse(
        request_id=request_id,
        allocations=[
            control_plane_pb2.ShardAllocation(
                shard_id=item.shard_id,
                allocated_memory_bytes=item.allocated_memory_bytes,
                recommended_q=item.recommended_q,
                recommended_r=item.recommended_r,
                recommended_m=item.recommended_m,
            )
            for item in plan_items
        ],
        confidence=confidence,
        model_version=model_version,
    )
