"""Integration and concurrency tests for the Python gRPC ControlPlaneAdvisor server."""

from __future__ import annotations

import concurrent.futures
import math
import sys
from pathlib import Path

import grpc
import numpy as np
import pytest

from aeris_sidecar.bandit import MAX_QUOTIENT_BITS, MIN_QUOTIENT_BITS, min_remainder_bits
from aeris_sidecar.proto import control_plane_pb2, control_plane_pb2_grpc
from aeris_sidecar.server import ThreadSafeLruCache, create_server
from aeris_sidecar.trainer import generate_seasonal_workload, train_models


def _build_realistic_tuning_request(request_id: str) -> control_plane_pb2.TuningRequest:
    rng = np.random.default_rng(42)
    ingest, qlen = generate_seasonal_workload(rng, num_steps=48, period=24.0, noise_scale=0.02)
    return control_plane_pb2.TuningRequest(
        request_id=request_id,
        shard_id=7,
        epoch=3,
        timestamp_ms=1_700_000_000_000,
        current_q=16,
        current_r=15,
        current_m=5,
        payload_budget_c=20,
        target_epsilon=1.0 / 256.0,
        load_factor=0.72,
        ingestion_rate_ewma=float(ingest[-1]),
        query_rate=15_000.0,
        observed_fpr=0.003,
        p95_query_len=float(qlen[-1]),
        p99_query_len=float(qlen[-1] * 1.5),
        reverse_map_entries=45_000,
        bulk_load_speed=500_000.0,
        recent_query_len_series=[float(v) for v in qlen],
        recent_ingest_rate_series=[float(v) for v in ingest],
    )


def _build_memory_request(request_id: str, total_budget: int) -> control_plane_pb2.MemoryAllocationRequest:
    return control_plane_pb2.MemoryAllocationRequest(
        request_id=request_id,
        epoch=5,
        total_memory_budget_bytes=total_budget,
        shards=[
            control_plane_pb2.ShardTelemetry(
                shard_id=1,
                key_count=150_000,
                current_memory_bytes=2_000_000,
                query_rate=45_000.0,
                observed_fpr=0.06,
                p95_query_len=256.0,
                load_factor=0.88,
                current_q=16,
                current_r=14,
                current_m=6,
            ),
            control_plane_pb2.ShardTelemetry(
                shard_id=2,
                key_count=150_000,
                current_memory_bytes=2_000_000,
                query_rate=5.0,
                observed_fpr=0.0005,
                p95_query_len=4.0,
                load_factor=0.22,
                current_q=16,
                current_r=14,
                current_m=6,
            ),
        ],
    )


def test_grpc_server_end_to_end_and_idempotency_cache() -> None:
    bundle = train_models(seed=42, model_version="py-e2e-v1")
    server, servicer, bound_address, bound_port = create_server(
        bind="127.0.0.1:0", bundle=bundle, max_workers=4
    )
    assert bound_port > 0
    server.start()
    try:
        with grpc.insecure_channel(bound_address) as channel:
            stub = control_plane_pb2_grpc.ControlPlaneAdvisorStub(channel)

            # 1. HealthCheck
            health = stub.HealthCheck(control_plane_pb2.HealthCheckRequest(), timeout=5.0)
            assert health.healthy is True
            assert health.model_version == "py-e2e-v1"
            assert health.uptime_ms >= 0

            # 2. ProposeTuning + idempotency cache hit on identical request_id
            req = _build_realistic_tuning_request("req-tune-001")
            resp1 = stub.ProposeTuning(req, timeout=5.0)
            resp2 = stub.ProposeTuning(req, timeout=5.0)

            assert resp1 == resp2
            assert servicer.tuning_calls == 2
            assert servicer.tuning_cache_hits == 1

            r_min = min_remainder_bits(1.0 / 256.0)
            assert resp1.request_id == "req-tune-001"
            assert resp1.proposed_r >= r_min
            assert resp1.proposed_r + resp1.proposed_m <= 20
            assert 0.50 <= resp1.proposed_target_alpha <= 0.90
            assert 1.0 <= resp1.proposed_proactive_margin_k <= 3.0
            assert 0.80 <= resp1.confidence <= 1.0
            assert math.isfinite(resp1.forecast_peak_ingest_rate)
            assert resp1.forecast_peak_ingest_rate >= 0.0

            # 3. AllocateMemory + idempotency cache hit
            budget = 16 * 1024 * 1024
            mem_req = _build_memory_request("req-mem-001", budget)
            m1 = stub.AllocateMemory(mem_req, timeout=5.0)
            m2 = stub.AllocateMemory(mem_req, timeout=5.0)
            assert m1 == m2
            assert servicer.memory_cache_hits == 1
            assert len(m1.allocations) == 2
            total_alloc = sum(a.allocated_memory_bytes for a in m1.allocations)
            assert total_alloc <= budget
            assert 0.80 <= m1.confidence <= 1.0

            by_id = {a.shard_id: a for a in m1.allocations}
            assert by_id[1].allocated_memory_bytes > by_id[2].allocated_memory_bytes
            for alloc in m1.allocations:
                assert MIN_QUOTIENT_BITS <= alloc.recommended_q <= MAX_QUOTIENT_BITS
                assert alloc.recommended_r >= r_min
                assert alloc.recommended_r + alloc.recommended_m <= 20
    finally:
        server.stop(grace=1.0).wait()


def test_grpc_server_concurrent_requests() -> None:
    bundle = train_models(seed=42)
    server, _, bound_address, _ = create_server(bind="127.0.0.1:0", bundle=bundle, max_workers=8)
    server.start()
    try:
        with grpc.insecure_channel(bound_address) as channel:
            stub = control_plane_pb2_grpc.ControlPlaneAdvisorStub(channel)

            def _worker(idx: int) -> tuple[str, float]:
                req = _build_realistic_tuning_request(f"concurrent-{idx}")
                resp = stub.ProposeTuning(req, timeout=5.0)
                return resp.request_id, resp.confidence

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(_worker, range(24)))

            assert len(results) == 24
            for idx, (rid, conf) in enumerate(results):
                assert rid == f"concurrent-{idx}"
                assert conf >= 0.80
    finally:
        server.stop(grace=1.0).wait()


def test_grpc_server_fault_injection_modes() -> None:
    bundle = train_models(seed=42)

    # internal_error mode
    srv_err, _, addr_err, _ = create_server(
        bind="127.0.0.1:0", bundle=bundle, fault_mode="internal_error"
    )
    srv_err.start()
    try:
        with grpc.insecure_channel(addr_err) as ch:
            stub = control_plane_pb2_grpc.ControlPlaneAdvisorStub(ch)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.ProposeTuning(_build_realistic_tuning_request("fault-1"), timeout=5.0)
            assert exc_info.value.code() == grpc.StatusCode.INTERNAL
    finally:
        srv_err.stop(grace=1.0).wait()

    # out_of_bounds mode
    srv_oob, _, addr_oob, _ = create_server(
        bind="127.0.0.1:0", bundle=bundle, fault_mode="out_of_bounds"
    )
    srv_oob.start()
    try:
        with grpc.insecure_channel(addr_oob) as ch:
            stub = control_plane_pb2_grpc.ControlPlaneAdvisorStub(ch)
            resp = stub.ProposeTuning(_build_realistic_tuning_request("fault-oob"), timeout=5.0)
            assert resp.proposed_r < min_remainder_bits(1.0 / 256.0)
            assert resp.proposed_target_alpha > 0.90
    finally:
        srv_oob.stop(grace=1.0).wait()


def test_lru_cache_eviction_bounds() -> None:
    cache: ThreadSafeLruCache[int] = ThreadSafeLruCache(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") == 1  # promotes 'a'; 'b' is now LRU
    cache.put("d", 4)
    assert len(cache) == 3
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


@pytest.mark.skipif(sys.platform == "win32", reason="Unix Domain Sockets are Unix-only")
def test_grpc_server_over_uds_on_unix(tmp_path: Path) -> None:
    sock_path = tmp_path / "aeris_sidecar.sock"
    uri = f"unix://{sock_path}"
    bundle = train_models(seed=42)
    server, _, bound_address, _ = create_server(bind=uri, bundle=bundle)
    server.start()
    try:
        with grpc.insecure_channel(bound_address) as channel:
            stub = control_plane_pb2_grpc.ControlPlaneAdvisorStub(channel)
            health = stub.HealthCheck(control_plane_pb2.HealthCheckRequest(), timeout=5.0)
            assert health.healthy is True
    finally:
        server.stop(grace=1.0).wait()
