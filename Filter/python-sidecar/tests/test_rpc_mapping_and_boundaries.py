"""Unit tests for newly extracted sidecar modules, boundary inputs, and failure paths."""

from __future__ import annotations

import math
from pathlib import Path

import grpc
import numpy as np
import pytest

from aeris_sidecar.bandit import (
    DEFAULT_ACTIONS,
    OfflineContextualBandit,
    ShardContext,
    extract_raw_features,
)
from aeris_sidecar.cache import ThreadSafeLruCache
from aeris_sidecar.forecaster import SeasonalWorkloadForecaster
from aeris_sidecar.proto import control_plane_pb2
from aeris_sidecar.rpc_mapping import (
    infer_payload_budget_c,
    select_optimal_rm,
    telemetry_to_shard_contexts,
)
from aeris_sidecar.server import create_server, main as server_main
from aeris_sidecar.trainer import TrainedBundle, train_models


def test_lru_cache_invalid_capacity_and_update_promotion() -> None:
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        ThreadSafeLruCache(capacity=0)

    cache: ThreadSafeLruCache[str] = ThreadSafeLruCache(capacity=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k1", "v1_updated")  # promotes k1, k2 becomes LRU
    cache.put("k3", "v3")
    assert cache.get("k2") is None
    assert cache.get("k1") == "v1_updated"
    assert cache.get("k3") == "v3"


def test_select_optimal_rm_and_payload_budget_inference() -> None:
    # Non-finite / invalid inputs fall back safely while respecting r_min.
    r_pt, m_pt = select_optimal_rm(
        forecast_p95_len=float("nan"),
        payload_budget_c=20,
        target_epsilon=float("inf"),
    )
    assert r_pt + m_pt == 20
    assert r_pt >= 12  # r_min(1/256) = 12

    r_rng, m_rng = select_optimal_rm(
        forecast_p95_len=512.0,
        payload_budget_c=20,
        target_epsilon=1.0 / 256.0,
    )
    assert r_rng + m_rng == 20
    assert m_rng >= 1
    assert r_rng >= 12

    assert infer_payload_budget_c([]) == 20
    req = control_plane_pb2.MemoryAllocationRequest(
        request_id="inf-c",
        epoch=1,
        total_memory_budget_bytes=1024,
        shards=[
            control_plane_pb2.ShardTelemetry(
                shard_id=1,
                key_count=10,
                current_memory_bytes=1024,
                query_rate=1.0,
                observed_fpr=0.0,
                p95_query_len=1.0,
                load_factor=0.1,
                current_q=8,
                current_r=14,
                current_m=10,
            )
        ],
    )
    contexts = telemetry_to_shard_contexts(req)
    assert len(contexts) == 1
    assert infer_payload_budget_c(contexts) == 24


def test_bandit_fit_and_allocate_failure_paths() -> None:
    bandit = OfflineContextualBandit(actions=DEFAULT_ACTIONS)

    with pytest.raises(ValueError, match="raw_contexts"):
        bandit.fit_offline(np.zeros((10, 4)), np.zeros(10, dtype=np.int64), np.zeros(10))
    with pytest.raises(ValueError, match="sample count"):
        bandit.fit_offline(np.zeros((10, 5)), np.zeros(9, dtype=np.int64), np.zeros(10))
    with pytest.raises(ValueError, match="insufficient"):
        bandit.fit_offline(np.zeros((3, 5)), np.zeros(3, dtype=np.int64), np.zeros(3))
    with pytest.raises(ValueError, match="out of valid range"):
        bandit.fit_offline(
            np.zeros((8, 5)),
            np.array([0, 1, 2, 3, 4, 99, 0, 1], dtype=np.int64),
            np.zeros(8),
        )
    with pytest.raises(ValueError, match="propensities"):
        bandit.fit_offline(
            np.ones((8, 5)),
            np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64),
            np.ones(8),
            propensities=np.ones(4),
        )
    with pytest.raises(ValueError, match="total_memory_budget_bytes"):
        bandit.allocate([], total_memory_budget_bytes=-1)

    nan_shard = ShardContext(
        shard_id=99,
        key_count=100,
        current_memory_bytes=1024,
        query_rate=float("nan"),
        observed_fpr=float("inf"),
        p95_query_len=float("-inf"),
        load_factor=float("nan"),
        current_q=10,
        current_r=14,
        current_m=4,
    )
    feats = extract_raw_features(nan_shard)
    assert np.all(np.isfinite(feats))


def test_forecaster_from_dict_corrupted_shapes_and_serialization_roundtrip() -> None:
    model = SeasonalWorkloadForecaster(period=12.0, num_harmonics=2)
    payload = model.to_dict()
    payload["ingest_weights"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="corrupted weight shapes"):
        SeasonalWorkloadForecaster.from_dict(payload)


def test_server_low_confidence_fault_mode_and_model_artifact_roundtrip(tmp_path: Path) -> None:
    bundle = train_models(seed=7, model_version="test-low-conf-v1")
    artifact = tmp_path / "model.json"
    bundle.save(artifact)
    loaded = TrainedBundle.load(artifact)
    assert loaded.model_version == "test-low-conf-v1"

    srv, _, addr, _ = create_server(
        bind="127.0.0.1:0", bundle=loaded, fault_mode="low_confidence"
    )
    srv.start()
    try:
        with grpc.insecure_channel(addr) as ch:
            from aeris_sidecar.proto import control_plane_pb2_grpc

            stub = control_plane_pb2_grpc.ControlPlaneAdvisorStub(ch)
            mem_resp = stub.AllocateMemory(
                control_plane_pb2.MemoryAllocationRequest(
                    request_id="low-conf-mem",
                    epoch=1,
                    total_memory_budget_bytes=4096,
                    shards=[],
                ),
                timeout=5.0,
            )
            assert math.isclose(mem_resp.confidence, 0.25)
    finally:
        srv.stop(grace=1.0).wait()


def test_server_cli_parser_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        server_main(["--help"])
    assert exc_info.value.code == 0
