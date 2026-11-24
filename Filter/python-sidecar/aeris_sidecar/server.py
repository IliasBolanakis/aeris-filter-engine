"""Out-of-process gRPC Control Plane Advisor server for Aeris.

Serves ``ProposeTuning``, ``AllocateMemory``, and ``HealthCheck`` over:
- ``unix://<path>`` (Unix Domain Sockets on Unix platforms), OR
- ``127.0.0.1:<port>`` (loopback TCP on Windows and all OSes; port ``0`` binds
  an ephemeral port and optionally writes the bound address to ``--port-file``).

Features a thread-safe bounded LRU ``request_id`` idempotency cache so retried
RPCs return identical cached responses without duplicate computation.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import threading
import time
from collections import OrderedDict
from concurrent import futures
from pathlib import Path
from typing import Generic, TypeVar

import grpc

from .bandit import (
    DEFAULT_M_MAX,
    OfflineContextualBandit,
    ShardContext,
    min_remainder_bits,
)
from .forecaster import SeasonalWorkloadForecaster
from .proto import control_plane_pb2, control_plane_pb2_grpc
from .trainer import DEFAULT_ARTIFACT_VERSION, TrainedBundle, train_models

logger = logging.getLogger("aeris_sidecar.server")

V = TypeVar("V")

DEFAULT_CACHE_CAPACITY = 4096
DEFAULT_CPU_PROBE_NS = 35.0
DEFAULT_DISK_IO_PENALTY_NS = 50_000.0
MIN_ALPHA = 0.50
MAX_ALPHA = 0.90
MIN_PROACTIVE_K = 1.0
MAX_PROACTIVE_K = 3.0


class ThreadSafeLruCache(Generic[V]):
    """Thread-safe bounded LRU cache keyed by RPC ``request_id``."""

    def __init__(self, capacity: int = DEFAULT_CACHE_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"cache capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._store: OrderedDict[str, V] = OrderedDict()

    def get(self, key: str) -> V | None:
        """Retrieves cached value and promotes it as most recently used."""
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def put(self, key: str, value: V) -> None:
        """Inserts or updates ``key``, evicting the least recently used item if full."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def select_optimal_rm(
    forecast_p95_len: float,
    payload_budget_c: int,
    target_epsilon: float,
    m_max: int = DEFAULT_M_MAX,
) -> tuple[int, int]:
    """Selects optimal discrete ``(r, m)`` minimizing expected probe + FPR cost
    while strictly respecting ``r >= r_min(epsilon)``, ``m <= m_max``, and
    ``r + m == payload_budget_c``.
    """
    eps = target_epsilon if (math.isfinite(target_epsilon) and 0.0 < target_epsilon < 0.5) else (1.0 / 256.0)
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


class ControlPlaneAdvisorServicer(
    control_plane_pb2_grpc.ControlPlaneAdvisorServicer
):
    """gRPC implementation of ``aeris.control.v1.ControlPlaneAdvisor``."""

    def __init__(
        self,
        forecaster: SeasonalWorkloadForecaster,
        bandit: OfflineContextualBandit,
        model_version: str = DEFAULT_ARTIFACT_VERSION,
        cache_capacity: int = DEFAULT_CACHE_CAPACITY,
        fault_mode: str = "normal",
        slow_delay_ms: int = 0,
    ) -> None:
        self._forecaster = forecaster
        self._bandit = bandit
        self._model_version = model_version
        self._fault_mode = fault_mode
        self._slow_delay_s = max(float(slow_delay_ms) / 1000.0, 0.0)
        self._started_monotonic = time.monotonic()

        self._tuning_cache: ThreadSafeLruCache[control_plane_pb2.TuningResponse] = (
            ThreadSafeLruCache(capacity=cache_capacity)
        )
        self._memory_cache: ThreadSafeLruCache[
            control_plane_pb2.MemoryAllocationResponse
        ] = ThreadSafeLruCache(capacity=cache_capacity)
        self._lock = threading.Lock()
        self.tuning_calls = 0
        self.tuning_cache_hits = 0
        self.memory_calls = 0
        self.memory_cache_hits = 0

    def ProposeTuning(
        self,
        request: control_plane_pb2.TuningRequest,
        context: grpc.ServicerContext,
    ) -> control_plane_pb2.TuningResponse:
        with self._lock:
            self.tuning_calls += 1

        if self._fault_mode == "internal_error":
            context.abort(grpc.StatusCode.INTERNAL, "injected internal sidecar fault")
        if self._fault_mode == "slow" and self._slow_delay_s > 0.0:
            time.sleep(self._slow_delay_s)

        req_id = request.request_id
        if req_id and self._fault_mode == "normal":
            cached = self._tuning_cache.get(req_id)
            if cached is not None:
                with self._lock:
                    self.tuning_cache_hits += 1
                logger.debug("ProposeTuning cache hit request_id=%s", req_id)
                return cached

        if self._fault_mode == "out_of_bounds":
            return control_plane_pb2.TuningResponse(
                request_id=req_id,
                proposed_r=1,
                proposed_m=int(request.payload_budget_c) + 4,
                proposed_target_alpha=0.98,
                proposed_proactive_margin_k=4.5,
                confidence=0.95,
                trigger_proactive_rebuild=True,
                model_version=self._model_version,
                rationale="injected out-of-bounds proposal",
                forecast_peak_ingest_rate=-5.0,
            )

        ingest_series = list(request.recent_ingest_rate_series)
        query_series = list(request.recent_query_len_series)

        forecast = self._forecaster.predict(
            ingest_series=ingest_series,
            query_len_series=query_series,
            horizon=6,
            fallback_ingest=max(float(request.ingestion_rate_ewma), 0.0),
            fallback_query_len=max(float(request.p95_query_len), 1.0),
        )

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
            min(max(1.25 + 0.50 * max(spike_ratio - 1.0, 0.0), MIN_PROACTIVE_K), MAX_PROACTIVE_K)
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

        confidence = 0.25 if self._fault_mode == "low_confidence" else forecast.confidence
        response = control_plane_pb2.TuningResponse(
            request_id=req_id,
            proposed_r=proposed_r,
            proposed_m=proposed_m,
            proposed_target_alpha=proposed_alpha,
            proposed_proactive_margin_k=proposed_k,
            confidence=confidence,
            trigger_proactive_rebuild=trigger_rebuild,
            model_version=self._model_version,
            rationale=(
                f"seasonal forecast peak_ingest={peak_rate:.1f}/s "
                f"p95_len={forecast.forecast_p95_query_len:.1f} "
                f"spike_ratio={spike_ratio:.2f}"
            ),
            forecast_peak_ingest_rate=peak_rate,
        )

        if req_id and self._fault_mode == "normal":
            self._tuning_cache.put(req_id, response)

        logger.info(
            "ProposeTuning shard_id=%d epoch=%d request_id=%s r=%d m=%d alpha=%.3f k=%.2f conf=%.3f",
            request.shard_id,
            request.epoch,
            req_id,
            proposed_r,
            proposed_m,
            proposed_alpha,
            proposed_k,
            confidence,
        )
        return response

    def AllocateMemory(
        self,
        request: control_plane_pb2.MemoryAllocationRequest,
        context: grpc.ServicerContext,
    ) -> control_plane_pb2.MemoryAllocationResponse:
        with self._lock:
            self.memory_calls += 1

        if self._fault_mode == "internal_error":
            context.abort(grpc.StatusCode.INTERNAL, "injected internal sidecar fault")
        if self._fault_mode == "slow" and self._slow_delay_s > 0.0:
            time.sleep(self._slow_delay_s)

        req_id = request.request_id
        if req_id and self._fault_mode == "normal":
            cached = self._memory_cache.get(req_id)
            if cached is not None:
                with self._lock:
                    self.memory_cache_hits += 1
                logger.debug("AllocateMemory cache hit request_id=%s", req_id)
                return cached

        if self._fault_mode == "out_of_bounds":
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
                request_id=req_id,
                allocations=allocs,
                confidence=0.95,
                model_version=self._model_version,
            )

        shard_contexts = [
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

        # Infer shared payload budget C from incoming shards if available.
        if shard_contexts:
            inferred_c = max(max(s.current_r + s.current_m for s in shard_contexts), 16)
        else:
            inferred_c = 20

        plan_items, confidence = self._bandit.allocate(
            shards=shard_contexts,
            total_memory_budget_bytes=int(request.total_memory_budget_bytes),
            target_epsilon=1.0 / 256.0,
            payload_budget_c=inferred_c,
        )
        if self._fault_mode == "low_confidence":
            confidence = 0.25

        response = control_plane_pb2.MemoryAllocationResponse(
            request_id=req_id,
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
            model_version=self._model_version,
        )

        if req_id and self._fault_mode == "normal":
            self._memory_cache.put(req_id, response)

        logger.info(
            "AllocateMemory epoch=%d request_id=%s shards=%d budget=%d conf=%.3f",
            request.epoch,
            req_id,
            len(plan_items),
            request.total_memory_budget_bytes,
            confidence,
        )
        return response

    def HealthCheck(
        self,
        request: control_plane_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> control_plane_pb2.HealthCheckResponse:
        del request
        if self._fault_mode == "internal_error":
            context.abort(grpc.StatusCode.INTERNAL, "injected internal health fault")
        if self._fault_mode == "slow" and self._slow_delay_s > 0.0:
            time.sleep(self._slow_delay_s)
        uptime_ms = int((time.monotonic() - self._started_monotonic) * 1000.0)
        return control_plane_pb2.HealthCheckResponse(
            healthy=True,
            model_version=self._model_version,
            uptime_ms=max(uptime_ms, 0),
        )


def create_server(
    bind: str = "127.0.0.1:0",
    bundle: TrainedBundle | None = None,
    max_workers: int = 4,
    fault_mode: str = "normal",
    slow_delay_ms: int = 0,
) -> tuple[grpc.Server, ControlPlaneAdvisorServicer, str, int]:
    """Creates and binds a gRPC ``ControlPlaneAdvisor`` server without blocking.

    Args:
        bind: Listen address (``127.0.0.1:<port>`` or ``unix:///path/to/sock``).
        bundle: Optional pre-trained :class:`TrainedBundle`; trains a deterministic
            default bundle in-memory when omitted.
        max_workers: Thread pool worker count for concurrent RPCs.
        fault_mode: Fault injection mode (``normal``, ``out_of_bounds``,
            ``low_confidence``, ``slow``, ``internal_error``).
        slow_delay_ms: Artificial delay in milliseconds when ``fault_mode == "slow"``.

    Returns:
        ``(grpc_server, servicer, bound_address, bound_port)``
    """
    active_bundle = bundle if bundle is not None else train_models(seed=42)
    servicer = ControlPlaneAdvisorServicer(
        forecaster=active_bundle.forecaster,
        bandit=active_bundle.bandit,
        model_version=active_bundle.model_version,
        fault_mode=fault_mode,
        slow_delay_ms=slow_delay_ms,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    control_plane_pb2_grpc.add_ControlPlaneAdvisorServicer_to_server(servicer, server)

    bound_port = server.add_insecure_port(bind)
    if bind.startswith("unix:"):
        bound_address = bind
    else:
        if bound_port == 0:
            raise RuntimeError(f"failed to bind gRPC sidecar on {bind}")
        host = bind.rsplit(":", 1)[0]
        bound_address = f"{host}:{bound_port}"

    return server, servicer, bound_address, bound_port


def _write_port_file(port_file: Path, bound_address: str, bound_port: int) -> None:
    """Atomically writes the bound port/address for subprocess test discovery."""
    port_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = port_file.with_suffix(port_file.suffix + ".tmp")
    content = str(bound_port) if bound_port > 0 else bound_address
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(port_file)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for running the out-of-process gRPC advisory sidecar."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Aeris Out-of-Process ML/RL Control Plane gRPC Sidecar"
    )
    parser.add_argument(
        "--bind",
        type=str,
        default="127.0.0.1:0",
        help="Bind address: '127.0.0.1:<port>' or 'unix:///path/to/socket' (Unix).",
    )
    parser.add_argument(
        "--port-file",
        type=Path,
        default=None,
        help="Optional file where the actual bound port (or unix URI) is written once listening.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to a trained JSON model artifact from aeris_sidecar.trainer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed if training an in-memory model on startup.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="gRPC server worker thread pool size.",
    )
    parser.add_argument(
        "--fault-mode",
        type=str,
        default="normal",
        choices=["normal", "out_of_bounds", "low_confidence", "slow", "internal_error"],
        help="Optional fault-injection mode for testing.",
    )
    parser.add_argument(
        "--slow-ms",
        type=int,
        default=0,
        help="Artificial RPC latency in milliseconds when --fault-mode=slow.",
    )
    args = parser.parse_args(argv)

    bundle = (
        TrainedBundle.load(args.model_path)
        if args.model_path is not None
        else train_models(seed=args.seed)
    )
    server, _, bound_address, bound_port = create_server(
        bind=args.bind,
        bundle=bundle,
        max_workers=args.max_workers,
        fault_mode=args.fault_mode,
        slow_delay_ms=args.slow_ms,
    )

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("received signal %d; initiating graceful sidecar shutdown", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    server.start()
    logger.info(
        "aeris python sidecar listening on %s model_version=%s fault_mode=%s",
        bound_address,
        bundle.model_version,
        args.fault_mode,
    )
    if args.port_file is not None:
        _write_port_file(args.port_file, bound_address, bound_port)

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)
    finally:
        logger.info("stopping aeris python sidecar server")
        server.stop(grace=2.0).wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
