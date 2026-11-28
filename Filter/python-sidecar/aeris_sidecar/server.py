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
import signal
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc

from .bandit import OfflineContextualBandit
from .cache import DEFAULT_CACHE_CAPACITY, ThreadSafeLruCache
from .forecaster import SeasonalWorkloadForecaster
from .proto import control_plane_pb2, control_plane_pb2_grpc
from .rpc_mapping import (
    DEFAULT_CPU_PROBE_NS,
    DEFAULT_DISK_IO_PENALTY_NS,
    MAX_ALPHA,
    MAX_PROACTIVE_K,
    MIN_ALPHA,
    MIN_PROACTIVE_K,
    build_memory_response,
    build_oob_memory_response,
    build_oob_tuning_response,
    build_tuning_response,
    compute_tuning_policy,
    infer_payload_budget_c,
    select_optimal_rm,
    telemetry_to_shard_contexts,
)
from .trainer import DEFAULT_ARTIFACT_VERSION, TrainedBundle, train_models

logger = logging.getLogger("aeris_sidecar.server")

__all__ = [
    "ControlPlaneAdvisorServicer",
    "DEFAULT_CACHE_CAPACITY",
    "DEFAULT_CPU_PROBE_NS",
    "DEFAULT_DISK_IO_PENALTY_NS",
    "MAX_ALPHA",
    "MAX_PROACTIVE_K",
    "MIN_ALPHA",
    "MIN_PROACTIVE_K",
    "ThreadSafeLruCache",
    "create_server",
    "main",
    "select_optimal_rm",
]


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
        """Initializes the servicer with fitted advisory models and idempotency caches.

        Args:
            forecaster: Fitted seasonal workload forecaster.
            bandit: Fitted offline contextual bandit allocator.
            model_version: Version identifier returned in responses.
            cache_capacity: Capacity of per-RPC LRU idempotency caches.
            fault_mode: Optional fault-injection mode for testing.
            slow_delay_ms: Artificial sleep duration when ``fault_mode == "slow"``.
        """
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
        self.tuning_calls: int = 0
        self.tuning_cache_hits: int = 0
        self.memory_calls: int = 0
        self.memory_cache_hits: int = 0

    def _apply_fault_preconditions(
        self, context: grpc.ServicerContext, error_message: str
    ) -> None:
        """Injects configured ``internal_error`` abort or ``slow`` delay."""
        if self._fault_mode == "internal_error":
            context.abort(grpc.StatusCode.INTERNAL, error_message)
        if self._fault_mode == "slow" and self._slow_delay_s > 0.0:
            time.sleep(self._slow_delay_s)

    def ProposeTuning(
        self,
        request: control_plane_pb2.TuningRequest,
        context: grpc.ServicerContext,
    ) -> control_plane_pb2.TuningResponse:
        """Forecasts workload demand and proposes optimal ``(r, m, alpha, k)``."""
        with self._lock:
            self.tuning_calls += 1

        self._apply_fault_preconditions(context, "injected internal sidecar fault")

        req_id = request.request_id
        if req_id and self._fault_mode == "normal":
            cached = self._tuning_cache.get(req_id)
            if cached is not None:
                with self._lock:
                    self.tuning_cache_hits += 1
                logger.debug("ProposeTuning cache hit request_id=%s", req_id)
                return cached

        if self._fault_mode == "out_of_bounds":
            return build_oob_tuning_response(request, self._model_version)

        forecast = self._forecaster.predict(
            ingest_series=list(request.recent_ingest_rate_series),
            query_len_series=list(request.recent_query_len_series),
            horizon=6,
            fallback_ingest=max(float(request.ingestion_rate_ewma), 0.0),
            fallback_query_len=max(float(request.p95_query_len), 1.0),
        )
        proposal = compute_tuning_policy(request, forecast)
        confidence = (
            0.25 if self._fault_mode == "low_confidence" else forecast.confidence
        )
        response = build_tuning_response(
            req_id, proposal, forecast, confidence, self._model_version
        )

        if req_id and self._fault_mode == "normal":
            self._tuning_cache.put(req_id, response)

        logger.info(
            "ProposeTuning shard_id=%d epoch=%d request_id=%s r=%d m=%d alpha=%.3f k=%.2f conf=%.3f",
            request.shard_id,
            request.epoch,
            req_id,
            proposal.proposed_r,
            proposal.proposed_m,
            proposal.proposed_alpha,
            proposal.proposed_k,
            confidence,
        )
        return response

    def AllocateMemory(
        self,
        request: control_plane_pb2.MemoryAllocationRequest,
        context: grpc.ServicerContext,
    ) -> control_plane_pb2.MemoryAllocationResponse:
        """Allocates global memory across shards using the offline contextual bandit."""
        with self._lock:
            self.memory_calls += 1

        self._apply_fault_preconditions(context, "injected internal sidecar fault")

        req_id = request.request_id
        if req_id and self._fault_mode == "normal":
            cached = self._memory_cache.get(req_id)
            if cached is not None:
                with self._lock:
                    self.memory_cache_hits += 1
                logger.debug("AllocateMemory cache hit request_id=%s", req_id)
                return cached

        if self._fault_mode == "out_of_bounds":
            return build_oob_memory_response(request, self._model_version)

        shard_contexts = telemetry_to_shard_contexts(request)
        inferred_c = infer_payload_budget_c(shard_contexts)
        plan_items, confidence = self._bandit.allocate(
            shards=shard_contexts,
            total_memory_budget_bytes=int(request.total_memory_budget_bytes),
            target_epsilon=1.0 / 256.0,
            payload_budget_c=inferred_c,
        )
        if self._fault_mode == "low_confidence":
            confidence = 0.25

        response = build_memory_response(
            req_id, plan_items, confidence, self._model_version
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
        """Reports liveness, loaded model version, and server uptime."""
        del request
        self._apply_fault_preconditions(context, "injected internal health fault")
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
        Tuple ``(grpc_server, servicer, bound_address, bound_port)``.

    Raises:
        RuntimeError: If binding a TCP socket fails (``bound_port == 0``).
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for ``aeris_sidecar.server``."""
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
    return parser


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """Registers SIGINT/SIGTERM handlers that trigger graceful shutdown."""

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("received signal %d; initiating graceful sidecar shutdown", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for running the out-of-process gRPC advisory sidecar.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (`0` on clean shutdown).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)

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
    _install_signal_handlers(stop_event)

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
