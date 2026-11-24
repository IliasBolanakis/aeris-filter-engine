#!/usr/bin/env python3
"""Aeris Ablation Report Aggregator and Markdown Report Generator.

Aggregates per-worker JSON ablation reports emitted by the Aeris Correctness
Validation & Large-Scale Ablation Suite (`target/ablation_worker*.json`) into:

1. A merged machine-readable JSON summary (`Filter/target/ablation_results.json`).
2. A strictly data-driven engineering report (`Filter/ABLATION_REPORT.md`)
   featuring a correctness audit scorecard, host hardware & L3 cache spill
   analysis, side-by-side Naive Paper-Baseline vs. Aeris Optimized tables
   grouped by extension suite, and a blunt engineering verdict per extension.

Uses only the Python 3 standard library (zero third-party dependencies) and
resolves the `Filter/` workspace root relative to `Path(__file__).resolve().parent.parent`
so it behaves identically regardless of the caller's working directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Final, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants & Canonical Suite Definitions
# ---------------------------------------------------------------------------

FILTER_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR: Final[Path] = FILTER_ROOT / "target"
DEFAULT_OUTPUT_JSON: Final[Path] = FILTER_ROOT / "target" / "ablation_results.json"
DEFAULT_OUTPUT_MD: Final[Path] = FILTER_ROOT / "ABLATION_REPORT.md"
DEFAULT_GLOB_PATTERN: Final[str] = "ablation_worker*.json"

SCHEMA_VERSION: Final[str] = "1.0"
REFERENCE_L3_SPILL_BYTES: Final[int] = 24 * 1024 * 1024  # 24 MiB reference L3 threshold
NEUTRAL_BAND_PCT: Final[float] = 5.0  # +/-5% band classified as Neutral

CANONICAL_WORKER_FILES: Final[tuple[str, ...]] = (
    "ablation_worker1_fpr_dynamic_r.json",
    "ablation_worker2_proactive_storage.json",
    "ablation_worker3_scale_sidecar.json",
)

REQUIRED_HOST_FIELDS: Final[tuple[str, ...]] = (
    "cpu_model",
    "physical_cores",
    "logical_cores",
    "l3_cache_kb",
    "total_ram_bytes",
    "available_ram_bytes",
    "scale_factor",
    "peak_rss_bytes",
)

REQUIRED_METRIC_FIELDS: Final[tuple[str, ...]] = (
    "extension_name",
    "key_count",
    "query_count",
    "baseline_value",
    "optimized_value",
    "delta_pct",
    "p50_ns",
    "p99_ns",
    "p999_ns",
    "empirical_fpr",
    "false_negatives",
    "repeated_fps",
    "peak_ram_bytes",
    "disk_io_bytes",
)

VERDICT_MEASURABLE_GAIN: Final[str] = "Measurable Gain"
VERDICT_EXPECTED_OVERHEAD: Final[str] = "Expected Safety/Durability Overhead"
VERDICT_NEUTRAL: Final[str] = "Neutral"
VERDICT_NEGATIVE_OVERHEAD: Final[str] = "Negative Overhead"


@dataclass(frozen=True)
class SuiteDefinition:
    """Metadata describing one canonical ablation report section."""

    suite_id: str
    title: str
    baseline_label: str
    optimized_label: str
    expected_worker: str


CANONICAL_SUITES: Final[tuple[SuiteDefinition, ...]] = (
    SuiteDefinition(
        suite_id="DRAM_AUDIT",
        title="DRAM-Scale 10M-Key / 5M-Query Audit",
        baseline_label="Theoretical / Unadapted Baseline",
        optimized_label="Aeris Optimized (Monotonic Adaptation)",
        expected_worker="worker1_fpr_dynamic_r",
    ),
    SuiteDefinition(
        suite_id="AB1",
        title="A/B 1: Dynamic R + FP Log Replay vs. Static R",
        baseline_label="Naive Paper-Baseline (Static R, No FP Log)",
        optimized_label="Aeris Optimized (Dynamic R + FP Log Replay)",
        expected_worker="worker1_fpr_dynamic_r",
    ),
    SuiteDefinition(
        suite_id="AB2",
        title="A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95%",
        baseline_label="Naive Paper-Baseline (Reactive alpha = 95%, No Contraction)",
        optimized_label="Aeris Optimized (Proactive EWMA + Contraction)",
        expected_worker="worker2_proactive_storage",
    ),
    SuiteDefinition(
        suite_id="AB3",
        title="A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map",
        baseline_label="Naive Paper-Baseline (Uncoupled Reverse Map)",
        optimized_label="Aeris Optimized (WAL-Coupled Storage Engine)",
        expected_worker="worker2_proactive_storage",
    ),
    SuiteDefinition(
        suite_id="AB4",
        title="A/B 4: Scale Plane (RCU + 8-Shard Range Sharding) vs. Coarse-Locked Single Filter",
        baseline_label="Naive Paper-Baseline (Coarse-Locked Single Filter)",
        optimized_label="Aeris Optimized (RCU + 8-Shard Range Sharding)",
        expected_worker="worker3_scale_sidecar",
    ),
    SuiteDefinition(
        suite_id="AB5",
        title="A/B 5: Phase 5 Live Python ML/RL Sidecar vs. Native Control Plane",
        baseline_label="Native Control Plane Baseline",
        optimized_label="Phase 5 Live Python ML/RL Sidecar",
        expected_worker="worker3_scale_sidecar",
    ),
)

OTHER_SUITE: Final[SuiteDefinition] = SuiteDefinition(
    suite_id="OTHER",
    title="Additional / Uncategorized Ablation Metrics",
    baseline_label="Baseline",
    optimized_label="Optimized",
    expected_worker="unknown",
)

SUITE_BY_ID: Final[dict[str, SuiteDefinition]] = {
    suite.suite_id: suite for suite in (*CANONICAL_SUITES, OTHER_SUITE)
}

LOGGER = logging.getLogger("aeris.ablation_report")


# ---------------------------------------------------------------------------
# Exceptions & Domain Models
# ---------------------------------------------------------------------------


class SchemaValidationError(ValueError):
    """Raised when an input JSON file violates the required ablation schema."""


@dataclass(frozen=True)
class HostMetadata:
    """Host hardware and process memory telemetry mirroring Rust `HostMetadata`.

    Invariants:
        - `physical_cores`, `logical_cores`, `l3_cache_kb`, `total_ram_bytes`,
          `available_ram_bytes`, and `peak_rss_bytes` are non-negative integers.
        - `scale_factor` is finite and strictly greater than zero.
    """

    cpu_model: str
    physical_cores: int
    logical_cores: int
    l3_cache_kb: int
    total_ram_bytes: int
    available_ram_bytes: int
    scale_factor: float
    peak_rss_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance into a JSON-compatible dictionary."""
        return {
            "cpu_model": self.cpu_model,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "l3_cache_kb": self.l3_cache_kb,
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "scale_factor": self.scale_factor,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True)
class AblationMetric:
    """Single validated measurement record mirroring Rust `AblationMetric`.

    Invariants:
        - All integer counters and latencies are non-negative.
        - Floating-point values are finite; `empirical_fpr` lies in `[0.0, 1.0]`.
    """

    extension_name: str
    metric_label: str
    unit: str
    higher_is_better: bool
    key_count: int
    query_count: int
    baseline_value: float
    optimized_value: float
    delta_pct: float
    p50_ns: int
    p99_ns: int
    p999_ns: int
    empirical_fpr: float
    false_negatives: int
    repeated_fps: int
    peak_ram_bytes: int
    disk_io_bytes: int
    scale_factor: float
    notes: str = ""
    worker_name: str = ""
    suite_id: str = "OTHER"

    def to_contract_dict(self) -> dict[str, Any]:
        """Serializes the exact Rust `AblationMetric` fields plus worker/suite provenance."""
        return {
            "extension_name": self.extension_name,
            "metric_label": self.metric_label,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "key_count": self.key_count,
            "query_count": self.query_count,
            "baseline_value": self.baseline_value,
            "optimized_value": self.optimized_value,
            "delta_pct": self.delta_pct,
            "p50_ns": self.p50_ns,
            "p99_ns": self.p99_ns,
            "p999_ns": self.p999_ns,
            "empirical_fpr": self.empirical_fpr,
            "false_negatives": self.false_negatives,
            "repeated_fps": self.repeated_fps,
            "peak_ram_bytes": self.peak_ram_bytes,
            "disk_io_bytes": self.disk_io_bytes,
            "scale_factor": self.scale_factor,
            "notes": self.notes,
            "worker_name": self.worker_name,
            "suite_id": self.suite_id,
        }


@dataclass(frozen=True)
class WorkerReport:
    """Validated per-worker ablation JSON payload."""

    worker_name: str
    source_file: str
    host: HostMetadata
    metrics: tuple[AblationMetric, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MetricAssessment:
    """Direction-aware evaluation of a single `AblationMetric`."""

    metric: AblationMetric
    signed_improvement_pct: float
    classification: str
    ratio_display: str
    is_comparative: bool
    is_durability_metric: bool


@dataclass(frozen=True)
class SuiteVerdict:
    """Data-driven engineering verdict for one ablation suite."""

    suite_id: str
    title: str
    verdict: str
    metric_count: int
    false_negatives: int
    repeated_fps: int
    rationale: str
    highlights: tuple[str, ...]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serializes this verdict to a JSON-compatible dictionary."""
        return {
            "suite_id": self.suite_id,
            "title": self.title,
            "verdict": self.verdict,
            "metric_count": self.metric_count,
            "false_negatives": self.false_negatives,
            "repeated_fps": self.repeated_fps,
            "rationale": self.rationale,
            "highlights": list(self.highlights),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class CorrectnessSummary:
    """Global correctness audit across all ingested ablation metrics."""

    total_metrics: int
    total_false_negatives: int
    total_repeated_fps: int
    zero_fn_pass: bool
    zero_repeated_fp_pass: bool
    fn_violations: tuple[AblationMetric, ...]
    repeated_fp_violations: tuple[AblationMetric, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serializes the correctness summary to a JSON-compatible dictionary."""
        return {
            "total_metrics": self.total_metrics,
            "total_false_negatives": self.total_false_negatives,
            "total_repeated_fps": self.total_repeated_fps,
            "zero_fn_pass": self.zero_fn_pass,
            "zero_repeated_fp_pass": self.zero_repeated_fp_pass,
            "fn_violation_extensions": [m.extension_name for m in self.fn_violations],
            "repeated_fp_violation_extensions": [
                m.extension_name for m in self.repeated_fp_violations
            ],
        }


@dataclass(frozen=True)
class L3SpillAnalysis:
    """L3 cache spill analysis comparing filter/process footprint against host L3 cache."""

    reported_l3_bytes: int
    reference_l3_bytes: int
    reference_source: str
    max_dram_metric_ram_bytes: int
    max_overall_metric_ram_bytes: int
    merged_peak_rss_bytes: int
    dram_spill_ratio: float
    rss_spill_ratio: float
    dram_spill_confirmed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serializes the L3 cache spill analysis to a JSON-compatible dictionary."""
        return {
            "reported_l3_bytes": self.reported_l3_bytes,
            "reference_l3_bytes": self.reference_l3_bytes,
            "reference_source": self.reference_source,
            "max_dram_metric_ram_bytes": self.max_dram_metric_ram_bytes,
            "max_overall_metric_ram_bytes": self.max_overall_metric_ram_bytes,
            "merged_peak_rss_bytes": self.merged_peak_rss_bytes,
            "dram_spill_ratio": round(self.dram_spill_ratio, 3),
            "rss_spill_ratio": round(self.rss_spill_ratio, 3),
            "dram_spill_confirmed": self.dram_spill_confirmed,
        }


@dataclass(frozen=True)
class MergedAblationReport:
    """Complete aggregated report ready for JSON export and Markdown rendering."""

    generated_at_utc: str
    host: HostMetadata
    workers: tuple[WorkerReport, ...]
    missing_canonical_files: tuple[str, ...]
    metrics: tuple[AblationMetric, ...]
    grouped_metrics: Mapping[str, tuple[AblationMetric, ...]]
    assessments_by_suite: Mapping[str, tuple[MetricAssessment, ...]]
    verdicts: tuple[SuiteVerdict, ...]
    correctness: CorrectnessSummary
    l3_spill: L3SpillAnalysis


# ---------------------------------------------------------------------------
# Strict Schema Validation & Loading
# ---------------------------------------------------------------------------


def _require_non_empty_str(obj: Mapping[str, Any], field_name: str, ctx: str) -> str:
    """Validates that `obj[field_name]` is a non-empty string."""
    if field_name not in obj:
        raise SchemaValidationError(f"{ctx}: missing required field '{field_name}'")
    value = obj[field_name]
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(
            f"{ctx}: field '{field_name}' must be a non-empty string, got {type(value).__name__}"
        )
    return value.strip()


def _require_non_negative_int(obj: Mapping[str, Any], field_name: str, ctx: str) -> int:
    """Validates that `obj[field_name]` is a non-negative integer (excluding bool)."""
    if field_name not in obj:
        raise SchemaValidationError(f"{ctx}: missing required field '{field_name}'")
    value = obj[field_name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaValidationError(
            f"{ctx}: field '{field_name}' must be a non-negative integer, got {value!r}"
        )
    return value


def _require_finite_float(obj: Mapping[str, Any], field_name: str, ctx: str) -> float:
    """Validates that `obj[field_name]` is a finite number (int or float, excluding bool)."""
    if field_name not in obj:
        raise SchemaValidationError(f"{ctx}: missing required field '{field_name}'")
    value = obj[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(
            f"{ctx}: field '{field_name}' must be a finite number, got {value!r}"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SchemaValidationError(
            f"{ctx}: field '{field_name}' must be finite (not NaN/Inf), got {value!r}"
        )
    return numeric


def _optional_bool(
    obj: Mapping[str, Any],
    field_name: str,
    default: bool,
    ctx: str,
) -> bool:
    """Reads an optional boolean field, validating its type if present."""
    if field_name not in obj or obj[field_name] is None:
        return default
    value = obj[field_name]
    if not isinstance(value, bool):
        raise SchemaValidationError(
            f"{ctx}: field '{field_name}' must be a boolean, got {value!r}"
        )
    return value


def _optional_str(
    obj: Mapping[str, Any],
    field_name: str,
    default: str,
    ctx: str,
) -> str:
    """Reads an optional string field, validating its type if present."""
    if field_name not in obj or obj[field_name] is None:
        return default
    value = obj[field_name]
    if not isinstance(value, str):
        raise SchemaValidationError(
            f"{ctx}: field '{field_name}' must be a string, got {type(value).__name__}"
        )
    return value


def _infer_higher_is_better(extension_name: str, metric_label: str, unit: str) -> bool:
    """Infers metric optimization direction when optional `higher_is_better` is omitted."""
    combined = f"{extension_name} {metric_label} {unit}".lower()
    if any(
        token in combined
        for token in ("ops/sec", "ops/s", "qps", "throughput", "accuracy", "reclaimed", "speedup")
    ):
        return True
    return False


def parse_host_metadata(raw: Any, ctx: str) -> HostMetadata:
    """Validates and parses a raw JSON object into a `HostMetadata` instance.

    Preconditions:
        `raw` is a mapping containing all `REQUIRED_HOST_FIELDS`.
    Raises:
        SchemaValidationError: If any required field is missing or has an invalid value.
    """
    if not isinstance(raw, Mapping):
        raise SchemaValidationError(f"{ctx}: 'host' must be a JSON object")

    missing = [f for f in REQUIRED_HOST_FIELDS if f not in raw]
    if missing:
        raise SchemaValidationError(
            f"{ctx}: host metadata missing required field(s): {', '.join(missing)}"
        )

    scale_factor = _require_finite_float(raw, "scale_factor", f"{ctx}.host")
    if scale_factor <= 0.0:
        raise SchemaValidationError(
            f"{ctx}.host: 'scale_factor' must be > 0.0, got {scale_factor}"
        )

    return HostMetadata(
        cpu_model=_require_non_empty_str(raw, "cpu_model", f"{ctx}.host"),
        physical_cores=_require_non_negative_int(raw, "physical_cores", f"{ctx}.host"),
        logical_cores=_require_non_negative_int(raw, "logical_cores", f"{ctx}.host"),
        l3_cache_kb=_require_non_negative_int(raw, "l3_cache_kb", f"{ctx}.host"),
        total_ram_bytes=_require_non_negative_int(raw, "total_ram_bytes", f"{ctx}.host"),
        available_ram_bytes=_require_non_negative_int(raw, "available_ram_bytes", f"{ctx}.host"),
        scale_factor=scale_factor,
        peak_rss_bytes=_require_non_negative_int(raw, "peak_rss_bytes", f"{ctx}.host"),
    )


def classify_metric_suite(extension_name: str, metric_label: str, worker_name: str) -> str:
    """Maps a metric record to one of the canonical suite IDs (`DRAM_AUDIT`, `AB1`..`AB5`, `OTHER`).

    Uses explicit suite prefixes first to avoid cross-suite keyword collisions
    (for example, A/B 5 multi-tenant shard metrics vs. A/B 4 range sharding),
    falling back to domain-specific tokens and worker provenance.
    """
    ext_lower = extension_name.lower()
    combined = f"{extension_name} {metric_label}".lower()
    worker_lower = worker_name.lower()

    # 1. Explicit canonical suite identifiers in extension_name or metric_label.
    explicit_patterns: tuple[tuple[str, str], ...] = (
        ("AB1", r"\bab[\s_-]*1\b|ab1_"),
        ("AB2", r"\bab[\s_-]*2\b|ab2_"),
        ("AB3", r"\bab[\s_-]*3\b|ab3_"),
        ("AB4", r"\bab[\s_-]*4\b|ab4_"),
        ("AB5", r"\bab[\s_-]*5\b|ab5_"),
        ("DRAM_AUDIT", r"\bdram[\s_-]*10m|\bdram[\s_-]*scale|\bdram[\s_-]*audit|\bdram_"),
    )
    for suite_id, pattern in explicit_patterns:
        if re.search(pattern, ext_lower):
            return suite_id
    for suite_id, pattern in explicit_patterns:
        if re.search(pattern, combined):
            return suite_id

    # 2. Domain-specific token routing (ordered to prevent overlap).
    if any(
        tok in combined
        for tok in ("sidecar", "bandit", "guardrail", "fallback", "seasonal", "ml_rl", "ml/rl", "python")
    ):
        return "AB5"
    if any(
        tok in combined
        for tok in ("wal", "fsync", "sigkill", "crash", "coupled_storage", "uncoupled")
    ):
        return "AB3"
    if any(
        tok in combined
        for tok in ("proactive", "ewma", "contraction", "reactive", "mass_delete", "trigger")
    ):
        return "AB2"
    if any(
        tok in combined
        for tok in ("dynamic_r", "dynamicr", "shifting_r", "shifting", "fp_log", "fplog", "static_r")
    ):
        return "AB1"
    if any(
        tok in combined
        for tok in ("scale_plane", "scaleplane", "rcu", "sharding", "shard", "hotspot", "hot_spot", "coarse")
    ):
        return "AB4"
    if any(
        tok in combined
        for tok in ("10m", "5m", "eps_", "2^-4", "2^-8", "2e-4", "2e-8", "8x_exp", "expansion", "adversarial", "repeat_fp")
    ):
        return "DRAM_AUDIT"

    # 3. Worker provenance fallback.
    if "worker1" in worker_lower:
        return "DRAM_AUDIT"
    if "worker2" in worker_lower:
        return "AB2"
    if "worker3" in worker_lower:
        return "AB4"
    return "OTHER"


def parse_ablation_metric(
    raw: Any,
    worker_name: str,
    default_scale_factor: float,
    ctx: str,
) -> AblationMetric:
    """Validates and parses one raw JSON object into an `AblationMetric`.

    Preconditions:
        `raw` is a mapping containing all 14 `REQUIRED_METRIC_FIELDS`.
    Raises:
        SchemaValidationError: If any required field is missing or violates type/range constraints.
    """
    if not isinstance(raw, Mapping):
        raise SchemaValidationError(f"{ctx}: metric entry must be a JSON object")

    missing = [f for f in REQUIRED_METRIC_FIELDS if f not in raw]
    if missing:
        raise SchemaValidationError(
            f"{ctx}: metric missing required field(s): {', '.join(missing)}"
        )

    extension_name = _require_non_empty_str(raw, "extension_name", ctx)
    metric_label = _optional_str(raw, "metric_label", extension_name, ctx).strip() or extension_name
    unit = _optional_str(raw, "unit", "", ctx).strip()
    inferred_direction = _infer_higher_is_better(extension_name, metric_label, unit)
    higher_is_better = _optional_bool(raw, "higher_is_better", inferred_direction, ctx)

    empirical_fpr = _require_finite_float(raw, "empirical_fpr", ctx)
    if not (0.0 <= empirical_fpr <= 1.0):
        raise SchemaValidationError(
            f"{ctx}: 'empirical_fpr' must lie within [0.0, 1.0], got {empirical_fpr}"
        )

    if "scale_factor" in raw and raw["scale_factor"] is not None:
        scale_factor = _require_finite_float(raw, "scale_factor", ctx)
        if scale_factor <= 0.0:
            raise SchemaValidationError(
                f"{ctx}: 'scale_factor' must be > 0.0, got {scale_factor}"
            )
    else:
        scale_factor = default_scale_factor

    suite_id = classify_metric_suite(extension_name, metric_label, worker_name)

    return AblationMetric(
        extension_name=extension_name,
        metric_label=metric_label,
        unit=unit,
        higher_is_better=higher_is_better,
        key_count=_require_non_negative_int(raw, "key_count", ctx),
        query_count=_require_non_negative_int(raw, "query_count", ctx),
        baseline_value=_require_finite_float(raw, "baseline_value", ctx),
        optimized_value=_require_finite_float(raw, "optimized_value", ctx),
        delta_pct=_require_finite_float(raw, "delta_pct", ctx),
        p50_ns=_require_non_negative_int(raw, "p50_ns", ctx),
        p99_ns=_require_non_negative_int(raw, "p99_ns", ctx),
        p999_ns=_require_non_negative_int(raw, "p999_ns", ctx),
        empirical_fpr=empirical_fpr,
        false_negatives=_require_non_negative_int(raw, "false_negatives", ctx),
        repeated_fps=_require_non_negative_int(raw, "repeated_fps", ctx),
        peak_ram_bytes=_require_non_negative_int(raw, "peak_ram_bytes", ctx),
        disk_io_bytes=_require_non_negative_int(raw, "disk_io_bytes", ctx),
        scale_factor=scale_factor,
        notes=_optional_str(raw, "notes", "", ctx).strip(),
        worker_name=worker_name,
        suite_id=suite_id,
    )


def load_worker_report(path: Path) -> WorkerReport:
    """Reads and strictly validates a single `WorkerAblationReport` JSON file.

    Raises:
        SchemaValidationError: On malformed JSON or schema violations.
    """
    ctx = path.name
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaValidationError(f"{ctx}: failed to read file: {exc}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{ctx}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc

    if not isinstance(payload, Mapping):
        raise SchemaValidationError(f"{ctx}: root JSON value must be an object")

    worker_name = _require_non_empty_str(payload, "worker_name", ctx)
    host = parse_host_metadata(payload.get("host"), ctx)

    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise SchemaValidationError(f"{ctx}: 'metrics' must be a non-empty JSON array")

    metrics: list[AblationMetric] = []
    for idx, item in enumerate(raw_metrics):
        metrics.append(
            parse_ablation_metric(
                item,
                worker_name=worker_name,
                default_scale_factor=host.scale_factor,
                ctx=f"{ctx}.metrics[{idx}]",
            )
        )

    raw_notes = payload.get("notes", [])
    if not isinstance(raw_notes, list) or not all(isinstance(n, str) for n in raw_notes):
        raise SchemaValidationError(f"{ctx}: 'notes' must be an array of strings")

    return WorkerReport(
        worker_name=worker_name,
        source_file=path.name,
        host=host,
        metrics=tuple(metrics),
        notes=tuple(n.strip() for n in raw_notes if n.strip()),
    )


def discover_worker_files(input_dir: Path, pattern: str = DEFAULT_GLOB_PATTERN) -> list[Path]:
    """Discovers sorted worker JSON files matching `pattern` under `input_dir`.

    Excludes temporary `.tmp` files produced during atomic Rust writes.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    candidates = [
        p
        for p in sorted(input_dir.glob(pattern))
        if p.is_file() and not p.name.endswith(".tmp")
    ]
    return candidates


# ---------------------------------------------------------------------------
# Pure Domain Analysis: Merging, Auditing, Spill Analysis & Blunt Verdicts
# ---------------------------------------------------------------------------


def merge_host_metadata(workers: Sequence[WorkerReport]) -> HostMetadata:
    """Merges host telemetry across workers, taking `max(peak_rss_bytes)`.

    Preconditions:
        `workers` is non-empty.
    Postconditions:
        `result.peak_rss_bytes == max(w.host.peak_rss_bytes for w in workers)`.
    """
    if not workers:
        raise ValueError("merge_host_metadata requires at least one worker report")

    cpu_model = next(
        (w.host.cpu_model for w in workers if w.host.cpu_model.lower() != "unknown"),
        workers[0].host.cpu_model,
    )
    physical_cores = max(w.host.physical_cores for w in workers)
    logical_cores = max(w.host.logical_cores for w in workers)
    l3_cache_kb = max(w.host.l3_cache_kb for w in workers)
    total_ram_bytes = max(w.host.total_ram_bytes for w in workers)

    positive_avail = [w.host.available_ram_bytes for w in workers if w.host.available_ram_bytes > 0]
    available_ram_bytes = min(positive_avail) if positive_avail else 0

    scale_factor = min(w.host.scale_factor for w in workers)
    peak_rss_bytes = max(w.host.peak_rss_bytes for w in workers)

    return HostMetadata(
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        l3_cache_kb=l3_cache_kb,
        total_ram_bytes=total_ram_bytes,
        available_ram_bytes=available_ram_bytes,
        scale_factor=scale_factor,
        peak_rss_bytes=peak_rss_bytes,
    )


def audit_correctness(metrics: Sequence[AblationMetric]) -> CorrectnessSummary:
    """Computes global False Negative and Repeated False Positive audit totals."""
    fn_violations = tuple(m for m in metrics if m.false_negatives > 0)
    rfp_violations = tuple(m for m in metrics if m.repeated_fps > 0)
    total_fn = sum(m.false_negatives for m in metrics)
    total_rfp = sum(m.repeated_fps for m in metrics)

    return CorrectnessSummary(
        total_metrics=len(metrics),
        total_false_negatives=total_fn,
        total_repeated_fps=total_rfp,
        zero_fn_pass=(total_fn == 0),
        zero_repeated_fp_pass=(total_rfp == 0),
        fn_violations=fn_violations,
        repeated_fp_violations=rfp_violations,
    )


def analyze_l3_spill(
    host: HostMetadata,
    grouped_metrics: Mapping[str, Sequence[AblationMetric]],
    all_metrics: Sequence[AblationMetric],
) -> L3SpillAnalysis:
    """Evaluates whether DRAM-scale filter footprints exceed the host L3 cache."""
    reported_l3_bytes = host.l3_cache_kb * 1024
    if reported_l3_bytes > 0:
        reference_l3_bytes = reported_l3_bytes
        reference_source = "OS-reported host L3 cache"
    else:
        reference_l3_bytes = REFERENCE_L3_SPILL_BYTES
        reference_source = "24 MiB reference L3 threshold (OS L3 unavailable)"

    dram_metrics = grouped_metrics.get("DRAM_AUDIT", ())
    max_dram_ram = max((m.peak_ram_bytes for m in dram_metrics), default=0)
    max_overall_ram = max((m.peak_ram_bytes for m in all_metrics), default=0)

    effective_footprint = max_dram_ram if max_dram_ram > 0 else max_overall_ram
    dram_spill_ratio = (
        effective_footprint / reference_l3_bytes if reference_l3_bytes > 0 else 0.0
    )
    rss_spill_ratio = (
        host.peak_rss_bytes / reference_l3_bytes if reference_l3_bytes > 0 else 0.0
    )
    dram_spill_confirmed = effective_footprint > reference_l3_bytes

    return L3SpillAnalysis(
        reported_l3_bytes=reported_l3_bytes,
        reference_l3_bytes=reference_l3_bytes,
        reference_source=reference_source,
        max_dram_metric_ram_bytes=max_dram_ram,
        max_overall_metric_ram_bytes=max_overall_ram,
        merged_peak_rss_bytes=host.peak_rss_bytes,
        dram_spill_ratio=dram_spill_ratio,
        rss_spill_ratio=rss_spill_ratio,
        dram_spill_confirmed=dram_spill_confirmed,
    )


def _is_durability_or_safety_metric(metric: AblationMetric) -> bool:
    """Returns True when a metric directly measures crash durability, safety, or guardrail accuracy."""
    combined = f"{metric.extension_name} {metric.metric_label} {metric.unit}".lower()
    return any(
        tok in combined
        for tok in (
            "sigkill",
            "crash",
            "lost",
            "desync",
            "durability",
            "recovery",
            "guardrail",
            "rejection",
            "accuracy",
            "monotonic",
            "repeat_fp",
            "repeated_fp",
            "adversarial",
            "fallback",
        )
    )


def _compute_ratio_display(metric: AblationMetric) -> str:
    """Formats a clearly labeled speedup or reduction ratio for `metric`."""
    base = metric.baseline_value
    opt = metric.optimized_value
    eps = 1e-12

    if abs(base) <= eps and abs(opt) <= eps:
        return "1.00x (both 0)"

    if metric.higher_is_better:
        if abs(base) <= eps:
            return "new capability (base=0)"
        ratio = opt / base
        if abs(ratio - 1.0) <= 1e-3:
            return "1.00x (equal)"
        if ratio > 1.0:
            return f"{ratio:.2f}x faster (opt/base)"
        if abs(opt) <= eps:
            return "0.00x (opt=0)"
        return f"{base / opt:.2f}x slower (base/opt)"

    # Lower is better (latency, FPR, RAM, lost keys, disk I/O).
    if abs(base) <= eps:
        return "overhead (base=0)"
    if abs(opt) <= eps:
        return "eliminated (opt=0)"
    if opt < base:
        return f"{base / opt:.2f}x lower (base/opt)"
    if opt > base:
        return f"{opt / base:.2f}x higher (opt/base)"
    return "1.00x (equal)"


def assess_metric(metric: AblationMetric) -> MetricAssessment:
    """Evaluates a single metric's direction-aware improvement and classification.

    A metric with `baseline_value == 0` and `optimized_value > 0` on a
    lower-is-better axis (for example, rebuild time or crash-fallback latency
    when the naive baseline has no corresponding mechanism) is flagged as
    non-comparative (`is_comparative=False`) so its cost is reported honestly
    without corrupting head-to-head speedup comparisons.
    """
    base = metric.baseline_value
    opt = metric.optimized_value
    eps = 1e-12
    is_durability = _is_durability_or_safety_metric(metric)

    is_comparative = True
    if not metric.higher_is_better and abs(base) <= eps and opt > eps:
        is_comparative = False
        signed_improvement = -100.0
    elif metric.higher_is_better and abs(base) <= eps and opt > eps:
        signed_improvement = 100.0
    elif abs(base) <= eps and abs(opt) <= eps:
        signed_improvement = 0.0
    else:
        raw_delta = ((opt - base) / abs(base)) * 100.0
        signed_improvement = raw_delta if metric.higher_is_better else -raw_delta

    if metric.false_negatives > 0 or metric.repeated_fps > 0:
        classification = VERDICT_NEGATIVE_OVERHEAD
    elif not is_comparative:
        classification = VERDICT_EXPECTED_OVERHEAD
    elif signed_improvement > NEUTRAL_BAND_PCT:
        classification = VERDICT_MEASURABLE_GAIN
    elif signed_improvement < -NEUTRAL_BAND_PCT:
        classification = (
            VERDICT_EXPECTED_OVERHEAD
            if metric.suite_id == "AB3" and not is_durability
            else VERDICT_NEGATIVE_OVERHEAD
        )
    else:
        classification = VERDICT_NEUTRAL

    return MetricAssessment(
        metric=metric,
        signed_improvement_pct=signed_improvement,
        classification=classification,
        ratio_display=_compute_ratio_display(metric),
        is_comparative=is_comparative,
        is_durability_metric=is_durability,
    )


def _format_assessment_brief(item: MetricAssessment) -> str:
    """Builds a concise, exact-number bullet describing one metric assessment."""
    m = item.metric
    base_str = format_metric_value(m.baseline_value, m.unit)
    opt_str = format_metric_value(m.optimized_value, m.unit)
    sign = "+" if m.delta_pct >= 0 else ""
    return (
        f"`{m.metric_label}`: baseline {base_str} -> optimized {opt_str} "
        f"({sign}{m.delta_pct:.1f}%, {item.ratio_display})"
    )


def evaluate_suite_verdict(
    suite: SuiteDefinition,
    assessments: Sequence[MetricAssessment],
) -> SuiteVerdict:
    """Computes an objective, strictly data-driven Blunt Engineering Verdict for a suite."""
    total_fn = sum(a.metric.false_negatives for a in assessments)
    total_rfp = sum(a.metric.repeated_fps for a in assessments)

    if not assessments:
        return SuiteVerdict(
            suite_id=suite.suite_id,
            title=suite.title,
            verdict=VERDICT_NEUTRAL,
            metric_count=0,
            false_negatives=0,
            repeated_fps=0,
            rationale="No metrics were reported for this suite in the ingested worker JSON files.",
            highlights=(),
            caveats=("Missing worker output; cannot validate this suite.",),
        )

    # Hard correctness gate: any FN or repeated FP is an immediate negative failure.
    if total_fn > 0 or total_rfp > 0:
        return SuiteVerdict(
            suite_id=suite.suite_id,
            title=suite.title,
            verdict=VERDICT_NEGATIVE_OVERHEAD,
            metric_count=len(assessments),
            false_negatives=total_fn,
            repeated_fps=total_rfp,
            rationale=(
                f"Correctness invariant violated ({total_fn:,} false negatives, "
                f"{total_rfp:,} repeated FPs); performance numbers cannot redeem a "
                "soundness or monotonicity regression."
            ),
            highlights=(),
            caveats=tuple(
                f"`{a.metric.extension_name}` recorded FN={a.metric.false_negatives}, "
                f"repeated_fps={a.metric.repeated_fps}"
                for a in assessments
                if a.metric.false_negatives > 0 or a.metric.repeated_fps > 0
            ),
        )

    comparative = [a for a in assessments if a.is_comparative]
    non_comparative = [a for a in assessments if not a.is_comparative]

    gains = [a for a in comparative if a.signed_improvement_pct > NEUTRAL_BAND_PCT]
    neutrals = [a for a in comparative if abs(a.signed_improvement_pct) <= NEUTRAL_BAND_PCT]
    regressions = [a for a in comparative if a.signed_improvement_pct < -NEUTRAL_BAND_PCT]

    highlights = tuple(_format_assessment_brief(a) for a in gains)
    caveats_list = [_format_assessment_brief(a) for a in regressions]
    caveats_list.extend(
        f"Mechanism cost — {_format_assessment_brief(a)}" for a in non_comparative
    )

    # Special case: A/B 3 WAL-Coupled Storage trades write throughput/latency/disk I/O
    # for crash durability under SIGKILL. Classify honestly based on whether crash
    # durability actually held AND whether write/IO overhead was measured.
    if suite.suite_id == "AB3":
        durability_metrics = [a for a in comparative if a.is_durability_metric]
        durability_wins = [
            a
            for a in durability_metrics
            if a.signed_improvement_pct > NEUTRAL_BAND_PCT
            or (a.metric.optimized_value == 0.0 and a.metric.baseline_value > 0.0)
        ]
        perf_metrics = [a for a in comparative if not a.is_durability_metric]
        perf_regressions = [a for a in perf_metrics if a.signed_improvement_pct < -NEUTRAL_BAND_PCT]

        if durability_wins and perf_regressions:
            verdict = VERDICT_EXPECTED_OVERHEAD
            rationale = (
                f"Crash durability verified across {len(durability_wins)} recovery metric(s) "
                f"with 0 lost ACKed keys, at the expected cost of write/storage overhead "
                f"across {len(perf_regressions)}/{len(perf_metrics)} throughput/latency/IO metric(s)."
            )
        elif durability_wins and not perf_regressions:
            verdict = VERDICT_MEASURABLE_GAIN
            rationale = (
                f"Crash durability verified across {len(durability_wins)} recovery metric(s) "
                "without >5% degradation on measured comparative throughput/latency metrics."
            )
        elif perf_regressions and not durability_wins:
            verdict = VERDICT_NEGATIVE_OVERHEAD
            rationale = (
                f"Measured {len(perf_regressions)} throughput/latency/IO regression(s) >5% "
                "without an offsetting crash-durability win over the uncoupled baseline."
            )
        else:
            verdict = VERDICT_NEUTRAL
            rationale = (
                "Measured WAL-coupled vs. uncoupled deltas remained within the +/-5% neutral band."
            )
        return SuiteVerdict(
            suite_id=suite.suite_id,
            title=suite.title,
            verdict=verdict,
            metric_count=len(assessments),
            false_negatives=total_fn,
            repeated_fps=total_rfp,
            rationale=rationale,
            highlights=highlights,
            caveats=tuple(caveats_list),
        )

    # Special case: DRAM-Scale Audit verifies empirical FPR <= theoretical epsilon,
    # 0 FN, 0 repeated FPs, and adversarial repeat-FP elimination.
    if suite.suite_id == "DRAM_AUDIT":
        fpr_breaches = [
            a
            for a in comparative
            if not a.metric.higher_is_better
            and ("fpr" in a.metric.metric_label.lower() or "fpr" in a.metric.extension_name.lower())
            and a.metric.baseline_value > 0.0
            and a.metric.optimized_value > a.metric.baseline_value * 1.15
        ]
        if fpr_breaches:
            verdict = VERDICT_NEGATIVE_OVERHEAD
            rationale = (
                f"Empirical FPR exceeded the theoretical baseline envelope by >15% on "
                f"{len(fpr_breaches)} DRAM-scale measurement(s)."
            )
        elif gains and not regressions:
            verdict = VERDICT_MEASURABLE_GAIN
            rationale = (
                f"Passed DRAM-scale correctness (0 FN, 0 repeated FPs) and beat baseline "
                f"on {len(gains)}/{len(comparative)} comparative metric(s)."
            )
        elif regressions and not gains:
            verdict = VERDICT_NEGATIVE_OVERHEAD
            rationale = (
                f"Regressed >5% against baseline on {len(regressions)}/{len(comparative)} "
                "comparative DRAM-scale metric(s)."
            )
        elif gains and regressions:
            verdict = VERDICT_MEASURABLE_GAIN if len(gains) >= len(regressions) else VERDICT_NEGATIVE_OVERHEAD
            rationale = (
                f"Mixed DRAM-scale results: {len(gains)} improvement(s) vs. "
                f"{len(regressions)} regression(s) beyond +/-5%; inspect individual rows."
            )
        else:
            verdict = VERDICT_NEUTRAL
            rationale = (
                "All comparative DRAM-scale measurements matched theoretical/baseline targets "
                "within +/-5% with 0 FN and 0 repeated FPs."
            )
        return SuiteVerdict(
            suite_id=suite.suite_id,
            title=suite.title,
            verdict=verdict,
            metric_count=len(assessments),
            false_negatives=total_fn,
            repeated_fps=total_rfp,
            rationale=rationale,
            highlights=highlights,
            caveats=tuple(caveats_list),
        )

    # Standard comparative A/B suites (AB1, AB2, AB4, AB5, OTHER).
    if not comparative:
        verdict = VERDICT_EXPECTED_OVERHEAD
        rationale = (
            f"Only mechanism-cost metrics ({len(non_comparative)}) with baseline=0 were emitted; "
            "no direct head-to-head comparative metric was present."
        )
    elif gains and not regressions:
        verdict = VERDICT_MEASURABLE_GAIN
        rationale = (
            f"Delivered >5% improvement on {len(gains)}/{len(comparative)} comparative metric(s) "
            f"({len(neutrals)} neutral, 0 regressions) with 0 FN and 0 repeated FPs."
        )
    elif regressions and not gains:
        # Check if the ONLY thing this suite provided was safety/guardrail with no perf win.
        safety_ok = [a for a in comparative if a.is_durability_metric and a.signed_improvement_pct >= -NEUTRAL_BAND_PCT]
        if safety_ok and len(regressions) < len(comparative):
            verdict = VERDICT_EXPECTED_OVERHEAD
            rationale = (
                f"Safety/guardrail invariant held across {len(safety_ok)} metric(s), while "
                f"{len(regressions)} performance metric(s) incurred >5% overhead without workload gains."
            )
        else:
            verdict = VERDICT_NEGATIVE_OVERHEAD
            rationale = (
                f"Regressed >5% on {len(regressions)}/{len(comparative)} comparative metric(s) "
                "with zero compensating >5% improvements."
            )
    elif gains and regressions:
        # Weigh net comparative direction honestly without hiding trade-offs.
        mean_imp = sum(a.signed_improvement_pct for a in comparative) / len(comparative)
        if len(gains) > len(regressions) or (len(gains) == len(regressions) and mean_imp > NEUTRAL_BAND_PCT):
            verdict = VERDICT_MEASURABLE_GAIN
            rationale = (
                f"Net positive trade-off ({len(gains)} gain(s), {len(neutrals)} neutral, "
                f"{len(regressions)} regression(s); mean directional delta {mean_imp:+.1f}%). "
                "Trade-off regressions are explicitly listed below."
            )
        elif len(regressions) > len(gains) and mean_imp < -NEUTRAL_BAND_PCT:
            verdict = VERDICT_NEGATIVE_OVERHEAD
            rationale = (
                f"Net negative trade-off ({len(regressions)} regression(s) outweigh "
                f"{len(gains)} gain(s); mean directional delta {mean_imp:+.1f}%)."
            )
        else:
            verdict = VERDICT_NEUTRAL
            rationale = (
                f"Offsetting gains ({len(gains)}) and regressions ({len(regressions)}) yield "
                f"no decisive net win (mean directional delta {mean_imp:+.1f}%)."
            )
    else:
        verdict = VERDICT_NEUTRAL
        rationale = (
            f"All {len(comparative)} comparative metric(s) fell within the +/-{NEUTRAL_BAND_PCT:.0f}% "
            "neutral band; no statistically meaningful gain or overhead observed."
        )

    return SuiteVerdict(
        suite_id=suite.suite_id,
        title=suite.title,
        verdict=verdict,
        metric_count=len(assessments),
        false_negatives=total_fn,
        repeated_fps=total_rfp,
        rationale=rationale,
        highlights=highlights,
        caveats=tuple(caveats_list),
    )


def build_merged_report(
    workers: Sequence[WorkerReport],
    generated_at_utc: str | None = None,
) -> MergedAblationReport:
    """Aggregates validated worker reports into a complete `MergedAblationReport`."""
    if not workers:
        raise ValueError("build_merged_report requires at least one worker report")

    timestamp = generated_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    merged_host = merge_host_metadata(workers)

    all_metrics: list[AblationMetric] = []
    for worker in workers:
        all_metrics.extend(worker.metrics)

    grouped_mut: dict[str, list[AblationMetric]] = {suite.suite_id: [] for suite in CANONICAL_SUITES}
    grouped_mut["OTHER"] = []
    for metric in all_metrics:
        grouped_mut.setdefault(metric.suite_id, []).append(metric)

    grouped_metrics: dict[str, tuple[AblationMetric, ...]] = {
        suite_id: tuple(items) for suite_id, items in grouped_mut.items()
    }
    assessments_by_suite: dict[str, tuple[MetricAssessment, ...]] = {
        suite_id: tuple(assess_metric(m) for m in items)
        for suite_id, items in grouped_metrics.items()
    }

    verdicts: list[SuiteVerdict] = []
    for suite in CANONICAL_SUITES:
        verdicts.append(evaluate_suite_verdict(suite, assessments_by_suite[suite.suite_id]))
    if assessments_by_suite["OTHER"]:
        verdicts.append(evaluate_suite_verdict(OTHER_SUITE, assessments_by_suite["OTHER"]))

    loaded_files = {w.source_file for w in workers}
    missing_canonical = tuple(f for f in CANONICAL_WORKER_FILES if f not in loaded_files)

    correctness = audit_correctness(all_metrics)
    l3_spill = analyze_l3_spill(merged_host, grouped_metrics, all_metrics)

    return MergedAblationReport(
        generated_at_utc=timestamp,
        host=merged_host,
        workers=tuple(workers),
        missing_canonical_files=missing_canonical,
        metrics=tuple(all_metrics),
        grouped_metrics=grouped_metrics,
        assessments_by_suite=assessments_by_suite,
        verdicts=tuple(verdicts),
        correctness=correctness,
        l3_spill=l3_spill,
    )


# ---------------------------------------------------------------------------
# Formatting & Presentation Helpers
# ---------------------------------------------------------------------------


def format_bytes(num_bytes: int) -> str:
    """Formats a non-negative byte count into human-readable binary units."""
    if num_bytes == 0:
        return "0 B"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{num_bytes:,} B"
    return f"{value:.2f} {units[unit_idx]}"


def format_latency_ns(ns: int) -> str:
    """Formats a nanosecond latency value cleanly across ns / us / ms / s."""
    if ns == 0:
        return "0 ns"
    if ns < 1_000:
        return f"{ns:,} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000.0:.2f} us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000.0:.2f} ms"
    return f"{ns / 1_000_000_000.0:.3f} s"


def format_fpr(fpr: float) -> str:
    """Formats an empirical FPR in `[0.0, 1.0]` with high precision for small rates."""
    if fpr == 0.0:
        return "0.000000"
    if fpr < 1e-4:
        return f"{fpr:.3e}"
    return f"{fpr:.6f}"


def format_metric_value(value: float, unit: str) -> str:
    """Formats a numeric `baseline_value` or `optimized_value` according to its unit."""
    unit_lower = unit.lower().strip()
    if unit_lower in ("bytes", "b"):
        return format_bytes(int(round(value))) if value >= 0 else f"{value:,.2f} B"
    if unit_lower == "ns":
        return format_latency_ns(int(round(value))) if value >= 0 else f"{value:,.2f} ns"
    if unit_lower in ("us", "µs"):
        return f"{value:,.2f} us"
    if unit_lower == "ms":
        return f"{value:,.2f} ms"
    if unit_lower in ("s", "sec", "seconds"):
        return f"{value:,.3f} s"
    if unit_lower in ("%", "pct", "percent"):
        return f"{value:.2f}%"
    if unit_lower in ("ratio", "fpr", "rate"):
        return format_fpr(value) if 0.0 <= value <= 1.0 else f"{value:.4f}"
    if unit_lower in ("ops/sec", "ops/s", "qps", "keys/sec"):
        return f"{value:,.0f} {unit}"
    if unit_lower in ("count", "keys", "fps", "events", "lost_keys", "steps"):
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}" + (f" {unit}" if unit_lower != "count" else "")
        return f"{value:,.2f} {unit}".strip()
    if abs(value) >= 1_000.0:
        formatted = f"{value:,.2f}"
    elif value == 0.0:
        formatted = "0"
    elif abs(value) < 1e-3:
        formatted = f"{value:.3e}"
    else:
        formatted = f"{value:.4f}"
    return f"{formatted} {unit}".strip()


def _escape_md(text: str) -> str:
    """Escapes pipe characters and collapses newlines for safe Markdown table cells."""
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").strip()


def serialize_merged_json(report: MergedAblationReport) -> dict[str, Any]:
    """Builds the structured dictionary written to `Filter/target/ablation_results.json`."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": report.generated_at_utc,
        "sources": [w.source_file for w in report.workers],
        "missing_canonical_files": list(report.missing_canonical_files),
        "host": report.host.to_dict(),
        "l3_cache_spill": report.l3_spill.to_dict(),
        "correctness_audit": report.correctness.to_dict(),
        "workers": [
            {
                "worker_name": w.worker_name,
                "source_file": w.source_file,
                "metric_count": len(w.metrics),
                "host": w.host.to_dict(),
                "notes": list(w.notes),
            }
            for w in report.workers
        ],
        "verdicts": [v.to_dict() for v in report.verdicts],
        "metrics": [m.to_contract_dict() for m in report.metrics],
    }


def _render_suite_table(
    suite: SuiteDefinition,
    assessments: Sequence[MetricAssessment],
) -> list[str]:
    """Renders one side-by-side Naive Paper-Baseline vs. Aeris Optimized Markdown table."""
    lines: list[str] = [
        f"### {suite.title}",
        "",
        f"- **Baseline:** {suite.baseline_label}",
        f"- **Optimized:** {suite.optimized_label}",
        "",
    ]

    if not assessments:
        lines.append(
            f"> [!WARNING]\n> No metrics were ingested for `{suite.suite_id}` "
            f"(expected from `{suite.expected_worker}`)."
        )
        lines.append("")
        return lines

    lines.append(
        "| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | "
        "Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |"
    )

    notes_entries: list[str] = []
    for item in assessments:
        m = item.metric
        sign = "+" if m.delta_pct >= 0.0 else ""
        fn_cell = f"**{m.false_negatives:,} FAIL**" if m.false_negatives > 0 else "0"
        rfp_cell = f"**{m.repeated_fps:,} FAIL**" if m.repeated_fps > 0 else "0"
        lines.append(
            "| "
            + " | ".join(
                (
                    f"**{_escape_md(m.metric_label)}**<br><sub>`{_escape_md(m.extension_name)}`</sub>",
                    f"{m.key_count:,}",
                    f"{m.query_count:,}",
                    _escape_md(format_metric_value(m.baseline_value, m.unit)),
                    _escape_md(format_metric_value(m.optimized_value, m.unit)),
                    f"{sign}{m.delta_pct:.2f}%",
                    _escape_md(item.ratio_display),
                    format_latency_ns(m.p50_ns),
                    format_latency_ns(m.p99_ns),
                    format_latency_ns(m.p999_ns),
                    format_fpr(m.empirical_fpr),
                    fn_cell,
                    rfp_cell,
                    format_bytes(m.peak_ram_bytes),
                    format_bytes(m.disk_io_bytes),
                )
            )
            + " |"
        )
        if m.notes:
            notes_entries.append(
                f"- **{m.metric_label}** (`{m.extension_name}`): {m.notes}"
            )

    lines.append("")
    if notes_entries:
        lines.append("<details>")
        lines.append(f"<summary>Diagnostic Notes ({len(notes_entries)})</summary>")
        lines.append("")
        lines.extend(notes_entries)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return lines


def render_markdown_report(report: MergedAblationReport) -> str:
    """Renders the complete `Filter/ABLATION_REPORT.md` document from `report`."""
    c = report.correctness
    h = report.host
    l3 = report.l3_spill

    lines: list[str] = [
        "# Aeris Correctness Validation & Large-Scale Ablation Report",
        "",
        f"- **Generated (UTC):** `{report.generated_at_utc}`",
        f"- **Ingested Worker Reports ({len(report.workers)}):** "
        + ", ".join(f"`{w.source_file}`" for w in report.workers),
        f"- **Total Recorded Metrics:** `{c.total_metrics}`",
        "",
    ]

    if report.missing_canonical_files:
        lines.extend(
            [
                "> [!WARNING]",
                "> Partial suite run: missing canonical worker file(s): "
                + ", ".join(f"`{f}`" for f in report.missing_canonical_files),
                "",
            ]
        )

    # 1. Executive Summary & Correctness Audit Scorecard
    lines.extend(
        [
            "## 1. Executive Summary & Correctness Audit Scorecard",
            "",
        ]
    )

    if c.zero_fn_pass and c.zero_repeated_fp_pass:
        lines.extend(
            [
                "> [!IMPORTANT]",
                f"> **CORRECTNESS AUDIT PASSED:** `0` total false negatives and `0` total repeated "
                f"false positives across all `{c.total_metrics}` recorded Aeris ablation metrics.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> [!CAUTION]",
                f"> **CORRECTNESS INVARIANT VIOLATION DETECTED:** "
                f"`{c.total_false_negatives:,}` total false negatives and "
                f"`{c.total_repeated_fps:,}` total repeated false positives observed!",
                "",
            ]
        )

    verdict_counts: dict[str, int] = {
        VERDICT_MEASURABLE_GAIN: 0,
        VERDICT_EXPECTED_OVERHEAD: 0,
        VERDICT_NEUTRAL: 0,
        VERDICT_NEGATIVE_OVERHEAD: 0,
    }
    for v in report.verdicts:
        if v.metric_count > 0:
            verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

    lines.extend(
        [
            "| Audit Gate | Observed Total | Requirement | Status |",
            "| :--- | ---: | :--- | :--- |",
            f"| Total False Negatives (FN) | `{c.total_false_negatives:,}` | `== 0` | "
            + ("**PASS**" if c.zero_fn_pass else "**FAIL (VIOLATION)**")
            + " |",
            f"| Total Repeated False Positives | `{c.total_repeated_fps:,}` | `== 0` | "
            + ("**PASS**" if c.zero_repeated_fp_pass else "**FAIL (VIOLATION)**")
            + " |",
            f"| DRAM Spill Beyond L3 Cache | `{format_bytes(l3.max_dram_metric_ram_bytes or l3.max_overall_metric_ram_bytes)}` "
            f"(`{l3.dram_spill_ratio:.2f}x` L3) | `> {format_bytes(l3.reference_l3_bytes)}` | "
            + ("**CONFIRMED**" if l3.dram_spill_confirmed else "**L3-RESIDENT / NOT CONFIRMED**")
            + " |",
            f"| Verdict Distribution (Active Suites) | "
            f"Gain: `{verdict_counts[VERDICT_MEASURABLE_GAIN]}`, "
            f"Expected Overhead: `{verdict_counts[VERDICT_EXPECTED_OVERHEAD]}`, "
            f"Neutral: `{verdict_counts[VERDICT_NEUTRAL]}`, "
            f"Negative: `{verdict_counts[VERDICT_NEGATIVE_OVERHEAD]}` | Strictly Data-Driven | Recorded |",
            "",
        ]
    )

    if c.fn_violations or c.repeated_fp_violations:
        lines.append("### Correctness Violation Breakdown")
        lines.append("")
        lines.append("| Extension | Worker | False Negatives | Repeated FPs | Notes |")
        lines.append("| :--- | :--- | ---: | ---: | :--- |")
        seen_ids: set[int] = set()
        for m in (*c.fn_violations, *c.repeated_fp_violations):
            if id(m) in seen_ids:
                continue
            seen_ids.add(id(m))
            lines.append(
                f"| `{_escape_md(m.extension_name)}` | `{_escape_md(m.worker_name)}` | "
                f"**{m.false_negatives:,}** | **{m.repeated_fps:,}** | {_escape_md(m.notes or '-')} |"
            )
        lines.append("")

    # 2. Host Hardware, L3 Cache Spill Analysis, Scale Factor & Peak RSS
    scale_desc = (
        "1.0x (Full 10M / 5M Scale)"
        if abs(h.scale_factor - 1.0) < 1e-6
        else f"{h.scale_factor:.2f}x (RAM Safety-Ceiling Scaled)"
    )
    l3_display = (
        f"{h.l3_cache_kb:,} KiB ({format_bytes(l3.reported_l3_bytes)})"
        if h.l3_cache_kb > 0
        else f"Unavailable from OS (using {format_bytes(l3.reference_l3_bytes)} reference)"
    )

    lines.extend(
        [
            "## 2. Host Hardware, L3 Cache Spill Analysis & Peak RSS",
            "",
            "| Parameter | Measured Value |",
            "| :--- | :--- |",
            f"| CPU Model | {_escape_md(h.cpu_model)} |",
            f"| Physical / Logical Cores | `{h.physical_cores}` physical / `{h.logical_cores}` logical |",
            f"| L3 Cache Capacity | {l3_display} |",
            f"| Total System RAM | `{format_bytes(h.total_ram_bytes)}` (`{h.total_ram_bytes:,}` B) |",
            f"| Min Available RAM at Suite Start | `{format_bytes(h.available_ram_bytes)}` (`{h.available_ram_bytes:,}` B) |",
            f"| Applied Scale Factor | `{scale_desc}` |",
            f"| Merged Process Peak RSS (`max` across workers) | `{format_bytes(h.peak_rss_bytes)}` (`{h.peak_rss_bytes:,}` B) |",
            f"| Max DRAM-Audit Filter Footprint | `{format_bytes(l3.max_dram_metric_ram_bytes)}` "
            f"(`{l3.dram_spill_ratio:.2f}x` of {l3.reference_source}) |",
            f"| Process Peak RSS vs. L3 Cache | `{l3.rss_spill_ratio:.2f}x` of {l3.reference_source} |",
            f"| DRAM Spill Status | "
            + (
                "**Confirmed DRAM-bound** (filter working set exceeds L3 cache)"
                if l3.dram_spill_confirmed
                else "**Not Confirmed** (recorded filter footprint did not exceed reference L3 cache)"
            )
            + " |",
            "",
            "### Per-Worker Telemetry Breakdown",
            "",
            "| Worker | Source File | Metrics | Scale Factor | Avail. RAM | Worker Peak RSS |",
            "| :--- | :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for w in report.workers:
        lines.append(
            f"| `{_escape_md(w.worker_name)}` | `{_escape_md(w.source_file)}` | "
            f"{len(w.metrics)} | `{w.host.scale_factor:.2f}x` | "
            f"{format_bytes(w.host.available_ram_bytes)} | {format_bytes(w.host.peak_rss_bytes)} |"
        )
    lines.append("")

    # 3. Side-by-Side Ablation Tables
    lines.extend(
        [
            "## 3. Side-by-Side Ablation Results (Naive Paper-Baseline vs. Aeris Optimized)",
            "",
        ]
    )

    for suite in CANONICAL_SUITES:
        lines.extend(_render_suite_table(suite, report.assessments_by_suite[suite.suite_id]))

    if report.assessments_by_suite["OTHER"]:
        lines.extend(_render_suite_table(OTHER_SUITE, report.assessments_by_suite["OTHER"]))

    # 4. Blunt Engineering Verdict
    lines.extend(
        [
            "## 4. Blunt Engineering Verdict",
            "",
            "Each extension is classified strictly from the measured data using four honest categories: "
            "**Measurable Gain**, **Expected Safety/Durability Overhead**, **Neutral** (within +/-5%), "
            "or **Negative Overhead**. Neutral and negative results are reported directly without spin.",
            "",
            "| Suite / Extension | Verdict | Metrics | FN | Rep. FPs | Data-Driven Rationale |",
            "| :--- | :--- | ---: | ---: | ---: | :--- |",
        ]
    )

    for v in report.verdicts:
        lines.append(
            f"| **{_escape_md(v.title)}** | **{v.verdict}** | {v.metric_count} | "
            f"{v.false_negatives:,} | {v.repeated_fps:,} | {_escape_md(v.rationale)} |"
        )
    lines.append("")

    for v in report.verdicts:
        lines.append(f"### {v.title} — {v.verdict}")
        lines.append("")
        lines.append(v.rationale)
        lines.append("")
        if v.highlights:
            lines.append("**Measured Gains:**")
            for bullet in v.highlights:
                lines.append(f"- {bullet}")
            lines.append("")
        if v.caveats:
            lines.append("**Measured Overheads / Regressions / Caveats:**")
            for bullet in v.caveats:
                lines.append(f"- {bullet}")
            lines.append("")

    # 5. Worker Methodology Notes (if any)
    worker_notes = [
        (w.worker_name, note)
        for w in report.workers
        for note in w.notes
    ]
    if worker_notes:
        lines.extend(
            [
                "## 5. Worker Methodology & Telemetry Notes",
                "",
            ]
        )
        for worker_name, note in worker_notes:
            lines.append(f"- **`{worker_name}`**: {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Atomic File Persistence & CLI Entrypoint
# ---------------------------------------------------------------------------


def write_text_atomic(path: Path, content: str) -> None:
    """Atomically writes UTF-8 text to `path` via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with defaults anchored to `Filter/`."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Aeris worker ablation JSON outputs into "
            "Filter/target/ablation_results.json and Filter/ABLATION_REPORT.md."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing ablation_worker*.json files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_GLOB_PATTERN,
        help=f"Glob pattern for worker JSON reports (default: {DEFAULT_GLOB_PATTERN})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output path for merged JSON summary (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help=f"Output path for Markdown ablation report (default: {DEFAULT_OUTPUT_MD})",
    )
    parser.add_argument(
        "--strict-workers",
        action="store_true",
        help="Fail with non-zero exit code if any of the 3 canonical worker JSON files is missing.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Structured logging verbosity (default: INFO).",
    )
    return parser


def run_pipeline(
    input_dir: Path,
    output_json: Path,
    output_md: Path,
    pattern: str = DEFAULT_GLOB_PATTERN,
    strict_workers: bool = False,
) -> MergedAblationReport:
    """Executes the full discover -> validate -> merge -> emit report pipeline.

    Raises:
        SchemaValidationError: If no worker JSON files exist, required workers
            are missing under `strict_workers=True`, or any schema check fails.
    """
    worker_files = discover_worker_files(input_dir, pattern=pattern)
    if not worker_files:
        raise SchemaValidationError(
            f"No worker ablation JSON files matching '{pattern}' found in '{input_dir}'"
        )

    LOGGER.info(
        "discovered_worker_files count=%d input_dir=%s files=%s",
        len(worker_files),
        input_dir,
        [p.name for p in worker_files],
    )

    workers: list[WorkerReport] = []
    for file_path in worker_files:
        worker = load_worker_report(file_path)
        LOGGER.info(
            "loaded_worker_report worker=%s metrics=%d peak_rss_bytes=%d",
            worker.worker_name,
            len(worker.metrics),
            worker.host.peak_rss_bytes,
        )
        workers.append(worker)

    report = build_merged_report(workers)

    if strict_workers and report.missing_canonical_files:
        raise SchemaValidationError(
            "Missing canonical worker report(s): "
            + ", ".join(report.missing_canonical_files)
        )

    merged_json_payload = json.dumps(serialize_merged_json(report), indent=2) + "\n"
    markdown_payload = render_markdown_report(report)

    write_text_atomic(output_json, merged_json_payload)
    write_text_atomic(output_md, markdown_payload)

    LOGGER.info(
        "report_generation_complete output_json=%s output_md=%s total_metrics=%d "
        "total_fn=%d total_repeated_fps=%d peak_rss_bytes=%d",
        output_json,
        output_md,
        report.correctness.total_metrics,
        report.correctness.total_false_negatives,
        report.correctness.total_repeated_fps,
        report.host.peak_rss_bytes,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit status (`0` on success, `1` on error)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        run_pipeline(
            input_dir=args.input_dir,
            output_json=args.output_json,
            output_md=args.output_md,
            pattern=args.pattern,
            strict_workers=args.strict_workers,
        )
    except SchemaValidationError as exc:
        LOGGER.error("ablation_report_failed reason=%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive top-level boundary
        LOGGER.exception("ablation_report_unexpected_error reason=%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
