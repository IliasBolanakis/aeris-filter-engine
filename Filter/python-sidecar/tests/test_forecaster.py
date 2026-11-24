"""Unit and statistical accuracy tests for SeasonalWorkloadForecaster."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aeris_sidecar.forecaster import SeasonalWorkloadForecaster
from aeris_sidecar.trainer import generate_seasonal_workload


def test_seasonal_forecast_accuracy_beats_persistence_and_has_high_confidence() -> None:
    rng = np.random.default_rng(2026)
    ingest, qlen = generate_seasonal_workload(rng, num_steps=192, period=24.0, noise_scale=0.02)

    train_in, val_in = ingest[:144], ingest[144:]
    train_ql, val_ql = qlen[:144], qlen[144:]

    model = SeasonalWorkloadForecaster(period=24.0, num_harmonics=3, damping=0.92)
    model.fit(train_in, train_ql)

    result = model.predict(train_in, train_ql, horizon=len(val_in))
    preds = np.asarray(result.ingest_trajectory)
    model_rmse = float(np.sqrt(np.mean((val_in - preds) ** 2)))
    persistence_rmse = float(np.sqrt(np.mean((val_in - train_in[-1]) ** 2)))

    assert model_rmse < persistence_rmse * 0.35
    assert result.confidence >= 0.85
    assert result.forecast_peak_ingest_rate == pytest.approx(float(np.max(preds)))
    assert abs(result.forecast_peak_ingest_rate - float(np.max(val_in))) / float(np.max(val_in)) < 0.12
    assert result.forecast_p95_query_len >= 1.0


def test_phase_shifted_seasonal_window_adapts_cleanly() -> None:
    rng = np.random.default_rng(7)
    ingest, qlen = generate_seasonal_workload(rng, num_steps=200, period=24.0, noise_scale=0.02)

    model = SeasonalWorkloadForecaster(period=24.0, num_harmonics=3)
    model.fit(ingest[:120], qlen[:120])

    # Slice starting at an arbitrary phase offset t=37 (shifted by 13 steps mod 24).
    window_in = ingest[37:85]
    window_ql = qlen[37:85]
    expected_next_in = ingest[85:91]

    res = model.predict(window_in, window_ql, horizon=6)
    preds = np.asarray(res.ingest_trajectory)
    rel_err = float(np.sqrt(np.mean((expected_next_in - preds) ** 2))) / float(np.std(window_in))

    assert rel_err < 0.25
    assert res.confidence >= 0.85


def test_low_confidence_on_sparse_and_noisy_ood_series() -> None:
    rng = np.random.default_rng(99)
    ingest, qlen = generate_seasonal_workload(rng, num_steps=144, period=24.0, noise_scale=0.02)
    model = SeasonalWorkloadForecaster(period=24.0, num_harmonics=3, min_confident_points=8)
    model.fit(ingest, qlen)

    # Empty series -> very low confidence, safe non-negative fallback
    empty_res = model.predict([], [], horizon=4, fallback_ingest=500.0, fallback_query_len=32.0)
    assert empty_res.confidence < 0.50
    assert empty_res.forecast_peak_ingest_rate == pytest.approx(500.0)
    assert empty_res.forecast_p95_query_len == pytest.approx(32.0)

    # Short series (3 samples < min_confident_points=8) -> low confidence
    short_res = model.predict(ingest[:3], qlen[:3], horizon=4)
    assert short_res.confidence < 0.60

    # High-variance pure white noise -> poor seasonal fit -> low confidence
    noisy_in = rng.normal(10_000.0, 5_000.0, size=48)
    noisy_ql = np.maximum(rng.normal(200.0, 150.0, size=48), 1.0)
    noisy_res = model.predict(noisy_in, noisy_ql, horizon=6)
    assert noisy_res.confidence < 0.60
    assert math.isfinite(noisy_res.forecast_peak_ingest_rate)
    assert noisy_res.forecast_peak_ingest_rate >= 0.0


def test_non_finite_inputs_are_sanitized() -> None:
    rng = np.random.default_rng(11)
    ingest, qlen = generate_seasonal_workload(rng, num_steps=120, period=24.0)
    model = SeasonalWorkloadForecaster()
    model.fit(ingest, qlen)

    corrupted_in = [1000.0, float("nan"), float("inf"), -float("inf")]
    corrupted_ql = [16.0, float("nan")]
    res = model.predict(
        corrupted_in,
        corrupted_ql,
        horizon=3,
        fallback_ingest=float("nan"),
        fallback_query_len=float("inf"),
    )
    assert math.isfinite(res.forecast_peak_ingest_rate)
    assert res.forecast_peak_ingest_rate >= 0.0
    assert math.isfinite(res.forecast_p95_query_len)
    assert res.forecast_p95_query_len >= 1.0
    assert 0.0 <= res.confidence <= 1.0


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    ingest, qlen = generate_seasonal_workload(rng, num_steps=144, period=24.0)
    model = SeasonalWorkloadForecaster(period=24.0, num_harmonics=3, model_version="test-v2")
    model.fit(ingest[:120], qlen[:120])

    artifact = tmp_path / "forecaster.json"
    model.save(artifact)
    restored = SeasonalWorkloadForecaster.load(artifact)

    assert restored.model_version == "test-v2"
    assert np.allclose(restored.ingest_weights, model.ingest_weights)
    assert np.allclose(restored.query_weights, model.query_weights)

    r1 = model.predict(ingest[96:132], qlen[96:132], horizon=6)
    r2 = restored.predict(ingest[96:132], qlen[96:132], horizon=6)
    assert r1 == r2


def test_invalid_hyperparameters_and_inputs_raise() -> None:
    with pytest.raises(ValueError):
        SeasonalWorkloadForecaster(period=1.0)
    with pytest.raises(ValueError):
        SeasonalWorkloadForecaster(num_harmonics=0)
    with pytest.raises(ValueError):
        SeasonalWorkloadForecaster(damping=1.0)
    with pytest.raises(ValueError):
        SeasonalWorkloadForecaster(ridge_lambda=0.0)

    model = SeasonalWorkloadForecaster()
    with pytest.raises(ValueError):
        model.fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        model.predict([1.0, 2.0], [1.0, 2.0], horizon=0)
