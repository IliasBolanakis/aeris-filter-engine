"""Offline-trained seasonal time-series workload forecaster (pure NumPy).

Models periodic ingestion spikes and range-query span drifts using a damped
linear trend combined with Fourier seasonal harmonics fitted via Ridge OLS.
At inference time, performs regularized MAP adaptation around the offline
prior weights and computes a calibrated confidence score that scales down on
sparse series or high out-of-distribution residual variance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
    """Output of a workload forecast over an upcoming time horizon.

    Attributes:
        forecast_peak_ingest_rate: Maximum forecasted key ingestion rate
            (keys/sec) across the upcoming horizon (guaranteed finite and >= 0).
        forecast_p95_query_len: Forecasted 95th-percentile query span across
            the upcoming horizon (guaranteed finite and >= 1.0).
        confidence: Calibrated model confidence score in ``[0.0, 1.0]``.
            Scales downward when input history is short or noisy/OOD.
        ingest_trajectory: Per-step forecasted ingestion rates over the horizon.
        query_len_trajectory: Per-step forecasted query spans over the horizon.
    """

    forecast_peak_ingest_rate: float
    forecast_p95_query_len: float
    confidence: float
    ingest_trajectory: tuple[float, ...]
    query_len_trajectory: tuple[float, ...]


class SeasonalWorkloadForecaster:
    """Damped-trend + Fourier-harmonic seasonal workload forecaster.

    Feature basis at discrete step ``t``:
        ``[1, (1 - damping^t) / ((1 - damping) * period), sin(2*pi*k*t/P), cos(2*pi*k*t/P) ...]``
    for harmonics ``k = 1..num_harmonics``.

    All numerical operations use pure NumPy and deterministic linear algebra.
    """

    def __init__(
        self,
        period: float = 24.0,
        num_harmonics: int = 3,
        damping: float = 0.92,
        ridge_lambda: float = 1e-2,
        prior_strength: float = 0.25,
        min_confident_points: int = 8,
        model_version: str = "seasonal-forecaster-v1",
    ) -> None:
        if period <= 1.0:
            raise ValueError(f"period must be > 1.0, got {period}")
        if num_harmonics < 1:
            raise ValueError(f"num_harmonics must be >= 1, got {num_harmonics}")
        if not (0.0 < damping < 1.0):
            raise ValueError(f"damping must be in (0, 1), got {damping}")
        if ridge_lambda <= 0.0:
            raise ValueError(f"ridge_lambda must be > 0, got {ridge_lambda}")
        if prior_strength < 0.0:
            raise ValueError(f"prior_strength must be >= 0, got {prior_strength}")
        if min_confident_points < 2:
            raise ValueError(
                f"min_confident_points must be >= 2, got {min_confident_points}"
            )

        self.period = float(period)
        self.num_harmonics = int(num_harmonics)
        self.damping = float(damping)
        self.ridge_lambda = float(ridge_lambda)
        self.prior_strength = float(prior_strength)
        self.min_confident_points = int(min_confident_points)
        self.model_version = str(model_version)

        n_features = self.num_features
        self.ingest_weights: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self.ingest_residual_std: float = 1.0
        self.ingest_scale: float = 1.0

        self.query_weights: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self.query_residual_std: float = 1.0
        self.query_scale: float = 1.0
        self.is_fitted: bool = False

    @property
    def num_features(self) -> int:
        """Total number of basis features (intercept + damped trend + 2*K harmonics)."""
        return 2 + 2 * self.num_harmonics

    def _design_matrix(self, steps: np.ndarray) -> np.ndarray:
        """Builds the `(N, num_features)` damped-trend + Fourier design matrix."""
        t = np.asarray(steps, dtype=np.float64)
        n = t.shape[0]
        x = np.empty((n, self.num_features), dtype=np.float64)
        x[:, 0] = 1.0
        # Damped cumulative trend normalized by period for stable conditioning.
        x[:, 1] = (1.0 - np.power(self.damping, t)) / ((1.0 - self.damping) * self.period)
        base_angle = (2.0 * math.pi / self.period) * t
        for k in range(1, self.num_harmonics + 1):
            angle = k * base_angle
            x[:, 2 * k] = np.sin(angle)
            x[:, 2 * k + 1] = np.cos(angle)
        return x

    def _fit_single_series(
        self, values: np.ndarray, log_transform: bool
    ) -> tuple[np.ndarray, float, float]:
        """Fits Ridge OLS on a single 1D historical series."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] < self.num_features + 2:
            raise ValueError(
                f"training series must be 1D with at least {self.num_features + 2} points, "
                f"got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("training series contains non-finite values")

        y = np.log1p(np.maximum(arr, 0.0)) if log_transform else arr
        steps = np.arange(y.shape[0], dtype=np.float64)
        x = self._design_matrix(steps)

        reg = np.eye(self.num_features, dtype=np.float64) * self.ridge_lambda
        reg[0, 0] = 1e-8  # Do not shrink the intercept.
        lhs = x.T @ x + reg
        rhs = x.T @ y
        weights = np.linalg.solve(lhs, rhs)

        residuals = y - (x @ weights)
        resid_std = float(np.sqrt(np.mean(residuals * residuals)))
        scale = float(np.std(y))
        return weights, max(resid_std, 1e-6), max(scale, 1e-6)

    def fit(
        self,
        ingest_series: Sequence[float] | np.ndarray,
        query_len_series: Sequence[float] | np.ndarray,
    ) -> SeasonalWorkloadForecaster:
        """Fits both ingestion-rate and query-span models offline via Ridge OLS.

        Args:
            ingest_series: Historical key ingestion rate observations (keys/sec).
            query_len_series: Historical p95 query length observations.

        Returns:
            ``self`` for method chaining.
        """
        (
            self.ingest_weights,
            self.ingest_residual_std,
            self.ingest_scale,
        ) = self._fit_single_series(
            np.asarray(ingest_series, dtype=np.float64), log_transform=False
        )
        (
            self.query_weights,
            self.query_residual_std,
            self.query_scale,
        ) = self._fit_single_series(
            np.asarray(query_len_series, dtype=np.float64), log_transform=True
        )
        self.is_fitted = True
        return self

    def _predict_single(
        self,
        raw_series: Sequence[float] | np.ndarray,
        prior_weights: np.ndarray,
        train_scale: float,
        horizon: int,
        fallback_value: float,
        min_value: float,
        log_transform: bool,
    ) -> tuple[np.ndarray, float]:
        """Adapts offline weights via regularized MAP on `raw_series` and
        forecasts `horizon` steps ahead with calibrated confidence.
        """
        arr = np.asarray(raw_series, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = int(arr.shape[0])

        if n < 2:
            val = max(float(fallback_value) if math.isfinite(fallback_value) else min_value, min_value)
            preds = np.full(horizon, val, dtype=np.float64)
            # Sparse series carries low confidence so guardrails can reject or defer.
            sparse_conf = 0.25 if n == 0 else 0.35
            return preds, sparse_conf

        y = np.log1p(np.maximum(arr, 0.0)) if log_transform else arr
        steps = np.arange(n, dtype=np.float64)
        x = self._design_matrix(steps)

        # Prior precision decays smoothly as more live seasonal observations arrive,
        # allowing arbitrary phase alignment while keeping short windows regularized.
        effective_prior = self.prior_strength / max(n / self.period, 1.0)
        prior_diag = np.full(self.num_features, effective_prior, dtype=np.float64)
        prior_diag[0] = min(effective_prior * 0.1, 1e-3)
        prior_mat = np.diag(prior_diag)

        lhs = x.T @ x + prior_mat + np.eye(self.num_features) * self.ridge_lambda
        rhs = (x.T @ y) + (prior_mat @ prior_weights)
        w_post = np.linalg.solve(lhs, rhs)

        fitted = x @ w_post
        residuals = y - fitted
        resid_rmse = float(np.sqrt(np.mean(residuals * residuals)))

        obs_std = float(np.std(y))
        denom = max(obs_std, train_scale * 0.25, 1e-6)
        nrmse = resid_rmse / denom

        # Penalize both insufficient sample length and high unexplained variance.
        length_factor = min(1.0, n / float(self.min_confident_points))
        fit_quality = math.exp(-1.5 * nrmse)
        confidence = float(np.clip(length_factor * fit_quality, 0.05, 0.99))

        future_steps = np.arange(n, n + horizon, dtype=np.float64)
        x_future = self._design_matrix(future_steps)
        raw_preds = x_future @ w_post

        if log_transform:
            preds = np.expm1(np.clip(raw_preds, -20.0, 40.0))
        else:
            preds = raw_preds
        preds = np.maximum(preds, min_value)
        return preds, confidence

    def predict(
        self,
        ingest_series: Sequence[float] | np.ndarray,
        query_len_series: Sequence[float] | np.ndarray,
        horizon: int = 6,
        fallback_ingest: float = 0.0,
        fallback_query_len: float = 1.0,
    ) -> ForecastResult:
        """Predicts upcoming peak ingestion rate, p95 query span, and confidence.

        Args:
            ingest_series: Recent ingestion rate observations (keys/sec).
            query_len_series: Recent p95 query length observations.
            horizon: Number of future steps to forecast (`>= 1`).
            fallback_ingest: Scalar fallback if `ingest_series` is sparse.
            fallback_query_len: Scalar fallback if `query_len_series` is sparse.

        Returns:
            A :class:`ForecastResult` containing peak forecasts, trajectories,
            and calibrated confidence in ``[0.0, 1.0]``.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        ingest_preds, ingest_conf = self._predict_single(
            raw_series=ingest_series,
            prior_weights=self.ingest_weights,
            train_scale=self.ingest_scale,
            horizon=horizon,
            fallback_value=fallback_ingest,
            min_value=0.0,
            log_transform=False,
        )
        query_preds, query_conf = self._predict_single(
            raw_series=query_len_series,
            prior_weights=self.query_weights,
            train_scale=self.query_scale,
            horizon=horizon,
            fallback_value=fallback_query_len,
            min_value=1.0,
            log_transform=True,
        )

        # Conservative combination: if either series is sparse or OOD, lower confidence.
        confidence = float(np.clip(min(ingest_conf, query_conf), 0.0, 1.0))
        peak_ingest = float(np.max(ingest_preds))
        p95_query = float(np.percentile(query_preds, 95))

        return ForecastResult(
            forecast_peak_ingest_rate=max(peak_ingest, 0.0),
            forecast_p95_query_len=max(p95_query, 1.0),
            confidence=confidence,
            ingest_trajectory=tuple(float(v) for v in ingest_preds),
            query_len_trajectory=tuple(float(v) for v in query_preds),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes model hyperparameters and trained weights to a JSON-safe dict."""
        return {
            "model_version": self.model_version,
            "period": self.period,
            "num_harmonics": self.num_harmonics,
            "damping": self.damping,
            "ridge_lambda": self.ridge_lambda,
            "prior_strength": self.prior_strength,
            "min_confident_points": self.min_confident_points,
            "is_fitted": self.is_fitted,
            "ingest_weights": self.ingest_weights.tolist(),
            "ingest_residual_std": self.ingest_residual_std,
            "ingest_scale": self.ingest_scale,
            "query_weights": self.query_weights.tolist(),
            "query_residual_std": self.query_residual_std,
            "query_scale": self.query_scale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SeasonalWorkloadForecaster:
        """Restores a :class:`SeasonalWorkloadForecaster` from a dictionary."""
        model = cls(
            period=float(payload["period"]),
            num_harmonics=int(payload["num_harmonics"]),
            damping=float(payload["damping"]),
            ridge_lambda=float(payload["ridge_lambda"]),
            prior_strength=float(payload["prior_strength"]),
            min_confident_points=int(payload["min_confident_points"]),
            model_version=str(payload.get("model_version", "seasonal-forecaster-v1")),
        )
        model.ingest_weights = np.asarray(payload["ingest_weights"], dtype=np.float64)
        model.ingest_residual_std = float(payload["ingest_residual_std"])
        model.ingest_scale = float(payload["ingest_scale"])
        model.query_weights = np.asarray(payload["query_weights"], dtype=np.float64)
        model.query_residual_std = float(payload["query_residual_std"])
        model.query_scale = float(payload["query_scale"])
        model.is_fitted = bool(payload.get("is_fitted", True))

        expected = (model.num_features,)
        if model.ingest_weights.shape != expected or model.query_weights.shape != expected:
            raise ValueError(
                f"corrupted weight shapes: expected {expected}, got "
                f"{model.ingest_weights.shape} and {model.query_weights.shape}"
            )
        return model

    def save(self, path: str | Path) -> None:
        """Persists trained forecaster weights as UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SeasonalWorkloadForecaster:
        """Loads a previously saved forecaster from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
