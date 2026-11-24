"""Offline contextual bandit for multi-tenant shard RAM allocation (pure NumPy).

Implements a Ridge-regularized LinUCB / Doubly-Robust (DR) offline contextual
bandit mapping per-shard context features:
    ``[log1p(key_count), log1p(query_rate), observed_fpr, log1p(p95_query_len), load_factor]``
to discrete RAM share multiplier actions. Ensures strict global budget
conservation (``sum(allocated_memory_bytes) <= total_memory_budget_bytes``)
and valid power-of-two ``recommended_q`` and Goswami-compliant ``(r, m)``
bit allocations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_ACTIONS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
MIN_QUOTIENT_BITS: int = 6
MAX_QUOTIENT_BITS: int = 40
DEFAULT_M_MAX: int = 20
NUM_RAW_FEATURES: int = 5


@dataclass(frozen=True)
class ShardContext:
    """Normalized per-shard telemetry passed to :class:`OfflineContextualBandit`."""

    shard_id: int
    key_count: int
    current_memory_bytes: int
    query_rate: float
    observed_fpr: float
    p95_query_len: float
    load_factor: float
    current_q: int
    current_r: int
    current_m: int


@dataclass(frozen=True)
class ShardAllocationPlanItem:
    """Per-shard memory and geometry recommendation produced by the bandit."""

    shard_id: int
    allocated_memory_bytes: int
    recommended_q: int
    recommended_r: int
    recommended_m: int


def min_remainder_bits(target_epsilon: float) -> int:
    """Computes Goswami's monotonic adaptivity floor ``r_min(epsilon)``."""
    if not math.isfinite(target_epsilon) or target_epsilon <= 0.0 or target_epsilon >= 0.5:
        raise ValueError(
            f"target_epsilon must be finite and in (0.0, 0.5), got {target_epsilon}"
        )
    inv = 1.0 / target_epsilon
    log_inv = math.log2(inv)
    return int(math.ceil(log_inv + math.log2(log_inv) + 1.0))


def extract_raw_features(shard: ShardContext) -> np.ndarray:
    """Extracts the 5-dimensional raw context vector for a shard."""
    key_count = max(float(shard.key_count), 0.0)
    query_rate = max(float(shard.query_rate), 0.0) if math.isfinite(shard.query_rate) else 0.0
    fpr = float(np.clip(shard.observed_fpr, 0.0, 1.0)) if math.isfinite(shard.observed_fpr) else 0.0
    p95 = max(float(shard.p95_query_len), 0.0) if math.isfinite(shard.p95_query_len) else 0.0
    alpha = float(np.clip(shard.load_factor, 0.0, 1.0)) if math.isfinite(shard.load_factor) else 0.0

    return np.array(
        [
            math.log1p(key_count),
            math.log1p(query_rate),
            fpr,
            math.log1p(p95),
            alpha,
        ],
        dtype=np.float64,
    )


class OfflineContextualBandit:
    """Ridge-regularized LinUCB / Doubly-Robust offline contextual bandit.

    Each discrete arm represents a relative memory-share multiplier applied to
    a shard's baseline demand. During offline fitting, direct Ridge reward
    models are refined via clipped Doubly-Robust (DR) pseudo-rewards when
    behavior-policy propensities are provided, and per-arm inverse covariance
    matrices ``A_a^{-1}`` quantify epistemic uncertainty for conservative action
    selection and calibrated confidence reporting.
    """

    def __init__(
        self,
        actions: Sequence[float] = DEFAULT_ACTIONS,
        ridge_lambda: float = 1.0,
        pessimism_beta: float = 0.15,
        ips_clip: float = 10.0,
        model_version: str = "offline-contextual-bandit-v1",
    ) -> None:
        if len(actions) < 2:
            raise ValueError("bandit requires at least 2 discrete actions")
        if any(a <= 0.0 or not math.isfinite(a) for a in actions):
            raise ValueError("all action multipliers must be finite and > 0")
        if ridge_lambda <= 0.0:
            raise ValueError(f"ridge_lambda must be > 0, got {ridge_lambda}")
        if pessimism_beta < 0.0:
            raise ValueError(f"pessimism_beta must be >= 0, got {pessimism_beta}")
        if ips_clip < 1.0:
            raise ValueError(f"ips_clip must be >= 1.0, got {ips_clip}")

        self.actions: tuple[float, ...] = tuple(float(a) for a in actions)
        self.ridge_lambda = float(ridge_lambda)
        self.pessimism_beta = float(pessimism_beta)
        self.ips_clip = float(ips_clip)
        self.model_version = str(model_version)

        dim = self.feature_dim
        k = len(self.actions)
        self.feature_mean: np.ndarray = np.zeros(NUM_RAW_FEATURES, dtype=np.float64)
        self.feature_std: np.ndarray = np.ones(NUM_RAW_FEATURES, dtype=np.float64)
        self.theta: np.ndarray = np.zeros((k, dim), dtype=np.float64)
        self.a_inv: np.ndarray = np.broadcast_to(
            np.eye(dim, dtype=np.float64) / self.ridge_lambda, (k, dim, dim)
        ).copy()
        self.median_uncertainty: float = 1.0
        self.is_fitted: bool = False

    @property
    def num_actions(self) -> int:
        """Number of discrete memory-share multiplier arms."""
        return len(self.actions)

    @property
    def feature_dim(self) -> int:
        """Augmented feature dimension (bias + 5 standardized shard features)."""
        return 1 + NUM_RAW_FEATURES

    def _standardize(self, raw_features: np.ndarray) -> np.ndarray:
        """Standardizes raw 5D features and prepends a constant bias term."""
        arr = np.asarray(raw_features, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[np.newaxis, :]
        if arr.shape[1] != NUM_RAW_FEATURES:
            raise ValueError(
                f"expected {NUM_RAW_FEATURES} raw features, got {arr.shape[1]}"
            )
        normed = (arr - self.feature_mean) / self.feature_std
        ones = np.ones((normed.shape[0], 1), dtype=np.float64)
        out = np.hstack([ones, normed])
        return out[0] if single else out

    def fit_offline(
        self,
        raw_contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> OfflineContextualBandit:
        """Trains per-arm Ridge LinUCB / Doubly-Robust parameters offline.

        Args:
            raw_contexts: Array of shape ``(N, 5)`` containing raw context vectors.
            actions: Integer array of shape ``(N,)`` with logged arm indices in
                ``0..num_actions``.
            rewards: Float array of shape ``(N,)`` with observed scalar rewards.
            propensities: Optional float array of shape ``(N,)`` giving behavior
                policy action probabilities ``p(a_i | x_i)`` for Doubly-Robust
                off-policy debiasing.

        Returns:
            ``self`` for method chaining.
        """
        x_raw = np.asarray(raw_contexts, dtype=np.float64)
        act = np.asarray(actions, dtype=np.int64)
        rew = np.asarray(rewards, dtype=np.float64)

        if x_raw.ndim != 2 or x_raw.shape[1] != NUM_RAW_FEATURES:
            raise ValueError(
                f"raw_contexts must have shape (N, {NUM_RAW_FEATURES}), got {x_raw.shape}"
            )
        n = x_raw.shape[0]
        if act.shape != (n,) or rew.shape != (n,):
            raise ValueError("actions and rewards must match sample count N")
        if n < self.num_actions:
            raise ValueError("insufficient offline samples to fit bandit")
        if np.any((act < 0) | (act >= self.num_actions)):
            raise ValueError("action indices out of valid range")

        self.feature_mean = np.mean(x_raw, axis=0)
        std = np.std(x_raw, axis=0)
        self.feature_std = np.where(std > 1e-6, std, 1.0)

        x = self._standardize(x_raw)
        dim = self.feature_dim
        eye = np.eye(dim, dtype=np.float64)

        # Stage 1: Direct Method (DM) Ridge regression + covariance inverse per arm.
        dm_theta = np.zeros((self.num_actions, dim), dtype=np.float64)
        a_inv_all = np.zeros((self.num_actions, dim, dim), dtype=np.float64)
        for a in range(self.num_actions):
            mask = act == a
            x_a = x[mask]
            r_a = rew[mask]
            cov = self.ridge_lambda * eye + (x_a.T @ x_a)
            a_inv = np.linalg.inv(cov)
            a_inv_all[a] = a_inv
            if x_a.shape[0] > 0:
                dm_theta[a] = a_inv @ (x_a.T @ r_a)

        # Stage 2: Doubly-Robust (DR) refinement when logging propensities exist.
        if propensities is not None:
            props = np.asarray(propensities, dtype=np.float64)
            if props.shape != (n,):
                raise ValueError("propensities must have shape (N,)")
            inv_props = np.clip(1.0 / np.maximum(props, 1e-3), 0.0, self.ips_clip)
            global_cov = self.ridge_lambda * eye + (x.T @ x)
            global_inv = np.linalg.inv(global_cov)
            dr_theta = np.zeros_like(dm_theta)
            for a in range(self.num_actions):
                pred_dm = x @ dm_theta[a]
                indicator = (act == a).astype(np.float64)
                dr_targets = pred_dm + indicator * inv_props * (rew - pred_dm)
                dr_theta[a] = global_inv @ (x.T @ dr_targets)
            self.theta = dr_theta
        else:
            self.theta = dm_theta

        self.a_inv = a_inv_all

        # Calibrate baseline in-distribution epistemic uncertainty.
        sample_unc = []
        for i in range(min(n, 512)):
            z = x[i]
            a_i = int(act[i])
            unc = math.sqrt(max(float(z @ self.a_inv[a_i] @ z), 0.0))
            sample_unc.append(unc)
        self.median_uncertainty = max(float(np.median(sample_unc)), 1e-6)
        self.is_fitted = True
        return self

    def select_action(
        self, raw_features: np.ndarray
    ) -> tuple[int, float, float, float]:
        """Selects the optimal discrete memory multiplier arm for one shard.

        Uses conservative offline lower-confidence scoring
        ``score(a) = z^T theta_a - pessimism_beta * sqrt(z^T A_a^{-1} z)``
        so poorly-supported actions are never over-optimistically selected.

        Returns:
            ``(action_idx, multiplier, expected_reward, uncertainty)``
        """
        z = self._standardize(raw_features)
        best_idx = 0
        best_score = -math.inf
        best_reward = 0.0
        best_unc = 1.0

        for a in range(self.num_actions):
            mean_reward = float(z @ self.theta[a])
            unc = math.sqrt(max(float(z @ self.a_inv[a] @ z), 0.0))
            score = mean_reward - self.pessimism_beta * unc
            if score > best_score:
                best_score = score
                best_idx = a
                best_reward = mean_reward
                best_unc = unc

        return best_idx, self.actions[best_idx], best_reward, best_unc

    def allocate(
        self,
        shards: Sequence[ShardContext],
        total_memory_budget_bytes: int,
        target_epsilon: float = 1.0 / 256.0,
        payload_budget_c: int = 20,
        m_max: int = DEFAULT_M_MAX,
    ) -> tuple[list[ShardAllocationPlanItem], float]:
        """Produces a budget-conserving multi-tenant memory allocation plan.

        Guarantees:
        - ``sum(item.allocated_memory_bytes) <= total_memory_budget_bytes``.
        - ``MIN_QUOTIENT_BITS <= item.recommended_q <= MAX_QUOTIENT_BITS``.
        - ``item.recommended_r >= r_min(target_epsilon)`` and
          ``item.recommended_r + item.recommended_m <= payload_budget_c``.

        Returns:
            ``(allocations, calibrated_confidence)``
        """
        if total_memory_budget_bytes < 0:
            raise ValueError("total_memory_budget_bytes must be >= 0")

        r_min = min_remainder_bits(target_epsilon)
        effective_c = max(int(payload_budget_c), r_min)
        max_valid_m = min(int(m_max), effective_c - r_min)
        bytes_per_slot = max((effective_c + 4.0) / 8.0, 1.0)

        if not shards:
            return [], 0.95

        demands: list[float] = []
        uncertainties: list[float] = []
        for shard in shards:
            raw = extract_raw_features(shard)
            _, mult, _, unc = self.select_action(raw)
            base_demand = max(float(shard.key_count), 1.0)
            demands.append(base_demand * mult)
            uncertainties.append(unc)

        total_demand = max(sum(demands), 1e-9)
        remaining = int(total_memory_budget_bytes)
        allocations: list[ShardAllocationPlanItem] = []

        for shard, demand in zip(shards, demands):
            share = int(math.floor((demand / total_demand) * float(total_memory_budget_bytes)))
            allocated = max(0, min(share, remaining))
            remaining -= allocated

            max_slots = max(float(allocated) / bytes_per_slot, float(1 << MIN_QUOTIENT_BITS))
            rec_q = int(
                np.clip(
                    int(math.floor(math.log2(max_slots))),
                    MIN_QUOTIENT_BITS,
                    MAX_QUOTIENT_BITS,
                )
            )
            span = max(
                int(math.ceil(shard.p95_query_len))
                if math.isfinite(shard.p95_query_len) and shard.p95_query_len >= 1.0
                else 1,
                1,
            )
            desired_m = 0 if span <= 1 else int(math.ceil(math.log2(span)))
            rec_m = min(desired_m, max_valid_m)
            rec_r = effective_c - rec_m

            allocations.append(
                ShardAllocationPlanItem(
                    shard_id=int(shard.shard_id),
                    allocated_memory_bytes=allocated,
                    recommended_q=rec_q,
                    recommended_r=rec_r,
                    recommended_m=rec_m,
                )
            )

        mean_unc = float(np.mean(uncertainties))
        unc_ratio = mean_unc / max(self.median_uncertainty, 1e-6)
        confidence = float(np.clip(1.0 / (1.0 + 0.12 * unc_ratio), 0.10, 0.99))
        return allocations, confidence

    def to_dict(self) -> dict[str, Any]:
        """Serializes bandit parameters and covariance matrices to JSON-safe dict."""
        return {
            "model_version": self.model_version,
            "actions": list(self.actions),
            "ridge_lambda": self.ridge_lambda,
            "pessimism_beta": self.pessimism_beta,
            "ips_clip": self.ips_clip,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "theta": self.theta.tolist(),
            "a_inv": self.a_inv.tolist(),
            "median_uncertainty": self.median_uncertainty,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OfflineContextualBandit:
        """Restores an :class:`OfflineContextualBandit` from a dictionary."""
        model = cls(
            actions=payload["actions"],
            ridge_lambda=float(payload["ridge_lambda"]),
            pessimism_beta=float(payload["pessimism_beta"]),
            ips_clip=float(payload["ips_clip"]),
            model_version=str(payload.get("model_version", "offline-contextual-bandit-v1")),
        )
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        model.feature_std = np.asarray(payload["feature_std"], dtype=np.float64)
        model.theta = np.asarray(payload["theta"], dtype=np.float64)
        model.a_inv = np.asarray(payload["a_inv"], dtype=np.float64)
        model.median_uncertainty = float(payload["median_uncertainty"])
        model.is_fitted = bool(payload.get("is_fitted", True))

        if model.theta.shape != (model.num_actions, model.feature_dim):
            raise ValueError(f"unexpected theta shape {model.theta.shape}")
        if model.a_inv.shape != (model.num_actions, model.feature_dim, model.feature_dim):
            raise ValueError(f"unexpected a_inv shape {model.a_inv.shape}")
        return model

    def save(self, path: str | Path) -> None:
        """Persists trained bandit model weights as UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> OfflineContextualBandit:
        """Loads a previously saved bandit model from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
