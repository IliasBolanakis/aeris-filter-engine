"""Deterministic offline trainer for Phase 5 ML/RL sidecar models.

Generates seeded synthetic seasonal workload traces and off-policy multi-tenant
shard trajectories using ``numpy.random.default_rng(seed)``, trains both
:class:`SeasonalWorkloadForecaster` and :class:`OfflineContextualBandit`
offline, evaluates held-out RMSE and bandit reward gain over a uniform
allocation baseline, and persists a versioned JSON model artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bandit import DEFAULT_ACTIONS, OfflineContextualBandit
from .forecaster import SeasonalWorkloadForecaster

logger = logging.getLogger("aeris_sidecar.trainer")

DEFAULT_ARTIFACT_VERSION = "aeris-sidecar-ml-v1"


@dataclass(frozen=True)
class TrainingMetrics:
    """Held-out evaluation metrics produced by offline model training."""

    ingest_val_rmse: float
    ingest_persistence_rmse: float
    query_len_val_rmse: float
    bandit_eval_reward: float
    uniform_baseline_reward: float
    bandit_reward_gain: float

    def to_dict(self) -> dict[str, float]:
        """Serializes metrics into a JSON-compatible dictionary."""
        return {
            "ingest_val_rmse": self.ingest_val_rmse,
            "ingest_persistence_rmse": self.ingest_persistence_rmse,
            "query_len_val_rmse": self.query_len_val_rmse,
            "bandit_eval_reward": self.bandit_eval_reward,
            "uniform_baseline_reward": self.uniform_baseline_reward,
            "bandit_reward_gain": self.bandit_reward_gain,
        }


@dataclass(frozen=True)
class TrainedBundle:
    """Combined versioned model artifact containing both offline-trained models."""

    model_version: str
    seed: int
    forecaster: SeasonalWorkloadForecaster
    bandit: OfflineContextualBandit
    metrics: TrainingMetrics

    def to_dict(self) -> dict[str, Any]:
        """Serializes the complete bundle into a JSON-compatible dictionary."""
        return {
            "model_version": self.model_version,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "forecaster": self.forecaster.to_dict(),
            "bandit": self.bandit.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        """Persists the versioned artifact JSON to disk atomically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> TrainedBundle:
        """Loads a previously saved versioned model bundle from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics_raw = payload["metrics"]
        metrics = TrainingMetrics(
            ingest_val_rmse=float(metrics_raw["ingest_val_rmse"]),
            ingest_persistence_rmse=float(metrics_raw["ingest_persistence_rmse"]),
            query_len_val_rmse=float(metrics_raw["query_len_val_rmse"]),
            bandit_eval_reward=float(metrics_raw["bandit_eval_reward"]),
            uniform_baseline_reward=float(metrics_raw["uniform_baseline_reward"]),
            bandit_reward_gain=float(metrics_raw["bandit_reward_gain"]),
        )
        return cls(
            model_version=str(payload["model_version"]),
            seed=int(payload["seed"]),
            forecaster=SeasonalWorkloadForecaster.from_dict(payload["forecaster"]),
            bandit=OfflineContextualBandit.from_dict(payload["bandit"]),
            metrics=metrics,
        )


def generate_seasonal_workload(
    rng: np.random.Generator,
    num_steps: int = 240,
    period: float = 24.0,
    noise_scale: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates seeded synthetic diurnal ingestion and query-span time-series."""
    t = np.arange(num_steps, dtype=np.float64)
    angle = (2.0 * math.pi / period) * t
    damped_trend = (1.0 - np.power(0.92, t)) / (1.0 - 0.92)

    ingest_signal = (
        12_000.0
        + 120.0 * damped_trend
        + 5_500.0 * np.sin(angle)
        + 2_100.0 * np.cos(2.0 * angle)
        + 900.0 * np.sin(3.0 * angle)
    )
    ingest_noise = rng.normal(0.0, noise_scale * np.std(ingest_signal), size=num_steps)
    ingest_series = np.maximum(ingest_signal + ingest_noise, 50.0)

    log_qlen_signal = (
        4.2
        + 0.03 * damped_trend
        + 1.1 * np.sin(angle - 0.4)
        + 0.45 * np.cos(2.0 * angle)
    )
    qlen_noise = rng.normal(0.0, noise_scale * np.std(log_qlen_signal), size=num_steps)
    query_len_series = np.maximum(np.expm1(log_qlen_signal + qlen_noise), 1.0)
    return ingest_series, query_len_series


def ideal_multiplier_from_context(raw_context: np.ndarray) -> float:
    """Ground-truth target memory share multiplier for synthetic bandit logs.

    Raw context layout:
        ``[log1p(key_count), log1p(query_rate), observed_fpr, log1p(p95_query_len), load_factor]``
    """
    log_qrate = float(raw_context[1])
    fpr = float(raw_context[2])
    alpha = float(raw_context[4])

    heat = np.clip((log_qrate - 2.0) / 8.0, 0.0, 1.0)
    fpr_pressure = np.clip(fpr / 0.04, 0.0, 1.0)
    alpha_pressure = np.clip((alpha - 0.4) / 0.5, 0.0, 1.0)

    score = 0.45 * heat + 0.40 * fpr_pressure + 0.15 * alpha_pressure
    return float(0.5 + 2.5 * score)


def compute_bandit_reward(raw_context: np.ndarray, action_multiplier: float) -> float:
    """Deterministic expected reward for choosing `action_multiplier` on `raw_context`."""
    ideal = ideal_multiplier_from_context(raw_context)
    # Asymmetric penalty: under-allocating hot/high-FPR shards hurts more than slight excess.
    diff = action_multiplier - ideal
    penalty_weight = 1.15 if diff < 0.0 else 0.75
    return float(1.0 - penalty_weight * (diff * diff) / (1.0 + 0.35 * ideal))


def generate_offline_bandit_dataset(
    rng: np.random.Generator,
    num_samples: int = 2400,
    actions: tuple[float, ...] = DEFAULT_ACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a seeded off-policy multi-tenant shard bandit dataset."""
    num_actions = len(actions)
    raw_contexts = np.empty((num_samples, 5), dtype=np.float64)
    logged_actions = np.empty(num_samples, dtype=np.int64)
    rewards = np.empty(num_samples, dtype=np.float64)
    propensities = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        # Mix cold, warm, and hot/high-FPR shard regimes.
        regime = int(rng.integers(0, 3))
        if regime == 0:  # Cold / idle shard
            key_count = float(rng.uniform(5_000, 120_000))
            query_rate = float(rng.uniform(0.0, 25.0))
            fpr = float(rng.uniform(0.0, 0.002))
            p95 = float(rng.uniform(1.0, 32.0))
            alpha = float(rng.uniform(0.15, 0.50))
        elif regime == 1:  # Moderate shard
            key_count = float(rng.uniform(50_000, 500_000))
            query_rate = float(rng.uniform(100.0, 2_500.0))
            fpr = float(rng.uniform(0.002, 0.015))
            p95 = float(rng.uniform(16.0, 256.0))
            alpha = float(rng.uniform(0.45, 0.75))
        else:  # Hot / high-FPR shard
            key_count = float(rng.uniform(80_000, 800_000))
            query_rate = float(rng.uniform(5_000.0, 60_000.0))
            fpr = float(rng.uniform(0.02, 0.08))
            p95 = float(rng.uniform(64.0, 2048.0))
            alpha = float(rng.uniform(0.70, 0.92))

        ctx = np.array(
            [
                math.log1p(key_count),
                math.log1p(query_rate),
                fpr,
                math.log1p(p95),
                alpha,
            ],
            dtype=np.float64,
        )
        raw_contexts[i] = ctx

        # Biased stochastic logging policy (epsilon-greedy around noisy heuristic).
        ideal = ideal_multiplier_from_context(ctx)
        heuristic_arm = int(np.argmin([abs(a - ideal) for a in actions]))
        probs = np.full(num_actions, 0.45 / num_actions, dtype=np.float64)
        probs[heuristic_arm] += 0.55
        chosen = int(rng.choice(num_actions, p=probs))
        logged_actions[i] = chosen
        propensities[i] = float(probs[chosen])

        true_reward = compute_bandit_reward(ctx, actions[chosen])
        rewards[i] = true_reward + float(rng.normal(0.0, 0.05))

    return raw_contexts, logged_actions, rewards, propensities


def train_models(
    seed: int = 42,
    num_series_steps: int = 240,
    num_bandit_samples: int = 2400,
    model_version: str = DEFAULT_ARTIFACT_VERSION,
) -> TrainedBundle:
    """Runs the complete deterministic offline training and validation pipeline."""
    rng = np.random.default_rng(seed)

    # 1. Train & validate SeasonalWorkloadForecaster
    ingest_all, qlen_all = generate_seasonal_workload(
        rng, num_steps=num_series_steps, period=24.0
    )
    split = int(num_series_steps * 0.75)
    train_ingest, val_ingest = ingest_all[:split], ingest_all[split:]
    train_qlen, val_qlen = qlen_all[:split], qlen_all[split:]

    forecaster = SeasonalWorkloadForecaster(
        period=24.0,
        num_harmonics=3,
        damping=0.92,
        ridge_lambda=1e-2,
        prior_strength=0.25,
        model_version=f"{model_version}:forecaster",
    )
    forecaster.fit(train_ingest, train_qlen)

    horizon = len(val_ingest)
    forecast = forecaster.predict(train_ingest, train_qlen, horizon=horizon)
    ingest_preds = np.asarray(forecast.ingest_trajectory, dtype=np.float64)
    qlen_preds = np.asarray(forecast.query_len_trajectory, dtype=np.float64)

    ingest_val_rmse = float(np.sqrt(np.mean((val_ingest - ingest_preds) ** 2)))
    ingest_persistence_rmse = float(
        np.sqrt(np.mean((val_ingest - train_ingest[-1]) ** 2))
    )
    query_len_val_rmse = float(
        np.sqrt(np.mean((np.log1p(val_qlen) - np.log1p(qlen_preds)) ** 2))
    )

    # 2. Train & evaluate OfflineContextualBandit
    train_ctx, train_act, train_rew, train_prop = generate_offline_bandit_dataset(
        rng, num_samples=num_bandit_samples
    )
    eval_ctx, _, _, _ = generate_offline_bandit_dataset(rng, num_samples=600)

    bandit = OfflineContextualBandit(
        actions=DEFAULT_ACTIONS,
        ridge_lambda=1.0,
        pessimism_beta=0.10,
        model_version=f"{model_version}:bandit",
    )
    bandit.fit_offline(train_ctx, train_act, train_rew, propensities=train_prop)

    bandit_rewards = []
    uniform_rewards = []
    for ctx in eval_ctx:
        _, chosen_mult, _, _ = bandit.select_action(ctx)
        bandit_rewards.append(compute_bandit_reward(ctx, chosen_mult))
        uniform_rewards.append(compute_bandit_reward(ctx, 1.0))

    bandit_eval_reward = float(np.mean(bandit_rewards))
    uniform_baseline_reward = float(np.mean(uniform_rewards))
    bandit_reward_gain = bandit_eval_reward - uniform_baseline_reward

    metrics = TrainingMetrics(
        ingest_val_rmse=ingest_val_rmse,
        ingest_persistence_rmse=ingest_persistence_rmse,
        query_len_val_rmse=query_len_val_rmse,
        bandit_eval_reward=bandit_eval_reward,
        uniform_baseline_reward=uniform_baseline_reward,
        bandit_reward_gain=bandit_reward_gain,
    )

    logger.info(
        "offline training complete seed=%d ingest_rmse=%.2f baseline_rmse=%.2f "
        "bandit_reward=%.4f uniform_reward=%.4f gain=%.4f",
        seed,
        ingest_val_rmse,
        ingest_persistence_rmse,
        bandit_eval_reward,
        uniform_baseline_reward,
        bandit_reward_gain,
    )

    return TrainedBundle(
        model_version=model_version,
        seed=seed,
        forecaster=forecaster,
        bandit=bandit,
        metrics=metrics,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for offline Phase 5 model training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train Aeris Phase 5 offline seasonal forecaster and contextual bandit."
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aeris_sidecar_model.json"),
        help="Destination path for versioned JSON artifact.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=DEFAULT_ARTIFACT_VERSION,
        help="Model version identifier embedded in the artifact.",
    )
    args = parser.parse_args(argv)

    bundle = train_models(seed=args.seed, model_version=args.model_version)
    bundle.save(args.output)
    logger.info("saved versioned model artifact to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
