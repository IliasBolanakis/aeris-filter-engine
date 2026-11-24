"""Tests for the deterministic offline training pipeline and artifact CLI."""

from __future__ import annotations

from pathlib import Path

from aeris_sidecar.trainer import TrainedBundle, main as trainer_main, train_models


def test_deterministic_offline_training_pipeline_and_metrics() -> None:
    b1 = train_models(seed=42, model_version="aeris-ml-test-v1")
    b2 = train_models(seed=42, model_version="aeris-ml-test-v1")

    assert b1.to_dict() == b2.to_dict()
    assert b1.metrics.ingest_val_rmse < b1.metrics.ingest_persistence_rmse * 0.5
    assert b1.metrics.bandit_reward_gain > 0.20
    assert b1.forecaster.is_fitted
    assert b1.bandit.is_fitted


def test_trainer_cli_persists_and_reloads_versioned_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "models" / "sidecar_v1.json"
    exit_code = trainer_main(
        [
            "--seed",
            "1337",
            "--output",
            str(artifact_path),
            "--model-version",
            "aeris-cli-v1",
        ]
    )
    assert exit_code == 0
    assert artifact_path.exists()

    reloaded = TrainedBundle.load(artifact_path)
    assert reloaded.model_version == "aeris-cli-v1"
    assert reloaded.seed == 1337
    assert reloaded.metrics.bandit_reward_gain > 0.15
    assert reloaded.forecaster.is_fitted
    assert reloaded.bandit.is_fitted
