"""Unit and convergence tests for OfflineContextualBandit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aeris_sidecar.bandit import (
    DEFAULT_ACTIONS,
    MAX_QUOTIENT_BITS,
    MIN_QUOTIENT_BITS,
    OfflineContextualBandit,
    ShardContext,
    min_remainder_bits,
)
from aeris_sidecar.trainer import compute_bandit_reward, generate_offline_bandit_dataset


def test_offline_training_converges_and_beats_uniform_baseline() -> None:
    rng = np.random.default_rng(101)
    train_ctx, train_act, train_rew, train_prop = generate_offline_bandit_dataset(
        rng, num_samples=2500
    )
    eval_ctx, _, _, _ = generate_offline_bandit_dataset(rng, num_samples=600)

    bandit = OfflineContextualBandit(actions=DEFAULT_ACTIONS, ridge_lambda=1.0, pessimism_beta=0.1)
    bandit.fit_offline(train_ctx, train_act, train_rew, propensities=train_prop)

    bandit_scores = []
    uniform_scores = []
    for ctx in eval_ctx:
        _, mult, _, _ = bandit.select_action(ctx)
        bandit_scores.append(compute_bandit_reward(ctx, mult))
        uniform_scores.append(compute_bandit_reward(ctx, 1.0))

    mean_bandit = float(np.mean(bandit_scores))
    mean_uniform = float(np.mean(uniform_scores))
    assert mean_bandit > mean_uniform + 0.20


def test_budget_conservation_and_hot_vs_idle_shard_prioritization() -> None:
    rng = np.random.default_rng(42)
    train_ctx, train_act, train_rew, train_prop = generate_offline_bandit_dataset(
        rng, num_samples=2500
    )
    bandit = OfflineContextualBandit(actions=DEFAULT_ACTIONS)
    bandit.fit_offline(train_ctx, train_act, train_rew, propensities=train_prop)

    # Identical key counts, vastly different heat & FPR pressure.
    hot_shard = ShardContext(
        shard_id=1,
        key_count=200_000,
        current_memory_bytes=2 * 1024 * 1024,
        query_rate=40_000.0,
        observed_fpr=0.06,
        p95_query_len=512.0,
        load_factor=0.88,
        current_q=16,
        current_r=14,
        current_m=6,
    )
    warm_shard = ShardContext(
        shard_id=2,
        key_count=200_000,
        current_memory_bytes=2 * 1024 * 1024,
        query_rate=800.0,
        observed_fpr=0.008,
        p95_query_len=64.0,
        load_factor=0.60,
        current_q=16,
        current_r=14,
        current_m=6,
    )
    idle_shard = ShardContext(
        shard_id=3,
        key_count=200_000,
        current_memory_bytes=2 * 1024 * 1024,
        query_rate=2.0,
        observed_fpr=0.0002,
        p95_query_len=4.0,
        load_factor=0.20,
        current_q=16,
        current_r=14,
        current_m=6,
    )

    eps = 1.0 / 256.0
    r_min = min_remainder_bits(eps)
    budget_c = 20

    for total_budget in (0, 513, 1024 * 1024, 16 * 1024 * 1024 + 7):
        allocs, confidence = bandit.allocate(
            shards=[hot_shard, warm_shard, idle_shard],
            total_memory_budget_bytes=total_budget,
            target_epsilon=eps,
            payload_budget_c=budget_c,
        )
        assert len(allocs) == 3
        total_allocated = sum(a.allocated_memory_bytes for a in allocs)
        assert total_allocated <= total_budget
        assert 0.0 <= confidence <= 1.0

        for item in allocs:
            assert MIN_QUOTIENT_BITS <= item.recommended_q <= MAX_QUOTIENT_BITS
            assert item.recommended_r >= r_min
            assert item.recommended_r + item.recommended_m <= budget_c

        if total_budget >= 1024 * 1024:
            by_id = {a.shard_id: a for a in allocs}
            assert by_id[1].allocated_memory_bytes > by_id[2].allocated_memory_bytes
            assert by_id[2].allocated_memory_bytes > by_id[3].allocated_memory_bytes
            assert by_id[1].allocated_memory_bytes >= 2 * by_id[3].allocated_memory_bytes
            assert confidence >= 0.80


def test_empty_shards_and_ood_uncertainty_penalty() -> None:
    rng = np.random.default_rng(19)
    train_ctx, train_act, train_rew, train_prop = generate_offline_bandit_dataset(
        rng, num_samples=1500
    )
    bandit = OfflineContextualBandit()
    bandit.fit_offline(train_ctx, train_act, train_rew, propensities=train_prop)

    empty_allocs, empty_conf = bandit.allocate([], total_memory_budget_bytes=4096)
    assert empty_allocs == []
    assert empty_conf >= 0.80

    in_dist_shard = ShardContext(
        shard_id=10,
        key_count=150_000,
        current_memory_bytes=1_000_000,
        query_rate=1_000.0,
        observed_fpr=0.01,
        p95_query_len=64.0,
        load_factor=0.65,
        current_q=15,
        current_r=14,
        current_m=6,
    )
    ood_shard = ShardContext(
        shard_id=11,
        key_count=10**18,
        current_memory_bytes=1_000_000,
        query_rate=1e15,
        observed_fpr=0.99,
        p95_query_len=1e12,
        load_factor=1.0,
        current_q=15,
        current_r=14,
        current_m=6,
    )
    _, in_conf = bandit.allocate([in_dist_shard], total_memory_budget_bytes=8_000_000)
    _, ood_conf = bandit.allocate([ood_shard], total_memory_budget_bytes=8_000_000)
    assert ood_conf < in_conf


def test_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    train_ctx, train_act, train_rew, train_prop = generate_offline_bandit_dataset(
        rng, num_samples=1200
    )
    bandit = OfflineContextualBandit(model_version="bandit-test-v3")
    bandit.fit_offline(train_ctx, train_act, train_rew, propensities=train_prop)

    path = tmp_path / "bandit.json"
    bandit.save(path)
    restored = OfflineContextualBandit.load(path)

    assert restored.model_version == "bandit-test-v3"
    assert np.allclose(restored.theta, bandit.theta)
    assert np.allclose(restored.a_inv, bandit.a_inv)

    probe = train_ctx[0]
    assert restored.select_action(probe) == bandit.select_action(probe)


def test_invalid_bandit_configs_raise() -> None:
    with pytest.raises(ValueError):
        OfflineContextualBandit(actions=(1.0,))
    with pytest.raises(ValueError):
        OfflineContextualBandit(actions=(-1.0, 2.0))
    with pytest.raises(ValueError):
        OfflineContextualBandit(ridge_lambda=0.0)
    with pytest.raises(ValueError):
        min_remainder_bits(0.5)
    with pytest.raises(ValueError):
        min_remainder_bits(0.0)
