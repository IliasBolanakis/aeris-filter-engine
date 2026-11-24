//! Milestone M3 Exit Gate Integration & Fault-Tolerance Test Suite.
//!
//! All tests run exclusively against REAL Data Plane (`AerisFilter`), REAL Storage
//! Plane (`RocksDbReverseMap` + `ReferenceKvStore`), and REAL Control Plane
//! (`SlidingWindowEstimator`, `DiscreteCostOptimizer`, `ProactiveTrigger`,
//! `FalsePositiveLog`, `AsyncRebuildCoordinator`, `ControlledAerisEngine`) with
//! deterministic seeded `ChaCha8Rng` distributions.

use aeris_control::{
    AsyncRebuildCoordinator, ControlledAerisEngine, ControlledEngineConfig, EstimatorConfig,
    FalsePositiveLog, FpLogConfig, IoThrottleConfig, ProactiveTriggerConfig, RebuildReason,
    TriggerAction,
};
use aeris_core::AerisFilter;
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, CostModelParams, FilterConfig, Key, QueryRange, RangeFilter, ReverseMapStore,
    WorkloadEstimator, MAX_LOAD_FACTOR,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

/// Helper that generates `count` unique keys spaced so natural range gaps exist.
fn generate_seeded_keys(seed: u64, count: usize, stride: u64) -> Vec<Key> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut keys = BTreeSet::new();
    while keys.len() < count {
        let bucket = rng.gen_range(0..(count as u64 * 4));
        let offset = rng.gen_range(0..stride.clamp(1, 64));
        keys.insert(bucket * stride + offset);
    }
    keys.into_iter().collect()
}

/// Gate 1: 100% monotonicity across dynamic `R` changes (`m = 5 -> m = 8`) with
/// concurrent `DeltaBuffer` inserts AND deletes on `F_old`.
#[test]
fn gate_1_monotonicity_across_dynamic_r_rebuild_with_concurrent_delta_mutations() {
    let dir = TempDir::new().unwrap();
    // eps = 2^-4 (0.0625) -> r_min = 7; C = 15 supports m in 0..=8 under fixed RAM budget.
    // Start at m = 5 (R = 32, r = 10, q = 9 -> 512 slots) so FPs arise naturally on both F_old and F_new.
    let initial_cfg = FilterConfig::new(9, 10, 32, 15, 1.0 / 16.0).unwrap();
    let mut engine_cfg = ControlledEngineConfig::new(initial_cfg.clone());
    // Light throttling ensures the background thread stays actively running while
    // foreground DeltaBuffer inserts/deletes execute concurrently.
    engine_cfg.throttle_config = IoThrottleConfig::limited(6_000, 80);

    let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();

    // 1. Populate 380 seeded keys in the low keyspace [0, 5_000_000).
    let initial_keys = generate_seeded_keys(0x0003_0001, 380, 2_000);
    let mut live_truth: BTreeSet<Key> = BTreeSet::new();
    for &k in &initial_keys {
        engine.insert(k, &k.to_le_bytes()).unwrap();
        live_truth.insert(k);
    }

    // 2. Discover genuine empty false positives on F_old (both R = 32 and incoming R = 256
    // regime-shift queries, adapted and persisted into FalsePositiveLog).
    let mut rng = ChaCha8Rng::seed_from_u64(0x0003_0002);
    let mut prior_fps: Vec<QueryRange> = Vec::new();
    let mut prior_fp_set: BTreeSet<QueryRange> = BTreeSet::new();

    for probe_idx in 0..250_000usize {
        let start = rng.gen_range(10_000_000..1_000_000_000);
        let span = if probe_idx < 25_000 { 32 } else { 256 };
        let range = QueryRange::with_len(start, span).unwrap();
        if live_truth
            .range(range.start()..=range.end())
            .next()
            .is_some()
        {
            continue;
        }
        let outcome = engine.query_range(range).unwrap();
        assert!(
            !outcome.non_empty,
            "empty probe must report non_empty=false"
        );
        if outcome.was_false_positive {
            assert!(outcome.adapted);
            if prior_fp_set.insert(range) {
                prior_fps.push(range);
            }
            // Monotonic immediately on F_old.
            assert!(
                !engine.filter().query_range(range),
                "adapted range must immediately return false on F_old"
            );
            if prior_fps.len() >= 600 {
                break;
            }
        }
    }
    assert!(
        prior_fps.len() >= 400,
        "expected at least 400 natural FPs on F_old, found {}",
        prior_fps.len()
    );
    assert_eq!(engine.fp_log().len(), prior_fps.len());

    // 3. Launch async rebuild to m = 8 (R = 256, r = 7, C = 15).
    let target_cfg = initial_cfg.reconfigure_for_m(8).unwrap();
    engine
        .trigger_async_rebuild(
            target_cfg.clone(),
            RebuildReason::DynamicRangeChange { old_m: 5, new_m: 8 },
        )
        .unwrap();

    // Overlapping rebuild must be rejected with RebuildInProgress.
    let overlap_err = engine.trigger_async_rebuild(
        target_cfg,
        RebuildReason::DynamicRangeChange { old_m: 5, new_m: 8 },
    );
    assert!(matches!(overlap_err, Err(AerisError::RebuildInProgress)));

    // 4. Concurrently apply DeltaBuffer inserts AND deletes while background rebuild runs!
    // Delta inserts live in [5_000_000, 9_000_000) so they remain strictly disjoint from prior_fps.
    let delta_insert_keys: Vec<Key> = (0..80u64).map(|i| 5_000_000 + i * 257).collect();
    for &k in &delta_insert_keys {
        engine.insert(k, &k.to_le_bytes()).unwrap();
        live_truth.insert(k);
    }

    // Delete 60 original keys while rebuild is active.
    let delta_delete_keys: Vec<Key> = initial_keys.iter().copied().take(60).collect();
    for &k in &delta_delete_keys {
        assert!(engine.delete(k).unwrap());
        live_truth.remove(&k);
    }

    // Also probe additional empty ranges during rebuild to exercise incremental FP catch-up.
    for _ in 0..10_000 {
        let start = rng.gen_range(2_000_000_000..3_000_000_000);
        let range = QueryRange::with_len(start, 256).unwrap();
        let outcome = engine.query_range(range).unwrap();
        if outcome.was_false_positive && prior_fp_set.insert(range) {
            prior_fps.push(range);
        }
    }

    // 5. Complete rebuild, drain DeltaBuffer, replay FP Log, and atomically swap F_old -> F_new.
    let report = engine.finish_rebuild().unwrap();
    assert_eq!(engine.epoch(), 2);
    assert_eq!(engine.filter().config().m, 8);
    assert_eq!(engine.filter().config().r, 7);
    assert_eq!(engine.filter().config().max_range_r, 256);
    assert!(report.keys_loaded >= 320);
    assert!(report.fp_replay_initial.replayed_entries >= 400);
    assert!(
        report.fp_replay_initial.collisions_resolved + report.fp_replay_catchup.collisions_resolved
            > 0,
        "expected FP Log replay to actively resolve collisions on F_new"
    );
    assert_eq!(engine.filter().verify_invariants().ok(), Some(()));

    // 6. Assert 100% monotonicity: ZERO repeated FPs on F_new!
    for &fp_range in &prior_fps {
        assert!(
            !engine.filter().query_range(fp_range),
            "monotonicity violation: prior FP range {:?} repeated on F_new!",
            fp_range
        );
    }

    // 7. Assert ZERO false negatives across all surviving original + delta-inserted keys!
    for &k in &live_truth {
        assert!(
            engine.filter().query_point(k),
            "false negative on F_new for live key {k}"
        );
        assert!(
            engine.filter().query_range(QueryRange::point(k)),
            "range false negative on F_new for live key {k}"
        );
    }

    // 8. Assert all delta-deleted keys were cleanly removed from KV store, filter, and Reverse Map!
    assert_eq!(engine.filter().len(), live_truth.len());
    assert_eq!(engine.kv_store().len().unwrap(), live_truth.len());
    let rm_keys: BTreeSet<Key> = engine
        .reverse_map()
        .scan_all_keys()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(rm_keys, live_truth);
    for &deleted in &delta_delete_keys {
        assert!(!rm_keys.contains(&deleted));
        assert_eq!(engine.kv_store().get(deleted).unwrap(), None);
    }
}

/// Gate 2: Zero rebuild thrashing under oscillating query lengths (`R = 16` <-> `R = 512`)
/// across 10,000 queries (tested under both per-query alternation and 100-query burst oscillation).
#[test]
fn gate_2_zero_thrashing_under_oscillating_query_lengths() {
    for (pattern_name, burst_len) in [("alternating", 1u64), ("burst_100", 100u64)] {
        let dir = TempDir::new().unwrap();
        let cfg = FilterConfig::new(10, 14, 64, 20, 1.0 / 256.0).unwrap();
        let mut engine_cfg = ControlledEngineConfig::new(cfg);
        engine_cfg.estimator_config = EstimatorConfig::default();
        engine_cfg.cost_model = CostModelParams::default();

        let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();
        for &k in &generate_seeded_keys(0x0003_0020, 200, 4_000) {
            engine.insert(k, &k.to_le_bytes()).unwrap();
        }

        let mut rng = ChaCha8Rng::seed_from_u64(0x0003_0021);
        let mut rebuilds_triggered = 0u64;

        for q_idx in 0..10_000u64 {
            let phase = (q_idx / burst_len) % 2;
            let span = if phase == 0 { 16 } else { 512 };
            let start = rng.gen_range(10_000_000..500_000_000);
            let range = QueryRange::with_len(start, span).unwrap();
            let _ = engine.query_range(range).unwrap();

            if (q_idx + 1) % 64 == 0 {
                let decision = engine.evaluate_control_plane();
                if decision.optimization.should_rebuild {
                    let next_cfg = engine
                        .filter()
                        .config()
                        .reconfigure_for_m(decision.optimization.recommended_m)
                        .unwrap();
                    let old_m = engine.filter().config().m;
                    let new_m = next_cfg.m;
                    engine
                        .trigger_async_rebuild(
                            next_cfg,
                            RebuildReason::DynamicRangeChange { old_m, new_m },
                        )
                        .unwrap();
                    let _ = engine.finish_rebuild().unwrap();
                    rebuilds_triggered += 1;
                }
            }
        }

        // At most 1 initial convergence rebuild is allowed; zero oscillatory thrashing!
        assert!(
            rebuilds_triggered <= 1,
            "oscillating pattern '{pattern_name}' caused rebuild thrashing: {rebuilds_triggered} rebuilds across 10,000 queries"
        );
    }
}

/// Gate 3: Poisoning resistance against 8% extreme outlier queries (`span = 10_000_000`)
/// injected into `R = 32` baseline traffic.
#[test]
fn gate_3_poisoning_resistance_against_extreme_outlier_queries() {
    let dir = TempDir::new().unwrap();
    let cfg = FilterConfig::new(10, 15, 32, 20, 1.0 / 256.0).unwrap();
    let engine_cfg = ControlledEngineConfig::new(cfg);
    let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();

    for &k in &generate_seeded_keys(0x0003_0030, 300, 3_000) {
        engine.insert(k, &k.to_le_bytes()).unwrap();
    }

    // Compute the clean unpoisoned optimal m* for R = 32 baseline first.
    let clean_baseline = engine.evaluate_control_plane();
    let baseline_optimal_m = clean_baseline.optimization.recommended_m;

    // Verify that an unclipped 10_000_000 span WOULD indeed trip bypass (proving clipping is essential).
    let poisoned_unclipped = aeris_types::WorkloadStats {
        effective_query_span: 10_000_000,
        ..aeris_types::WorkloadStats::default()
    };
    let unclipped_decision = aeris_types::CostOptimizer::optimize(
        &engine.optimizer(),
        &poisoned_unclipped,
        engine.filter().config(),
        &CostModelParams::default(),
    );
    assert!(unclipped_decision.should_bypass_filter);

    let mut rng = ChaCha8Rng::seed_from_u64(0x0003_0031);
    let mut any_rebuild_recommended = false;
    let mut any_bypass_recommended = false;

    for i in 0..4_096u64 {
        let is_poison = i % 25 < 2; // 8% adversarial poison spikes
        let span = if is_poison {
            10_000_000
        } else {
            rng.gen_range(24..=40)
        };
        let start = rng.gen_range(50_000_000..500_000_000);
        let range = QueryRange::with_len(start, span).unwrap();

        // Feed queries into the estimator (recording directly for poison spans to keep test fast
        // while executing real filter probes on all baseline R=32 traffic).
        if is_poison {
            engine.estimator_mut().record_query(range, false);
        } else {
            let _ = engine.query_range(range).unwrap();
        }

        if (i + 1) % 256 == 0 {
            let decision = engine.evaluate_control_plane();
            if decision.optimization.should_rebuild {
                any_rebuild_recommended = true;
            }
            if decision.optimization.should_bypass_filter {
                any_bypass_recommended = true;
            }
        }
    }

    let final_decision = engine.evaluate_control_plane();
    assert!(
        final_decision.workload.clipped_outliers >= 300,
        "expected ~8% (327) outliers clipped, got {}",
        final_decision.workload.clipped_outliers
    );
    assert!(
        (24..=48).contains(&final_decision.workload.effective_query_span),
        "effective_query_span poisoned: {}",
        final_decision.workload.effective_query_span
    );
    assert_eq!(
        final_decision.optimization.recommended_m, baseline_optimal_m,
        "optimizer must keep clean-baseline optimal m={baseline_optimal_m} under poison spikes"
    );
    assert!(
        !any_bypass_recommended,
        "poison spikes must not trigger filter bypass"
    );
    assert!(
        !any_rebuild_recommended,
        "poison spikes must not trigger rebuild thrashing"
    );
    assert!(!engine.bypass_active());
    assert_eq!(engine.total_bypassed_queries(), 0);
}

/// Gate 4: Proactive EWMA trigger fires strictly before saturation (`alpha < 0.95`)
/// AND Contraction (`q -> q - 1`) reclaims memory cleanly after bulk deletes with
/// 0 false negatives and 0 repeated FPs.
#[test]
fn gate_4_proactive_ewma_trigger_and_contraction_after_bulk_deletes() {
    let dir = TempDir::new().unwrap();
    // Start at q = 10 (1024 slots), m = 5, r = 10, C = 15 (eps = 1/16 -> r_min = 7)
    // so genuine FPs arise naturally even after 2x expansion.
    let cfg = FilterConfig::new(10, 10, 32, 15, 1.0 / 16.0).unwrap();
    let mut engine_cfg = ControlledEngineConfig::new(cfg);
    engine_cfg.trigger_config = ProactiveTriggerConfig {
        margin: 1.5,
        hard_expand_alpha: 0.90,
        contraction_alpha_threshold: 0.25,
        contraction_sustained_checks: 4,
        max_post_contraction_alpha: 0.65,
    };

    let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();
    // Calibrate bulk-load speed to a realistic disk-bound rate so Eq. (3) fires proactively.
    engine.estimator_mut().record_bulk_load_speed(4_000.0);

    let all_keys = generate_seeded_keys(0x0003_0040, 800, 1_500);
    let mut live_keys = BTreeSet::new();
    let mut proactive_fired_alpha: Option<f64> = None;

    // 1. Ingest keys in batches while reporting sustained high ingestion rate (2_000 keys/sec).
    for chunk in all_keys.chunks(50) {
        for &k in chunk {
            engine.insert(k, &k.to_le_bytes()).unwrap();
            live_keys.insert(k);
        }
        engine.record_ingest_sample(chunk.len() as u64, 0.025); // 2_000 keys/sec
        let decision = engine.evaluate_control_plane();

        if decision.trigger.action == TriggerAction::ProactiveExpand {
            let alpha = engine.filter().load_factor();
            proactive_fired_alpha = Some(alpha);
            assert!(
                alpha < MAX_LOAD_FACTOR,
                "proactive trigger must fire before saturation (alpha={alpha})"
            );
            assert!(
                alpha < 0.90,
                "Eq. (3) proactive trigger should fire strictly before hard_expand_alpha (0.90), got alpha={alpha}"
            );

            // Execute proactive async expansion q=10 -> q=11.
            let old_q = engine.filter().config().q;
            let expanded_cfg = engine.filter().config().expanded().unwrap();
            let new_q = expanded_cfg.q;
            engine
                .trigger_async_rebuild(
                    expanded_cfg,
                    RebuildReason::ProactiveExpansion { old_q, new_q },
                )
                .unwrap();
            let _ = engine.finish_rebuild().unwrap();
            assert_eq!(engine.filter().config().q, 11);
            break;
        }
    }

    assert!(
        proactive_fired_alpha.is_some(),
        "expected ProactiveExpand to fire during ingestion ramp"
    );

    // 2. Collect genuine FPs on the expanded q=11 filter into FalsePositiveLog.
    let mut rng = ChaCha8Rng::seed_from_u64(0x0003_0041);
    let mut logged_fps = Vec::new();
    for _ in 0..80_000 {
        let start = rng.gen_range(50_000_000..500_000_000);
        let range = QueryRange::with_len(start, 32).unwrap();
        let outcome = engine.query_range(range).unwrap();
        if outcome.was_false_positive {
            logged_fps.push(range);
            if logged_fps.len() >= 10 {
                break;
            }
        }
    }
    assert!(
        !logged_fps.is_empty(),
        "expected at least one natural FP before contraction"
    );

    // 3. Bulk-delete >75% of keys and drop ingestion rate to 0 so filter is underutilized (alpha <= 0.25).
    let keys_to_delete: Vec<Key> = live_keys
        .iter()
        .copied()
        .take((live_keys.len() * 4) / 5)
        .collect();
    for &k in &keys_to_delete {
        assert!(engine.delete(k).unwrap());
        live_keys.remove(&k);
    }
    // Feed idle ingestion samples so T_full is large and contraction is safe.
    for _ in 0..10 {
        engine.record_ingest_sample(0, 1.0);
    }
    assert!(
        engine.filter().load_factor() <= 0.25,
        "expected low load factor after bulk deletes, got {}",
        engine.filter().load_factor()
    );

    // 4. Evaluate sustained low-load checks until Contract fires, then execute contraction!
    let mut contract_seen = false;
    for _ in 0..4 {
        let d = engine.evaluate_control_plane();
        if d.trigger.action == TriggerAction::Contract {
            contract_seen = true;
            break;
        }
    }
    assert!(
        contract_seen,
        "expected TriggerAction::Contract after 4 sustained low-load checks"
    );

    let contract_report = engine.contract_if_needed().unwrap();
    assert!(
        contract_report.is_some(),
        "expected contraction rebuild to execute"
    );
    assert_eq!(engine.filter().config().q, 10);
    assert_eq!(engine.total_contractions(), 1);
    assert_eq!(engine.filter().verify_invariants().ok(), Some(()));

    // 5. Verify 0 false negatives across surviving keys and 0 repeated FPs!
    for &k in &live_keys {
        assert!(
            engine.filter().query_point(k),
            "false negative after contraction for surviving key {k}"
        );
    }
    for &fp_range in &logged_fps {
        assert!(
            !engine.filter().query_range(fp_range),
            "repeated FP after contraction for range {:?}",
            fp_range
        );
    }
}

/// Gate 5: Token-bucket I/O throttling demonstrably limits background rebuild
/// Reverse Map scan throughput according to the configured rate cap.
#[test]
fn gate_5_io_throttling_limits_background_rebuild_scan_throughput() {
    let dir = TempDir::new().unwrap();
    let src_rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("src_rm"), ReverseMapConfig::default()).unwrap(),
    );
    let dst_rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("dst_rm"), ReverseMapConfig::default()).unwrap(),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
    );

    let cfg = FilterConfig::new(12, 15, 32, 20, 1.0 / 256.0).unwrap();
    let keys = generate_seeded_keys(0x0003_0050, 2_500, 1_000);
    let _src_filter = AerisFilter::bulk_load(cfg.clone(), &keys, src_rm.as_ref()).unwrap();

    // Limit scan throughput to 5_000 keys/sec in 250-key batches -> >= 500 ms for 2_500 keys.
    let throttle = IoThrottleConfig::limited(5_000, 250);
    let coordinator = AsyncRebuildCoordinator::new();
    let target_cfg = cfg.reconfigure_for_m(7).unwrap();

    let start = Instant::now();
    coordinator
        .start_rebuild(
            src_rm,
            dst_rm,
            fp_log,
            target_cfg,
            RebuildReason::DynamicRangeChange { old_m: 5, new_m: 7 },
            throttle,
        )
        .unwrap();

    let key_set: BTreeSet<Key> = keys.iter().copied().collect();
    let completed = coordinator
        .finish_and_drain_deltas(|r| Ok(key_set.range(r.start()..=r.end()).next().is_none()))
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(completed.report.keys_loaded, 2_500);
    assert_eq!(completed.report.throttle_metrics.keys_paced, 2_500);
    assert!(
        completed.report.throttle_metrics.throttle_events >= 5,
        "expected multiple throttle events, got {}",
        completed.report.throttle_metrics.throttle_events
    );
    assert!(
        completed.report.throttle_metrics.total_throttled_ns >= 350_000_000,
        "expected >= 350ms cumulative throttle delay, got {} ns",
        completed.report.throttle_metrics.total_throttled_ns
    );
    assert!(
        elapsed.as_millis() >= 420,
        "expected throttled rebuild to take >= 420ms, finished in {:?}",
        elapsed
    );

    let effective_keys_per_sec = 2_500.0 / elapsed.as_secs_f64();
    assert!(
        effective_keys_per_sec <= 6_000.0,
        "effective throughput ({effective_keys_per_sec:.1} keys/s) exceeded 5_000 keys/s limit"
    );

    // Zero false negatives across all 2,500 keys on F_new.
    for &k in &keys {
        assert!(completed.filter.query_point(k));
    }
    assert_eq!(completed.filter.verify_invariants().ok(), Some(()));
}

/// Gate 6: Restart persistence — reopening `ControlledAerisEngine` reconstructs the
/// filter from durable `ReferenceKvStore` and automatically replays persisted `FalsePositiveLog`.
#[test]
fn gate_6_engine_reopen_replays_persisted_fp_log_and_recovers_keys() {
    let dir = TempDir::new().unwrap();
    let cfg = FilterConfig::new(10, 15, 32, 20, 1.0 / 256.0).unwrap();
    let keys = generate_seeded_keys(0x0003_0060, 400, 2_000);
    let mut logged_fps = Vec::new();

    {
        let mut engine =
            ControlledAerisEngine::open(dir.path(), ControlledEngineConfig::new(cfg.clone()))
                .unwrap();
        for &k in &keys {
            engine.insert(k, &k.to_le_bytes()).unwrap();
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x0003_0061);
        for _ in 0..80_000 {
            let start = rng.gen_range(20_000_000..400_000_000);
            let range = QueryRange::with_len(start, 32).unwrap();
            let outcome = engine.query_range(range).unwrap();
            if outcome.was_false_positive {
                logged_fps.push(range);
                if logged_fps.len() >= 8 {
                    break;
                }
            }
        }
        assert!(!logged_fps.is_empty());
        engine.kv_store().flush().unwrap();
        engine.fp_log().sync().unwrap();
    }

    // Reopen fresh engine at same directory.
    let reopened =
        ControlledAerisEngine::open(dir.path(), ControlledEngineConfig::new(cfg)).unwrap();
    assert_eq!(reopened.filter().len(), keys.len());
    assert_eq!(reopened.fp_log().len(), logged_fps.len());

    // Every persisted key is present with zero false negatives.
    for &k in &keys {
        assert!(reopened.filter().query_point(k));
    }
    // Every persisted FP was replayed on open and returns false immediately.
    for &fp_range in &logged_fps {
        assert!(
            !reopened.filter().query_range(fp_range),
            "persisted FP {:?} repeated after reopen",
            fp_range
        );
    }
}

/// Gate 7: Fault/concurrency edge cases — auto-expansion during undersized rebuild target,
/// delta Insert->Delete & Delete->Insert cancellation/reinsertion, delta insert landing
/// inside a previously logged FP range (zero false negatives!), concurrent multi-threaded
/// `FalsePositiveLog` + `DeltaBuffer` writes synchronized via `Barrier`, and non-blocking
/// `complete_rebuild_if_ready` polling.
#[test]
fn gate_7_delta_edge_cases_auto_expand_and_concurrent_fp_log_stress() {
    use std::sync::Barrier;
    use std::thread;

    let dir = TempDir::new().unwrap();
    let cfg = FilterConfig::new(8, 10, 32, 15, 1.0 / 16.0).unwrap(); // q=8 (256 slots)
    let mut engine_cfg = ControlledEngineConfig::new(cfg.clone());
    engine_cfg.throttle_config = IoThrottleConfig::limited(4_000, 50);

    let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();

    // Populate 320 keys -> triggers automatic expansion from q=8 -> q=9 on F_old.
    let initial_keys = generate_seeded_keys(0x0003_0070, 320, 1_000);
    let mut live_truth: BTreeSet<Key> = BTreeSet::new();
    for &k in &initial_keys {
        engine.insert(k, &k.to_le_bytes()).unwrap();
        live_truth.insert(k);
    }
    assert!(engine.filter().config().q >= 9);

    // Collect natural empty FPs into FalsePositiveLog.
    let mut rng = ChaCha8Rng::seed_from_u64(0x0003_0071);
    let mut fp_ranges = Vec::new();
    for _ in 0..80_000 {
        let start = rng.gen_range(20_000_000..400_000_000);
        let range = QueryRange::with_len(start, 32).unwrap();
        let outcome = engine.query_range(range).unwrap();
        if outcome.was_false_positive {
            fp_ranges.push(range);
            if fp_ranges.len() >= 10 {
                break;
            }
        }
    }
    assert!(fp_ranges.len() >= 2);
    let populated_fp_range = fp_ranges[0];
    let remaining_empty_fps = &fp_ranges[1..];

    // Deliberately pass undersized target q=7 (128 slots < 320 keys) to verify auto-expansion
    // during background bulk_load + m=5 -> m=7 reconfiguration!
    let undersized_target = FilterConfig::new(7, 8, 128, 15, 1.0 / 16.0).unwrap();
    engine
        .trigger_async_rebuild(
            undersized_target,
            RebuildReason::DynamicRangeChange { old_m: 5, new_m: 7 },
        )
        .unwrap();

    // Exercise edge-case delta sequences while rebuild runs:
    // (a) Insert a key directly inside a previously logged empty FP interval (must NOT cause FN!).
    let key_inside_prev_fp = populated_fp_range.start() + 5;
    engine
        .insert(key_inside_prev_fp, &key_inside_prev_fp.to_le_bytes())
        .unwrap();
    live_truth.insert(key_inside_prev_fp);

    // (b) Insert then immediately delete the same delta key (net absent).
    let ephemeral_key = 900_000_001u64;
    engine
        .insert(ephemeral_key, &ephemeral_key.to_le_bytes())
        .unwrap();
    assert!(engine.delete(ephemeral_key).unwrap());

    // (c) Delete an existing key then re-insert it (net present).
    let resurrected_key = initial_keys[0];
    assert!(engine.delete(resurrected_key).unwrap());
    engine
        .insert(resurrected_key, &resurrected_key.to_le_bytes())
        .unwrap();
    live_truth.insert(resurrected_key);

    // (d) Concurrent multi-threaded FP log appends while rebuild runs (synchronized via Barrier).
    let fp_log_shared = engine.fp_log();
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    for tid in 0..4u64 {
        let log = Arc::clone(&fp_log_shared);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            for j in 0..25u64 {
                let s = 1_000_000_000 + tid * 100_000 + j * 100;
                let r = QueryRange::new(s, s + 31).unwrap();
                let _ = log.record(r, 10_000 + tid * 100 + j).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Complete rebuild via non-blocking poll or blocking finish.
    let report = match engine.complete_rebuild_if_ready().unwrap() {
        Some(r) => r,
        None => engine.finish_rebuild().unwrap(),
    };
    assert!(
        report.auto_expansions >= 2,
        "expected undersized q=7 target to auto-expand at least twice, got {}",
        report.auto_expansions
    );
    assert!(engine.filter().config().q >= 9);
    assert_eq!(engine.filter().config().m, 7);
    assert_eq!(engine.filter().verify_invariants().ok(), Some(()));

    // Zero false negatives including key inserted inside previously logged FP range + resurrected key.
    assert!(engine.filter().query_point(key_inside_prev_fp));
    assert!(engine.filter().query_range(populated_fp_range));
    assert!(engine.filter().query_point(resurrected_key));
    for &k in &live_truth {
        assert!(engine.filter().query_point(k), "false negative for key {k}");
    }

    // All still-empty prior FPs remain strictly false on F_new.
    for &r in remaining_empty_fps {
        assert!(!engine.filter().query_range(r));
    }

    // Ephemeral insert->delete key must be absent from KV store and Reverse Map.
    assert_eq!(engine.kv_store().get(ephemeral_key).unwrap(), None);
    let rm_keys: BTreeSet<Key> = engine
        .reverse_map()
        .scan_all_keys()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(rm_keys, live_truth);
}
