//! Unit tests for [`super::DeltaBuffer`], [`super::IoThrottler`],
//! [`super::AsyncRebuildCoordinator`], and [`super::ControlledAerisEngine`].

use super::*;
use crate::fp_log::{FalsePositiveLog, FpLogConfig};
use crate::trigger::TriggerAction;
use aeris_core::AerisFilter;
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, FilterConfig, Key, QueryRange, RangeFilter, ReverseMapStore, WorkloadEstimator,
};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::thread;

#[test]
fn delta_buffer_activation_fifo_and_atomic_drain() {
    let buf = DeltaBuffer::new();
    assert!(!buf.is_active());
    assert!(buf.is_empty());

    // Inactive buffer drops writes cleanly.
    assert!(!buf.record_insert(10));
    assert!(!buf.record_delete(20));
    assert_eq!(buf.len(), 0);
    assert!(buf.drain_batch(0).is_empty());

    buf.activate();
    assert!(buf.is_active());
    assert!(buf.record_insert(10));
    assert!(buf.record_delete(20));
    assert!(buf.record_insert(30));
    assert_eq!(buf.len(), 3);

    let drained = buf.drain_all();
    assert_eq!(
        drained,
        vec![
            DeltaMutation::Insert(10),
            DeltaMutation::Delete(20),
            DeltaMutation::Insert(30),
        ]
    );
    assert!(buf.is_active());
    assert!(buf.is_empty());

    assert!(buf.record_delete(99));
    let final_ops = buf.drain_and_deactivate();
    assert_eq!(final_ops, vec![DeltaMutation::Delete(99)]);
    assert!(!buf.is_active());
    assert!(!buf.record_insert(100));
}

#[test]
fn io_throttler_virtual_clock_enforces_exact_token_bucket_rate() {
    assert!(IoThrottleConfig::limited(0, 100).validate().is_err());
    assert!(IoThrottleConfig::limited(1000, 0).validate().is_err());

    // 10_000 keys/sec, batch size 1_000 -> each 1_000-key batch requires exactly 100 ms.
    let mut throttler = IoThrottler::new(IoThrottleConfig::limited(10_000, 1_000)).unwrap();

    // Batch 1 at t = 0: needs 100 ms (100_000_000 ns).
    let w1 = throttler.consume_at(1_000, 0);
    assert_eq!(w1, 100_000_000);

    // Batch 2 arrives at t = 140 ms (40 ms of I/O work elapsed after w1 -> 400 tokens refilled):
    // deficit = 600 keys -> needs 60 ms.
    let w2 = throttler.consume_at(1_000, 140_000_000);
    assert_eq!(w2, 60_000_000);

    // Batch 3 arrives after a 150 ms pause (t = 350 ms -> bucket refills to cap = 1_000):
    // zero wait required!
    let w3 = throttler.consume_at(1_000, 350_000_000);
    assert_eq!(w3, 0);

    // Zero-key pacing never sleeps.
    assert_eq!(throttler.consume_at(0, 400_000_000), 0);

    let metrics = throttler.metrics();
    assert_eq!(metrics.keys_paced, 3_000);
    assert_eq!(metrics.throttle_events, 2);
    assert_eq!(metrics.total_throttled_ns, 160_000_000);
}

#[test]
fn io_throttler_unthrottled_never_waits() {
    let mut throttler = IoThrottler::new(IoThrottleConfig::default()).unwrap();
    assert_eq!(throttler.consume_at(50_000, 0), 0);
    assert_eq!(throttler.consume_at(50_000, 1_000), 0);
    let metrics = throttler.metrics();
    assert_eq!(metrics.keys_paced, 100_000);
    assert_eq!(metrics.throttle_events, 0);
    assert_eq!(metrics.total_throttled_ns, 0);
}

#[test]
fn two_phase_incremental_delta_drain_reduces_tail_and_preserves_zero_fn() {
    let dir = tempfile::TempDir::new().unwrap();
    let src_rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_src"), ReverseMapConfig::default()).unwrap(),
    );
    let dst_rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_dst"), ReverseMapConfig::default()).unwrap(),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
    );

    let base_cfg = FilterConfig::new(10, 15, 32, 20, 1.0 / 256.0).unwrap();
    let seed_keys: Vec<Key> = (0..400u64).map(|i| i * 64 + 7).collect();
    let _seed_filter =
        AerisFilter::bulk_load(base_cfg.clone(), &seed_keys, src_rm.as_ref()).unwrap();

    let coord = AsyncRebuildCoordinator::new();
    let target_cfg = base_cfg.expanded().unwrap();
    coord
        .start_rebuild(
            Arc::clone(&src_rm) as Arc<dyn ReverseMapStore>,
            Arc::clone(&dst_rm) as Arc<dyn ReverseMapStore>,
            Arc::clone(&fp_log),
            target_cfg,
            RebuildReason::ProactiveExpansion {
                old_q: 10,
                new_q: 11,
            },
            IoThrottleConfig::default(),
        )
        .unwrap();

    let delta_buf = coord.delta_buffer();
    let mut expected_live: BTreeSet<Key> = seed_keys.iter().copied().collect();
    for i in 0..2_500u64 {
        let k = 100_000 + i * 32;
        assert!(delta_buf.record_insert(k));
        expected_live.insert(k);
    }
    for i in 0..300u64 {
        let k = 100_000 + i * 32;
        assert!(delta_buf.record_delete(k));
        expected_live.remove(&k);
    }

    while !coord.is_ready() {
        thread::yield_now();
    }
    let mut staged = coord.prepare_finish().unwrap();
    assert!(
        coord.is_in_progress(),
        "Draining slot must block overlapping rebuilds"
    );
    assert!(matches!(
        coord.start_rebuild(
            Arc::clone(&src_rm) as Arc<dyn ReverseMapStore>,
            Arc::clone(&dst_rm) as Arc<dyn ReverseMapStore>,
            Arc::clone(&fp_log),
            base_cfg,
            RebuildReason::ProactiveExpansion {
                old_q: 10,
                new_q: 11
            },
            IoThrottleConfig::default(),
        ),
        Err(AerisError::RebuildInProgress)
    ));

    let drained_off_lock = staged.pre_drain_until_small(256, 512, 32).unwrap();
    assert!(
        drained_off_lock >= 2_500,
        "expected off-lock pre-drain to absorb bulk delta backlog, got {drained_off_lock}"
    );
    assert!(staged.pre_drain_passes() >= 5);
    assert!(
        staged.pending_delta_len() <= 256,
        "residual tail {} exceeded threshold 256",
        staged.pending_delta_len()
    );
    staged.verify_pre_drained_keys().unwrap();

    for tail_i in 0..16u64 {
        let k = 9_000_000 + tail_i * 64;
        assert!(delta_buf.record_insert(k));
        expected_live.insert(k);
    }

    let completed = staged.finish_tail_with_live_keys().unwrap();
    assert!(!coord.is_in_progress());
    assert!(!delta_buf.is_active());
    assert_eq!(completed.report.delta_inserts_applied, 2_516);
    assert_eq!(completed.report.delta_deletes_applied, 300);

    for &k in &expected_live {
        assert!(
            completed.filter.query_point(k),
            "false negative for live key {k} after two-phase delta drain"
        );
    }
    let rm_keys: BTreeSet<Key> = dst_rm.scan_all_keys().unwrap().into_iter().collect();
    assert_eq!(rm_keys, expected_live);
}

#[test]
fn staged_rebuild_drop_guard_recovers_coordinator_to_idle() {
    let dir = tempfile::TempDir::new().unwrap();
    let src_rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_src"), ReverseMapConfig::default()).unwrap(),
    );
    let dst_rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_dst"), ReverseMapConfig::default()).unwrap(),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
    );
    let cfg = FilterConfig::new(8, 15, 32, 20, 1.0 / 256.0).unwrap();

    let coord = AsyncRebuildCoordinator::new();
    // Calling prepare_finish when idle fails cleanly without leaving Draining state.
    assert!(matches!(
        coord.prepare_finish(),
        Err(AerisError::InternalInvariant(_))
    ));
    assert!(!coord.is_in_progress());

    coord
        .start_rebuild(
            Arc::clone(&src_rm) as Arc<dyn ReverseMapStore>,
            Arc::clone(&dst_rm) as Arc<dyn ReverseMapStore>,
            Arc::clone(&fp_log),
            cfg,
            RebuildReason::ProactiveExpansion { old_q: 8, new_q: 9 },
            IoThrottleConfig::default(),
        )
        .unwrap();

    while !coord.is_ready() {
        thread::yield_now();
    }
    let staged = coord.prepare_finish().unwrap();
    assert!(coord.is_in_progress());
    assert!(coord.delta_buffer().is_active());

    drop(staged);
    assert!(!coord.is_in_progress());
    assert!(!coord.delta_buffer().is_active());
}

#[test]
fn static_r_bypass_skips_estimator_prevents_dynamic_rebuilds_and_preserves_monotonic_fp_adapt() {
    let dir = tempfile::TempDir::new().unwrap();
    let cfg = FilterConfig::new(9, 10, 32, 15, 1.0 / 16.0).unwrap();
    let default_engine_cfg = ControlledEngineConfig::new(cfg.clone());
    assert!(default_engine_cfg.dynamic_r_enabled);
    assert!(default_engine_cfg.estimator_config.enabled);

    let mut engine_cfg = ControlledEngineConfig::static_r_only(cfg);
    assert!(!engine_cfg.dynamic_r_enabled);
    assert!(!engine_cfg.estimator_config.enabled);
    engine_cfg.estimator_config.window_size = 128;
    engine_cfg.estimator_config.cooldown_queries = 32;
    engine_cfg.estimator_config.required_sustained_evals = 2;

    let mut engine = ControlledAerisEngine::open(dir.path(), engine_cfg).unwrap();
    assert!(!engine.is_dynamic_r_enabled());
    assert!(!engine.estimator().is_enabled());

    let mut live_keys = Vec::with_capacity(350);
    for i in 0..350u64 {
        let k = i * 1_000 + 13;
        engine.insert(k, &k.to_le_bytes()).unwrap();
        live_keys.push(k);
    }

    for &k in &live_keys {
        let r = QueryRange::point(k);
        assert!(engine.query_range_filter_only(r));
        let outcome = engine.query_range(r).unwrap();
        assert!(outcome.non_empty);
        assert!(!outcome.was_false_positive);
    }
    assert_eq!(engine.estimator().total_queries(), 0);

    let mut adapted_fps = Vec::new();
    for probe in 0..40_000u64 {
        let start = 10_000_000 + probe * 1_024;
        let range = QueryRange::with_len(start, 512).unwrap();
        let filter_only_before = engine.query_range_filter_only(range);
        let outcome = engine.query_range(range).unwrap();
        assert!(!outcome.non_empty);
        assert_eq!(outcome.filter_returned_positive, filter_only_before);
        if outcome.was_false_positive {
            assert!(outcome.adapted);
            assert!(!engine.query_range_filter_only(range));
            adapted_fps.push(range);
            if adapted_fps.len() >= 10 {
                break;
            }
        }
    }
    assert!(adapted_fps.len() >= 5);
    assert_eq!(engine.fp_log().len(), adapted_fps.len());
    assert_eq!(engine.total_false_positives(), adapted_fps.len() as u64);
    assert_eq!(engine.estimator().total_queries(), 0);

    engine.estimator_mut().record_bulk_load_speed(1_000.0);
    engine.record_ingest_sample(10_000, 1.0);
    let decision = engine.evaluate_control_plane();
    assert!(!decision.optimization.should_rebuild);
    assert_eq!(decision.trigger.action, TriggerAction::ProactiveExpand);

    engine.set_dynamic_r_enabled(true);
    assert!(engine.is_dynamic_r_enabled());
    assert!(engine.estimator().is_enabled());
    for i in 0..256u64 {
        let r = QueryRange::with_len(500_000_000 + i * 2_048, 512).unwrap();
        let _ = engine.query_range(r).unwrap();
    }
    assert_eq!(engine.estimator().total_queries(), 256);

    engine.set_dynamic_r_enabled(false);
    assert!(!engine.is_dynamic_r_enabled());
    assert!(!engine.estimator().is_enabled());
    let post_toggle = engine.evaluate_control_plane();
    assert!(!post_toggle.optimization.should_rebuild);

    for &k in &live_keys {
        assert!(engine.query_range_filter_only(QueryRange::point(k)));
    }
    for &r in &adapted_fps {
        assert!(!engine.query_range_filter_only(r));
    }
}

#[test]
fn controlled_engine_config_validation_rejects_invalid_subconfigs() {
    let valid = FilterConfig::new(8, 12, 32, 17, 1.0 / 256.0).unwrap();
    let mut bad = ControlledEngineConfig::new(valid);
    bad.throttle_config.batch_size = 0;
    assert!(matches!(bad.validate(), Err(AerisError::InvalidConfig(_))));
}
