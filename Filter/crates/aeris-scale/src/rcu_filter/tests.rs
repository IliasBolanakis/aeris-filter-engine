//! Unit and rollback tests for [`super::RcuAerisFilter`] and [`super::FilterGeneration`].

use super::*;
use aeris_control::{FalsePositiveLog, FpLogConfig, IoThrottleConfig, RebuildReason};
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use std::sync::atomic::AtomicBool;
use std::sync::Barrier;
use std::thread;
use tempfile::TempDir;

fn test_config(q: u8, m: u8) -> FilterConfig {
    let max_range_r = 1u64 << m;
    let payload_budget_c = 20;
    let r = payload_budget_c - m;
    FilterConfig::new(q, r, max_range_r, payload_budget_c, 1.0 / 256.0).unwrap()
}

#[test]
fn rcu_insert_query_delete_and_auto_expand() {
    let dir = TempDir::new().unwrap();
    let rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm"), ReverseMapConfig::default()).unwrap(),
    );
    let cfg = test_config(7, 4); // 128 slots -> triggers auto-expand quickly
    let filter = AerisFilter::new(cfg).unwrap();
    let rcu = RcuAerisFilter::new(0, filter, rm);

    let initial_epoch = rcu.epoch();
    for k in 0..300u64 {
        let key = k * 17;
        rcu.insert(key).unwrap();
        assert!(rcu.query_point(key));
    }

    assert!(
        rcu.epoch() > initial_epoch,
        "auto-expansion must advance RCU epoch"
    );
    assert!(rcu.capacity() > 128);
    assert_eq!(rcu.len(), 300);
    rcu.verify_invariants().unwrap();

    for k in 0..300u64 {
        assert!(rcu.query_point(k * 17));
    }
    for k in 0..100u64 {
        assert!(rcu.delete(k * 17).unwrap().is_some());
    }
    assert_eq!(rcu.len(), 200);
    for k in 100..300u64 {
        assert!(rcu.query_point(k * 17));
    }
    rcu.verify_invariants().unwrap();
}

#[test]
fn rcu_async_rebuild_changes_m_and_preserves_monotonicity() {
    let dir = TempDir::new().unwrap();
    let rm0 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_0"), ReverseMapConfig::default()).unwrap(),
    );
    let rm1 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_1"), ReverseMapConfig::default()).unwrap(),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
    );

    let cfg_old = test_config(9, 4); // m = 4 (R = 16)
    let filter = AerisFilter::new(cfg_old.clone()).unwrap();
    let rcu = RcuAerisFilter::new(7, filter, rm0);

    let mut inserted = Vec::new();
    for i in 0..200u64 {
        let k = i * 1000;
        rcu.insert(k).unwrap();
        inserted.push(k);
    }

    // Find real empty-range false positives and adapt before rebuild.
    let mut adapted_ranges = Vec::new();
    for probe in 0..20_000u64 {
        let start = probe * 1000 + 100;
        let r = QueryRange::new(start, start + 15).unwrap();
        if rcu.query_range(r) {
            fp_log.record(r, 1).unwrap();
            rcu.adapt(r).unwrap();
            assert!(!rcu.query_range(r));
            adapted_ranges.push(r);
            if adapted_ranges.len() >= 10 {
                break;
            }
        }
    }

    let cfg_new = cfg_old.reconfigure_for_m(6).unwrap(); // m = 6 (R = 64)
    let report = rcu
        .rebuild_async_and_swap(
            rm1,
            fp_log,
            cfg_new,
            RebuildReason::DynamicRangeChange { old_m: 4, new_m: 6 },
            IoThrottleConfig::default(),
        )
        .unwrap();

    assert!(report.new_epoch > report.old_epoch);
    assert_eq!(rcu.config().m, 6);
    assert_eq!(rcu.config().r, 14);
    rcu.verify_invariants().unwrap();

    // Zero false negatives across all inserted keys.
    for k in inserted {
        assert!(rcu.query_point(k), "false negative for key {k}");
    }
    // 100% monotonic adaptivity across dynamic-R RCU swap.
    for r in adapted_ranges {
        assert!(
            !rcu.query_range(r),
            "previously adapted empty range {r:?} regressed to FP after RCU rebuild"
        );
    }
}

#[test]
fn rcu_swap_filter_drains_delta_buffer_and_gates_concurrent_rebuild() {
    let dir = TempDir::new().unwrap();
    let rm0 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm0"), ReverseMapConfig::default()).unwrap(),
    );
    let rm1 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm1"), ReverseMapConfig::default()).unwrap(),
    );
    let rm2 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm2"), ReverseMapConfig::default()).unwrap(),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap(),
    );

    let cfg = test_config(8, 4);
    let rcu = RcuAerisFilter::new(3, AerisFilter::new(cfg.clone()).unwrap(), rm0);
    for k in 0..40u64 {
        rcu.insert(k * 10).unwrap();
    }

    // Manual swap_filter drains active DeltaBuffer inserts and deletes atomically.
    rcu.delta_buffer().activate();
    let offline_keys: Vec<Key> = (0..40u64).map(|k| k * 10).collect();
    let f_new = AerisFilter::bulk_load(cfg.clone(), &offline_keys, rm1.as_ref()).unwrap();

    // Concurrent mutations arriving after offline snapshot are captured in DeltaBuffer.
    rcu.insert(9999).unwrap();
    rcu.insert(8888).unwrap();
    assert!(rcu.delete(10).unwrap().is_some());

    let new_ep = rcu
        .swap_filter(f_new, Arc::clone(&rm1) as Arc<dyn ReverseMapStore>)
        .unwrap();
    assert_eq!(new_ep, 2);
    assert!(!rcu.delta_buffer().is_active());
    assert!(rcu.query_point(9999));
    assert!(rcu.query_point(8888));
    assert!(rcu.query_point(0));
    rcu.verify_invariants().unwrap();

    // Two-phase async rebuild rejects overlapping start_async_rebuild with RebuildInProgress.
    let cfg_m5 = cfg.reconfigure_for_m(5).unwrap();
    rcu.start_async_rebuild(
        Arc::clone(&rm2) as Arc<dyn ReverseMapStore>,
        Arc::clone(&fp_log),
        cfg_m5.clone(),
        RebuildReason::DynamicRangeChange { old_m: 4, new_m: 5 },
        IoThrottleConfig::default(),
    )
    .unwrap();
    assert!(rcu.is_rebuild_in_progress());

    let dup = rcu.start_async_rebuild(
        rm2,
        fp_log,
        cfg_m5,
        RebuildReason::DynamicRangeChange { old_m: 4, new_m: 5 },
        IoThrottleConfig::default(),
    );
    assert!(matches!(dup, Err(AerisError::RebuildInProgress)));

    // Foreground writes proceed cleanly while rebuild is active and survive swap.
    rcu.insert(77_777).unwrap();
    let rep = rcu
        .finish_async_rebuild_and_swap_from_reverse_map()
        .unwrap();
    assert_eq!(rep.new_epoch, 3);
    assert!(!rcu.is_rebuild_in_progress());
    assert!(rcu.query_point(77_777));
    assert!(rcu.query_point(9999));
    assert_eq!(rcu.config().m, 5);
    rcu.verify_invariants().unwrap();
}

#[test]
fn rcu_insert_releases_filter_lock_before_reverse_map_io_and_rolls_back_on_error() {
    struct ControlledReverseMap {
        inner: RocksDbReverseMap,
        fail_append: AtomicBool,
        fail_remove: AtomicBool,
        io_entered: Option<Arc<Barrier>>,
        io_release: Option<Arc<Barrier>>,
    }

    impl ReverseMapStore for ControlledReverseMap {
        fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
            if let Some(ref b) = self.io_entered {
                b.wait();
            }
            if let Some(ref b) = self.io_release {
                b.wait();
            }
            if self.fail_append.load(Ordering::Acquire) {
                return Err(AerisError::StorageError("injected append failure".into()));
            }
            self.inner.append_key(fp, key)
        }
        fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
            self.inner.append_batch(entries)
        }
        fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
            if self.fail_remove.load(Ordering::Acquire) {
                return Err(AerisError::StorageError("injected remove failure".into()));
            }
            self.inner.remove_key(fp, key)
        }
        fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
            self.inner.get_keys(fp)
        }
        fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
            self.inner.tombstone(fp)
        }
        fn replace_fingerprint(
            &self,
            old_fp: &Fingerprint,
            new_entries: &[(Fingerprint, Key)],
        ) -> AerisResult<()> {
            self.inner.replace_fingerprint(old_fp, new_entries)
        }
        fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
            self.inner.scan_all_keys()
        }
        fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
            self.inner.iter_all_keys()
        }
        fn flush(&self) -> AerisResult<()> {
            self.inner.flush()
        }
        fn compact(&self) -> AerisResult<()> {
            self.inner.compact()
        }
        fn entry_count_estimate(&self) -> u64 {
            self.inner.entry_count_estimate()
        }
    }

    // Part 1: Verify rollback restores filter state and `len()` on storage errors.
    let dir = TempDir::new().unwrap();
    let inner =
        RocksDbReverseMap::open(dir.path().join("rm_err"), ReverseMapConfig::default()).unwrap();
    let rm = Arc::new(ControlledReverseMap {
        inner,
        fail_append: AtomicBool::new(false),
        fail_remove: AtomicBool::new(false),
        io_entered: None,
        io_release: None,
    });
    let rcu = RcuAerisFilter::new(
        1,
        AerisFilter::new(test_config(8, 4)).unwrap(),
        Arc::clone(&rm) as Arc<dyn ReverseMapStore>,
    );

    rcu.insert(100).unwrap();
    assert_eq!(rcu.len(), 1);

    rm.fail_append.store(true, Ordering::Release);
    assert!(matches!(rcu.insert(200), Err(AerisError::StorageError(_))));
    assert_eq!(rcu.len(), 1);
    assert!(!rcu.query_point(200));
    rcu.verify_invariants().unwrap();

    rm.fail_append.store(false, Ordering::Release);
    rm.fail_remove.store(true, Ordering::Release);
    assert!(matches!(rcu.delete(100), Err(AerisError::StorageError(_))));
    assert_eq!(rcu.len(), 1);
    assert!(rcu.query_point(100));
    rcu.verify_invariants().unwrap();

    // Part 2: Prove concurrent readers complete while a writer is parked inside
    // `reverse_map.append_key` (would deadlock if `gen.filter.write()` were held).
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let inner2 =
        RocksDbReverseMap::open(dir.path().join("rm_block"), ReverseMapConfig::default()).unwrap();
    let rm2 = Arc::new(ControlledReverseMap {
        inner: inner2,
        fail_append: AtomicBool::new(false),
        fail_remove: AtomicBool::new(false),
        io_entered: Some(Arc::clone(&entered)),
        io_release: Some(Arc::clone(&release)),
    });
    let rcu2 = Arc::new(RcuAerisFilter::new(
        2,
        AerisFilter::new(test_config(8, 4)).unwrap(),
        rm2 as Arc<dyn ReverseMapStore>,
    ));

    let writer_rcu = Arc::clone(&rcu2);
    let writer = thread::spawn(move || {
        writer_rcu.insert(42_000).unwrap();
    });

    entered.wait();
    assert!(rcu2.query_point(42_000));
    assert_eq!(rcu2.len(), 1);
    release.wait();

    writer.join().expect("writer finished cleanly");
    rcu2.verify_invariants().unwrap();
}

#[test]
fn rcu_direct_methods_match_concurrent_api_across_auto_expansions() {
    let dir = TempDir::new().unwrap();
    let rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_direct"), ReverseMapConfig::default()).unwrap(),
    );
    let cfg = test_config(7, 4); // 128 slots -> triggers multiple auto-expansions
    let mut rcu = RcuAerisFilter::new(5, AerisFilter::new(cfg).unwrap(), rm);

    let initial_epoch = rcu.epoch();
    for k in 0..320u64 {
        let key = k * 37 + 5;
        rcu.insert_direct(key).unwrap();
        assert!(rcu.query_point_direct(key));
        assert!(rcu.query_point(key));
    }

    assert!(rcu.epoch() > initial_epoch);
    assert_eq!(rcu.len(), 320);
    rcu.verify_invariants().unwrap();

    // Bit-exact agreement on present, absent, point, and range queries.
    for probe in 0..500u64 {
        let key = probe * 19;
        assert_eq!(rcu.query_point_direct(key), rcu.query_point(key));
        let range = QueryRange::new(key, key + 31).unwrap();
        assert_eq!(rcu.query_range_direct(range), rcu.query_range(range));
    }
}

#[test]
fn rcu_rebuild_errors_and_filter_generation_surface() {
    let dir = TempDir::new().unwrap();
    let rm = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_gen"), ReverseMapConfig::default()).unwrap(),
    );
    let filter = AerisFilter::new(test_config(8, 4)).unwrap();
    let gen = FilterGeneration::new(42, filter, Arc::clone(&rm) as Arc<dyn ReverseMapStore>, 16);
    assert_eq!(gen.epoch, 42);
    assert!(!gen.is_retired());
    assert_eq!(gen.reverse_map().entry_count_estimate(), 0);

    let rcu = RcuAerisFilter::new(9, AerisFilter::new(test_config(8, 4)).unwrap(), rm);
    assert!(rcu.is_empty());
    assert_eq!(rcu.shard_id(), 9);
    assert!(!rcu.is_rebuild_ready());
    assert_eq!(rcu.expansion_cycles(), 0);
    assert!(rcu.memory_usage_bytes() > 0);
    assert_eq!(rcu.occupied_slots(), 0);

    let finish_err = rcu.finish_async_rebuild_and_swap(|_| Ok(true));
    assert!(matches!(finish_err, Err(AerisError::InternalInvariant(_))));

    let finish_rm_err = rcu.finish_async_rebuild_and_swap_from_reverse_map();
    assert!(matches!(
        finish_rm_err,
        Err(AerisError::InternalInvariant(_))
    ));

    // When epoch has already advanced, `expand_rcu_if_epoch` skips redundant expansion.
    let stale_epoch = rcu.epoch();
    let rep = rcu.expand_rcu().unwrap();
    assert_eq!(rep.old_epoch, stale_epoch);
    assert_eq!(rep.new_epoch, stale_epoch + 1);
    assert!(rcu.expand_rcu_if_epoch(stale_epoch).unwrap().is_none());
}
