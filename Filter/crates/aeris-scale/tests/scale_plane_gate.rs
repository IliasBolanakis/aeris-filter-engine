//! Milestone M4 Scale Plane integration, E2E, and concurrency gate suite.
//!
//! All tests run against the REAL Data Plane (`AerisFilter`), REAL Storage Plane
//! (`RocksDbReverseMap`), and REAL Control Plane (`FalsePositiveLog`,
//! `AsyncRebuildCoordinator`, `SlidingWindowEstimator`) with zero mocks and zero
//! `thread::sleep()` calls.

use aeris_control::{FalsePositiveLog, FpLogConfig, IoThrottleConfig, RebuildReason};
use aeris_core::AerisFilter;
use aeris_scale::{RcuAerisFilter, ShardedAerisSystem, ShardingConfig};
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{AerisError, FilterConfig, Key, QueryRange, ShardBounds};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

fn make_filter_config(q: u8, m: u8) -> FilterConfig {
    // eps = 1/64 -> r_min = ceil(6 + log2(6) + 1) = 10, giving ~1/1024 raw partition
    // collision rate so tests deterministically observe and adapt genuine FPs rapidly.
    let target_epsilon = 1.0 / 64.0;
    let r = 10u8;
    let payload_budget_c = r + m;
    let max_range_r = 1u64 << m;
    FilterConfig::new(q, r, max_range_r, payload_budget_c, target_epsilon)
        .expect("valid test FilterConfig")
}

fn make_dynamic_r_config(q: u8, m: u8) -> FilterConfig {
    // Budget C = 16 with eps = 1/64 (r_min = 10) permits dynamic m in [0..=6] while
    // producing abundant genuine FPs for monotonicity verification.
    let target_epsilon = 1.0 / 64.0;
    let payload_budget_c = 16u8;
    let r = payload_budget_c - m;
    let max_range_r = 1u64 << m;
    FilterConfig::new(q, r, max_range_r, payload_budget_c, target_epsilon)
        .expect("valid dynamic-R FilterConfig")
}

fn lean_rm_config() -> ReverseMapConfig {
    ReverseMapConfig::default()
        .with_write_buffer_size(2 * 1024 * 1024)
        .with_sync_writes(false)
}

fn fast_fp_log_config() -> FpLogConfig {
    FpLogConfig {
        max_entries: 16_384,
        sync_on_append: false,
    }
}

/// Gate 1: Zero-stall RCU atomic swap (`F_old -> F_new`) under 8 concurrent readers,
/// 2 concurrent writers, repeated `expand_rcu` swaps, and live dynamic-`R` async rebuilds.
#[test]
fn gate1_rcu_zero_stall_concurrent_readers_writers_and_dynamic_r_swaps() {
    const READERS: usize = 8;
    const WRITERS: usize = 2;
    const KEYS_PER_WRITER: usize = 350;

    let dir = TempDir::new().expect("temp dir");
    let rm0 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_gen_0"), lean_rm_config())
            .expect("open rm_gen_0"),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(dir.path().join("fp.log"), fast_fp_log_config())
            .expect("open fp.log"),
    );

    let base_cfg = make_dynamic_r_config(8, 4); // 256 slots, m = 4 (R = 16), C = 16
    let filter = AerisFilter::new(base_cfg.clone()).expect("init AerisFilter");
    let rcu = Arc::new(RcuAerisFilter::new(1, filter, rm0));

    // Seed baseline keys in [0, 500_000) with ChaCha8Rng (spaced cleanly from writer keys).
    let mut rng = ChaCha8Rng::seed_from_u64(0x4D34_0001);
    let mut baseline_set = BTreeSet::new();
    while baseline_set.len() < 180 {
        baseline_set.insert(rng.gen_range(0..500_000u64));
    }
    let baseline_keys: Arc<Vec<Key>> = Arc::new(baseline_set.iter().copied().collect());
    for &k in baseline_keys.iter() {
        rcu.insert(k).expect("baseline insert");
    }

    // Record and adapt real empty-range FPs strictly in a reserved empty zone [10_000_000, 20_000_000).
    let mut adapted_empty_ranges = Vec::new();
    for probe in 0..80_000u64 {
        let start = 10_000_000 + probe * 64;
        let range = QueryRange::new(start, start + 15).expect("valid range");
        if rcu.query_range(range) {
            fp_log.record(range, probe + 1).expect("record fp");
            rcu.adapt(range).expect("adapt fp");
            assert!(!rcu.query_range(range));
            adapted_empty_ranges.push(range);
            if adapted_empty_ranges.len() >= 12 {
                break;
            }
        }
    }
    assert!(
        adapted_empty_ranges.len() >= 5,
        "expected >= 5 genuine FPs to verify monotonicity across RCU dynamic-R swaps, found {}",
        adapted_empty_ranges.len()
    );

    let initial_epoch = rcu.epoch();
    let barrier = Arc::new(Barrier::new(READERS + WRITERS + 1));
    let done = Arc::new(AtomicBool::new(false));
    let total_reads = Arc::new(AtomicUsize::new(0));

    // Spawn 8 continuous zero-stall reader threads.
    let mut reader_handles = Vec::with_capacity(READERS);
    for tid in 0..READERS {
        let rcu_ref = Arc::clone(&rcu);
        let keys_ref = Arc::clone(&baseline_keys);
        let bar = Arc::clone(&barrier);
        let done_ref = Arc::clone(&done);
        let reads_ref = Arc::clone(&total_reads);

        reader_handles.push(thread::spawn(move || {
            let mut local_rng = ChaCha8Rng::seed_from_u64(0xBEEF_0000 + tid as u64);
            let mut prev_epoch = 0u64;
            let mut ops = 0usize;
            bar.wait();

            while !done_ref.load(Ordering::Acquire) || ops < 2_000 {
                let ep = rcu_ref.epoch();
                assert!(
                    ep >= prev_epoch,
                    "reader {tid} observed epoch regression: {ep} < {prev_epoch}"
                );
                prev_epoch = ep;

                let key = keys_ref[local_rng.gen_range(0..keys_ref.len())];
                assert!(
                    rcu_ref.query_point(key),
                    "FALSE NEGATIVE on baseline point query key={key} at epoch={ep}"
                );
                let enclosing =
                    QueryRange::new(key.saturating_sub(3), key.saturating_add(3)).unwrap();
                assert!(
                    rcu_ref.query_range(enclosing),
                    "FALSE NEGATIVE on baseline range query {enclosing:?} at epoch={ep}"
                );
                ops += 2;
            }
            reads_ref.fetch_add(ops, Ordering::Relaxed);
        }));
    }

    // Spawn 2 writer threads inserting disjoint key ranges in [1_000_000, 5_000_000).
    let mut writer_handles = Vec::with_capacity(WRITERS);
    for wid in 0..WRITERS {
        let rcu_ref = Arc::clone(&rcu);
        let bar = Arc::clone(&barrier);
        writer_handles.push(thread::spawn(move || {
            let mut local_rng = ChaCha8Rng::seed_from_u64(0xCAFE_1000 + wid as u64);
            let base = 1_000_000u64 + (wid as u64) * 1_000_000;
            let mut inserted = BTreeSet::new();
            while inserted.len() < KEYS_PER_WRITER {
                inserted.insert(base + local_rng.gen_range(0..500_000u64));
            }
            let keys: Vec<Key> = inserted.into_iter().collect();
            bar.wait();

            let mut live = Vec::with_capacity(keys.len());
            for (idx, &k) in keys.iter().enumerate() {
                rcu_ref.insert(k).expect("concurrent writer insert");
                assert!(
                    rcu_ref.query_point(k),
                    "immediate read-after-write false negative for key={k}"
                );
                // Delete every 10th inserted key to stress concurrent delete + delta drain.
                if idx % 10 == 9 {
                    let removed = rcu_ref.delete(k).expect("concurrent writer delete");
                    assert!(removed.is_some(), "just-inserted key {k} missing on delete");
                } else {
                    live.push(k);
                }
            }
            live
        }));
    }

    // Coordinator thread triggers explicit RCU 2x expansion + two dynamic-R async rebuild swaps.
    barrier.wait();

    let exp_rep = rcu.expand_rcu().expect("explicit expand_rcu");
    assert!(exp_rep.new_epoch > exp_rep.old_epoch);

    let rm_rebuilt_1 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_gen_1"), lean_rm_config())
            .expect("open rm_gen_1"),
    );
    let target_cfg_1 = rcu.config().reconfigure_for_m(6).expect("reconfigure m=6");
    let reb1 = rcu
        .rebuild_async_and_swap(
            rm_rebuilt_1,
            Arc::clone(&fp_log),
            target_cfg_1,
            RebuildReason::DynamicRangeChange { old_m: 4, new_m: 6 },
            IoThrottleConfig::default(),
        )
        .expect("dynamic-R rebuild m=4 -> m=6");
    assert!(reb1.new_epoch > reb1.old_epoch);
    assert_eq!(rcu.config().m, 6);

    let rm_rebuilt_2 = Arc::new(
        RocksDbReverseMap::open(dir.path().join("rm_gen_2"), lean_rm_config())
            .expect("open rm_gen_2"),
    );
    let target_cfg_2 = rcu.config().reconfigure_for_m(5).expect("reconfigure m=5");
    let reb2 = rcu
        .rebuild_async_and_swap(
            rm_rebuilt_2,
            Arc::clone(&fp_log),
            target_cfg_2,
            RebuildReason::DynamicRangeChange { old_m: 6, new_m: 5 },
            IoThrottleConfig::default(),
        )
        .expect("dynamic-R rebuild m=6 -> m=5");
    assert!(reb2.new_epoch > reb2.old_epoch);
    assert_eq!(rcu.config().m, 5);

    let mut all_writer_live_keys = Vec::new();
    for h in writer_handles {
        all_writer_live_keys.extend(h.join().expect("writer thread panicked"));
    }
    done.store(true, Ordering::Release);
    for h in reader_handles {
        h.join().expect("reader thread panicked");
    }

    // Post-join strict verification.
    assert!(rcu.epoch() >= initial_epoch + 3);
    assert!(total_reads.load(Ordering::Relaxed) >= READERS * 2_000);
    rcu.verify_invariants()
        .expect("post-stress filter invariants");

    // 0 false negatives across ALL baseline and writer-inserted surviving keys.
    for &k in baseline_keys.iter().chain(all_writer_live_keys.iter()) {
        assert!(rcu.query_point(k), "post-join FALSE NEGATIVE for key={k}");
    }
    // 100% monotonic adaptivity across all adapted empty ranges.
    for r in adapted_empty_ranges {
        assert!(
            !rcu.query_range(r),
            "monotonicity violation: empty range {r:?} returned FP after RCU dynamic-R swaps"
        );
    }
}

/// Gate 2: Skewed hot-spot dynamic shard split over 8 uniform shards.
/// Verifies data-driven median splitting, smooth incremental RAM growth (not 2x whole-system jump),
/// contiguous routing, 0 false negatives, and 0 repeated FPs across splits.
#[test]
fn gate2_skewed_hotspot_dynamic_shard_split_incremental_ram_and_monotonicity() {
    let dir = TempDir::new().expect("temp dir");
    let base_cfg = make_filter_config(10, 5); // 1024 slots/shard, m = 5 (R = 32)
    let sharding_cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(8)
        .with_split_load_threshold(0.75)
        .with_split_min_keys(128)
        .with_max_shard_q(10)
        .with_merge_load_threshold(0.15)
        .with_reverse_map_config(lean_rm_config())
        .with_fp_log_config(fast_fp_log_config());

    let sys = ShardedAerisSystem::open(dir.path(), sharding_cfg).expect("open ShardedAerisSystem");
    sys.verify_routing_invariants().expect("initial routing");
    assert_eq!(sys.shard_count(), 8);

    let initial_ram_bytes = sys.total_memory_bytes();
    assert!(initial_ram_bytes > 0);

    // Light uniform keys across non-hot shards (shards 1..8) so they remain healthy & unmerged.
    let mut rng = ChaCha8Rng::seed_from_u64(0x4D34_0002);
    let mut all_inserted = BTreeSet::new();
    for (_, bounds) in sys.shard_bounds().into_iter().skip(1) {
        let base = bounds.min_key + 10_000;
        for offset in 0..220u64 {
            let k = base + offset * 97;
            sys.insert(k).expect("cold shard insert");
            all_inserted.insert(k);
        }
    }

    // Hammer ONE narrow hot-spot interval [1_000_000, 1_040_000] (all inside shard 0!)
    // with 1,500 distinct keys so even with keepsake box compression the single hot shard
    // exceeds `split_load_threshold` / `max_shard_q` while shards 1..7 remain untouched.
    let mut hotspot_keys = BTreeSet::new();
    while hotspot_keys.len() < 1_500 {
        hotspot_keys.insert(rng.gen_range(1_000_000..=1_040_000u64));
    }
    let hot_shard_id = sys.route_shard_id(1_000_000);
    for &k in &hotspot_keys {
        let assigned = sys.insert(k).expect("hotspot insert");
        assert_eq!(assigned, hot_shard_id, "all hotspot keys land in shard 0");
        all_inserted.insert(k);
    }

    // Discover and adapt real empty-range FPs on BOTH sides of the hotspot median
    // within Shard 0 (left band [100_000, 999_000) and right band (1_050_000, 5_000_000))
    // so both split children inherit and replay real FP log entries!
    let mut adapted_empty_ranges = Vec::new();
    let mut left_fps = 0usize;
    let mut right_fps = 0usize;
    for _probe in 0..80_000u64 {
        let start = if left_fps <= right_fps {
            rng.gen_range(100_000..990_000u64)
        } else {
            rng.gen_range(1_050_000..5_000_000u64)
        };
        let range = QueryRange::new(start, start + 31).unwrap();
        let is_empty = all_inserted
            .range(range.start()..=range.end())
            .next()
            .is_none();
        if is_empty && sys.query_range(range) {
            sys.adapt_range(range).expect("adapt hotspot FP");
            assert!(!sys.query_range(range));
            if start < 1_000_000 {
                left_fps += 1;
            } else {
                right_fps += 1;
            }
            adapted_empty_ranges.push(range);
            if left_fps >= 6 && right_fps >= 6 {
                break;
            }
        }
    }
    assert!(
        left_fps >= 4 && right_fps >= 4,
        "expected genuine FPs on both sides of split median: left_fps={left_fps}, right_fps={right_fps}"
    );

    // Trigger automatic rebalance -> hot shard splits cleanly at exact data median!
    let rebalance = sys.rebalance_if_needed().expect("rebalance_if_needed");
    assert!(
        !rebalance.splits.is_empty(),
        "hot shard must trigger at least one split"
    );
    assert!(
        sys.shard_count() > 8,
        "shard count must increase after hot-spot split"
    );

    // Every split produced balanced children (|left_keys - right_keys| <= 1) inside the hotspot!
    for split in &rebalance.splits {
        let diff = split.left_keys.abs_diff(split.right_keys);
        assert!(
            diff <= 1,
            "data-driven median split must balance children: left={}, right={}",
            split.left_keys,
            split.right_keys
        );
        assert!(
            split.split_key >= 1_000_000 && split.split_key <= 1_040_000,
            "split boundary {} must track skewed hotspot data median",
            split.split_key
        );
    }

    // Verify smooth incremental RAM growth (well under a 2x whole-system monolithic jump!).
    let post_split_ram_bytes = sys.total_memory_bytes();
    let ram_growth_ratio = (post_split_ram_bytes as f64) / (initial_ram_bytes as f64);
    assert!(
        ram_growth_ratio < 1.50,
        "expected incremental shard RAM growth (< 1.50x), got {ram_growth_ratio:.3}x (initial={initial_ram_bytes}, post={post_split_ram_bytes})"
    );

    // Verify routing, Reverse Map bounds, Quotient Filter invariants, telemetry, 0 FNs, 0 repeated FPs.
    sys.verify_all_invariants()
        .expect("post-split system invariants");

    let telemetry = sys.shard_telemetry_snapshots();
    assert_eq!(telemetry.len(), sys.shard_count());
    let telemetry_keys: u64 = telemetry.iter().map(|t| t.key_count).sum();
    assert_eq!(telemetry_keys as usize, all_inserted.len());

    for &k in &all_inserted {
        assert!(sys.query_point(k), "post-split FALSE NEGATIVE for key={k}");
    }
    for r in adapted_empty_ranges {
        assert!(
            !sys.query_range(r),
            "post-split monotonicity violation: empty range {r:?} returned FP"
        );
    }
}

/// Gate 3: Dynamic shard merge after bulk deletions.
/// Verifies shard count decreases, RAM is reclaimed, routing remains contiguous,
/// 0 false negatives, and 0 repeated FPs across merge.
#[test]
fn gate3_dynamic_shard_merge_after_bulk_deletions_preserves_monotonicity() {
    let dir = TempDir::new().expect("temp dir");
    let base_cfg = make_filter_config(9, 4); // 512 slots/shard, m = 4 (R = 16)
    let sharding_cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(4)
        .with_split_load_threshold(0.85)
        .with_merge_load_threshold(0.20)
        .with_min_shards(2)
        .with_reverse_map_config(lean_rm_config())
        .with_fp_log_config(fast_fp_log_config());

    let sys = ShardedAerisSystem::open_with_boundaries(
        dir.path(),
        sharding_cfg,
        &[9_999_999, 19_999_999, 29_999_999],
    )
    .expect("open 4-shard system");
    assert_eq!(sys.shard_count(), 4);

    let mut rng = ChaCha8Rng::seed_from_u64(0x4D34_0003);
    let mut shard_keys: Vec<Vec<Key>> = vec![Vec::new(); 4];
    let ranges = [
        (0u64, 9_999_999u64),
        (10_000_000, 19_999_999),
        (20_000_000, 29_999_999),
        (30_000_000, 39_999_999),
    ];

    for (idx, &(lo, hi)) in ranges.iter().enumerate() {
        let mut set = BTreeSet::new();
        while set.len() < 180 {
            set.insert(rng.gen_range(lo..=hi));
        }
        for &k in &set {
            sys.insert(k).expect("insert key");
        }
        shard_keys[idx] = set.into_iter().collect();
    }

    // All 4 shards currently have load ~ 180/512 ~ 0.35 > merge_load_threshold (0.20).
    let noop = sys.rebalance_if_needed().expect("rebalance before deletes");
    assert!(noop.merges.is_empty());
    assert_eq!(sys.shard_count(), 4);

    // Bulk delete 140 of 180 keys from shard 1 and shard 2 (leaving 40 keys each, load ~ 0.08 < 0.20).
    let mut surviving = BTreeSet::new();
    for (idx, keys) in shard_keys.iter().enumerate() {
        if idx == 1 || idx == 2 {
            for &k in &keys[..140] {
                assert!(sys.delete(k).expect("bulk delete"));
            }
            for &k in &keys[140..] {
                surviving.insert(k);
            }
        } else {
            for &k in keys {
                surviving.insert(k);
            }
        }
    }

    // Find and adapt real empty-range FPs across shards 1 and 2 before merging.
    let mut adapted_empty_ranges = Vec::new();
    for _probe in 0..200_000u64 {
        let start = rng.gen_range(10_000_000..29_999_900u64);
        let range = QueryRange::new(start, start + 15).unwrap();
        let is_empty = surviving
            .range(range.start()..=range.end())
            .next()
            .is_none();
        if is_empty && sys.query_range(range) {
            sys.adapt_range(range).expect("adapt pre-merge FP");
            assert!(!sys.query_range(range));
            adapted_empty_ranges.push(range);
            if adapted_empty_ranges.len() >= 10 {
                break;
            }
        }
    }
    assert!(
        adapted_empty_ranges.len() >= 4,
        "expected >= 4 genuine FPs before merge to verify monotonicity, found {}",
        adapted_empty_ranges.len()
    );

    let ram_before_merge = sys.total_memory_bytes();
    let count_before_merge = sys.shard_count();

    // Trigger rebalance -> shards 1 & 2 merge automatically!
    let rebalance = sys.rebalance_if_needed().expect("rebalance after deletes");
    assert_eq!(rebalance.merges.len(), 1);
    assert_eq!(sys.shard_count(), count_before_merge - 1);
    assert!(
        sys.total_memory_bytes() < ram_before_merge,
        "merging two shards must reclaim RAM: before={ram_before_merge}, after={}",
        sys.total_memory_bytes()
    );

    // Verify merged shard covers [10_000_000, 29_999_999] contiguously with all 80 surviving keys.
    let merge_rep = rebalance.merges[0];
    assert_eq!(
        merge_rep.merged_bounds,
        ShardBounds::new(10_000_000, 29_999_999).unwrap()
    );
    assert_eq!(merge_rep.merged_keys, 80);

    sys.verify_all_invariants()
        .expect("post-merge system invariants");

    // 0 false negatives across ALL surviving keys.
    for &k in &surviving {
        assert!(sys.query_point(k), "post-merge FALSE NEGATIVE for key={k}");
    }
    // 0 repeated FPs across ALL adapted empty ranges (monotonicity preserved across merge!).
    for r in adapted_empty_ranges {
        assert!(
            !sys.query_range(r),
            "post-merge monotonicity violation: empty range {r:?} returned FP"
        );
    }
}

/// Gate 4: Concurrent multi-threaded reads + writes + live splits AND merges.
/// Coordinated strictly via `Barrier` + `AtomicBool` (zero `thread::sleep()`!).
/// Verifies 100% of inserted keys return `true` (0 lost writes, 0 false negatives)
/// and full routing/storage/filter invariants hold after live topology churn.
#[test]
fn gate4_concurrent_reads_writes_and_live_splits_merges_zero_lost_writes() {
    const READERS: usize = 4;
    const WRITERS: usize = 4;
    const KEYS_PER_WRITER: usize = 250;

    let dir = TempDir::new().expect("temp dir");
    let base_cfg = make_filter_config(8, 4); // 256 slots/shard
    let sharding_cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(4)
        .with_split_load_threshold(0.65)
        .with_split_min_keys(40)
        .with_max_shard_q(9)
        .with_merge_load_threshold(0.25)
        .with_min_shards(2)
        .with_max_shards(32)
        .with_reverse_map_config(lean_rm_config())
        .with_fp_log_config(fast_fp_log_config());

    let sys = Arc::new(
        ShardedAerisSystem::open_with_boundaries(
            dir.path(),
            sharding_cfg,
            &[99_999, 199_999, 299_999],
        )
        .expect("open 4-shard system"),
    );

    // Pre-populate baseline keys across all 4 shards so readers continuously probe live data.
    let mut baseline = BTreeSet::new();
    for shard_idx in 0..4u64 {
        let base = shard_idx * 100_000;
        for i in 0..30u64 {
            let k = base + i * 1_000 + 7;
            sys.insert(k).expect("baseline insert");
            baseline.insert(k);
        }
    }
    let baseline_vec: Arc<Vec<Key>> = Arc::new(baseline.iter().copied().collect());

    let barrier = Arc::new(Barrier::new(READERS + WRITERS + 1));
    let writers_done = Arc::new(AtomicBool::new(false));
    let all_done = Arc::new(AtomicBool::new(false));
    let total_reads = Arc::new(AtomicUsize::new(0));

    // Spawn 4 continuous reader threads verifying 0 false negatives & monotonic epochs.
    let mut reader_handles = Vec::with_capacity(READERS);
    for rid in 0..READERS {
        let sys_ref = Arc::clone(&sys);
        let base_ref = Arc::clone(&baseline_vec);
        let bar = Arc::clone(&barrier);
        let done_ref = Arc::clone(&all_done);
        let reads_ref = Arc::clone(&total_reads);

        reader_handles.push(thread::spawn(move || {
            let mut local_rng = ChaCha8Rng::seed_from_u64(0xDEAD_0000 + rid as u64);
            let mut last_epoch = 0u64;
            let mut local_ops = 0usize;
            bar.wait();

            while !done_ref.load(Ordering::Acquire) || local_ops < 2_000 {
                let ep = sys_ref.routing_epoch();
                assert!(
                    ep >= last_epoch,
                    "routing epoch regressed: {ep} < {last_epoch}"
                );
                last_epoch = ep;

                let k = base_ref[local_rng.gen_range(0..base_ref.len())];
                assert!(
                    sys_ref.query_point(k),
                    "FALSE NEGATIVE on concurrent point read key={k} during live split/merge"
                );
                let span = QueryRange::new(k.saturating_sub(5), k.saturating_add(5)).unwrap();
                assert!(
                    sys_ref.query_range(span),
                    "FALSE NEGATIVE on concurrent range read {span:?} during live split/merge"
                );
                local_ops += 2;
            }
            reads_ref.fetch_add(local_ops, Ordering::Relaxed);
        }));
    }

    // Spawn 4 writer threads inserting disjoint + overlapping cross-shard key sets.
    let mut writer_handles = Vec::with_capacity(WRITERS);
    for wid in 0..WRITERS {
        let sys_ref = Arc::clone(&sys);
        let bar = Arc::clone(&barrier);
        writer_handles.push(thread::spawn(move || {
            let mut local_rng = ChaCha8Rng::seed_from_u64(0xF00D_0000 + wid as u64);
            let mut keys: Vec<Key> = Vec::with_capacity(KEYS_PER_WRITER);
            // Half disjoint per-writer band, half interleaved across [0, 400_000) to hit split boundaries!
            for i in 0..(KEYS_PER_WRITER / 2) {
                let k = (wid as u64) * 100_000 + 20_000 + (i as u64) * 13 + (wid as u64);
                keys.push(k);
            }
            for i in 0..(KEYS_PER_WRITER - KEYS_PER_WRITER / 2) {
                let k = 400_000 + (i as u64) * 100 + (wid as u64) * 17 + 1;
                keys.push(k);
            }
            keys.shuffle(&mut local_rng);

            bar.wait();

            let mut acknowledged = Vec::with_capacity(keys.len());
            for &k in &keys {
                let _sid = sys_ref.insert(k).expect("concurrent insert");
                assert!(
                    sys_ref.query_point(k),
                    "immediate read-after-write false negative for key={k}"
                );
                acknowledged.push(k);
            }
            acknowledged
        }));
    }

    // Coordinator thread runs live splits and live merges concurrently with readers & writers!
    let sys_coord = Arc::clone(&sys);
    let writers_done_coord = Arc::clone(&writers_done);
    let coord_handle = thread::spawn(move || {
        barrier.wait();
        let mut live_splits = 0usize;
        let mut live_merges = 0usize;
        let mut rounds = 0usize;

        while !writers_done_coord.load(Ordering::Acquire) || rounds < 6 {
            // Rebalance (triggers splits on growing shards).
            let reb = sys_coord
                .rebalance_if_needed()
                .expect("concurrent rebalance");
            live_splits += reb.splits.len();
            live_merges += reb.merges.len();

            // Also proactively exercise explicit live split on the heaviest shard.
            if sys_coord.shard_count() < 12 {
                let snaps = sys_coord.shard_telemetry_snapshots();
                if let Some(busiest) = snaps.iter().max_by_key(|s| s.key_count) {
                    if busiest.key_count >= 20 && sys_coord.split_shard(busiest.shard_id).is_ok() {
                        live_splits += 1;
                    }
                }
            }

            // Exercise explicit live merge on the two lightest adjacent shards when >= 5 shards exist.
            if sys_coord.shard_count() >= 5 {
                let snaps = sys_coord.shard_telemetry_snapshots();
                if let Some(pair) = snaps
                    .windows(2)
                    .min_by_key(|w| w[0].key_count + w[1].key_count)
                {
                    if sys_coord
                        .merge_adjacent_shards(pair[0].shard_id, pair[1].shard_id)
                        .is_ok()
                    {
                        live_merges += 1;
                    }
                }
            }

            rounds += 1;
            thread::yield_now();
        }
        (live_splits, live_merges)
    });

    // Wait for writers to finish, signal coordinator, then signal readers.
    let mut all_acknowledged: BTreeSet<Key> = baseline;
    for h in writer_handles {
        let writer_keys = h.join().expect("writer thread panicked");
        all_acknowledged.extend(writer_keys);
    }
    writers_done.store(true, Ordering::Release);
    let (live_splits, live_merges) = coord_handle.join().expect("coordinator thread panicked");
    all_done.store(true, Ordering::Release);
    for h in reader_handles {
        h.join().expect("reader thread panicked");
    }

    assert!(
        live_splits >= 2,
        "expected multiple live splits, got {live_splits}"
    );
    assert!(
        live_merges >= 1,
        "expected at least one live merge, got {live_merges}"
    );
    assert!(total_reads.load(Ordering::Relaxed) >= READERS * 2_000);
    assert!(sys.routing_epoch() > 1);

    // CRITICAL ASSERTION: 100% of inserted keys MUST return `true` (0 lost writes, 0 FNs)!
    for &k in &all_acknowledged {
        assert!(
            sys.query_point(k),
            "LOST WRITE / FALSE NEGATIVE for acknowledged key={k}"
        );
        let r = QueryRange::new(k.saturating_sub(2), k.saturating_add(2)).unwrap();
        assert!(
            sys.query_range(r),
            "LOST WRITE / FALSE NEGATIVE on range {r:?} for acknowledged key={k}"
        );
    }

    sys.verify_all_invariants()
        .expect("post-concurrency full routing + filter + Reverse Map invariants");
    assert_eq!(sys.total_keys(), all_acknowledged.len());
}

/// Gate 5: Boundary keys (`0`, `1`, `u64::MAX - 1`, `u64::MAX`) and strict error-path contracts.
#[test]
fn gate5_error_paths_and_u64_boundary_invariants() {
    let dir = TempDir::new().expect("temp dir");
    let base_cfg = make_filter_config(8, 4);

    // Invalid ShardingConfig rejections.
    assert!(matches!(
        ShardingConfig::new(base_cfg.clone())
            .with_min_shards(0)
            .validate(),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        ShardingConfig::new(base_cfg.clone())
            .with_min_shards(4)
            .with_max_shards(2)
            .validate(),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        ShardingConfig::new(base_cfg.clone())
            .with_split_load_threshold(0.50)
            .with_merge_load_threshold(0.60)
            .validate(),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        ShardingConfig::new(base_cfg.clone())
            .with_split_load_threshold(f64::NAN)
            .validate(),
        Err(AerisError::InvalidConfig(_))
    ));

    let cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(3)
        .with_min_shards(2)
        .with_max_shards(4)
        .with_reverse_map_config(lean_rm_config())
        .with_fp_log_config(fast_fp_log_config());

    let sys = ShardedAerisSystem::open_with_boundaries(dir.path(), cfg, &[0, u64::MAX - 1])
        .expect("open extreme boundary shards");

    // Shard 0 is single-key [0, 0] -> splitting it MUST fail with InvalidConfig!
    let bounds = sys.shard_bounds();
    assert_eq!(bounds[0].1, ShardBounds::new(0, 0).unwrap());
    assert_eq!(bounds[2].1, ShardBounds::new(u64::MAX, u64::MAX).unwrap());
    assert!(matches!(
        sys.split_shard(bounds[0].0),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        sys.split_shard(999_999),
        Err(AerisError::ShardNotFound(999_999))
    ));

    // Merging non-adjacent or reversed shards MUST fail with InvalidConfig!
    assert!(matches!(
        sys.merge_adjacent_shards(bounds[0].0, bounds[2].0),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        sys.merge_adjacent_shards(bounds[1].0, bounds[0].0),
        Err(AerisError::InvalidConfig(_))
    ));

    // Insert extreme u64 boundary keys, split middle shard, merge back, verify zero FNs.
    let extreme_keys = [0u64, 1, 2, u64::MAX - 2, u64::MAX - 1, u64::MAX];
    for &k in &extreme_keys {
        sys.insert(k).expect("insert extreme key");
        assert!(sys.query_point(k));
    }

    let split_rep = sys.split_shard(bounds[1].0).expect("split middle shard");
    assert_eq!(sys.shard_count(), 4);
    // Now at max_shards == 4 -> another split MUST fail!
    assert!(matches!(
        sys.split_shard(split_rep.left_shard_id),
        Err(AerisError::InvalidConfig(_))
    ));

    sys.merge_adjacent_shards(split_rep.left_shard_id, split_rep.right_shard_id)
        .expect("merge back middle shards");
    assert_eq!(sys.shard_count(), 3);

    for &k in &extreme_keys {
        assert!(sys.query_point(k), "extreme boundary key {k} lost");
    }
    sys.verify_all_invariants()
        .expect("extreme boundary invariants");

    // Delete extreme keys and verify clean removal.
    for &k in &extreme_keys {
        assert!(sys.delete(k).expect("delete extreme key"));
    }
    assert_eq!(sys.total_keys(), 0);
    assert!(!sys.query_range(QueryRange::new(0, u64::MAX).unwrap()));
}

/// Gate 6: Cross-boundary false-positive monotonicity across both shard merges
/// (range straddling `[left.max, right.min]` before merge) and shard splits
/// (adapted ranges split across children).
#[test]
fn gate6_cross_boundary_fp_monotonicity_across_split_and_merge() {
    let dir = TempDir::new().expect("temp dir");
    let base_cfg = make_filter_config(8, 4); // 256 slots, m = 4 (P = 16), r = 10
    let cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(2)
        .with_min_shards(1)
        .with_max_shards(8)
        .with_reverse_map_config(lean_rm_config())
        .with_fp_log_config(fast_fp_log_config());

    // Place shard boundary mid-partition: 1_000_005 = partition 62_500, memento 5!
    // That forces cross-boundary ranges [1_000_000, 1_000_015] to share the SAME partition
    // across both shards, rigorously exercising clamped FP replay across merge!
    let split_boundary = 1_000_005u64;
    let sys = ShardedAerisSystem::open_with_boundaries(dir.path(), cfg, &[split_boundary])
        .expect("open 2-shard system");

    let mut rng = ChaCha8Rng::seed_from_u64(0x4D34_0006);
    let mut live_keys = BTreeSet::new();
    // Populate both shards away from the immediate boundary neighborhood so boundary ranges are empty.
    while live_keys.len() < 120 {
        live_keys.insert(rng.gen_range(0..900_000u64));
    }
    while live_keys.len() < 240 {
        live_keys.insert(rng.gen_range(1_100_000..2_000_000u64));
    }
    for &k in &live_keys {
        sys.insert(k).expect("insert key");
    }

    // Explicitly adapt a cross-boundary range straddling `split_boundary` PLUS discovered FPs.
    let mut tracked_empty = Vec::new();
    let cross_boundary = QueryRange::new(split_boundary - 5, split_boundary + 10).unwrap();
    sys.adapt_range(cross_boundary)
        .expect("adapt cross-boundary range");
    assert!(!sys.query_range(cross_boundary));
    tracked_empty.push(cross_boundary);

    for _ in 0..80_000u64 {
        let start = rng.gen_range(900_010..1_099_900u64);
        let r = QueryRange::new(start, start + 15).unwrap();
        if sys.query_range(r) {
            sys.adapt_range(r).expect("adapt empty FP");
            assert!(!sys.query_range(r));
            tracked_empty.push(r);
            if tracked_empty.len() >= 12 {
                break;
            }
        }
    }
    assert!(
        tracked_empty.len() >= 5,
        "expected >= 5 genuine FPs before merge/split cycle, found {}",
        tracked_empty.len()
    );

    // 1. Merge the 2 shards into 1 -> verify 0 FNs and 0 repeated FPs (including cross_boundary!).
    let bounds = sys.shard_bounds();
    let merge_rep = sys
        .merge_adjacent_shards(bounds[0].0, bounds[1].0)
        .expect("merge into single shard");
    assert_eq!(sys.shard_count(), 1);
    for &k in &live_keys {
        assert!(sys.query_point(k));
    }
    for &r in &tracked_empty {
        assert!(
            !sys.query_range(r),
            "cross-boundary FP {r:?} regressed after merge"
        );
    }

    // 2. Split the merged shard back into 2 at its data median -> verify 0 FNs and 0 repeated FPs!
    let split_rep = sys
        .split_shard(merge_rep.merged_shard_id)
        .expect("split merged shard");
    assert_eq!(sys.shard_count(), 2);
    assert!(split_rep.left_keys.abs_diff(split_rep.right_keys) <= 1);
    for &k in &live_keys {
        assert!(sys.query_point(k));
    }
    for &r in &tracked_empty {
        assert!(
            !sys.query_range(r),
            "adapted empty range {r:?} regressed after split"
        );
    }
    sys.verify_all_invariants().expect("final invariants");
}
