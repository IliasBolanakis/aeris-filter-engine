//! Concurrency, RCU Swap, Dynamic Sharding, and Multi-Core Scaling
//! Verification Suite.
//!
//! # Testing Contract
//! - Uses 100% real Data Plane ([`AerisFilter`]), Storage Plane
//!   ([`RocksDbReverseMap`]), Control Plane ([`FalsePositiveLog`]), and Scale Plane
//!   ([`RcuAerisFilter`], [`ShardedAerisSystem`]) implementations — zero mocks/stubs.
//! - Uses strictly deterministic concurrency synchronization (`Barrier`, `AtomicBool`,
//!   `AtomicU64`, `AtomicUsize`) — zero `thread::sleep()`.
//! - Must pass cleanly in both debug and `--release` profiles.

mod common;

use aeris_control::{FalsePositiveLog, FpLogConfig, IoThrottleConfig, RebuildReason};
use aeris_core::AerisFilter;
use aeris_scale::{RcuAerisFilter, ShardedAerisSystem, ShardingConfig};
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{AerisError, FilterConfig, Key, QueryRange, ReverseMapStore, ShardId};
use common::{enclosing_range, init_test_tracing, LatencySummary};
use parking_lot::RwLock;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tracing::info;

/// Keyspace partition boundaries guaranteeing zero overlap between live keys and
/// permanently-empty ranges used to verify 100% FP monotonicity under concurrency.
const RCU_SEED_KEY_MAX: u64 = 1_000_000_000;
const RCU_WRITER_BASE: u64 = 1_200_000_000;
const RCU_WRITER_STRIDE: u64 = 200_000_000;
const RCU_EMPTY_ZONE_BASE: u64 = 3_000_000_000;
const RCU_EMPTY_ZONE_STRIDE: u64 = 200_000_000;

/// Reserved permanently-empty band in `sharded_concurrent_hotspot_splits_and_merges_stress`
/// for concurrent `adapt_range` testing alongside live splits and merges.
const SHARDED_EMPTY_ZONE_START: u64 = 50_000_000_000;
const SHARDED_EMPTY_ZONE_END: u64 = 51_000_000_000;

#[inline]
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Generates `count` unique keys strictly inside `[min_k, max_k]` with a fixed
/// 3-bit low tag `tag in 0..8` so disjoint threads can never produce duplicate keys.
fn generate_tagged_keys(seed: u64, min_k: u64, max_k: u64, tag: u64, count: usize) -> Vec<Key> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut seen = HashSet::with_capacity(count);
    let mut keys = Vec::with_capacity(count);
    let span = max_k - min_k;

    while keys.len() < count {
        let raw = min_k + (rng.gen::<u64>() % span.max(1));
        let tagged = (raw & !7u64) | (tag & 7u64);
        if tagged >= min_k
            && tagged <= max_k
            && !(SHARDED_EMPTY_ZONE_START..=SHARDED_EMPTY_ZONE_END).contains(&tagged)
            && seen.insert(tagged)
        {
            keys.push(tagged);
        }
    }
    keys
}

/// Stress-tests 8 concurrent reader threads, 3 concurrent insert/adapt writer threads,
/// and 1 live swap coordinator executing repeated `expand_rcu` and dynamic-`R`
/// asynchronous rebuild swaps (`F_old -> F_new`) over real `AerisFilter`,
/// `RocksDbReverseMap`, `FalsePositiveLog`, and `RcuAerisFilter`.
///
/// Verifies:
/// - **0 false negatives** across all seed and writer-acknowledged keys (point & range).
/// - **100% false-positive monotonicity** across concurrent adaptations, `2x` RCU
///   expansions, and dynamic-`R` swaps (`m: 5 -> 8 -> 6 -> 7 -> 5`).
/// - **Strictly monotonic epoch progression** and full structural invariants.
#[test]
fn rcu_concurrent_readers_writers_and_swaps_stress() {
    init_test_tracing();

    const NUM_READERS: usize = 8;
    const NUM_WRITERS: usize = 3;
    const SEED_KEYS_COUNT: usize = 512;
    const WRITER_INSERTS_EACH: usize = 220;
    const WRITER_EMPTY_PROBES_EACH: usize = 4_500;

    let temp_dir = TempDir::new().expect("create temp dir for RCU stress test");
    let rm_gen0 = Arc::new(
        RocksDbReverseMap::open(
            temp_dir.path().join("rm_gen_0"),
            ReverseMapConfig::default(),
        )
        .expect("open initial RocksDbReverseMap"),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(
            temp_dir.path().join("fp.log"),
            FpLogConfig {
                max_entries: 16_384,
                sync_on_append: false,
            },
        )
        .expect("open FalsePositiveLog"),
    );

    // eps = 1/16 -> r_min = 7; fixed payload budget C = 15 allows m in {5, 6, 7, 8}
    // (R = 32, 64, 128, 256) with r in {10, 9, 8, 7} >= r_min while generating genuine
    // FPs for concurrent adaptation and 100% monotonicity verification across RCU swaps.
    let initial_cfg =
        FilterConfig::new(10, 10, 32, 15, 1.0 / 16.0).expect("valid initial RCU FilterConfig");

    let seed_keys: Arc<Vec<Key>> = Arc::new(generate_tagged_keys(
        0xAEE1_5001,
        1_000,
        RCU_SEED_KEY_MAX,
        0,
        SEED_KEYS_COUNT,
    ));

    let rm_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&rm_gen0) as _;
    let initial_filter =
        AerisFilter::bulk_load(initial_cfg, &seed_keys, rm_dyn.as_ref()).expect("bulk_load seed");
    let rcu = Arc::new(RcuAerisFilter::new(0, initial_filter, rm_dyn));
    let initial_epoch = rcu.epoch();

    let confirmed_keys: Arc<RwLock<Vec<Key>>> = Arc::new(RwLock::new(Vec::with_capacity(
        NUM_WRITERS * WRITER_INSERTS_EACH,
    )));
    let adapted_fps: Arc<RwLock<Vec<QueryRange>>> = Arc::new(RwLock::new(Vec::new()));

    let start_barrier = Arc::new(Barrier::new(NUM_READERS + NUM_WRITERS + 1));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let writers_done = Arc::new(AtomicUsize::new(0));
    let total_reader_queries = Arc::new(AtomicU64::new(0));
    let total_reader_fn = Arc::new(AtomicU64::new(0));
    let total_reader_repeated_fps = Arc::new(AtomicU64::new(0));
    let reader_latencies: Arc<RwLock<Vec<u64>>> = Arc::new(RwLock::new(Vec::new()));

    let test_start = Instant::now();

    // Spawn 8 continuous zero-stall reader threads.
    let mut reader_handles = Vec::with_capacity(NUM_READERS);
    for reader_id in 0..NUM_READERS {
        let rcu = Arc::clone(&rcu);
        let seed_keys = Arc::clone(&seed_keys);
        let confirmed_keys = Arc::clone(&confirmed_keys);
        let adapted_fps = Arc::clone(&adapted_fps);
        let barrier = Arc::clone(&start_barrier);
        let stop = Arc::clone(&stop_flag);
        let q_counter = Arc::clone(&total_reader_queries);
        let fn_counter = Arc::clone(&total_reader_fn);
        let rfp_counter = Arc::clone(&total_reader_repeated_fps);
        let lat_sink = Arc::clone(&reader_latencies);

        reader_handles.push(thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(0x9EAD_0000 + reader_id as u64);
            let mut local_queries = 0u64;
            let mut local_fn = 0u64;
            let mut local_rfp = 0u64;
            let mut local_lat_ns = Vec::with_capacity(256);

            barrier.wait();

            while !stop.load(Ordering::Acquire) {
                let batch_start = Instant::now();

                // 32 point + enclosing range queries against known-present seed keys.
                for _ in 0..16 {
                    let k = seed_keys[rng.gen_range(0..seed_keys.len())];
                    if !rcu.query_point(k) {
                        local_fn += 1;
                    }
                    if !rcu.query_range(enclosing_range(k, 12)) {
                        local_fn += 1;
                    }
                    local_queries += 2;
                }

                // Sample recently confirmed writer-inserted keys for 0-FN guarantee.
                {
                    let live = confirmed_keys.read();
                    if !live.is_empty() {
                        for _ in 0..8 {
                            let k = live[rng.gen_range(0..live.len())];
                            if !rcu.query_point(k) {
                                local_fn += 1;
                            }
                            if !rcu.query_range(enclosing_range(k, 8)) {
                                local_fn += 1;
                            }
                            local_queries += 2;
                        }
                    }
                }

                // Verify 100% FP monotonicity on all adapted permanently-empty intervals.
                {
                    let fps = adapted_fps.read();
                    for &fp_range in fps.iter() {
                        if rcu.query_range(fp_range) {
                            local_rfp += 1;
                        }
                        local_queries += 1;
                    }
                }

                if local_lat_ns.len() < 256 {
                    let per_q = (batch_start.elapsed().as_nanos() as u64) / 32;
                    local_lat_ns.push(per_q);
                }
            }

            q_counter.fetch_add(local_queries, Ordering::Relaxed);
            fn_counter.fetch_add(local_fn, Ordering::Relaxed);
            rfp_counter.fetch_add(local_rfp, Ordering::Relaxed);
            lat_sink.write().extend(local_lat_ns);
        }));
    }

    // Spawn 3 concurrent writer threads inserting disjoint keys and adapting discovered FPs.
    let mut writer_handles = Vec::with_capacity(NUM_WRITERS);
    for writer_id in 0..NUM_WRITERS {
        let rcu = Arc::clone(&rcu);
        let fp_log = Arc::clone(&fp_log);
        let confirmed_keys = Arc::clone(&confirmed_keys);
        let adapted_fps = Arc::clone(&adapted_fps);
        let barrier = Arc::clone(&start_barrier);
        let done_counter = Arc::clone(&writers_done);

        writer_handles.push(thread::spawn(move || {
            let min_k = RCU_WRITER_BASE + (writer_id as u64) * RCU_WRITER_STRIDE;
            let max_k = min_k + RCU_WRITER_STRIDE - 16;
            let keys_to_insert = generate_tagged_keys(
                0xC117_0000 + writer_id as u64,
                min_k,
                max_k,
                (writer_id + 1) as u64,
                WRITER_INSERTS_EACH,
            );

            let empty_min = RCU_EMPTY_ZONE_BASE + (writer_id as u64) * RCU_EMPTY_ZONE_STRIDE;
            let empty_max = empty_min + RCU_EMPTY_ZONE_STRIDE - 512;
            let mut rng = ChaCha8Rng::seed_from_u64(0xE000_1000 + writer_id as u64);
            let mut local_adapted_set = HashSet::new();
            let probes_per_insert = WRITER_EMPTY_PROBES_EACH / WRITER_INSERTS_EACH.max(1);

            barrier.wait();

            for &key in &keys_to_insert {
                rcu.insert(key).expect("concurrent RCU insert must succeed");
                assert!(
                    rcu.query_point(key),
                    "immediate false negative for newly inserted key {key}"
                );
                confirmed_keys.write().push(key);

                for _ in 0..probes_per_insert {
                    let start = rng.gen_range(empty_min..empty_max);
                    let range = QueryRange::with_len(start, 32).expect("valid empty range");
                    if local_adapted_set.contains(&range) {
                        continue;
                    }
                    if rcu.query_range(range) {
                        // Record into persistent FP Log FIRST, then adapt active RCU generation.
                        fp_log
                            .record(range, current_unix_ms())
                            .expect("FP log record failed");
                        rcu.adapt(range).expect("concurrent RCU adapt must succeed");
                        assert!(
                            !rcu.query_range(range),
                            "adapted range [{}, {}] immediately returned FP",
                            range.start(),
                            range.end()
                        );
                        local_adapted_set.insert(range);
                        adapted_fps.write().push(range);
                    }
                }
            }

            done_counter.fetch_add(1, Ordering::Release);
        }));
    }

    // Spawn 1 live swap coordinator executing repeated `expand_rcu` and dynamic-`R` swaps.
    let swap_rcu = Arc::clone(&rcu);
    let swap_fp_log = Arc::clone(&fp_log);
    let swap_barrier = Arc::clone(&start_barrier);
    let swap_writers_done = Arc::clone(&writers_done);
    let swap_stop = Arc::clone(&stop_flag);
    let swap_root = temp_dir.path().to_path_buf();

    let swap_handle = thread::spawn(move || {
        swap_barrier.wait();

        let mut observed_epochs = vec![swap_rcu.epoch()];
        let mut expand_swaps = 0usize;
        let mut dynamic_r_swaps = 0usize;
        let mut rm_gen = 1u64;

        let schedule: &[(bool, u8)] = &[
            (true, 0),  // expand_rcu
            (false, 8), // dynamic R: m=5 -> m=8 (R=256)
            (false, 6), // dynamic R: m=8 -> m=6 (R=64)
            (true, 0),  // expand_rcu
            (false, 7), // dynamic R: m=6 -> m=7 (R=128)
            (false, 5), // dynamic R: m=7 -> m=5 (R=32)
        ];

        for &(is_expand, target_m) in schedule {
            let prev_epoch = swap_rcu.epoch();
            if is_expand {
                let rep = swap_rcu
                    .expand_rcu()
                    .expect("concurrent expand_rcu must succeed");
                assert!(
                    rep.new_epoch > prev_epoch,
                    "expand_rcu epoch did not advance: {} -> {}",
                    prev_epoch,
                    rep.new_epoch
                );
                observed_epochs.push(rep.new_epoch);
                expand_swaps += 1;
            } else {
                let current_cfg = swap_rcu.config();
                let target_cfg = current_cfg
                    .reconfigure_for_m(target_m)
                    .expect("reconfigure_for_m must satisfy budget C=15 and r_min=7");
                let dest_rm: Arc<dyn ReverseMapStore> = Arc::new(
                    RocksDbReverseMap::open(
                        swap_root.join(format!("rm_gen_{rm_gen}")),
                        ReverseMapConfig::default(),
                    )
                    .expect("open destination RocksDbReverseMap"),
                );
                rm_gen += 1;

                let rep = swap_rcu
                    .rebuild_async_and_swap(
                        dest_rm,
                        Arc::clone(&swap_fp_log),
                        target_cfg,
                        RebuildReason::DynamicRangeChange {
                            old_m: current_cfg.m,
                            new_m: target_m,
                        },
                        IoThrottleConfig::default(),
                    )
                    .expect("concurrent dynamic-R rebuild_async_and_swap must succeed");
                assert!(
                    rep.new_epoch > prev_epoch,
                    "dynamic-R swap epoch did not advance: {} -> {}",
                    prev_epoch,
                    rep.new_epoch
                );
                observed_epochs.push(rep.new_epoch);
                dynamic_r_swaps += 1;
            }
            thread::yield_now();
        }

        // Wait deterministically for all writer threads to finish (zero thread::sleep).
        while swap_writers_done.load(Ordering::Acquire) < NUM_WRITERS {
            thread::yield_now();
        }
        swap_stop.store(true, Ordering::Release);

        (observed_epochs, expand_swaps, dynamic_r_swaps)
    });

    for h in writer_handles {
        h.join().expect("writer thread panicked");
    }
    let (observed_epochs, expand_swaps, dynamic_r_swaps) =
        swap_handle.join().expect("swap coordinator panicked");
    for h in reader_handles {
        h.join().expect("reader thread panicked");
    }

    let elapsed_secs = test_start.elapsed().as_secs_f64().max(1e-9);
    let final_epoch = rcu.epoch();
    let total_queries = total_reader_queries.load(Ordering::Relaxed);
    let reader_fn = total_reader_fn.load(Ordering::Relaxed);
    let reader_rfp = total_reader_repeated_fps.load(Ordering::Relaxed);

    assert_eq!(reader_fn, 0, "readers observed {reader_fn} false negatives");
    assert_eq!(
        reader_rfp, 0,
        "readers observed {reader_rfp} repeated false positives (monotonicity violated)"
    );
    assert!(
        final_epoch > initial_epoch,
        "final epoch ({final_epoch}) must exceed initial epoch ({initial_epoch})"
    );
    for pair in observed_epochs.windows(2) {
        assert!(
            pair[1] > pair[0],
            "epoch progression was not strictly monotonic: {} -> {}",
            pair[0],
            pair[1]
        );
    }

    // Post-quiescence exhaustive check of 100% of seed + writer keys (0 FNs).
    let final_confirmed = confirmed_keys.read().clone();
    assert_eq!(final_confirmed.len(), NUM_WRITERS * WRITER_INSERTS_EACH);
    for &k in seed_keys.iter().chain(final_confirmed.iter()) {
        assert!(
            rcu.query_point(k),
            "post-stress false negative on query_point({k})"
        );
        assert!(
            rcu.query_range(enclosing_range(k, 16)),
            "post-stress false negative on enclosing range for {k}"
        );
    }

    // Post-quiescence exhaustive check of 100% FP monotonicity.
    let final_fps = adapted_fps.read().clone();
    assert!(
        !final_fps.is_empty(),
        "expected at least 1 real FP adaptation during RCU stress run"
    );
    for &fp_range in &final_fps {
        assert!(
            !rcu.query_range(fp_range),
            "post-stress repeated FP on adapted range [{}, {}]",
            fp_range.start(),
            fp_range.end()
        );
    }

    rcu.verify_invariants()
        .expect("RCU filter structural invariants must hold post-stress");

    let lat_summary = LatencySummary::from_samples(reader_latencies.read().clone());
    let qps = (total_queries as f64) / elapsed_secs;
    info!(
        test = "rcu_concurrent_readers_writers_and_swaps_stress",
        readers = NUM_READERS,
        writers = NUM_WRITERS,
        total_keys = seed_keys.len() + final_confirmed.len(),
        adapted_fps = final_fps.len(),
        expand_swaps,
        dynamic_r_swaps,
        initial_epoch,
        final_epoch,
        total_reader_queries = total_queries,
        reader_qps = format!("{qps:.0}"),
        p50_ns = lat_summary.p50_ns,
        p99_ns = lat_summary.p99_ns,
        false_negatives = 0,
        repeated_fps = 0,
        "RCU concurrent stress passed"
    );
}

/// Stress-tests 4 readers, 4 writers (2 skewed hot-spot + 2 uniform), and 1 live
/// topology coordinator invoking `rebalance_if_needed()`, `split_shard()`, and
/// `merge_adjacent_shards()` concurrently on a real [`ShardedAerisSystem`].
///
/// Verifies:
/// - **0 lost writes** and **0 false negatives** across 100% of acknowledged inserts.
/// - **Exact key conservation** (`system.total_keys() == all_unique_keys.len()`).
/// - **Contiguous `[0, u64::MAX]` routing** at every point and full
///   [`ShardedAerisSystem::verify_all_invariants`].
#[test]
fn sharded_concurrent_hotspot_splits_and_merges_stress() {
    init_test_tracing();

    const NUM_READERS: usize = 4;
    const NUM_WRITERS: usize = 4;
    const SEED_KEYS: usize = 64;
    const HOT_WRITER_INSERTS: usize = 260;
    const UNIFORM_WRITER_INSERTS: usize = 90;

    let temp_dir = TempDir::new().expect("create temp dir for sharded stress test");
    let base_cfg =
        FilterConfig::new(8, 12, 32, 17, 1.0 / 256.0).expect("valid sharded base FilterConfig");

    let sharding_cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(4)
        .with_min_shards(2)
        .with_max_shards(24)
        .with_split_load_threshold(0.60)
        .with_split_min_keys(40)
        .with_split_key_threshold(Some(100))
        .with_max_shard_q(9)
        .with_merge_load_threshold(0.25)
        .with_fp_log_config(FpLogConfig {
            max_entries: 4_096,
            sync_on_append: false,
        });

    let system = Arc::new(
        ShardedAerisSystem::open(temp_dir.path(), sharding_cfg).expect("open ShardedAerisSystem"),
    );

    // Seed baseline keys spread across the 64-bit key space (low 3-bit tag = 0).
    let seed_keys: Arc<Vec<Key>> = Arc::new(generate_tagged_keys(
        0x5A00_0001,
        1_000,
        u64::MAX - 1_000,
        0,
        SEED_KEYS,
    ));
    for &k in seed_keys.iter() {
        system
            .insert(k)
            .expect("seed insert into ShardedAerisSystem");
    }

    let acked_keys: Arc<RwLock<Vec<Key>>> = Arc::new(RwLock::new(Vec::with_capacity(
        2 * HOT_WRITER_INSERTS + 2 * UNIFORM_WRITER_INSERTS,
    )));
    let start_barrier = Arc::new(Barrier::new(NUM_READERS + NUM_WRITERS + 1));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let writers_done = Arc::new(AtomicUsize::new(0));
    let reader_queries = Arc::new(AtomicU64::new(0));
    let reader_fn = Arc::new(AtomicU64::new(0));

    let started_at = Instant::now();

    // Spawn 4 concurrent reader threads continuously verifying 0 FNs and contiguous routing.
    let mut reader_handles = Vec::with_capacity(NUM_READERS);
    for reader_id in 0..NUM_READERS {
        let sys = Arc::clone(&system);
        let seeds = Arc::clone(&seed_keys);
        let acked = Arc::clone(&acked_keys);
        let barrier = Arc::clone(&start_barrier);
        let stop = Arc::clone(&stop_flag);
        let q_cnt = Arc::clone(&reader_queries);
        let fn_cnt = Arc::clone(&reader_fn);

        reader_handles.push(thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(0x52ED_0000 + reader_id as u64);
            let mut local_q = 0u64;
            let mut local_fn = 0u64;

            barrier.wait();

            while !stop.load(Ordering::Acquire) {
                for _ in 0..16 {
                    let k = seeds[rng.gen_range(0..seeds.len())];
                    if !sys.query_point(k) || !sys.query_range(enclosing_range(k, 15)) {
                        local_fn += 1;
                    }
                    local_q += 2;
                }

                {
                    let live = acked.read();
                    if !live.is_empty() {
                        for _ in 0..16 {
                            let k = live[rng.gen_range(0..live.len())];
                            if !sys.query_point(k) || !sys.query_range(enclosing_range(k, 15)) {
                                local_fn += 1;
                            }
                            local_q += 2;
                        }
                    }
                }

                sys.verify_routing_invariants()
                    .expect("routing invariants violated during live split/merge");
            }

            q_cnt.fetch_add(local_q, Ordering::Relaxed);
            fn_cnt.fetch_add(local_fn, Ordering::Relaxed);
        }));
    }

    // Spawn 4 writers: 2 skewed hot-spot writers targeting Shard 0 + 2 uniform writers.
    let mut writer_handles = Vec::with_capacity(NUM_WRITERS);
    for writer_id in 0..NUM_WRITERS {
        let sys = Arc::clone(&system);
        let acked = Arc::clone(&acked_keys);
        let barrier = Arc::clone(&start_barrier);
        let done = Arc::clone(&writers_done);

        writer_handles.push(thread::spawn(move || {
            let tag = (writer_id + 1) as u64;
            let keys = if writer_id < 2 {
                // Concentrated hot-spot band inside Shard 0 driving repeated median splits.
                generate_tagged_keys(
                    0x4075_0000 + writer_id as u64,
                    1_000_000,
                    2_000_000,
                    tag,
                    HOT_WRITER_INSERTS,
                )
            } else {
                // Sparse uniform writes across the upper 75% of the keyspace.
                generate_tagged_keys(
                    0x01F0_0000 + writer_id as u64,
                    1u64 << 62,
                    u64::MAX - 1_000,
                    tag,
                    UNIFORM_WRITER_INSERTS,
                )
            };

            let mut rng = ChaCha8Rng::seed_from_u64(0xAD40_0000 + writer_id as u64);

            barrier.wait();

            for (idx, &k) in keys.iter().enumerate() {
                sys.insert(k)
                    .expect("concurrent sharded insert must succeed");
                assert!(
                    sys.query_point(k),
                    "immediate false negative on sharded insert {k}"
                );
                acked.write().push(k);

                // Periodically probe and adapt any FP in the permanently-empty zone.
                if idx.is_multiple_of(16) {
                    let start =
                        rng.gen_range(SHARDED_EMPTY_ZONE_START..(SHARDED_EMPTY_ZONE_END - 64));
                    let probe = QueryRange::with_len(start, 32).expect("valid empty probe");
                    if sys.query_range(probe) {
                        let _ = sys
                            .adapt_range(probe)
                            .expect("concurrent adapt_range failed");
                    }
                }
            }

            done.fetch_add(1, Ordering::Release);
        }));
    }

    // Spawn 1 live topology coordinator driving rebalance_if_needed, split_shard, and merge_adjacent_shards.
    let topo_sys = Arc::clone(&system);
    let topo_barrier = Arc::clone(&start_barrier);
    let topo_writers_done = Arc::clone(&writers_done);
    let topo_stop = Arc::clone(&stop_flag);

    let topo_handle = thread::spawn(move || {
        topo_barrier.wait();

        let mut total_splits = 0usize;
        let mut total_merges = 0usize;
        let mut explicit_splits = 0usize;
        let mut explicit_merges = 0usize;
        let mut round = 0usize;

        while topo_writers_done.load(Ordering::Acquire) < NUM_WRITERS {
            let reb = topo_sys
                .rebalance_if_needed()
                .expect("live rebalance_if_needed failed");
            total_splits += reb.splits.len();
            total_merges += reb.merges.len();

            if round.is_multiple_of(2) {
                if let Some(heavy_id) = select_explicit_split_candidate(&topo_sys) {
                    match topo_sys.split_shard(heavy_id) {
                        Ok(_) => {
                            total_splits += 1;
                            explicit_splits += 1;
                        }
                        Err(AerisError::InvalidConfig(_) | AerisError::ShardNotFound(_)) => {}
                        Err(err) => panic!("unexpected split_shard error: {err}"),
                    }
                }
            } else if let Some((left_id, right_id)) = select_explicit_merge_candidate(&topo_sys) {
                match topo_sys.merge_adjacent_shards(left_id, right_id) {
                    Ok(_) => {
                        total_merges += 1;
                        explicit_merges += 1;
                    }
                    Err(AerisError::InvalidConfig(_) | AerisError::ShardNotFound(_)) => {}
                    Err(err) => panic!("unexpected merge_adjacent_shards error: {err}"),
                }
            }

            topo_sys
                .verify_routing_invariants()
                .expect("routing invariants failed after live topology round");
            round += 1;
            thread::yield_now();
        }

        // Final post-write rebalance + ensure explicit split & merge paths were both exercised.
        let final_reb = topo_sys
            .rebalance_if_needed()
            .expect("final rebalance_if_needed failed");
        total_splits += final_reb.splits.len();
        total_merges += final_reb.merges.len();

        if explicit_splits == 0 {
            if let Some(heavy_id) = select_explicit_split_candidate(&topo_sys) {
                if topo_sys.split_shard(heavy_id).is_ok() {
                    total_splits += 1;
                    explicit_splits += 1;
                }
            }
        }
        if explicit_merges == 0 {
            if let Some((left_id, right_id)) = select_explicit_merge_candidate(&topo_sys) {
                if topo_sys.merge_adjacent_shards(left_id, right_id).is_ok() {
                    total_merges += 1;
                    explicit_merges += 1;
                }
            }
        }

        topo_stop.store(true, Ordering::Release);
        (total_splits, total_merges, explicit_splits, explicit_merges)
    });

    for h in writer_handles {
        h.join().expect("sharded writer panicked");
    }
    let (total_splits, total_merges, explicit_splits, explicit_merges) =
        topo_handle.join().expect("topology coordinator panicked");
    for h in reader_handles {
        h.join().expect("sharded reader panicked");
    }

    let final_acked = acked_keys.read().clone();
    let expected_total_keys = SEED_KEYS + 2 * HOT_WRITER_INSERTS + 2 * UNIFORM_WRITER_INSERTS;
    assert_eq!(
        final_acked.len(),
        2 * HOT_WRITER_INSERTS + 2 * UNIFORM_WRITER_INSERTS
    );
    assert_eq!(
        reader_fn.load(Ordering::Relaxed),
        0,
        "readers observed false negatives during live splits/merges"
    );
    assert!(
        total_splits >= 2 && explicit_splits >= 1,
        "expected >= 2 hotspot splits (>= 1 explicit), got total={total_splits}, explicit={explicit_splits}"
    );
    assert!(
        total_merges >= 1 && explicit_merges >= 1,
        "expected >= 1 adjacent shard merge (>= 1 explicit), got total={total_merges}, explicit={explicit_merges}"
    );

    // Exhaustive zero-lost-write & zero-FN verification across 100% of keys.
    for &k in seed_keys.iter().chain(final_acked.iter()) {
        assert!(
            system.query_point(k),
            "LOST WRITE / false negative for acknowledged key {k}"
        );
        assert!(
            system.query_range(enclosing_range(k, 16)),
            "false negative on enclosing range for acknowledged key {k}"
        );
    }

    assert_eq!(
        system.total_keys(),
        expected_total_keys,
        "exact key conservation violated across live splits/merges"
    );

    // Verify contiguous [0, u64::MAX] routing coverage and all shard QF + ReverseMap invariants.
    let bounds = system.shard_bounds();
    assert!(!bounds.is_empty());
    assert_eq!(bounds.first().expect("non-empty").1.min_key, 0);
    assert_eq!(bounds.last().expect("non-empty").1.max_key, u64::MAX);
    for pair in bounds.windows(2) {
        assert_eq!(
            pair[0].1.max_key.checked_add(1),
            Some(pair[1].1.min_key),
            "routing gap or overlap detected between shards {:?} and {:?}",
            pair[0],
            pair[1]
        );
    }
    system
        .verify_all_invariants()
        .expect("ShardedAerisSystem::verify_all_invariants failed");

    info!(
        test = "sharded_concurrent_hotspot_splits_and_merges_stress",
        total_keys = expected_total_keys,
        final_shards = system.shard_count(),
        routing_epoch = system.routing_epoch(),
        total_splits,
        explicit_splits,
        total_merges,
        explicit_merges,
        reader_queries = reader_queries.load(Ordering::Relaxed),
        elapsed_ms = started_at.elapsed().as_millis(),
        "sharded live split/merge stress passed with 0 lost writes and 0 FNs"
    );
}

fn select_explicit_split_candidate(system: &ShardedAerisSystem) -> Option<ShardId> {
    if system.shard_count() >= system.config().max_shards {
        return None;
    }
    system
        .shard_telemetry_snapshots()
        .into_iter()
        .filter(|s| s.key_count >= 40)
        .max_by_key(|s| s.key_count)
        .map(|s| s.shard_id)
}

fn select_explicit_merge_candidate(system: &ShardedAerisSystem) -> Option<(ShardId, ShardId)> {
    if system.shard_count() <= system.config().min_shards {
        return None;
    }
    let snaps = system.shard_telemetry_snapshots();
    snaps
        .windows(2)
        .filter(|pair| pair[0].key_count + pair[1].key_count <= 200)
        .min_by_key(|pair| pair[0].key_count + pair[1].key_count)
        .map(|pair| (pair[0].shard_id, pair[1].shard_id))
}

/// Benchmarks 1 vs 2 vs 4 vs 8 concurrent reader threads on a seeded 8-shard
/// [`ShardedAerisSystem`] holding 50,000 uniform keys and verifies multi-threaded
/// aggregate read throughput scales cleanly over single-threaded throughput with
/// zero data races, errors, or false negatives.
#[test]
fn multi_core_read_throughput_scaling() {
    init_test_tracing();

    const NUM_KEYS: usize = 50_000;
    const NUM_SHARDS: usize = 8;
    const SEED_THREADS: usize = 8;
    let queries_per_thread: usize = if cfg!(debug_assertions) {
        25_000
    } else {
        200_000
    };

    let temp_dir = TempDir::new().expect("create temp dir for multi-core scaling test");
    // q = 14 (16,384 slots) * 8 shards = 131,072 total slots -> ~38% load factor at 50k keys.
    let base_cfg =
        FilterConfig::new(14, 12, 32, 17, 1.0 / 256.0).expect("valid scaling base FilterConfig");
    let sharding_cfg = ShardingConfig::new(base_cfg)
        .with_initial_shard_count(NUM_SHARDS)
        .with_min_shards(NUM_SHARDS)
        .with_max_shards(NUM_SHARDS)
        .with_reverse_map_config(ReverseMapConfig::for_wal_coupled());

    let system = Arc::new(
        ShardedAerisSystem::open(temp_dir.path(), sharding_cfg)
            .expect("open 8-shard ShardedAerisSystem"),
    );

    // Parallel deterministic seeding partitioned across the 8 shard key intervals.
    let bounds = system.shard_bounds();
    assert_eq!(bounds.len(), NUM_SHARDS);
    let keys_per_shard = NUM_KEYS / SEED_THREADS;
    let mut seed_handles = Vec::with_capacity(SEED_THREADS);

    for (idx, &(_, shard_bounds)) in bounds.iter().enumerate() {
        let sys = Arc::clone(&system);
        seed_handles.push(thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(0x5CA1_E000 + idx as u64);
            let mut local_keys = Vec::with_capacity(keys_per_shard);
            let mut seen = HashSet::with_capacity(keys_per_shard);
            let span = shard_bounds.max_key - shard_bounds.min_key;

            while local_keys.len() < keys_per_shard {
                let k = shard_bounds.min_key + (rng.gen::<u64>() % span);
                if seen.insert(k) {
                    sys.insert(k).expect("parallel seed insert failed");
                    local_keys.push(k);
                }
            }
            local_keys
        }));
    }

    let mut all_keys = Vec::with_capacity(NUM_KEYS);
    for h in seed_handles {
        all_keys.extend(h.join().expect("seed thread panicked"));
    }
    let all_keys = Arc::new(all_keys);
    assert_eq!(system.total_keys(), NUM_KEYS);

    let thread_counts = [1usize, 2, 4, 8];
    let mut results: Vec<(usize, f64, u64)> = Vec::with_capacity(thread_counts.len());

    for &num_threads in &thread_counts {
        // Pre-build each thread's deterministic query workload outside the timed region.
        let mut workloads: Vec<Vec<(QueryRange, bool)>> = Vec::with_capacity(num_threads);
        for tid in 0..num_threads {
            let mut rng =
                ChaCha8Rng::seed_from_u64(0x7000_0000 + (num_threads as u64) * 100 + tid as u64);
            let mut batch = Vec::with_capacity(queries_per_thread);
            for i in 0..queries_per_thread {
                if i.is_multiple_of(4) {
                    let k = all_keys[rng.gen_range(0..all_keys.len())];
                    batch.push((enclosing_range(k, 15), true));
                } else {
                    let start = rng.gen::<u64>().min(u64::MAX - 32);
                    batch.push((QueryRange::with_len(start, 32).expect("valid range"), false));
                }
            }
            workloads.push(batch);
        }

        let start_barrier = Arc::new(Barrier::new(num_threads + 1));
        let end_barrier = Arc::new(Barrier::new(num_threads + 1));
        let false_negatives = Arc::new(AtomicU64::new(0));
        let positive_hits = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::with_capacity(num_threads);
        for workload in workloads {
            let sys = Arc::clone(&system);
            let start_b = Arc::clone(&start_barrier);
            let end_b = Arc::clone(&end_barrier);
            let fn_cnt = Arc::clone(&false_negatives);
            let hit_cnt = Arc::clone(&positive_hits);

            handles.push(thread::spawn(move || {
                let mut local_fn = 0u64;
                let mut local_hits = 0u64;

                start_b.wait();
                for (range, must_be_present) in workload {
                    let res = sys.query_range(range);
                    if must_be_present && !res {
                        local_fn += 1;
                    }
                    if res {
                        local_hits += 1;
                    }
                }
                end_b.wait();

                fn_cnt.fetch_add(local_fn, Ordering::Relaxed);
                hit_cnt.fetch_add(local_hits, Ordering::Relaxed);
            }));
        }

        start_barrier.wait();
        let t0 = Instant::now();
        end_barrier.wait();
        let elapsed = t0.elapsed();

        for h in handles {
            h.join().expect("scaling reader thread panicked");
        }

        assert_eq!(
            false_negatives.load(Ordering::Relaxed),
            0,
            "false negatives observed at {num_threads} threads"
        );
        let total_ops = (num_threads * queries_per_thread) as u64;
        let ops_per_sec = (total_ops as f64) / elapsed.as_secs_f64().max(1e-9);
        results.push((
            num_threads,
            ops_per_sec,
            positive_hits.load(Ordering::Relaxed),
        ));
    }

    let ops_1t = results[0].1;
    let best_multi_ops = results
        .iter()
        .skip(1)
        .map(|&(_, ops, _)| ops)
        .fold(0.0_f64, f64::max);
    let scaling_factor = best_multi_ops / ops_1t.max(1.0);

    for &(threads, ops, hits) in &results {
        info!(
            test = "multi_core_read_throughput_scaling",
            threads,
            ops_per_sec = format!("{ops:.0}"),
            speedup_vs_1t = format!("{:.2}x", ops / ops_1t.max(1.0)),
            positive_hits = hits,
            "read scaling measurement"
        );
    }

    let hw_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if hw_threads >= 2 {
        assert!(
            best_multi_ops > ops_1t * 1.15,
            "expected multi-core read throughput to scale over 1T (1T={ops_1t:.0} ops/s, best_multi={best_multi_ops:.0} ops/s, scaling={scaling_factor:.2}x, hw_threads={hw_threads})"
        );
    }

    system
        .verify_routing_invariants()
        .expect("routing invariants intact after scaling benchmark");
}
