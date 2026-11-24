//! Large-Scale Ablation Suite: A/B 2 (Proactive EWMA Trigger + Contraction vs.
//! Reactive `alpha = 95%` Expansion) and A/B 3 (WAL-Coupled Storage vs. Uncoupled
//! Reverse Map, including unbatched `fsync`, batched WAL commits, and multi-cycle
//! real subprocess `SIGKILL` crash recovery).
//!
//! # Architecture & Execution
//! - Invoked via `cargo test --release --test ablation_proactive_and_storage -- --nocapture`.
//! - Top-level `#[test] fn run_ablation_proactive_and_storage()` dispatches into
//!   the subprocess `SIGKILL` child driver when `AERIS_AB3_SIGKILL_CHILD` is set;
//!   otherwise it orchestrates A/B 2 and A/B 3 sequentially and persists a single
//!   [`ablation_common::WorkerAblationReport`] to
//!   `Filter/target/ablation_worker2_proactive_storage.json`.
//! - Uses 100% real Data, Storage, Control, and Scale Plane components with zero
//!   mocks/stubs and zero `thread::sleep()`.

mod ablation_common;

use ablation_common::{
    compute_delta_pct, directory_size_bytes, enclosing_range, generate_uniform_empty_queries,
    generate_uniform_keys_oracle, init_ablation_tracing, write_worker_report, AblationMetric,
    HostMetadata, LatencyStats, SortedKeyOracle, WorkerAblationReport,
};
use aeris_control::{
    EstimatorConfig, FalsePositiveLog, FpLogConfig, IoThrottleConfig, ProactiveTrigger,
    ProactiveTriggerConfig, RebuildReason, SlidingWindowEstimator, TriggerAction, TriggerInput,
};
use aeris_core::AerisFilter;
use aeris_scale::RcuAerisFilter;
use aeris_storage::{
    encode_wal_frame, CoupledStorageEngine, KvStoreConfig, ReferenceKvStore, ReverseMapConfig,
    RocksDbReverseMap, StorageConfig, WalSyncPolicy, WAL_FILE_NAME, WAL_FORMAT_VERSION,
    WAL_HEADER_LEN, WAL_MAGIC,
};
use aeris_types::{
    AerisError, FilterConfig, Fingerprint, Key, Lsn, PartitionHasher, QueryRange, RangeFilter,
    ReverseMapStore, WalOp, WorkloadEstimator, MAX_LOAD_FACTOR,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tracing::{info, warn};

/// Worker report identifier written to `Filter/target/ablation_worker2_proactive_storage.json`.
const WORKER_NAME: &str = "worker2_proactive_storage";

/// Canonical extension identifiers for report grouping.
const EXT_AB2: &str = "AB2_Proactive_EWMA_Contraction";
const EXT_AB3: &str = "AB3_WAL_Coupled_Storage";

/// A/B 2 workload parameters (`q = 20`, 1,000,000 keys, 85% -> 95%+ transition, 70% mass deletes).
const AB2_INITIAL_Q: u8 = 20;
const AB2_TOTAL_KEYS: usize = 1_000_000;
const AB2_INITIAL_85PCT_KEYS: usize = 891_290;
const AB2_DELETE_COUNT: usize = 700_000;
const AB2_SURVIVING_KEYS: usize = AB2_TOTAL_KEYS - AB2_DELETE_COUNT;
const AB2_TARGET_EPSILON: f64 = 1.0 / 256.0;
const AB2_RANGE_R: u64 = 32;
const AB2_PAYLOAD_C: u8 = 17;
const AB2_REMAINDER_R: u8 = 12;
const AB2_HASH_SEED: u64 = 0xAB02_2026;
const AB2_FP_DISCOVERY_QUERIES: usize = 16_384;
const AB2_MIXED_QUERY_POOL_SIZE: usize = 16_384;
const AB2_QUERIES_PER_INSERT: usize = 192;
const AB2_READ_LATENCY_SAMPLE_STRIDE: usize = 64;

/// A/B 3 storage benchmark parameters.
const AB3_UNBATCHED_KEYS: usize = 5_000;
const AB3_BATCHED_KEYS: usize = 250_000;
const AB3_BATCH_SIZE: usize = 1_000;
const AB3_HASH_SEED: u64 = 0xAB03_2026;
const AB3_FP_BITS: u16 = 22;

/// A/B 3 subprocess `SIGKILL` crash-recovery parameters (`5,000` keys across 5 crash cycles).
const AB3_SIGKILL_CHILD_ENV: &str = "AERIS_AB3_SIGKILL_CHILD";
const AB3_SIGKILL_DIR_ENV: &str = "AERIS_AB3_SIGKILL_DIR";
const AB3_SIGKILL_CYCLE_ENV: &str = "AERIS_AB3_SIGKILL_CYCLE";
const AB3_SIGKILL_CHECKPOINT_AT_ENV: &str = "AERIS_AB3_SIGKILL_CHECKPOINT_AT";
const AB3_SIGKILL_MODE_COUPLED: &str = "coupled";
const AB3_SIGKILL_MODE_UNCOUPLED: &str = "uncoupled";
const AB3_SIGKILL_CYCLES: u64 = 5;
const AB3_SIGKILL_ACKS_PER_CYCLE: usize = 1_000;
const AB3_SIGKILL_CHILD_CAP: usize = 100_000;
const AB3_UNCOUPLED_RM_BATCH_SIZE: usize = 400;

/// Unified test entrypoint for A/B 2 and A/B 3.
///
/// When invoked as a child subprocess with `AERIS_AB3_SIGKILL_CHILD=coupled|uncoupled`,
/// executes the crash-injection driver loop until terminated via `SIGKILL`.
#[test]
fn run_ablation_proactive_and_storage() {
    if let Ok(mode) = env::var(AB3_SIGKILL_CHILD_ENV) {
        run_sigkill_child_driver(&mode);
        return;
    }

    init_ablation_tracing();
    let mut host = HostMetadata::detect();
    let scale_factor = host.scale_factor;

    info!(
        worker = WORKER_NAME,
        scale_factor, "starting worker2_proactive_storage ablation suite (A/B 2 + A/B 3)"
    );

    let mut metrics = Vec::new();
    let mut report_notes = Vec::new();

    // Execute A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95%.
    let (ab2_metrics, ab2_note) = run_ab2_proactive_ewma_and_contraction(scale_factor);
    metrics.extend(ab2_metrics);
    report_notes.push(ab2_note);
    host.refresh_peak_rss();

    // Execute A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map.
    let (ab3_metrics, ab3_note) = run_ab3_wal_coupled_vs_uncoupled(scale_factor);
    metrics.extend(ab3_metrics);
    report_notes.push(ab3_note);
    host.refresh_peak_rss();

    let report = WorkerAblationReport {
        worker_name: WORKER_NAME.to_string(),
        host,
        metrics,
        notes: report_notes,
    };
    let path = write_worker_report(&report);
    assert!(
        path.exists(),
        "expected worker2 ablation JSON report at {}",
        path.display()
    );
}

// =============================================================================
// A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95%
// =============================================================================

/// Telemetry captured from the Naive Reactive (`alpha = 95%`, no contraction) A/B 2 run.
struct Ab2NaiveOutcome {
    insert_stats: LatencyStats,
    read_stats: LatencyStats,
    reactive_expansions: usize,
    ram_before_deletes: u64,
    ram_after_deletes: u64,
    false_negatives: u64,
}

/// Telemetry captured from the Optimized Proactive EWMA + Contraction A/B 2 run.
struct Ab2OptimizedOutcome {
    insert_stats: LatencyStats,
    read_stats: LatencyStats,
    trigger_alpha: f64,
    swap_alpha: f64,
    proactive_expansions: usize,
    reactive_fallbacks: usize,
    contractions: usize,
    peak_ram_bytes: u64,
    ram_before_deletes: u64,
    ram_after_deletes: u64,
    reclaimed_ram_bytes: u64,
    reclaimed_ram_pct: f64,
    final_q: u8,
    adapted_fp_count: usize,
    false_negatives: u64,
    repeated_fps: u64,
}

/// Runs A/B 2 comparing Reactive `alpha = 95%` synchronous expansion (no contraction)
/// against Proactive EWMA-triggered asynchronous RCU expansion + sustained-low-load
/// contraction over `1,000,000` keys starting at `q = 20`.
fn run_ab2_proactive_ewma_and_contraction(scale_factor: f64) -> (Vec<AblationMetric>, String) {
    info!(
        total_keys = AB2_TOTAL_KEYS,
        initial_85pct = AB2_INITIAL_85PCT_KEYS,
        deletes = AB2_DELETE_COUNT,
        surviving = AB2_SURVIVING_KEYS,
        initial_q = AB2_INITIAL_Q,
        "starting A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95%"
    );

    let full_oracle = Arc::new(generate_uniform_keys_oracle(0xAB02_0001, AB2_TOTAL_KEYS));
    let insert_order = shuffled_keys(full_oracle.as_slice(), 0xAB02_0002);

    let initial_keys = &insert_order[..AB2_INITIAL_85PCT_KEYS];
    let transition_keys = &insert_order[AB2_INITIAL_85PCT_KEYS..];
    let delete_keys = &insert_order[..AB2_DELETE_COUNT];
    let surviving_keys = &insert_order[AB2_DELETE_COUNT..];
    let surviving_oracle = SortedKeyOracle::from_keys(surviving_keys.to_vec());

    // Empty queries disjoint from ALL 1,000,000 keys (therefore valid throughout transition & post-delete).
    let fp_discovery_queries = generate_uniform_empty_queries(
        0xAB02_0003,
        &full_oracle,
        AB2_RANGE_R,
        AB2_FP_DISCOVERY_QUERIES,
    );
    let mixed_query_pool = generate_uniform_empty_queries(
        0xAB02_0004,
        &full_oracle,
        AB2_RANGE_R,
        AB2_MIXED_QUERY_POOL_SIZE,
    );

    let base_config = FilterConfig::new(
        AB2_INITIAL_Q,
        AB2_REMAINDER_R,
        AB2_RANGE_R,
        AB2_PAYLOAD_C,
        AB2_TARGET_EPSILON,
    )
    .expect("valid A/B 2 FilterConfig")
    .with_hash_seed(AB2_HASH_SEED)
    .with_max_load_factor(MAX_LOAD_FACTOR)
    .expect("MAX_LOAD_FACTOR is valid");

    let naive = run_ab2_naive_baseline(
        &base_config,
        initial_keys,
        transition_keys,
        delete_keys,
        surviving_keys,
        &mixed_query_pool,
    );
    assert_eq!(
        naive.false_negatives, 0,
        "A/B 2 Naive baseline produced false negatives"
    );

    let opt = run_ab2_optimized_proactive(
        &base_config,
        initial_keys,
        transition_keys,
        delete_keys,
        surviving_keys,
        &fp_discovery_queries,
        &mixed_query_pool,
        Arc::clone(&full_oracle),
        &surviving_oracle,
    );

    assert_eq!(
        opt.false_negatives, 0,
        "A/B 2 Optimized Aeris produced false negatives across 300,000 surviving keys"
    );
    assert_eq!(
        opt.repeated_fps, 0,
        "A/B 2 Optimized Aeris violated monotonicity across expansion/contraction"
    );
    assert!(
        opt.proactive_expansions >= 1,
        "A/B 2 Optimized Aeris must trigger at least one proactive expansion"
    );
    assert!(
        opt.contractions >= 1,
        "A/B 2 Optimized Aeris must trigger contraction after 70% mass deletes"
    );
    assert_eq!(
        opt.final_q, AB2_INITIAL_Q,
        "A/B 2 Optimized Aeris must contract back from q=21 to q=20"
    );
    assert!(
        opt.ram_after_deletes < opt.ram_before_deletes,
        "A/B 2 contraction must reclaim filter RAM"
    );

    let transition_count = transition_keys.len();

    let metrics = vec![
        AblationMetric {
            extension_name: EXT_AB2.to_string(),
            metric_label: "Transition Insert Latency (p99, 85%->95%)".to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: AB2_TOTAL_KEYS,
            query_count: transition_count,
            baseline_value: naive.insert_stats.p99_ns as f64,
            optimized_value: opt.insert_stats.p99_ns as f64,
            delta_pct: compute_delta_pct(
                naive.insert_stats.p99_ns as f64,
                opt.insert_stats.p99_ns as f64,
            ),
            p50_ns: opt.insert_stats.p50_ns,
            p99_ns: opt.insert_stats.p99_ns,
            p999_ns: opt.insert_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: opt.false_negatives,
            repeated_fps: opt.repeated_fps,
            peak_ram_bytes: opt.peak_ram_bytes,
            disk_io_bytes: 0,
            scale_factor,
            notes: format!(
                "Naive(p50={}ns,p99={}ns,p999={}ns,max={}ns,reactive_exp={}) vs Opt(p50={}ns,p99={}ns,p999={}ns,max={}ns,trigger_alpha={:.4},swap_alpha={:.4},reactive_fallbacks={})",
                naive.insert_stats.p50_ns,
                naive.insert_stats.p99_ns,
                naive.insert_stats.p999_ns,
                naive.insert_stats.max_ns,
                naive.reactive_expansions,
                opt.insert_stats.p50_ns,
                opt.insert_stats.p99_ns,
                opt.insert_stats.p999_ns,
                opt.insert_stats.max_ns,
                opt.trigger_alpha,
                opt.swap_alpha,
                opt.reactive_fallbacks,
            ),
        },
        AblationMetric {
            extension_name: EXT_AB2.to_string(),
            metric_label: "Transition Insert Latency (p99.9, 85%->95%)".to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: AB2_TOTAL_KEYS,
            query_count: transition_count,
            baseline_value: naive.insert_stats.p999_ns as f64,
            optimized_value: opt.insert_stats.p999_ns as f64,
            delta_pct: compute_delta_pct(
                naive.insert_stats.p999_ns as f64,
                opt.insert_stats.p999_ns as f64,
            ),
            p50_ns: opt.insert_stats.p50_ns,
            p99_ns: opt.insert_stats.p99_ns,
            p999_ns: opt.insert_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: opt.false_negatives,
            repeated_fps: opt.repeated_fps,
            peak_ram_bytes: opt.peak_ram_bytes,
            disk_io_bytes: 0,
            scale_factor,
            notes: format!(
                "Concurrent read latency: Naive(p50={}ns,p99={}ns,max={}ns) vs Opt(p50={}ns,p99={}ns,max={}ns)",
                naive.read_stats.p50_ns,
                naive.read_stats.p99_ns,
                naive.read_stats.max_ns,
                opt.read_stats.p50_ns,
                opt.read_stats.p99_ns,
                opt.read_stats.max_ns,
            ),
        },
        AblationMetric {
            extension_name: EXT_AB2.to_string(),
            metric_label: "Transition Insert Max Stall (85%->95%)".to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: AB2_TOTAL_KEYS,
            query_count: transition_count,
            baseline_value: naive.insert_stats.max_ns as f64,
            optimized_value: opt.insert_stats.max_ns as f64,
            delta_pct: compute_delta_pct(
                naive.insert_stats.max_ns as f64,
                opt.insert_stats.max_ns as f64,
            ),
            p50_ns: opt.insert_stats.p50_ns,
            p99_ns: opt.insert_stats.p99_ns,
            p999_ns: opt.insert_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: opt.false_negatives,
            repeated_fps: opt.repeated_fps,
            peak_ram_bytes: opt.peak_ram_bytes,
            disk_io_bytes: 0,
            scale_factor,
            notes: format!(
                "Naive synchronous filter.expand() at alpha=0.95 stalled {}ms vs Opt background RCU swap max={}ms (reader max={}us)",
                naive.insert_stats.max_ns / 1_000_000,
                opt.insert_stats.max_ns / 1_000_000,
                opt.read_stats.max_ns / 1_000,
            ),
        },
        AblationMetric {
            extension_name: EXT_AB2.to_string(),
            metric_label: "Post-70%-Delete Filter RAM Footprint".to_string(),
            unit: "bytes".to_string(),
            higher_is_better: false,
            key_count: AB2_SURVIVING_KEYS,
            query_count: AB2_DELETE_COUNT,
            baseline_value: naive.ram_after_deletes as f64,
            optimized_value: opt.ram_after_deletes as f64,
            delta_pct: compute_delta_pct(
                naive.ram_after_deletes as f64,
                opt.ram_after_deletes as f64,
            ),
            p50_ns: opt.insert_stats.p50_ns,
            p99_ns: opt.insert_stats.p99_ns,
            p999_ns: opt.insert_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: opt.false_negatives,
            repeated_fps: opt.repeated_fps,
            peak_ram_bytes: opt.peak_ram_bytes,
            disk_io_bytes: 0,
            scale_factor,
            notes: format!(
                "Pre-delete RAM: Naive={}B, Opt={}B; Post-70%-delete RAM: Naive={}B (q=21, 0B reclaimed) vs Opt={}B (q={}, reclaimed={}B / {:.2}%); {} surviving keys verified with 0 FN and {} adapted FPs verified with 0 repeated FPs",
                naive.ram_before_deletes,
                opt.ram_before_deletes,
                naive.ram_after_deletes,
                opt.ram_after_deletes,
                opt.final_q,
                opt.reclaimed_ram_bytes,
                opt.reclaimed_ram_pct,
                AB2_SURVIVING_KEYS,
                opt.adapted_fp_count,
            ),
        },
    ];

    let summary_note = format!(
        "A/B 2 completed at q=20->21->20 over {} keys: ProactiveTrigger fired at alpha={:.4} and swapped at alpha={:.4}; contraction reclaimed {} bytes ({:.2}%) after {} deletes; 0 FN across {} surviving keys, 0 repeated FPs across {} adapted FPs.",
        AB2_TOTAL_KEYS,
        opt.trigger_alpha,
        opt.swap_alpha,
        opt.reclaimed_ram_bytes,
        opt.reclaimed_ram_pct,
        AB2_DELETE_COUNT,
        AB2_SURVIVING_KEYS,
        opt.adapted_fp_count,
    );

    info!(
        naive_p99_ns = naive.insert_stats.p99_ns,
        opt_p99_ns = opt.insert_stats.p99_ns,
        naive_max_ns = naive.insert_stats.max_ns,
        opt_max_ns = opt.insert_stats.max_ns,
        reclaimed_ram_bytes = opt.reclaimed_ram_bytes,
        reclaimed_ram_pct = opt.reclaimed_ram_pct,
        "completed A/B 2 proactive EWMA + contraction comparison"
    );

    (metrics, summary_note)
}

/// Executes the A/B 2 Naive Paper-Baseline: synchronous expansion ONLY when `alpha >= 0.95`
/// (`AerisError::FilterFull`), blocking the foreground writer, with zero contraction after deletes.
fn run_ab2_naive_baseline(
    base_config: &FilterConfig,
    initial_keys: &[Key],
    transition_keys: &[Key],
    delete_keys: &[Key],
    surviving_keys: &[Key],
    mixed_query_pool: &[QueryRange],
) -> Ab2NaiveOutcome {
    let dir = TempDir::new().expect("create A/B 2 naive temp dir");
    let rm = RocksDbReverseMap::open(dir.path().join("rm"), ReverseMapConfig::for_wal_coupled())
        .expect("open A/B 2 naive ReverseMap");

    let mut filter = AerisFilter::bulk_load(base_config.clone(), initial_keys, &rm)
        .expect("bulk_load 85% naive filter");

    info!(
        initial_alpha = filter.load_factor(),
        initial_ram_bytes = filter.memory_usage_bytes(),
        "A/B 2 Naive baseline loaded to ~85%"
    );

    let mut insert_latencies = Vec::with_capacity(transition_keys.len());
    let mut read_latencies = Vec::with_capacity(
        (transition_keys.len() * AB2_QUERIES_PER_INSERT) / AB2_READ_LATENCY_SAMPLE_STRIDE + 1,
    );
    let mut reactive_expansions = 0usize;
    let pool_len = mixed_query_pool.len();
    let mut q_cursor = 0usize;

    for &key in transition_keys {
        let t0 = Instant::now();
        let fp = match filter.insert(key) {
            Ok(fp) => fp,
            Err(AerisError::FilterFull { .. }) => {
                let _exp = filter
                    .expand(&rm)
                    .expect("naive reactive expand must succeed");
                reactive_expansions += 1;
                filter
                    .insert(key)
                    .expect("naive insert after reactive expand must succeed")
            }
            Err(err) => panic!("unexpected naive insert error: {err}"),
        };
        rm.append_key(&fp, key)
            .expect("naive reverse_map append_key");
        insert_latencies.push(t0.elapsed().as_nanos() as u64);

        for step in 0..AB2_QUERIES_PER_INSERT {
            let range = mixed_query_pool[q_cursor % pool_len];
            q_cursor = q_cursor.wrapping_add(1);
            if step % AB2_READ_LATENCY_SAMPLE_STRIDE == 0 {
                let qt = Instant::now();
                let hit = filter.query_range(range);
                black_box(hit);
                read_latencies.push(qt.elapsed().as_nanos() as u64);
            } else {
                black_box(filter.query_range(range));
            }
        }
    }

    let ram_before_deletes = filter.memory_usage_bytes() as u64;

    // 70% mass deletes (no contraction in Naive baseline).
    for &key in delete_keys {
        if let Some(fp) = filter.delete(key).expect("naive filter delete") {
            rm.remove_key(&fp, key)
                .expect("naive reverse_map remove_key");
        }
    }

    let ram_after_deletes = filter.memory_usage_bytes() as u64;
    let false_negatives = count_false_negatives_filter(&filter, surviving_keys);

    info!(
        reactive_expansions,
        q_after_deletes = filter.config().q,
        alpha_after_deletes = filter.load_factor(),
        ram_before_deletes,
        ram_after_deletes,
        false_negatives,
        "A/B 2 Naive baseline finished"
    );

    Ab2NaiveOutcome {
        insert_stats: LatencyStats::from_samples(insert_latencies),
        read_stats: LatencyStats::from_samples(read_latencies),
        reactive_expansions,
        ram_before_deletes,
        ram_after_deletes,
        false_negatives,
    }
}

/// Executes the A/B 2 Optimized Aeris path using real `RcuAerisFilter`,
/// `RocksDbReverseMap`, `SlidingWindowEstimator`, `ProactiveTrigger`, and
/// `FalsePositiveLog` for non-blocking proactive expansion and post-delete contraction.
#[allow(clippy::too_many_arguments)]
fn run_ab2_optimized_proactive(
    base_config: &FilterConfig,
    initial_keys: &[Key],
    transition_keys: &[Key],
    delete_keys: &[Key],
    surviving_keys: &[Key],
    fp_discovery_queries: &[QueryRange],
    mixed_query_pool: &[QueryRange],
    full_oracle: Arc<SortedKeyOracle>,
    surviving_oracle: &SortedKeyOracle,
) -> Ab2OptimizedOutcome {
    let dir = TempDir::new().expect("create A/B 2 optimized temp dir");
    let rm_gen0 = Arc::new(
        RocksDbReverseMap::open(
            dir.path().join("rm_gen0"),
            ReverseMapConfig::for_wal_coupled(),
        )
        .expect("open A/B 2 optimized initial ReverseMap"),
    );
    let fp_log = Arc::new(
        FalsePositiveLog::open(
            dir.path().join("fp.log"),
            FpLogConfig {
                max_entries: 65_536,
                sync_on_append: false,
            },
        )
        .expect("open A/B 2 FalsePositiveLog"),
    );

    let bulk_start = Instant::now();
    let mut initial_filter =
        AerisFilter::bulk_load(base_config.clone(), initial_keys, rm_gen0.as_ref())
            .expect("bulk_load 85% optimized filter");
    let bulk_elapsed_secs = bulk_start.elapsed().as_secs_f64().max(1e-6);
    // Conservative end-to-end rebuild throughput (scan + bulk_load + RM populate) ~ 0.35x in-RAM bulk_load.
    let initial_bulk_speed = ((initial_keys.len() as f64) / bulk_elapsed_secs) * 0.35;

    // Discover and adapt genuine false positives into FalsePositiveLog before transition so
    // we verify 100% monotonicity across both proactive expansion and post-delete contraction.
    let mut adapted_fp_ranges = Vec::new();
    for &range in fp_discovery_queries {
        if initial_filter.query_range(range) {
            initial_filter
                .adapt(range, rm_gen0.as_ref())
                .expect("adapt initial FP before A/B 2 transition");
            let _ = fp_log
                .record(range, current_unix_ms())
                .expect("record adapted FP in FalsePositiveLog");
            adapted_fp_ranges.push(range);
        }
    }

    let rcu_filter = Arc::new(RcuAerisFilter::new(
        0,
        initial_filter,
        Arc::clone(&rm_gen0) as Arc<dyn ReverseMapStore>,
    ));

    let inserted_counter = Arc::new(AtomicUsize::new(0));
    let writer_done = Arc::new(AtomicBool::new(false));
    let expand_rm_path = dir.path().join("rm_gen1_expand");

    let ctrl_rcu = Arc::clone(&rcu_filter);
    let ctrl_fp_log = Arc::clone(&fp_log);
    let ctrl_inserted = Arc::clone(&inserted_counter);
    let ctrl_done = Arc::clone(&writer_done);
    let ctrl_oracle = Arc::clone(&full_oracle);
    let target_expand_cfg = base_config.expanded().expect("q=21 config valid");

    let trigger_cfg = ProactiveTriggerConfig {
        margin: 2.0,
        hard_expand_alpha: 0.90,
        contraction_alpha_threshold: 0.25,
        contraction_sustained_checks: 4,
        max_post_contraction_alpha: 0.65,
    };

    // Dedicated Control Plane thread running SlidingWindowEstimator + ProactiveTrigger.
    let control_handle = thread::Builder::new()
        .name("ab2-control-plane".to_string())
        .spawn(move || {
            let mut estimator =
                SlidingWindowEstimator::with_initial_span(EstimatorConfig::default(), AB2_RANGE_R)
                    .expect("valid SlidingWindowEstimator");
            estimator.record_bulk_load_speed(initial_bulk_speed);

            let mut trigger =
                ProactiveTrigger::new(trigger_cfg).expect("valid ProactiveTriggerConfig");

            let mut last_inserted = 0usize;
            let mut last_tick = Instant::now();
            let mut rebuild_started = false;
            let mut rebuild_completed = false;
            let mut trigger_alpha = 0.0_f64;
            let mut swap_alpha = 0.0_f64;
            let mut proactive_expansions = 0usize;

            loop {
                let current_inserted = ctrl_inserted.load(Ordering::Acquire);
                let delta = current_inserted.saturating_sub(last_inserted);

                if !rebuild_started && delta >= 256 {
                    let dt_secs = last_tick.elapsed().as_secs_f64().max(1e-6);
                    estimator.record_ingest(delta as u64, dt_secs);
                    last_inserted = current_inserted;
                    last_tick = Instant::now();

                    let stats = estimator.snapshot();
                    let cur_cfg = ctrl_rcu.config();
                    let cur_alpha = ctrl_rcu.load_factor();
                    let eval = trigger.evaluate(&TriggerInput {
                        capacity_slots: ctrl_rcu.capacity(),
                        current_q: cur_cfg.q,
                        current_alpha: cur_alpha,
                        ingestion_rate_ewma: stats.ingestion_rate_ewma,
                        reverse_map_entries: ctrl_rcu.active_reverse_map().entry_count_estimate(),
                        bulk_load_speed: stats.bulk_load_speed,
                    });

                    if eval.action == TriggerAction::ProactiveExpand {
                        trigger_alpha = cur_alpha;
                        let dest_rm = Arc::new(
                            RocksDbReverseMap::open(
                                &expand_rm_path,
                                ReverseMapConfig::for_wal_coupled(),
                            )
                            .expect("open A/B 2 expansion ReverseMap"),
                        );
                        ctrl_rcu
                            .start_async_rebuild(
                                dest_rm as Arc<dyn ReverseMapStore>,
                                Arc::clone(&ctrl_fp_log),
                                target_expand_cfg.clone(),
                                RebuildReason::ProactiveExpansion {
                                    old_q: cur_cfg.q,
                                    new_q: target_expand_cfg.q,
                                },
                                IoThrottleConfig::default(),
                            )
                            .expect("start proactive async rebuild");
                        rebuild_started = true;
                        info!(
                            trigger_alpha,
                            ingestion_rate_ewma = stats.ingestion_rate_ewma,
                            t_full_secs = eval.t_full_secs,
                            t_rebuild_secs = eval.t_rebuild_secs,
                            "A/B 2 ProactiveTrigger launched background q=20->21 rebuild"
                        );
                    }
                }

                if rebuild_started && !rebuild_completed && ctrl_rcu.is_rebuild_ready() {
                    swap_alpha = ctrl_rcu.load_factor();
                    let swap_report = ctrl_rcu
                        .finish_async_rebuild_and_swap(|r| Ok(ctrl_oracle.is_range_empty(r)))
                        .expect("finish proactive RCU rebuild swap");
                    estimator.record_bulk_load_speed(swap_report.rebuild.observed_bulk_load_speed);
                    rebuild_completed = true;
                    proactive_expansions += 1;
                    info!(
                        swap_alpha,
                        new_q = ctrl_rcu.config().q,
                        new_alpha = ctrl_rcu.load_factor(),
                        delta_inserts = swap_report.rebuild.delta_inserts_applied,
                        "A/B 2 completed proactive RCU swap"
                    );
                }

                if ctrl_done.load(Ordering::Acquire) {
                    if rebuild_started && !rebuild_completed {
                        swap_alpha = ctrl_rcu.load_factor();
                        let _ = ctrl_rcu
                            .finish_async_rebuild_and_swap(|r| Ok(ctrl_oracle.is_range_empty(r)))
                            .expect("finalize in-flight proactive rebuild");
                        proactive_expansions += 1;
                    }
                    break;
                }

                thread::yield_now();
            }

            (
                estimator,
                trigger,
                trigger_alpha,
                swap_alpha,
                proactive_expansions,
            )
        })
        .expect("spawn ab2-control-plane thread");

    // Foreground thread streams the exact same 85% -> 100% transition + mixed read workload.
    let mut insert_latencies = Vec::with_capacity(transition_keys.len());
    let mut read_latencies = Vec::with_capacity(
        (transition_keys.len() * AB2_QUERIES_PER_INSERT) / AB2_READ_LATENCY_SAMPLE_STRIDE + 1,
    );
    let pool_len = mixed_query_pool.len();
    let mut q_cursor = 0usize;

    for &key in transition_keys {
        let t0 = Instant::now();
        let _fp = rcu_filter
            .insert(key)
            .expect("optimized RcuAerisFilter insert");
        insert_latencies.push(t0.elapsed().as_nanos() as u64);
        inserted_counter.fetch_add(1, Ordering::Release);

        for step in 0..AB2_QUERIES_PER_INSERT {
            let range = mixed_query_pool[q_cursor % pool_len];
            q_cursor = q_cursor.wrapping_add(1);
            if step % AB2_READ_LATENCY_SAMPLE_STRIDE == 0 {
                let qt = Instant::now();
                let hit = rcu_filter.query_range(range);
                black_box(hit);
                read_latencies.push(qt.elapsed().as_nanos() as u64);
            } else {
                black_box(rcu_filter.query_range(range));
            }
        }
    }

    writer_done.store(true, Ordering::Release);
    let (estimator, mut trigger, trigger_alpha, swap_alpha, proactive_expansions) = control_handle
        .join()
        .expect("join ab2-control-plane thread");

    let reactive_fallbacks = rcu_filter
        .epoch()
        .saturating_sub(1 + (proactive_expansions as u64)) as usize;
    let ram_before_deletes = rcu_filter.memory_usage_bytes() as u64;
    let peak_ram_bytes = ram_before_deletes;

    // 70% mass deletes (`700,000` keys leaving `300,000` surviving keys).
    for &key in delete_keys {
        let _ = rcu_filter
            .delete(key)
            .expect("optimized RcuAerisFilter delete");
    }

    let post_delete_alpha = rcu_filter.load_factor();
    let post_delete_q = rcu_filter.config().q;
    let stats = estimator.snapshot();
    let mut contract_fired = false;

    // Evaluate ProactiveTrigger across `contraction_sustained_checks` (4) checks with zero ingest.
    let required_checks = trigger.config().contraction_sustained_checks;
    for check_idx in 1..=required_checks {
        let eval = trigger.evaluate(&TriggerInput {
            capacity_slots: rcu_filter.capacity(),
            current_q: post_delete_q,
            current_alpha: post_delete_alpha,
            ingestion_rate_ewma: 0.0,
            reverse_map_entries: rcu_filter.active_reverse_map().entry_count_estimate(),
            bulk_load_speed: stats.bulk_load_speed,
        });
        if check_idx < required_checks {
            assert_eq!(
                eval.action,
                TriggerAction::None,
                "contraction must honor sustained low-alpha hysteresis before check {required_checks}"
            );
        } else if eval.action == TriggerAction::Contract {
            contract_fired = true;
        }
    }

    let mut contractions = 0usize;
    if contract_fired {
        let contracted_cfg = rcu_filter
            .config()
            .contracted()
            .expect("contracted q=20 config valid");
        let contract_rm = Arc::new(
            RocksDbReverseMap::open(
                dir.path().join("rm_gen2_contract"),
                ReverseMapConfig::for_wal_coupled(),
            )
            .expect("open contraction destination ReverseMap"),
        );
        let contract_report = rcu_filter
            .rebuild_async_and_swap_with_verifier(
                contract_rm as Arc<dyn ReverseMapStore>,
                Arc::clone(&fp_log),
                contracted_cfg.clone(),
                RebuildReason::Contraction {
                    old_q: post_delete_q,
                    new_q: contracted_cfg.q,
                },
                IoThrottleConfig::default(),
                |r| Ok(surviving_oracle.is_range_empty(r)),
            )
            .expect("proactive contraction RCU rebuild must succeed");
        contractions += 1;
        info!(
            old_q = post_delete_q,
            new_q = rcu_filter.config().q,
            alpha_after_contraction = rcu_filter.load_factor(),
            keys_loaded = contract_report.rebuild.keys_loaded,
            fp_collisions_resolved = contract_report
                .rebuild
                .fp_replay_initial
                .collisions_resolved
                + contract_report
                    .rebuild
                    .fp_replay_catchup
                    .collisions_resolved,
            "A/B 2 completed post-delete contraction"
        );
    }

    rcu_filter
        .verify_invariants()
        .expect("optimized RcuAerisFilter invariants must hold after expansion and contraction");

    let ram_after_deletes = rcu_filter.memory_usage_bytes() as u64;
    let reclaimed_ram_bytes = ram_before_deletes.saturating_sub(ram_after_deletes);
    let reclaimed_ram_pct = if ram_before_deletes > 0 {
        (reclaimed_ram_bytes as f64 / ram_before_deletes as f64) * 100.0
    } else {
        0.0
    };

    // Verify 0 false negatives across ALL 300,000 surviving keys (point + enclosing range).
    let false_negatives = count_false_negatives_rcu(&rcu_filter, surviving_keys);

    // Verify 0 repeated FPs across every adapted FP range recorded in FalsePositiveLog.
    let mut repeated_fps = 0u64;
    for &fp_range in &adapted_fp_ranges {
        if rcu_filter.query_range(fp_range) {
            repeated_fps += 1;
        }
    }

    Ab2OptimizedOutcome {
        insert_stats: LatencyStats::from_samples(insert_latencies),
        read_stats: LatencyStats::from_samples(read_latencies),
        trigger_alpha,
        swap_alpha,
        proactive_expansions,
        reactive_fallbacks,
        contractions,
        peak_ram_bytes,
        ram_before_deletes,
        ram_after_deletes,
        reclaimed_ram_bytes,
        reclaimed_ram_pct,
        final_q: rcu_filter.config().q,
        adapted_fp_count: adapted_fp_ranges.len(),
        false_negatives,
        repeated_fps,
    }
}

// =============================================================================
// A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map
// =============================================================================

/// Runs all three A/B 3 storage experiments:
/// 1. Unbatched per-op `fsync` overhead (`5,000` keys).
/// 2. Batched WAL commit overhead (`250,000` keys, batch size `1,000`).
/// 3. Subprocess `SIGKILL` crash recovery (`5,000` keys across 5 crash cycles).
fn run_ab3_wal_coupled_vs_uncoupled(scale_factor: f64) -> (Vec<AblationMetric>, String) {
    info!(
        unbatched_keys = AB3_UNBATCHED_KEYS,
        batched_keys = AB3_BATCHED_KEYS,
        batch_size = AB3_BATCH_SIZE,
        sigkill_cycles = AB3_SIGKILL_CYCLES,
        sigkill_acks_per_cycle = AB3_SIGKILL_ACKS_PER_CYCLE,
        "starting A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map"
    );

    let mut metrics = Vec::new();
    let (unbatched_metrics, unbatched_summary) = run_ab3_unbatched_fsync_overhead(scale_factor);
    metrics.extend(unbatched_metrics);

    let (batched_metrics, batched_summary) = run_ab3_batched_wal_overhead(scale_factor);
    metrics.extend(batched_metrics);

    let (sigkill_metrics, sigkill_summary) = run_ab3_sigkill_crash_comparison(scale_factor);
    metrics.extend(sigkill_metrics);

    let combined_note =
        format!("A/B 3 summary: {unbatched_summary} | {batched_summary} | {sigkill_summary}");
    (metrics, combined_note)
}

/// A/B 3.1: Measures unbatched per-op `fsync` overhead (`5,000` keys) comparing
/// `CoupledStorageEngine::put` against uncoupled `ReferenceKvStore` + `RocksDbReverseMap`.
fn run_ab3_unbatched_fsync_overhead(scale_factor: f64) -> (Vec<AblationMetric>, String) {
    let keys = generate_uniform_keys_oracle(0xAB03_1001, AB3_UNBATCHED_KEYS).into_vec();
    let hasher = PartitionHasher::new(AB3_HASH_SEED);
    let prepared: Vec<(Key, Vec<u8>, Fingerprint)> = keys
        .iter()
        .map(|&k| {
            let fp = Fingerprint::from_hasher(&hasher, k >> 5, AB3_FP_BITS);
            (k, deterministic_payload(k), fp)
        })
        .collect();

    // Baseline: Uncoupled ReferenceKvStore + RocksDbReverseMap (sync_writes: false, no CoupledWal).
    let uncoupled_dir = TempDir::new().expect("create A/B 3 unbatched uncoupled dir");
    let uncoupled_kv = ReferenceKvStore::open_with_config(
        uncoupled_dir.path().join("kv_db"),
        KvStoreConfig::for_wal_coupled(),
    )
    .expect("open uncoupled ReferenceKvStore");
    let uncoupled_rm = RocksDbReverseMap::open(
        uncoupled_dir.path().join("reverse_map_db"),
        ReverseMapConfig::for_wal_coupled(),
    )
    .expect("open uncoupled RocksDbReverseMap");

    let mut uncoupled_latencies = Vec::with_capacity(AB3_UNBATCHED_KEYS);
    let uncoupled_start = Instant::now();
    for (key, payload, fp) in &prepared {
        let t0 = Instant::now();
        uncoupled_kv.put(*key, payload).expect("uncoupled kv put");
        uncoupled_rm
            .append_key(fp, *key)
            .expect("uncoupled rm append_key");
        uncoupled_latencies.push(t0.elapsed().as_nanos() as u64);
    }
    let uncoupled_elapsed = uncoupled_start.elapsed().as_secs_f64().max(1e-9);
    uncoupled_kv.flush().expect("flush uncoupled kv");
    uncoupled_rm.flush().expect("flush uncoupled rm");
    let uncoupled_disk_bytes = directory_size_bytes(uncoupled_dir.path());
    let uncoupled_ops_sec = (AB3_UNBATCHED_KEYS as f64) / uncoupled_elapsed;
    let uncoupled_stats = LatencyStats::from_samples(uncoupled_latencies);

    // Explicit per-op durability hook: CoupledStorageEngine with StorageConfig::strict_per_op_fsync().
    let coupled_dir = TempDir::new().expect("create A/B 3 unbatched coupled dir");
    let (coupled_engine, _) = CoupledStorageEngine::open_and_recover_with_config(
        coupled_dir.path(),
        StorageConfig::strict_per_op_fsync(),
    )
    .expect("open CoupledStorageEngine for unbatched test");

    let mut coupled_latencies = Vec::with_capacity(AB3_UNBATCHED_KEYS);
    let coupled_start = Instant::now();
    for (key, payload, fp) in &prepared {
        let t0 = Instant::now();
        let _lsn = coupled_engine
            .put(*key, payload, fp)
            .expect("coupled engine put");
        coupled_latencies.push(t0.elapsed().as_nanos() as u64);
    }
    let coupled_elapsed = coupled_start.elapsed().as_secs_f64().max(1e-9);
    coupled_engine.kv_store().flush().expect("flush coupled kv");
    coupled_engine
        .reverse_map()
        .flush()
        .expect("flush coupled rm");
    let coupled_disk_bytes = directory_size_bytes(coupled_dir.path());
    let coupled_ops_sec = (AB3_UNBATCHED_KEYS as f64) / coupled_elapsed;
    let coupled_stats = LatencyStats::from_samples(coupled_latencies);

    let tp_delta = compute_delta_pct(uncoupled_ops_sec, coupled_ops_sec);
    let p99_delta = compute_delta_pct(uncoupled_stats.p99_ns as f64, coupled_stats.p99_ns as f64);

    info!(
        uncoupled_ops_sec,
        coupled_ops_sec,
        tp_delta_pct = tp_delta,
        uncoupled_p99_ns = uncoupled_stats.p99_ns,
        coupled_p99_ns = coupled_stats.p99_ns,
        uncoupled_disk_bytes,
        coupled_disk_bytes,
        "completed A/B 3.1 unbatched per-op fsync overhead benchmark"
    );

    let metrics = vec![
        AblationMetric {
            extension_name: EXT_AB3.to_string(),
            metric_label: "Unbatched Per-Op fsync Throughput (5k keys)".to_string(),
            unit: "ops/sec".to_string(),
            higher_is_better: true,
            key_count: AB3_UNBATCHED_KEYS,
            query_count: AB3_UNBATCHED_KEYS,
            baseline_value: uncoupled_ops_sec,
            optimized_value: coupled_ops_sec,
            delta_pct: tp_delta,
            p50_ns: coupled_stats.p50_ns,
            p99_ns: coupled_stats.p99_ns,
            p999_ns: coupled_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: 0,
            disk_io_bytes: coupled_disk_bytes,
            scale_factor,
            notes: format!(
                "Uncoupled(p50={}ns,p99={}ns,p999={}ns,disk={}B) vs CoupledWalPerOpFsync(p50={}ns,p99={}ns,p999={}ns,disk={}B)",
                uncoupled_stats.p50_ns,
                uncoupled_stats.p99_ns,
                uncoupled_stats.p999_ns,
                uncoupled_disk_bytes,
                coupled_stats.p50_ns,
                coupled_stats.p99_ns,
                coupled_stats.p999_ns,
                coupled_disk_bytes,
            ),
        },
        AblationMetric {
            extension_name: EXT_AB3.to_string(),
            metric_label: "Unbatched Per-Op fsync Latency (p99, 5k keys)".to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: AB3_UNBATCHED_KEYS,
            query_count: AB3_UNBATCHED_KEYS,
            baseline_value: uncoupled_stats.p99_ns as f64,
            optimized_value: coupled_stats.p99_ns as f64,
            delta_pct: p99_delta,
            p50_ns: coupled_stats.p50_ns,
            p99_ns: coupled_stats.p99_ns,
            p999_ns: coupled_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: 0,
            disk_io_bytes: coupled_disk_bytes,
            scale_factor,
            notes: format!(
                "Per-op fsync durability barrier overhead: baseline_disk={}B, coupled_disk={}B",
                uncoupled_disk_bytes, coupled_disk_bytes
            ),
        },
    ];

    let summary = format!(
        "Unbatched(5k): uncoupled={:.0} ops/s vs coupled={:.0} ops/s ({:+.1}%)",
        uncoupled_ops_sec, coupled_ops_sec, tp_delta
    );
    (metrics, summary)
}

/// A/B 3.2: Measures batched WAL commit overhead (`250,000` keys, batch size `1,000`)
/// comparing batched `CoupledWal` (`1,000` frames + single `wal.sync()`) + `ReferenceKvStore`
/// + `RocksDbReverseMap::append_batch` against uncoupled batched KV + Reverse Map.
fn run_ab3_batched_wal_overhead(scale_factor: f64) -> (Vec<AblationMetric>, String) {
    let keys = generate_uniform_keys_oracle(0xAB03_2001, AB3_BATCHED_KEYS).into_vec();
    let hasher = PartitionHasher::new(AB3_HASH_SEED);
    let num_batches = AB3_BATCHED_KEYS / AB3_BATCH_SIZE;

    // Precompute batch fingerprints & payloads outside the timed region so measurement
    // reflects pure storage/WAL persistence overhead.
    let prepared: Vec<(Key, Vec<u8>, Fingerprint)> = keys
        .iter()
        .map(|&k| {
            let fp = Fingerprint::from_hasher(&hasher, k >> 5, AB3_FP_BITS);
            (k, deterministic_payload(k), fp)
        })
        .collect();

    // Baseline: Uncoupled batched ReferenceKvStore + RocksDbReverseMap::append_batch (no CoupledWal).
    let uncoupled_dir = TempDir::new().expect("create A/B 3 batched uncoupled dir");
    let uncoupled_kv = ReferenceKvStore::open_with_config(
        uncoupled_dir.path().join("kv_db"),
        KvStoreConfig::for_wal_coupled(),
    )
    .expect("open uncoupled batched kv");
    let uncoupled_rm = RocksDbReverseMap::open(
        uncoupled_dir.path().join("reverse_map_db"),
        ReverseMapConfig::for_wal_coupled(),
    )
    .expect("open uncoupled batched rm");

    let mut uncoupled_batch_ns = Vec::with_capacity(num_batches);
    let mut uncoupled_per_key_ns = Vec::with_capacity(num_batches);
    let mut rm_batch: Vec<(Fingerprint, Key)> = Vec::with_capacity(AB3_BATCH_SIZE);

    let uncoupled_start = Instant::now();
    for chunk in prepared.chunks(AB3_BATCH_SIZE) {
        rm_batch.clear();
        for (key, _, fp) in chunk {
            rm_batch.push((fp.clone(), *key));
        }
        let t0 = Instant::now();
        for (key, payload, _) in chunk {
            uncoupled_kv
                .put(*key, payload)
                .expect("uncoupled batched kv put");
        }
        uncoupled_rm
            .append_batch(&rm_batch)
            .expect("uncoupled batched rm append_batch");
        let dt_ns = t0.elapsed().as_nanos() as u64;
        uncoupled_batch_ns.push(dt_ns);
        uncoupled_per_key_ns.push(dt_ns / (chunk.len() as u64));
    }
    let uncoupled_elapsed = uncoupled_start.elapsed().as_secs_f64().max(1e-9);
    uncoupled_kv.flush().expect("flush uncoupled batched kv");
    uncoupled_rm.flush().expect("flush uncoupled batched rm");
    let uncoupled_disk_bytes = directory_size_bytes(uncoupled_dir.path());
    let uncoupled_ops_sec = (AB3_BATCHED_KEYS as f64) / uncoupled_elapsed;
    let uncoupled_batch_stats = LatencyStats::from_samples(uncoupled_batch_ns);
    let uncoupled_key_stats = LatencyStats::from_samples(uncoupled_per_key_ns);

    // Optimized: Default CoupledStorageEngine (WalSyncPolicy::GroupCommit { batch_size: 1_000 })
    // executing first-class put_batch() (1,000 WalOp::Put frames -> single wal.sync() -> KV + RM batch).
    let coupled_dir = TempDir::new().expect("create A/B 3 batched coupled dir");
    let (coupled_engine, _) = CoupledStorageEngine::open_and_recover(coupled_dir.path())
        .expect("open default group-commit CoupledStorageEngine for batched test");
    assert_eq!(
        coupled_engine.wal_sync_policy(),
        WalSyncPolicy::GroupCommit {
            batch_size: AB3_BATCH_SIZE,
        },
        "default StorageConfig must enforce B >= 1000 group-commit batching"
    );

    let mut coupled_batch_ns = Vec::with_capacity(num_batches);
    let mut coupled_per_key_ns = Vec::with_capacity(num_batches);
    let mut batch_entries: Vec<(Key, &[u8], &Fingerprint)> = Vec::with_capacity(AB3_BATCH_SIZE);

    let coupled_start = Instant::now();
    for chunk in prepared.chunks(AB3_BATCH_SIZE) {
        batch_entries.clear();
        for (key, payload, fp) in chunk {
            batch_entries.push((*key, payload.as_slice(), fp));
        }
        let t0 = Instant::now();
        let _lsn = coupled_engine
            .put_batch(&batch_entries)
            .expect("default group-commit put_batch must succeed");
        let dt_ns = t0.elapsed().as_nanos() as u64;
        coupled_batch_ns.push(dt_ns);
        coupled_per_key_ns.push(dt_ns / (chunk.len() as u64));
    }
    let coupled_elapsed = coupled_start.elapsed().as_secs_f64().max(1e-9);
    assert_eq!(
        coupled_engine.last_lsn(),
        AB3_BATCHED_KEYS as Lsn,
        "batched CoupledStorageEngine must record all 250,000 LSNs"
    );
    coupled_engine
        .kv_store()
        .flush()
        .expect("flush coupled batched kv");
    coupled_engine
        .reverse_map()
        .flush()
        .expect("flush coupled batched rm");
    let coupled_disk_bytes = directory_size_bytes(coupled_dir.path());
    let coupled_ops_sec = (AB3_BATCHED_KEYS as f64) / coupled_elapsed;
    let coupled_batch_stats = LatencyStats::from_samples(coupled_batch_ns);
    let coupled_key_stats = LatencyStats::from_samples(coupled_per_key_ns);

    let tp_delta = compute_delta_pct(uncoupled_ops_sec, coupled_ops_sec);
    let batch_p99_delta = compute_delta_pct(
        uncoupled_batch_stats.p99_ns as f64,
        coupled_batch_stats.p99_ns as f64,
    );

    info!(
        uncoupled_ops_sec,
        coupled_ops_sec,
        tp_delta_pct = tp_delta,
        uncoupled_batch_p99_ns = uncoupled_batch_stats.p99_ns,
        coupled_batch_p99_ns = coupled_batch_stats.p99_ns,
        coupled_amortized_p99_ns = coupled_key_stats.p99_ns,
        uncoupled_disk_bytes,
        coupled_disk_bytes,
        "completed A/B 3.2 batched WAL commit overhead benchmark"
    );

    let metrics = vec![
        AblationMetric {
            extension_name: EXT_AB3.to_string(),
            metric_label: "Batched WAL Commit Throughput (250k keys, B=1000)".to_string(),
            unit: "ops/sec".to_string(),
            higher_is_better: true,
            key_count: AB3_BATCHED_KEYS,
            query_count: AB3_BATCHED_KEYS,
            baseline_value: uncoupled_ops_sec,
            optimized_value: coupled_ops_sec,
            delta_pct: tp_delta,
            p50_ns: coupled_key_stats.p50_ns,
            p99_ns: coupled_key_stats.p99_ns,
            p999_ns: coupled_key_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: 0,
            disk_io_bytes: coupled_disk_bytes,
            scale_factor,
            notes: format!(
                "Amortized/key: Uncoupled(p50={}ns,p99={}ns,p999={}ns,disk={}B) vs CoupledWalBatch(p50={}ns,p99={}ns,p999={}ns,disk={}B)",
                uncoupled_key_stats.p50_ns,
                uncoupled_key_stats.p99_ns,
                uncoupled_key_stats.p999_ns,
                uncoupled_disk_bytes,
                coupled_key_stats.p50_ns,
                coupled_key_stats.p99_ns,
                coupled_key_stats.p999_ns,
                coupled_disk_bytes,
            ),
        },
        AblationMetric {
            extension_name: EXT_AB3.to_string(),
            metric_label: "Batched WAL Commit Latency (p99/batch, B=1000)".to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: AB3_BATCHED_KEYS,
            query_count: num_batches,
            baseline_value: uncoupled_batch_stats.p99_ns as f64,
            optimized_value: coupled_batch_stats.p99_ns as f64,
            delta_pct: batch_p99_delta,
            p50_ns: coupled_batch_stats.p50_ns,
            p99_ns: coupled_batch_stats.p99_ns,
            p999_ns: coupled_batch_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: 0,
            disk_io_bytes: coupled_disk_bytes,
            scale_factor,
            notes: format!(
                "Per-batch(1000 keys): Uncoupled(p50={}ns,p99={}ns,p999={}ns) vs CoupledWalBatch(p50={}ns,p99={}ns,p999={}ns)",
                uncoupled_batch_stats.p50_ns,
                uncoupled_batch_stats.p99_ns,
                uncoupled_batch_stats.p999_ns,
                coupled_batch_stats.p50_ns,
                coupled_batch_stats.p99_ns,
                coupled_batch_stats.p999_ns,
            ),
        },
    ];

    let summary = format!(
        "Batched(250k,B=1000): uncoupled={:.0} ops/s vs coupled={:.0} ops/s ({:+.1}%)",
        uncoupled_ops_sec, coupled_ops_sec, tp_delta
    );
    (metrics, summary)
}

/// Summary of a multi-cycle real subprocess `SIGKILL` crash-recovery campaign.
struct SigkillCampaignOutcome {
    total_acked: usize,
    lost_kv_keys: usize,
    lost_rm_keys: usize,
    desync_keys: usize,
    post_rebuild_false_negatives: u64,
    failure_rate_pct: f64,
    recovery_latencies_ns: Vec<u64>,
    disk_bytes: u64,
}

/// A/B 3.3: Runs identical 5-cycle real subprocess `SIGKILL` (`TerminateProcess` / `kill -9`)
/// crash-recovery campaigns (`5,000+` ACKed keys total) against:
/// - **Optimized (`CoupledStorageEngine`):** WAL-coupled durability + torn-tail truncation +
///   `open_and_recover` + `rebuild_filter_and_reconcile`.
/// - **Naive Uncoupled (`ReferenceKvStore` + `RocksDbReverseMap`, `sync_writes: false`,
///   uncoupled batched Reverse Map, no `CoupledWal`):** measures genuine lost Reverse Map keys,
///   KV <-> Reverse Map divergence, and post-rebuild false negatives.
fn run_ab3_sigkill_crash_comparison(scale_factor: f64) -> (Vec<AblationMetric>, String) {
    let coupled = run_sigkill_coupled_campaign();
    assert_eq!(
        coupled.lost_kv_keys, 0,
        "CoupledStorageEngine lost ACKed KV keys across SIGKILL cycles"
    );
    assert_eq!(
        coupled.lost_rm_keys, 0,
        "CoupledStorageEngine lost ACKed Reverse Map keys across SIGKILL cycles"
    );
    assert_eq!(
        coupled.desync_keys, 0,
        "CoupledStorageEngine KV and Reverse Map diverged across SIGKILL cycles"
    );
    assert_eq!(
        coupled.post_rebuild_false_negatives, 0,
        "CoupledStorageEngine rebuild_filter_and_reconcile produced false negatives"
    );

    let uncoupled = run_sigkill_uncoupled_campaign();
    let rec_stats = LatencyStats::from_samples(coupled.recovery_latencies_ns.clone());

    info!(
        coupled_acked = coupled.total_acked,
        coupled_lost_kv = coupled.lost_kv_keys,
        coupled_lost_rm = coupled.lost_rm_keys,
        coupled_desync = coupled.desync_keys,
        coupled_fn = coupled.post_rebuild_false_negatives,
        uncoupled_acked = uncoupled.total_acked,
        uncoupled_lost_kv = uncoupled.lost_kv_keys,
        uncoupled_lost_rm = uncoupled.lost_rm_keys,
        uncoupled_desync = uncoupled.desync_keys,
        uncoupled_fn = uncoupled.post_rebuild_false_negatives,
        uncoupled_failure_rate_pct = uncoupled.failure_rate_pct,
        "completed A/B 3.3 subprocess SIGKILL crash-recovery comparison"
    );

    let metric = AblationMetric {
        extension_name: EXT_AB3.to_string(),
        metric_label: "SIGKILL Crash Lost/Desync Key Rate (5k keys, 5 cycles)".to_string(),
        unit: "pct".to_string(),
        higher_is_better: false,
        key_count: coupled.total_acked,
        query_count: AB3_SIGKILL_CYCLES as usize,
        baseline_value: uncoupled.failure_rate_pct,
        optimized_value: coupled.failure_rate_pct,
        delta_pct: compute_delta_pct(uncoupled.failure_rate_pct, coupled.failure_rate_pct),
        p50_ns: rec_stats.p50_ns,
        p99_ns: rec_stats.p99_ns,
        p999_ns: rec_stats.p999_ns,
        empirical_fpr: 0.0,
        false_negatives: coupled.post_rebuild_false_negatives,
        repeated_fps: 0,
        peak_ram_bytes: 0,
        disk_io_bytes: coupled.disk_bytes,
        scale_factor,
        notes: format!(
            "Coupled({} ACKed, 0 lost KV, 0 lost RM, 0 desync, 0 FN, torn-tail recovery verified) vs Uncoupled({} ACKed, {} lost KV, {} lost RM, {} KV<->RM desync, {} post-rebuild FN, failure_rate={:.2}%; KvStoreConfig/ReverseMapConfig expose sync_writes=false with uncoupled batched RM writes and no CoupledWal)",
            coupled.total_acked,
            uncoupled.total_acked,
            uncoupled.lost_kv_keys,
            uncoupled.lost_rm_keys,
            uncoupled.desync_keys,
            uncoupled.post_rebuild_false_negatives,
            uncoupled.failure_rate_pct,
        ),
    };

    let summary = format!(
        "SIGKILL(5 cycles): coupled=0.00% loss/0 FN ({} ACKed) vs uncoupled={:.2}% lost/desync ({} lost RM, {} post-rebuild FN out of {} ACKed)",
        coupled.total_acked,
        uncoupled.failure_rate_pct,
        uncoupled.lost_rm_keys,
        uncoupled.post_rebuild_false_negatives,
        uncoupled.total_acked,
    );
    (vec![metric], summary)
}

/// Parent coordinator for the 5-cycle `CoupledStorageEngine` `SIGKILL` campaign.
fn run_sigkill_coupled_campaign() -> SigkillCampaignOutcome {
    let dir = TempDir::new().expect("create coupled SIGKILL temp dir");
    let exe = env::current_exe().expect("resolve current test binary");

    let mut all_acked: BTreeMap<Key, Lsn> = BTreeMap::new();
    let mut prev_max_lsn: Lsn = 0;
    let mut recovery_latencies_ns = Vec::with_capacity(AB3_SIGKILL_CYCLES as usize);

    for cycle in 0..AB3_SIGKILL_CYCLES {
        let mut cmd = Command::new(&exe);
        cmd.arg("--exact")
            .arg("run_ablation_proactive_and_storage")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(AB3_SIGKILL_CHILD_ENV, AB3_SIGKILL_MODE_COUPLED)
            .env(AB3_SIGKILL_DIR_ENV, dir.path())
            .env(AB3_SIGKILL_CYCLE_ENV, cycle.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if cycle == AB3_SIGKILL_CYCLES - 1 {
            cmd.env(AB3_SIGKILL_CHECKPOINT_AT_ENV, "500");
        }

        let mut child = cmd.spawn().expect("spawn coupled SIGKILL child");
        let stdout = child.stdout.take().expect("coupled child stdout missing");
        let mut reader = BufReader::new(stdout);

        let mut cycle_acks = 0usize;
        let mut line = String::new();
        while cycle_acks < AB3_SIGKILL_ACKS_PER_CYCLE {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .expect("read coupled child stdout");
            assert!(
                n > 0,
                "coupled child exited prematurely in cycle {cycle} after {cycle_acks} ACKs"
            );
            if let Some((lsn, key)) = parse_ack_line(&line) {
                all_acked.insert(key, lsn);
                cycle_acks += 1;
            }
        }

        // Violently kill child mid-flight (TerminateProcess on Windows / SIGKILL on Unix).
        child.kill().expect("kill coupled child");
        let status = child.wait().expect("wait for killed coupled child");
        assert!(
            !status.success(),
            "cycle {cycle}: killed coupled child unexpectedly exited successfully"
        );

        // Drain complete ACK lines flushed to pipe prior to termination.
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .expect("drain coupled child stdout");
            if n == 0 {
                break;
            }
            if line.ends_with('\n') {
                if let Some((lsn, key)) = parse_ack_line(&line) {
                    all_acked.insert(key, lsn);
                }
            }
        }

        let injected_torn_bytes = inject_torn_tail_for_cycle(dir.path(), cycle, &all_acked);

        let rec_start = Instant::now();
        let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path())
            .unwrap_or_else(|err| panic!("coupled cycle {cycle} open_and_recover failed: {err}"));
        let rec_ns = rec_start.elapsed().as_nanos() as u64;
        recovery_latencies_ns.push(rec_ns);

        if injected_torn_bytes > 0 {
            assert!(
                report.truncated_tail_bytes >= injected_torn_bytes,
                "cycle {cycle}: expected >= {injected_torn_bytes} truncated tail bytes, got {}",
                report.truncated_tail_bytes
            );
        }
        assert!(
            report.last_lsn > prev_max_lsn,
            "cycle {cycle}: WAL LSN did not advance ({} <= {prev_max_lsn})",
            report.last_lsn
        );
        prev_max_lsn = report.last_lsn;

        let kv_keys = engine
            .kv_store()
            .scan_all_keys()
            .expect("scan coupled kv keys");
        let rm_keys = engine
            .reverse_map()
            .scan_all_keys()
            .expect("scan coupled rm keys");
        assert_eq!(
            kv_keys, rm_keys,
            "coupled cycle {cycle}: KV and Reverse Map diverged after open_and_recover"
        );

        let rm_set: BTreeSet<Key> = rm_keys.iter().copied().collect();
        for (&acked_key, &acked_lsn) in &all_acked {
            let expected_val = deterministic_payload(acked_key);
            let actual_val = engine
                .kv_store()
                .get(acked_key)
                .expect("read recovered coupled kv");
            assert_eq!(
                actual_val.as_deref(),
                Some(expected_val.as_slice()),
                "coupled cycle {cycle}: lost ACKed key {acked_key} (LSN {acked_lsn}) in KV"
            );
            assert!(
                rm_set.contains(&acked_key),
                "coupled cycle {cycle}: lost ACKed key {acked_key} (LSN {acked_lsn}) in Reverse Map"
            );
        }

        // Rebuild real AerisFilter via rebuild_filter_and_reconcile (starting at q=12 so
        // it auto-expands to q=13 across 5,000+ keys) and verify 0 false negatives.
        let filter_cfg = FilterConfig::new(12, 12, 32, 17, 1.0 / 256.0)
            .expect("valid sigkill rebuild FilterConfig")
            .with_hash_seed(AB3_HASH_SEED);
        let mut rebuilt_filter =
            AerisFilter::new(filter_cfg).expect("create AerisFilter for rebuild");
        let loaded = engine
            .rebuild_filter_and_reconcile(&mut rebuilt_filter)
            .expect("rebuild_filter_and_reconcile must succeed");
        assert_eq!(loaded, kv_keys.len());

        let acked_slice: Vec<Key> = all_acked.keys().copied().collect();
        let cycle_fns = count_false_negatives_filter(&rebuilt_filter, &acked_slice);
        assert_eq!(
            cycle_fns, 0,
            "coupled cycle {cycle}: rebuild_filter_and_reconcile produced {cycle_fns} false negatives"
        );
    }

    let disk_bytes = directory_size_bytes(dir.path());

    SigkillCampaignOutcome {
        total_acked: all_acked.len(),
        lost_kv_keys: 0,
        lost_rm_keys: 0,
        desync_keys: 0,
        post_rebuild_false_negatives: 0,
        failure_rate_pct: 0.0,
        recovery_latencies_ns,
        disk_bytes,
    }
}

/// Parent coordinator for the 5-cycle Naive Uncoupled `SIGKILL` campaign
/// (`ReferenceKvStore` + `RocksDbReverseMap`, `sync_writes: false`, uncoupled batched
/// Reverse Map writes, zero `CoupledWal`).
fn run_sigkill_uncoupled_campaign() -> SigkillCampaignOutcome {
    let dir = TempDir::new().expect("create uncoupled SIGKILL temp dir");
    let exe = env::current_exe().expect("resolve current test binary");

    let mut all_acked: BTreeSet<Key> = BTreeSet::new();
    let mut cumulative_lost_kv = 0usize;
    let mut cumulative_lost_rm = 0usize;
    let mut cumulative_desync = 0usize;
    let mut final_post_rebuild_fns = 0u64;
    let mut recovery_latencies_ns = Vec::with_capacity(AB3_SIGKILL_CYCLES as usize);

    for cycle in 0..AB3_SIGKILL_CYCLES {
        let mut cmd = Command::new(&exe);
        cmd.arg("--exact")
            .arg("run_ablation_proactive_and_storage")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(AB3_SIGKILL_CHILD_ENV, AB3_SIGKILL_MODE_UNCOUPLED)
            .env(AB3_SIGKILL_DIR_ENV, dir.path())
            .env(AB3_SIGKILL_CYCLE_ENV, cycle.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().expect("spawn uncoupled SIGKILL child");
        let stdout = child.stdout.take().expect("uncoupled child stdout missing");
        let mut reader = BufReader::new(stdout);

        let mut cycle_acks = 0usize;
        let mut line = String::new();
        while cycle_acks < AB3_SIGKILL_ACKS_PER_CYCLE {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .expect("read uncoupled child stdout");
            assert!(
                n > 0,
                "uncoupled child exited prematurely in cycle {cycle} after {cycle_acks} ACKs"
            );
            if let Some((_seq, key)) = parse_ack_line(&line) {
                all_acked.insert(key);
                cycle_acks += 1;
            }
        }

        child.kill().expect("kill uncoupled child");
        let status = child.wait().expect("wait for killed uncoupled child");
        assert!(
            !status.success(),
            "cycle {cycle}: killed uncoupled child unexpectedly exited successfully"
        );

        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .expect("drain uncoupled child stdout");
            if n == 0 {
                break;
            }
            if line.ends_with('\n') {
                if let Some((_seq, key)) = parse_ack_line(&line) {
                    all_acked.insert(key);
                }
            }
        }

        // Reopen both independent stores with zero CoupledWal recovery.
        let reopen_start = Instant::now();
        let kv = ReferenceKvStore::open_with_config(
            dir.path().join("kv_db"),
            KvStoreConfig::for_wal_coupled(),
        )
        .expect("reopen uncoupled kv after SIGKILL");
        let rm = RocksDbReverseMap::open(
            dir.path().join("reverse_map_db"),
            ReverseMapConfig::for_wal_coupled(),
        )
        .expect("reopen uncoupled rm after SIGKILL");
        recovery_latencies_ns.push(reopen_start.elapsed().as_nanos() as u64);

        let kv_set: BTreeSet<Key> = kv
            .scan_all_keys()
            .expect("scan uncoupled kv")
            .into_iter()
            .collect();
        let rm_set: BTreeSet<Key> = rm
            .scan_all_keys()
            .expect("scan uncoupled rm")
            .into_iter()
            .collect();

        let lost_kv = all_acked.iter().filter(|k| !kv_set.contains(k)).count();
        let lost_rm = all_acked.iter().filter(|k| !rm_set.contains(k)).count();
        let desync = kv_set.symmetric_difference(&rm_set).count();

        cumulative_lost_kv = lost_kv;
        cumulative_lost_rm = lost_rm;
        cumulative_desync = desync;

        warn!(
            cycle,
            total_acked = all_acked.len(),
            lost_kv,
            lost_rm,
            kv_rm_desync = desync,
            "uncoupled SIGKILL crash audit"
        );

        if cycle == AB3_SIGKILL_CYCLES - 1 {
            // Rebuild an AerisFilter strictly from the surviving uncoupled Reverse Map
            // (matching AsyncRebuildCoordinator) to measure real post-rebuild false negatives.
            let rm_keys_vec: Vec<Key> = rm_set.iter().copied().collect();
            let rebuild_dir = TempDir::new().expect("create temp dir for uncoupled filter rebuild");
            let rebuild_rm = RocksDbReverseMap::open(
                rebuild_dir.path().join("rm"),
                ReverseMapConfig::for_wal_coupled(),
            )
            .expect("open temp rm for uncoupled rebuild");
            let filter_cfg = FilterConfig::new(14, 12, 32, 17, 1.0 / 256.0)
                .expect("valid q=14 config for uncoupled rebuild")
                .with_hash_seed(AB3_HASH_SEED);
            let rebuilt = AerisFilter::bulk_load(filter_cfg, &rm_keys_vec, &rebuild_rm)
                .expect("bulk_load uncoupled surviving Reverse Map keys");

            let acked_vec: Vec<Key> = all_acked.iter().copied().collect();
            final_post_rebuild_fns = count_false_negatives_filter(&rebuilt, &acked_vec);
        }
    }

    let disk_bytes = directory_size_bytes(dir.path());
    let failed_keys = cumulative_lost_kv
        .max(cumulative_lost_rm)
        .max(cumulative_desync);
    let failure_rate_pct = if all_acked.is_empty() {
        0.0
    } else {
        (failed_keys as f64 / all_acked.len() as f64) * 100.0
    };

    SigkillCampaignOutcome {
        total_acked: all_acked.len(),
        lost_kv_keys: cumulative_lost_kv,
        lost_rm_keys: cumulative_lost_rm,
        desync_keys: cumulative_desync,
        post_rebuild_false_negatives: final_post_rebuild_fns,
        failure_rate_pct,
        recovery_latencies_ns,
        disk_bytes,
    }
}

/// Child subprocess driver for both `"coupled"` and `"uncoupled"` `SIGKILL` modes.
fn run_sigkill_child_driver(mode: &str) {
    let dir = env::var(AB3_SIGKILL_DIR_ENV).expect("AERIS_AB3_SIGKILL_DIR must be set");
    let cycle: u64 = env::var(AB3_SIGKILL_CYCLE_ENV)
        .expect("AERIS_AB3_SIGKILL_CYCLE must be set")
        .parse()
        .expect("AERIS_AB3_SIGKILL_CYCLE must parse as u64");
    let checkpoint_at: Option<usize> = env::var(AB3_SIGKILL_CHECKPOINT_AT_ENV)
        .ok()
        .and_then(|v| v.parse().ok());

    let mut rng = ChaCha8Rng::seed_from_u64(0x516B_0000_u64.wrapping_add(cycle.wrapping_mul(997)));
    let hasher = PartitionHasher::new(AB3_HASH_SEED);
    let stdout = io::stdout();
    let mut out = stdout.lock();

    match mode {
        AB3_SIGKILL_MODE_COUPLED => {
            let (engine, _report) = CoupledStorageEngine::open_and_recover(&dir)
                .expect("coupled child open_and_recover failed");
            for iter in 0..AB3_SIGKILL_CHILD_CAP {
                let key = next_cycle_key(cycle, &mut rng);
                let payload = deterministic_payload(key);
                let fp = Fingerprint::from_hasher(&hasher, key >> 5, AB3_FP_BITS);
                let lsn = engine
                    .put(key, &payload, &fp)
                    .expect("coupled child put failed");
                writeln!(out, "ACK:{lsn}:{key}").expect("coupled child stdout write failed");
                out.flush().expect("coupled child stdout flush failed");
                if checkpoint_at == Some(iter + 1) {
                    engine
                        .checkpoint()
                        .expect("coupled child checkpoint failed");
                }
            }
        }
        AB3_SIGKILL_MODE_UNCOUPLED => {
            let root = Path::new(&dir);
            let kv = ReferenceKvStore::open_with_config(
                root.join("kv_db"),
                KvStoreConfig::for_wal_coupled(),
            )
            .expect("uncoupled child open kv failed");
            let rm = RocksDbReverseMap::open(
                root.join("reverse_map_db"),
                ReverseMapConfig::for_wal_coupled(),
            )
            .expect("uncoupled child open rm failed");

            let mut rm_pending: Vec<(Fingerprint, Key)> =
                Vec::with_capacity(AB3_UNCOUPLED_RM_BATCH_SIZE);
            for iter in 0..AB3_SIGKILL_CHILD_CAP {
                let key = next_cycle_key(cycle, &mut rng);
                let payload = deterministic_payload(key);
                let fp = Fingerprint::from_hasher(&hasher, key >> 5, AB3_FP_BITS);

                // Uncoupled write path: primary KV write returns and is ACKed while Reverse Map
                // updates flush independently in batches with zero CoupledWal coordination.
                kv.put(key, &payload)
                    .expect("uncoupled child kv put failed");
                rm_pending.push((fp, key));
                if rm_pending.len() >= AB3_UNCOUPLED_RM_BATCH_SIZE {
                    rm.append_batch(&rm_pending)
                        .expect("uncoupled child rm append_batch failed");
                    rm_pending.clear();
                }

                let seq = (cycle * (AB3_SIGKILL_CHILD_CAP as u64)) + (iter as u64) + 1;
                writeln!(out, "ACK:{seq}:{key}").expect("uncoupled child stdout write failed");
                out.flush().expect("uncoupled child stdout flush failed");
            }
        }
        other => panic!("unsupported {AB3_SIGKILL_CHILD_ENV} mode: '{other}'"),
    }
}

// =============================================================================
// Shared Pure Helpers
// =============================================================================

/// Deterministically shuffles `keys` using Fisher-Yates seeded by `seed`.
fn shuffled_keys(keys: &[Key], seed: u64) -> Vec<Key> {
    let mut out = keys.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in (1..out.len()).rev() {
        let j = rng.gen_range(0..=i);
        out.swap(i, j);
    }
    out
}

/// Counts false negatives on `filter` across all `keys` (point + enclosing range of radius 8).
fn count_false_negatives_filter(filter: &AerisFilter, keys: &[Key]) -> u64 {
    let mut fns = 0u64;
    for &k in keys {
        if !filter.query_point(k) || !filter.query_range(enclosing_range(k, 8)) {
            fns += 1;
        }
    }
    fns
}

/// Counts false negatives on `RcuAerisFilter` across all `keys` (point + enclosing range of radius 8).
fn count_false_negatives_rcu(filter: &RcuAerisFilter, keys: &[Key]) -> u64 {
    let mut fns = 0u64;
    for &k in keys {
        if !filter.query_point(k) || !filter.query_range(enclosing_range(k, 8)) {
            fns += 1;
        }
    }
    fns
}

/// Generates a cycle-partitioned deterministic 48-bit key disjoint across crash cycles.
#[inline]
fn next_cycle_key(cycle: u64, rng: &mut ChaCha8Rng) -> Key {
    ((cycle + 1) << 40) | (rng.gen::<u64>() & 0x0000_00FF_FFFF_FFFF)
}

/// Builds a deterministic 24-byte KV payload for `key`.
fn deterministic_payload(key: Key) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(b"AERIS-KV");
    out.extend_from_slice(&key.to_be_bytes());
    out.extend_from_slice(&(!key).to_le_bytes());
    out
}

/// Parses an `ACK:<lsn_or_seq>:<key>` stdout line emitted by a `SIGKILL` child driver.
fn parse_ack_line(line: &str) -> Option<(Lsn, Key)> {
    let rest = line.trim().strip_prefix("ACK:")?;
    let (lsn_str, key_str) = rest.split_once(':')?;
    let lsn: Lsn = lsn_str.parse().ok()?;
    let key: Key = key_str.parse().ok()?;
    Some((lsn, key))
}

/// Appends realistic torn-tail WAL bytes on cycles 1..=3 (mid-header, mid-payload, bad CRC32)
/// to exercise `CoupledWal` torn-tail truncation during crash recovery.
fn inject_torn_tail_for_cycle(dir: &Path, cycle: u64, acked: &BTreeMap<Key, Lsn>) -> u64 {
    let wal_path = dir.join(WAL_FILE_NAME);
    let next_lsn = acked.values().copied().max().unwrap_or(0) + 10_000;
    let hasher = PartitionHasher::new(AB3_HASH_SEED);
    let fake_key = 0xDEAD_BEEF_0000_0000 | cycle;
    let fake_op = WalOp::Put {
        key: fake_key,
        value: vec![0xAB; 64],
        fp: Fingerprint::from_hasher(&hasher, fake_key >> 5, AB3_FP_BITS),
    };
    let (mut frame, _) = encode_wal_frame(next_lsn, &fake_op).expect("encode torn-tail WAL frame");

    let torn_bytes: Vec<u8> = match cycle {
        0 | 4 => return 0,
        1 => {
            let mut partial = Vec::with_capacity(11);
            partial.extend_from_slice(&WAL_MAGIC);
            partial.push(WAL_FORMAT_VERSION);
            partial.extend_from_slice(&[0x40, 0x00, 0x00, 0x00, 0x01, 0x02]);
            partial
        }
        2 => frame[..WAL_HEADER_LEN + 20].to_vec(),
        3 => {
            let last = frame.len() - 1;
            frame[last] ^= 0xFF;
            frame
        }
        _ => return 0,
    };

    let mut file = OpenOptions::new()
        .append(true)
        .open(&wal_path)
        .expect("open WAL for torn-tail injection");
    file.write_all(&torn_bytes)
        .expect("write torn-tail WAL bytes");
    file.sync_all().expect("fsync torn-tail WAL bytes");
    torn_bytes.len() as u64
}

/// Returns current Unix timestamp in milliseconds (`0` if system clock precedes epoch).
#[inline]
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
