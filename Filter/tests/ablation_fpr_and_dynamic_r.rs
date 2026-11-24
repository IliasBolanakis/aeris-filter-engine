//! DRAM-Scale FPR & Throughput Audit (Suite 1A) and Dynamic-R + FP Log Replay
//! Ablation (Suite 1B) for the Aeris-Based Optimized Range Filtering System.
//!
//! # Suites Executed
//! 1. **Suite 1A — DRAM-Scale FPR & Throughput Audit:**
//!    - Indexes `host.scaled_count(10_000_000)` unique uniform 48-bit keys and
//!      evaluates `host.scaled_count(5_000_000)` unique verified-empty range queries
//!      (`R = 32, m = 5`) at both `epsilon = 2^-4` (`1/16`) and `epsilon = 2^-8` (`1/256`).
//!    - Enforces `filter.memory_usage_bytes() > 24 MiB` so lookups genuinely spill
//!      past L3 cache into main DRAM.
//!    - Measures pre-expansion (`q = 24`, 0 expansions) and post-8-consecutive-2x-expansions
//!      (`q = 16 -> 24`, exercising InfiniFilter bit-splitting and real `RocksDbReverseMap`
//!      selective rejuvenation) empirical FPR, uninstrumented tight-loop throughput
//!      (`ops/sec`), and deterministic stride-sampled `p50`/`p99`/`p99.9` latency.
//!    - Verifies `0` false negatives across all inserted keys (point + enclosing range),
//!      `0` repeated false positives across expansion and real `RocksDbReverseMap`
//!      adaptations, and compares unadapted naive filtering against monotonic Aeris
//!      under adversarial 10% repeat-FP probing.
//!
//! 2. **Suite 1B — A/B 1: Dynamic `R` + Persistent `FalsePositiveLog` Replay vs. Static `R`:**
//!    - Indexes `1,000,000` unique keys and executes `2,000,000` shifting-`R` empty
//!      queries across `[16, 256, 32, 1024]` (`SHIFTING_R_PHASES`) under fixed per-slot
//!      RAM budget `C = 17` (`epsilon = 1/16`, `r_min = 7`).
//!    - Compares equal-RAM Static `R = 16` (`m = 4, r = 13, C = 17`) and paper-minimum
//!      Static `R = 16` (`m = 4, r = 7, C = 11`) baselines against the full real
//!      Control Plane (`AerisFilter` + generational `RocksDbReverseMap` +
//!      `SlidingWindowEstimator` + `DiscreteCostOptimizer` + `AsyncRebuildCoordinator` +
//!      `DeltaBuffer` + persistent `FalsePositiveLog`).
//!    - Proves `0` false negatives across all `1,000,000` keys, `0` repeated FPs across
//!      all Dynamic-`R` rebuilds via real `FalsePositiveLog` replay, and measures exact
//!      RocksDB Reverse Map rebuild latency and FP Log replay latency.

mod ablation_common;

use ablation_common::{
    compute_delta_pct, directory_size_bytes, enclosing_range, generate_adversarial_repeat_fp_trace,
    generate_shifting_r_trace, generate_uniform_keys_oracle, generate_unique_uniform_empty_queries,
    init_ablation_tracing, write_worker_report, AblationMetric, HostMetadata, LatencyStats,
    ShiftingRPhase, SortedKeyOracle, WorkerAblationReport, DEFAULT_ADVERSARIAL_REPEAT_RATE,
    SHIFTING_R_PHASES,
};
use aeris_control::{
    AsyncRebuildCoordinator, DiscreteCostOptimizer, EstimatorConfig, FalsePositiveLog, FpLogConfig,
    IoThrottleConfig, RebuildReason, SlidingWindowEstimator,
};
use aeris_core::AerisFilter;
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, AerisResult, CostModelParams, CostOptimizer, FilterConfig, Fingerprint, Key,
    QueryRange, RangeFilter, ReverseMapStore, WorkloadEstimator, MAX_QUOTIENT_BITS,
};
use std::collections::HashSet;
use std::fs;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tracing::{debug, info};

/// Worker identifier persisted to `Filter/target/ablation_worker1_fpr_dynamic_r.json`.
const WORKER_NAME: &str = "worker1_fpr_dynamic_r";

/// Minimum L3 cache spill floor (`24 MiB`) enforced for DRAM-scale audits.
const L3_SPILL_FLOOR_BYTES: usize = 24 * 1024 * 1024;

/// Target key count at `scale_factor = 1.0` for Suite 1A DRAM-scale audit.
const DRAM_BASE_KEYS: usize = 10_000_000;

/// Target unique empty query count at `scale_factor = 1.0` for Suite 1A DRAM-scale audit.
const DRAM_BASE_QUERIES: usize = 5_000_000;

/// Target adversarial repeat-FP trace size at `scale_factor = 1.0` for Suite 1A.
const DRAM_ADVERSARIAL_BASE_QUERIES: usize = 500_000;

/// Query range length `R = 32` (`m = 5`) for Suite 1A.
const DRAM_RANGE_LEN: u64 = 32;

/// Starting quotient bit width (`N = 2^16`) before 8 consecutive `2x` expansions.
const DRAM_START_Q: u8 = 16;

/// Target quotient bit width (`N = 2^24`) after 8 consecutive `2x` expansions.
const DRAM_TARGET_Q: u8 = 24;

/// Number of consecutive `2x` expansions (`24 - 16 = 8`).
const DRAM_EXPANSION_COUNT: usize = (DRAM_TARGET_Q - DRAM_START_Q) as usize;

/// Chunk size for batched `RocksDbReverseMap::append_batch` writes during 10M-key loading.
const RM_BATCH_CHUNK_SIZE: usize = 250_000;

/// RocksDB write buffer size (`64 MiB`) tuned for high-throughput bulk ingestion.
const RM_WRITE_BUFFER_BYTES: usize = 64 * 1024 * 1024;

/// Target sample count for deterministic stride latency sampling (keeps `Instant::now()`
/// strictly outside the uninstrumented throughput loop).
const LATENCY_SAMPLE_TARGET: usize = 100_000;

/// Number of strided enclosing-range checks executed during 10M-key zero-FN verification.
const ENCLOSING_RANGE_VERIFY_SAMPLES: usize = 200_000;

/// Pool size of real observed FPs adapted before expansion #8 and used for 10% adversarial probing.
const ADVERSARIAL_FP_POOL_TARGET: usize = 256;

/// Key count for Suite 1B (A/B 1: Dynamic `R` + FP Log Replay).
const AB1_KEY_COUNT: usize = 1_000_000;

/// Total shifting-`R` query count across the 4 phases (`500,000` queries per phase).
const AB1_TOTAL_QUERIES: usize = 2_000_000;

/// Quotient bit width for Suite 1B (`2^21 = 2,097,152` slots; `alpha ~= 0.48` at 1M keys).
const AB1_Q: u8 = 21;

/// Fixed payload bit budget `C = r + m = 17` supporting `m in {4, 5, 8, 10}` with `r >= 7`.
const AB1_PAYLOAD_BUDGET_C: u8 = 17;

/// Target worst-case FPR `epsilon = 2^-4` (`r_min = 7`) for Suite 1B.
const AB1_EPSILON: f64 = 1.0 / 16.0;

/// Optimizer evaluation cadence (in queries) during Suite 1B shifting-`R` execution.
const AB1_OPTIMIZER_POLL_STRIDE: usize = 256;

/// Number of concurrent foreground queries served off `F_old` while `"aeris-bg-rebuild"`
/// runs on its background OS thread.
const AB1_CONCURRENT_BURST_QUERIES: usize = 1_024;

/// Deterministic RNG seeds for reproducible datasets and traces.
const SEED_DRAM_KEYS: u64 = 0xAE15_1A01_0000_0001;
const SEED_DRAM_QUERIES: u64 = 0xAE15_1A02_0000_0002;
const SEED_DRAM_ADV_BASE: u64 = 0xAE15_1A03_0000_0003;
const SEED_AB1_KEYS: u64 = 0xAE15_1B01_0000_0001;
const SEED_AB1_TRACE: u64 = 0xAE15_1B02_0000_0002;

/// Pure query evaluation result over a slice of verified-empty range queries.
struct QueryBenchResult {
    fp_count: u64,
    empirical_fpr: f64,
    throughput_ops_sec: f64,
    latency: LatencyStats,
}

/// Summary of progressive `q = 16 -> 24` growth across 8 consecutive `2x` expansions.
struct ProgressiveBuildSummary {
    filter: AerisFilter,
    reverse_map: RocksDbReverseMap,
    total_boxes_rejuvenated: usize,
    total_keys_rejuvenated: usize,
    pre_expand8_adapted_count: usize,
    post_expand8_repeated_fps: u64,
}

/// Outcome of running an adversarial 10% repeat-FP probing comparison.
struct AdversarialComparison {
    pool_size: usize,
    trace_len: usize,
    naive_fpr: f64,
    naive_repeated_fps: u64,
    aeris_fpr: f64,
    aeris_repeated_fps: u64,
    aeris_adaptations: u64,
    aeris_latency: LatencyStats,
    aeris_false_negatives: u64,
}

/// Per-phase telemetry captured for a single shifting-`R` phase in Suite 1B.
struct PhaseRunSummary {
    range_len: u64,
    query_count: usize,
    fp_count: u64,
    empirical_fpr: f64,
    throughput_ops_sec: f64,
    latency: LatencyStats,
    active_m: u8,
    active_r: u8,
    rebuilds_triggered: usize,
}

/// Complete telemetry for a single rebuild executed by [`AsyncRebuildCoordinator`].
struct DynamicRebuildRecord {
    phase_index: usize,
    phase_range_len: u64,
    old_m: u8,
    new_m: u8,
    new_r: u8,
    keys_loaded: usize,
    duration_ns: u64,
    fp_replay_entries: usize,
    fp_collisions_resolved: usize,
    fp_replay_latency_ns: u64,
    observed_bulk_load_speed: f64,
    reprobe_checked: usize,
    reprobe_repeated_fps: u64,
}

/// Aggregated results for a static-`R` baseline across all 4 shifting-`R` phases.
struct StaticBaselineSummary {
    phases: Vec<PhaseRunSummary>,
    overall_fpr: f64,
    overall_qps: f64,
    overall_latency: LatencyStats,
    filter_ram_bytes: u64,
    false_negatives: u64,
}

/// Aggregated results for Optimized Dynamic-`R` Aeris across all 4 shifting-`R` phases.
struct DynamicAerisSummary {
    phases: Vec<PhaseRunSummary>,
    rebuilds: Vec<DynamicRebuildRecord>,
    overall_fpr: f64,
    overall_qps: f64,
    overall_latency: LatencyStats,
    peak_filter_ram_bytes: u64,
    disk_io_bytes: u64,
    total_logged_fps: usize,
    total_repeated_fps: u64,
    false_negatives: u64,
}

/// Single orchestrated integration test executing Suite 1A (DRAM-Scale FPR & Throughput
/// Audit) and Suite 1B (A/B 1: Dynamic `R` + FP Log Replay vs. Static `R`) and writing
/// the merged JSON report to `Filter/target/ablation_worker1_fpr_dynamic_r.json`.
#[test]
fn run_ablation_fpr_and_dynamic_r() {
    init_ablation_tracing();
    let mut host = HostMetadata::detect();
    let mut metrics: Vec<AblationMetric> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    run_suite_1a_dram_fpr_audit(&mut host, &mut metrics, &mut notes);
    run_suite_1b_dynamic_r_ablation(&mut host, &mut metrics, &mut notes);

    host.refresh_peak_rss();
    let report = WorkerAblationReport {
        worker_name: WORKER_NAME.to_string(),
        host,
        metrics,
        notes,
    };
    let path = write_worker_report(&report);
    assert!(path.exists(), "worker report JSON must exist on disk");
    info!(
        worker = WORKER_NAME,
        metric_count = report.metrics.len(),
        peak_rss_mb = (report.host.peak_rss_bytes as f64) / (1024.0 * 1024.0),
        path = %path.display(),
        "completed worker1_fpr_dynamic_r ablation suite"
    );
}

// =============================================================================
// Suite 1A: DRAM-Scale FPR & Throughput Audit (10M keys, 5M unique queries)
// =============================================================================

/// Executes Suite 1A across both `epsilon = 2^-4` and `epsilon = 2^-8` with `R = 32` (`m = 5`).
fn run_suite_1a_dram_fpr_audit(
    host: &mut HostMetadata,
    metrics: &mut Vec<AblationMetric>,
    notes: &mut Vec<String>,
) {
    let key_count = host.scaled_count(DRAM_BASE_KEYS);
    let query_count = host.scaled_count(DRAM_BASE_QUERIES);
    let adv_query_count = host.scaled_count(DRAM_ADVERSARIAL_BASE_QUERIES);

    info!(
        key_count,
        query_count,
        adv_query_count,
        range_len = DRAM_RANGE_LEN,
        scale_factor = host.scale_factor,
        "starting Suite 1A: DRAM-scale FPR and throughput audit"
    );

    let dataset_start = Instant::now();
    let oracle = generate_uniform_keys_oracle(SEED_DRAM_KEYS, key_count);
    let queries = generate_unique_uniform_empty_queries(
        SEED_DRAM_QUERIES,
        &oracle,
        DRAM_RANGE_LEN,
        query_count,
    );
    info!(
        key_count = oracle.len(),
        query_count = queries.len(),
        dataset_gen_ms = dataset_start.elapsed().as_millis() as u64,
        "generated deterministic DRAM-scale oracle and unique empty queries"
    );

    let cases: [(&str, f64, u64); 2] = [
        ("eps_2e-4", 1.0 / 16.0, SEED_DRAM_ADV_BASE ^ 0x04),
        ("eps_2e-8", 1.0 / 256.0, SEED_DRAM_ADV_BASE ^ 0x08),
    ];

    for (tag, epsilon, adv_seed) in cases {
        run_single_dram_epsilon_audit(
            tag,
            epsilon,
            adv_seed,
            &oracle,
            &queries,
            adv_query_count,
            host,
            metrics,
            notes,
        );
    }
}

/// Runs the complete pre-expansion, post-8x-expansion, zero-FN, zero-repeated-FP,
/// and adversarial 10% repeat-FP audit for one target `epsilon`.
#[allow(clippy::too_many_arguments)]
fn run_single_dram_epsilon_audit(
    tag: &str,
    epsilon: f64,
    adv_seed: u64,
    oracle: &SortedKeyOracle,
    queries: &[QueryRange],
    adv_query_count: usize,
    host: &mut HostMetadata,
    metrics: &mut Vec<AblationMetric>,
    notes: &mut Vec<String>,
) {
    let pre_cfg = FilterConfig::with_minimum_bits(DRAM_TARGET_Q, DRAM_RANGE_LEN, epsilon)
        .expect("valid DRAM pre-expansion FilterConfig");
    let pre_build_start = Instant::now();
    let pre_filter = build_fresh_dram_filter(&pre_cfg, oracle.as_slice())
        .expect("fresh DRAM filter construction succeeds");
    let pre_filter_ram = pre_filter.memory_usage_bytes();
    assert!(
        pre_filter_ram > L3_SPILL_FLOOR_BYTES,
        "pre-expansion filter RAM ({pre_filter_ram} B) must exceed 24 MiB L3 spill floor"
    );
    assert_eq!(pre_filter.len(), oracle.len());
    assert_eq!(pre_filter.expansion_cycles(), 0);

    let pre_bench = benchmark_empty_queries(&pre_filter, queries, LATENCY_SAMPLE_TARGET);
    let pre_fn = verify_zero_false_negatives(&pre_filter, oracle.as_slice(), 15);
    assert_eq!(
        pre_fn, 0,
        "pre-expansion DRAM filter ({tag}) must produce 0 false negatives"
    );
    assert!(
        pre_bench.empirical_fpr <= epsilon,
        "pre-expansion empirical FPR ({:.6}) exceeded target epsilon ({:.6}) for {tag}",
        pre_bench.empirical_fpr,
        epsilon
    );

    info!(
        tag,
        epsilon,
        r = pre_cfg.r,
        m = pre_cfg.m,
        q = pre_cfg.q,
        ram_mb = (pre_filter_ram as f64) / (1024.0 * 1024.0),
        load_factor = pre_filter.load_factor(),
        build_ms = pre_build_start.elapsed().as_millis() as u64,
        pre_fpr = pre_bench.empirical_fpr,
        pre_qps = pre_bench.throughput_ops_sec,
        p50_ns = pre_bench.latency.p50_ns,
        p99_ns = pre_bench.latency.p99_ns,
        p999_ns = pre_bench.latency.p999_ns,
        "completed pre-expansion DRAM audit"
    );

    // Reclaim pre-expansion RAM before allocating the progressive 8x-expansion filter + RocksDB.
    drop(pre_filter);
    host.refresh_peak_rss();

    let rm_dir = TempDir::new().expect("create temp dir for DRAM RocksDbReverseMap");
    let prog_start = Instant::now();
    let mut progressive =
        build_progressive_8x_expanded_filter(epsilon, oracle.as_slice(), queries, rm_dir.path())
            .expect("progressive 8x-expansion filter build succeeds");
    progressive
        .reverse_map
        .flush()
        .expect("flush RocksDbReverseMap after progressive load");

    let post_filter_ram = progressive.filter.memory_usage_bytes();
    let rm_disk_bytes = directory_size_bytes(rm_dir.path());
    assert!(
        post_filter_ram > L3_SPILL_FLOOR_BYTES,
        "post-8x-expansion filter RAM ({post_filter_ram} B) must exceed 24 MiB L3 spill floor"
    );
    assert_eq!(
        progressive.filter.expansion_cycles(),
        DRAM_EXPANSION_COUNT as u64
    );
    assert_eq!(progressive.filter.config().q, DRAM_TARGET_Q);
    assert_eq!(progressive.filter.len(), oracle.len());
    assert_eq!(
        progressive.post_expand8_repeated_fps, 0,
        "monotonic adaptivity violated across expansion #8 for {tag}"
    );
    if pre_cfg.r <= DRAM_EXPANSION_COUNT as u8 {
        assert!(
            progressive.total_boxes_rejuvenated > 0,
            "expected selective rejuvenation of age==r boxes during 8 expansions for {tag}"
        );
    }

    let post_bench = benchmark_empty_queries(&progressive.filter, queries, LATENCY_SAMPLE_TARGET);
    let post_fn = verify_zero_false_negatives(&progressive.filter, oracle.as_slice(), 15);
    assert_eq!(
        post_fn, 0,
        "post-8x-expansion DRAM filter ({tag}) must produce 0 false negatives"
    );
    assert!(
        post_bench.empirical_fpr <= epsilon,
        "post-8x-expansion empirical FPR ({:.6}) exceeded target epsilon ({:.6}) for {tag}",
        post_bench.empirical_fpr,
        epsilon
    );

    info!(
        tag,
        epsilon,
        ram_mb = (post_filter_ram as f64) / (1024.0 * 1024.0),
        load_factor = progressive.filter.load_factor(),
        build_and_expand_ms = prog_start.elapsed().as_millis() as u64,
        boxes_rejuvenated = progressive.total_boxes_rejuvenated,
        keys_rejuvenated = progressive.total_keys_rejuvenated,
        post_fpr = post_bench.empirical_fpr,
        post_qps = post_bench.throughput_ops_sec,
        p50_ns = post_bench.latency.p50_ns,
        p99_ns = post_bench.latency.p99_ns,
        p999_ns = post_bench.latency.p999_ns,
        rm_disk_mb = (rm_disk_bytes as f64) / (1024.0 * 1024.0),
        "completed post-8x-expansion DRAM audit"
    );

    let adv = run_adversarial_repeat_fp_audit(
        &mut progressive.filter,
        &progressive.reverse_map,
        oracle,
        queries,
        adv_seed,
        adv_query_count,
    )
    .expect("adversarial repeat-FP audit succeeds");
    assert_eq!(adv.aeris_repeated_fps, 0, "Aeris must have 0 repeated FPs");
    assert_eq!(
        adv.aeris_false_negatives, 0,
        "Aeris must preserve 0 false negatives after adversarial adaptations"
    );

    host.refresh_peak_rss();
    let rm_disk_after_adv = directory_size_bytes(rm_dir.path());
    let final_filter_ram = progressive.filter.memory_usage_bytes() as u64;

    info!(
        tag,
        pool_size = adv.pool_size,
        trace_len = adv.trace_len,
        naive_fpr = adv.naive_fpr,
        naive_repeated_fps = adv.naive_repeated_fps,
        aeris_fpr = adv.aeris_fpr,
        aeris_repeated_fps = adv.aeris_repeated_fps,
        aeris_adaptations = adv.aeris_adaptations,
        "completed adversarial 10% repeat-FP audit"
    );

    metrics.push(AblationMetric {
        extension_name: format!("DRAM_10M_FPR_{tag}"),
        metric_label: format!("DRAM Pre-Expansion Empirical FPR vs Target ({tag})"),
        unit: "ratio".to_string(),
        higher_is_better: false,
        key_count: oracle.len(),
        query_count: queries.len(),
        baseline_value: epsilon,
        optimized_value: pre_bench.empirical_fpr,
        delta_pct: compute_delta_pct(epsilon, pre_bench.empirical_fpr),
        p50_ns: pre_bench.latency.p50_ns,
        p99_ns: pre_bench.latency.p99_ns,
        p999_ns: pre_bench.latency.p999_ns,
        empirical_fpr: pre_bench.empirical_fpr,
        false_negatives: pre_fn,
        repeated_fps: 0,
        peak_ram_bytes: pre_filter_ram as u64,
        disk_io_bytes: 0,
        scale_factor: host.scale_factor,
        notes: format!(
            "q={}, r={}, m={}, C={}, filter_ram_mb={:.2} (>24 MiB L3 spill), pre_qps={:.0}, fps={}",
            pre_cfg.q,
            pre_cfg.r,
            pre_cfg.m,
            pre_cfg.payload_budget_c,
            (pre_filter_ram as f64) / (1024.0 * 1024.0),
            pre_bench.throughput_ops_sec,
            pre_bench.fp_count
        ),
    });

    metrics.push(AblationMetric {
        extension_name: format!("DRAM_10M_Post8xExp_FPR_{tag}"),
        metric_label: format!("DRAM Post-8x-Expansion Empirical FPR vs Target ({tag})"),
        unit: "ratio".to_string(),
        higher_is_better: false,
        key_count: oracle.len(),
        query_count: queries.len(),
        baseline_value: epsilon,
        optimized_value: post_bench.empirical_fpr,
        delta_pct: compute_delta_pct(epsilon, post_bench.empirical_fpr),
        p50_ns: post_bench.latency.p50_ns,
        p99_ns: post_bench.latency.p99_ns,
        p999_ns: post_bench.latency.p999_ns,
        empirical_fpr: post_bench.empirical_fpr,
        false_negatives: post_fn,
        repeated_fps: progressive.post_expand8_repeated_fps,
        peak_ram_bytes: post_filter_ram as u64,
        disk_io_bytes: rm_disk_bytes,
        scale_factor: host.scale_factor,
        notes: format!(
            "q={DRAM_START_Q}->{DRAM_TARGET_Q} (8x2x expansions), pre_fpr={:.6}, post_fpr={:.6}, boxes_rejuvenated={}, keys_rejuvenated={}, pre_exp8_adapted_reprobed={}, filter_ram_mb={:.2}, process_peak_rss_mb={:.1}",
            pre_bench.empirical_fpr,
            post_bench.empirical_fpr,
            progressive.total_boxes_rejuvenated,
            progressive.total_keys_rejuvenated,
            progressive.pre_expand8_adapted_count,
            (post_filter_ram as f64) / (1024.0 * 1024.0),
            (host.peak_rss_bytes as f64) / (1024.0 * 1024.0)
        ),
    });

    metrics.push(AblationMetric {
        extension_name: format!("DRAM_10M_Throughput_{tag}"),
        metric_label: format!("DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion ({tag})"),
        unit: "ops/sec".to_string(),
        higher_is_better: true,
        key_count: oracle.len(),
        query_count: queries.len(),
        baseline_value: pre_bench.throughput_ops_sec,
        optimized_value: post_bench.throughput_ops_sec,
        delta_pct: compute_delta_pct(pre_bench.throughput_ops_sec, post_bench.throughput_ops_sec),
        p50_ns: post_bench.latency.p50_ns,
        p99_ns: post_bench.latency.p99_ns,
        p999_ns: post_bench.latency.p999_ns,
        empirical_fpr: post_bench.empirical_fpr,
        false_negatives: post_fn,
        repeated_fps: 0,
        peak_ram_bytes: post_filter_ram as u64,
        disk_io_bytes: rm_disk_bytes,
        scale_factor: host.scale_factor,
        notes: format!(
            "uninstrumented 5M tight loop; pre_p50={}ns pre_p99={}ns post_p50={}ns post_p99={}ns",
            pre_bench.latency.p50_ns,
            pre_bench.latency.p99_ns,
            post_bench.latency.p50_ns,
            post_bench.latency.p99_ns
        ),
    });

    metrics.push(AblationMetric {
        extension_name: format!("DRAM_10M_Adversarial_RepeatFP_{tag}"),
        metric_label: format!("Adversarial 10% Repeat-FP Probing FPR ({tag})"),
        unit: "ratio".to_string(),
        higher_is_better: false,
        key_count: oracle.len(),
        query_count: adv.trace_len,
        baseline_value: adv.naive_fpr,
        optimized_value: adv.aeris_fpr,
        delta_pct: compute_delta_pct(adv.naive_fpr, adv.aeris_fpr),
        p50_ns: adv.aeris_latency.p50_ns,
        p99_ns: adv.aeris_latency.p99_ns,
        p999_ns: adv.aeris_latency.p999_ns,
        empirical_fpr: adv.aeris_fpr,
        false_negatives: adv.aeris_false_negatives,
        repeated_fps: adv.aeris_repeated_fps,
        peak_ram_bytes: final_filter_ram,
        disk_io_bytes: rm_disk_after_adv,
        scale_factor: host.scale_factor,
        notes: format!(
            "known_fp_pool={}, naive_repeated_fps={}, aeris_repeated_fps=0, aeris_adaptations={}, repeat_rate={:.0}%",
            adv.pool_size,
            adv.naive_repeated_fps,
            adv.aeris_adaptations,
            DEFAULT_ADVERSARIAL_REPEAT_RATE * 100.0
        ),
    });

    notes.push(format!(
        "DRAM 10M ({tag}): filter_ram={:.2} MiB (>24 MiB L3), pre_fpr={:.6}, post_8x_fpr={:.6} (target={:.6}), post_qps={:.0}, boxes_rejuv={}, adv_fpr {:.5}->{:.5} (0 repeated FPs, 0 FNs).",
        (post_filter_ram as f64) / (1024.0 * 1024.0),
        pre_bench.empirical_fpr,
        post_bench.empirical_fpr,
        epsilon,
        post_bench.throughput_ops_sec,
        progressive.total_boxes_rejuvenated,
        adv.naive_fpr,
        adv.aeris_fpr
    ));
}

/// Inserts all `keys` into a fresh [`AerisFilter`] at `config` purely in RAM.
fn build_fresh_dram_filter(config: &FilterConfig, keys: &[Key]) -> AerisResult<AerisFilter> {
    let mut filter = AerisFilter::new(config.clone())?;
    for &key in keys {
        let _ = filter.insert(key)?;
    }
    Ok(filter)
}

/// Deterministically assigns each key to one of 9 dyadic growth waves (`0..=8`) using
/// a SplitMix64 finalizer so every expansion wave is uniformly distributed across the
/// 48-bit keyspace with `O(1)` extra RAM.
#[inline]
fn dyadic_wave_of_key(key: Key) -> usize {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let h = ((z ^ (z >> 31)) >> 56) as u8;
    if h == 0 {
        0
    } else {
        (8 - h.leading_zeros()) as usize
    }
}

/// Builds a progressive [`AerisFilter`] starting at `q = 16` and executing 8 consecutive
/// `2x` expansions (`q = 16 -> 24`) paired with a real [`RocksDbReverseMap`] populated
/// via 250k-entry `append_batch` chunks.
///
/// Before expansion #8 (`q = 23 -> 24`), adapts up to [`ADVERSARIAL_FP_POOL_TARGET`]
/// genuine false positives against the real Reverse Map and re-probes every adapted range
/// immediately after expansion #8 (including selective rejuvenation when `age == r`) to
/// verify `0` repeated FPs across expansion boundaries.
fn build_progressive_8x_expanded_filter(
    epsilon: f64,
    keys: &[Key],
    empty_queries: &[QueryRange],
    rm_path: &std::path::Path,
) -> AerisResult<ProgressiveBuildSummary> {
    let start_cfg = FilterConfig::with_minimum_bits(DRAM_START_Q, DRAM_RANGE_LEN, epsilon)?;
    let mut filter = AerisFilter::new(start_cfg)?;
    let rm_cfg = ReverseMapConfig::for_wal_coupled().with_write_buffer_size(RM_WRITE_BUFFER_BYTES);
    let reverse_map = RocksDbReverseMap::open(rm_path, rm_cfg)?;
    let mut batch: Vec<(Fingerprint, Key)> = Vec::with_capacity(RM_BATCH_CHUNK_SIZE);

    let mut total_boxes_rejuvenated = 0usize;
    let mut total_keys_rejuvenated = 0usize;
    let mut pre_expand8_adapted: Vec<QueryRange> = Vec::new();
    let mut post_expand8_repeated_fps = 0u64;

    for wave in 0..=DRAM_EXPANSION_COUNT {
        for &key in keys {
            if dyadic_wave_of_key(key) != wave {
                continue;
            }
            let fp = insert_key_with_auto_expand(&mut filter, &reverse_map, &mut batch, key)?;
            batch.push((fp, key));
            if batch.len() >= RM_BATCH_CHUNK_SIZE {
                reverse_map.append_batch(&batch)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            reverse_map.append_batch(&batch)?;
            batch.clear();
        }

        if wave < DRAM_EXPANSION_COUNT {
            // Right before the 8th expansion (q=23->24), adapt real FPs and prove 0 repeats
            // across expansion + Stage 2 selective rejuvenation.
            if wave + 1 == DRAM_EXPANSION_COUNT {
                pre_expand8_adapted =
                    collect_real_fps(&filter, empty_queries, ADVERSARIAL_FP_POOL_TARGET);
                for &range in &pre_expand8_adapted {
                    adapt_with_auto_expand(&mut filter, range, &reverse_map)?;
                }
            }

            let exp_report = filter.expand(&reverse_map)?;
            total_boxes_rejuvenated += exp_report.boxes_rejuvenated;
            total_keys_rejuvenated += exp_report.keys_rejuvenated;
            debug!(
                wave,
                old_q = exp_report.old_q,
                new_q = exp_report.new_q,
                boxes_transferred = exp_report.boxes_transferred,
                boxes_rejuvenated = exp_report.boxes_rejuvenated,
                keys_rejuvenated = exp_report.keys_rejuvenated,
                "completed progressive 2x expansion wave"
            );

            if wave + 1 == DRAM_EXPANSION_COUNT {
                for &range in &pre_expand8_adapted {
                    if filter.query_range(range) {
                        post_expand8_repeated_fps += 1;
                    }
                }
            }
        }
    }

    Ok(ProgressiveBuildSummary {
        filter,
        reverse_map,
        total_boxes_rejuvenated,
        total_keys_rejuvenated,
        pre_expand8_adapted_count: pre_expand8_adapted.len(),
        post_expand8_repeated_fps,
    })
}

/// Inserts `key`, flushing any pending Reverse Map batch before invoking `expand()`
/// if `FilterFull` is ever encountered.
fn insert_key_with_auto_expand(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    pending_batch: &mut Vec<(Fingerprint, Key)>,
    key: Key,
) -> AerisResult<Fingerprint> {
    let mut expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if expansions < MAX_QUOTIENT_BITS => {
                if !pending_batch.is_empty() {
                    reverse_map.append_batch(pending_batch)?;
                    pending_batch.clear();
                }
                let _ = filter.expand(reverse_map)?;
                expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Adapts `filter` to `range`, auto-expanding if `FilterFull` occurs.
fn adapt_with_auto_expand(
    filter: &mut AerisFilter,
    range: QueryRange,
    reverse_map: &dyn ReverseMapStore,
) -> AerisResult<()> {
    let mut expansions = 0u8;
    loop {
        match filter.adapt(range, reverse_map) {
            Ok(_) => return Ok(()),
            Err(AerisError::FilterFull { .. }) if expansions < MAX_QUOTIENT_BITS => {
                let _ = filter.expand(reverse_map)?;
                expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Scans `queries` and collects up to `max_fps` verified-empty ranges that currently
/// return `true` on `filter`.
fn collect_real_fps(
    filter: &AerisFilter,
    queries: &[QueryRange],
    max_fps: usize,
) -> Vec<QueryRange> {
    let mut fps = Vec::with_capacity(max_fps);
    for &q in queries {
        if filter.query_range(q) {
            fps.push(q);
            if fps.len() >= max_fps {
                break;
            }
        }
    }
    fps
}

/// Executes an uninstrumented tight loop over `queries` for true DRAM-bound throughput
/// (`ops/sec`) and exact empirical FPR, followed by a deterministic stride sample for
/// `p50`/`p99`/`p99.9` latency without skewing throughput.
fn benchmark_empty_queries(
    filter: &AerisFilter,
    queries: &[QueryRange],
    sample_target: usize,
) -> QueryBenchResult {
    let start = Instant::now();
    let mut fp_count = 0u64;
    for &q in queries {
        if filter.query_range(q) {
            fp_count += 1;
        }
    }
    let elapsed_secs = start.elapsed().as_secs_f64().max(1e-9);
    let empirical_fpr = (fp_count as f64) / (queries.len().max(1) as f64);
    let throughput_ops_sec = (queries.len() as f64) / elapsed_secs;

    let stride = (queries.len() / sample_target.max(1)).max(1);
    let mut samples_ns = Vec::with_capacity(queries.len() / stride + 1);
    let mut idx = 0usize;
    let mut black_box_hits = 0u64;
    while idx < queries.len() {
        let q = queries[idx];
        let t0 = Instant::now();
        let hit = filter.query_range(q);
        let dt = t0.elapsed().as_nanos() as u64;
        black_box_hits += u64::from(hit);
        samples_ns.push(dt);
        idx += stride;
    }
    std::hint::black_box(black_box_hits);

    QueryBenchResult {
        fp_count,
        empirical_fpr,
        throughput_ops_sec,
        latency: LatencyStats::from_samples(samples_ns),
    }
}

/// Verifies `0` false negatives across ALL `keys` via point queries and across a
/// deterministic strided subset via enclosing closed range queries `[k - r, k + r]`.
fn verify_zero_false_negatives(
    filter: &AerisFilter,
    keys: &[Key],
    enclosing_half_width: u64,
) -> u64 {
    let mut false_negatives = 0u64;
    for &k in keys {
        if !filter.query_point(k) {
            false_negatives += 1;
        }
    }
    let stride = (keys.len() / ENCLOSING_RANGE_VERIFY_SAMPLES).max(1);
    let mut idx = 0usize;
    while idx < keys.len() {
        let range = enclosing_range(keys[idx], enclosing_half_width);
        if !filter.query_range(range) {
            false_negatives += 1;
        }
        idx += stride;
    }
    false_negatives
}

/// Runs an adversarial 10% repeat-FP probing trace comparing unadapted naive querying
/// against live monotonic `AerisFilter::adapt` backed by `RocksDbReverseMap`.
fn run_adversarial_repeat_fp_audit(
    filter: &mut AerisFilter,
    reverse_map: &RocksDbReverseMap,
    oracle: &SortedKeyOracle,
    empty_queries: &[QueryRange],
    seed: u64,
    trace_len: usize,
) -> AerisResult<AdversarialComparison> {
    let known_fp_pool = collect_real_fps(filter, empty_queries, ADVERSARIAL_FP_POOL_TARGET);
    assert!(
        !known_fp_pool.is_empty(),
        "expected non-empty real FP pool across 5M empty queries"
    );

    let trace = generate_adversarial_repeat_fp_trace(
        seed,
        oracle,
        &known_fp_pool,
        DRAM_RANGE_LEN,
        DEFAULT_ADVERSARIAL_REPEAT_RATE,
        trace_len,
    );

    // 1. Naive Unadapted Baseline: read-only replay of the identical trace without `adapt()`.
    let mut naive_seen: HashSet<QueryRange> = HashSet::new();
    let mut naive_fps = 0u64;
    let mut naive_repeated_fps = 0u64;
    for &q in &trace {
        if filter.query_range(q) {
            naive_fps += 1;
            if !naive_seen.insert(q) {
                naive_repeated_fps += 1;
            }
        }
    }
    let naive_fpr = (naive_fps as f64) / (trace.len().max(1) as f64);

    // 2. Monotonic Adaptive Aeris: real `adapt()` against `RocksDbReverseMap` on every FP.
    let mut adapted_set: HashSet<QueryRange> = HashSet::new();
    let mut aeris_fps = 0u64;
    let mut aeris_repeated_fps = 0u64;
    let mut aeris_adaptations = 0u64;
    let sample_stride = (trace.len() / LATENCY_SAMPLE_TARGET).max(1);
    let mut latency_samples = Vec::with_capacity(trace.len() / sample_stride + 1);

    for (idx, &q) in trace.iter().enumerate() {
        let should_sample = idx.is_multiple_of(sample_stride);
        let t0 = if should_sample {
            Some(Instant::now())
        } else {
            None
        };
        let hit = filter.query_range(q);
        if let Some(start) = t0 {
            latency_samples.push(start.elapsed().as_nanos() as u64);
        }

        if hit {
            aeris_fps += 1;
            if !adapted_set.insert(q) {
                aeris_repeated_fps += 1;
            }
            adapt_with_auto_expand(filter, q, reverse_map)?;
            aeris_adaptations += 1;
        }
    }

    // Re-probe every adapted range once more to verify strict monotonicity post-trace.
    for &q in &adapted_set {
        if filter.query_range(q) {
            aeris_repeated_fps += 1;
        }
    }

    let aeris_false_negatives = verify_zero_false_negatives(filter, oracle.as_slice(), 15);
    let aeris_fpr = (aeris_fps as f64) / (trace.len().max(1) as f64);

    Ok(AdversarialComparison {
        pool_size: known_fp_pool.len(),
        trace_len: trace.len(),
        naive_fpr,
        naive_repeated_fps,
        aeris_fpr,
        aeris_repeated_fps,
        aeris_adaptations,
        aeris_latency: LatencyStats::from_samples(latency_samples),
        aeris_false_negatives,
    })
}

// =============================================================================
// Suite 1B: A/B 1 — Dynamic R + FP Log Replay vs. Static R Naive Aeris
// =============================================================================

/// Executes Suite 1B (`1,000,000` keys, `2,000,000` shifting-`R` queries `[16, 256, 32, 1024]`)
/// comparing Static `R = 16` baselines (equal-RAM `C = 17` and paper-minimum `C = 11`)
/// against real Dynamic-`R` Aeris with generational `RocksDbReverseMap` rebuilds and
/// persistent `FalsePositiveLog` replay.
fn run_suite_1b_dynamic_r_ablation(
    host: &mut HostMetadata,
    metrics: &mut Vec<AblationMetric>,
    notes: &mut Vec<String>,
) {
    info!(
        keys = AB1_KEY_COUNT,
        total_queries = AB1_TOTAL_QUERIES,
        phases = ?SHIFTING_R_PHASES,
        payload_budget_c = AB1_PAYLOAD_BUDGET_C,
        epsilon = AB1_EPSILON,
        "starting Suite 1B: A/B 1 Dynamic R + FP Log Replay vs. Static R"
    );

    let oracle = generate_uniform_keys_oracle(SEED_AB1_KEYS, AB1_KEY_COUNT);
    let phases = generate_shifting_r_trace(
        SEED_AB1_TRACE,
        &oracle,
        &SHIFTING_R_PHASES,
        AB1_TOTAL_QUERIES,
    );

    // Baseline A (Equal-RAM Static R=16, m=4, r=13, C=17).
    let equal_ram_cfg = FilterConfig::new(
        AB1_Q,
        AB1_PAYLOAD_BUDGET_C - 4,
        16,
        AB1_PAYLOAD_BUDGET_C,
        AB1_EPSILON,
    )
    .expect("valid equal-RAM static R=16 config");
    let static_equal_ram = run_static_r_baseline(&equal_ram_cfg, &oracle, &phases)
        .expect("equal-RAM static R=16 baseline succeeds");
    assert_eq!(static_equal_ram.false_negatives, 0);

    // Baseline B (Paper-Minimum Static R=16, m=4, r=7, C=11) quantifying classic FPR blowup.
    let paper_min_cfg = FilterConfig::with_minimum_bits(AB1_Q, 16, AB1_EPSILON)
        .expect("valid paper-minimum static R=16 config");
    let static_paper_min = run_static_r_baseline(&paper_min_cfg, &oracle, &phases)
        .expect("paper-minimum static R=16 baseline succeeds");
    assert_eq!(static_paper_min.false_negatives, 0);

    // Optimized Aeris (Equal-RAM C=17, starting at m=4, r=13, with full Control Plane).
    let dynamic_dir = TempDir::new().expect("create temp dir for Suite 1B Dynamic-R Aeris");
    let dynamic_summary = run_dynamic_r_aeris(&equal_ram_cfg, &oracle, &phases, dynamic_dir.path())
        .expect("Dynamic-R Aeris execution succeeds");

    assert_eq!(
        dynamic_summary.false_negatives, 0,
        "Dynamic-R Aeris must maintain 0 false negatives across all 1M keys"
    );
    assert_eq!(
        dynamic_summary.total_repeated_fps, 0,
        "Dynamic-R Aeris with FalsePositiveLog replay must maintain 0 repeated FPs across all rebuilds"
    );
    assert!(
        !dynamic_summary.rebuilds.is_empty(),
        "Dynamic-R Aeris must trigger at least one real background rebuild on shifting R=[16,256,32,1024]"
    );

    host.refresh_peak_rss();
    emit_ab1_metrics(
        &static_equal_ram,
        &static_paper_min,
        &dynamic_summary,
        host.scale_factor,
        metrics,
        notes,
    );
}

/// Runs a static-`R` baseline filter (no dynamic-`R` rebuild, no FP Log replay) across all
/// phases of the shifting-`R` trace.
fn run_static_r_baseline(
    config: &FilterConfig,
    oracle: &SortedKeyOracle,
    phases: &[ShiftingRPhase],
) -> AerisResult<StaticBaselineSummary> {
    let filter = build_fresh_dram_filter(config, oracle.as_slice())?;
    let filter_ram_bytes = filter.memory_usage_bytes() as u64;

    let mut phase_summaries = Vec::with_capacity(phases.len());
    let mut total_fps = 0u64;
    let mut total_queries = 0usize;
    let mut total_elapsed_secs = 0.0_f64;
    let mut all_latency_samples: Vec<u64> = Vec::with_capacity(LATENCY_SAMPLE_TARGET);
    let per_phase_samples = (LATENCY_SAMPLE_TARGET / phases.len().max(1)).max(1);

    for phase in phases {
        let bench = benchmark_empty_queries(&filter, &phase.queries, per_phase_samples);
        let phase_secs = (phase.queries.len() as f64) / bench.throughput_ops_sec.max(1e-9);
        total_elapsed_secs += phase_secs;
        total_fps += bench.fp_count;
        total_queries += phase.queries.len();

        // Collect representative latency samples for overall distribution.
        let stride = (phase.queries.len() / per_phase_samples).max(1);
        let mut idx = 0usize;
        while idx < phase.queries.len() {
            let t0 = Instant::now();
            let hit = filter.query_range(phase.queries[idx]);
            std::hint::black_box(hit);
            all_latency_samples.push(t0.elapsed().as_nanos() as u64);
            idx += stride;
        }

        info!(
            static_m = config.m,
            static_r = config.r,
            static_c = config.payload_budget_c,
            phase_r = phase.range_len,
            fpr = bench.empirical_fpr,
            qps = bench.throughput_ops_sec,
            p50_ns = bench.latency.p50_ns,
            p99_ns = bench.latency.p99_ns,
            "completed static-R baseline phase"
        );

        phase_summaries.push(PhaseRunSummary {
            range_len: phase.range_len,
            query_count: phase.queries.len(),
            fp_count: bench.fp_count,
            empirical_fpr: bench.empirical_fpr,
            throughput_ops_sec: bench.throughput_ops_sec,
            latency: bench.latency,
            active_m: config.m,
            active_r: config.r,
            rebuilds_triggered: 0,
        });
    }

    let false_negatives = verify_zero_false_negatives(&filter, oracle.as_slice(), 8);
    let overall_fpr = (total_fps as f64) / (total_queries.max(1) as f64);
    let overall_qps = (total_queries as f64) / total_elapsed_secs.max(1e-9);

    Ok(StaticBaselineSummary {
        phases: phase_summaries,
        overall_fpr,
        overall_qps,
        overall_latency: LatencyStats::from_samples(all_latency_samples),
        filter_ram_bytes,
        false_negatives,
    })
}

/// Runs Optimized Aeris over the shifting-`R` trace using real `AerisFilter`, generational
/// `RocksDbReverseMap`, `SlidingWindowEstimator`, `DiscreteCostOptimizer`,
/// `AsyncRebuildCoordinator` (`DeltaBuffer` + `"aeris-bg-rebuild"`), and persistent
/// `FalsePositiveLog` replay.
fn run_dynamic_r_aeris(
    initial_cfg: &FilterConfig,
    oracle: &SortedKeyOracle,
    phases: &[ShiftingRPhase],
    root_dir: &std::path::Path,
) -> AerisResult<DynamicAerisSummary> {
    let rm_cfg = ReverseMapConfig::for_wal_coupled().with_write_buffer_size(RM_WRITE_BUFFER_BYTES);
    let mut rm_gen = 0u64;
    let rm0_path = root_dir.join(format!("rm_gen_{rm_gen}"));
    let mut reverse_map = Arc::new(RocksDbReverseMap::open(&rm0_path, rm_cfg.clone())?);

    let mut filter =
        AerisFilter::bulk_load(initial_cfg.clone(), oracle.as_slice(), reverse_map.as_ref())?;
    // Flush initial setup memtables before timing Phase 0 so background setup I/O
    // does not skew Phase 0 RAM query throughput relative to the static baseline.
    reverse_map.flush()?;
    let mut peak_filter_ram = filter.memory_usage_bytes() as u64;

    // Persist every FP in 40-byte CRC32 frames; explicitly fsync before every rebuild boundary.
    let fp_log_path = root_dir.join("ab1_fp.log");
    let fp_log = Arc::new(FalsePositiveLog::open(
        &fp_log_path,
        FpLogConfig {
            max_entries: 65_536,
            sync_on_append: false,
        },
    )?);

    let mut estimator = SlidingWindowEstimator::with_initial_span(
        EstimatorConfig::default(),
        initial_cfg.max_range_r,
    )?;
    let optimizer = DiscreteCostOptimizer::new();
    let cost_model = CostModelParams::default();
    let coordinator = AsyncRebuildCoordinator::new();

    let mut adapted_fps: HashSet<QueryRange> = HashSet::new();
    let mut total_repeated_fps = 0u64;
    let mut rebuilds: Vec<DynamicRebuildRecord> = Vec::new();
    let mut phase_summaries: Vec<PhaseRunSummary> = Vec::with_capacity(phases.len());
    let mut all_latency_samples: Vec<u64> = Vec::with_capacity(LATENCY_SAMPLE_TARGET);
    let per_phase_samples = (LATENCY_SAMPLE_TARGET / phases.len().max(1)).max(1);

    let mut total_fps = 0u64;
    let mut total_queries = 0usize;
    let mut total_query_secs = 0.0_f64;

    for (phase_idx, phase) in phases.iter().enumerate() {
        let mut phase_fps = 0u64;
        let mut phase_rebuilds = 0usize;
        let mut phase_query_ns = 0u128;
        let mut phase_latency_samples: Vec<u64> = Vec::with_capacity(per_phase_samples + 16);
        let sample_stride = (phase.queries.len() / per_phase_samples).max(1);
        let mut batch_fps: Vec<QueryRange> = Vec::with_capacity(64);

        let mut q_idx = 0usize;
        while q_idx < phase.queries.len() {
            let batch_end = (q_idx + AB1_OPTIMIZER_POLL_STRIDE).min(phase.queries.len());
            let batch_len = batch_end - q_idx;
            batch_fps.clear();

            // Timed hot-path query batch integrating real bitmask-sampled
            // `SlidingWindowEstimator::record_query` inline on every query.
            let batch_start = Instant::now();
            for &query in &phase.queries[q_idx..batch_end] {
                let hit = filter.query_range(query);
                estimator.record_query(query, hit);
                if hit {
                    batch_fps.push(query);
                }
            }
            phase_query_ns += batch_start.elapsed().as_nanos();

            // Deterministic stride latency sampling outside the throughput timing block.
            for offset in 0..batch_len {
                if (q_idx + offset).is_multiple_of(sample_stride) {
                    let t0 = Instant::now();
                    let hit = filter.query_range(phase.queries[q_idx + offset]);
                    std::hint::black_box(hit);
                    let sample_ns = t0.elapsed().as_nanos() as u64;
                    phase_latency_samples.push(sample_ns);
                    all_latency_samples.push(sample_ns);
                }
            }

            // Monotonic adaptation + persistent FalsePositiveLog recording for every FP.
            let mut batch_seen: HashSet<QueryRange> = HashSet::new();
            for &fp_range in &batch_fps {
                phase_fps += 1;
                if !batch_seen.insert(fp_range) {
                    continue;
                }
                if !adapted_fps.insert(fp_range) {
                    total_repeated_fps += 1;
                }
                let _ = fp_log.record(fp_range, current_unix_ms())?;
                adapt_with_auto_expand(&mut filter, fp_range, reverse_map.as_ref())?;
            }

            q_idx = batch_end;

            let stats = estimator.snapshot();
            let decision = optimizer.optimize(&stats, filter.config(), &cost_model);
            if decision.should_rebuild {
                let old_m = filter.config().m;
                let target_cfg = filter.config().reconfigure_for_m(decision.recommended_m)?;

                fp_log.sync()?;
                rm_gen += 1;
                let dest_path = root_dir.join(format!("rm_gen_{rm_gen}"));
                let dest_rm = Arc::new(RocksDbReverseMap::open(&dest_path, rm_cfg.clone())?);

                let source_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&reverse_map) as _;
                let dest_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&dest_rm) as _;
                coordinator.start_rebuild(
                    source_dyn,
                    dest_dyn,
                    Arc::clone(&fp_log),
                    target_cfg.clone(),
                    RebuildReason::DynamicRangeChange {
                        old_m,
                        new_m: target_cfg.m,
                    },
                    IoThrottleConfig::default(),
                )?;

                // Serve a concurrent foreground burst on F_old while "aeris-bg-rebuild"
                // executes on its dedicated OS thread, verifying zero-downtime serving and
                // catch-up FalsePositiveLog replay.
                let burst_end = (q_idx + AB1_CONCURRENT_BURST_QUERIES).min(phase.queries.len());
                let mut burst_fps: Vec<QueryRange> = Vec::new();
                let burst_start = Instant::now();
                for &bq in &phase.queries[q_idx..burst_end] {
                    if filter.query_range(bq) {
                        burst_fps.push(bq);
                    }
                }
                phase_query_ns += burst_start.elapsed().as_nanos();

                let burst_fp_set: HashSet<QueryRange> = burst_fps.iter().copied().collect();
                for &bq in &phase.queries[q_idx..burst_end] {
                    estimator.record_query(bq, burst_fp_set.contains(&bq));
                }
                for &fp_range in &burst_fps {
                    phase_fps += 1;
                    if !adapted_fps.insert(fp_range) {
                        total_repeated_fps += 1;
                    }
                    let _ = fp_log.record(fp_range, current_unix_ms())?;
                    adapt_with_auto_expand(&mut filter, fp_range, reverse_map.as_ref())?;
                }
                q_idx = burst_end;

                let completed =
                    coordinator.finish_and_drain_deltas(|r| Ok(oracle.is_range_empty(r)))?;
                let old_rm_path = reverse_map.path().to_path_buf();
                let old_rm = std::mem::replace(&mut reverse_map, dest_rm);
                drop(completed.reverse_map);
                drop(old_rm);
                if old_rm_path != reverse_map.path() {
                    let _ = fs::remove_dir_all(&old_rm_path);
                }

                filter = completed.filter;
                peak_filter_ram = peak_filter_ram.max(filter.memory_usage_bytes() as u64);
                estimator.record_bulk_load_speed(completed.report.observed_bulk_load_speed);

                // Immediately re-probe EVERY previously adapted FP range on F_new to prove
                // 100% monotonicity across the Dynamic-R rebuild.
                let mut reprobe_repeated = 0u64;
                for &adapted_range in &adapted_fps {
                    if filter.query_range(adapted_range) {
                        reprobe_repeated += 1;
                    }
                }
                total_repeated_fps += reprobe_repeated;

                let fp_replay_entries = completed.report.fp_replay_initial.replayed_entries
                    + completed.report.fp_replay_catchup.replayed_entries;
                let fp_collisions_resolved = completed.report.fp_replay_initial.collisions_resolved
                    + completed.report.fp_replay_catchup.collisions_resolved;
                let fp_replay_latency_ns = completed.report.fp_replay_initial.latency_ns
                    + completed.report.fp_replay_catchup.latency_ns;

                info!(
                    phase_idx,
                    phase_r = phase.range_len,
                    old_m,
                    new_m = filter.config().m,
                    new_r = filter.config().r,
                    keys_loaded = completed.report.keys_loaded,
                    rebuild_ms = (completed.report.duration_ns as f64) / 1e6,
                    fp_replay_entries,
                    fp_collisions_resolved,
                    fp_replay_us = (fp_replay_latency_ns as f64) / 1e3,
                    reprobe_checked = adapted_fps.len(),
                    reprobe_repeated,
                    "swapped Dynamic-R AerisFilter and verified post-rebuild monotonicity"
                );

                rebuilds.push(DynamicRebuildRecord {
                    phase_index: phase_idx,
                    phase_range_len: phase.range_len,
                    old_m,
                    new_m: filter.config().m,
                    new_r: filter.config().r,
                    keys_loaded: completed.report.keys_loaded,
                    duration_ns: completed.report.duration_ns,
                    fp_replay_entries,
                    fp_collisions_resolved,
                    fp_replay_latency_ns,
                    observed_bulk_load_speed: completed.report.observed_bulk_load_speed,
                    reprobe_checked: adapted_fps.len(),
                    reprobe_repeated_fps: reprobe_repeated,
                });
                phase_rebuilds += 1;
            }
        }

        let phase_secs = ((phase_query_ns as f64) / 1e9).max(1e-9);
        total_query_secs += phase_secs;
        total_fps += phase_fps;
        total_queries += phase.queries.len();

        let phase_fpr = (phase_fps as f64) / (phase.queries.len().max(1) as f64);
        let phase_qps = (phase.queries.len() as f64) / phase_secs;
        let phase_lat = LatencyStats::from_samples(phase_latency_samples);

        info!(
            phase_idx,
            phase_r = phase.range_len,
            active_m = filter.config().m,
            active_r = filter.config().r,
            phase_rebuilds,
            phase_fpr,
            phase_qps,
            p50_ns = phase_lat.p50_ns,
            p99_ns = phase_lat.p99_ns,
            "completed Dynamic-R Aeris phase"
        );

        phase_summaries.push(PhaseRunSummary {
            range_len: phase.range_len,
            query_count: phase.queries.len(),
            fp_count: phase_fps,
            empirical_fpr: phase_fpr,
            throughput_ops_sec: phase_qps,
            latency: phase_lat,
            active_m: filter.config().m,
            active_r: filter.config().r,
            rebuilds_triggered: phase_rebuilds,
        });
    }

    fp_log.sync()?;
    reverse_map.flush()?;
    let disk_io_bytes = directory_size_bytes(root_dir);
    let false_negatives = verify_zero_false_negatives(&filter, oracle.as_slice(), 8);

    // Final end-to-end monotonicity check over all adapted FPs across all phases.
    for &adapted_range in &adapted_fps {
        if filter.query_range(adapted_range) {
            total_repeated_fps += 1;
        }
    }

    let overall_fpr = (total_fps as f64) / (total_queries.max(1) as f64);
    let overall_qps = (total_queries as f64) / total_query_secs.max(1e-9);

    Ok(DynamicAerisSummary {
        phases: phase_summaries,
        rebuilds,
        overall_fpr,
        overall_qps,
        overall_latency: LatencyStats::from_samples(all_latency_samples),
        peak_filter_ram_bytes: peak_filter_ram,
        disk_io_bytes,
        total_logged_fps: fp_log.len(),
        total_repeated_fps,
        false_negatives,
    })
}

/// Emits all structured [`AblationMetric`] records for Suite 1B (overall, per-phase,
/// rebuild duration, and FP Log replay latency).
fn emit_ab1_metrics(
    static_eq: &StaticBaselineSummary,
    static_min: &StaticBaselineSummary,
    dynamic: &DynamicAerisSummary,
    scale_factor: f64,
    metrics: &mut Vec<AblationMetric>,
    notes: &mut Vec<String>,
) {
    let rebuild_path: String = dynamic
        .rebuilds
        .iter()
        .map(|r| format!("m:{}->{}@R={}", r.old_m, r.new_m, r.phase_range_len))
        .collect::<Vec<_>>()
        .join(", ");

    metrics.push(AblationMetric {
        extension_name: "AB1_DynamicR_Overall_Throughput".to_string(),
        metric_label: "A/B 1 Overall Shifting-R Throughput (Equal-RAM Static R=16 vs Dynamic R)"
            .to_string(),
        unit: "ops/sec".to_string(),
        higher_is_better: true,
        key_count: AB1_KEY_COUNT,
        query_count: AB1_TOTAL_QUERIES,
        baseline_value: static_eq.overall_qps,
        optimized_value: dynamic.overall_qps,
        delta_pct: compute_delta_pct(static_eq.overall_qps, dynamic.overall_qps),
        p50_ns: dynamic.overall_latency.p50_ns,
        p99_ns: dynamic.overall_latency.p99_ns,
        p999_ns: dynamic.overall_latency.p999_ns,
        empirical_fpr: dynamic.overall_fpr,
        false_negatives: dynamic.false_negatives,
        repeated_fps: dynamic.total_repeated_fps,
        peak_ram_bytes: dynamic.peak_filter_ram_bytes.max(static_eq.filter_ram_bytes),
        disk_io_bytes: dynamic.disk_io_bytes,
        scale_factor,
        notes: format!(
            "C={AB1_PAYLOAD_BUDGET_C}, eps={AB1_EPSILON}, rebuilds={} [{rebuild_path}], static_c17_fpr={:.6}, dynamic_c17_fpr={:.6}, paper_min_c11_fpr={:.6}",
            dynamic.rebuilds.len(),
            static_eq.overall_fpr,
            dynamic.overall_fpr,
            static_min.overall_fpr
        ),
    });

    metrics.push(AblationMetric {
        extension_name: "AB1_DynamicR_Overall_Latency_p99".to_string(),
        metric_label: "A/B 1 Overall Shifting-R p99 Latency (Equal-RAM Static R=16 vs Dynamic R)"
            .to_string(),
        unit: "ns".to_string(),
        higher_is_better: false,
        key_count: AB1_KEY_COUNT,
        query_count: AB1_TOTAL_QUERIES,
        baseline_value: static_eq.overall_latency.p99_ns as f64,
        optimized_value: dynamic.overall_latency.p99_ns as f64,
        delta_pct: compute_delta_pct(
            static_eq.overall_latency.p99_ns as f64,
            dynamic.overall_latency.p99_ns as f64,
        ),
        p50_ns: dynamic.overall_latency.p50_ns,
        p99_ns: dynamic.overall_latency.p99_ns,
        p999_ns: dynamic.overall_latency.p999_ns,
        empirical_fpr: dynamic.overall_fpr,
        false_negatives: dynamic.false_negatives,
        repeated_fps: dynamic.total_repeated_fps,
        peak_ram_bytes: dynamic.peak_filter_ram_bytes,
        disk_io_bytes: dynamic.disk_io_bytes,
        scale_factor,
        notes: format!(
            "static_p50={}ns static_p99={}ns static_p999={}ns vs dynamic_p50={}ns dynamic_p99={}ns dynamic_p999={}ns",
            static_eq.overall_latency.p50_ns,
            static_eq.overall_latency.p99_ns,
            static_eq.overall_latency.p999_ns,
            dynamic.overall_latency.p50_ns,
            dynamic.overall_latency.p99_ns,
            dynamic.overall_latency.p999_ns
        ),
    });

    metrics.push(AblationMetric {
        extension_name: "AB1_DynamicR_Overall_FPR_vs_PaperMin".to_string(),
        metric_label:
            "A/B 1 Overall Shifting-R FPR (Paper-Min Static R=16 C=11 vs Dynamic-R Aeris C=17)"
                .to_string(),
        unit: "ratio".to_string(),
        higher_is_better: false,
        key_count: AB1_KEY_COUNT,
        query_count: AB1_TOTAL_QUERIES,
        baseline_value: static_min.overall_fpr,
        optimized_value: dynamic.overall_fpr,
        delta_pct: compute_delta_pct(static_min.overall_fpr, dynamic.overall_fpr),
        p50_ns: dynamic.overall_latency.p50_ns,
        p99_ns: dynamic.overall_latency.p99_ns,
        p999_ns: dynamic.overall_latency.p999_ns,
        empirical_fpr: dynamic.overall_fpr,
        false_negatives: dynamic.false_negatives,
        repeated_fps: dynamic.total_repeated_fps,
        peak_ram_bytes: dynamic.peak_filter_ram_bytes,
        disk_io_bytes: dynamic.disk_io_bytes,
        scale_factor,
        notes: format!(
            "paper_min(r=7,m=4,C=11) overall_fpr={:.6} (R=1024 blowup={:.6} vs target eps={AB1_EPSILON}), equal_ram(r=13,m=4,C=17) overall_fpr={:.6}, dynamic_aeris overall_fpr={:.6}",
            static_min.overall_fpr,
            static_min.phases.last().map(|p| p.empirical_fpr).unwrap_or(0.0),
            static_eq.overall_fpr,
            dynamic.overall_fpr
        ),
    });

    for (idx, ((st_eq, st_min), dyn_p)) in static_eq
        .phases
        .iter()
        .zip(static_min.phases.iter())
        .zip(dynamic.phases.iter())
        .enumerate()
    {
        metrics.push(AblationMetric {
            extension_name: format!("AB1_DynamicR_Phase{idx}_R{}_Throughput", dyn_p.range_len),
            metric_label: format!(
                "A/B 1 Phase {idx} (R={}) Throughput (Equal-RAM Static vs Dynamic R)",
                dyn_p.range_len
            ),
            unit: "ops/sec".to_string(),
            higher_is_better: true,
            key_count: AB1_KEY_COUNT,
            query_count: dyn_p.query_count,
            baseline_value: st_eq.throughput_ops_sec,
            optimized_value: dyn_p.throughput_ops_sec,
            delta_pct: compute_delta_pct(st_eq.throughput_ops_sec, dyn_p.throughput_ops_sec),
            p50_ns: dyn_p.latency.p50_ns,
            p99_ns: dyn_p.latency.p99_ns,
            p999_ns: dyn_p.latency.p999_ns,
            empirical_fpr: dyn_p.empirical_fpr,
            false_negatives: dynamic.false_negatives,
            repeated_fps: 0,
            peak_ram_bytes: dynamic.peak_filter_ram_bytes,
            disk_io_bytes: dynamic.disk_io_bytes,
            scale_factor,
            notes: format!(
                "active_m={}, active_r={}, rebuilds_in_phase={}, static_p99={}ns, dyn_p99={}ns, static_c17_fpr={:.6}, paper_min_c11_fpr={:.6}, dyn_fpr={:.6}",
                dyn_p.active_m,
                dyn_p.active_r,
                dyn_p.rebuilds_triggered,
                st_eq.latency.p99_ns,
                dyn_p.latency.p99_ns,
                st_eq.empirical_fpr,
                st_min.empirical_fpr,
                dyn_p.empirical_fpr
            ),
        });

        metrics.push(AblationMetric {
            extension_name: format!("AB1_DynamicR_Phase{idx}_R{}_FPR", dyn_p.range_len),
            metric_label: format!(
                "A/B 1 Phase {idx} (R={}) FPR (Paper-Min Static R=16 vs Dynamic R)",
                dyn_p.range_len
            ),
            unit: "ratio".to_string(),
            higher_is_better: false,
            key_count: AB1_KEY_COUNT,
            query_count: dyn_p.query_count,
            baseline_value: st_min.empirical_fpr,
            optimized_value: dyn_p.empirical_fpr,
            delta_pct: compute_delta_pct(st_min.empirical_fpr, dyn_p.empirical_fpr),
            p50_ns: dyn_p.latency.p50_ns,
            p99_ns: dyn_p.latency.p99_ns,
            p999_ns: dyn_p.latency.p999_ns,
            empirical_fpr: dyn_p.empirical_fpr,
            false_negatives: dynamic.false_negatives,
            repeated_fps: 0,
            peak_ram_bytes: dynamic.peak_filter_ram_bytes,
            disk_io_bytes: dynamic.disk_io_bytes,
            scale_factor,
            notes: format!(
                "equal_ram_static_fpr={:.6} (fps={}), paper_min_static_fpr={:.6} (fps={}), dynamic_fpr={:.6} (fps={}), target_eps={AB1_EPSILON}, active_m={}, active_r={}",
                st_eq.empirical_fpr,
                st_eq.fp_count,
                st_min.empirical_fpr,
                st_min.fp_count,
                dyn_p.empirical_fpr,
                dyn_p.fp_count,
                dyn_p.active_m,
                dyn_p.active_r
            ),
        });
    }

    let rebuild_durations: Vec<u64> = dynamic.rebuilds.iter().map(|r| r.duration_ns).collect();
    let rebuild_lat = LatencyStats::from_samples(rebuild_durations);
    let total_rebuild_ns: u64 = dynamic.rebuilds.iter().map(|r| r.duration_ns).sum();
    let rebuild_details: String = dynamic
        .rebuilds
        .iter()
        .map(|r| {
            format!(
                "phase{}(R={}, m:{}->{}, r={}, keys={}, {:.1}ms, bulk={:.0}k/s, reprobe_rep={})",
                r.phase_index,
                r.phase_range_len,
                r.old_m,
                r.new_m,
                r.new_r,
                r.keys_loaded,
                (r.duration_ns as f64) / 1e6,
                r.observed_bulk_load_speed / 1e3,
                r.reprobe_repeated_fps
            )
        })
        .collect::<Vec<_>>()
        .join("; ");

    metrics.push(AblationMetric {
        extension_name: "AB1_DynamicR_RebuildTime".to_string(),
        metric_label: "A/B 1 Background RocksDB Reverse Map Rebuild Time (Mean)".to_string(),
        unit: "ns".to_string(),
        higher_is_better: false,
        key_count: AB1_KEY_COUNT,
        query_count: dynamic.rebuilds.len(),
        baseline_value: 0.0,
        optimized_value: rebuild_lat.mean_ns,
        delta_pct: compute_delta_pct(0.0, rebuild_lat.mean_ns),
        p50_ns: rebuild_lat.p50_ns,
        p99_ns: rebuild_lat.p99_ns,
        p999_ns: rebuild_lat.p999_ns,
        empirical_fpr: dynamic.overall_fpr,
        false_negatives: dynamic.false_negatives,
        repeated_fps: dynamic.total_repeated_fps,
        peak_ram_bytes: dynamic.peak_filter_ram_bytes,
        disk_io_bytes: dynamic.disk_io_bytes,
        scale_factor,
        notes: format!(
            "total_rebuilds={}, total_rebuild_ms={:.1}; {rebuild_details}",
            dynamic.rebuilds.len(),
            (total_rebuild_ns as f64) / 1e6
        ),
    });

    let replay_latencies: Vec<u64> = dynamic
        .rebuilds
        .iter()
        .map(|r| r.fp_replay_latency_ns)
        .collect();
    let replay_lat = LatencyStats::from_samples(replay_latencies);
    let total_replay_ns: u64 = dynamic
        .rebuilds
        .iter()
        .map(|r| r.fp_replay_latency_ns)
        .sum();
    let total_replayed_entries: usize = dynamic.rebuilds.iter().map(|r| r.fp_replay_entries).sum();
    let total_collisions_resolved: usize = dynamic
        .rebuilds
        .iter()
        .map(|r| r.fp_collisions_resolved)
        .sum();
    let total_reprobes: usize = dynamic.rebuilds.iter().map(|r| r.reprobe_checked).sum();

    metrics.push(AblationMetric {
        extension_name: "AB1_DynamicR_FpReplayLatency".to_string(),
        metric_label: "A/B 1 Persistent FP Log Replay Latency Across Rebuilds (Total)".to_string(),
        unit: "ns".to_string(),
        higher_is_better: false,
        key_count: AB1_KEY_COUNT,
        query_count: total_replayed_entries,
        baseline_value: 0.0,
        optimized_value: total_replay_ns as f64,
        delta_pct: compute_delta_pct(0.0, total_replay_ns as f64),
        p50_ns: replay_lat.p50_ns,
        p99_ns: replay_lat.p99_ns,
        p999_ns: replay_lat.p999_ns,
        empirical_fpr: dynamic.overall_fpr,
        false_negatives: dynamic.false_negatives,
        repeated_fps: dynamic.total_repeated_fps,
        peak_ram_bytes: dynamic.peak_filter_ram_bytes,
        disk_io_bytes: dynamic.disk_io_bytes,
        scale_factor,
        notes: format!(
            "logged_fps={}, replayed_entries={}, collisions_resolved={}, post_rebuild_reprobes={}, repeated_fps={} (100% monotonic)",
            dynamic.total_logged_fps,
            total_replayed_entries,
            total_collisions_resolved,
            total_reprobes,
            dynamic.total_repeated_fps
        ),
    });

    notes.push(format!(
        "A/B 1 Dynamic-R: overall_qps {:.0}->{:.0} ({:+.1}%), p99 {}ns->{}ns, paper_min_fpr {:.5}->{:.5}, rebuilds={} ({rebuild_path}), total_fp_replay={:.2}ms, repeated_fps=0, fn=0.",
        static_eq.overall_qps,
        dynamic.overall_qps,
        compute_delta_pct(static_eq.overall_qps, dynamic.overall_qps),
        static_eq.overall_latency.p99_ns,
        dynamic.overall_latency.p99_ns,
        static_min.overall_fpr,
        dynamic.overall_fpr,
        dynamic.rebuilds.len(),
        (total_replay_ns as f64) / 1e6
    ));
}

/// Returns current Unix timestamp in milliseconds for [`FalsePositiveLog::record`].
#[inline]
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
