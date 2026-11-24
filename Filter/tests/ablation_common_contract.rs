//! Contract verification unit tests for `tests/ablation_common/mod.rs`.
//!
//! Validates determinism, ground-truth emptiness invariants, latency percentiles,
//! host metadata detection, and JSON round-trip serialization of [`AblationMetric`]
//! before parallel ablation suites execute.

mod ablation_common;

use ablation_common::{
    compute_delta_pct, generate_adversarial_repeat_fp_trace,
    generate_correlated_zipfian_empty_queries, generate_seasonal_ingestion_trace,
    generate_shifting_r_trace, generate_skewed_hotspot_write_trace,
    generate_uncorrelated_zipfian_empty_queries, generate_uniform_empty_queries,
    generate_uniform_keys_oracle, generate_unique_uniform_empty_queries, init_ablation_tracing,
    AblationMetric, HostMetadata, LatencyStats, DEFAULT_ADVERSARIAL_REPEAT_RATE,
    DEFAULT_CORRELATION_D, DEFAULT_ZIPF_ALPHA, SHIFTING_R_PHASES,
};
use std::collections::HashSet;

#[test]
fn ablation_common_generators_and_json_contract_hold() {
    init_ablation_tracing();

    // 1. Uniform keys oracle + uniqueness + determinism
    let oracle_a = generate_uniform_keys_oracle(0xAB1A_0001, 2_000);
    let oracle_b = generate_uniform_keys_oracle(0xAB1A_0001, 2_000);
    assert_eq!(oracle_a.len(), 2_000);
    assert_eq!(oracle_a.as_slice(), oracle_b.as_slice());

    // 2. Uniform & unique empty queries
    let uniform_q = generate_uniform_empty_queries(0xAB1A_0002, &oracle_a, 32, 500);
    assert_eq!(uniform_q.len(), 500);
    assert!(uniform_q.iter().all(|&q| oracle_a.is_range_empty(q)));

    let unique_q = generate_unique_uniform_empty_queries(0xAB1A_0003, &oracle_a, 64, 500);
    assert_eq!(unique_q.len(), 500);
    let unique_starts: HashSet<u64> = unique_q.iter().map(|q| q.start()).collect();
    assert_eq!(unique_starts.len(), 500);
    assert!(unique_q.iter().all(|&q| oracle_a.is_range_empty(q)));

    // 3. Correlated & uncorrelated Zipfian empty queries
    let corr_q = generate_correlated_zipfian_empty_queries(
        0xAB1A_0004,
        &oracle_a,
        32,
        DEFAULT_ZIPF_ALPHA,
        DEFAULT_CORRELATION_D,
        500,
    );
    assert_eq!(corr_q.len(), 500);
    assert!(corr_q.iter().all(|&q| oracle_a.is_range_empty(q)));

    let uncorr_q = generate_uncorrelated_zipfian_empty_queries(
        0xAB1A_0005,
        &oracle_a,
        32,
        DEFAULT_ZIPF_ALPHA,
        256,
        500,
    );
    assert_eq!(uncorr_q.len(), 500);
    assert!(uncorr_q.iter().all(|&q| oracle_a.is_range_empty(q)));

    // 4. Adversarial repeat-FP trace (~10% repeat rate)
    let fp_pool = &unique_q[..10];
    let adv_trace = generate_adversarial_repeat_fp_trace(
        0xAB1A_0006,
        &oracle_a,
        fp_pool,
        64,
        DEFAULT_ADVERSARIAL_REPEAT_RATE,
        1_000,
    );
    assert_eq!(adv_trace.len(), 1_000);
    assert!(adv_trace.iter().all(|&q| oracle_a.is_range_empty(q)));

    // 5. Shifting-R trace (16 -> 256 -> 32 -> 1024)
    let phases = generate_shifting_r_trace(0xAB1A_0007, &oracle_a, &SHIFTING_R_PHASES, 800);
    assert_eq!(phases.len(), 4);
    for (phase, &expected_r) in phases.iter().zip(SHIFTING_R_PHASES.iter()) {
        assert_eq!(phase.range_len, expected_r);
        assert_eq!(phase.queries.len(), 200);
        assert!(phase.queries.iter().all(|&q| oracle_a.is_range_empty(q)));
    }

    // 6. Seasonal ingestion trace
    let seasonal = generate_seasonal_ingestion_trace(0xAB1A_0008, 72, 24, 15_000.0, 6_000.0);
    assert_eq!(seasonal.len(), 72);
    assert!(seasonal
        .iter()
        .all(|p| p.ingest_rate_keys_per_sec > 0.0 && p.p95_query_len >= 8.0));

    // 7. Skewed hot-spot write trace
    let skewed = generate_skewed_hotspot_write_trace(0xAB1A_0009, 2_000, 0.85, 1_000_000, 100_000);
    assert_eq!(skewed.len(), 2_000);
    let unique_skewed: HashSet<u64> = skewed.iter().copied().collect();
    assert_eq!(unique_skewed.len(), 2_000);

    // 8. LatencyStats (p50 / p99 / p999) + AblationMetric JSON serialization + HostMetadata
    let stats = LatencyStats::from_samples((1..=1_000u64).collect());
    assert_eq!(stats.p50_ns, 500);
    assert_eq!(stats.p99_ns, 990);
    assert_eq!(stats.p999_ns, 999);

    let host = HostMetadata::detect();
    assert!(host.logical_cores >= 1);
    assert!(host.scale_factor == 1.0 || host.scale_factor == 0.5);

    let metric = AblationMetric {
        extension_name: "contract_check".to_string(),
        metric_label: "Throughput".to_string(),
        unit: "ops/sec".to_string(),
        higher_is_better: true,
        key_count: 2_000,
        query_count: 1_000,
        baseline_value: 100.0,
        optimized_value: 150.0,
        delta_pct: compute_delta_pct(100.0, 150.0),
        p50_ns: stats.p50_ns,
        p99_ns: stats.p99_ns,
        p999_ns: stats.p999_ns,
        empirical_fpr: 0.001,
        false_negatives: 0,
        repeated_fps: 0,
        peak_ram_bytes: host.peak_rss_bytes,
        disk_io_bytes: 0,
        scale_factor: host.scale_factor,
        notes: "contract smoke".to_string(),
    };
    let json = serde_json::to_string(&metric).expect("serialize AblationMetric");
    let decoded: AblationMetric = serde_json::from_str(&json).expect("deserialize AblationMetric");
    assert_eq!(metric, decoded);
}
