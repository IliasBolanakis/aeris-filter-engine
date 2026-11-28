//! Core unit tests for [`super::SlidingWindowEstimator`], config validation, and quantile clipping.

use super::quantile::{compute_clipped_quantiles, replace_sorted_sample};
use super::*;
use aeris_types::WorkloadEstimator;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn config_validation_rejects_invalid_parameters() {
    assert!(EstimatorConfig::default().validate().is_ok());

    let cases = [
        EstimatorConfig {
            window_size: 0,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            outlier_median_multiplier: 0,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            outlier_floor_span: 0,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            max_clip_span: 0,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            hysteresis_ratio: 0.99,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            hysteresis_ratio: f64::NAN,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            required_sustained_evals: 0,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            ewma_alpha: 0.0,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            ewma_alpha: 1.01,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            ewma_alpha: f64::INFINITY,
            ..EstimatorConfig::default()
        },
    ];
    for bad in &cases {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn empty_window_returns_safe_defaults() {
    let est = SlidingWindowEstimator::new(EstimatorConfig::default()).unwrap();
    let snap = est.snapshot();
    assert_eq!(snap.window_query_count, 0);
    assert_eq!(snap.observed_fpr, 0.0);
    assert_eq!(snap.p50_query_len, 1);
    assert_eq!(snap.p95_query_len, 1);
    assert_eq!(snap.p99_query_len, 1);
    assert_eq!(snap.effective_query_span, 1);
    assert_eq!(snap.clipped_outliers, 0);
}

#[test]
fn outlier_clipping_resists_adversarial_poison_spikes() {
    let cfg = EstimatorConfig::default();
    let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xA3E1_0001);

    // 85% baseline R=32 queries mixed with 15% adversarial 10^7-span poison queries.
    for i in 0..4096u64 {
        let span = if i % 20 < 3 {
            10_000_000
        } else {
            rng.gen_range(24..=40)
        };
        let start = rng.gen_range(0..1_000_000_000);
        let range = QueryRange::with_len(start, span).unwrap();
        est.record_query(range, false);
    }

    let snap = est.snapshot();
    assert_eq!(snap.window_query_count, 4096);
    assert!(
        snap.clipped_outliers >= 600,
        "expected ~15% poison spikes clipped"
    );
    assert!(
        (24..=40).contains(&snap.p50_query_len),
        "p50 shifted unexpectedly: {}",
        snap.p50_query_len
    );
    assert!(
        (32..=40).contains(&snap.p95_query_len),
        "p95 shifted under poison attack: {}",
        snap.p95_query_len
    );
    assert_eq!(
        snap.effective_query_span, 32,
        "effective_query_span must not shift under 15% poison spikes"
    );
    assert_eq!(est.span_transitions(), 0);
}

#[test]
fn legitimate_regime_shift_updates_span_after_cooldown_and_hysteresis() {
    let cfg = EstimatorConfig {
        window_size: 256,
        min_samples_for_clipping: 32,
        cooldown_queries: 64,
        required_sustained_evals: 4,
        hysteresis_ratio: 2.0,
        ..EstimatorConfig::default()
    };
    let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();

    // Warm up with 256 queries of span 32.
    for i in 0..256u64 {
        est.record_query(QueryRange::with_len(i * 100, 32).unwrap(), false);
    }
    assert_eq!(est.effective_query_span(), 32);
    assert_eq!(est.span_transitions(), 0);

    // Shift workload legitimately to span 256 (> 50% of window -> median shifts).
    for i in 0..300u64 {
        est.record_query(QueryRange::with_len(i * 1000, 256).unwrap(), false);
    }
    assert_eq!(est.effective_query_span(), 256);
    assert_eq!(est.span_transitions(), 1);
}

#[test]
fn cooldown_and_sustained_evals_prevent_premature_transition() {
    let cfg = EstimatorConfig {
        window_size: 64,
        min_samples_for_clipping: 16,
        cooldown_queries: 100,
        required_sustained_evals: 4,
        hysteresis_ratio: 2.0,
        ..EstimatorConfig::default()
    };
    let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();

    // Send 99 queries of span 512: still inside cooldown (< 100).
    for i in 0..99u64 {
        est.record_query(QueryRange::with_len(i * 1000, 512).unwrap(), false);
    }
    assert_eq!(est.effective_query_span(), 32);
    assert!(!est.evaluate_span_transition());

    // 100th query crosses cooldown; still needs 4 consecutive evaluations.
    est.record_query(QueryRange::with_len(100_000, 512).unwrap(), false);
    assert_eq!(est.effective_query_span(), 32);

    let mut transitions = 0;
    for _ in 0..4 {
        if est.evaluate_span_transition() {
            transitions += 1;
        }
    }
    assert_eq!(transitions, 1);
    assert_eq!(est.effective_query_span(), 512);
}

#[test]
fn ewma_and_sliding_fpr_tracking_handle_valid_and_invalid_inputs() {
    let cfg = EstimatorConfig {
        window_size: 100,
        ewma_alpha: 0.25,
        ..EstimatorConfig::default()
    };
    let mut est = SlidingWindowEstimator::new(cfg).unwrap();

    // First sample initializes directly; subsequent samples smooth with alpha = 0.25.
    est.record_ingest(1000, 1.0);
    assert!((est.ingestion_rate_ewma() - 1000.0).abs() < 1e-9);
    est.record_ingest(2000, 1.0);
    assert!((est.ingestion_rate_ewma() - 1250.0).abs() < 1e-9);

    // Invalid elapsed_secs / speeds / mu must be ignored without corrupting state.
    est.record_ingest(9999, 0.0);
    est.record_ingest(9999, -1.0);
    est.record_ingest(9999, f64::NAN);
    assert!((est.ingestion_rate_ewma() - 1250.0).abs() < 1e-9);

    est.record_bulk_load_speed(400_000.0);
    est.record_bulk_load_speed(800_000.0);
    assert!((est.snapshot().bulk_load_speed - 500_000.0).abs() < 1e-9);
    est.record_bulk_load_speed(-10.0);
    assert!((est.snapshot().bulk_load_speed - 500_000.0).abs() < 1e-9);

    est.record_query_rate(100, 0.5);
    est.record_query_rate(100, 0.0);
    est.record_query_rate(100, f64::NAN);
    assert!((est.snapshot().query_rate - 200.0).abs() < 1e-9);

    est.set_avg_keys_per_partition(3.5);
    est.set_avg_keys_per_partition(-1.0);
    est.set_avg_keys_per_partition(f64::NAN);
    assert!((est.snapshot().avg_keys_per_partition - 3.5).abs() < 1e-9);

    // Sliding window FPR: 10 FPs out of 100 queries -> 0.10, then evicted by 100 non-FPs -> 0.0.
    for i in 0..100u64 {
        est.record_query(QueryRange::point(i), i < 10);
    }
    assert!((est.snapshot().observed_fpr - 0.10).abs() < 1e-9);

    for i in 100..200u64 {
        est.record_query(QueryRange::point(i), false);
    }
    assert!((est.snapshot().observed_fpr - 0.0).abs() < 1e-9);
}

#[test]
fn legitimate_full_table_scan_regime_caps_at_max_clip_span() {
    let cfg = EstimatorConfig {
        window_size: 128,
        min_samples_for_clipping: 32,
        cooldown_queries: 16,
        required_sustained_evals: 2,
        ..EstimatorConfig::default()
    };
    let max_cap = cfg.max_clip_span;
    let mut est = SlidingWindowEstimator::with_initial_span(cfg, 32).unwrap();

    for i in 0..160u64 {
        est.record_query(QueryRange::with_len(i, 10_000_000).unwrap(), false);
    }
    let snap = est.snapshot();
    assert_eq!(snap.p95_query_len, max_cap);
    assert_eq!(snap.effective_query_span, max_cap);
    assert_eq!(snap.clipped_outliers, 128);
}

#[test]
fn sample_stride_validation_requires_power_of_two() {
    for bad in [0u64, 3, 5, 6, 48, 65, 100] {
        let cfg = EstimatorConfig {
            sample_stride: bad,
            ..EstimatorConfig::default()
        };
        assert!(
            cfg.validate().is_err(),
            "expected sample_stride={bad} to fail validation"
        );
    }
    for good in [1u64, 2, 4, 16, 64, 128] {
        let cfg = EstimatorConfig {
            sample_stride: good,
            ..EstimatorConfig::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "expected sample_stride={good} to pass validation"
        );
    }
}

#[test]
fn sampled_quantile_accuracy_matches_unsampled_and_detects_shifts() {
    let unsampled_cfg = EstimatorConfig {
        sample_stride: 1,
        ..EstimatorConfig::default()
    };
    let sampled_cfg = EstimatorConfig::default(); // sample_stride = 64
    assert_eq!(sampled_cfg.sample_stride, DEFAULT_SAMPLE_STRIDE);

    let mut exact = SlidingWindowEstimator::with_initial_span(unsampled_cfg, 16).unwrap();
    let mut sampled = SlidingWindowEstimator::with_initial_span(sampled_cfg, 16).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0x5A17_0064);

    // Phase 1: 8,192 queries centered around R=32 with 10% adversarial 10^7 poison spikes.
    for i in 0..8_192u64 {
        let span = if i % 10 == 0 {
            10_000_000
        } else {
            rng.gen_range(24..=40)
        };
        let q = QueryRange::with_len(i * 100, span).unwrap();
        exact.record_query(q, false);
        sampled.record_query(q, false);
    }

    let s_exact = exact.snapshot();
    let s_sampled = sampled.snapshot();
    assert_eq!(s_sampled.window_query_count, 4096);
    assert!(s_sampled.p50_query_len.abs_diff(s_exact.p50_query_len) <= 4);
    assert!(s_sampled.p95_query_len.abs_diff(s_exact.p95_query_len) <= 4);
    assert!((24..=40).contains(&s_sampled.effective_query_span));

    // Phase 2: genuine shift to R=256 (within 4,096 queries, both estimators detect shift).
    for i in 0..4_096u64 {
        let q = QueryRange::with_len(10_000_000 + i * 512, 256).unwrap();
        exact.record_query(q, false);
        sampled.record_query(q, false);
    }
    assert_eq!(exact.effective_query_span(), 256);
    assert_eq!(sampled.effective_query_span(), 256);

    // Phase 3: shift back down to R=32.
    for i in 0..4_096u64 {
        let q = QueryRange::with_len(20_000_000 + i * 64, 32).unwrap();
        exact.record_query(q, false);
        sampled.record_query(q, false);
    }
    assert_eq!(exact.effective_query_span(), 32);
    assert_eq!(sampled.effective_query_span(), 32);
}

#[test]
fn disabled_estimator_fast_path_skips_recording_and_supports_dynamic_toggle() {
    let default_cfg = EstimatorConfig::default();
    assert!(default_cfg.enabled);

    let disabled_cfg = EstimatorConfig {
        window_size: 128,
        cooldown_queries: 32,
        required_sustained_evals: 2,
        ..EstimatorConfig::default()
    }
    .with_enabled(false);
    assert!(!disabled_cfg.enabled);

    let mut est = SlidingWindowEstimator::with_initial_span(disabled_cfg, 32).unwrap();
    assert!(!est.is_enabled());

    for i in 0..2_048u64 {
        est.record_query(QueryRange::with_len(i * 1_000, 512).unwrap(), i % 7 == 0);
    }
    assert_eq!(est.total_queries(), 0);
    assert_eq!(est.effective_query_span(), 32);
    assert_eq!(est.span_transitions(), 0);
    let snap = est.snapshot();
    assert_eq!(snap.window_query_count, 0);
    assert_eq!(snap.observed_fpr, 0.0);

    est.set_enabled(true);
    assert!(est.is_enabled());
    assert!(est.config().enabled);
    for i in 0..512u64 {
        est.record_query(QueryRange::with_len(i * 1_000, 512).unwrap(), false);
    }
    assert_eq!(est.total_queries(), 512);
    assert_eq!(est.effective_query_span(), 512);
    assert_eq!(est.span_transitions(), 1);

    est.set_enabled(false);
    assert!(!est.is_enabled());
    for i in 0..512u64 {
        est.record_query(QueryRange::with_len(i * 1_000, 16).unwrap(), true);
    }
    assert_eq!(est.total_queries(), 512);
    assert_eq!(est.effective_query_span(), 512);
}

#[test]
fn below_min_clip_samples_still_caps_quantiles_at_max_clip_span() {
    let cfg = EstimatorConfig {
        max_clip_span: 64,
        ..EstimatorConfig::default()
    };
    let q = compute_clipped_quantiles(&[8, 16, 1024, 2048], &cfg, 10);
    assert_eq!(q.clipped_outliers, 2);
    assert_eq!(q.p50, 64);
    assert_eq!(q.p95, 64);
    assert_eq!(q.p99, 64);

    // Defensive fallback when evicted value is absent from sorted vec.
    let mut sorted = vec![10, 30];
    replace_sorted_sample(&mut sorted, 999, 20);
    assert_eq!(sorted, vec![10, 20, 30]);
}
