//! Reference-model equivalence and boundary invariant tests for [`super::SlidingWindowEstimator`].

use super::quantile::{
    exact_median_sorted, nearest_rank_quantile_ratio, replace_sorted_sample, ClippedQuantiles,
};
use super::*;
use aeris_types::WorkloadEstimator;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

#[test]
fn integer_nearest_rank_quantile_matches_float_ceil_across_all_sizes() {
    let cases = [
        (3usize, 5usize, 0.60_f64),
        (19, 20, 0.95_f64),
        (99, 100, 0.99_f64),
    ];
    for &(num, den, p) in &cases {
        for n in 1..=100_000usize {
            let int_rank = n.saturating_mul(num).div_ceil(den);
            let int_idx = int_rank.saturating_sub(1).min(n - 1);
            let flt_rank = ((n as f64) * p).ceil() as usize;
            let flt_idx = flt_rank.saturating_sub(1).min(n - 1);
            assert_eq!(
                int_idx, flt_idx,
                "index mismatch at n={n}, p={p}: int={int_idx}, flt={flt_idx}"
            );
            assert_eq!(
                nearest_rank_quantile_ratio(&[42], num, den),
                42,
                "single-element slice invariant failed"
            );
        }
    }
}

#[test]
fn replace_sorted_sample_preserves_multiset_and_sorted_order_exhaustively() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE_50A7);
    for cap in [1usize, 2, 3, 7, 16, 64] {
        let mut ring: Vec<u64> = (0..cap)
            .map(|_| match rng.gen_range(0..6) {
                0 => 1,
                1 => u64::MAX,
                2 => 16,
                _ => rng.gen_range(1..=64),
            })
            .collect();
        let mut sorted = ring.clone();
        sorted.sort_unstable();

        let mut head = 0usize;
        for _ in 0..1_000 {
            let new_val = match rng.gen_range(0..8) {
                0 => 1,
                1 => u64::MAX,
                2 => ring[head],
                _ => rng.gen_range(1..=64),
            };
            let old_val = ring[head];
            ring[head] = new_val;
            head = (head + 1) % cap;

            if old_val != new_val {
                replace_sorted_sample(&mut sorted, old_val, new_val);
            }

            let mut expected = ring.clone();
            expected.sort_unstable();
            assert_eq!(sorted, expected, "sorted multiset diverged for cap={cap}");
        }
    }
}

#[test]
fn steady_state_equal_eviction_still_transitions_after_cooldown_and_breach_streak() {
    let cfg = EstimatorConfig {
        window_size: 256,
        min_samples_for_clipping: 32,
        cooldown_queries: 512,
        required_sustained_evals: 4,
        hysteresis_ratio: 2.0,
        sample_stride: 64,
        ..EstimatorConfig::default()
    };
    let mut est = SlidingWindowEstimator::with_initial_span(cfg, 16).unwrap();

    for i in 0..511u64 {
        est.record_query(QueryRange::with_len(i * 512, 256).unwrap(), false);
    }
    assert_eq!(est.effective_query_span(), 16);
    assert_eq!(est.span_transitions(), 0);

    for i in 511..1_024u64 {
        est.record_query(QueryRange::with_len(i * 512, 256).unwrap(), false);
    }
    assert_eq!(est.effective_query_span(), 256);
    assert_eq!(est.span_transitions(), 1);
}

#[test]
fn optimized_estimator_matches_reference_model_on_shifting_and_poisoned_workloads() {
    struct RefEstimator {
        cfg: EstimatorConfig,
        mask: u64,
        phase_step: u64,
        phase: u64,
        next_sample: u64,
        cap: usize,
        window: VecDeque<u64>,
        fps: VecDeque<u64>,
        total: u64,
        since_change: u64,
        since_eval: u64,
        eff: u64,
        dir: Option<BreachDirection>,
        evals: u32,
        transitions: u64,
    }
    impl RefEstimator {
        fn new(cfg: EstimatorConfig, initial: u64) -> Self {
            let cap = cfg.sample_window_capacity();
            Self {
                mask: cfg.sample_stride - 1,
                phase_step: golden_odd_phase_step(cfg.sample_stride),
                phase: 0,
                next_sample: 0,
                cap,
                window: VecDeque::with_capacity(cap),
                fps: VecDeque::new(),
                total: 0,
                since_change: 0,
                since_eval: 0,
                eff: initial.clamp(1, cfg.max_clip_span),
                dir: None,
                evals: 0,
                transitions: 0,
                cfg,
            }
        }
        fn quantiles(&self) -> ClippedQuantiles {
            let mut sorted: Vec<u64> = self.window.iter().copied().collect();
            sorted.sort_unstable();
            let n = sorted.len();
            if n == 0 {
                return ClippedQuantiles::EMPTY;
            }
            let q_at = |s: &[u64], p: f64| -> u64 {
                let r = ((s.len() as f64) * p).ceil() as usize;
                s[r.saturating_sub(1).min(s.len() - 1)]
            };
            let min_clip = self.cfg.effective_min_clip_samples();
            let (inliers, clipped) = if n >= min_clip {
                let raw_p60 = q_at(&sorted, 0.60);
                let outlier_cap = raw_p60
                    .saturating_mul(self.cfg.outlier_median_multiplier)
                    .max(self.cfg.outlier_floor_span);
                let clip_cap = outlier_cap.min(self.cfg.max_clip_span);
                let ic = sorted.partition_point(|&v| v <= outlier_cap).max(1);
                let wc = sorted.partition_point(|&v| v <= clip_cap);
                (&sorted[..ic], (n - wc) as u64)
            } else {
                let wm = sorted.partition_point(|&v| v <= self.cfg.max_clip_span);
                (&sorted[..], (n - wm) as u64)
            };
            let cap = self.cfg.max_clip_span;
            ClippedQuantiles {
                p50: exact_median_sorted(inliers).min(cap).max(1),
                p95: q_at(inliers, 0.95).min(cap).max(1),
                p99: q_at(inliers, 0.99).min(cap).max(1),
                clipped_outliers: clipped,
            }
        }
        fn eval(&mut self) {
            let min_eval = (self.cap / 2).clamp(1, 16);
            if self.window.len() < min_eval || self.since_change < self.cfg.cooldown_queries {
                self.dir = None;
                self.evals = 0;
                return;
            }
            let q = self.quantiles();
            let cand = q.p95.max(1);
            let eff_f = self.eff.max(1) as f64;
            let cand_f = cand as f64;
            let breach = if cand_f > eff_f * self.cfg.hysteresis_ratio {
                Some(BreachDirection::Up)
            } else if cand_f < eff_f / self.cfg.hysteresis_ratio {
                Some(BreachDirection::Down)
            } else {
                None
            };
            let Some(d) = breach else {
                self.dir = None;
                self.evals = 0;
                return;
            };
            if self.dir == Some(d) {
                self.evals += 1;
            } else {
                self.dir = Some(d);
                self.evals = 1;
            }
            if self.evals >= self.cfg.required_sustained_evals {
                self.eff = cand;
                self.since_change = 0;
                self.since_eval = 0;
                self.dir = None;
                self.evals = 0;
                self.transitions += 1;
            }
        }
        fn record(&mut self, range: QueryRange, is_fp: bool) {
            let seq = self.total;
            self.total += 1;
            self.since_change += 1;
            self.since_eval += 1;
            if is_fp {
                let cutoff = self.total.saturating_sub(self.cfg.window_size as u64);
                while self.fps.front().is_some_and(|&f| f < cutoff) {
                    self.fps.pop_front();
                }
                self.fps.push_back(seq);
            }
            if seq == self.next_sample {
                self.phase = self.phase.wrapping_add(self.phase_step) & self.mask;
                let next_block = (seq & !self.mask).saturating_add(self.cfg.sample_stride);
                self.next_sample = next_block.wrapping_add(self.phase);
                if self.window.len() == self.cap {
                    self.window.pop_front();
                }
                self.window.push_back(range.len_saturated().max(1));
                let stride = self.cfg.auto_eval_stride();
                if self.since_change >= self.cfg.cooldown_queries
                    && (self.since_eval >= stride || self.cfg.sample_stride >= stride)
                {
                    self.since_eval = 0;
                    self.eval();
                }
            }
        }
        fn active_fps(&self) -> u64 {
            let cutoff = self.total.saturating_sub(self.cfg.window_size as u64);
            let exp = self.fps.partition_point(|&s| s < cutoff);
            (self.fps.len() - exp) as u64
        }
    }

    let configs = [
        EstimatorConfig::default(),
        EstimatorConfig {
            window_size: 64,
            sample_stride: 64,
            cooldown_queries: 128,
            ..EstimatorConfig::default()
        },
        EstimatorConfig {
            window_size: 256,
            sample_stride: 1,
            cooldown_queries: 100,
            required_sustained_evals: 4,
            ..EstimatorConfig::default()
        },
    ];

    for cfg in configs {
        let mut opt = SlidingWindowEstimator::with_initial_span(cfg.clone(), 16).unwrap();
        let mut ref_est = RefEstimator::new(cfg.clone(), 16);
        let mut rng = ChaCha8Rng::seed_from_u64(0xE0A1_7E57);

        for step in 0..20_000u64 {
            let span = match step / 4_000 {
                0 => 16,
                1 => 256,
                2 => {
                    if step % 12 == 0 {
                        10_000_000
                    } else {
                        rng.gen_range(24..=40)
                    }
                }
                3 => 1024,
                _ => 16,
            };
            let is_fp = step % 313 == 0;
            let range = QueryRange::with_len(step * 2_048, span).unwrap();
            opt.record_query(range, is_fp);
            ref_est.record(range, is_fp);

            if step % 64 == 0 || step == 19_999 {
                let s_opt = opt.snapshot();
                let q_ref = ref_est.quantiles();
                let w_cnt = ref_est.total.min(cfg.window_size as u64);
                let ref_fpr = if w_cnt == 0 {
                    0.0
                } else {
                    (ref_est.active_fps() as f64) / (w_cnt as f64)
                };
                let ref_clipped = q_ref
                    .clipped_outliers
                    .saturating_mul(cfg.sample_stride)
                    .min(w_cnt);

                assert_eq!(opt.total_queries(), ref_est.total);
                assert_eq!(opt.queries_since_last_change(), ref_est.since_change);
                assert_eq!(opt.effective_query_span(), ref_est.eff);
                assert_eq!(opt.span_transitions(), ref_est.transitions);
                assert_eq!(s_opt.window_query_count, w_cnt);
                assert_eq!(s_opt.p50_query_len, q_ref.p50);
                assert_eq!(s_opt.p95_query_len, q_ref.p95);
                assert_eq!(s_opt.p99_query_len, q_ref.p99);
                assert_eq!(s_opt.clipped_outliers, ref_clipped);
                assert!((s_opt.observed_fpr - ref_fpr).abs() < 1e-12);
            }
        }
    }
}
