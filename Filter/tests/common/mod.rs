//! Shared deterministic statistical workload generators, exact ground-truth
//! oracles, latency percentile aggregators, and structured tracing bootstrap
//! consumed by the integration, concurrency, and E2E test suites.
//!
//! All workloads are generated deterministically from seeded [`ChaCha8Rng`]
//! streams so false-positive bounds, throughput scaling, and adversarial
//! experiments are 100% reproducible without hardcoded toy datasets.

#![allow(dead_code)]

use aeris_types::{Key, QueryRange};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

/// Default 48-bit keyspace upper bound (`[0, 2^48)`) keeping uniform keys
/// well-spaced while preserving genuine 64-bit integer arithmetic and full
/// partition/memento coverage without wrap-around overflow.
pub const KEYSPACE_MASK: u64 = (1u64 << 48) - 1;

/// Default Zipfian skew parameter matching `Aeris Filters.pdf` Section 5.2.
pub const DEFAULT_ZIPF_ALPHA: f64 = 1.5;

/// Default correlation parameter `D` matching `Aeris Filters.pdf` Section 5.2.
pub const DEFAULT_CORRELATION_D: f64 = 0.8;

static TRACING_INIT: Once = Once::new();

/// Initializes structured `tracing` subscriber once per test binary.
///
/// Honors `RUST_LOG` when set; otherwise defaults to `info` level so structured
/// telemetry reports appear cleanly under `--nocapture` without raw `println!`.
pub fn init_test_tracing() {
    TRACING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_test_writer()
            .try_init();
    });
}

/// Deterministic discrete Zipfian sampler over ranks `0..domain_size` with
/// probability mass `P(rank = i) proportional to 1 / (i + 1)^alpha`.
#[derive(Clone, Debug)]
pub struct ZipfianSampler {
    cdf: Vec<f64>,
}

impl ZipfianSampler {
    /// Constructs an exact inverse-CDF Zipfian sampler for `domain_size >= 1`
    /// and skew `alpha > 0.0`.
    ///
    /// # Panics
    /// Panics if `domain_size == 0` or `alpha <= 0.0`.
    ///
    /// # Complexity
    /// `O(domain_size)` construction time and space; `O(log domain_size)` per sample.
    #[must_use]
    pub fn new(domain_size: usize, alpha: f64) -> Self {
        assert!(domain_size > 0, "Zipfian domain_size must be at least 1");
        assert!(
            alpha.is_finite() && alpha > 0.0,
            "Zipfian alpha must be finite and > 0.0"
        );

        let mut cdf = Vec::with_capacity(domain_size);
        let mut cumulative = 0.0_f64;
        for rank in 1..=domain_size {
            cumulative += 1.0 / (rank as f64).powf(alpha);
            cdf.push(cumulative);
        }
        let inv_total = 1.0 / cumulative;
        for entry in &mut cdf {
            *entry *= inv_total;
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self { cdf }
    }

    /// Samples a zero-based rank in `0..self.cdf.len()` from `rng`.
    #[inline]
    #[must_use]
    pub fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let u: f64 = rng.gen();
        let idx = self.cdf.partition_point(|&c| c < u);
        idx.min(self.cdf.len() - 1)
    }
}

/// Generates `count` unique uniformly distributed keys within `[0, KEYSPACE_MASK]`
/// using a seeded [`ChaCha8Rng`], returning `(sorted_keys, key_set)`.
#[must_use]
pub fn generate_uniform_keys(seed: u64, count: usize) -> (Vec<Key>, BTreeSet<Key>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut set = BTreeSet::new();
    while set.len() < count {
        let k = rng.gen::<u64>() & KEYSPACE_MASK;
        set.insert(k);
    }
    let vec: Vec<Key> = set.iter().copied().collect();
    (vec, set)
}

/// Exact ground-truth emptiness check against an in-memory sorted key set.
#[inline]
#[must_use]
pub fn is_range_empty(key_set: &BTreeSet<Key>, range: QueryRange) -> bool {
    key_set.range(range.start()..=range.end()).next().is_none()
}

/// Generates `count` verified-empty uniform closed range queries of exact length
/// `range_len >= 1` disjoint from `key_set`.
#[must_use]
pub fn generate_uniform_empty_queries(
    seed: u64,
    key_set: &BTreeSet<Key>,
    range_len: u64,
    count: usize,
) -> Vec<QueryRange> {
    assert!(range_len >= 1, "range_len must be >= 1");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);
    let mut queries = Vec::with_capacity(count);

    while queries.len() < count {
        let start = rng.gen_range(0..=max_start);
        let range = QueryRange::with_len(start, range_len).expect("range_len >= 1");
        if is_range_empty(key_set, range) {
            queries.push(range);
        }
    }
    queries
}

/// Generates `count` verified-empty correlated Zipfian range queries matching
/// `Aeris Filters.pdf` Section 5.2 (`alpha = 1.5`, correlation degree `D = 0.8`).
///
/// With probability `D`, anchors a query to a Zipfian-selected existing key `k`
/// and places an empty range of length `range_len` within distance
/// `[1, 2^round(30 * (1 - D))]` of `k` (probing adjacent/same partitions).
/// With probability `1 - D`, emits a uniformly random empty range.
#[must_use]
pub fn generate_correlated_zipfian_empty_queries(
    seed: u64,
    sorted_keys: &[Key],
    key_set: &BTreeSet<Key>,
    range_len: u64,
    alpha: f64,
    correlation_d: f64,
    count: usize,
) -> Vec<QueryRange> {
    assert!(!sorted_keys.is_empty(), "sorted_keys must be non-empty");
    assert!(range_len >= 1, "range_len must be >= 1");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let hot_domain = sorted_keys.len().min(4_096);
    let zipf = ZipfianSampler::new(hot_domain, alpha);

    // Deterministic permutation of anchor indices so hot Zipfian ranks spread across the keyspace.
    let stride = (sorted_keys.len() / hot_domain).max(1);
    let exp_bits = ((30.0 * (1.0 - correlation_d.clamp(0.0, 1.0))).round() as u32).clamp(1, 30);
    let max_offset = (1u64 << exp_bits).max(2);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);

    // Precompute deterministic hot correlated offsets per rank so Zipfian skew genuinely
    // revisits the same correlated empty neighborhoods (matching Section 5.2).
    let mut rank_queries: Vec<QueryRange> = Vec::with_capacity(hot_domain);
    for rank in 0..hot_domain {
        let anchor_idx = (rank.wrapping_mul(stride) + (rank.wrapping_mul(2654435761) % stride))
            % sorted_keys.len();
        let anchor = sorted_keys[anchor_idx];
        let mut chosen = None;
        for attempt in 0..32u64 {
            let offset = rng.gen_range(1..=max_offset).saturating_add(attempt);
            let start = anchor.saturating_add(offset).min(max_start);
            let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
            if is_range_empty(key_set, candidate) {
                chosen = Some(candidate);
                break;
            }
        }
        if let Some(q) = chosen {
            rank_queries.push(q);
        } else {
            loop {
                let start = rng.gen_range(0..=max_start);
                let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
                if is_range_empty(key_set, candidate) {
                    rank_queries.push(candidate);
                    break;
                }
            }
        }
    }

    let mut queries = Vec::with_capacity(count);
    while queries.len() < count {
        let draw: f64 = rng.gen();
        if draw < correlation_d {
            let rank = zipf.sample(&mut rng);
            queries.push(rank_queries[rank]);
        } else {
            let start = rng.gen_range(0..=max_start);
            let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
            if is_range_empty(key_set, candidate) {
                queries.push(candidate);
            }
        }
    }
    queries
}

/// Generates `count` verified-empty uncorrelated Zipfian range queries (`alpha = 1.5`)
/// drawn over a seeded pool of empty intervals independent of dataset keys.
#[must_use]
pub fn generate_uncorrelated_zipfian_empty_queries(
    seed: u64,
    key_set: &BTreeSet<Key>,
    range_len: u64,
    alpha: f64,
    pool_size: usize,
    count: usize,
) -> Vec<QueryRange> {
    let pool = generate_uniform_empty_queries(seed ^ 0x51A7_900D, key_set, range_len, pool_size);
    let zipf = ZipfianSampler::new(pool.len(), alpha);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_1234);

    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        let rank = zipf.sample(&mut rng);
        queries.push(pool[rank]);
    }
    queries
}

/// Builds an enclosing closed [`QueryRange`] of length `<= span` containing `key`
/// without overflowing `u64` boundaries.
#[inline]
#[must_use]
pub fn enclosing_range(key: Key, half_width: u64) -> QueryRange {
    let start = key.saturating_sub(half_width);
    let end = key.saturating_add(half_width);
    QueryRange::new(start, end).expect("start <= end")
}

/// Summary latency statistics in nanoseconds computed over a sorted sample slice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatencySummary {
    /// Minimum observed latency (ns).
    pub min_ns: u64,
    /// Median (`p50`) latency (ns).
    pub p50_ns: u64,
    /// 95th percentile latency (ns).
    pub p95_ns: u64,
    /// 99th percentile latency (ns).
    pub p99_ns: u64,
    /// Maximum observed latency (ns).
    pub max_ns: u64,
    /// Arithmetic mean latency (ns).
    pub mean_ns: f64,
}

impl LatencySummary {
    /// Computes exact order-statistic latency percentiles from `samples_ns`.
    #[must_use]
    pub fn from_samples(mut samples_ns: Vec<u64>) -> Self {
        if samples_ns.is_empty() {
            return Self::default();
        }
        samples_ns.sort_unstable();
        let n = samples_ns.len();
        let sum: u128 = samples_ns.iter().map(|&v| u128::from(v)).sum();
        Self {
            min_ns: samples_ns[0],
            p50_ns: samples_ns[percentile_idx(n, 50)],
            p95_ns: samples_ns[percentile_idx(n, 95)],
            p99_ns: samples_ns[percentile_idx(n, 99)],
            max_ns: samples_ns[n - 1],
            mean_ns: (sum as f64) / (n as f64),
        }
    }
}

#[inline]
fn percentile_idx(len: usize, pct: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let num = (len - 1).saturating_mul(pct);
    (num / 100).min(len - 1)
}
