//! Shared filter probing, ground-truth verification, and read-only pass helpers.
//!
//! All hot-path probe functions are `#[inline]` and perform zero heap allocations.

use crate::io::QueryRecord;
use crate::metrics::{compute_fpr, compute_p50_p99, MAX_LATENCY_SAMPLES};
use aeris_core::AerisFilter;
use aeris_types::{Key, QueryRange, RangeFilter};
use std::time::Instant;

/// Summary metrics returned by [`run_readonly_query_pass`].
pub struct ReadonlyPassStats {
    /// Measured throughput over the uninstrumented pass (queries/sec).
    pub qps: f64,
    /// Total observed false positives.
    pub fp_count: u64,
    /// Total observed false negatives on non-empty queries.
    pub fn_count: u64,
    /// Measured false-positive rate over empty queries.
    pub fpr: f64,
    /// Median sampled query latency (ns).
    pub p50_ns: u64,
    /// 99th-percentile sampled query latency (ns).
    pub p99_ns: u64,
}

/// Exact `O(log N)` closed-range emptiness check against a sorted key slice.
#[inline]
pub fn is_range_empty(sorted_keys: &[Key], start: u64, end: u64) -> bool {
    let idx = sorted_keys.partition_point(|&k| k < start);
    match sorted_keys.get(idx) {
        Some(&k) => k > end,
        None => true,
    }
}

/// Evaluates `query_point` for singleton ranges and `query_range` otherwise.
#[inline]
pub fn probe_filter(filter: &AerisFilter, range: QueryRange) -> bool {
    if range.start() == range.end() {
        filter.query_point(range.start())
    } else {
        filter.query_range(range)
    }
}

/// Verifies 100% of inserted keys return `true` under `query_point` (Zero FN invariant).
pub fn verify_all_keys_zero_fn(filter: &AerisFilter, keys: &[Key]) -> u64 {
    let mut missed = 0u64;
    for &k in keys {
        if !filter.query_point(k) {
            missed += 1;
        }
    }
    missed
}

/// Runs an uninstrumented read-only query pass followed by a strided latency pass
/// (`<= MAX_LATENCY_SAMPLES` samples) over `queries`.
pub fn run_readonly_query_pass(filter: &AerisFilter, queries: &[QueryRecord]) -> ReadonlyPassStats {
    if queries.is_empty() {
        return ReadonlyPassStats {
            qps: 0.0,
            fp_count: 0,
            fn_count: 0,
            fpr: 0.0,
            p50_ns: 0,
            p99_ns: 0,
        };
    }

    let (qps, fp_count, fn_count, fpr) = measure_readonly_throughput(filter, queries);
    let (p50_ns, p99_ns) = sample_readonly_latency(filter, queries);

    ReadonlyPassStats {
        qps,
        fp_count,
        fn_count,
        fpr,
        p50_ns,
        p99_ns,
    }
}

fn measure_readonly_throughput(
    filter: &AerisFilter,
    queries: &[QueryRecord],
) -> (f64, u64, u64, f64) {
    let mut fp_count = 0u64;
    let mut fn_count = 0u64;
    let mut empty_count = 0u64;

    let t_q = Instant::now();
    for qr in queries {
        let hit = probe_filter(filter, qr.range);
        if qr.expected == 0 {
            empty_count += 1;
            if hit {
                fp_count += 1;
            }
        } else if !hit {
            fn_count += 1;
        }
    }
    let elapsed_s = t_q.elapsed().as_secs_f64();
    let qps = (queries.len() as f64) / elapsed_s.max(1e-12);
    let fpr = compute_fpr(fp_count, empty_count);
    (qps, fp_count, fn_count, fpr)
}

fn sample_readonly_latency(filter: &AerisFilter, queries: &[QueryRecord]) -> (u64, u64) {
    let stride = (queries.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut lat_samples = Vec::with_capacity(queries.len().min(MAX_LATENCY_SAMPLES) + 1);
    let mut sink = 0u64;
    for idx in (0..queries.len()).step_by(stride) {
        let qr = queries[idx];
        let t0 = Instant::now();
        let hit = probe_filter(filter, qr.range);
        let dt = t0.elapsed().as_nanos() as u64;
        sink = sink.wrapping_add(u64::from(hit));
        lat_samples.push(dt);
    }
    std::hint::black_box(sink);
    compute_p50_p99(lat_samples)
}
