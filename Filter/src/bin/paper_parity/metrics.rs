//! Telemetry aggregation, latency quantiles, and parity JSON assembly.
//!
//! All system introspection (`query_peak_rss_mb`, `detect_compiler_version`) runs
//! strictly outside timed benchmark loops.

use crate::cli::CliArgs;
use aeris_types::FilterConfig;
use serde::Serialize;
use std::process::Command;

/// Maximum number of strided latency samples collected per run.
pub const MAX_LATENCY_SAMPLES: usize = 100_000;

/// Cross-language JSON schema emitted by both C++ and Rust paper-parity drivers.
#[derive(Clone, Debug, Serialize)]
pub struct ParityResult {
    /// Driver implementation name (`rust_naive` or `rust_optimized`).
    #[serde(rename = "impl")]
    pub impl_name: String,
    /// Benchmark scenario label.
    pub scenario: String,
    /// Number of loaded keys.
    pub keys: usize,
    /// Number of executed queries.
    pub queries: usize,
    /// Final quotient bit width `q`.
    pub q_bits: u8,
    /// Final remainder bit width `r`.
    pub r_bits: u8,
    /// Final range-extension bit width `m`.
    pub m_bits: u8,
    /// Final slot occupancy fraction.
    pub load_factor: f64,
    /// Query throughput (queries/second).
    pub qps: f64,
    /// Bulk-load throughput (keys/second).
    pub bulk_load_keys_per_sec: f64,
    /// Incremental insert throughput (keys/second).
    pub insert_keys_per_sec: f64,
    /// Median sampled operation latency (nanoseconds).
    pub p50_ns: u64,
    /// 99th-percentile sampled operation latency (nanoseconds).
    pub p99_ns: u64,
    /// Measured false-positive rate over empty queries.
    pub fpr: f64,
    /// Total observed false positives.
    pub fp_count: u64,
    /// Total observed false negatives (must be `0`).
    pub fn_count: u64,
    /// Observed repeated false positives across identical ranges.
    pub repeated_fp_count: u64,
    /// Successful filter adaptations invoked.
    pub adaptations: u64,
    /// Amortized bits per key from final filter memory footprint.
    pub bits_per_key: f64,
    /// Final filter heap/array footprint in bytes.
    pub filter_bytes: usize,
    /// Process peak resident set size in MiB.
    pub peak_rss_mb: f64,
    /// Rust compiler version string.
    pub compiler: String,
    /// Release optimization flags string.
    pub flags: String,
    /// Structured semicolon-separated mode telemetry notes.
    pub notes: String,
}

/// Mode execution summary passed to [`assemble_result`].
pub struct ModeOutcome {
    /// Final filter geometry configuration.
    pub cfg: FilterConfig,
    /// Final filter load factor.
    pub load_factor: f64,
    /// Total key count.
    pub keys: usize,
    /// Total query count.
    pub queries: usize,
    /// Measured query throughput (queries/sec).
    pub qps: f64,
    /// Measured bulk-load throughput (keys/sec).
    pub bulk_load_keys_per_sec: f64,
    /// Measured incremental insert throughput (keys/sec).
    pub insert_keys_per_sec: f64,
    /// Median sampled latency (ns).
    pub p50_ns: u64,
    /// 99th-percentile sampled latency (ns).
    pub p99_ns: u64,
    /// Measured false-positive rate.
    pub fpr: f64,
    /// Observed false positives.
    pub fp_count: u64,
    /// Observed false negatives.
    pub fn_count: u64,
    /// Observed repeated false positives.
    pub repeated_fp_count: u64,
    /// Completed adaptations.
    pub adaptations: u64,
    /// Final filter memory footprint in bytes.
    pub filter_bytes: usize,
    /// Mode-specific structured notes string.
    pub notes: String,
}

/// Computes `(p50_ns, p99_ns)` from an unsorted vector of nanosecond latency samples.
pub fn compute_p50_p99(mut samples: Vec<u64>) -> (u64, u64) {
    if samples.is_empty() {
        return (0, 0);
    }
    samples.sort_unstable();
    let n = samples.len();
    let p50_idx = ((n - 1) * 500) / 1000;
    let p99_idx = ((n - 1) * 990) / 1000;
    (samples[p50_idx], samples[p99_idx])
}

/// Computes amortized bits per key for `filter_bytes` and `key_count`.
#[inline]
pub fn compute_bits_per_key(filter_bytes: usize, key_count: usize) -> f64 {
    if key_count == 0 {
        0.0
    } else {
        (filter_bytes as f64) * 8.0 / (key_count as f64)
    }
}

/// Computes false-positive rate over `empty_count` verified-empty queries.
#[inline]
pub fn compute_fpr(fp_count: u64, empty_count: u64) -> f64 {
    if empty_count > 0 {
        (fp_count as f64) / (empty_count as f64)
    } else {
        0.0
    }
}

/// Queries the current process Peak Resident Set Size in MiB without `unsafe` FFI.
///
/// Invoked strictly after all benchmark timers have stopped.
pub fn query_peak_rss_mb() -> f64 {
    #[cfg(target_os = "windows")]
    {
        let pid = std::process::id();
        let script = format!("(Get-Process -Id {pid}).PeakWorkingSet64");
        if let Ok(output) = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
        {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    if let Ok(bytes) = text.trim().parse::<u64>() {
                        return (bytes as f64) / (1024.0 * 1024.0);
                    }
                }
            }
        }
        0.0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(kb_str) = rest.split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return (kb as f64) / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        0.0
    }
}

/// Queries `rustc --version` once (strictly outside timed regions).
pub fn detect_compiler_version() -> String {
    if let Ok(output) = Command::new("rustc").arg("--version").output() {
        if output.status.success() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    "rustc".to_string()
}

/// Assembles the canonical 24-field [`ParityResult`] after querying Peak RSS and
/// compiler metadata outside all benchmark timers.
pub fn assemble_result(args: &CliArgs, outcome: ModeOutcome) -> ParityResult {
    let peak_rss_mb = query_peak_rss_mb();
    let compiler = detect_compiler_version();
    let flags = "-C opt-level=3 -C lto=fat -C codegen-units=1 -C target-cpu=native".to_string();
    let bits_per_key = compute_bits_per_key(outcome.filter_bytes, outcome.keys);

    ParityResult {
        impl_name: args.impl_variant.as_str().to_string(),
        scenario: args.scenario.clone(),
        keys: outcome.keys,
        queries: outcome.queries,
        q_bits: outcome.cfg.q,
        r_bits: outcome.cfg.r,
        m_bits: outcome.cfg.m,
        load_factor: outcome.load_factor,
        qps: outcome.qps,
        bulk_load_keys_per_sec: outcome.bulk_load_keys_per_sec,
        insert_keys_per_sec: outcome.insert_keys_per_sec,
        p50_ns: outcome.p50_ns,
        p99_ns: outcome.p99_ns,
        fpr: outcome.fpr,
        fp_count: outcome.fp_count,
        fn_count: outcome.fn_count,
        repeated_fp_count: outcome.repeated_fp_count,
        adaptations: outcome.adaptations,
        bits_per_key,
        filter_bytes: outcome.filter_bytes,
        peak_rss_mb,
        compiler,
        flags,
        notes: outcome.notes,
    }
}
