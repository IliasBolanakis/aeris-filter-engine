//! Paper-parity benchmark driver for the Rust Aeris Filtering System.
//!
//! Consumes the exact binary key (`*.keys.bin`) and query (`*.queries.bin`) trace
//! files produced by `trace_gen` and shared verbatim with the C++ SIGMOD 2026
//! reference driver, emitting a single structured JSON record conforming to the
//! cross-language parity schema.
//!
//! # Modes
//! - `query_adapt`: Bulk-load with [`InMemReverseMap`], execute queries with
//!   monotonic `adapt()` on every false positive, verify 100% key FN == 0.
//! - `bulk_load`: Bulk-load with [`NoopReverseMap`] (pure filter scaling), run
//!   optional read-only query throughput + strided latency passes, verify 100% FN == 0.
//! - `dynamic_insert`: Incremental `insert()` of every key with strided latency
//!   sampling, optional read-only query pass, verify 100% FN == 0.
//! - `adversarial`: Deterministic 10% repeat-FP adversarial schedule (`fp_pool` cap 256)
//!   with monotonic adaptation, verifying `repeated_fp_count == 0` and `fn_count == 0`.
//! - `shifting_r`: Multi-phase shifting range length (`[16, 256, 32, 1024]`) comparing
//!   static-`m` (`--impl rust_naive`) against Control-Plane dynamic-`m` reconfiguration
//!   with [`SlidingWindowEstimator`], [`DiscreteCostOptimizer`], and [`FalsePositiveLog`]
//!   replay (`--impl rust_optimized`).

use aeris_control::{
    DiscreteCostOptimizer, EstimatorConfig, FalsePositiveLog, FpLogConfig, SlidingWindowEstimator,
};
use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, CostModelParams, CostOptimizer, FilterConfig, Fingerprint, Key,
    QueryRange, RangeFilter, ReverseMapStore, WorkloadEstimator,
};
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

/// 8-byte magic prefix for binary key trace files.
const KEYS_MAGIC: [u8; 8] = *b"AERISKEY";

/// 8-byte magic prefix for binary query trace files.
const QUERIES_MAGIC: [u8; 8] = *b"AERISQRY";

/// Sequential I/O read buffer size (16 MiB).
const IO_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Maximum number of strided latency samples collected per run.
const MAX_LATENCY_SAMPLES: usize = 100_000;

/// Maximum adversarial repeat-FP pool capacity (matches C++ driver).
const ADVERSARIAL_FP_POOL_CAP: usize = 256;

/// Control-plane optimizer evaluation cadence (queries) in `shifting_r`.
const OPTIMIZER_EVAL_INTERVAL: usize = 256;

/// Default target false-positive epsilon (`r_min = 7`).
const DEFAULT_TARGET_EPSILON: f64 = 0.0625;

/// Default deterministic hash seed.
const DEFAULT_SEED: u64 = 1380;

/// Supported benchmark modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchMode {
    QueryAdapt,
    BulkLoad,
    DynamicInsert,
    Adversarial,
    ShiftingR,
}

impl BenchMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "query_adapt" => Ok(Self::QueryAdapt),
            "bulk_load" => Ok(Self::BulkLoad),
            "dynamic_insert" => Ok(Self::DynamicInsert),
            "adversarial" => Ok(Self::Adversarial),
            "shifting_r" => Ok(Self::ShiftingR),
            other => Err(format!(
                "unsupported --mode '{other}': expected query_adapt | bulk_load | dynamic_insert | adversarial | shifting_r"
            )),
        }
    }

    const fn requires_queries(self) -> bool {
        matches!(self, Self::QueryAdapt | Self::Adversarial | Self::ShiftingR)
    }
}

/// Supported Rust driver implementation variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImplVariant {
    RustNaive,
    RustOptimized,
}

impl ImplVariant {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "rust_naive" => Ok(Self::RustNaive),
            "rust_optimized" => Ok(Self::RustOptimized),
            other => Err(format!(
                "unsupported --impl '{other}': expected rust_naive | rust_optimized"
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::RustNaive => "rust_naive",
            Self::RustOptimized => "rust_optimized",
        }
    }
}

/// Parsed CLI configuration for `paper_parity`.
struct CliArgs {
    mode: BenchMode,
    impl_variant: ImplVariant,
    scenario: String,
    keys_path: PathBuf,
    queries_path: Option<PathBuf>,
    q: u8,
    r: u8,
    m: u8,
    budget_c: Option<u8>,
    target_epsilon: Option<f64>,
    seed: u64,
    static_bypass: bool,
    out_path: Option<PathBuf>,
}

/// Decoded binary query trace record (`start <= end`).
#[derive(Clone, Copy, Debug)]
struct QueryRecord {
    range: QueryRange,
    expected: u64,
}

/// Cross-language JSON schema emitted by both C++ and Rust paper-parity drivers.
#[derive(Clone, Debug, Serialize)]
struct ParityResult {
    #[serde(rename = "impl")]
    impl_name: String,
    scenario: String,
    keys: usize,
    queries: usize,
    q_bits: u8,
    r_bits: u8,
    m_bits: u8,
    load_factor: f64,
    qps: f64,
    bulk_load_keys_per_sec: f64,
    insert_keys_per_sec: f64,
    p50_ns: u64,
    p99_ns: u64,
    fpr: f64,
    fp_count: u64,
    fn_count: u64,
    repeated_fp_count: u64,
    adaptations: u64,
    bits_per_key: f64,
    filter_bytes: usize,
    peak_rss_mb: f64,
    compiler: String,
    flags: String,
    notes: String,
}

/// Zero-overhead no-op [`ReverseMapStore`] for pure filter `bulk_load` and
/// `dynamic_insert` scaling runs (parity with C++ pure `QF` benchmarks).
struct NoopReverseMap;

impl ReverseMapStore for NoopReverseMap {
    fn append_key(&self, _fp: &Fingerprint, _key: Key) -> AerisResult<()> {
        Ok(())
    }

    fn append_batch(&self, _entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
        Ok(())
    }

    fn remove_key(&self, _fp: &Fingerprint, _key: Key) -> AerisResult<()> {
        Ok(())
    }

    fn get_keys(&self, _fp: &Fingerprint) -> AerisResult<Vec<Key>> {
        Ok(Vec::new())
    }

    fn tombstone(&self, _fp: &Fingerprint) -> AerisResult<()> {
        Ok(())
    }

    fn replace_fingerprint(
        &self,
        _old_fp: &Fingerprint,
        _new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        Ok(())
    }

    fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        Ok(Vec::new())
    }

    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
        Ok(Box::new(std::iter::empty()))
    }

    fn entry_count_estimate(&self) -> u64 {
        0
    }

    fn flush(&self) -> AerisResult<()> {
        Ok(())
    }

    fn compact(&self) -> AerisResult<()> {
        Ok(())
    }

    #[inline]
    fn is_noop(&self) -> bool {
        true
    }
}

/// High-speed in-memory [`ReverseMapStore`] backed by `parking_lot::RwLock<HashMap>`
/// for DRAM-parity adaptive benchmarks (matching C++ `InMemArqf`'s `unordered_multimap`).
struct InMemReverseMap {
    map: RwLock<HashMap<Fingerprint, Vec<Key>>>,
    entry_count: AtomicU64,
}

impl InMemReverseMap {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(capacity)),
            entry_count: AtomicU64::new(0),
        }
    }
}

impl ReverseMapStore for InMemReverseMap {
    fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let mut guard = self.map.write();
        guard.entry(fp.clone()).or_default().push(key);
        self.entry_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut guard = self.map.write();
        if guard.is_empty() {
            guard.reserve(entries.len());
        }
        for (fp, key) in entries {
            guard.entry(fp.clone()).or_default().push(*key);
        }
        self.entry_count
            .fetch_add(entries.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let mut guard = self.map.write();
        let mut should_remove = false;
        if let Some(vec) = guard.get_mut(fp) {
            if let Some(pos) = vec.iter().position(|&k| k == key) {
                vec.swap_remove(pos);
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
            }
            should_remove = vec.is_empty();
        }
        if should_remove {
            guard.remove(fp);
        }
        Ok(())
    }

    fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
        let guard = self.map.read();
        Ok(guard.get(fp).cloned().unwrap_or_default())
    }

    fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
        let mut guard = self.map.write();
        if let Some(removed) = guard.remove(fp) {
            self.entry_count
                .fetch_sub(removed.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn replace_fingerprint(
        &self,
        old_fp: &Fingerprint,
        new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        let mut guard = self.map.write();
        let removed_count = guard.remove(old_fp).map(|v| v.len()).unwrap_or(0);
        for (fp, key) in new_entries {
            guard.entry(fp.clone()).or_default().push(*key);
        }
        let added = new_entries.len();
        if added >= removed_count {
            self.entry_count
                .fetch_add((added - removed_count) as u64, Ordering::Relaxed);
        } else {
            self.entry_count
                .fetch_sub((removed_count - added) as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        let guard = self.map.read();
        let mut all = Vec::with_capacity(self.entry_count.load(Ordering::Relaxed) as usize);
        for keys in guard.values() {
            all.extend_from_slice(keys);
        }
        all.sort_unstable();
        all.dedup();
        Ok(all)
    }

    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
        let keys = self.scan_all_keys()?;
        Ok(Box::new(keys.into_iter().map(Ok)))
    }

    fn entry_count_estimate(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }

    fn flush(&self) -> AerisResult<()> {
        Ok(())
    }

    fn compact(&self) -> AerisResult<()> {
        Ok(())
    }
}

/// RAII cleanup guard for temporary `FalsePositiveLog` files created during `shifting_r`.
struct TempFileGuard {
    path: PathBuf,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let mut tmp_os = self.path.as_os_str().to_os_string();
        tmp_os.push(".compact.tmp");
        let _ = fs::remove_file(PathBuf::from(tmp_os));
    }
}

/// Per-phase telemetry collected during `shifting_r`.
#[derive(Clone, Debug)]
struct PhaseMetrics {
    phase_idx: usize,
    range_len: u64,
    queries: usize,
    fp_count: u64,
    pure_query_s: f64,
    wall_s: f64,
    p99_ns: u64,
    end_m: u8,
}

/// Reads all little-endian 64-bit keys from a `*.keys.bin` file.
fn read_keys_file(path: &Path) -> Result<Vec<Key>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open keys file '{}': {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);

    let mut header = [0u8; 16];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("failed to read 16B keys header: {e}"))?;
    if header[0..8] != KEYS_MAGIC {
        return Err(format!(
            "invalid keys magic in '{}': expected {:?}, got {:?}",
            path.display(),
            KEYS_MAGIC,
            &header[0..8]
        ));
    }
    let count = u64::from_le_bytes(header[8..16].try_into().expect("8B count"));
    let count_usize =
        usize::try_from(count).map_err(|_| format!("key count {count} exceeds usize"))?;

    let mut keys = Vec::with_capacity(count_usize);
    let chunk_keys = 131_072usize; // 1 MiB
    let mut buf = vec![0u8; chunk_keys * 8];
    let mut remaining = count_usize;

    while remaining > 0 {
        let batch = remaining.min(chunk_keys);
        let byte_len = batch * 8;
        reader
            .read_exact(&mut buf[..byte_len])
            .map_err(|e| format!("truncated keys payload: {e}"))?;
        for c in buf[..byte_len].as_chunks::<8>().0 {
            keys.push(u64::from_le_bytes(*c));
        }
        remaining -= batch;
    }

    Ok(keys)
}

/// Reads all 24-byte query records from a `*.queries.bin` file.
fn read_queries_file(path: &Path) -> Result<(Vec<QueryRecord>, u64), String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open queries file '{}': {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);

    let mut header = [0u8; 32];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("failed to read 32B queries header: {e}"))?;
    if header[0..8] != QUERIES_MAGIC {
        return Err(format!(
            "invalid queries magic in '{}': expected {:?}, got {:?}",
            path.display(),
            QUERIES_MAGIC,
            &header[0..8]
        ));
    }
    let count = u64::from_le_bytes(header[8..16].try_into().expect("8B count"));
    let max_range_r = u64::from_le_bytes(header[16..24].try_into().expect("8B max_range_r"));
    let _flags = u64::from_le_bytes(header[24..32].try_into().expect("8B flags"));

    let count_usize =
        usize::try_from(count).map_err(|_| format!("query count {count} exceeds usize"))?;
    let mut queries = Vec::with_capacity(count_usize);
    let chunk_records = 65_536usize; // 1.5 MiB
    let mut buf = vec![0u8; chunk_records * 24];
    let mut remaining = count_usize;

    while remaining > 0 {
        let batch = remaining.min(chunk_records);
        let byte_len = batch * 24;
        reader
            .read_exact(&mut buf[..byte_len])
            .map_err(|e| format!("truncated queries payload: {e}"))?;
        for c in buf[..byte_len].as_chunks::<24>().0 {
            let start = u64::from_le_bytes(c[0..8].try_into().expect("8B start"));
            let end = u64::from_le_bytes(c[8..16].try_into().expect("8B end"));
            let expected = u64::from_le_bytes(c[16..24].try_into().expect("8B expected"));
            let range = QueryRange::new(start, end)
                .map_err(|e| format!("invalid query range [{start}, {end}]: {e}"))?;
            queries.push(QueryRecord { range, expected });
        }
        remaining -= batch;
    }

    Ok((queries, max_range_r))
}

/// Exact `O(log N)` closed-range emptiness check against a sorted key slice.
#[inline]
fn is_range_empty(sorted_keys: &[Key], start: u64, end: u64) -> bool {
    let idx = sorted_keys.partition_point(|&k| k < start);
    match sorted_keys.get(idx) {
        Some(&k) => k > end,
        None => true,
    }
}

/// Evaluates `query_point` for singleton ranges and `query_range` otherwise.
#[inline]
fn probe_filter(filter: &AerisFilter, range: QueryRange) -> bool {
    if range.start() == range.end() {
        filter.query_point(range.start())
    } else {
        filter.query_range(range)
    }
}

/// Verifies 100% of inserted keys return `true` under `query_point` (Zero FN invariant).
fn verify_all_keys_zero_fn(filter: &AerisFilter, keys: &[Key]) -> u64 {
    let mut missed = 0u64;
    for &k in keys {
        if !filter.query_point(k) {
            missed += 1;
        }
    }
    missed
}

/// Computes `(p50_ns, p99_ns)` from an unsorted vector of nanosecond latency samples.
fn compute_p50_p99(mut samples: Vec<u64>) -> (u64, u64) {
    if samples.is_empty() {
        return (0, 0);
    }
    samples.sort_unstable();
    let n = samples.len();
    let p50_idx = ((n - 1) * 500) / 1000;
    let p99_idx = ((n - 1) * 990) / 1000;
    (samples[p50_idx], samples[p99_idx])
}

/// Resolves the validated [`FilterConfig`] from CLI parameters.
///
/// Passes `max_range_r = 1u64 << m` so `FilterConfig::ceil_log2_u64(max_range_r) == m`
/// holds identically for point (`m = 0` or `m = 2`) and range (`m = 4..=10`) configs.
/// When `--target-epsilon` is omitted and `r < 7` (e.g., low-bit static paper curves),
/// relaxes `target_epsilon` to the tightest valid value satisfying `r >= r_min(eps)`.
fn build_filter_config(args: &CliArgs) -> Result<FilterConfig, String> {
    if args.m > 63 {
        return Err(format!("--m ({}) must be <= 63", args.m));
    }
    let max_range_r = 1u64
        .checked_shl(u32::from(args.m))
        .ok_or_else(|| format!("--m ({}) overflows 1u64 << m", args.m))?;
    let budget_c = args.budget_c.unwrap_or_else(|| {
        args.r
            .checked_add(args.m)
            .expect("r + m fits in u8 payload budget")
    });

    let target_epsilon = match args.target_epsilon {
        Some(eps) => eps,
        None => {
            let default_r_min = FilterConfig::min_remainder_bits(DEFAULT_TARGET_EPSILON)
                .map_err(|e| e.to_string())?;
            if args.r >= default_r_min {
                DEFAULT_TARGET_EPSILON
            } else if args.r >= 6 {
                0.125
            } else if args.r >= 4 {
                0.25
            } else {
                0.45
            }
        }
    };

    FilterConfig::new(args.q, args.r, max_range_r, budget_c, target_epsilon)
        .and_then(|c| c.with_max_load_factor(0.95))
        .map(|c| c.with_hash_seed(args.seed))
        .map_err(|e| format!("invalid FilterConfig: {e}"))
}

/// Queries the current process Peak Resident Set Size in MiB without `unsafe` FFI.
/// Invoked strictly AFTER all benchmark timers have stopped.
fn query_peak_rss_mb() -> f64 {
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
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
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
fn detect_compiler_version() -> String {
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

/// Runs an uninstrumented read-only query pass followed by a strided latency pass
/// (`<= 100,000` samples) over `queries`.
fn run_readonly_query_pass(
    filter: &AerisFilter,
    queries: &[QueryRecord],
) -> (f64, u64, u64, f64, u64, u64) {
    if queries.is_empty() {
        return (0.0, 0, 0, 0.0, 0, 0);
    }
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
    let fpr = if empty_count > 0 {
        (fp_count as f64) / (empty_count as f64)
    } else {
        0.0
    };

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
    let (p50_ns, p99_ns) = compute_p50_p99(lat_samples);
    (qps, fp_count, fn_count, fpr, p50_ns, p99_ns)
}

/// Executes `query_adapt` mode.
fn run_query_adapt(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let rm = InMemReverseMap::with_capacity(keys.len());
    let t_load = Instant::now();
    let mut filter = AerisFilter::bulk_load(cfg, keys, &rm)
        .map_err(|e| format!("bulk_load failed in query_adapt: {e}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let bulk_load_keys_per_sec = (keys.len() as f64) / load_s.max(1e-12);

    let stride = (queries.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut lat_samples = Vec::with_capacity(queries.len().min(MAX_LATENCY_SAMPLES) + 1);
    let mut seen_fps: HashSet<(u64, u64)> = HashSet::new();
    let mut fp_count = 0u64;
    let mut fn_count = 0u64;
    let mut empty_count = 0u64;
    let mut repeated_fp_count = 0u64;
    let mut adaptations = 0u64;
    let mut adapt_skips_full = 0u64;

    let t_q = Instant::now();
    for (idx, qr) in queries.iter().enumerate() {
        let sample = (idx % stride) == 0;
        let t0 = if sample { Some(Instant::now()) } else { None };

        let hit = probe_filter(&filter, qr.range);
        if qr.expected == 0 {
            empty_count += 1;
            if hit {
                fp_count += 1;
                if !seen_fps.insert((qr.range.start(), qr.range.end())) {
                    repeated_fp_count += 1;
                }
                match filter.adapt(qr.range, &rm) {
                    Ok(_) => adaptations += 1,
                    Err(AerisError::FilterFull { .. }) => adapt_skips_full += 1,
                    Err(err) => return Err(format!("adapt failed at query {idx}: {err}")),
                }
            }
        } else if !hit {
            fn_count += 1;
        }

        if let Some(start_ts) = t0 {
            lat_samples.push(start_ts.elapsed().as_nanos() as u64);
        }
    }
    let query_s = t_q.elapsed().as_secs_f64();
    let qps = (queries.len() as f64) / query_s.max(1e-12);
    let (p50_ns, p99_ns) = compute_p50_p99(lat_samples);

    let key_fn = verify_all_keys_zero_fn(&filter, keys);
    fn_count += key_fn;

    let filter_bytes = filter.memory_usage_bytes();
    let bits_per_key = if keys.is_empty() {
        0.0
    } else {
        (filter_bytes as f64) * 8.0 / (keys.len() as f64)
    };
    let fpr = if empty_count > 0 {
        (fp_count as f64) / (empty_count as f64)
    } else {
        0.0
    };
    let notes = format!(
        "mode=query_adapt;adapt_skips_full={adapt_skips_full};unique_fps={};verified_keys={}",
        seen_fps.len(),
        keys.len()
    );

    Ok(assemble_result(
        args,
        filter.config(),
        filter.load_factor(),
        keys.len(),
        queries.len(),
        qps,
        bulk_load_keys_per_sec,
        0.0,
        p50_ns,
        p99_ns,
        fpr,
        fp_count,
        fn_count,
        repeated_fp_count,
        adaptations,
        bits_per_key,
        filter_bytes,
        notes,
    ))
}

/// Executes `bulk_load` mode (pure filter with [`NoopReverseMap`]).
fn run_bulk_load(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let noop_rm = NoopReverseMap;
    let t_load = Instant::now();
    let filter = AerisFilter::bulk_load(cfg, keys, &noop_rm)
        .map_err(|e| format!("bulk_load failed: {e}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let bulk_load_keys_per_sec = (keys.len() as f64) / load_s.max(1e-12);

    let (qps, fp_count, mut fn_count, fpr, p50_ns, p99_ns) =
        run_readonly_query_pass(&filter, queries);

    let key_fn = verify_all_keys_zero_fn(&filter, keys);
    fn_count += key_fn;

    let filter_bytes = filter.memory_usage_bytes();
    let bits_per_key = if keys.is_empty() {
        0.0
    } else {
        (filter_bytes as f64) * 8.0 / (keys.len() as f64)
    };
    let notes = format!(
        "mode=bulk_load;reverse_map=noop;load_ms={:.3};verified_keys={}",
        load_s * 1000.0,
        keys.len()
    );

    Ok(assemble_result(
        args,
        filter.config(),
        filter.load_factor(),
        keys.len(),
        queries.len(),
        qps,
        bulk_load_keys_per_sec,
        0.0,
        p50_ns,
        p99_ns,
        fpr,
        fp_count,
        fn_count,
        0,
        0,
        bits_per_key,
        filter_bytes,
        notes,
    ))
}

/// Executes `dynamic_insert` mode (incremental per-key insertion + optional query pass).
fn run_dynamic_insert(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let mut filter = AerisFilter::new(cfg)
        .map_err(|e| format!("AerisFilter::new failed in dynamic_insert: {e}"))?;

    let stride = (keys.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut insert_samples = Vec::with_capacity(keys.len().min(MAX_LATENCY_SAMPLES) + 1);

    let t_ins = Instant::now();
    for (idx, &k) in keys.iter().enumerate() {
        let sample = (idx % stride) == 0;
        let t0 = if sample { Some(Instant::now()) } else { None };
        filter.insert(k).map_err(|e| {
            format!(
                "insert failed at key index {idx} (alpha={:.4}): {e}",
                filter.load_factor()
            )
        })?;
        if let Some(start_ts) = t0 {
            insert_samples.push(start_ts.elapsed().as_nanos() as u64);
        }
    }
    let ins_s = t_ins.elapsed().as_secs_f64();
    let insert_keys_per_sec = (keys.len() as f64) / ins_s.max(1e-12);
    let (ins_p50_ns, ins_p99_ns) = compute_p50_p99(insert_samples);

    let (qps, fp_count, mut fn_count, fpr, q_p50_ns, q_p99_ns) =
        run_readonly_query_pass(&filter, queries);

    let key_fn = verify_all_keys_zero_fn(&filter, keys);
    fn_count += key_fn;

    let filter_bytes = filter.memory_usage_bytes();
    let bits_per_key = if keys.is_empty() {
        0.0
    } else {
        (filter_bytes as f64) * 8.0 / (keys.len() as f64)
    };
    let notes = format!(
        "mode=dynamic_insert;insert_ms={:.3};query_p50_ns={q_p50_ns};query_p99_ns={q_p99_ns};verified_keys={}",
        ins_s * 1000.0,
        keys.len()
    );

    Ok(assemble_result(
        args,
        filter.config(),
        filter.load_factor(),
        keys.len(),
        queries.len(),
        qps,
        0.0,
        insert_keys_per_sec,
        ins_p50_ns,
        ins_p99_ns,
        fpr,
        fp_count,
        fn_count,
        0,
        0,
        bits_per_key,
        filter_bytes,
        notes,
    ))
}

/// Executes `adversarial` mode with deterministic 10% repeat-FP probing (`fp_pool` cap 256).
fn run_adversarial(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let rm = InMemReverseMap::with_capacity(keys.len());
    let t_load = Instant::now();
    let mut filter = AerisFilter::bulk_load(cfg, keys, &rm)
        .map_err(|e| format!("bulk_load failed in adversarial: {e}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let bulk_load_keys_per_sec = (keys.len() as f64) / load_s.max(1e-12);

    let stride = (queries.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut lat_samples = Vec::with_capacity(queries.len().min(MAX_LATENCY_SAMPLES) + 1);
    let mut fp_pool: Vec<QueryRecord> = Vec::with_capacity(ADVERSARIAL_FP_POOL_CAP);
    let mut seen_fps: HashSet<(u64, u64)> = HashSet::new();

    let mut fp_count = 0u64;
    let mut fn_count = 0u64;
    let mut empty_count = 0u64;
    let mut repeated_fp_count = 0u64;
    let mut adaptations = 0u64;
    let mut adapt_skips_full = 0u64;
    let mut adversarial_reprobes = 0u64;

    let t_q = Instant::now();
    for i in 0..queries.len() {
        let qr = if !fp_pool.is_empty() && (i % 10 == 0) {
            adversarial_reprobes += 1;
            fp_pool[(i / 10) % fp_pool.len()]
        } else {
            queries[i]
        };

        let sample = (i % stride) == 0;
        let t0 = if sample { Some(Instant::now()) } else { None };

        let hit = probe_filter(&filter, qr.range);
        if qr.expected == 0 {
            empty_count += 1;
            if hit {
                fp_count += 1;
                let is_new = seen_fps.insert((qr.range.start(), qr.range.end()));
                if !is_new {
                    repeated_fp_count += 1;
                } else if fp_pool.len() < ADVERSARIAL_FP_POOL_CAP {
                    fp_pool.push(qr);
                }
                match filter.adapt(qr.range, &rm) {
                    Ok(_) => adaptations += 1,
                    Err(AerisError::FilterFull { .. }) => adapt_skips_full += 1,
                    Err(err) => return Err(format!("adapt failed at adversarial step {i}: {err}")),
                }
            }
        } else if !hit {
            fn_count += 1;
        }

        if let Some(start_ts) = t0 {
            lat_samples.push(start_ts.elapsed().as_nanos() as u64);
        }
    }
    let query_s = t_q.elapsed().as_secs_f64();
    let qps = (queries.len() as f64) / query_s.max(1e-12);
    let (p50_ns, p99_ns) = compute_p50_p99(lat_samples);

    let key_fn = verify_all_keys_zero_fn(&filter, keys);
    fn_count += key_fn;

    let filter_bytes = filter.memory_usage_bytes();
    let bits_per_key = if keys.is_empty() {
        0.0
    } else {
        (filter_bytes as f64) * 8.0 / (keys.len() as f64)
    };
    let fpr = if empty_count > 0 {
        (fp_count as f64) / (empty_count as f64)
    } else {
        0.0
    };
    let notes = format!(
        "mode=adversarial;fp_pool_size={};adversarial_reprobes={adversarial_reprobes};adapt_skips_full={adapt_skips_full};verified_keys={}",
        fp_pool.len(),
        keys.len()
    );

    Ok(assemble_result(
        args,
        filter.config(),
        filter.load_factor(),
        keys.len(),
        queries.len(),
        qps,
        bulk_load_keys_per_sec,
        0.0,
        p50_ns,
        p99_ns,
        fpr,
        fp_count,
        fn_count,
        repeated_fp_count,
        adaptations,
        bits_per_key,
        filter_bytes,
        notes,
    ))
}

/// Identifies contiguous `(start_idx, end_idx, range_len)` phases in `queries`.
fn detect_query_phases(queries: &[QueryRecord]) -> Vec<(usize, usize, u64)> {
    if queries.is_empty() {
        return Vec::new();
    }
    let mut phases = Vec::new();
    let mut start = 0usize;
    let mut cur_len = queries[0].range.len_saturated();
    for (idx, qr) in queries.iter().enumerate().skip(1) {
        let len = qr.range.len_saturated();
        if len != cur_len {
            phases.push((start, idx, cur_len));
            start = idx;
            cur_len = len;
        }
    }
    phases.push((start, queries.len(), cur_len));
    phases
}

/// Executes `shifting_r` mode for either `rust_naive` (static `m`) or `rust_optimized`
/// (Control Plane `SlidingWindowEstimator` + `DiscreteCostOptimizer` + `FalsePositiveLog` replay).
fn run_shifting_r(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let mut rm = InMemReverseMap::with_capacity(keys.len());
    let t_load = Instant::now();
    let mut filter = AerisFilter::bulk_load(cfg, keys, &rm)
        .map_err(|e| format!("bulk_load failed in shifting_r: {e}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let bulk_load_keys_per_sec = (keys.len() as f64) / load_s.max(1e-12);

    let is_optimized = args.impl_variant == ImplVariant::RustOptimized && !args.static_bypass;
    let phases = detect_query_phases(queries);

    // Pre-build sorted ground-truth oracle and real persistent FalsePositiveLog for `rust_optimized`.
    let sorted_keys: Vec<Key> = if is_optimized {
        let mut s = keys.to_vec();
        s.sort_unstable();
        s
    } else {
        Vec::new()
    };

    let tmp_fp_path = env::temp_dir().join(format!(
        "aeris_paper_parity_fplog_{}_{}.fplog",
        std::process::id(),
        args.seed
    ));
    let _fp_log_guard = TempFileGuard::new(tmp_fp_path.clone());
    let fp_log = if is_optimized {
        let _ = fs::remove_file(&tmp_fp_path);
        Some(
            FalsePositiveLog::open(
                &tmp_fp_path,
                FpLogConfig {
                    max_entries: 262_144,
                    sync_on_append: false,
                },
            )
            .map_err(|e| format!("failed to open FalsePositiveLog: {e}"))?,
        )
    } else {
        None
    };

    let mut estimator = if args.impl_variant == ImplVariant::RustOptimized {
        let est_cfg = if args.static_bypass {
            EstimatorConfig::default().with_enabled(false)
        } else {
            EstimatorConfig::default()
        };
        Some(
            SlidingWindowEstimator::with_initial_span(est_cfg, filter.config().max_range_r)
                .map_err(|e| format!("failed to initialize SlidingWindowEstimator: {e}"))?,
        )
    } else {
        None
    };
    let optimizer = DiscreteCostOptimizer::new();
    let cost_model = CostModelParams::default();

    let stride = (queries.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut all_lat_samples = Vec::with_capacity(queries.len().min(MAX_LATENCY_SAMPLES) + 1);
    let mut seen_fps: HashSet<(u64, u64)> = HashSet::new();

    let mut fp_count = 0u64;
    let mut fn_count = 0u64;
    let mut empty_count = 0u64;
    let mut repeated_fp_count = 0u64;
    let mut adaptations = 0u64;
    let mut adapt_skips_full = 0u64;

    let mut rebuild_count = 0u64;
    let mut total_rebuild_s = 0.0f64;
    let mut total_replay_us = 0u64;
    let mut total_replayed_entries = 0usize;
    let mut total_replay_resolved = 0usize;
    let mut m_trajectory = vec![filter.config().m];
    let mut phase_metrics = Vec::with_capacity(phases.len());

    let mut global_idx = 0usize;
    let t_total = Instant::now();

    for (phase_idx, &(p_start, p_end, phase_r)) in phases.iter().enumerate() {
        let phase_slice = &queries[p_start..p_end];
        let phase_stride =
            (phase_slice.len() / (MAX_LATENCY_SAMPLES / phases.len().max(1)).max(1)).max(1);
        let mut phase_lat = Vec::with_capacity(phase_slice.len().min(MAX_LATENCY_SAMPLES) + 1);
        let mut phase_fps = 0u64;
        let mut phase_rebuild_s = 0.0f64;
        let t_phase = Instant::now();

        for (local_idx, qr) in phase_slice.iter().enumerate() {
            let sample_global = global_idx.is_multiple_of(stride);
            let sample_phase = local_idx.is_multiple_of(phase_stride);
            let t0 = if sample_global || sample_phase {
                Some(Instant::now())
            } else {
                None
            };

            let hit = probe_filter(&filter, qr.range);
            let mut is_fp = false;

            if qr.expected == 0 {
                empty_count += 1;
                if hit {
                    is_fp = true;
                    fp_count += 1;
                    phase_fps += 1;
                    if !seen_fps.insert((qr.range.start(), qr.range.end())) {
                        repeated_fp_count += 1;
                    }
                    if let Some(ref log) = fp_log {
                        log.record(qr.range, global_idx as u64)
                            .map_err(|e| format!("FalsePositiveLog::record failed: {e}"))?;
                    }
                    match filter.adapt(qr.range, &rm) {
                        Ok(_) => adaptations += 1,
                        Err(AerisError::FilterFull { .. }) => adapt_skips_full += 1,
                        Err(err) => {
                            return Err(format!("adapt failed at query {global_idx}: {err}"))
                        }
                    }
                }
            } else if !hit {
                fn_count += 1;
            }

            if let Some(ref mut est) = estimator {
                est.record_query(qr.range, is_fp);
            }

            if let Some(start_ts) = t0 {
                let dt = start_ts.elapsed().as_nanos() as u64;
                if sample_global {
                    all_lat_samples.push(dt);
                }
                if sample_phase {
                    phase_lat.push(dt);
                }
            }

            global_idx += 1;

            if is_optimized && global_idx.is_multiple_of(OPTIMIZER_EVAL_INTERVAL) {
                let stats = estimator.as_ref().expect("estimator present").snapshot();
                let decision = optimizer.optimize(&stats, filter.config(), &cost_model);
                if decision.should_rebuild && decision.recommended_m != filter.config().m {
                    let t_reb = Instant::now();
                    let new_cfg = filter
                        .config()
                        .reconfigure_for_m(decision.recommended_m)
                        .map_err(|e| format!("reconfigure_for_m failed: {e}"))?;
                    let new_rm = InMemReverseMap::with_capacity(keys.len());
                    let mut new_filter = AerisFilter::bulk_load(new_cfg, keys, &new_rm)
                        .map_err(|e| format!("rebuild bulk_load failed: {e}"))?;

                    let log_ref = fp_log.as_ref().expect("fp_log present");
                    log_ref
                        .sync()
                        .map_err(|e| format!("fp_log sync failed: {e}"))?;
                    let t_rep = Instant::now();
                    let rep = log_ref
                        .replay_onto(&mut new_filter, &new_rm, |r| {
                            Ok(is_range_empty(&sorted_keys, r.start(), r.end()))
                        })
                        .map_err(|e| format!("FalsePositiveLog::replay_onto failed: {e}"))?;
                    let replay_us = t_rep.elapsed().as_micros() as u64;
                    let reb_s = t_reb.elapsed().as_secs_f64();

                    info!(
                        old_m = filter.config().m,
                        new_m = new_filter.config().m,
                        at_query = global_idx,
                        phase_r,
                        rebuild_ms = reb_s * 1000.0,
                        replay_us,
                        replayed = rep.replayed_entries,
                        resolved = rep.collisions_resolved,
                        "completed dynamic-R rebuild and monotonic FP replay"
                    );

                    filter = new_filter;
                    rm = new_rm;
                    rebuild_count += 1;
                    total_rebuild_s += reb_s;
                    phase_rebuild_s += reb_s;
                    total_replay_us += replay_us;
                    total_replayed_entries += rep.replayed_entries;
                    total_replay_resolved += rep.collisions_resolved;
                    m_trajectory.push(filter.config().m);
                }
            }
        }

        let phase_wall_s = t_phase.elapsed().as_secs_f64();
        let phase_pure_s = (phase_wall_s - phase_rebuild_s).max(1e-12);
        let (_, phase_p99) = compute_p50_p99(phase_lat);
        phase_metrics.push(PhaseMetrics {
            phase_idx,
            range_len: phase_r,
            queries: phase_slice.len(),
            fp_count: phase_fps,
            pure_query_s: phase_pure_s,
            wall_s: phase_wall_s,
            p99_ns: phase_p99,
            end_m: filter.config().m,
        });
    }

    let total_wall_s = t_total.elapsed().as_secs_f64();
    let pure_query_s = (total_wall_s - total_rebuild_s).max(1e-12);
    let wall_qps = (queries.len() as f64) / total_wall_s.max(1e-12);
    let pure_query_qps = (queries.len() as f64) / pure_query_s;
    let (p50_ns, p99_ns) = compute_p50_p99(all_lat_samples);

    let key_fn = verify_all_keys_zero_fn(&filter, keys);
    fn_count += key_fn;

    let filter_bytes = filter.memory_usage_bytes();
    let bits_per_key = if keys.is_empty() {
        0.0
    } else {
        (filter_bytes as f64) * 8.0 / (keys.len() as f64)
    };
    let fpr = if empty_count > 0 {
        (fp_count as f64) / (empty_count as f64)
    } else {
        0.0
    };

    let m_traj_str = m_trajectory
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join("->");
    let phase_str = phase_metrics
        .iter()
        .map(|pm| {
            let p_fpr = if pm.queries > 0 {
                (pm.fp_count as f64) / (pm.queries as f64)
            } else {
                0.0
            };
            let p_pure_qps = (pm.queries as f64) / pm.pure_query_s.max(1e-12);
            let p_wall_qps = (pm.queries as f64) / pm.wall_s.max(1e-12);
            format!(
                "P{}(R={},m={},pure_qps={:.0},wall_qps={:.0},fpr={:.6},p99_ns={})",
                pm.phase_idx, pm.range_len, pm.end_m, p_pure_qps, p_wall_qps, p_fpr, pm.p99_ns
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let notes = format!(
        "mode=shifting_r;pure_query_qps={pure_query_qps:.2};wall_qps={wall_qps:.2};rebuilds={rebuild_count};total_rebuild_ms={:.3};total_replay_us={total_replay_us};replayed_entries={total_replayed_entries};replay_resolved={total_replay_resolved};m_trajectory=[{m_traj_str}];adapt_skips_full={adapt_skips_full};phases=[{phase_str}]",
        total_rebuild_s * 1000.0
    );

    Ok(assemble_result(
        args,
        filter.config(),
        filter.load_factor(),
        keys.len(),
        queries.len(),
        wall_qps,
        bulk_load_keys_per_sec,
        0.0,
        p50_ns,
        p99_ns,
        fpr,
        fp_count,
        fn_count,
        repeated_fp_count,
        adaptations,
        bits_per_key,
        filter_bytes,
        notes,
    ))
}

/// Assembles the canonical 24-field [`ParityResult`] after querying Peak RSS and
/// compiler metadata outside all benchmark timers.
#[allow(clippy::too_many_arguments)]
fn assemble_result(
    args: &CliArgs,
    cfg: &FilterConfig,
    load_factor: f64,
    keys: usize,
    queries: usize,
    qps: f64,
    bulk_load_keys_per_sec: f64,
    insert_keys_per_sec: f64,
    p50_ns: u64,
    p99_ns: u64,
    fpr: f64,
    fp_count: u64,
    fn_count: u64,
    repeated_fp_count: u64,
    adaptations: u64,
    bits_per_key: f64,
    filter_bytes: usize,
    notes: String,
) -> ParityResult {
    let peak_rss_mb = query_peak_rss_mb();
    let compiler = detect_compiler_version();
    let flags = "-C opt-level=3 -C lto=fat -C codegen-units=1 -C target-cpu=native".to_string();

    ParityResult {
        impl_name: args.impl_variant.as_str().to_string(),
        scenario: args.scenario.clone(),
        keys,
        queries,
        q_bits: cfg.q,
        r_bits: cfg.r,
        m_bits: cfg.m,
        load_factor,
        qps,
        bulk_load_keys_per_sec,
        insert_keys_per_sec,
        p50_ns,
        p99_ns,
        fpr,
        fp_count,
        fn_count,
        repeated_fp_count,
        adaptations,
        bits_per_key,
        filter_bytes,
        peak_rss_mb,
        compiler,
        flags,
        notes,
    }
}

/// Parses CLI arguments matching the shared C++/Rust parity interface.
fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    if args.len() < 2 {
        return Err(usage_string());
    }

    let mut mode: Option<BenchMode> = None;
    let mut impl_variant: Option<ImplVariant> = None;
    let mut scenario: Option<String> = None;
    let mut keys_path: Option<PathBuf> = None;
    let mut queries_path: Option<PathBuf> = None;
    let mut q: Option<u8> = None;
    let mut r: Option<u8> = None;
    let mut m: Option<u8> = None;
    let mut budget_c: Option<u8> = None;
    let mut target_epsilon: Option<f64> = None;
    let mut seed: u64 = DEFAULT_SEED;
    let mut static_bypass = false;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--static-bypass" {
            static_bypass = true;
            i += 1;
            continue;
        }
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--mode" => mode = Some(BenchMode::parse(val)?),
            "--impl" => impl_variant = Some(ImplVariant::parse(val)?),
            "--scenario" => scenario = Some(val.clone()),
            "--keys" => keys_path = Some(PathBuf::from(val)),
            "--queries" => queries_path = Some(PathBuf::from(val)),
            "--q" => {
                q = Some(
                    val.parse::<u8>()
                        .map_err(|e| format!("invalid --q '{val}': {e}"))?,
                );
            }
            "--r" => {
                r = Some(
                    val.parse::<u8>()
                        .map_err(|e| format!("invalid --r '{val}': {e}"))?,
                );
            }
            "--m" => {
                m = Some(
                    val.parse::<u8>()
                        .map_err(|e| format!("invalid --m '{val}': {e}"))?,
                );
            }
            "--budget-c" => {
                budget_c = Some(
                    val.parse::<u8>()
                        .map_err(|e| format!("invalid --budget-c '{val}': {e}"))?,
                );
            }
            "--target-epsilon" => {
                target_epsilon = Some(
                    val.parse::<f64>()
                        .map_err(|e| format!("invalid --target-epsilon '{val}': {e}"))?,
                );
            }
            "--seed" => {
                seed = val
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --seed '{val}': {e}"))?;
            }
            "--out" => out_path = Some(PathBuf::from(val)),
            other => return Err(format!("unknown flag '{other}'\n{}", usage_string())),
        }
        i += 2;
    }

    let mode = mode.ok_or_else(|| "missing required --mode".to_string())?;
    let impl_variant = impl_variant.ok_or_else(|| "missing required --impl".to_string())?;
    let scenario = scenario.ok_or_else(|| "missing required --scenario".to_string())?;
    let keys_path = keys_path.ok_or_else(|| "missing required --keys".to_string())?;
    let q = q.ok_or_else(|| "missing required --q".to_string())?;
    let r = r.ok_or_else(|| "missing required --r".to_string())?;
    let m = m.ok_or_else(|| "missing required --m".to_string())?;

    if mode.requires_queries() && queries_path.is_none() {
        return Err(format!("mode {mode:?} requires --queries <path>"));
    }

    Ok(CliArgs {
        mode,
        impl_variant,
        scenario,
        keys_path,
        queries_path,
        q,
        r,
        m,
        budget_c,
        target_epsilon,
        seed,
        static_bypass,
        out_path,
    })
}

fn usage_string() -> String {
    concat!(
        "Usage: paper_parity --mode <query_adapt|bulk_load|dynamic_insert|adversarial|shifting_r> \\\n",
        "  --impl <rust_naive|rust_optimized> --scenario <name> --keys <path> [--queries <path>] \\\n",
        "  --q <u8> --r <u8> --m <u8> [--budget-c <u8>] [--target-epsilon <f64>] [--seed <u64>] [--out <path>]"
    )
    .to_string()
}

fn init_stderr_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(io::stderr)
        .with_target(false)
        .try_init();
}

fn main() -> ExitCode {
    init_stderr_tracing();
    let raw_args: Vec<String> = env::args().collect();
    let args = match parse_args(&raw_args) {
        Ok(a) => a,
        Err(err) => {
            error!(error = %err, "invalid paper_parity CLI arguments");
            return ExitCode::FAILURE;
        }
    };

    let cfg = match build_filter_config(&args) {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "configuration validation failed");
            return ExitCode::FAILURE;
        }
    };

    info!(
        mode = ?args.mode,
        impl_variant = args.impl_variant.as_str(),
        scenario = %args.scenario,
        keys_file = %args.keys_path.display(),
        q = cfg.q,
        r = cfg.r,
        m = cfg.m,
        budget_c = cfg.payload_budget_c,
        target_epsilon = cfg.target_epsilon,
        "starting paper_parity benchmark"
    );

    let keys = match read_keys_file(&args.keys_path) {
        Ok(k) => k,
        Err(err) => {
            error!(error = %err, "failed to load binary keys trace");
            return ExitCode::FAILURE;
        }
    };

    let queries = match &args.queries_path {
        Some(q_path) => match read_queries_file(q_path) {
            Ok((qs, _max_r)) => qs,
            Err(err) => {
                error!(error = %err, "failed to load binary queries trace");
                return ExitCode::FAILURE;
            }
        },
        None => Vec::new(),
    };

    let result = match args.mode {
        BenchMode::QueryAdapt => run_query_adapt(&args, cfg, &keys, &queries),
        BenchMode::BulkLoad => run_bulk_load(&args, cfg, &keys, &queries),
        BenchMode::DynamicInsert => run_dynamic_insert(&args, cfg, &keys, &queries),
        BenchMode::Adversarial => run_adversarial(&args, cfg, &keys, &queries),
        BenchMode::ShiftingR => run_shifting_r(&args, cfg, &keys, &queries),
    };

    let res = match result {
        Ok(r) => r,
        Err(err) => {
            error!(error = %err, "paper_parity execution failed");
            return ExitCode::FAILURE;
        }
    };

    let json_line = match serde_json::to_string(&res) {
        Ok(s) => s,
        Err(err) => {
            error!(error = %err, "failed to serialize ParityResult JSON");
            return ExitCode::FAILURE;
        }
    };

    if let Some(ref out_path) = args.out_path {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    error!(error = %err, path = %parent.display(), "failed to create --out directory");
                    return ExitCode::FAILURE;
                }
            }
        }
        let pretty = serde_json::to_string_pretty(&res).unwrap_or_else(|_| json_line.clone());
        if let Err(err) = fs::write(out_path, pretty) {
            error!(error = %err, path = %out_path.display(), "failed to write --out JSON file");
            return ExitCode::FAILURE;
        }
    }

    {
        let mut stdout = io::stdout().lock();
        if writeln!(stdout, "{json_line}").is_err() {
            return ExitCode::FAILURE;
        }
    }

    if res.fn_count > 0 {
        error!(
            fn_count = res.fn_count,
            scenario = %res.scenario,
            "ZERO FALSE NEGATIVE INVARIANT VIOLATED"
        );
        return ExitCode::FAILURE;
    }

    if args.mode == BenchMode::Adversarial && res.repeated_fp_count > 0 {
        warn!(
            repeated_fp_count = res.repeated_fp_count,
            scenario = %res.scenario,
            "repeated false positives observed in adversarial mode"
        );
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeris_types::PartitionHasher;

    #[test]
    fn in_mem_reverse_map_satisfies_store_contract() {
        let rm = InMemReverseMap::with_capacity(16);
        let hasher = PartitionHasher::new(42);
        let fp1 = Fingerprint::from_hasher(&hasher, 10, 20);
        let fp2 = Fingerprint::from_hasher(&hasher, 20, 24);

        rm.append_batch(&[(fp1.clone(), 100), (fp1.clone(), 101)])
            .unwrap();
        assert_eq!(rm.entry_count_estimate(), 2);
        assert_eq!(rm.get_keys(&fp1).unwrap(), vec![100, 101]);

        rm.remove_key(&fp1, 100).unwrap();
        assert_eq!(rm.get_keys(&fp1).unwrap(), vec![101]);
        assert_eq!(rm.entry_count_estimate(), 1);

        rm.replace_fingerprint(&fp1, &[(fp2.clone(), 101), (fp2.clone(), 202)])
            .unwrap();
        assert!(rm.get_keys(&fp1).unwrap().is_empty());
        assert_eq!(rm.get_keys(&fp2).unwrap(), vec![101, 202]);
        assert_eq!(rm.scan_all_keys().unwrap(), vec![101, 202]);

        rm.tombstone(&fp2).unwrap();
        assert!(rm.get_keys(&fp2).unwrap().is_empty());
        assert_eq!(rm.entry_count_estimate(), 0);
    }

    #[test]
    fn compute_p50_p99_handles_empty_and_sorted_percentiles() {
        assert_eq!(compute_p50_p99(vec![]), (0, 0));
        let samples: Vec<u64> = (1..=100).rev().collect();
        let (p50, p99) = compute_p50_p99(samples);
        assert_eq!(p50, 50);
        assert_eq!(p99, 99);
    }

    #[test]
    fn shifting_r_optimized_reconfigures_and_preserves_zero_fn_and_zero_repeated_fp() {
        let keys: Vec<Key> = (0..4_000u64).map(|i| i * 25_000 + 7).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();

        let mut queries = Vec::new();
        for &phase_r in &[16u64, 256, 32, 1024] {
            for j in 0..3_000u64 {
                let start = j * 25_000 + 2_000;
                let end = start + phase_r - 1;
                if is_range_empty(&sorted, start, end) {
                    queries.push(QueryRecord {
                        range: QueryRange::new(start, end).unwrap(),
                        expected: 0,
                    });
                }
            }
        }

        let args = CliArgs {
            mode: BenchMode::ShiftingR,
            impl_variant: ImplVariant::RustOptimized,
            scenario: "unit_shifting_r".to_string(),
            keys_path: PathBuf::from("unit.keys.bin"),
            queries_path: Some(PathBuf::from("unit.queries.bin")),
            q: 13,
            r: 13,
            m: 4,
            budget_c: Some(17),
            target_epsilon: Some(0.0625),
            seed: 1380,
            static_bypass: false,
            out_path: None,
        };
        let cfg = build_filter_config(&args).unwrap();
        let res = run_shifting_r(&args, cfg, &keys, &queries).unwrap();

        assert_eq!(res.fn_count, 0, "expected zero false negatives");
        assert_eq!(res.repeated_fp_count, 0, "expected zero repeated FPs");
        assert!(
            res.notes.contains("rebuilds=") && !res.notes.contains("rebuilds=0"),
            "expected dynamic-R rebuilds in notes: {}",
            res.notes
        );
    }
}
