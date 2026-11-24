//! Shared deterministic workload generators, scale-aware ground-truth oracles,
//! host hardware/RSS telemetry probes, latency percentile calculators, and
//! JSON-serializable [`AblationMetric`] contracts for the Aeris Correctness
//! Validation & Large-Scale Ablation Suite.
//!
//! # Determinism & Performance
//! - Every workload generator is seeded exclusively via [`ChaCha8Rng`], ensuring
//!   bit-identical keys, queries, and traces across runs.
//! - [`SortedKeyOracle`] stores keys in a sorted contiguous `Vec<Key>` so
//!   10,000,000-key datasets allocate ~80 MB (instead of ~320 MB in a `BTreeSet`)
//!   and answer exact closed range emptiness in `O(log N)` binary search.
//! - Zero `unsafe` code, zero `dbg!`, and zero `thread::sleep()`.

#![allow(dead_code)]

use aeris_types::{Key, QueryRange};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Default 48-bit keyspace upper bound (`[0, 2^48)`), preserving genuine 64-bit
/// arithmetic and full partition/memento coverage without wrap-around overflow.
pub const KEYSPACE_MASK: u64 = (1u64 << 48) - 1;

/// Default Zipfian skew parameter matching `Aeris Filters.pdf` Section 5.2.
pub const DEFAULT_ZIPF_ALPHA: f64 = 1.5;

/// Default Zipfian correlation degree `D` matching `Aeris Filters.pdf` Section 5.2.
pub const DEFAULT_CORRELATION_D: f64 = 0.8;

/// Default adversarial repeat-FP rate (`10%`).
pub const DEFAULT_ADVERSARIAL_REPEAT_RATE: f64 = 0.10;

/// Free-RAM threshold below which 10M/5M scale tiers automatically scale down by 2x.
pub const RAM_SAFETY_CEILING_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Canonical shifting-`R` phase sequence for A/B 1 (`16 -> 256 -> 32 -> 1024`).
pub const SHIFTING_R_PHASES: [u64; 4] = [16, 256, 32, 1024];

static TRACING_INIT: Once = Once::new();

/// Initializes structured `tracing` subscriber once per ablation test binary.
///
/// Respects `RUST_LOG` if present; otherwise defaults to `info` level so
/// structured ablation telemetry renders cleanly under `--nocapture`.
pub fn init_ablation_tracing() {
    TRACING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_test_writer()
            .try_init();
    });
}

/// Host hardware and memory safety-ceiling metadata recorded in every worker report
/// and merged into `target/ablation_results.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HostMetadata {
    /// Host CPU model string.
    pub cpu_model: String,
    /// Number of physical CPU cores reported by the OS.
    pub physical_cores: usize,
    /// Number of logical CPU threads reported by the OS.
    pub logical_cores: usize,
    /// L3 cache size in kilobytes (`0` if unavailable from OS).
    pub l3_cache_kb: u64,
    /// Total physical system RAM in bytes.
    pub total_ram_bytes: u64,
    /// Available (free) physical system RAM in bytes at suite start.
    pub available_ram_bytes: u64,
    /// Applied scale factor (`1.0` when free RAM >= 8 GiB, `0.5` otherwise).
    pub scale_factor: f64,
    /// Peak resident set size (working set) of this test process in bytes.
    pub peak_rss_bytes: u64,
}

impl HostMetadata {
    /// Probes live host hardware specifications, available RAM, and process Peak RSS,
    /// computing the automatic safety-ceiling `scale_factor` (`1.0` or `0.5`).
    #[must_use]
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);

        let mut meta = Self {
            cpu_model: "unknown".to_string(),
            physical_cores: logical_cores,
            logical_cores,
            l3_cache_kb: 0,
            total_ram_bytes: 0,
            available_ram_bytes: 0,
            scale_factor: 1.0,
            peak_rss_bytes: query_process_peak_rss_bytes(),
        };

        probe_os_hardware(&mut meta);

        if meta.available_ram_bytes > 0 && meta.available_ram_bytes < RAM_SAFETY_CEILING_BYTES {
            meta.scale_factor = 0.5;
            warn!(
                available_ram_bytes = meta.available_ram_bytes,
                ceiling_bytes = RAM_SAFETY_CEILING_BYTES,
                scale_factor = meta.scale_factor,
                "free RAM below 8 GiB safety ceiling; scaling 10M/5M tiers by 0.5x"
            );
        } else {
            meta.scale_factor = 1.0;
        }

        info!(
            cpu_model = %meta.cpu_model,
            physical_cores = meta.physical_cores,
            logical_cores = meta.logical_cores,
            l3_cache_kb = meta.l3_cache_kb,
            total_ram_gb = (meta.total_ram_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
            available_ram_gb = (meta.available_ram_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
            scale_factor = meta.scale_factor,
            "probed host hardware metadata"
        );

        meta
    }

    /// Scales a high-tier (`10M`/`5M`) target count by `self.scale_factor`.
    #[inline]
    #[must_use]
    pub fn scaled_count(&self, base_count: usize) -> usize {
        ((base_count as f64) * self.scale_factor).round().max(1.0) as usize
    }

    /// Refreshes `peak_rss_bytes` with the latest OS high-water mark for this process.
    pub fn refresh_peak_rss(&mut self) {
        let rss = query_process_peak_rss_bytes();
        if rss > self.peak_rss_bytes {
            self.peak_rss_bytes = rss;
        }
    }
}

/// Queries the current process Peak Resident Set Size (Windows `PeakWorkingSet64`
/// or Linux `/proc/self/status` `VmHWM`) in bytes without `unsafe` FFI.
#[must_use]
pub fn query_process_peak_rss_bytes() -> u64 {
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
                        return bytes;
                    }
                }
            }
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(kb_str) = rest.split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb.saturating_mul(1024);
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        0
    }
}

#[cfg(target_os = "windows")]
fn probe_os_hardware(meta: &mut HostMetadata) {
    let script = concat!(
        "$os = Get-CimInstance Win32_OperatingSystem; ",
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; ",
        "@{ cpu_model = $cpu.Name; ",
        "physical_cores = [int]$cpu.NumberOfCores; ",
        "logical_cores = [int]$cpu.NumberOfLogicalProcessors; ",
        "l3_cache_kb = [uint64]$cpu.L3CacheSize; ",
        "total_ram_bytes = [uint64]$os.TotalVisibleMemorySize * 1024; ",
        "available_ram_bytes = [uint64]$os.FreePhysicalMemory * 1024 } ",
        "| ConvertTo-Json -Compress"
    );

    #[derive(Deserialize)]
    struct WinHwProbe {
        cpu_model: Option<String>,
        physical_cores: Option<usize>,
        logical_cores: Option<usize>,
        l3_cache_kb: Option<u64>,
        total_ram_bytes: Option<u64>,
        available_ram_bytes: Option<u64>,
    }

    if let Ok(output) = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
    {
        if output.status.success() {
            if let Ok(parsed) = serde_json::from_slice::<WinHwProbe>(&output.stdout) {
                if let Some(model) = parsed.cpu_model {
                    meta.cpu_model = model.trim().to_string();
                }
                if let Some(c) = parsed.physical_cores.filter(|&v| v > 0) {
                    meta.physical_cores = c;
                }
                if let Some(l) = parsed.logical_cores.filter(|&v| v > 0) {
                    meta.logical_cores = l;
                }
                if let Some(l3) = parsed.l3_cache_kb {
                    meta.l3_cache_kb = l3;
                }
                if let Some(tot) = parsed.total_ram_bytes {
                    meta.total_ram_bytes = tot;
                }
                if let Some(avail) = parsed.available_ram_bytes {
                    meta.available_ram_bytes = avail;
                }
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn probe_os_hardware(meta: &mut HostMetadata) {
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                if let Some(kb) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    meta.total_ram_bytes = kb.saturating_mul(1024);
                }
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                if let Some(kb) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    meta.available_ram_bytes = kb.saturating_mul(1024);
                }
            }
        }
    }
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some((_, val)) = rest.split_once(':') {
                    meta.cpu_model = val.trim().to_string();
                    break;
                }
            }
        }
    }
}

/// Shared JSON-serializable ablation measurement record emitted by every worker
/// and aggregated into `target/ablation_results.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AblationMetric {
    /// Canonical extension or experiment identifier (e.g., `"DRAM_10M_FPR_eps_2e-8"`,
    /// `"AB1_DynamicR_ShiftingTrace"`, `"AB2_Proactive_EWMA_Contraction"`,
    /// `"AB3_WAL_Coupled_Storage"`, `"AB4_ScalePlane_RCU_Sharding"`, `"AB5_Sidecar_ML_RL"`).
    pub extension_name: String,
    /// Human-readable metric name shown in report tables (e.g., `"Empirical FPR"`, `"Write p99 Latency"`).
    pub metric_label: String,
    /// Unit string for `baseline_value` and `optimized_value` (e.g., `"ops/sec"`, `"ns"`, `"ratio"`, `"bytes"`).
    pub unit: String,
    /// Whether a larger value represents an improvement (`true` for throughput, `false` for latency/FPR/RAM).
    pub higher_is_better: bool,
    /// Number of keys indexed in this experiment.
    pub key_count: usize,
    /// Number of queries or operations executed in this experiment.
    pub query_count: usize,
    /// Measured Naive Paper-Baseline value.
    pub baseline_value: f64,
    /// Measured Aeris Optimized System value.
    pub optimized_value: f64,
    /// Relative percentage change `((optimized - baseline) / |baseline|) * 100.0`.
    pub delta_pct: f64,
    /// Median (`p50`) operation latency in nanoseconds.
    pub p50_ns: u64,
    /// 99th percentile (`p99`) operation latency in nanoseconds.
    pub p99_ns: u64,
    /// 99.9th percentile (`p99.9`) operation latency in nanoseconds.
    pub p999_ns: u64,
    /// Observed empirical false positive rate in `[0.0, 1.0]`.
    pub empirical_fpr: f64,
    /// Total observed false negatives (MUST be `0` across all valid runs).
    pub false_negatives: u64,
    /// Total observed repeated false positives (MUST be `0` for monotonic Aeris runs).
    pub repeated_fps: u64,
    /// Peak RAM usage in bytes (filter/system footprint or process Peak RSS).
    pub peak_ram_bytes: u64,
    /// Total measured disk I/O bytes (WAL + RocksDB files written/read).
    pub disk_io_bytes: u64,
    /// Applied scale factor (`1.0` at full scale or `0.5` under RAM ceiling).
    pub scale_factor: f64,
    /// Free-form structured diagnostic context for the Markdown report.
    #[serde(default)]
    pub notes: String,
}

/// Computes signed percentage delta `((optimized - baseline) / |baseline|) * 100.0`.
#[inline]
#[must_use]
pub fn compute_delta_pct(baseline: f64, optimized: f64) -> f64 {
    if baseline.abs() > 1e-12 {
        ((optimized - baseline) / baseline.abs()) * 100.0
    } else if optimized.abs() <= 1e-12 {
        0.0
    } else {
        100.0 * optimized.signum()
    }
}

/// Per-worker JSON envelope written to `target/ablation_worker_<id>.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WorkerAblationReport {
    /// Worker identifier (e.g., `"worker1_fpr_dynamic_r"`).
    pub worker_name: String,
    /// Host hardware and Peak RSS telemetry captured by this worker.
    pub host: HostMetadata,
    /// Ordered list of [`AblationMetric`] measurements recorded by this worker.
    pub metrics: Vec<AblationMetric>,
    /// Additional structured notes or methodology observations.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Returns the `Filter/target/` directory path for persisting worker JSON files.
#[must_use]
pub fn ablation_target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target")
}

/// Atomically writes a [`WorkerAblationReport`] to
/// `Filter/target/ablation_<worker_name>.json` (temp file + rename) so concurrent
/// readers never observe truncated JSON.
///
/// # Panics
/// Panics if directory creation, JSON serialization, or atomic rename fails.
pub fn write_worker_report(report: &WorkerAblationReport) -> PathBuf {
    let target_dir = ablation_target_dir();
    fs::create_dir_all(&target_dir).expect("create target directory for ablation JSON");
    let final_path = target_dir.join(format!("ablation_{}.json", report.worker_name));
    let tmp_path = target_dir.join(format!("ablation_{}.json.tmp", report.worker_name));

    let payload =
        serde_json::to_string_pretty(report).expect("serialize WorkerAblationReport to JSON");
    fs::write(&tmp_path, payload).expect("write temporary ablation JSON");
    if final_path.exists() {
        let _ = fs::remove_file(&final_path);
    }
    fs::rename(&tmp_path, &final_path).expect("atomically move ablation JSON into place");
    info!(
        worker = %report.worker_name,
        metrics = report.metrics.len(),
        path = %final_path.display(),
        "persisted worker ablation report JSON"
    );
    final_path
}

/// Recursively computes total file size in bytes beneath `path` (directory or file),
/// used to measure real WAL / RocksDB on-disk storage footprints.
#[must_use]
pub fn directory_size_bytes(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    let mut total = 0u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            total = total.saturating_add(directory_size_bytes(&entry_path));
        } else if let Ok(meta) = entry.metadata() {
            total = total.saturating_add(meta.len());
        }
    }
    total
}

/// Compact sorted-vector exact ground-truth oracle supporting `O(log N)` point and
/// closed-range emptiness checks at 10M+ key scale with ~8 bytes/key RAM footprint.
#[derive(Clone, Debug)]
pub struct SortedKeyOracle {
    sorted_keys: Vec<Key>,
}

impl SortedKeyOracle {
    /// Builds a deduplicated, sorted [`SortedKeyOracle`] from `keys`.
    ///
    /// # Complexity
    /// `O(N log N)` time, `O(N)` space.
    #[must_use]
    pub fn from_keys(mut keys: Vec<Key>) -> Self {
        keys.sort_unstable();
        keys.dedup();
        Self { sorted_keys: keys }
    }

    /// Returns the sorted slice of unique keys.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Key] {
        &self.sorted_keys
    }

    /// Consumes the oracle and returns the underlying sorted `Vec<Key>`.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<Key> {
        self.sorted_keys
    }

    /// Returns the number of unique keys in the oracle.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.sorted_keys.len()
    }

    /// Returns `true` if the oracle contains zero keys.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorted_keys.is_empty()
    }

    /// Exact point membership check in `O(log N)`.
    #[inline]
    #[must_use]
    pub fn contains(&self, key: Key) -> bool {
        self.sorted_keys.binary_search(&key).is_ok()
    }

    /// Exact closed-range emptiness check (`true` iff no key lies in `[start, end]`)
    /// in `O(log N)` time with zero heap allocations.
    #[inline]
    #[must_use]
    pub fn is_range_empty(&self, range: QueryRange) -> bool {
        let idx = self.sorted_keys.partition_point(|&k| k < range.start());
        match self.sorted_keys.get(idx) {
            Some(&k) => k > range.end(),
            None => true,
        }
    }
}

/// Deterministic discrete Zipfian sampler over ranks `0..domain_size` with
/// `P(rank = i) proportional to 1 / (i + 1)^alpha`.
#[derive(Clone, Debug)]
pub struct ZipfianSampler {
    cdf: Vec<f64>,
}

impl ZipfianSampler {
    /// Constructs an inverse-CDF Zipfian sampler for `domain_size >= 1`, `alpha > 0.0`.
    ///
    /// # Panics
    /// Panics if `domain_size == 0` or `alpha` is non-finite or `<= 0.0`.
    #[must_use]
    pub fn new(domain_size: usize, alpha: f64) -> Self {
        assert!(domain_size > 0, "Zipfian domain_size must be >= 1");
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

    /// Samples a zero-based rank in `0..self.cdf.len()`.
    #[inline]
    #[must_use]
    pub fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let u: f64 = rng.gen();
        let idx = self.cdf.partition_point(|&c| c < u);
        idx.min(self.cdf.len() - 1)
    }
}

/// Generates `count` unique uniformly distributed keys within `[0, KEYSPACE_MASK]`
/// backed by a compact [`SortedKeyOracle`].
///
/// # Complexity
/// `O(count log count)` time and `8 * count` bytes peak memory.
#[must_use]
pub fn generate_uniform_keys_oracle(seed: u64, count: usize) -> SortedKeyOracle {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(count);
    for _ in 0..count {
        keys.push(rng.gen::<u64>() & KEYSPACE_MASK);
    }
    keys.sort_unstable();
    keys.dedup();
    while keys.len() < count {
        let deficit = count - keys.len();
        keys.reserve(deficit);
        for _ in 0..deficit {
            keys.push(rng.gen::<u64>() & KEYSPACE_MASK);
        }
        keys.sort_unstable();
        keys.dedup();
    }
    keys.truncate(count);
    SortedKeyOracle { sorted_keys: keys }
}

/// Generates `count` verified-empty uniform closed range queries of exact length
/// `range_len >= 1` disjoint from `oracle`.
#[must_use]
pub fn generate_uniform_empty_queries(
    seed: u64,
    oracle: &SortedKeyOracle,
    range_len: u64,
    count: usize,
) -> Vec<QueryRange> {
    assert!(range_len >= 1, "range_len must be >= 1");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);
    let mut queries = Vec::with_capacity(count);

    while queries.len() < count {
        let start = rng.gen_range(0..=max_start);
        let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
        if oracle.is_range_empty(candidate) {
            queries.push(candidate);
        }
    }
    queries
}

/// Generates `count` **unique** verified-empty uniform closed range queries of exact
/// length `range_len >= 1` disjoint from `oracle`.
///
/// Uses a sorted-start uniqueness pass so 5,000,000 unique empty queries can be
/// generated deterministically without `HashSet` overhead, then shuffled with a
/// seeded Fisher-Yates pass to defeat spatial prefetching.
#[must_use]
pub fn generate_unique_uniform_empty_queries(
    seed: u64,
    oracle: &SortedKeyOracle,
    range_len: u64,
    count: usize,
) -> Vec<QueryRange> {
    assert!(range_len >= 1, "range_len must be >= 1");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);
    let mut starts: Vec<u64> = Vec::with_capacity(count);

    while starts.len() < count {
        let deficit = count - starts.len();
        let batch = deficit.max(1024);
        starts.reserve(batch);
        let mut added = 0usize;
        while added < batch {
            let start = rng.gen_range(0..=max_start);
            let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
            if oracle.is_range_empty(candidate) {
                starts.push(start);
                added += 1;
            }
        }
        starts.sort_unstable();
        starts.dedup();
    }
    starts.truncate(count);

    // Deterministic Fisher-Yates shuffle to ensure genuinely random DRAM access order.
    for i in (1..starts.len()).rev() {
        let j = rng.gen_range(0..=i);
        starts.swap(i, j);
    }

    starts
        .into_iter()
        .map(|s| QueryRange::with_len(s, range_len).expect("range_len >= 1"))
        .collect()
}

/// Generates `count` verified-empty correlated Zipfian range queries
/// (`alpha = 1.5`, correlation degree `D = 0.8`) matching `Aeris Filters.pdf` Section 5.2.
#[must_use]
pub fn generate_correlated_zipfian_empty_queries(
    seed: u64,
    oracle: &SortedKeyOracle,
    range_len: u64,
    alpha: f64,
    correlation_d: f64,
    count: usize,
) -> Vec<QueryRange> {
    let sorted_keys = oracle.as_slice();
    assert!(!sorted_keys.is_empty(), "oracle must be non-empty");
    assert!(range_len >= 1, "range_len must be >= 1");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let hot_domain = sorted_keys.len().min(4_096);
    let zipf = ZipfianSampler::new(hot_domain, alpha);
    let stride = (sorted_keys.len() / hot_domain).max(1);
    let exp_bits = ((30.0 * (1.0 - correlation_d.clamp(0.0, 1.0))).round() as u32).clamp(1, 30);
    let max_offset = (1u64 << exp_bits).max(2);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);

    let mut rank_queries = Vec::with_capacity(hot_domain);
    for rank in 0..hot_domain {
        let anchor_idx = (rank.wrapping_mul(stride) + (rank.wrapping_mul(2_654_435_761) % stride))
            % sorted_keys.len();
        let anchor = sorted_keys[anchor_idx];
        let mut chosen = None;
        for attempt in 0..32u64 {
            let offset = rng.gen_range(1..=max_offset).saturating_add(attempt);
            let start = anchor.saturating_add(offset).min(max_start);
            let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
            if oracle.is_range_empty(candidate) {
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
                if oracle.is_range_empty(candidate) {
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
            if oracle.is_range_empty(candidate) {
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
    oracle: &SortedKeyOracle,
    range_len: u64,
    alpha: f64,
    pool_size: usize,
    count: usize,
) -> Vec<QueryRange> {
    let pool = generate_uniform_empty_queries(seed ^ 0x51A7_900D, oracle, range_len, pool_size);
    let zipf = ZipfianSampler::new(pool.len(), alpha);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_1234);

    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        let rank = zipf.sample(&mut rng);
        queries.push(pool[rank]);
    }
    queries
}

/// Generates an adversarial repeat-FP probing trace of length `count` where a
/// configured `repeat_rate` (default `10%`) replays known false-positive ranges
/// drawn from `known_fp_pool`, interleaved with fresh verified-empty queries.
///
/// # Panics
/// Panics if `known_fp_pool` is empty or `repeat_rate` is outside `[0.0, 1.0]`.
#[must_use]
pub fn generate_adversarial_repeat_fp_trace(
    seed: u64,
    oracle: &SortedKeyOracle,
    known_fp_pool: &[QueryRange],
    range_len: u64,
    repeat_rate: f64,
    count: usize,
) -> Vec<QueryRange> {
    assert!(!known_fp_pool.is_empty(), "known_fp_pool must be non-empty");
    assert!(
        (0.0..=1.0).contains(&repeat_rate),
        "repeat_rate must be in [0.0, 1.0]"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);
    let mut trace = Vec::with_capacity(count);

    while trace.len() < count {
        let draw: f64 = rng.gen();
        if draw < repeat_rate {
            let idx = rng.gen_range(0..known_fp_pool.len());
            trace.push(known_fp_pool[idx]);
        } else {
            let start = rng.gen_range(0..=max_start);
            let candidate = QueryRange::with_len(start, range_len).expect("range_len >= 1");
            if oracle.is_range_empty(candidate) {
                trace.push(candidate);
            }
        }
    }
    trace
}

/// A single phase of a shifting-`R` query workload (`R = 16 -> 256 -> 32 -> 1024`).
#[derive(Clone, Debug)]
pub struct ShiftingRPhase {
    /// Target query range length `R` for this phase.
    pub range_len: u64,
    /// Verified-empty range queries of exact length `range_len`.
    pub queries: Vec<QueryRange>,
}

/// Generates a deterministic multi-phase shifting-`R` empty-query trace covering
/// `phase_lengths` (canonically [`SHIFTING_R_PHASES`] = `[16, 256, 32, 1024]`)
/// with `total_queries` distributed evenly across phases.
#[must_use]
pub fn generate_shifting_r_trace(
    seed: u64,
    oracle: &SortedKeyOracle,
    phase_lengths: &[u64],
    total_queries: usize,
) -> Vec<ShiftingRPhase> {
    assert!(!phase_lengths.is_empty(), "phase_lengths must be non-empty");
    let base_per_phase = total_queries / phase_lengths.len();
    let remainder = total_queries % phase_lengths.len();

    phase_lengths
        .iter()
        .enumerate()
        .map(|(idx, &range_len)| {
            let phase_count = base_per_phase + usize::from(idx < remainder);
            let phase_seed = seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let queries =
                generate_uniform_empty_queries(phase_seed, oracle, range_len, phase_count);
            ShiftingRPhase { range_len, queries }
        })
        .collect()
}

/// One time-step of a deterministic diurnal/seasonal ingestion and query-length trace
/// consumed by proactive EWMA and ML/RL advisory sidecar ablations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SeasonalTracePoint {
    /// Zero-based time bucket index.
    pub step: usize,
    /// Instantaneous ingestion rate (keys/sec) at this step.
    pub ingest_rate_keys_per_sec: f64,
    /// Dominant query range length (`p95`) at this step.
    pub p95_query_len: f64,
    /// Instantaneous query arrival rate (queries/sec) at this step.
    pub query_rate_qps: f64,
}

/// Generates a deterministic periodic diurnal/seasonal ingestion and query-length trace
/// with realistic multi-harmonic surges and troughs (matching the Python sidecar's
/// seasonal Holt-Winters + Fourier forecaster period `24`).
#[must_use]
pub fn generate_seasonal_ingestion_trace(
    seed: u64,
    steps: usize,
    period: usize,
    base_ingest_rate: f64,
    peak_amplitude: f64,
) -> Vec<SeasonalTracePoint> {
    assert!(steps > 0, "steps must be > 0");
    assert!(period > 0, "period must be > 0");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let period_f = period as f64;
    let mut trace = Vec::with_capacity(steps);

    for step in 0..steps {
        let t = step as f64;
        let angle = (2.0 * PI / period_f) * t;
        let damped_trend = (1.0 - 0.94_f64.powf(t)) / (1.0 - 0.94);
        let noise: f64 = (rng.gen::<f64>() - 0.5) * 0.04 * peak_amplitude;

        let ingest = base_ingest_rate
            + 0.02 * base_ingest_rate * damped_trend
            + peak_amplitude * angle.sin()
            + 0.35 * peak_amplitude * (2.0 * angle).cos()
            + 0.15 * peak_amplitude * (3.0 * angle).sin()
            + noise;

        let log_qlen =
            4.0 + 0.025 * damped_trend + 1.15 * (angle - 0.35).sin() + 0.45 * (2.0 * angle).cos();
        let p95_qlen = log_qlen.exp_m1().clamp(8.0, 2048.0);

        let qps = (15_000.0 + 12_000.0 * (angle + 0.5).cos()).max(1_000.0);

        trace.push(SeasonalTracePoint {
            step,
            ingest_rate_keys_per_sec: ingest.max(250.0),
            p95_query_len: p95_qlen,
            query_rate_qps: qps,
        });
    }
    trace
}

/// Generates `count` deterministic keys exhibiting a skewed hot-spot write distribution:
/// `hot_fraction` (e.g., `0.85`) of writes fall inside a narrow `[hot_start, hot_start + hot_span)`
/// keyspace band while the remaining `1 - hot_fraction` are spread uniformly across `[0, KEYSPACE_MASK]`.
///
/// Guaranteed to produce `count` unique keys in deterministic generation order (not sorted),
/// exercising live Range Sharding hot-shard splits without artificial key collisions.
#[must_use]
pub fn generate_skewed_hotspot_write_trace(
    seed: u64,
    count: usize,
    hot_fraction: f64,
    hot_start: Key,
    hot_span: u64,
) -> Vec<Key> {
    assert!(
        (0.0..=1.0).contains(&hot_fraction),
        "hot_fraction must be in [0.0, 1.0]"
    );
    assert!(hot_span >= 2, "hot_span must be >= 2");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(count);
    let mut seen = std::collections::HashSet::with_capacity(count.min(2_000_000));
    let mut hot_cursor = 0u64;

    while keys.len() < count {
        let draw: f64 = rng.gen();
        let candidate = if draw < hot_fraction {
            // Mix pseudo-random offset inside the hot band; fall back to stride on collision.
            let raw = hot_start.wrapping_add(rng.gen_range(0..hot_span)) & KEYSPACE_MASK;
            if seen.contains(&raw) {
                hot_cursor = hot_cursor.wrapping_add(1);
                hot_start.wrapping_add(hot_cursor % hot_span) & KEYSPACE_MASK
            } else {
                raw
            }
        } else {
            rng.gen::<u64>() & KEYSPACE_MASK
        };

        if seen.insert(candidate) {
            keys.push(candidate);
        }
    }
    keys
}

/// Builds a closed [`QueryRange`] centered on `key` with radius `half_width` without overflow.
#[inline]
#[must_use]
pub fn enclosing_range(key: Key, half_width: u64) -> QueryRange {
    let start = key.saturating_sub(half_width);
    let end = key.saturating_add(half_width);
    QueryRange::new(start, end).expect("start <= end")
}

/// Exact order-statistic latency summary in nanoseconds including `p50`, `p95`, `p99`, and `p99.9`.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LatencyStats {
    /// Minimum observed latency (ns).
    pub min_ns: u64,
    /// Median (`p50`) latency (ns).
    pub p50_ns: u64,
    /// 95th percentile (`p95`) latency (ns).
    pub p95_ns: u64,
    /// 99th percentile (`p99`) latency (ns).
    pub p99_ns: u64,
    /// 99.9th percentile (`p99.9`) latency (ns).
    pub p999_ns: u64,
    /// Maximum observed latency (ns).
    pub max_ns: u64,
    /// Arithmetic mean latency (ns).
    pub mean_ns: f64,
}

impl LatencyStats {
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
            p50_ns: samples_ns[permille_idx(n, 500)],
            p95_ns: samples_ns[permille_idx(n, 950)],
            p99_ns: samples_ns[permille_idx(n, 990)],
            p999_ns: samples_ns[permille_idx(n, 999)],
            max_ns: samples_ns[n - 1],
            mean_ns: (sum as f64) / (n as f64),
        }
    }
}

#[inline]
fn permille_idx(len: usize, permille: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let num = (len - 1).saturating_mul(permille);
    (num / 1000).min(len - 1)
}
