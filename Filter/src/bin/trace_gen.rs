//! Deterministic high-throughput binary trace generator for Aeris paper-parity
//! benchmarks (shared verbatim between the Rust and C++ drivers).
//!
//! # Shared Little-Endian Binary Formats
//! - **Keys (`*.keys.bin`)**:
//!   - 16-byte header: `b"AERISKEY"` (8B) + `count: u64 LE` (8B)
//!   - Payload: `count` unique 48-bit keys (`u64 LE` in `[0, 2^48)`) in deterministic
//!     shuffled order produced by a 6-round 48-bit Feistel pseudorandom permutation
//!     (`O(N)` time, `O(1)` extra RAM).
//! - **Queries (`*.queries.bin`)**:
//!   - 32-byte header: `b"AERISQRY"` (8B) + `count: u64 LE` (8B)
//!     + `max_range_r: u64 LE` (8B) + `flags: u64 LE = 1` (8B)
//!   - Payload: `count` records of 24 bytes each:
//!     `{ start: u64 LE, end: u64 LE, expected: u64 LE }`.
//!
//! # CLI Usage
//! ```text
//! trace_gen keys --count <N> --seed <u64> --out <path>
//! trace_gen queries --keys <keys.bin> --kind <zipf_corr|zipf_uncorr|uniform_empty|shifting_r> \
//!   --count <Q> --range-len <R> --alpha <1.5> --corr-d <0.8> --seed <u64> --out <path>
//! ```

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

/// 8-byte magic prefix for binary key trace files.
const KEYS_MAGIC: [u8; 8] = *b"AERISKEY";

/// 8-byte magic prefix for binary query trace files.
const QUERIES_MAGIC: [u8; 8] = *b"AERISQRY";

/// Binary query format flags (bit 0 = includes explicit `expected` field).
const QUERIES_FLAGS: u64 = 1;

/// 48-bit keyspace upper bound (`[0, 2^48)`).
const KEYSPACE_BITS: u32 = 48;
const KEYSPACE_SIZE: u64 = 1u64 << KEYSPACE_BITS;
const KEYSPACE_MASK: u64 = KEYSPACE_SIZE - 1;

/// Sequential I/O buffer size (16 MiB) for disk-speed streaming up to 5e8 keys.
const IO_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Chunk size (in u64 keys) assembled in memory before writing to the buffered writer.
const KEY_WRITE_CHUNK_LEN: usize = 131_072; // 1 MiB

/// Default Zipfian pool size for uncorrelated and correlated hot-anchor queries.
const ZIPF_POOL_SIZE: usize = 4_096;

/// Canonical 4-phase range lengths for `shifting_r` workload.
const SHIFTING_R_PHASES: [u64; 4] = [16, 256, 32, 1024];

/// Supported query workload distributions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueryKind {
    /// SIGMOD 2026 Section 5.2 Zipfian (`alpha`) correlated (`D`) verified-empty queries.
    ZipfCorr,
    /// Uncorrelated Zipfian (`alpha`) over a pool of 4,096 uniform verified-empty queries.
    ZipfUncorr,
    /// Uniform verified-empty ranges of fixed width `R`.
    UniformEmpty,
    /// 4 equal phases of uniform verified-empty ranges with widths `[16, 256, 32, 1024]`.
    ShiftingR,
}

impl QueryKind {
    /// Parses a CLI `--kind` argument string.
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "zipf_corr" => Ok(Self::ZipfCorr),
            "zipf_uncorr" => Ok(Self::ZipfUncorr),
            "uniform_empty" => Ok(Self::UniformEmpty),
            "shifting_r" => Ok(Self::ShiftingR),
            other => Err(format!(
                "unsupported --kind '{other}': expected zipf_corr | zipf_uncorr | uniform_empty | shifting_r"
            )),
        }
    }

    /// Returns the canonical string identifier for this query kind.
    const fn as_str(self) -> &'static str {
        match self {
            Self::ZipfCorr => "zipf_corr",
            Self::ZipfUncorr => "zipf_uncorr",
            Self::UniformEmpty => "uniform_empty",
            Self::ShiftingR => "shifting_r",
        }
    }
}

/// Parsed CLI subcommand.
enum Command {
    Keys {
        count: u64,
        seed: u64,
        out: PathBuf,
    },
    Queries {
        keys_path: PathBuf,
        kind: QueryKind,
        count: u64,
        range_len: u64,
        alpha: f64,
        corr_d: f64,
        seed: u64,
        out: PathBuf,
    },
}

/// Structured JSON summary emitted to stdout upon successful trace generation.
#[derive(Serialize)]
struct TraceGenSummary {
    command: &'static str,
    kind: String,
    count: u64,
    max_range_r: u64,
    seed: u64,
    out: String,
    bytes_written: u64,
    elapsed_ms: f64,
}

/// SplitMix64 mixing function used for deterministic Feistel round keys and PRFs.
#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// 6-round balanced 24-bit Feistel pseudorandom permutation over `[0, 2^48)`.
///
/// # Mathematical Guarantees
/// - Strictly bijective on `[0, 2^48)`: mapping `i = 0..N` (`N <= 2^48`) produces
///   100% unique uniform 48-bit keys in shuffled order with `O(1)` extra RAM.
#[derive(Clone, Copy, Debug)]
struct Feistel48 {
    round_keys: [u64; 6],
}

impl Feistel48 {
    const HALF_MASK: u64 = (1u64 << 24) - 1;

    /// Derives 6 independent 64-bit round keys from `seed`.
    const fn from_seed(seed: u64) -> Self {
        let mut s = seed ^ 0xA3E1_5F01_9C4D_7B23;
        let mut rk = [0u64; 6];
        let mut i = 0;
        while i < 6 {
            s = splitmix64(s);
            rk[i] = s;
            i += 1;
        }
        Self { round_keys: rk }
    }

    /// Evaluates the bijective 48-bit permutation at index `x in [0, 2^48)`.
    #[inline]
    const fn permute(&self, x: u64) -> u64 {
        let mut l = (x >> 24) & Self::HALF_MASK;
        let mut r = x & Self::HALF_MASK;
        let mut i = 0;
        while i < 6 {
            let f = splitmix64(r ^ self.round_keys[i]) & Self::HALF_MASK;
            let next_l = r;
            let next_r = l ^ f;
            l = next_l;
            r = next_r;
            i += 1;
        }
        (l << 24) | r
    }
}

/// Deterministic discrete Zipfian sampler over ranks `0..domain_size` with
/// `P(rank = i) proportional to 1 / (i + 1)^alpha`.
struct ZipfianSampler {
    cdf: Vec<f64>,
}

impl ZipfianSampler {
    /// Builds an inverse-CDF Zipfian table in `O(domain_size)` time and space.
    fn new(domain_size: usize, alpha: f64) -> Result<Self, String> {
        if domain_size == 0 {
            return Err("Zipfian domain_size must be >= 1".to_string());
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(format!(
                "Zipfian alpha must be finite and > 0.0, got {alpha}"
            ));
        }
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
        Ok(Self { cdf })
    }

    /// Samples a zero-based rank in `0..self.cdf.len()` in `O(log domain_size)`.
    #[inline]
    fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let u: f64 = rng.gen();
        let idx = self.cdf.partition_point(|&c| c < u);
        idx.min(self.cdf.len() - 1)
    }
}

/// Exact closed-range emptiness check against a sorted key slice in `O(log N)`.
#[inline]
fn is_range_empty(sorted_keys: &[u64], start: u64, end: u64) -> bool {
    let idx = sorted_keys.partition_point(|&k| k < start);
    match sorted_keys.get(idx) {
        Some(&k) => k > end,
        None => true,
    }
}

/// Ensures the parent directory of `path` exists before creating an output file.
fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Streams `count` unique 48-bit keys to `out` using a 6-round Feistel PRP and
/// 16 MiB buffered sequential writes.
///
/// # Complexity
/// `O(count)` time and `O(1)` extra RAM (`~17 MiB` fixed I/O buffers).
fn generate_keys_file(count: u64, seed: u64, out: &Path) -> Result<u64, String> {
    if count == 0 {
        return Err("--count must be >= 1".to_string());
    }
    if count > KEYSPACE_SIZE {
        return Err(format!(
            "--count ({count}) exceeds 48-bit keyspace capacity ({KEYSPACE_SIZE})"
        ));
    }
    ensure_parent_dir(out).map_err(|e| format!("failed to create output directory: {e}"))?;

    let file = File::create(out)
        .map_err(|e| format!("failed to create key file '{}': {e}", out.display()))?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, file);

    // 16-byte header: magic + count LE
    writer
        .write_all(&KEYS_MAGIC)
        .and_then(|()| writer.write_all(&count.to_le_bytes()))
        .map_err(|e| format!("failed to write key file header: {e}"))?;

    let prp = Feistel48::from_seed(seed);
    let mut chunk_buf = vec![0u8; KEY_WRITE_CHUNK_LEN * 8];
    let mut emitted = 0u64;

    while emitted < count {
        let batch = ((count - emitted) as usize).min(KEY_WRITE_CHUNK_LEN);
        for i in 0..batch {
            let key = prp.permute(emitted + (i as u64));
            let off = i * 8;
            chunk_buf[off..off + 8].copy_from_slice(&key.to_le_bytes());
        }
        writer
            .write_all(&chunk_buf[..batch * 8])
            .map_err(|e| format!("failed to write key payload: {e}"))?;
        emitted += batch as u64;
    }

    writer
        .flush()
        .map_err(|e| format!("failed to flush key file: {e}"))?;
    Ok(16 + count * 8)
}

/// Reads all keys from a `*.keys.bin` file and returns them sorted ascending for
/// `O(log N)` emptiness verification.
fn read_and_sort_keys(path: &Path) -> Result<Vec<u64>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open key file '{}': {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);

    let mut header = [0u8; 16];
    reader.read_exact(&mut header).map_err(|e| {
        format!(
            "failed to read 16B key header from '{}': {e}",
            path.display()
        )
    })?;
    if header[0..8] != KEYS_MAGIC {
        return Err(format!(
            "invalid key file magic in '{}': expected {:?}, got {:?}",
            path.display(),
            KEYS_MAGIC,
            &header[0..8]
        ));
    }
    let count = u64::from_le_bytes(header[8..16].try_into().expect("8B count"));
    let count_usize =
        usize::try_from(count).map_err(|_| format!("key count {count} exceeds usize"))?;

    let mut keys = Vec::with_capacity(count_usize);
    let mut buf = vec![0u8; KEY_WRITE_CHUNK_LEN * 8];
    let mut remaining = count_usize;

    while remaining > 0 {
        let batch = remaining.min(KEY_WRITE_CHUNK_LEN);
        let byte_len = batch * 8;
        reader
            .read_exact(&mut buf[..byte_len])
            .map_err(|e| format!("truncated key payload in '{}': {e}", path.display()))?;
        for chunk in buf[..byte_len].as_chunks::<8>().0 {
            keys.push(u64::from_le_bytes(*chunk));
        }
        remaining -= batch;
    }

    keys.sort_unstable();
    Ok(keys)
}

/// Draws a single uniform verified-empty closed range `[start, start + range_len - 1]`
/// inside `[0, KEYSPACE_MASK]`.
#[inline]
fn sample_uniform_empty(
    rng: &mut ChaCha8Rng,
    sorted_keys: &[u64],
    range_len: u64,
    max_start: u64,
) -> (u64, u64) {
    loop {
        let start = rng.gen_range(0..=max_start);
        let end = start + (range_len - 1);
        if is_range_empty(sorted_keys, start, end) {
            return (start, end);
        }
    }
}

/// Buffered 24-byte query record writer.
struct QueryStreamWriter {
    writer: BufWriter<File>,
    buf: Vec<u8>,
    written_records: u64,
}

impl QueryStreamWriter {
    const CHUNK_RECORDS: usize = 65_536; // 1.5 MiB

    fn create(out: &Path, count: u64, max_range_r: u64) -> Result<Self, String> {
        ensure_parent_dir(out).map_err(|e| format!("failed to create output directory: {e}"))?;
        let file = File::create(out)
            .map_err(|e| format!("failed to create query file '{}': {e}", out.display()))?;
        let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, file);

        // 32-byte header: magic + count + max_range_r + flags
        writer
            .write_all(&QUERIES_MAGIC)
            .and_then(|()| writer.write_all(&count.to_le_bytes()))
            .and_then(|()| writer.write_all(&max_range_r.to_le_bytes()))
            .and_then(|()| writer.write_all(&QUERIES_FLAGS.to_le_bytes()))
            .map_err(|e| format!("failed to write query header: {e}"))?;

        Ok(Self {
            writer,
            buf: Vec::with_capacity(Self::CHUNK_RECORDS * 24),
            written_records: 0,
        })
    }

    #[inline]
    fn push(&mut self, start: u64, end: u64, expected: u64) -> Result<(), String> {
        debug_assert!(start <= end);
        self.buf.extend_from_slice(&start.to_le_bytes());
        self.buf.extend_from_slice(&end.to_le_bytes());
        self.buf.extend_from_slice(&expected.to_le_bytes());
        self.written_records += 1;
        if self.buf.len() >= Self::CHUNK_RECORDS * 24 {
            self.flush_chunk()?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> Result<(), String> {
        if !self.buf.is_empty() {
            self.writer
                .write_all(&self.buf)
                .map_err(|e| format!("failed to write query records: {e}"))?;
            self.buf.clear();
        }
        Ok(())
    }

    fn finish(mut self) -> Result<u64, String> {
        self.flush_chunk()?;
        self.writer
            .flush()
            .map_err(|e| format!("failed to flush query file: {e}"))?;
        Ok(32 + self.written_records * 24)
    }
}

/// Generates a binary query trace file verified empty against `keys_path`.
#[allow(clippy::too_many_arguments)]
fn generate_queries_file(
    keys_path: &Path,
    kind: QueryKind,
    count: u64,
    range_len: u64,
    alpha: f64,
    corr_d: f64,
    seed: u64,
    out: &Path,
) -> Result<(u64, u64), String> {
    if count == 0 {
        return Err("--count must be >= 1".to_string());
    }
    if range_len == 0 || range_len > KEYSPACE_SIZE {
        return Err(format!(
            "--range-len ({range_len}) must be in [1, {KEYSPACE_SIZE}]"
        ));
    }
    if !corr_d.is_finite() || !(0.0..=1.0).contains(&corr_d) {
        return Err(format!(
            "--corr-d must be finite and in [0.0, 1.0], got {corr_d}"
        ));
    }

    let sorted_keys = read_and_sort_keys(keys_path)?;
    if sorted_keys.is_empty() {
        return Err("key file contains zero keys".to_string());
    }

    let max_range_r = match kind {
        QueryKind::ShiftingR => *SHIFTING_R_PHASES.iter().max().expect("non-empty phases"),
        _ => range_len,
    };

    let mut stream = QueryStreamWriter::create(out, count, max_range_r)?;
    let max_start = KEYSPACE_MASK.saturating_sub(range_len - 1);

    match kind {
        QueryKind::UniformEmpty => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..count {
                let (s, e) = sample_uniform_empty(&mut rng, &sorted_keys, range_len, max_start);
                stream.push(s, e, 0)?;
            }
        }
        QueryKind::ZipfUncorr => {
            let mut pool_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x51A7_900D);
            let mut pool = Vec::with_capacity(ZIPF_POOL_SIZE);
            for _ in 0..ZIPF_POOL_SIZE {
                pool.push(sample_uniform_empty(
                    &mut pool_rng,
                    &sorted_keys,
                    range_len,
                    max_start,
                ));
            }
            let zipf = ZipfianSampler::new(pool.len(), alpha)?;
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_1234);
            for _ in 0..count {
                let rank = zipf.sample(&mut rng);
                let (s, e) = pool[rank];
                stream.push(s, e, 0)?;
            }
        }
        QueryKind::ZipfCorr => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let hot_domain = sorted_keys.len().min(ZIPF_POOL_SIZE);
            let zipf = ZipfianSampler::new(hot_domain, alpha)?;
            let stride = (sorted_keys.len() / hot_domain).max(1);
            let exp_bits = ((30.0 * (1.0 - corr_d.clamp(0.0, 1.0))).round() as u32).clamp(1, 30);
            let max_offset = (1u64 << exp_bits).max(2);

            let mut rank_queries = Vec::with_capacity(hot_domain);
            for rank in 0..hot_domain {
                let anchor_idx = (rank
                    .wrapping_mul(stride)
                    .wrapping_add(rank.wrapping_mul(2_654_435_761) % stride))
                    % sorted_keys.len();
                let anchor = sorted_keys[anchor_idx];
                let mut chosen = None;
                for attempt in 0..64u64 {
                    let offset = rng.gen_range(1..=max_offset).saturating_add(attempt);
                    let start = anchor.saturating_add(offset).min(max_start);
                    let end = start + (range_len - 1);
                    if is_range_empty(&sorted_keys, start, end) {
                        chosen = Some((start, end));
                        break;
                    }
                }
                let q = chosen.unwrap_or_else(|| {
                    sample_uniform_empty(&mut rng, &sorted_keys, range_len, max_start)
                });
                rank_queries.push(q);
            }

            for _ in 0..count {
                let draw: f64 = rng.gen();
                let (s, e) = if draw < corr_d {
                    let rank = zipf.sample(&mut rng);
                    rank_queries[rank]
                } else {
                    sample_uniform_empty(&mut rng, &sorted_keys, range_len, max_start)
                };
                stream.push(s, e, 0)?;
            }
        }
        QueryKind::ShiftingR => {
            let num_phases = SHIFTING_R_PHASES.len() as u64;
            let base = count / num_phases;
            let rem = count % num_phases;
            for (idx, &phase_r) in SHIFTING_R_PHASES.iter().enumerate() {
                let phase_count = base + u64::from((idx as u64) < rem);
                let phase_seed = seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = ChaCha8Rng::seed_from_u64(phase_seed);
                let phase_max_start = KEYSPACE_MASK.saturating_sub(phase_r - 1);
                for _ in 0..phase_count {
                    let (s, e) =
                        sample_uniform_empty(&mut rng, &sorted_keys, phase_r, phase_max_start);
                    stream.push(s, e, 0)?;
                }
            }
        }
    }

    let bytes = stream.finish()?;
    Ok((bytes, max_range_r))
}

/// Parses CLI arguments into a strongly typed [`Command`].
fn parse_args(args: &[String]) -> Result<Command, String> {
    if args.len() < 2 {
        return Err(usage_string());
    }
    let sub = args[1].as_str();
    let mut i = 2usize;

    match sub {
        "keys" => {
            let mut count: Option<u64> = None;
            let mut seed: u64 = 1380;
            let mut out: Option<PathBuf> = None;

            while i < args.len() {
                let flag = args[i].as_str();
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--count" => {
                        count = Some(
                            val.parse::<u64>()
                                .map_err(|e| format!("invalid --count '{val}': {e}"))?,
                        );
                    }
                    "--seed" => {
                        seed = val
                            .parse::<u64>()
                            .map_err(|e| format!("invalid --seed '{val}': {e}"))?;
                    }
                    "--out" => {
                        out = Some(PathBuf::from(val));
                    }
                    other => return Err(format!("unknown flag '{other}' for `keys`")),
                }
                i += 2;
            }

            Ok(Command::Keys {
                count: count.ok_or_else(|| "missing required --count".to_string())?,
                seed,
                out: out.ok_or_else(|| "missing required --out".to_string())?,
            })
        }
        "queries" => {
            let mut keys_path: Option<PathBuf> = None;
            let mut kind: Option<QueryKind> = None;
            let mut count: Option<u64> = None;
            let mut range_len: u64 = 1;
            let mut alpha: f64 = 1.5;
            let mut corr_d: f64 = 0.8;
            let mut seed: u64 = 1380;
            let mut out: Option<PathBuf> = None;

            while i < args.len() {
                let flag = args[i].as_str();
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--keys" => keys_path = Some(PathBuf::from(val)),
                    "--kind" => kind = Some(QueryKind::parse(val)?),
                    "--count" => {
                        count = Some(
                            val.parse::<u64>()
                                .map_err(|e| format!("invalid --count '{val}': {e}"))?,
                        );
                    }
                    "--range-len" => {
                        range_len = val
                            .parse::<u64>()
                            .map_err(|e| format!("invalid --range-len '{val}': {e}"))?;
                    }
                    "--alpha" => {
                        alpha = val
                            .parse::<f64>()
                            .map_err(|e| format!("invalid --alpha '{val}': {e}"))?;
                    }
                    "--corr-d" => {
                        corr_d = val
                            .parse::<f64>()
                            .map_err(|e| format!("invalid --corr-d '{val}': {e}"))?;
                    }
                    "--seed" => {
                        seed = val
                            .parse::<u64>()
                            .map_err(|e| format!("invalid --seed '{val}': {e}"))?;
                    }
                    "--out" => out = Some(PathBuf::from(val)),
                    other => return Err(format!("unknown flag '{other}' for `queries`")),
                }
                i += 2;
            }

            Ok(Command::Queries {
                keys_path: keys_path.ok_or_else(|| "missing required --keys".to_string())?,
                kind: kind.ok_or_else(|| "missing required --kind".to_string())?,
                count: count.ok_or_else(|| "missing required --count".to_string())?,
                range_len,
                alpha,
                corr_d,
                seed,
                out: out.ok_or_else(|| "missing required --out".to_string())?,
            })
        }
        other => Err(format!("unknown subcommand '{other}'\n{}", usage_string())),
    }
}

fn usage_string() -> String {
    concat!(
        "Usage:\n",
        "  trace_gen keys --count <N> --seed <u64> --out <path>\n",
        "  trace_gen queries --keys <keys.bin> --kind <zipf_corr|zipf_uncorr|uniform_empty|shifting_r> \\\n",
        "    --count <Q> --range-len <R> --alpha <1.5> --corr-d <0.8> --seed <u64> --out <path>"
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
    let args: Vec<String> = env::args().collect();
    let cmd = match parse_args(&args) {
        Ok(c) => c,
        Err(msg) => {
            error!(error = %msg, "invalid trace_gen arguments");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let summary = match cmd {
        Command::Keys { count, seed, out } => {
            info!(count, seed, out = %out.display(), "generating binary keys trace");
            match generate_keys_file(count, seed, &out) {
                Ok(bytes_written) => TraceGenSummary {
                    command: "keys",
                    kind: "uniform_48bit_feistel".to_string(),
                    count,
                    max_range_r: 1,
                    seed,
                    out: out.display().to_string(),
                    bytes_written,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                },
                Err(err) => {
                    error!(error = %err, "key trace generation failed");
                    return ExitCode::FAILURE;
                }
            }
        }
        Command::Queries {
            keys_path,
            kind,
            count,
            range_len,
            alpha,
            corr_d,
            seed,
            out,
        } => {
            info!(
                keys = %keys_path.display(),
                kind = kind.as_str(),
                count,
                range_len,
                alpha,
                corr_d,
                seed,
                out = %out.display(),
                "generating binary queries trace"
            );
            match generate_queries_file(
                &keys_path, kind, count, range_len, alpha, corr_d, seed, &out,
            ) {
                Ok((bytes_written, max_range_r)) => TraceGenSummary {
                    command: "queries",
                    kind: kind.as_str().to_string(),
                    count,
                    max_range_r,
                    seed,
                    out: out.display().to_string(),
                    bytes_written,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                },
                Err(err) => {
                    error!(error = %err, "query trace generation failed");
                    return ExitCode::FAILURE;
                }
            }
        }
    };

    info!(
        command = summary.command,
        kind = %summary.kind,
        count = summary.count,
        bytes_written = summary.bytes_written,
        elapsed_ms = summary.elapsed_ms,
        "trace generation complete"
    );

    match serde_json::to_string(&summary) {
        Ok(line) => {
            let mut stdout = io::stdout().lock();
            if writeln!(stdout, "{line}").is_err() {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            error!(error = %err, "failed to serialize summary JSON");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn feistel48_is_unique_and_within_48bit_domain() {
        let prp = Feistel48::from_seed(1380);
        let n = 50_000u64;
        let mut seen = HashSet::with_capacity(n as usize);
        for i in 0..n {
            let k = prp.permute(i);
            assert!(k <= KEYSPACE_MASK, "key {k} exceeds 48-bit domain");
            assert!(seen.insert(k), "duplicate key generated at index {i}");
        }
        // Determinism check
        assert_eq!(prp.permute(42), Feistel48::from_seed(1380).permute(42));
        // Different seeds produce different outputs
        assert_ne!(prp.permute(42), Feistel48::from_seed(1381).permute(42));
    }

    #[test]
    fn keys_and_all_query_kinds_round_trip_and_verify_empty() {
        let tmp_dir = env::temp_dir().join(format!("aeris_trace_gen_test_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let keys_path = tmp_dir.join("test.keys.bin");

        let bytes = generate_keys_file(2_000, 1380, &keys_path).unwrap();
        assert_eq!(bytes, 16 + 2_000 * 8);

        let sorted_keys = read_and_sort_keys(&keys_path).unwrap();
        assert_eq!(sorted_keys.len(), 2_000);
        assert!(sorted_keys.windows(2).all(|w| w[0] < w[1]));

        for (kind, range_len, expected_max_r) in [
            (QueryKind::ZipfCorr, 32u64, 32u64),
            (QueryKind::ZipfUncorr, 16u64, 16u64),
            (QueryKind::UniformEmpty, 1u64, 1u64),
            (QueryKind::ShiftingR, 16u64, 1024u64),
        ] {
            let q_path = tmp_dir.join(format!("test_{}.queries.bin", kind.as_str()));
            let (q_bytes, max_r) =
                generate_queries_file(&keys_path, kind, 400, range_len, 1.5, 0.8, 1380, &q_path)
                    .unwrap();
            assert_eq!(q_bytes, 32 + 400 * 24);
            assert_eq!(max_r, expected_max_r);

            let raw = fs::read(&q_path).unwrap();
            assert_eq!(&raw[0..8], &QUERIES_MAGIC);
            assert_eq!(u64::from_le_bytes(raw[8..16].try_into().unwrap()), 400);
            assert_eq!(
                u64::from_le_bytes(raw[16..24].try_into().unwrap()),
                expected_max_r
            );
            assert_eq!(
                u64::from_le_bytes(raw[24..32].try_into().unwrap()),
                QUERIES_FLAGS
            );

            for (idx, rec) in raw[32..].as_chunks::<24>().0.iter().enumerate() {
                let start = u64::from_le_bytes(rec[0..8].try_into().unwrap());
                let end = u64::from_le_bytes(rec[8..16].try_into().unwrap());
                let expected = u64::from_le_bytes(rec[16..24].try_into().unwrap());
                assert!(start <= end);
                assert_eq!(expected, 0);
                assert!(is_range_empty(&sorted_keys, start, end));
                if kind == QueryKind::ShiftingR {
                    let phase_r = SHIFTING_R_PHASES[idx / 100];
                    assert_eq!(end - start + 1, phase_r);
                } else {
                    assert_eq!(end - start + 1, range_len);
                }
            }
        }

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
