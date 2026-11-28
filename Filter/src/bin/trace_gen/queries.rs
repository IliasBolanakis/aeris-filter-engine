//! Deterministic verified-empty query workload generation (`uniform_empty`,
//! `zipf_uncorr`, `zipf_corr`, `shifting_r`).

use crate::cli::{QueriesCliConfig, QueryKind};
use crate::io::{read_and_sort_keys, QueryStreamWriter};
use crate::keys::{KEYSPACE_MASK, KEYSPACE_SIZE};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Default Zipfian pool size for uncorrelated and correlated hot-anchor queries.
const ZIPF_POOL_SIZE: usize = 4_096;

/// Canonical 4-phase range lengths for `shifting_r` workload.
pub const SHIFTING_R_PHASES: [u64; 4] = [16, 256, 32, 1024];

/// Deterministic discrete Zipfian sampler over ranks `0..domain_size` with
/// `P(rank = i) proportional to 1 / (i + 1)^alpha`.
pub struct ZipfianSampler {
    cdf: Vec<f64>,
}

impl ZipfianSampler {
    /// Builds an inverse-CDF Zipfian table in `O(domain_size)` time and space.
    ///
    /// # Errors
    /// Returns an error if `domain_size == 0` or `alpha` is non-finite / `<= 0.0`.
    pub fn new(domain_size: usize, alpha: f64) -> Result<Self, String> {
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
    pub fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let u: f64 = rng.gen();
        let idx = self.cdf.partition_point(|&c| c < u);
        idx.min(self.cdf.len() - 1)
    }
}

/// Exact closed-range emptiness check against a sorted key slice in `O(log N)`.
#[inline]
pub fn is_range_empty(sorted_keys: &[u64], start: u64, end: u64) -> bool {
    let idx = sorted_keys.partition_point(|&k| k < start);
    match sorted_keys.get(idx) {
        Some(&k) => k > end,
        None => true,
    }
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

/// Generates a binary query trace file verified empty against `cfg.keys_path`.
///
/// Returns `(bytes_written, max_range_r)` on success.
///
/// # Errors
/// Returns an error on invalid count/range-len/corr-d parameters, empty key files,
/// or I/O failures.
pub fn generate_queries_file(cfg: &QueriesCliConfig) -> Result<(u64, u64), String> {
    validate_query_config(cfg)?;

    let sorted_keys = read_and_sort_keys(&cfg.keys_path)?;
    if sorted_keys.is_empty() {
        return Err("key file contains zero keys".to_string());
    }

    let max_range_r = match cfg.kind {
        QueryKind::ShiftingR => *SHIFTING_R_PHASES.iter().max().expect("non-empty phases"),
        _ => cfg.range_len,
    };

    let mut stream = QueryStreamWriter::create(&cfg.out, cfg.count, max_range_r)?;
    let max_start = KEYSPACE_MASK.saturating_sub(cfg.range_len - 1);

    match cfg.kind {
        QueryKind::UniformEmpty => {
            emit_uniform_empty(
                &mut stream,
                &sorted_keys,
                cfg.count,
                cfg.range_len,
                max_start,
                cfg.seed,
            )?;
        }
        QueryKind::ZipfUncorr => {
            emit_zipf_uncorr(&mut stream, &sorted_keys, cfg, max_start)?;
        }
        QueryKind::ZipfCorr => {
            emit_zipf_corr(&mut stream, &sorted_keys, cfg, max_start)?;
        }
        QueryKind::ShiftingR => {
            emit_shifting_r(&mut stream, &sorted_keys, cfg.count, cfg.seed)?;
        }
    }

    let bytes = stream.finish()?;
    Ok((bytes, max_range_r))
}

fn validate_query_config(cfg: &QueriesCliConfig) -> Result<(), String> {
    if cfg.count == 0 {
        return Err("--count must be >= 1".to_string());
    }
    if cfg.range_len == 0 || cfg.range_len > KEYSPACE_SIZE {
        return Err(format!(
            "--range-len ({}) must be in [1, {KEYSPACE_SIZE}]",
            cfg.range_len
        ));
    }
    if !cfg.corr_d.is_finite() || !(0.0..=1.0).contains(&cfg.corr_d) {
        return Err(format!(
            "--corr-d must be finite and in [0.0, 1.0], got {}",
            cfg.corr_d
        ));
    }
    Ok(())
}

fn emit_uniform_empty(
    stream: &mut QueryStreamWriter,
    sorted_keys: &[u64],
    count: u64,
    range_len: u64,
    max_start: u64,
    seed: u64,
) -> Result<(), String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..count {
        let (s, e) = sample_uniform_empty(&mut rng, sorted_keys, range_len, max_start);
        stream.push(s, e, 0)?;
    }
    Ok(())
}

fn emit_zipf_uncorr(
    stream: &mut QueryStreamWriter,
    sorted_keys: &[u64],
    cfg: &QueriesCliConfig,
    max_start: u64,
) -> Result<(), String> {
    let mut pool_rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x51A7_900D);
    let mut pool = Vec::with_capacity(ZIPF_POOL_SIZE);
    for _ in 0..ZIPF_POOL_SIZE {
        pool.push(sample_uniform_empty(
            &mut pool_rng,
            sorted_keys,
            cfg.range_len,
            max_start,
        ));
    }
    let zipf = ZipfianSampler::new(pool.len(), cfg.alpha)?;
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0xA5A5_1234);
    for _ in 0..cfg.count {
        let rank = zipf.sample(&mut rng);
        let (s, e) = pool[rank];
        stream.push(s, e, 0)?;
    }
    Ok(())
}

fn emit_zipf_corr(
    stream: &mut QueryStreamWriter,
    sorted_keys: &[u64],
    cfg: &QueriesCliConfig,
    max_start: u64,
) -> Result<(), String> {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let hot_domain = sorted_keys.len().min(ZIPF_POOL_SIZE);
    let zipf = ZipfianSampler::new(hot_domain, cfg.alpha)?;
    let rank_queries = build_zipf_corr_anchors(
        &mut rng,
        sorted_keys,
        hot_domain,
        cfg.range_len,
        cfg.corr_d,
        max_start,
    );

    for _ in 0..cfg.count {
        let draw: f64 = rng.gen();
        let (s, e) = if draw < cfg.corr_d {
            let rank = zipf.sample(&mut rng);
            rank_queries[rank]
        } else {
            sample_uniform_empty(&mut rng, sorted_keys, cfg.range_len, max_start)
        };
        stream.push(s, e, 0)?;
    }
    Ok(())
}

fn build_zipf_corr_anchors(
    rng: &mut ChaCha8Rng,
    sorted_keys: &[u64],
    hot_domain: usize,
    range_len: u64,
    corr_d: f64,
    max_start: u64,
) -> Vec<(u64, u64)> {
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
            if is_range_empty(sorted_keys, start, end) {
                chosen = Some((start, end));
                break;
            }
        }
        let q =
            chosen.unwrap_or_else(|| sample_uniform_empty(rng, sorted_keys, range_len, max_start));
        rank_queries.push(q);
    }
    rank_queries
}

fn emit_shifting_r(
    stream: &mut QueryStreamWriter,
    sorted_keys: &[u64],
    count: u64,
    seed: u64,
) -> Result<(), String> {
    let num_phases = SHIFTING_R_PHASES.len() as u64;
    let base = count / num_phases;
    let rem = count % num_phases;
    for (idx, &phase_r) in SHIFTING_R_PHASES.iter().enumerate() {
        let phase_count = base + u64::from((idx as u64) < rem);
        let phase_seed = seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = ChaCha8Rng::seed_from_u64(phase_seed);
        let phase_max_start = KEYSPACE_MASK.saturating_sub(phase_r - 1);
        for _ in 0..phase_count {
            let (s, e) = sample_uniform_empty(&mut rng, sorted_keys, phase_r, phase_max_start);
            stream.push(s, e, 0)?;
        }
    }
    Ok(())
}
