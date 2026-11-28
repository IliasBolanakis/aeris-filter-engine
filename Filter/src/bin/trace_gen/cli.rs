//! CLI subcommand and flag parsing for `trace_gen`.

use std::path::PathBuf;

const DEFAULT_SEED: u64 = 1380;
const DEFAULT_RANGE_LEN: u64 = 1;
const DEFAULT_ALPHA: f64 = 1.5;
const DEFAULT_CORR_D: f64 = 0.8;

/// Supported query workload distributions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryKind {
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
    ///
    /// # Errors
    /// Returns a descriptive error string when `raw` does not match a known kind.
    pub fn parse(raw: &str) -> Result<Self, String> {
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
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ZipfCorr => "zipf_corr",
            Self::ZipfUncorr => "zipf_uncorr",
            Self::UniformEmpty => "uniform_empty",
            Self::ShiftingR => "shifting_r",
        }
    }
}

/// Configuration for `trace_gen queries` subcommand.
#[derive(Clone, Debug, PartialEq)]
pub struct QueriesCliConfig {
    /// Path to input `*.keys.bin` file.
    pub keys_path: PathBuf,
    /// Workload distribution kind.
    pub kind: QueryKind,
    /// Number of queries to emit.
    pub count: u64,
    /// Fixed range width `R` (ignored for `shifting_r` phase widths).
    pub range_len: u64,
    /// Zipfian skew exponent `alpha`.
    pub alpha: f64,
    /// Correlation probability `D in [0.0, 1.0]` for `zipf_corr`.
    pub corr_d: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Output `*.queries.bin` path.
    pub out: PathBuf,
}

/// Parsed CLI subcommand.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Generate a binary key trace file (`*.keys.bin`).
    Keys {
        /// Number of unique 48-bit keys to generate.
        count: u64,
        /// Deterministic Feistel permutation seed.
        seed: u64,
        /// Output `*.keys.bin` path.
        out: PathBuf,
    },
    /// Generate a verified-empty binary query trace file (`*.queries.bin`).
    Queries(QueriesCliConfig),
}

/// Parses CLI arguments into a strongly typed [`Command`].
///
/// # Errors
/// Returns usage text or flag diagnostics when subcommands, required flags, or
/// numeric values are missing or malformed.
pub fn parse_args(args: &[String]) -> Result<Command, String> {
    if args.len() < 2 {
        return Err(usage_string());
    }
    match args[1].as_str() {
        "keys" => parse_keys_args(&args[2..]),
        "queries" => parse_queries_args(&args[2..]),
        other => Err(format!("unknown subcommand '{other}'\n{}", usage_string())),
    }
}

fn parse_keys_args(tail: &[String]) -> Result<Command, String> {
    let mut count: Option<u64> = None;
    let mut seed: u64 = DEFAULT_SEED;
    let mut out: Option<PathBuf> = None;
    let mut i = 0usize;

    while i < tail.len() {
        let flag = tail[i].as_str();
        let val = tail
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--count" => count = Some(parse_typed(flag, val)?),
            "--seed" => seed = parse_typed(flag, val)?,
            "--out" => out = Some(PathBuf::from(val)),
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

fn parse_queries_args(tail: &[String]) -> Result<Command, String> {
    let mut keys_path: Option<PathBuf> = None;
    let mut kind: Option<QueryKind> = None;
    let mut count: Option<u64> = None;
    let mut range_len: u64 = DEFAULT_RANGE_LEN;
    let mut alpha: f64 = DEFAULT_ALPHA;
    let mut corr_d: f64 = DEFAULT_CORR_D;
    let mut seed: u64 = DEFAULT_SEED;
    let mut out: Option<PathBuf> = None;
    let mut i = 0usize;

    while i < tail.len() {
        let flag = tail[i].as_str();
        let val = tail
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--keys" => keys_path = Some(PathBuf::from(val)),
            "--kind" => kind = Some(QueryKind::parse(val)?),
            "--count" => count = Some(parse_typed(flag, val)?),
            "--range-len" => range_len = parse_typed(flag, val)?,
            "--alpha" => alpha = parse_typed(flag, val)?,
            "--corr-d" => corr_d = parse_typed(flag, val)?,
            "--seed" => seed = parse_typed(flag, val)?,
            "--out" => out = Some(PathBuf::from(val)),
            other => return Err(format!("unknown flag '{other}' for `queries`")),
        }
        i += 2;
    }

    Ok(Command::Queries(QueriesCliConfig {
        keys_path: keys_path.ok_or_else(|| "missing required --keys".to_string())?,
        kind: kind.ok_or_else(|| "missing required --kind".to_string())?,
        count: count.ok_or_else(|| "missing required --count".to_string())?,
        range_len,
        alpha,
        corr_d,
        seed,
        out: out.ok_or_else(|| "missing required --out".to_string())?,
    }))
}

fn parse_typed<T: std::str::FromStr>(flag: &str, val: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    val.parse::<T>()
        .map_err(|e| format!("invalid {flag} '{val}': {e}"))
}

/// Returns the CLI usage instructions string.
pub fn usage_string() -> String {
    concat!(
        "Usage:\n",
        "  trace_gen keys --count <N> --seed <u64> --out <path>\n",
        "  trace_gen queries --keys <keys.bin> --kind <zipf_corr|zipf_uncorr|uniform_empty|shifting_r> \\\n",
        "    --count <Q> --range-len <R> --alpha <1.5> --corr-d <0.8> --seed <u64> --out <path>"
    )
    .to_string()
}
