//! CLI argument parsing and filter configuration validation for `paper_parity`.
//!
//! Preserves the shared C++/Rust paper-parity flag surface (`--mode`, `--impl`,
//! `--scenario`, `--keys`, `--queries`, `--q`, `--r`, `--m`, `--budget-c`,
//! `--target-epsilon`, `--seed`, `--static-bypass`, `--out`).

use aeris_types::FilterConfig;
use std::path::PathBuf;

/// Default target false-positive epsilon (`r_min = 7`).
pub const DEFAULT_TARGET_EPSILON: f64 = 0.0625;

/// Default deterministic hash seed.
pub const DEFAULT_SEED: u64 = 1380;

/// Supported benchmark execution modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchMode {
    /// Bulk-load with in-memory reverse map and adapt monotonically on every FP.
    QueryAdapt,
    /// Bulk-load with no-op reverse map and run read-only query throughput pass.
    BulkLoad,
    /// Incremental per-key insertion with strided latency sampling.
    DynamicInsert,
    /// Deterministic 10% repeat-FP adversarial schedule (`fp_pool` cap 256).
    Adversarial,
    /// Multi-phase shifting range length comparing static-`m` vs dynamic-`m`.
    ShiftingR,
}

impl BenchMode {
    /// Parses a `--mode` flag value into a [`BenchMode`].
    ///
    /// # Errors
    /// Returns a descriptive error string if `raw` is not one of the supported modes.
    pub fn parse(raw: &str) -> Result<Self, String> {
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

    /// Returns `true` when the benchmark mode requires `--queries <path>`.
    #[inline]
    pub const fn requires_queries(self) -> bool {
        matches!(self, Self::QueryAdapt | Self::Adversarial | Self::ShiftingR)
    }
}

/// Supported Rust driver implementation variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplVariant {
    /// Static-`m` baseline configuration.
    RustNaive,
    /// Control-Plane dynamic-`m` reconfiguration variant.
    RustOptimized,
}

impl ImplVariant {
    /// Parses an `--impl` flag value into an [`ImplVariant`].
    ///
    /// # Errors
    /// Returns a descriptive error string if `raw` is unrecognized.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "rust_naive" => Ok(Self::RustNaive),
            "rust_optimized" => Ok(Self::RustOptimized),
            other => Err(format!(
                "unsupported --impl '{other}': expected rust_naive | rust_optimized"
            )),
        }
    }

    /// Returns the canonical string identifier emitted in `ParityResult.impl`.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RustNaive => "rust_naive",
            Self::RustOptimized => "rust_optimized",
        }
    }
}

/// Parsed CLI configuration for `paper_parity`.
#[derive(Clone, Debug)]
pub struct CliArgs {
    /// Selected benchmark mode.
    pub mode: BenchMode,
    /// Selected driver implementation variant.
    pub impl_variant: ImplVariant,
    /// Scenario label recorded verbatim in output JSON.
    pub scenario: String,
    /// Input binary key trace (`*.keys.bin`).
    pub keys_path: PathBuf,
    /// Optional input binary query trace (`*.queries.bin`).
    pub queries_path: Option<PathBuf>,
    /// Quotient bit width `q`.
    pub q: u8,
    /// Remainder bit width `r`.
    pub r: u8,
    /// Range-extension bit width `m`.
    pub m: u8,
    /// Optional explicit payload budget `C` (`r + m` by default).
    pub budget_c: Option<u8>,
    /// Optional target false-positive bound `epsilon`.
    pub target_epsilon: Option<f64>,
    /// Deterministic hash seed.
    pub seed: u64,
    /// Disables dynamic control-plane reconfiguration even under `rust_optimized`.
    pub static_bypass: bool,
    /// Optional path for pretty-printed JSON output.
    pub out_path: Option<PathBuf>,
}

/// Intermediate accumulator for optional and required CLI flags.
#[derive(Default)]
struct RawCliBuilder {
    mode: Option<BenchMode>,
    impl_variant: Option<ImplVariant>,
    scenario: Option<String>,
    keys_path: Option<PathBuf>,
    queries_path: Option<PathBuf>,
    q: Option<u8>,
    r: Option<u8>,
    m: Option<u8>,
    budget_c: Option<u8>,
    target_epsilon: Option<f64>,
    seed: Option<u64>,
    static_bypass: bool,
    out_path: Option<PathBuf>,
}

impl RawCliBuilder {
    fn apply_flag(&mut self, flag: &str, val: &str) -> Result<(), String> {
        match flag {
            "--mode" => self.mode = Some(BenchMode::parse(val)?),
            "--impl" => self.impl_variant = Some(ImplVariant::parse(val)?),
            "--scenario" => self.scenario = Some(val.to_string()),
            "--keys" => self.keys_path = Some(PathBuf::from(val)),
            "--queries" => self.queries_path = Some(PathBuf::from(val)),
            "--q" => self.q = Some(parse_typed(flag, val)?),
            "--r" => self.r = Some(parse_typed(flag, val)?),
            "--m" => self.m = Some(parse_typed(flag, val)?),
            "--budget-c" => self.budget_c = Some(parse_typed(flag, val)?),
            "--target-epsilon" => self.target_epsilon = Some(parse_typed(flag, val)?),
            "--seed" => self.seed = Some(parse_typed(flag, val)?),
            "--out" => self.out_path = Some(PathBuf::from(val)),
            other => return Err(format!("unknown flag '{other}'\n{}", usage_string())),
        }
        Ok(())
    }

    fn build(self) -> Result<CliArgs, String> {
        let mode = self
            .mode
            .ok_or_else(|| "missing required --mode".to_string())?;
        let impl_variant = self
            .impl_variant
            .ok_or_else(|| "missing required --impl".to_string())?;
        let scenario = self
            .scenario
            .ok_or_else(|| "missing required --scenario".to_string())?;
        let keys_path = self
            .keys_path
            .ok_or_else(|| "missing required --keys".to_string())?;
        let q = self.q.ok_or_else(|| "missing required --q".to_string())?;
        let r = self.r.ok_or_else(|| "missing required --r".to_string())?;
        let m = self.m.ok_or_else(|| "missing required --m".to_string())?;

        if mode.requires_queries() && self.queries_path.is_none() {
            return Err(format!("mode {mode:?} requires --queries <path>"));
        }

        Ok(CliArgs {
            mode,
            impl_variant,
            scenario,
            keys_path,
            queries_path: self.queries_path,
            q,
            r,
            m,
            budget_c: self.budget_c,
            target_epsilon: self.target_epsilon,
            seed: self.seed.unwrap_or(DEFAULT_SEED),
            static_bypass: self.static_bypass,
            out_path: self.out_path,
        })
    }
}

fn parse_typed<T: std::str::FromStr>(flag: &str, val: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    val.parse::<T>()
        .map_err(|e| format!("invalid {flag} '{val}': {e}"))
}

/// Parses CLI arguments matching the shared C++/Rust parity interface.
///
/// # Errors
/// Returns usage or validation diagnostics when required flags are missing,
/// unknown flags appear, or numeric values fail parsing.
pub fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    if args.len() < 2 {
        return Err(usage_string());
    }

    let mut builder = RawCliBuilder::default();
    let mut i = 1usize;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--static-bypass" {
            builder.static_bypass = true;
            i += 1;
            continue;
        }
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        builder.apply_flag(flag, val)?;
        i += 2;
    }

    builder.build()
}

/// Returns the formatted CLI usage string.
pub fn usage_string() -> String {
    concat!(
        "Usage: paper_parity --mode <query_adapt|bulk_load|dynamic_insert|adversarial|shifting_r> \\\n",
        "  --impl <rust_naive|rust_optimized> --scenario <name> --keys <path> [--queries <path>] \\\n",
        "  --q <u8> --r <u8> --m <u8> [--budget-c <u8>] [--target-epsilon <f64>] [--seed <u64>] [--out <path>]"
    )
    .to_string()
}

/// Resolves the validated [`FilterConfig`] from parsed CLI parameters.
///
/// Passes `max_range_r = 1u64 << m` so `FilterConfig::ceil_log2_u64(max_range_r) == m`
/// holds identically for point (`m = 0` or `m = 2`) and range (`m = 4..=10`) configs.
/// When `--target-epsilon` is omitted and `r < 7` (e.g., low-bit static paper curves),
/// relaxes `target_epsilon` to the tightest valid value satisfying `r >= r_min(eps)`.
///
/// # Errors
/// Returns an error if `m > 63`, `r + m` overflows `u8`, or `FilterConfig::new`
/// rejects the resulting geometry.
pub fn build_filter_config(args: &CliArgs) -> Result<FilterConfig, String> {
    if args.m > 63 {
        return Err(format!("--m ({}) must be <= 63", args.m));
    }
    let max_range_r = 1u64
        .checked_shl(u32::from(args.m))
        .ok_or_else(|| format!("--m ({}) overflows 1u64 << m", args.m))?;
    let budget_c = match args.budget_c {
        Some(c) => c,
        None => args.r.checked_add(args.m).ok_or_else(|| {
            format!(
                "r ({}) + m ({}) overflows u8 payload budget",
                args.r, args.m
            )
        })?,
    };

    let target_epsilon = resolve_target_epsilon(args.r, args.target_epsilon)?;

    FilterConfig::new(args.q, args.r, max_range_r, budget_c, target_epsilon)
        .and_then(|c| c.with_max_load_factor(0.95))
        .map(|c| c.with_hash_seed(args.seed))
        .map_err(|e| format!("invalid FilterConfig: {e}"))
}

fn resolve_target_epsilon(r: u8, explicit: Option<f64>) -> Result<f64, String> {
    if let Some(eps) = explicit {
        return Ok(eps);
    }
    let default_r_min =
        FilterConfig::min_remainder_bits(DEFAULT_TARGET_EPSILON).map_err(|e| e.to_string())?;
    if r >= default_r_min {
        Ok(DEFAULT_TARGET_EPSILON)
    } else if r >= 6 {
        Ok(0.125)
    } else if r >= 4 {
        Ok(0.25)
    } else {
        Ok(0.45)
    }
}
