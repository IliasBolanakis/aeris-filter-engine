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
//!   with `SlidingWindowEstimator`, `DiscreteCostOptimizer`, and `FalsePositiveLog`
//!   replay (`--impl rust_optimized`).

#![forbid(unsafe_code)]

#[path = "paper_parity/adversarial.rs"]
mod adversarial;
#[path = "paper_parity/bulk_load.rs"]
mod bulk_load;
#[path = "paper_parity/cli.rs"]
mod cli;
#[path = "paper_parity/dynamic_insert.rs"]
mod dynamic_insert;
#[path = "paper_parity/io.rs"]
mod io;
#[path = "paper_parity/metrics.rs"]
mod metrics;
#[path = "paper_parity/probe.rs"]
mod probe;
#[path = "paper_parity/query_adapt.rs"]
mod query_adapt;
#[path = "paper_parity/reverse_map.rs"]
mod reverse_map;
#[path = "paper_parity/shifting_r.rs"]
mod shifting_r;
#[cfg(test)]
#[path = "paper_parity/tests.rs"]
mod tests;

use crate::adversarial::run_adversarial;
use crate::bulk_load::run_bulk_load;
use crate::cli::{build_filter_config, parse_args, BenchMode, CliArgs};
use crate::dynamic_insert::run_dynamic_insert;
use crate::io::{read_keys_file, read_queries_file, QueryRecord};
use crate::metrics::ParityResult;
use crate::query_adapt::run_query_adapt;
use crate::shifting_r::run_shifting_r;
use aeris_types::{FilterConfig, Key};
use std::env;
use std::fs;
use std::io::{self as stdio, Write};
use std::process::ExitCode;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

fn init_stderr_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(stdio::stderr)
        .with_target(false)
        .try_init();
}

fn dispatch_mode(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    match args.mode {
        BenchMode::QueryAdapt => run_query_adapt(args, cfg, keys, queries),
        BenchMode::BulkLoad => run_bulk_load(args, cfg, keys, queries),
        BenchMode::DynamicInsert => run_dynamic_insert(args, cfg, keys, queries),
        BenchMode::Adversarial => run_adversarial(args, cfg, keys, queries),
        BenchMode::ShiftingR => run_shifting_r(args, cfg, keys, queries),
    }
}

fn write_outputs(args: &CliArgs, res: &ParityResult) -> Result<(), String> {
    let json_line = serde_json::to_string(res)
        .map_err(|e| format!("failed to serialize ParityResult JSON: {e}"))?;

    if let Some(ref out_path) = args.out_path {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "failed to create --out directory '{}': {e}",
                        parent.display()
                    )
                })?;
            }
        }
        let pretty = serde_json::to_string_pretty(res).unwrap_or_else(|_| json_line.clone());
        fs::write(out_path, pretty).map_err(|e| {
            format!(
                "failed to write --out JSON file '{}': {e}",
                out_path.display()
            )
        })?;
    }

    let mut stdout = stdio::stdout().lock();
    writeln!(stdout, "{json_line}").map_err(|e| format!("failed to write stdout JSON: {e}"))
}

fn run_cli() -> Result<ExitCode, String> {
    let raw_args: Vec<String> = env::args().collect();
    let args = parse_args(&raw_args)?;
    let cfg = build_filter_config(&args)?;

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

    let keys = read_keys_file(&args.keys_path)?;
    let queries = match &args.queries_path {
        Some(q_path) => read_queries_file(q_path)?.0,
        None => Vec::new(),
    };

    let res = dispatch_mode(&args, cfg, &keys, &queries)?;
    write_outputs(&args, &res)?;

    if res.fn_count > 0 {
        error!(
            fn_count = res.fn_count,
            scenario = %res.scenario,
            "ZERO FALSE NEGATIVE INVARIANT VIOLATED"
        );
        return Ok(ExitCode::FAILURE);
    }

    if args.mode == BenchMode::Adversarial && res.repeated_fp_count > 0 {
        warn!(
            repeated_fp_count = res.repeated_fp_count,
            scenario = %res.scenario,
            "repeated false positives observed in adversarial mode"
        );
    }

    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    init_stderr_tracing();
    match run_cli() {
        Ok(code) => code,
        Err(err) => {
            error!(error = %err, "paper_parity execution failed");
            ExitCode::FAILURE
        }
    }
}
