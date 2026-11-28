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

#![forbid(unsafe_code)]

#[path = "trace_gen/cli.rs"]
mod cli;
#[path = "trace_gen/io.rs"]
mod io;
#[path = "trace_gen/keys.rs"]
mod keys;
#[path = "trace_gen/queries.rs"]
mod queries;
#[cfg(test)]
#[path = "trace_gen/tests.rs"]
mod tests;

use crate::cli::{parse_args, Command};
use crate::keys::generate_keys_file;
use crate::queries::generate_queries_file;
use serde::Serialize;
use std::env;
use std::io::{self as stdio, Write};
use std::process::ExitCode;
use std::time::Instant;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

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

fn init_stderr_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(stdio::stderr)
        .with_target(false)
        .try_init();
}

fn execute_command(cmd: Command, start: Instant) -> Result<TraceGenSummary, String> {
    match cmd {
        Command::Keys { count, seed, out } => {
            info!(count, seed, out = %out.display(), "generating binary keys trace");
            let bytes_written = generate_keys_file(count, seed, &out)?;
            Ok(TraceGenSummary {
                command: "keys",
                kind: "uniform_48bit_feistel".to_string(),
                count,
                max_range_r: 1,
                seed,
                out: out.display().to_string(),
                bytes_written,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            })
        }
        Command::Queries(cfg) => {
            info!(
                keys = %cfg.keys_path.display(),
                kind = cfg.kind.as_str(),
                count = cfg.count,
                range_len = cfg.range_len,
                alpha = cfg.alpha,
                corr_d = cfg.corr_d,
                seed = cfg.seed,
                out = %cfg.out.display(),
                "generating binary queries trace"
            );
            let (bytes_written, max_range_r) = generate_queries_file(&cfg)?;
            Ok(TraceGenSummary {
                command: "queries",
                kind: cfg.kind.as_str().to_string(),
                count: cfg.count,
                max_range_r,
                seed: cfg.seed,
                out: cfg.out.display().to_string(),
                bytes_written,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            })
        }
    }
}

fn run_cli() -> Result<ExitCode, String> {
    let args: Vec<String> = env::args().collect();
    let cmd = parse_args(&args)?;
    let start = Instant::now();
    let summary = execute_command(cmd, start)?;

    info!(
        command = summary.command,
        kind = %summary.kind,
        count = summary.count,
        bytes_written = summary.bytes_written,
        elapsed_ms = summary.elapsed_ms,
        "trace generation complete"
    );

    let line = serde_json::to_string(&summary)
        .map_err(|e| format!("failed to serialize summary JSON: {e}"))?;
    let mut stdout = stdio::stdout().lock();
    writeln!(stdout, "{line}").map_err(|e| format!("failed to write stdout summary: {e}"))?;
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    init_stderr_tracing();
    match run_cli() {
        Ok(code) => code,
        Err(err) => {
            error!(error = %err, "trace_gen failed");
            ExitCode::FAILURE
        }
    }
}
