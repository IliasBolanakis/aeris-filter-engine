//! `bulk_load` benchmark mode: pure filter bulk-load with [`NoopReverseMap`].

use crate::cli::CliArgs;
use crate::io::QueryRecord;
use crate::metrics::{assemble_result, ModeOutcome, ParityResult};
use crate::probe::{run_readonly_query_pass, verify_all_keys_zero_fn};
use crate::reverse_map::NoopReverseMap;
use aeris_core::AerisFilter;
use aeris_types::{FilterConfig, Key, RangeFilter};
use std::time::Instant;

/// Executes `bulk_load` mode (pure filter scaling with [`NoopReverseMap`]).
///
/// # Errors
/// Returns an error if `AerisFilter::bulk_load` fails.
pub fn run_bulk_load(
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

    let pass = run_readonly_query_pass(&filter, queries);
    let fn_count = pass.fn_count + verify_all_keys_zero_fn(&filter, keys);
    let notes = format!(
        "mode=bulk_load;reverse_map=noop;load_ms={:.3};verified_keys={}",
        load_s * 1000.0,
        keys.len()
    );

    Ok(assemble_result(
        args,
        ModeOutcome {
            cfg: filter.config().clone(),
            load_factor: filter.load_factor(),
            keys: keys.len(),
            queries: queries.len(),
            qps: pass.qps,
            bulk_load_keys_per_sec,
            insert_keys_per_sec: 0.0,
            p50_ns: pass.p50_ns,
            p99_ns: pass.p99_ns,
            fpr: pass.fpr,
            fp_count: pass.fp_count,
            fn_count,
            repeated_fp_count: 0,
            adaptations: 0,
            filter_bytes: filter.memory_usage_bytes(),
            notes,
        },
    ))
}
