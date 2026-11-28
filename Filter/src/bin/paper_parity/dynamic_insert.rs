//! `dynamic_insert` benchmark mode: incremental per-key insertion + read-only pass.

use crate::cli::CliArgs;
use crate::io::QueryRecord;
use crate::metrics::{
    assemble_result, compute_p50_p99, ModeOutcome, ParityResult, MAX_LATENCY_SAMPLES,
};
use crate::probe::{run_readonly_query_pass, verify_all_keys_zero_fn};
use aeris_core::AerisFilter;
use aeris_types::{FilterConfig, Key, RangeFilter};
use std::time::Instant;

/// Executes `dynamic_insert` mode (incremental per-key `insert()` + optional query pass).
///
/// # Errors
/// Returns an error if filter creation or any `insert()` call fails.
pub fn run_dynamic_insert(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let mut filter = AerisFilter::new(cfg)
        .map_err(|e| format!("AerisFilter::new failed in dynamic_insert: {e}"))?;

    let (insert_keys_per_sec, ins_s, ins_p50_ns, ins_p99_ns) =
        measure_incremental_inserts(&mut filter, keys)?;

    let pass = run_readonly_query_pass(&filter, queries);
    let fn_count = pass.fn_count + verify_all_keys_zero_fn(&filter, keys);
    let notes = format!(
        "mode=dynamic_insert;insert_ms={:.3};query_p50_ns={};query_p99_ns={};verified_keys={}",
        ins_s * 1000.0,
        pass.p50_ns,
        pass.p99_ns,
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
            bulk_load_keys_per_sec: 0.0,
            insert_keys_per_sec,
            p50_ns: ins_p50_ns,
            p99_ns: ins_p99_ns,
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

#[inline]
fn measure_incremental_inserts(
    filter: &mut AerisFilter,
    keys: &[Key],
) -> Result<(f64, f64, u64, u64), String> {
    let stride = (keys.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut insert_samples = Vec::with_capacity(keys.len().min(MAX_LATENCY_SAMPLES) + 1);

    let t_ins = Instant::now();
    for (idx, &k) in keys.iter().enumerate() {
        let t0 = if (idx % stride) == 0 {
            Some(Instant::now())
        } else {
            None
        };
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
    Ok((insert_keys_per_sec, ins_s, ins_p50_ns, ins_p99_ns))
}
