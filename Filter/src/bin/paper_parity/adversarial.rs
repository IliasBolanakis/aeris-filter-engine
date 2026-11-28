//! `adversarial` benchmark mode: deterministic 10% repeat-FP probing schedule.

use crate::cli::CliArgs;
use crate::io::QueryRecord;
use crate::metrics::{
    assemble_result, compute_fpr, compute_p50_p99, ModeOutcome, ParityResult, MAX_LATENCY_SAMPLES,
};
use crate::probe::{probe_filter, verify_all_keys_zero_fn};
use crate::reverse_map::InMemReverseMap;
use aeris_core::AerisFilter;
use aeris_types::{AerisError, FilterConfig, Key, RangeFilter};
use std::collections::HashSet;
use std::time::Instant;

/// Maximum adversarial repeat-FP pool capacity (matches C++ driver).
const ADVERSARIAL_FP_POOL_CAP: usize = 256;

struct AdversarialCounters {
    lat_samples: Vec<u64>,
    fp_pool: Vec<QueryRecord>,
    fp_count: u64,
    fn_count: u64,
    empty_count: u64,
    repeated_fp_count: u64,
    adaptations: u64,
    adapt_skips_full: u64,
    adversarial_reprobes: u64,
}

/// Executes `adversarial` mode with deterministic 10% repeat-FP probing (`fp_pool` cap 256).
///
/// # Errors
/// Returns an error if bulk-loading fails or `adapt()` returns an error other
/// than `FilterFull`.
pub fn run_adversarial(
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

    let (counters, query_s) = execute_adversarial_loop(&mut filter, &rm, queries)?;
    let qps = (queries.len() as f64) / query_s.max(1e-12);
    let (p50_ns, p99_ns) = compute_p50_p99(counters.lat_samples);
    let fn_count = counters.fn_count + verify_all_keys_zero_fn(&filter, keys);
    let fpr = compute_fpr(counters.fp_count, counters.empty_count);
    let notes = format!(
        "mode=adversarial;fp_pool_size={};adversarial_reprobes={};adapt_skips_full={};verified_keys={}",
        counters.fp_pool.len(),
        counters.adversarial_reprobes,
        counters.adapt_skips_full,
        keys.len()
    );

    Ok(assemble_result(
        args,
        ModeOutcome {
            cfg: filter.config().clone(),
            load_factor: filter.load_factor(),
            keys: keys.len(),
            queries: queries.len(),
            qps,
            bulk_load_keys_per_sec,
            insert_keys_per_sec: 0.0,
            p50_ns,
            p99_ns,
            fpr,
            fp_count: counters.fp_count,
            fn_count,
            repeated_fp_count: counters.repeated_fp_count,
            adaptations: counters.adaptations,
            filter_bytes: filter.memory_usage_bytes(),
            notes,
        },
    ))
}

#[inline]
fn execute_adversarial_loop(
    filter: &mut AerisFilter,
    rm: &InMemReverseMap,
    queries: &[QueryRecord],
) -> Result<(AdversarialCounters, f64), String> {
    let stride = (queries.len() / MAX_LATENCY_SAMPLES).max(1);
    let mut seen_fps: HashSet<(u64, u64)> = HashSet::new();
    let mut c = AdversarialCounters {
        lat_samples: Vec::with_capacity(queries.len().min(MAX_LATENCY_SAMPLES) + 1),
        fp_pool: Vec::with_capacity(ADVERSARIAL_FP_POOL_CAP),
        fp_count: 0,
        fn_count: 0,
        empty_count: 0,
        repeated_fp_count: 0,
        adaptations: 0,
        adapt_skips_full: 0,
        adversarial_reprobes: 0,
    };

    let t_q = Instant::now();
    for (i, &base_qr) in queries.iter().enumerate() {
        let qr = if !c.fp_pool.is_empty() && (i % 10 == 0) {
            c.adversarial_reprobes += 1;
            c.fp_pool[(i / 10) % c.fp_pool.len()]
        } else {
            base_qr
        };

        let t0 = if (i % stride) == 0 {
            Some(Instant::now())
        } else {
            None
        };

        let hit = probe_filter(filter, qr.range);
        if qr.expected == 0 {
            c.empty_count += 1;
            if hit {
                c.fp_count += 1;
                let is_new = seen_fps.insert((qr.range.start(), qr.range.end()));
                if !is_new {
                    c.repeated_fp_count += 1;
                } else if c.fp_pool.len() < ADVERSARIAL_FP_POOL_CAP {
                    c.fp_pool.push(qr);
                }
                match filter.adapt(qr.range, rm) {
                    Ok(_) => c.adaptations += 1,
                    Err(AerisError::FilterFull { .. }) => c.adapt_skips_full += 1,
                    Err(err) => return Err(format!("adapt failed at adversarial step {i}: {err}")),
                }
            }
        } else if !hit {
            c.fn_count += 1;
        }

        if let Some(start_ts) = t0 {
            c.lat_samples.push(start_ts.elapsed().as_nanos() as u64);
        }
    }
    Ok((c, t_q.elapsed().as_secs_f64()))
}
