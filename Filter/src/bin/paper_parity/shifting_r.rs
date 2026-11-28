//! `shifting_r` benchmark mode: static-`m` vs Control-Plane dynamic-`m` reconfiguration.

use crate::cli::{CliArgs, ImplVariant};
use crate::io::QueryRecord;
use crate::metrics::{
    assemble_result, compute_fpr, compute_p50_p99, ModeOutcome, ParityResult, MAX_LATENCY_SAMPLES,
};
use crate::probe::{is_range_empty, probe_filter, verify_all_keys_zero_fn};
use crate::reverse_map::InMemReverseMap;
use aeris_control::{
    DiscreteCostOptimizer, EstimatorConfig, FalsePositiveLog, FpLogConfig, SlidingWindowEstimator,
};
use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, CostModelParams, CostOptimizer, FilterConfig, Key, RangeFilter, WorkloadEstimator,
};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

/// Control-plane optimizer evaluation cadence (queries) in `shifting_r`.
const OPTIMIZER_EVAL_INTERVAL: usize = 256;

/// RAII cleanup guard for temporary `FalsePositiveLog` files created during `shifting_r`.
struct TempFileGuard {
    path: PathBuf,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let mut tmp_os = self.path.as_os_str().to_os_string();
        tmp_os.push(".compact.tmp");
        let _ = fs::remove_file(PathBuf::from(tmp_os));
    }
}

/// Per-phase telemetry collected during `shifting_r`.
#[derive(Clone, Debug)]
struct PhaseMetrics {
    phase_idx: usize,
    range_len: u64,
    queries: usize,
    fp_count: u64,
    pure_query_s: f64,
    wall_s: f64,
    p99_ns: u64,
    end_m: u8,
}

/// Mutable benchmark state threaded across `shifting_r` phases.
struct ShiftingRState {
    filter: AerisFilter,
    rm: InMemReverseMap,
    sorted_keys: Vec<Key>,
    fp_log: Option<FalsePositiveLog>,
    estimator: Option<SlidingWindowEstimator>,
    optimizer: DiscreteCostOptimizer,
    cost_model: CostModelParams,
    is_optimized: bool,
    all_lat_samples: Vec<u64>,
    seen_fps: HashSet<(u64, u64)>,
    fp_count: u64,
    fn_count: u64,
    empty_count: u64,
    repeated_fp_count: u64,
    adaptations: u64,
    adapt_skips_full: u64,
    rebuild_count: u64,
    total_rebuild_s: f64,
    total_replay_us: u64,
    total_replayed_entries: usize,
    total_replay_resolved: usize,
    m_trajectory: Vec<u8>,
    global_idx: usize,
}

/// Identifies contiguous `(start_idx, end_idx, range_len)` phases in `queries`.
pub fn detect_query_phases(queries: &[QueryRecord]) -> Vec<(usize, usize, u64)> {
    if queries.is_empty() {
        return Vec::new();
    }
    let mut phases = Vec::new();
    let mut start = 0usize;
    let mut cur_len = queries[0].range.len_saturated();
    for (idx, qr) in queries.iter().enumerate().skip(1) {
        let len = qr.range.len_saturated();
        if len != cur_len {
            phases.push((start, idx, cur_len));
            start = idx;
            cur_len = len;
        }
    }
    phases.push((start, queries.len(), cur_len));
    phases
}

/// Executes `shifting_r` mode for either `rust_naive` (static `m`) or `rust_optimized`
/// (Control Plane `SlidingWindowEstimator` + `DiscreteCostOptimizer` + `FalsePositiveLog` replay).
///
/// # Errors
/// Returns an error if bulk-loading, FP-log I/O, estimator setup, `adapt()`, or
/// dynamic rebuild/replay fails.
pub fn run_shifting_r(
    args: &CliArgs,
    cfg: FilterConfig,
    keys: &[Key],
    queries: &[QueryRecord],
) -> Result<ParityResult, String> {
    let rm = InMemReverseMap::with_capacity(keys.len());
    let t_load = Instant::now();
    let filter = AerisFilter::bulk_load(cfg, keys, &rm)
        .map_err(|e| format!("bulk_load failed in shifting_r: {e}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let bulk_load_keys_per_sec = (keys.len() as f64) / load_s.max(1e-12);

    let is_optimized = args.impl_variant == ImplVariant::RustOptimized && !args.static_bypass;
    let phases = detect_query_phases(queries);
    let (_fp_log_guard, mut state) =
        init_shifting_r_state(args, filter, rm, keys, queries.len(), is_optimized)?;

    let mut phase_metrics = Vec::with_capacity(phases.len());
    let stride = (queries.len() / MAX_LATENCY_SAMPLES).max(1);
    let t_total = Instant::now();

    for (phase_idx, &(p_start, p_end, phase_r)) in phases.iter().enumerate() {
        let pm = run_shifting_phase(
            &mut state,
            keys,
            &queries[p_start..p_end],
            phase_idx,
            phases.len(),
            phase_r,
            stride,
        )?;
        phase_metrics.push(pm);
    }

    let total_wall_s = t_total.elapsed().as_secs_f64();
    Ok(finalize_shifting_r_result(
        args,
        &state,
        &phase_metrics,
        keys,
        queries.len(),
        bulk_load_keys_per_sec,
        total_wall_s,
    ))
}

fn init_shifting_r_state(
    args: &CliArgs,
    filter: AerisFilter,
    rm: InMemReverseMap,
    keys: &[Key],
    query_count: usize,
    is_optimized: bool,
) -> Result<(TempFileGuard, ShiftingRState), String> {
    let sorted_keys: Vec<Key> = if is_optimized {
        let mut s = keys.to_vec();
        s.sort_unstable();
        s
    } else {
        Vec::new()
    };

    let tmp_fp_path = env::temp_dir().join(format!(
        "aeris_paper_parity_fplog_{}_{}.fplog",
        std::process::id(),
        args.seed
    ));
    let fp_log_guard = TempFileGuard::new(tmp_fp_path.clone());
    let fp_log = if is_optimized {
        let _ = fs::remove_file(&tmp_fp_path);
        Some(
            FalsePositiveLog::open(
                &tmp_fp_path,
                FpLogConfig {
                    max_entries: 262_144,
                    sync_on_append: false,
                },
            )
            .map_err(|e| format!("failed to open FalsePositiveLog: {e}"))?,
        )
    } else {
        None
    };

    let estimator = if args.impl_variant == ImplVariant::RustOptimized {
        let est_cfg = if args.static_bypass {
            EstimatorConfig::default().with_enabled(false)
        } else {
            EstimatorConfig::default()
        };
        Some(
            SlidingWindowEstimator::with_initial_span(est_cfg, filter.config().max_range_r)
                .map_err(|e| format!("failed to initialize SlidingWindowEstimator: {e}"))?,
        )
    } else {
        None
    };

    let initial_m = filter.config().m;
    let state = ShiftingRState {
        filter,
        rm,
        sorted_keys,
        fp_log,
        estimator,
        optimizer: DiscreteCostOptimizer::new(),
        cost_model: CostModelParams::default(),
        is_optimized,
        all_lat_samples: Vec::with_capacity(query_count.min(MAX_LATENCY_SAMPLES) + 1),
        seen_fps: HashSet::new(),
        fp_count: 0,
        fn_count: 0,
        empty_count: 0,
        repeated_fp_count: 0,
        adaptations: 0,
        adapt_skips_full: 0,
        rebuild_count: 0,
        total_rebuild_s: 0.0,
        total_replay_us: 0,
        total_replayed_entries: 0,
        total_replay_resolved: 0,
        m_trajectory: vec![initial_m],
        global_idx: 0,
    };

    Ok((fp_log_guard, state))
}

#[inline]
fn run_shifting_phase(
    state: &mut ShiftingRState,
    keys: &[Key],
    phase_slice: &[QueryRecord],
    phase_idx: usize,
    total_phases: usize,
    phase_r: u64,
    stride: usize,
) -> Result<PhaseMetrics, String> {
    let phase_stride =
        (phase_slice.len() / (MAX_LATENCY_SAMPLES / total_phases.max(1)).max(1)).max(1);
    let mut phase_lat = Vec::with_capacity(phase_slice.len().min(MAX_LATENCY_SAMPLES) + 1);
    let mut phase_fps = 0u64;
    let mut phase_rebuild_s = 0.0f64;
    let t_phase = Instant::now();

    for (local_idx, qr) in phase_slice.iter().enumerate() {
        let sample_global = state.global_idx.is_multiple_of(stride);
        let sample_phase = local_idx.is_multiple_of(phase_stride);
        let t0 = if sample_global || sample_phase {
            Some(Instant::now())
        } else {
            None
        };

        let hit = probe_filter(&state.filter, qr.range);
        let is_fp = handle_shifting_probe_result(state, qr, hit, &mut phase_fps)?;

        if let Some(ref mut est) = state.estimator {
            est.record_query(qr.range, is_fp);
        }

        if let Some(start_ts) = t0 {
            let dt = start_ts.elapsed().as_nanos() as u64;
            if sample_global {
                state.all_lat_samples.push(dt);
            }
            if sample_phase {
                phase_lat.push(dt);
            }
        }

        state.global_idx += 1;

        if state.is_optimized && state.global_idx.is_multiple_of(OPTIMIZER_EVAL_INTERVAL) {
            phase_rebuild_s += maybe_rebuild_filter(state, keys, phase_r)?;
        }
    }

    let phase_wall_s = t_phase.elapsed().as_secs_f64();
    let phase_pure_s = (phase_wall_s - phase_rebuild_s).max(1e-12);
    let (_, phase_p99) = compute_p50_p99(phase_lat);
    Ok(PhaseMetrics {
        phase_idx,
        range_len: phase_r,
        queries: phase_slice.len(),
        fp_count: phase_fps,
        pure_query_s: phase_pure_s,
        wall_s: phase_wall_s,
        p99_ns: phase_p99,
        end_m: state.filter.config().m,
    })
}

#[inline]
fn handle_shifting_probe_result(
    state: &mut ShiftingRState,
    qr: &QueryRecord,
    hit: bool,
    phase_fps: &mut u64,
) -> Result<bool, String> {
    if qr.expected != 0 {
        if !hit {
            state.fn_count += 1;
        }
        return Ok(false);
    }

    state.empty_count += 1;
    if !hit {
        return Ok(false);
    }

    state.fp_count += 1;
    *phase_fps += 1;
    if !state.seen_fps.insert((qr.range.start(), qr.range.end())) {
        state.repeated_fp_count += 1;
    }
    if let Some(ref log) = state.fp_log {
        log.record(qr.range, state.global_idx as u64)
            .map_err(|e| format!("FalsePositiveLog::record failed: {e}"))?;
    }
    match state.filter.adapt(qr.range, &state.rm) {
        Ok(_) => state.adaptations += 1,
        Err(AerisError::FilterFull { .. }) => state.adapt_skips_full += 1,
        Err(err) => return Err(format!("adapt failed at query {}: {err}", state.global_idx)),
    }
    Ok(true)
}

#[cold]
fn maybe_rebuild_filter(
    state: &mut ShiftingRState,
    keys: &[Key],
    phase_r: u64,
) -> Result<f64, String> {
    let stats = state
        .estimator
        .as_ref()
        .expect("estimator present")
        .snapshot();
    let decision = state
        .optimizer
        .optimize(&stats, state.filter.config(), &state.cost_model);
    if !decision.should_rebuild || decision.recommended_m == state.filter.config().m {
        return Ok(0.0);
    }

    let t_reb = Instant::now();
    let new_cfg = state
        .filter
        .config()
        .reconfigure_for_m(decision.recommended_m)
        .map_err(|e| format!("reconfigure_for_m failed: {e}"))?;
    let new_rm = InMemReverseMap::with_capacity(keys.len());
    let mut new_filter = AerisFilter::bulk_load(new_cfg, keys, &new_rm)
        .map_err(|e| format!("rebuild bulk_load failed: {e}"))?;

    let log_ref = state.fp_log.as_ref().expect("fp_log present");
    log_ref
        .sync()
        .map_err(|e| format!("fp_log sync failed: {e}"))?;
    let t_rep = Instant::now();
    let sorted_keys = &state.sorted_keys;
    let rep = log_ref
        .replay_onto(&mut new_filter, &new_rm, |r| {
            Ok(is_range_empty(sorted_keys, r.start(), r.end()))
        })
        .map_err(|e| format!("FalsePositiveLog::replay_onto failed: {e}"))?;
    let replay_us = t_rep.elapsed().as_micros() as u64;
    let reb_s = t_reb.elapsed().as_secs_f64();

    info!(
        old_m = state.filter.config().m,
        new_m = new_filter.config().m,
        at_query = state.global_idx,
        phase_r,
        rebuild_ms = reb_s * 1000.0,
        replay_us,
        replayed = rep.replayed_entries,
        resolved = rep.collisions_resolved,
        "completed dynamic-R rebuild and monotonic FP replay"
    );

    state.filter = new_filter;
    state.rm = new_rm;
    state.rebuild_count += 1;
    state.total_rebuild_s += reb_s;
    state.total_replay_us += replay_us;
    state.total_replayed_entries += rep.replayed_entries;
    state.total_replay_resolved += rep.collisions_resolved;
    state.m_trajectory.push(state.filter.config().m);
    Ok(reb_s)
}

fn finalize_shifting_r_result(
    args: &CliArgs,
    state: &ShiftingRState,
    phase_metrics: &[PhaseMetrics],
    keys: &[Key],
    query_count: usize,
    bulk_load_keys_per_sec: f64,
    total_wall_s: f64,
) -> ParityResult {
    let pure_query_s = (total_wall_s - state.total_rebuild_s).max(1e-12);
    let wall_qps = (query_count as f64) / total_wall_s.max(1e-12);
    let pure_query_qps = (query_count as f64) / pure_query_s;
    let (p50_ns, p99_ns) = compute_p50_p99(state.all_lat_samples.clone());
    let fn_count = state.fn_count + verify_all_keys_zero_fn(&state.filter, keys);
    let fpr = compute_fpr(state.fp_count, state.empty_count);
    let notes = format_shifting_r_notes(state, phase_metrics, pure_query_qps, wall_qps);

    assemble_result(
        args,
        ModeOutcome {
            cfg: state.filter.config().clone(),
            load_factor: state.filter.load_factor(),
            keys: keys.len(),
            queries: query_count,
            qps: wall_qps,
            bulk_load_keys_per_sec,
            insert_keys_per_sec: 0.0,
            p50_ns,
            p99_ns,
            fpr,
            fp_count: state.fp_count,
            fn_count,
            repeated_fp_count: state.repeated_fp_count,
            adaptations: state.adaptations,
            filter_bytes: state.filter.memory_usage_bytes(),
            notes,
        },
    )
}

fn format_shifting_r_notes(
    state: &ShiftingRState,
    phase_metrics: &[PhaseMetrics],
    pure_query_qps: f64,
    wall_qps: f64,
) -> String {
    let m_traj_str = state
        .m_trajectory
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("->");
    let phase_str = phase_metrics
        .iter()
        .map(|pm| {
            let p_fpr = compute_fpr(pm.fp_count, pm.queries as u64);
            let p_pure_qps = (pm.queries as f64) / pm.pure_query_s.max(1e-12);
            let p_wall_qps = (pm.queries as f64) / pm.wall_s.max(1e-12);
            format!(
                "P{}(R={},m={},pure_qps={:.0},wall_qps={:.0},fpr={:.6},p99_ns={})",
                pm.phase_idx, pm.range_len, pm.end_m, p_pure_qps, p_wall_qps, p_fpr, pm.p99_ns
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "mode=shifting_r;pure_query_qps={pure_query_qps:.2};wall_qps={wall_qps:.2};rebuilds={};total_rebuild_ms={:.3};total_replay_us={};replayed_entries={};replay_resolved={};m_trajectory=[{m_traj_str}];adapt_skips_full={};phases=[{phase_str}]",
        state.rebuild_count,
        state.total_rebuild_s * 1000.0,
        state.total_replay_us,
        state.total_replayed_entries,
        state.total_replay_resolved,
        state.adapt_skips_full,
    )
}
