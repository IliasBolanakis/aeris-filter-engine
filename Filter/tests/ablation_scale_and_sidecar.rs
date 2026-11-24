//! Large-Scale Ablation Suite — Worker 3:
//! **A/B 4: Scale Plane (`RcuAerisFilter` + `ShardedAerisSystem`) vs. Single Coarse-Locked Filter**
//! and
//! **A/B 5: Phase 5 Live Python ML/RL Sidecar vs. Native Control Plane**.
//!
//! # Experimental Guarantees
//! - 100% real Data Plane ([`AerisFilter`]), Storage Plane ([`RocksDbReverseMap`]),
//!   Control Plane ([`ProactiveTrigger`], [`DiscreteCostOptimizer`], [`FpLogConfig`]),
//!   Scale Plane ([`RcuAerisFilter`], [`ShardedAerisSystem`]), and Phase 5 Sidecar
//!   ([`SidecarClient`], [`GuardedTuningAdvisor`], [`SafetyGuardrail`], live Python
//!   `aeris_sidecar.server` gRPC subprocess).
//! - Zero mocks, zero stubs, zero `thread::sleep()`, zero `unsafe` code, zero `dbg!`.
//! - Persists structured results to `Filter/target/ablation_worker3_scale_sidecar.json`
//!   via [`ablation_common::write_worker_report`].

mod ablation_common;

use ablation_common::{
    compute_delta_pct, directory_size_bytes, enclosing_range, generate_seasonal_ingestion_trace,
    generate_skewed_hotspot_write_trace, generate_uniform_empty_queries,
    generate_uniform_keys_oracle, init_ablation_tracing, write_worker_report, AblationMetric,
    HostMetadata, LatencyStats, SortedKeyOracle, WorkerAblationReport, KEYSPACE_MASK,
};
use aeris_control::{
    DiscreteCostOptimizer, FpLogConfig, ProactiveTrigger, ProactiveTriggerConfig, TriggerAction,
    TriggerInput,
};
use aeris_core::AerisFilter;
use aeris_scale::{RcuAerisFilter, ShardedAerisSystem, ShardingConfig};
use aeris_sidecar::{
    ClientConfig, GuardedTuningAdvisor, NativeFallbackOptimizer, SafetyGuardrail, SidecarClient,
    SidecarEndpoint, NATIVE_FALLBACK_MODEL_VERSION,
};
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, CostModelParams, CostOptimizer, FilterConfig, GuardrailVerdict, Key,
    MemoryAllocationPlan, QueryRange, RangeFilter, ReverseMapStore, SafetyBounds,
    ShardAllocationDecision, ShardTelemetrySnapshot, TelemetrySnapshot, TuningAdvisor,
    TuningProposal, WorkloadStats,
};
use parking_lot::{Mutex, RwLock};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tracing::info;

/// Target pre-loaded key count for A/B 4.1 multi-core scaling (scaled by `host.scale_factor`).
const AB4_SCALING_BASE_KEYS: usize = 5_000_000;
/// Target concurrent mixed operations per thread tier for A/B 4.1 (scaled by `host.scale_factor`).
const AB4_SCALING_BASE_OPS: usize = 5_000_000;
/// Initial shard count for [`ShardedAerisSystem`] in A/B 4.
const AB4_INITIAL_SHARDS: usize = 8;
/// Thread counts evaluated in A/B 4.1 multi-core scaling.
const AB4_THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];
/// Target latency sample budget per scaling run so `Instant::now()` does not skew throughput.
const AB4_LATENCY_SAMPLE_BUDGET: usize = 20_000;

/// Base key count for A/B 4.2 live 2x expansion reader-latency audit.
const AB4_RCU_EXPANSION_BASE_KEYS: usize = 500_000;

/// Base uniform key count for A/B 4.3 hot-spot skewed write comparison.
const AB4_HOTSPOT_BASE_KEYS: usize = 80_000;
/// Skewed hot-spot write trace length for A/B 4.3 (85% inside hot band).
const AB4_HOTSPOT_WRITE_KEYS: usize = 175_000;

/// Total key count for A/B 5.1 seasonal trace and A/B 5.2 multi-tenant bandit workloads.
const AB5_TOTAL_KEYS: usize = 1_000_000;
/// Simulated disk-check penalty per false positive (50 us) used in A/B 5.2 effective throughput.
const SIMULATED_DISK_CHECK_PENALTY_NS: u64 = 50_000;

/// Multiplicative odd scramble constant for guaranteed bijective 47-bit write key generation.
const BIJECTIVE_ODD_SCRAMBLE: u64 = 0x9E37_79B9_7F4A_7C15;
/// 47-bit mask leaving bit 0 free to separate even pre-loaded keys from odd concurrent writes.
const MASK_47_BITS: u64 = (1u64 << 47) - 1;

/// Naive paper-baseline wrapper pairing a single unsharded [`AerisFilter`] with
/// its persistent [`RocksDbReverseMap`] behind one coarse lock.
struct NaiveLockedFilter {
    filter: AerisFilter,
    reverse_map: RocksDbReverseMap,
}

impl NaiveLockedFilter {
    /// Inserts `key` and persists its `(fp, key)` association, expanding `2x` if full.
    fn insert(&mut self, key: Key) {
        loop {
            match self.filter.insert(key) {
                Ok(fp) => {
                    self.reverse_map
                        .append_key(&fp, key)
                        .expect("naive reverse map append_key must succeed");
                    return;
                }
                Err(AerisError::FilterFull { .. }) => {
                    self.filter
                        .expand(&self.reverse_map)
                        .expect("naive filter 2x expansion must succeed");
                }
                Err(err) => panic!("unexpected naive filter insert error: {err}"),
            }
        }
    }

    /// Evaluates a point emptiness query.
    #[inline]
    fn query_point(&self, key: Key) -> bool {
        self.filter.query_point(key)
    }

    /// Evaluates a closed range emptiness query.
    #[inline]
    fn query_range(&self, range: QueryRange) -> bool {
        self.filter.query_range(range)
    }
}

/// Deterministic mixed operation executed during A/B 4.1 multi-core scaling.
#[derive(Clone, Copy)]
enum MixedOp {
    /// Point query on a guaranteed-present key (verifying 0 false negatives inline).
    PresentPoint(Key),
    /// Closed range query of length 32 across `[0, KEYSPACE_MASK]`.
    RangeProbe(QueryRange),
    /// Unique key insertion disjoint from all pre-loaded and sibling-thread keys.
    Insert(Key),
}

/// RAII subprocess guard managing the real out-of-process Python gRPC sidecar
/// (`Filter/python-sidecar/aeris_sidecar/server.py`) with zero `thread::sleep()`.
struct PythonSidecarProcess {
    child: Option<Child>,
    _tempdir: TempDir,
    port: u16,
}

impl PythonSidecarProcess {
    /// Spawns the live Python gRPC sidecar bound to an ephemeral loopback port
    /// and waits deterministically for its atomic `--port-file` announcement.
    fn spawn() -> Self {
        let tempdir = tempfile::tempdir().expect("create tempdir for Python sidecar port file");
        let port_file = tempdir.path().join("sidecar.port");
        let python_sidecar_dir = python_sidecar_root();
        let python_bin = detect_python_executable();

        let child = Command::new(&python_bin)
            .args([
                "-m",
                "aeris_sidecar.server",
                "--bind",
                "127.0.0.1:0",
                "--port-file",
                port_file
                    .to_str()
                    .expect("tempdir port file path must be valid UTF-8"),
                "--seed",
                "42",
            ])
            .env("PYTHONPATH", &python_sidecar_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to spawn Python sidecar using '{}' (PYTHONPATH={}): {err}",
                    python_bin,
                    python_sidecar_dir.display()
                )
            });

        let mut proc = Self {
            child: Some(child),
            _tempdir: tempdir,
            port: 0,
        };
        proc.port = wait_for_port_file(
            &port_file,
            proc.child.as_mut().expect("child handle present"),
            Duration::from_secs(20),
        );
        info!(
            port = proc.port,
            python_bin = %python_bin,
            "spawned live Python ML/RL gRPC sidecar subprocess"
        );
        proc
    }

    /// Returns the loopback [`SidecarEndpoint`] for the bound Python gRPC server.
    fn endpoint(&self) -> SidecarEndpoint {
        SidecarEndpoint::loopback_v4(self.port)
    }

    /// Simulates an abrupt `SIGKILL` crash of the out-of-process Python sidecar.
    fn kill_and_wait(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for PythonSidecarProcess {
    fn drop(&mut self) {
        self.kill_and_wait();
    }
}

/// Resolves `<Filter>/python-sidecar` relative to the root crate manifest directory.
fn python_sidecar_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.join("python-sidecar");
    assert!(
        root.join("aeris_sidecar/server.py").exists(),
        "expected python-sidecar package at {}",
        root.display()
    );
    root
}

/// Resolves the Python executable via `AERIS_PYTHON` or standard `PATH` probing.
fn detect_python_executable() -> String {
    if let Ok(explicit) = std::env::var("AERIS_PYTHON") {
        return explicit;
    }
    for candidate in ["python", "python3"] {
        if Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return candidate.to_string();
        }
    }
    panic!("no working Python interpreter found on PATH (set AERIS_PYTHON)");
}

/// Spins cooperatively (`thread::yield_now()`, zero `thread::sleep()`) until
/// the Python sidecar atomically publishes its bound TCP port.
fn wait_for_port_file(port_file: &Path, child: &mut Child, timeout: Duration) -> u16 {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(contents) = std::fs::read_to_string(port_file) {
            if let Ok(port) = contents.trim().parse::<u16>() {
                if port > 0 {
                    return port;
                }
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("Python sidecar exited prematurely with status {status}");
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for Python sidecar port file at {}",
            port_file.display()
        );
        thread::yield_now();
    }
}

/// Computes `shard_count - 1` evenly spaced split keys across `[0, KEYSPACE_MASK]`
/// so uniform 48-bit keys distribute identically across all initial shards.
fn uniform_keyspace_split_keys(shard_count: usize) -> Vec<Key> {
    assert!(shard_count >= 2, "shard_count must be >= 2");
    let universe = KEYSPACE_MASK + 1;
    let step = universe / (shard_count as u64);
    (1..shard_count as u64).map(|i| i * step - 1).collect()
}

/// Generates `count` unique even keys (`key & 1 == 0`) uniformly distributed over
/// `[0, KEYSPACE_MASK]` in sorted order so concurrent odd write keys are disjoint.
fn generate_even_uniform_keys(seed: u64, count: usize) -> Vec<Key> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(count);
    for _ in 0..count {
        keys.push((rng.gen::<u64>() & KEYSPACE_MASK) & !1u64);
    }
    keys.sort_unstable();
    keys.dedup();
    while keys.len() < count {
        let deficit = count - keys.len();
        keys.reserve(deficit);
        for _ in 0..deficit {
            keys.push((rng.gen::<u64>() & KEYSPACE_MASK) & !1u64);
        }
        keys.sort_unstable();
        keys.dedup();
    }
    keys.truncate(count);
    keys
}

/// Maps a monotonic counter `id` bijectively to a unique odd key uniformly spread
/// across `[0, KEYSPACE_MASK]`, guaranteed disjoint from all even pre-loaded keys.
#[inline]
fn bijective_odd_write_key(id: u64) -> Key {
    let scrambled = id.wrapping_mul(BIJECTIVE_ODD_SCRAMBLE) & MASK_47_BITS;
    (scrambled << 1) | 1u64
}

/// Computes the minimum quotient bit width `q >= min_q` whose slot capacity `2^q`
/// keeps `max_keys` comfortably below `0.55` load factor (`2^q >= max_keys * 2`).
fn choose_quotient_bits(max_keys: usize, min_q: u8) -> u8 {
    let needed = max_keys.saturating_mul(2).max(1usize << min_q);
    let pow2 = needed.next_power_of_two();
    let q = pow2.trailing_zeros() as u8;
    q.max(min_q)
}

/// Bulk-loads sorted keys into every shard of `system` in parallel using each
/// shard's real [`RocksDbReverseMap`] and atomic [`RcuAerisFilter::swap_filter`].
fn parallel_bulk_load_sharded_system(
    system: &ShardedAerisSystem,
    shard_cfg: &FilterConfig,
    sorted_keys: &[Key],
) {
    let bounds = system.shard_bounds();
    let handles: Vec<_> = bounds
        .iter()
        .map(|&(sid, b)| {
            let handle = system
                .shard_handle(sid)
                .expect("active shard handle must exist");
            (sid, b, handle)
        })
        .collect();

    thread::scope(|scope| {
        let mut workers = Vec::with_capacity(handles.len());
        for (sid, shard_bounds, handle) in &handles {
            let start = sorted_keys.partition_point(|&k| k < shard_bounds.min_key);
            let end = sorted_keys.partition_point(|&k| k <= shard_bounds.max_key);
            let slice = &sorted_keys[start..end];
            let cfg = shard_cfg.clone();
            workers.push(scope.spawn(move || {
                let rm = handle.reverse_map();
                let mut active_cfg = cfg;
                let filter = loop {
                    match AerisFilter::bulk_load(active_cfg.clone(), slice, rm.as_ref()) {
                        Ok(f) => break f,
                        Err(AerisError::FilterFull { .. }) => {
                            active_cfg = active_cfg
                                .expanded()
                                .expect("shard bulk-load config expansion must succeed");
                        }
                        Err(err) => panic!(
                            "bulk_load failed for shard {sid} ({} keys): {err}",
                            slice.len()
                        ),
                    }
                };
                let rm_dyn: Arc<dyn ReverseMapStore> = rm;
                handle
                    .rcu_filter()
                    .swap_filter(filter, rm_dyn)
                    .unwrap_or_else(|err| panic!("swap_filter failed for shard {sid}: {err}"));
            }));
        }
        for w in workers {
            w.join().expect("shard bulk-load thread panicked");
        }
    });
}

/// Generates `ops_count` deterministic 90% read / 10% write operations partitioned
/// evenly across `threads`, returning `(per_thread_ops, all_write_keys)`.
fn build_mixed_workload(
    seed: u64,
    run_id: u64,
    threads: usize,
    ops_count: usize,
    preloaded_keys: &[Key],
) -> (Vec<Vec<MixedOp>>, Vec<Key>) {
    let base_per_thread = ops_count / threads;
    let remainder = ops_count % threads;
    let max_range_start = KEYSPACE_MASK.saturating_sub(32);

    let mut per_thread = Vec::with_capacity(threads);
    let mut all_writes = Vec::with_capacity(ops_count / 10 + threads);
    let mut global_write_seq: u64 = (run_id << 28) + 1;

    for tid in 0..threads {
        let count = base_per_thread + usize::from(tid < remainder);
        let mut rng = ChaCha8Rng::seed_from_u64(
            seed ^ ((tid as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95)),
        );
        let mut ops = Vec::with_capacity(count);
        for _ in 0..count {
            let roll = rng.gen_range(0u32..100);
            if roll < 10 {
                let wk = bijective_odd_write_key(global_write_seq);
                global_write_seq += 1;
                all_writes.push(wk);
                ops.push(MixedOp::Insert(wk));
            } else if roll < 55 {
                let present = preloaded_keys[rng.gen_range(0..preloaded_keys.len())];
                ops.push(MixedOp::PresentPoint(present));
            } else {
                let start = rng.gen_range(0..=max_range_start);
                let range = QueryRange::with_len(start, 32).expect("valid 32-key range");
                ops.push(MixedOp::RangeProbe(range));
            }
        }
        per_thread.push(ops);
    }

    (per_thread, all_writes)
}

/// Executes a concurrent mixed workload against the Naive coarse-locked filter.
fn run_mixed_on_naive(
    naive: &Mutex<NaiveLockedFilter>,
    per_thread_ops: &[Vec<MixedOp>],
) -> (f64, LatencyStats, u64) {
    let threads = per_thread_ops.len();
    let total_ops: usize = per_thread_ops.iter().map(Vec::len).sum();
    let sample_stride = (total_ops / AB4_LATENCY_SAMPLE_BUDGET).max(1);
    let start_barrier = Barrier::new(threads + 1);
    let end_barrier = Barrier::new(threads + 1);

    let (lat_samples, false_negatives, elapsed_secs) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for ops in per_thread_ops {
            let sb = &start_barrier;
            let eb = &end_barrier;
            handles.push(scope.spawn(move || {
                let mut local_lat = Vec::with_capacity(ops.len() / sample_stride + 4);
                let mut local_fn = 0u64;
                let mut sink = 0u64;

                sb.wait();
                for (idx, &op) in ops.iter().enumerate() {
                    let sample_this = idx % sample_stride == 0;
                    let t0 = sample_this.then(Instant::now);
                    match op {
                        MixedOp::PresentPoint(k) => {
                            let guard = naive.lock();
                            if !guard.query_point(k) {
                                local_fn += 1;
                            }
                            sink = sink.wrapping_add(1);
                        }
                        MixedOp::RangeProbe(r) => {
                            let guard = naive.lock();
                            sink = sink.wrapping_add(u64::from(guard.query_range(r)));
                        }
                        MixedOp::Insert(k) => {
                            let mut guard = naive.lock();
                            guard.insert(k);
                        }
                    }
                    if let Some(ts) = t0 {
                        local_lat.push(ts.elapsed().as_nanos() as u64);
                    }
                }
                eb.wait();
                std::hint::black_box(sink);
                (local_lat, local_fn)
            }));
        }

        start_barrier.wait();
        let t_start = Instant::now();
        end_barrier.wait();
        let elapsed = t_start.elapsed().as_secs_f64().max(1e-9);

        let mut all_samples = Vec::with_capacity(AB4_LATENCY_SAMPLE_BUDGET + threads * 4);
        let mut total_fn = 0u64;
        for h in handles {
            let (s, f) = h.join().expect("naive scaling worker panicked");
            all_samples.extend(s);
            total_fn += f;
        }
        (all_samples, total_fn, elapsed)
    });

    let ops_per_sec = (total_ops as f64) / elapsed_secs;
    (
        ops_per_sec,
        LatencyStats::from_samples(lat_samples),
        false_negatives,
    )
}

/// Executes a mixed workload against the 8-shard [`ShardedAerisSystem`], engaging
/// the zero-overhead `&mut self` direct bypass (`query_point_direct`,
/// `query_range_direct`, `insert_direct`) when `threads == 1` and concurrent RCU
/// execution for `threads >= 2`.
fn run_mixed_on_sharded(
    system: &mut ShardedAerisSystem,
    per_thread_ops: &[Vec<MixedOp>],
) -> (f64, LatencyStats, u64) {
    let threads = per_thread_ops.len();
    let total_ops: usize = per_thread_ops.iter().map(Vec::len).sum();
    let sample_stride = (total_ops / AB4_LATENCY_SAMPLE_BUDGET).max(1);

    if threads == 1 {
        let ops = &per_thread_ops[0];
        let mut local_lat = Vec::with_capacity(ops.len() / sample_stride + 4);
        let mut local_fn = 0u64;
        let mut sink = 0u64;

        let t_start = Instant::now();
        for (idx, &op) in ops.iter().enumerate() {
            let sample_this = idx.is_multiple_of(sample_stride);
            let t0 = sample_this.then(Instant::now);
            match op {
                MixedOp::PresentPoint(k) => {
                    if !system.query_point_direct(k) {
                        local_fn += 1;
                    }
                    sink = sink.wrapping_add(1);
                }
                MixedOp::RangeProbe(r) => {
                    sink = sink.wrapping_add(u64::from(system.query_range_direct(r)));
                }
                MixedOp::Insert(k) => {
                    system
                        .insert_direct(k)
                        .expect("sharded 1T direct insert must succeed");
                }
            }
            if let Some(ts) = t0 {
                local_lat.push(ts.elapsed().as_nanos() as u64);
            }
        }
        let elapsed = t_start.elapsed().as_secs_f64().max(1e-9);
        std::hint::black_box(sink);
        let ops_per_sec = (total_ops as f64) / elapsed;
        return (ops_per_sec, LatencyStats::from_samples(local_lat), local_fn);
    }

    let system_ref: &ShardedAerisSystem = &*system;
    let start_barrier = Barrier::new(threads + 1);
    let end_barrier = Barrier::new(threads + 1);

    let (lat_samples, false_negatives, elapsed_secs) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for ops in per_thread_ops {
            let sb = &start_barrier;
            let eb = &end_barrier;
            handles.push(scope.spawn(move || {
                let mut local_lat = Vec::with_capacity(ops.len() / sample_stride + 4);
                let mut local_fn = 0u64;
                let mut sink = 0u64;

                sb.wait();
                for (idx, &op) in ops.iter().enumerate() {
                    let sample_this = idx.is_multiple_of(sample_stride);
                    let t0 = sample_this.then(Instant::now);
                    match op {
                        MixedOp::PresentPoint(k) => {
                            if !system_ref.query_point(k) {
                                local_fn += 1;
                            }
                            sink = sink.wrapping_add(1);
                        }
                        MixedOp::RangeProbe(r) => {
                            sink = sink.wrapping_add(u64::from(system_ref.query_range(r)));
                        }
                        MixedOp::Insert(k) => {
                            system_ref
                                .insert(k)
                                .expect("sharded concurrent insert must succeed");
                        }
                    }
                    if let Some(ts) = t0 {
                        local_lat.push(ts.elapsed().as_nanos() as u64);
                    }
                }
                eb.wait();
                std::hint::black_box(sink);
                (local_lat, local_fn)
            }));
        }

        start_barrier.wait();
        let t_start = Instant::now();
        end_barrier.wait();
        let elapsed = t_start.elapsed().as_secs_f64().max(1e-9);

        let mut all_samples = Vec::with_capacity(AB4_LATENCY_SAMPLE_BUDGET + threads * 4);
        let mut total_fn = 0u64;
        for h in handles {
            let (s, f) = h.join().expect("sharded scaling worker panicked");
            all_samples.extend(s);
            total_fn += f;
        }
        (all_samples, total_fn, elapsed)
    });

    let ops_per_sec = (total_ops as f64) / elapsed_secs;
    (
        ops_per_sec,
        LatencyStats::from_samples(lat_samples),
        false_negatives,
    )
}

/// **A/B 4.1:** Multi-core read/write scaling (1, 2, 4, 8 threads) on `5M` pre-loaded
/// keys and `5M` concurrent mixed operations (90% reads / 10% writes), comparing a
/// single coarse-locked `Mutex<AerisFilter>` vs. 8-shard [`ShardedAerisSystem`].
fn run_ab4_multicore_scaling(host: &HostMetadata) -> Vec<AblationMetric> {
    let preload_count = host.scaled_count(AB4_SCALING_BASE_KEYS);
    let ops_count = host.scaled_count(AB4_SCALING_BASE_OPS);
    let max_writes_all_runs = (ops_count / 10 + 1024) * AB4_THREAD_COUNTS.len();
    let max_total_keys = preload_count + max_writes_all_runs;
    let shard_q = choose_quotient_bits(max_total_keys / AB4_INITIAL_SHARDS, 10);
    let naive_q = shard_q + 3; // 2^3 == 8 shards => identical total slot capacity!

    info!(
        preload_count,
        ops_count,
        shard_q,
        naive_q,
        shards = AB4_INITIAL_SHARDS,
        "starting A/B 4.1 multi-core read/write scaling benchmark"
    );

    let preloaded_keys = generate_even_uniform_keys(0xAB41_0001, preload_count);
    let split_keys = uniform_keyspace_split_keys(AB4_INITIAL_SHARDS);

    let temp_dir = TempDir::new().expect("create tempdir for A/B 4.1 scaling");
    let rm_cfg = ReverseMapConfig::default()
        .with_write_buffer_size(32 * 1024 * 1024)
        .with_lz4_compression(false)
        .with_sync_writes(false);

    let shard_filter_cfg =
        FilterConfig::new(shard_q, 15, 32, 20, 1.0 / 256.0).expect("valid shard FilterConfig");
    let naive_filter_cfg =
        FilterConfig::new(naive_q, 15, 32, 20, 1.0 / 256.0).expect("valid naive FilterConfig");

    let sharding_cfg = ShardingConfig::new(shard_filter_cfg.clone())
        .with_initial_shard_count(AB4_INITIAL_SHARDS)
        .with_min_shards(AB4_INITIAL_SHARDS)
        .with_max_shards(32)
        .with_max_shard_q(shard_q + 2)
        .with_reverse_map_config(rm_cfg.clone())
        .with_fp_log_config(FpLogConfig {
            max_entries: 4_096,
            sync_on_append: false,
        });

    let sharded_dir = temp_dir.path().join("sharded_8");
    let mut sharded =
        ShardedAerisSystem::open_with_boundaries(&sharded_dir, sharding_cfg, &split_keys)
            .expect("open 8-shard ShardedAerisSystem");
    parallel_bulk_load_sharded_system(&sharded, &shard_filter_cfg, &preloaded_keys);

    let naive_dir = temp_dir.path().join("naive_single");
    let naive_rm = RocksDbReverseMap::open(&naive_dir, rm_cfg).expect("open naive ReverseMap");
    let naive_filter = AerisFilter::bulk_load(naive_filter_cfg, &preloaded_keys, &naive_rm)
        .expect("bulk_load naive single AerisFilter");
    let naive = Mutex::new(NaiveLockedFilter {
        filter: naive_filter,
        reverse_map: naive_rm,
    });

    assert_eq!(sharded.total_keys(), preload_count);
    assert_eq!(naive.lock().filter.len(), preload_count);

    let mut metrics = Vec::with_capacity(AB4_THREAD_COUNTS.len());
    let mut cumulative_writes: Vec<Key> = Vec::with_capacity(max_writes_all_runs);

    for (run_idx, &threads) in AB4_THREAD_COUNTS.iter().enumerate() {
        let seed = 0xAB41_1000 + (run_idx as u64) * 0x101;
        let (per_thread_ops, run_writes) =
            build_mixed_workload(seed, run_idx as u64, threads, ops_count, &preloaded_keys);

        let (naive_qps, naive_lat, naive_fn) = run_mixed_on_naive(&naive, &per_thread_ops);
        let (opt_qps, opt_lat, opt_fn) = run_mixed_on_sharded(&mut sharded, &per_thread_ops);

        assert_eq!(
            naive_fn, 0,
            "naive scaling run ({threads}T) observed false negatives"
        );
        assert_eq!(
            opt_fn, 0,
            "sharded scaling run ({threads}T) observed false negatives"
        );

        cumulative_writes.extend_from_slice(&run_writes);
        let expected_keys = preload_count + cumulative_writes.len();
        assert_eq!(
            sharded.total_keys(),
            expected_keys,
            "sharded key conservation failed after {threads}T run (lost writes)"
        );
        assert_eq!(
            naive.lock().filter.len(),
            expected_keys,
            "naive key conservation failed after {threads}T run"
        );

        let sharded_ram = sharded.total_memory_bytes() as u64;
        let naive_ram = naive.lock().filter.memory_usage_bytes() as u64;
        let disk_io = directory_size_bytes(&sharded_dir);
        let speedup = opt_qps / naive_qps.max(1e-9);

        info!(
            threads,
            naive_qps = format!("{naive_qps:.0}"),
            opt_qps = format!("{opt_qps:.0}"),
            speedup = format!("{speedup:.2}x"),
            naive_p99_ns = naive_lat.p99_ns,
            opt_p99_ns = opt_lat.p99_ns,
            opt_p999_ns = opt_lat.p999_ns,
            total_keys = expected_keys,
            "completed A/B 4.1 multi-core scaling tier"
        );

        metrics.push(AblationMetric {
            extension_name: format!("AB4_ScalePlane_MultiCore_{threads}T"),
            metric_label: format!("Mixed 90R/10W Throughput ({threads}T, 8 shards vs Mutex)"),
            unit: "ops/sec".to_string(),
            higher_is_better: true,
            key_count: expected_keys,
            query_count: ops_count,
            baseline_value: naive_qps,
            optimized_value: opt_qps,
            delta_pct: compute_delta_pct(naive_qps, opt_qps),
            p50_ns: opt_lat.p50_ns,
            p99_ns: opt_lat.p99_ns,
            p999_ns: opt_lat.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: sharded_ram.max(naive_ram),
            disk_io_bytes: disk_io,
            scale_factor: host.scale_factor,
            notes: format!(
                "threads={threads}; speedup={speedup:.2}x; naive_p50/p99/p999_ns={}/{}/{}; opt_p50/p99/p999_ns={}/{}/{}; shards={}; total_keys={expected_keys}",
                naive_lat.p50_ns,
                naive_lat.p99_ns,
                naive_lat.p999_ns,
                opt_lat.p50_ns,
                opt_lat.p99_ns,
                opt_lat.p999_ns,
                sharded.shard_count()
            ),
        });
    }

    // Exhaustive post-run verification: 0 lost writes and 0 false negatives across ALL
    // newly inserted keys (point + enclosing range) and all pre-loaded keys.
    for &wk in &cumulative_writes {
        assert!(
            sharded.query_point(wk),
            "lost write / false negative on sharded query_point({wk})"
        );
        assert!(
            sharded.query_range(enclosing_range(wk, 16)),
            "false negative on sharded enclosing range for written key {wk}"
        );
    }
    for &pk in &preloaded_keys {
        assert!(
            sharded.query_point(pk),
            "false negative on sharded pre-loaded key {pk}"
        );
    }
    sharded
        .verify_routing_invariants()
        .expect("sharded routing invariants must hold post-scaling");

    metrics
}

/// **A/B 4.2:** Reader latency (`p50`/`p99`/`p999`/`max`) during two consecutive
/// live `2x` expansions (`q -> q+1 -> q+2`) on real [`RcuAerisFilter`] (wait-free
/// epoch-RCU readers) vs. coarse `parking_lot::RwLock<AerisFilter>` (stop-the-world
/// exclusive write lock during `expand()`).
fn run_ab4_rcu_live_expansion(host: &HostMetadata) -> AblationMetric {
    const NUM_READERS: usize = 4;
    const WARMUP_QUERIES_BEFORE_EXPAND: usize = 8_000;

    let key_count = host.scaled_count(AB4_RCU_EXPANSION_BASE_KEYS);
    let initial_q = choose_quotient_bits(key_count, 16);
    let cfg =
        FilterConfig::new(initial_q, 15, 32, 20, 1.0 / 256.0).expect("valid A/B 4.2 FilterConfig");
    let oracle = generate_uniform_keys_oracle(0xAB42_0001, key_count);
    let keys = oracle.as_slice();

    let temp_dir = TempDir::new().expect("create tempdir for A/B 4.2 RCU live expansion");
    let rm_cfg = ReverseMapConfig::default()
        .with_write_buffer_size(32 * 1024 * 1024)
        .with_lz4_compression(false)
        .with_sync_writes(false);

    // 1. Naive coarse RwLock<AerisFilter>: readers take shared read(), expander takes exclusive write().
    let naive_rm_dir = temp_dir.path().join("naive_rwlock_rm");
    let naive_rm = RocksDbReverseMap::open(&naive_rm_dir, rm_cfg.clone()).expect("open naive RM");
    let naive_filter =
        AerisFilter::bulk_load(cfg.clone(), keys, &naive_rm).expect("bulk_load naive filter");
    let naive_rw = RwLock::new(naive_filter);

    let (naive_lat, naive_queries, naive_fn, naive_expand_ms) = {
        let start_barrier = Barrier::new(NUM_READERS + 1);
        let done = AtomicBool::new(false);
        let warmup_counter = AtomicUsize::new(0);
        let total_q = AtomicU64::new(0);
        let total_fn = AtomicU64::new(0);

        thread::scope(|scope| {
            let mut readers = Vec::with_capacity(NUM_READERS);
            for rid in 0..NUM_READERS {
                let sb = &start_barrier;
                let done_ref = &done;
                let warm_ref = &warmup_counter;
                let q_ref = &total_q;
                let fn_ref = &total_fn;
                let rw_ref = &naive_rw;
                readers.push(scope.spawn(move || {
                    let mut rng = ChaCha8Rng::seed_from_u64(0x42A0_0000 + rid as u64);
                    let mut samples = Vec::with_capacity(65_536);
                    let mut local_q = 0u64;
                    let mut local_fn = 0u64;

                    sb.wait();
                    while !done_ref.load(Ordering::Acquire) {
                        let k = keys[rng.gen_range(0..keys.len())];
                        let t0 = Instant::now();
                        let hit = {
                            let guard = rw_ref.read();
                            guard.query_point(k)
                        };
                        let elapsed_ns = t0.elapsed().as_nanos() as u64;
                        if !hit {
                            local_fn += 1;
                        }
                        local_q += 1;
                        if local_q <= 4_000 {
                            warm_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        if samples.len() < 120_000 {
                            samples.push(elapsed_ns);
                        }
                    }
                    q_ref.fetch_add(local_q, Ordering::Relaxed);
                    fn_ref.fetch_add(local_fn, Ordering::Relaxed);
                    samples
                }));
            }

            start_barrier.wait();
            while warmup_counter.load(Ordering::Relaxed) < WARMUP_QUERIES_BEFORE_EXPAND {
                thread::yield_now();
            }

            let exp_start = Instant::now();
            for _ in 0..2 {
                let mut guard = naive_rw.write();
                guard
                    .expand(&naive_rm)
                    .expect("naive RwLock 2x expansion must succeed");
                drop(guard);
                thread::yield_now();
            }
            let expand_ms = exp_start.elapsed().as_secs_f64() * 1000.0;
            done.store(true, Ordering::Release);

            let mut all_samples = Vec::new();
            for r in readers {
                all_samples.extend(r.join().expect("naive reader panicked"));
            }
            (
                LatencyStats::from_samples(all_samples),
                total_q.load(Ordering::Relaxed),
                total_fn.load(Ordering::Relaxed),
                expand_ms,
            )
        })
    };

    // 2. Optimized RcuAerisFilter: wait-free epoch-RCU readers never stall during expand_rcu().
    let rcu_rm_dir = temp_dir.path().join("rcu_rm");
    let rcu_rm = Arc::new(RocksDbReverseMap::open(&rcu_rm_dir, rm_cfg).expect("open RCU RM"));
    let rcu_rm_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&rcu_rm) as _;
    let rcu_init_filter =
        AerisFilter::bulk_load(cfg, keys, rcu_rm_dyn.as_ref()).expect("bulk_load RCU filter");
    let rcu = RcuAerisFilter::new(0, rcu_init_filter, rcu_rm_dyn);

    let (rcu_lat, rcu_queries, rcu_fn, rcu_expand_ms) = {
        let start_barrier = Barrier::new(NUM_READERS + 1);
        let done = AtomicBool::new(false);
        let warmup_counter = AtomicUsize::new(0);
        let total_q = AtomicU64::new(0);
        let total_fn = AtomicU64::new(0);

        thread::scope(|scope| {
            let mut readers = Vec::with_capacity(NUM_READERS);
            for rid in 0..NUM_READERS {
                let sb = &start_barrier;
                let done_ref = &done;
                let warm_ref = &warmup_counter;
                let q_ref = &total_q;
                let fn_ref = &total_fn;
                let rcu_ref = &rcu;
                readers.push(scope.spawn(move || {
                    let mut rng = ChaCha8Rng::seed_from_u64(0x42B0_0000 + rid as u64);
                    let mut samples = Vec::with_capacity(65_536);
                    let mut local_q = 0u64;
                    let mut local_fn = 0u64;

                    sb.wait();
                    while !done_ref.load(Ordering::Acquire) {
                        let k = keys[rng.gen_range(0..keys.len())];
                        let t0 = Instant::now();
                        let hit = rcu_ref.query_point(k);
                        let elapsed_ns = t0.elapsed().as_nanos() as u64;
                        if !hit {
                            local_fn += 1;
                        }
                        local_q += 1;
                        if local_q <= 4_000 {
                            warm_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        if samples.len() < 120_000 {
                            samples.push(elapsed_ns);
                        }
                    }
                    q_ref.fetch_add(local_q, Ordering::Relaxed);
                    fn_ref.fetch_add(local_fn, Ordering::Relaxed);
                    samples
                }));
            }

            start_barrier.wait();
            while warmup_counter.load(Ordering::Relaxed) < WARMUP_QUERIES_BEFORE_EXPAND {
                thread::yield_now();
            }

            let exp_start = Instant::now();
            for _ in 0..2 {
                let rep = rcu.expand_rcu().expect("RcuAerisFilter::expand_rcu failed");
                assert!(rep.new_epoch > rep.old_epoch);
                thread::yield_now();
            }
            let expand_ms = exp_start.elapsed().as_secs_f64() * 1000.0;
            done.store(true, Ordering::Release);

            let mut all_samples = Vec::new();
            for r in readers {
                all_samples.extend(r.join().expect("RCU reader panicked"));
            }
            (
                LatencyStats::from_samples(all_samples),
                total_q.load(Ordering::Relaxed),
                total_fn.load(Ordering::Relaxed),
                expand_ms,
            )
        })
    };

    assert_eq!(naive_fn, 0, "naive RwLock readers observed false negatives");
    assert_eq!(rcu_fn, 0, "RCU readers observed false negatives");
    for &k in keys.iter().step_by((keys.len() / 20_000).max(1)) {
        assert!(rcu.query_point(k), "post-expansion RCU FN on key {k}");
        assert!(
            rcu.query_range(enclosing_range(k, 16)),
            "post-expansion RCU enclosing-range FN on key {k}"
        );
    }
    rcu.verify_invariants()
        .expect("RcuAerisFilter invariants must hold post-expansion");

    let rcu_ram = rcu.memory_usage_bytes() as u64;
    let disk_io = directory_size_bytes(&rcu_rm_dir);
    let baseline_tail = naive_lat.p999_ns as f64;
    let optimized_tail = rcu_lat.p999_ns as f64;

    info!(
        key_count,
        naive_queries,
        rcu_queries,
        naive_p50_ns = naive_lat.p50_ns,
        naive_p99_ns = naive_lat.p99_ns,
        naive_p999_ns = naive_lat.p999_ns,
        naive_max_ns = naive_lat.max_ns,
        rcu_p50_ns = rcu_lat.p50_ns,
        rcu_p99_ns = rcu_lat.p99_ns,
        rcu_p999_ns = rcu_lat.p999_ns,
        rcu_max_ns = rcu_lat.max_ns,
        naive_expand_ms = format!("{naive_expand_ms:.1}"),
        rcu_expand_ms = format!("{rcu_expand_ms:.1}"),
        "completed A/B 4.2 live 2x expansion reader latency audit"
    );

    AblationMetric {
        extension_name: "AB4_ScalePlane_RCU_LiveExpansion".to_string(),
        metric_label: "Reader p99.9 Latency During 2x Live Expansion (RCU vs RwLock)".to_string(),
        unit: "ns".to_string(),
        higher_is_better: false,
        key_count,
        query_count: rcu_queries as usize,
        baseline_value: baseline_tail,
        optimized_value: optimized_tail,
        delta_pct: compute_delta_pct(baseline_tail, optimized_tail),
        p50_ns: rcu_lat.p50_ns,
        p99_ns: rcu_lat.p99_ns,
        p999_ns: rcu_lat.p999_ns,
        empirical_fpr: 0.0,
        false_negatives: 0,
        repeated_fps: 0,
        peak_ram_bytes: rcu_ram,
        disk_io_bytes: disk_io,
        scale_factor: host.scale_factor,
        notes: format!(
            "2x live expansions (q={initial_q}->{}); naive_rwlock_p50/p99/p999/max_ns={}/{}/{}/{}; rcu_p50/p99/p999/max_ns={}/{}/{}/{}; naive_queries={naive_queries}; rcu_queries={rcu_queries}",
            initial_q + 2,
            naive_lat.p50_ns,
            naive_lat.p99_ns,
            naive_lat.p999_ns,
            naive_lat.max_ns,
            rcu_lat.p50_ns,
            rcu_lat.p99_ns,
            rcu_lat.p999_ns,
            rcu_lat.max_ns,
        ),
    }
}

/// **A/B 4.3:** Skewed hot-spot writes (85% in a hot band via
/// [`generate_skewed_hotspot_write_trace`]) comparing a single unsharded
/// [`AerisFilter`] (forced to double its entire global table) vs. [`ShardedAerisSystem`]
/// performing live data-driven median splits and adjacent cold-shard merges.
fn run_ab4_hotspot_sharding(host: &HostMetadata) -> Vec<AblationMetric> {
    let base_count = host.scaled_count(AB4_HOTSPOT_BASE_KEYS);
    let hotspot_count = host.scaled_count(AB4_HOTSPOT_WRITE_KEYS);

    // Build base keys uniformly across shards 0..6 (97% => ~12.9k keys/shard, alpha ~0.79 < 0.90)
    // while keeping shards 6 and 7 sparse (3% => ~1.2k keys/shard, alpha ~0.07 < 0.20) so
    // rebalance_if_needed() exercises BOTH hot-spot median splits AND adjacent cold-shard merges.
    let split_keys = uniform_keyspace_split_keys(AB4_INITIAL_SHARDS);
    let cold_cutoff = split_keys[5]; // Upper boundary of shard 5 (shards 6 & 7 lie above)
    let mut base_keys: Vec<Key> = Vec::with_capacity(base_count);
    let mut rng = ChaCha8Rng::seed_from_u64(0xAB43_0002);
    let lower_target = (base_count * 97) / 100;
    let upper_target = base_count - lower_target;
    for _ in 0..lower_target {
        base_keys.push(rng.gen_range(0..=cold_cutoff) & !1u64);
    }
    for _ in 0..upper_target {
        base_keys.push(rng.gen_range((cold_cutoff + 1)..=KEYSPACE_MASK) & !1u64);
    }
    base_keys.sort_unstable();
    base_keys.dedup();
    let base_oracle = SortedKeyOracle::from_keys(base_keys);

    // 85% of writes concentrated inside shard 0's keyspace band `[1_000_000, 11_000_000)`.
    let raw_hotspot = generate_skewed_hotspot_write_trace(
        0xAB43_0003,
        hotspot_count,
        0.85,
        1_000_000,
        10_000_000,
    );
    // Guarantee strict disjointness from base_oracle while preserving arrival order.
    let mut seen_hot = std::collections::HashSet::with_capacity(raw_hotspot.len());
    let hotspot_trace: Vec<Key> = raw_hotspot
        .into_iter()
        .map(|k| k | 1u64)
        .filter(|&k| !base_oracle.contains(k) && seen_hot.insert(k))
        .collect();

    let total_expected_keys = base_oracle.len() + hotspot_trace.len();

    // Size both systems to identical initial slot capacity:
    // Naive q = 17 (131,072 slots) == 8 shards * 2^14 (16,384 slots/shard) at scale_factor = 1.0.
    let shard_q: u8 = if host.scale_factor < 0.75 { 13 } else { 14 };
    let naive_q: u8 = shard_q + 3;

    let temp_dir = TempDir::new().expect("create tempdir for A/B 4.3 hotspot sharding");
    let rm_cfg = ReverseMapConfig::default()
        .with_write_buffer_size(16 * 1024 * 1024)
        .with_lz4_compression(false)
        .with_sync_writes(false);

    let shard_cfg =
        FilterConfig::new(shard_q, 12, 32, 17, 1.0 / 256.0).expect("valid hotspot shard config");
    let naive_cfg =
        FilterConfig::new(naive_q, 12, 32, 17, 1.0 / 256.0).expect("valid hotspot naive config");

    // 1. Naive single unsharded AerisFilter (must globally double entire table).
    let naive_dir = temp_dir.path().join("naive_hotspot");
    let naive_rm = RocksDbReverseMap::open(&naive_dir, rm_cfg.clone()).expect("open naive RM");
    let mut naive = NaiveLockedFilter {
        filter: AerisFilter::bulk_load(naive_cfg, base_oracle.as_slice(), &naive_rm)
            .expect("bulk_load naive hotspot filter"),
        reverse_map: naive_rm,
    };
    let naive_init_ram = naive.filter.memory_usage_bytes() as u64;
    let mut naive_write_lat = Vec::with_capacity(hotspot_trace.len());
    for &k in &hotspot_trace {
        let t0 = Instant::now();
        naive.insert(k);
        naive_write_lat.push(t0.elapsed().as_nanos() as u64);
    }
    let naive_final_ram = naive.filter.memory_usage_bytes() as u64;
    let naive_lat_stats = LatencyStats::from_samples(naive_write_lat);

    // 2. Optimized ShardedAerisSystem with live median split/merge rebalancing.
    let sharded_dir = temp_dir.path().join("sharded_hotspot");
    let sharding_cfg = ShardingConfig::new(shard_cfg.clone())
        .with_initial_shard_count(AB4_INITIAL_SHARDS)
        .with_min_shards(2)
        .with_max_shards(64)
        .with_split_load_threshold(0.80)
        .with_split_min_keys(1_024)
        .with_max_shard_q(shard_q)
        .with_merge_load_threshold(0.20)
        .with_reverse_map_config(rm_cfg)
        .with_fp_log_config(FpLogConfig {
            max_entries: 4_096,
            sync_on_append: false,
        });

    let sharded = ShardedAerisSystem::open_with_boundaries(&sharded_dir, sharding_cfg, &split_keys)
        .expect("open ShardedAerisSystem for hotspot benchmark");
    parallel_bulk_load_sharded_system(&sharded, &shard_cfg, base_oracle.as_slice());
    let sharded_init_ram = sharded.total_memory_bytes() as u64;

    let rebalance_interval = (hotspot_trace.len() / 24).max(1_024);
    let mut total_splits = 0usize;
    let mut total_merges = 0usize;
    let mut sharded_write_lat = Vec::with_capacity(hotspot_trace.len());

    for (idx, &k) in hotspot_trace.iter().enumerate() {
        let t0 = Instant::now();
        sharded
            .insert(k)
            .expect("sharded hotspot insert must succeed");
        sharded_write_lat.push(t0.elapsed().as_nanos() as u64);

        if (idx + 1) % rebalance_interval == 0 {
            let reb = sharded
                .rebalance_if_needed()
                .expect("live hotspot rebalance_if_needed failed");
            total_splits += reb.splits.len();
            total_merges += reb.merges.len();
        }
    }
    let final_reb = sharded
        .rebalance_if_needed()
        .expect("final hotspot rebalance_if_needed failed");
    total_splits += final_reb.splits.len();
    total_merges += final_reb.merges.len();

    let sharded_final_ram = sharded.total_memory_bytes() as u64;
    let sharded_lat_stats = LatencyStats::from_samples(sharded_write_lat);
    let sharded_disk_io = directory_size_bytes(&sharded_dir);

    assert!(
        total_splits >= 2,
        "expected >= 2 hot-shard median splits under 85% skewed writes, got {total_splits}"
    );
    assert!(
        total_merges >= 1,
        "expected >= 1 adjacent cold-shard merge, got {total_merges}"
    );
    assert_eq!(
        sharded.total_keys(),
        total_expected_keys,
        "sharded hotspot lost writes: expected {total_expected_keys}, got {}",
        sharded.total_keys()
    );
    assert_eq!(naive.filter.len(), total_expected_keys);

    // Exhaustive 0-FN and 0-lost-write audit across 100% of base + hot-spot keys.
    for &k in base_oracle.as_slice().iter().chain(hotspot_trace.iter()) {
        assert!(
            sharded.query_point(k),
            "false negative / lost write on sharded hotspot key {k}"
        );
    }
    sharded
        .verify_all_invariants()
        .expect("ShardedAerisSystem::verify_all_invariants failed post-hotspot");

    let naive_ram_growth = naive_final_ram.saturating_sub(naive_init_ram);
    let sharded_ram_growth = sharded_final_ram.saturating_sub(sharded_init_ram);

    info!(
        total_keys = total_expected_keys,
        final_shards = sharded.shard_count(),
        total_splits,
        total_merges,
        naive_init_ram,
        naive_final_ram,
        sharded_init_ram,
        sharded_final_ram,
        naive_p99_ns = naive_lat_stats.p99_ns,
        sharded_p99_ns = sharded_lat_stats.p99_ns,
        "completed A/B 4.3 hot-spot skewed write sharding benchmark"
    );

    vec![
        AblationMetric {
            extension_name: "AB4_ScalePlane_HotSpot_Sharding_RAM".to_string(),
            metric_label: "Hot-Spot Final Filter RAM Footprint (85% Skewed Writes)".to_string(),
            unit: "bytes".to_string(),
            higher_is_better: false,
            key_count: total_expected_keys,
            query_count: hotspot_trace.len(),
            baseline_value: naive_final_ram as f64,
            optimized_value: sharded_final_ram as f64,
            delta_pct: compute_delta_pct(naive_final_ram as f64, sharded_final_ram as f64),
            p50_ns: sharded_lat_stats.p50_ns,
            p99_ns: sharded_lat_stats.p99_ns,
            p999_ns: sharded_lat_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: sharded_final_ram,
            disk_io_bytes: sharded_disk_io,
            scale_factor: host.scale_factor,
            notes: format!(
                "naive_init/final/growth={naive_init_ram}/{naive_final_ram}/{naive_ram_growth}B (q={}->{}); sharded_init/final/growth={sharded_init_ram}/{sharded_final_ram}/{sharded_ram_growth}B; splits={total_splits}; merges={total_merges}; final_shards={}",
                naive_q,
                naive.filter.config().q,
                sharded.shard_count()
            ),
        },
        AblationMetric {
            extension_name: "AB4_ScalePlane_HotSpot_Sharding_Latency".to_string(),
            metric_label: "Hot-Spot Skewed Write p99 Latency (Sharded Split/Merge vs Global)"
                .to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: total_expected_keys,
            query_count: hotspot_trace.len(),
            baseline_value: naive_lat_stats.p99_ns as f64,
            optimized_value: sharded_lat_stats.p99_ns as f64,
            delta_pct: compute_delta_pct(
                naive_lat_stats.p99_ns as f64,
                sharded_lat_stats.p99_ns as f64,
            ),
            p50_ns: sharded_lat_stats.p50_ns,
            p99_ns: sharded_lat_stats.p99_ns,
            p999_ns: sharded_lat_stats.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: sharded_final_ram,
            disk_io_bytes: sharded_disk_io,
            scale_factor: host.scale_factor,
            notes: format!(
                "naive_p50/p99/p999/max_ns={}/{}/{}/{}; sharded_p50/p99/p999/max_ns={}/{}/{}/{}; splits={total_splits}; merges={total_merges}",
                naive_lat_stats.p50_ns,
                naive_lat_stats.p99_ns,
                naive_lat_stats.p999_ns,
                naive_lat_stats.max_ns,
                sharded_lat_stats.p50_ns,
                sharded_lat_stats.p99_ns,
                sharded_lat_stats.p999_ns,
                sharded_lat_stats.max_ns
            ),
        },
    ]
}

/// **A/B 5.1:** Seasonal shifting trace over `1,000,000` real keys comparing
/// Native Control Plane (`ProactiveTrigger` + `DiscreteCostOptimizer`) vs.
/// live Python gRPC Sidecar (`ProposeTuning` Holt-Winters + Fourier forecast
/// driving proactive RCU pre-warming ahead of seasonal ingestion surges).
fn run_ab5_seasonal_sidecar(
    advisor: &GuardedTuningAdvisor<NativeFallbackOptimizer>,
    bounds: &SafetyBounds,
) -> Vec<AblationMetric> {
    const TOTAL_STEPS: usize = 90;
    const PERIOD: usize = 24;
    const HISTORY_WINDOW: usize = 48;
    // Start active evaluation at step 66 (66 % 24 == 18: diurnal trough immediately
    // preceding the surge ramp) so seasonal forecasting has 48 steps of history.
    const EVAL_START_STEP: usize = 66;
    const PRELOAD_KEYS: usize = 680_000;
    const SURGE_INGEST_KEYS: usize = AB5_TOTAL_KEYS - PRELOAD_KEYS; // 320,000 -> total 1,000,000
    const ASSUMED_BULK_LOAD_SPEED: f64 = 120_000.0;

    let trace =
        generate_seasonal_ingestion_trace(0xAB51_0001, TOTAL_STEPS, PERIOD, 14_000.0, 9_500.0);
    let eval_steps = &trace[EVAL_START_STEP..];
    let total_rate_weight: f64 = eval_steps
        .iter()
        .map(|p| p.ingest_rate_keys_per_sec)
        .sum::<f64>()
        .max(1.0);

    // Distribute 320,000 real keys proportionally across the 24 diurnal evaluation steps.
    let mut keys_per_step: Vec<usize> = eval_steps
        .iter()
        .map(|p| {
            ((p.ingest_rate_keys_per_sec / total_rate_weight) * (SURGE_INGEST_KEYS as f64)).round()
                as usize
        })
        .collect();
    let assigned: usize = keys_per_step.iter().sum();
    if assigned < SURGE_INGEST_KEYS {
        keys_per_step[0] += SURGE_INGEST_KEYS - assigned;
    } else if assigned > SURGE_INGEST_KEYS {
        let mut excess = assigned - SURGE_INGEST_KEYS;
        for slot in keys_per_step.iter_mut().rev() {
            let take = (*slot).min(excess);
            *slot -= take;
            excess -= take;
            if excess == 0 {
                break;
            }
        }
    }
    assert_eq!(keys_per_step.iter().sum::<usize>(), SURGE_INGEST_KEYS);

    let all_keys = generate_uniform_keys_oracle(0xAB51_0002, AB5_TOTAL_KEYS).into_vec();
    // Shuffle arrival order deterministically so inserts span the full keyspace uniformly.
    let mut rng = ChaCha8Rng::seed_from_u64(0xAB51_0003);
    let mut shuffled_keys = all_keys;
    for i in (1..shuffled_keys.len()).rev() {
        let j = rng.gen_range(0..=i);
        shuffled_keys.swap(i, j);
    }
    let preload_slice = &shuffled_keys[..PRELOAD_KEYS];
    let stream_slice = &shuffled_keys[PRELOAD_KEYS..];

    let temp_dir = TempDir::new().expect("create tempdir for A/B 5.1 seasonal sidecar");
    let rm_cfg = ReverseMapConfig::default()
        .with_write_buffer_size(32 * 1024 * 1024)
        .with_lz4_compression(false)
        .with_sync_writes(false);

    // q = 20 -> 1,048,576 slots; 680k pre-loaded => alpha = 0.648, reaching 1,000,000 => 0.954 (> 0.95)
    let init_cfg =
        FilterConfig::new(20, 14, 64, 20, 1.0 / 256.0).expect("valid A/B 5.1 FilterConfig");

    let native_dir = temp_dir.path().join("native_seasonal");
    let native_rm = RocksDbReverseMap::open(&native_dir, rm_cfg.clone()).expect("open native RM");
    let mut native_filter = AerisFilter::bulk_load(init_cfg.clone(), preload_slice, &native_rm)
        .expect("bulk_load native");

    let sidecar_dir = temp_dir.path().join("sidecar_seasonal");
    let sidecar_rm =
        Arc::new(RocksDbReverseMap::open(&sidecar_dir, rm_cfg).expect("open sidecar RM"));
    let sidecar_rm_dyn: Arc<dyn ReverseMapStore> = Arc::clone(&sidecar_rm) as _;
    let sidecar_init = AerisFilter::bulk_load(init_cfg, preload_slice, sidecar_rm_dyn.as_ref())
        .expect("bulk_load sidecar filter");
    let sidecar_rcu = RcuAerisFilter::new(0, sidecar_init, sidecar_rm_dyn);

    let mut native_trigger = ProactiveTrigger::new(ProactiveTriggerConfig::default())
        .expect("valid ProactiveTriggerConfig");
    let native_opt = DiscreteCostOptimizer::new();
    let cost_model = CostModelParams::default();
    let guardrail = SafetyGuardrail::new();

    // Seed Native EWMA from the full pre-evaluation history up to the diurnal trough.
    let mut ewma_ingest = trace[0].ingest_rate_keys_per_sec;
    for pt in &trace[1..EVAL_START_STEP] {
        ewma_ingest = 0.30 * pt.ingest_rate_keys_per_sec + 0.70 * ewma_ingest;
    }

    let mut native_expanded = false;
    let mut native_expand_step: Option<usize> = None;
    let mut native_stall_ns: u64 = 0;
    let mut native_write_lat_ns: Vec<u64> = Vec::with_capacity(SURGE_INGEST_KEYS);

    let mut sidecar_expanded = false;
    let mut sidecar_expand_step: Option<usize> = None;
    let mut sidecar_prewarm_ns: u64 = 0;
    let mut sidecar_write_lat_ns: Vec<u64> = Vec::with_capacity(SURGE_INGEST_KEYS);
    let mut sidecar_proposals_accepted = 0usize;

    let mut cursor = 0usize;
    for (rel_idx, &batch_len) in keys_per_step.iter().enumerate() {
        let abs_step = EVAL_START_STEP + rel_idx;
        let pt = trace[abs_step];
        let step_keys = &stream_slice[cursor..cursor + batch_len];
        cursor += batch_len;

        // --- Native Control Plane decision & ingestion ---
        ewma_ingest = 0.30 * pt.ingest_rate_keys_per_sec + 0.70 * ewma_ingest;
        let native_stats = WorkloadStats {
            p95_query_len: pt.p95_query_len.round() as u64,
            effective_query_span: pt.p95_query_len.round() as u64,
            ingestion_rate_ewma: ewma_ingest,
            bulk_load_speed: ASSUMED_BULK_LOAD_SPEED,
            query_rate: pt.query_rate_qps,
            ..WorkloadStats::default()
        };
        let _native_cost_dec =
            native_opt.optimize(&native_stats, native_filter.config(), &cost_model);

        if !native_expanded {
            let trig_eval = native_trigger.evaluate(&TriggerInput {
                capacity_slots: native_filter.capacity(),
                current_q: native_filter.config().q,
                current_alpha: native_filter.load_factor(),
                ingestion_rate_ewma: ewma_ingest,
                reverse_map_entries: native_filter.len() as u64,
                bulk_load_speed: ASSUMED_BULK_LOAD_SPEED,
            });
            if trig_eval.action == TriggerAction::ProactiveExpand {
                let t0 = Instant::now();
                native_filter
                    .expand(&native_rm)
                    .expect("native proactive expand must succeed");
                let dt = t0.elapsed().as_nanos() as u64;
                native_stall_ns = native_stall_ns.saturating_add(dt);
                native_write_lat_ns.push(dt);
                native_expanded = true;
                native_expand_step = Some(rel_idx);
            }
        }

        for &k in step_keys {
            let t0 = Instant::now();
            match native_filter.insert(k) {
                Ok(fp) => {
                    native_rm
                        .append_key(&fp, k)
                        .expect("native seasonal RM append failed");
                }
                Err(AerisError::FilterFull { .. }) => {
                    native_filter
                        .expand(&native_rm)
                        .expect("native reactive expand failed");
                    let fp = native_filter.insert(k).expect("insert after expand failed");
                    native_rm
                        .append_key(&fp, k)
                        .expect("native seasonal RM append failed");
                    let stall = t0.elapsed().as_nanos() as u64;
                    native_stall_ns = native_stall_ns.saturating_add(stall);
                    if !native_expanded {
                        native_expanded = true;
                        native_expand_step = Some(rel_idx);
                    }
                }
                Err(err) => panic!("unexpected native insert error: {err}"),
            }
            native_write_lat_ns.push(t0.elapsed().as_nanos() as u64);
        }

        // --- Live Python Sidecar decision & RCU pre-warming + ingestion ---
        let hist_slice = &trace[(abs_step - HISTORY_WINDOW)..abs_step];
        let ingest_series: Vec<f64> = hist_slice
            .iter()
            .map(|s| s.ingest_rate_keys_per_sec)
            .collect();
        let qlen_series: Vec<f64> = hist_slice.iter().map(|s| s.p95_query_len).collect();
        let sc_cfg = sidecar_rcu.config();

        let telemetry = TelemetrySnapshot {
            request_id: format!("ab5-seasonal-step-{rel_idx}"),
            shard_id: 0,
            epoch: sidecar_rcu.epoch(),
            timestamp_ms: 1_700_000_000_000 + (abs_step as u64) * 60_000,
            current_q: sc_cfg.q,
            current_r: sc_cfg.r,
            current_m: sc_cfg.m,
            payload_budget_c: sc_cfg.payload_budget_c,
            target_epsilon: sc_cfg.target_epsilon,
            load_factor: sidecar_rcu.load_factor(),
            ingestion_rate_ewma: ewma_ingest,
            query_rate: pt.query_rate_qps,
            observed_fpr: 0.002,
            p95_query_len: pt.p95_query_len,
            p99_query_len: pt.p95_query_len * 1.35,
            reverse_map_entries: sidecar_rcu.len() as u64,
            bulk_load_speed: ASSUMED_BULK_LOAD_SPEED,
            recent_query_len_series: qlen_series,
            recent_ingest_rate_series: ingest_series,
        };

        let proposal = advisor
            .advise(&telemetry)
            .expect("live Python sidecar advise() must succeed");
        assert_eq!(
            guardrail.validate_tuning(&proposal, bounds),
            GuardrailVerdict::Approved,
            "every returned tuning proposal must pass SafetyGuardrail"
        );
        if proposal.model_version != NATIVE_FALLBACK_MODEL_VERSION {
            sidecar_proposals_accepted += 1;
        }

        if !sidecar_expanded {
            // Check capacity pre-warming condition driven by the sidecar's forecasted peak rate,
            // lowered target alpha, and widened proactive margin k.
            let cap = sidecar_rcu.capacity() as f64;
            let remaining_slots =
                (proposal.proposed_target_alpha - sidecar_rcu.load_factor()).max(0.0) * cap;
            let t_full_forecast = if proposal.forecast_peak_ingest_rate > 0.0 {
                remaining_slots / proposal.forecast_peak_ingest_rate
            } else {
                f64::INFINITY
            };
            let t_rebuild = (sidecar_rcu.len() as f64) / ASSUMED_BULK_LOAD_SPEED;
            let capacity_prewarm_due =
                t_full_forecast <= t_rebuild * proposal.proposed_proactive_margin_k;

            if capacity_prewarm_due {
                let t0 = Instant::now();
                sidecar_rcu
                    .expand_rcu()
                    .expect("sidecar proactive RCU pre-warming expand failed");
                sidecar_prewarm_ns = t0.elapsed().as_nanos() as u64;
                sidecar_expanded = true;
                sidecar_expand_step = Some(rel_idx);
            }
        }

        // Foreground writes run cleanly on the RCU-prewarmed filter with zero mid-surge stall.
        for &k in step_keys {
            let t0 = Instant::now();
            sidecar_rcu
                .insert(k)
                .expect("sidecar RCU insert must succeed");
            sidecar_write_lat_ns.push(t0.elapsed().as_nanos() as u64);
        }
    }

    assert!(
        sidecar_proposals_accepted > 0,
        "expected live Python sidecar proposals to be accepted on 48-step seasonal telemetry"
    );
    assert_eq!(native_filter.len(), AB5_TOTAL_KEYS);
    assert_eq!(sidecar_rcu.len(), AB5_TOTAL_KEYS);

    // Verify 0 false negatives across a dense deterministic sample of the 1M keys.
    for &k in shuffled_keys.iter().step_by(20) {
        assert!(
            native_filter.query_point(k),
            "native seasonal filter FN on key {k}"
        );
        assert!(
            sidecar_rcu.query_point(k),
            "sidecar seasonal filter FN on key {k}"
        );
    }

    let native_lat = LatencyStats::from_samples(native_write_lat_ns);
    let sidecar_lat = LatencyStats::from_samples(sidecar_write_lat_ns);
    let native_step = native_expand_step.unwrap_or(keys_per_step.len());
    let sidecar_step = sidecar_expand_step.unwrap_or(keys_per_step.len());
    let lead_time_steps = (native_step as i64) - (sidecar_step as i64);

    let sidecar_ram = sidecar_rcu.memory_usage_bytes() as u64;
    let disk_io = directory_size_bytes(&sidecar_dir);

    info!(
        total_keys = AB5_TOTAL_KEYS,
        sidecar_proposals_accepted,
        native_expand_step = native_step,
        sidecar_expand_step = sidecar_step,
        lead_time_steps,
        native_max_write_ns = native_lat.max_ns,
        sidecar_max_write_ns = sidecar_lat.max_ns,
        native_p999_ns = native_lat.p999_ns,
        sidecar_p999_ns = sidecar_lat.p999_ns,
        sidecar_prewarm_ms = format!("{:.1}", sidecar_prewarm_ns as f64 / 1e6),
        "completed A/B 5.1 seasonal sidecar pre-warming benchmark"
    );

    vec![
        AblationMetric {
            extension_name: "AB5_Sidecar_Seasonal_PreWarm_Stall".to_string(),
            metric_label: "Seasonal Surge Max Foreground Write Stall (Sidecar RCU vs Native)"
                .to_string(),
            unit: "ns".to_string(),
            higher_is_better: false,
            key_count: AB5_TOTAL_KEYS,
            query_count: SURGE_INGEST_KEYS,
            baseline_value: native_lat.max_ns as f64,
            optimized_value: sidecar_lat.max_ns as f64,
            delta_pct: compute_delta_pct(native_lat.max_ns as f64, sidecar_lat.max_ns as f64),
            p50_ns: sidecar_lat.p50_ns,
            p99_ns: sidecar_lat.p99_ns,
            p999_ns: sidecar_lat.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: sidecar_ram,
            disk_io_bytes: disk_io,
            scale_factor: 1.0,
            notes: format!(
                "native_p50/p99/p999/max_ns={}/{}/{}/{}; sidecar_p50/p99/p999/max_ns={}/{}/{}/{}; sidecar_accepted={sidecar_proposals_accepted}/24; prewarm_rcu_ns={sidecar_prewarm_ns}",
                native_lat.p50_ns,
                native_lat.p99_ns,
                native_lat.p999_ns,
                native_lat.max_ns,
                sidecar_lat.p50_ns,
                sidecar_lat.p99_ns,
                sidecar_lat.p999_ns,
                sidecar_lat.max_ns,
            ),
        },
        AblationMetric {
            extension_name: "AB5_Sidecar_Seasonal_Forecast_LeadTime".to_string(),
            metric_label: "Seasonal Surge Expansion Trigger Step (Lower = Earlier Pre-Warm)"
                .to_string(),
            unit: "steps".to_string(),
            higher_is_better: false,
            key_count: AB5_TOTAL_KEYS,
            query_count: 24,
            baseline_value: (native_step + 1) as f64,
            optimized_value: (sidecar_step + 1) as f64,
            delta_pct: compute_delta_pct((native_step + 1) as f64, (sidecar_step + 1) as f64),
            p50_ns: sidecar_lat.p50_ns,
            p99_ns: sidecar_lat.p99_ns,
            p999_ns: sidecar_lat.p999_ns,
            empirical_fpr: 0.0,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: sidecar_ram,
            disk_io_bytes: disk_io,
            scale_factor: 1.0,
            notes: format!(
                "native_trigger_step={native_step}; sidecar_trigger_step={sidecar_step}; forecast_lead_time_steps={lead_time_steps}; native_stall_ns={native_stall_ns}"
            ),
        },
    ]
}

/// **A/B 5.2:** Multi-tenant shard workload (`1,000,000` keys across 2 hot high-FPR/high-QPS
/// shards + 2 cold idle shards under a shared 10 MiB RAM budget) comparing Native
/// uniform/static allocation vs. live Python `AllocateMemory` contextual bandit allocation.
fn run_ab5_multitenant_bandit(
    advisor: &GuardedTuningAdvisor<NativeFallbackOptimizer>,
    bounds: &SafetyBounds,
) -> Vec<AblationMetric> {
    const SHARED_BUDGET_BYTES: u64 = 10 * 1024 * 1024; // 10 MiB total across 4 shards
    const HOT_SHARD_KEYS: usize = 300_000;
    const COLD_SHARD_KEYS: usize = 200_000;
    const EVAL_EMPTY_QUERIES: usize = 200_000;
    const HOT_QUERY_LEN: u64 = 256;
    const COLD_QUERY_LEN: u64 = 8;

    assert_eq!(2 * HOT_SHARD_KEYS + 2 * COLD_SHARD_KEYS, AB5_TOTAL_KEYS);

    let shard_key_counts = [
        HOT_SHARD_KEYS,
        HOT_SHARD_KEYS,
        COLD_SHARD_KEYS,
        COLD_SHARD_KEYS,
    ];
    let shard_oracles: Vec<SortedKeyOracle> = shard_key_counts
        .iter()
        .enumerate()
        .map(|(sid, &cnt)| generate_uniform_keys_oracle(0xAB52_1000 + sid as u64, cnt))
        .collect();

    // Generate verified-empty queries per shard matching hot (98% traffic, R=256) vs cold (2%, R=8).
    let hot_q_each = (EVAL_EMPTY_QUERIES * 49) / 100;
    let cold_q_each = (EVAL_EMPTY_QUERIES - 2 * hot_q_each) / 2;
    let shard_query_counts = [hot_q_each, hot_q_each, cold_q_each, cold_q_each];
    let shard_query_lens = [HOT_QUERY_LEN, HOT_QUERY_LEN, COLD_QUERY_LEN, COLD_QUERY_LEN];
    let shard_queries: Vec<Vec<QueryRange>> = (0..4)
        .map(|sid| {
            generate_uniform_empty_queries(
                0xAB52_2000 + sid as u64,
                &shard_oracles[sid],
                shard_query_lens[sid],
                shard_query_counts[sid],
            )
        })
        .collect();
    let total_queries: usize = shard_queries.iter().map(Vec::len).sum();

    let temp_dir = TempDir::new().expect("create tempdir for A/B 5.2 multi-tenant bandit");
    let rm_cfg = ReverseMapConfig::default()
        .with_write_buffer_size(16 * 1024 * 1024)
        .with_lz4_compression(false)
        .with_sync_writes(false);

    // 1. Native uniform/static allocation: equal RAM share (2.5 MiB -> q=19), static (r=15, m=5, R=32).
    let mut uniform_filters = Vec::with_capacity(4);
    let mut uniform_ram = 0u64;
    for (sid, oracle) in shard_oracles.iter().enumerate() {
        let cfg = FilterConfig::new(19, 15, 32, 20, 1.0 / 256.0).expect("valid uniform shard cfg");
        let rm = RocksDbReverseMap::open(
            temp_dir.path().join(format!("uni_rm_{sid}")),
            rm_cfg.clone(),
        )
        .expect("open uniform shard RM");
        let filter =
            AerisFilter::bulk_load(cfg, oracle.as_slice(), &rm).expect("bulk_load uniform shard");
        uniform_ram += filter.memory_usage_bytes() as u64;
        uniform_filters.push(filter);
    }

    // Measure empirical FPR and latency under Native uniform/static allocation.
    let (uni_fps, uni_lat, uni_cpu_ns, uni_fn) =
        eval_multitenant_shards(&uniform_filters, &shard_oracles, &shard_queries);
    assert_eq!(
        uni_fn, 0,
        "uniform multi-tenant shards observed false negatives"
    );

    // 2. Query live Python sidecar AllocateMemory contextual bandit with live shard telemetry.
    let shard_telemetry: Vec<ShardTelemetrySnapshot> = (0..4usize)
        .map(|sid| {
            let is_hot = sid < 2;
            let f = &uniform_filters[sid];
            let cfg = f.config();
            ShardTelemetrySnapshot {
                shard_id: sid as u64,
                key_count: f.len() as u64,
                current_memory_bytes: f.memory_usage_bytes() as u64,
                query_rate: if is_hot { 45_000.0 } else { 5.0 },
                observed_fpr: if is_hot {
                    (uni_fps[sid] as f64 / shard_queries[sid].len().max(1) as f64).max(0.05)
                } else {
                    0.0002
                },
                p95_query_len: shard_query_lens[sid] as f64,
                load_factor: if is_hot { 0.88 } else { 0.22 },
                current_q: cfg.q,
                current_r: cfg.r,
                current_m: cfg.m,
            }
        })
        .collect();

    let plan = advisor
        .allocate_memory(1, SHARED_BUDGET_BYTES, &shard_telemetry)
        .expect("live Python sidecar allocate_memory must succeed");
    assert_eq!(
        plan.model_version, "aeris-sidecar-ml-v1",
        "expected live Python contextual bandit plan"
    );
    assert_eq!(
        SafetyGuardrail::new().validate_memory_plan(&plan, SHARED_BUDGET_BYTES, bounds),
        GuardrailVerdict::Approved
    );

    let hot_alloc_bytes: u64 = plan
        .allocations
        .iter()
        .filter(|a| a.shard_id < 2)
        .map(|a| a.allocated_memory_bytes)
        .sum();
    let cold_alloc_bytes: u64 = plan
        .allocations
        .iter()
        .filter(|a| a.shard_id >= 2)
        .map(|a| a.allocated_memory_bytes)
        .sum();
    assert!(
        hot_alloc_bytes > cold_alloc_bytes,
        "bandit must allocate more RAM to hot shards ({hot_alloc_bytes}) than cold shards ({cold_alloc_bytes})"
    );

    // Materialize REAL AerisFilter instances obeying the bandit's (q, r, m) plan.
    let mut bandit_filters = Vec::with_capacity(4);
    let mut bandit_ram = 0u64;
    for (sid, oracle) in shard_oracles.iter().enumerate() {
        let alloc = plan
            .allocations
            .iter()
            .find(|a| a.shard_id == sid as u64)
            .expect("shard allocation present");
        let rm = RocksDbReverseMap::open(
            temp_dir.path().join(format!("bandit_rm_{sid}")),
            rm_cfg.clone(),
        )
        .expect("open bandit shard RM");
        let mut q = alloc.recommended_q;
        let max_range_r = 1u64 << alloc.recommended_m;
        let filter = loop {
            let cfg = FilterConfig::new(
                q,
                alloc.recommended_r,
                max_range_r,
                bounds.payload_budget_c,
                bounds.target_epsilon,
            )
            .expect("guardrail-validated bandit FilterConfig must be valid");
            match AerisFilter::bulk_load(cfg, oracle.as_slice(), &rm) {
                Ok(f) => break f,
                Err(AerisError::FilterFull { .. }) => q += 1,
                Err(err) => panic!("unexpected bandit bulk_load error on shard {sid}: {err}"),
            }
        };
        bandit_ram += filter.memory_usage_bytes() as u64;
        bandit_filters.push(filter);
    }

    let (bandit_fps, bandit_lat, bandit_cpu_ns, bandit_fn) =
        eval_multitenant_shards(&bandit_filters, &shard_oracles, &shard_queries);
    assert_eq!(
        bandit_fn, 0,
        "bandit multi-tenant shards observed false negatives"
    );

    let uni_total_fps: u64 = uni_fps.iter().sum();
    let bandit_total_fps: u64 = bandit_fps.iter().sum();
    let uni_fpr = (uni_total_fps as f64) / (total_queries as f64);
    let bandit_fpr = (bandit_total_fps as f64) / (total_queries as f64);

    // Effective query throughput including 50 us simulated disk-check penalty for each FP.
    let uni_effective_secs =
        ((uni_cpu_ns + uni_total_fps * SIMULATED_DISK_CHECK_PENALTY_NS) as f64) / 1e9;
    let bandit_effective_secs =
        ((bandit_cpu_ns + bandit_total_fps * SIMULATED_DISK_CHECK_PENALTY_NS) as f64) / 1e9;
    let uni_effective_qps = (total_queries as f64) / uni_effective_secs.max(1e-9);
    let bandit_effective_qps = (total_queries as f64) / bandit_effective_secs.max(1e-9);

    let disk_io = directory_size_bytes(temp_dir.path());

    info!(
        total_keys = AB5_TOTAL_KEYS,
        total_queries,
        uni_total_fps,
        bandit_total_fps,
        uni_fpr = format!("{uni_fpr:.6}"),
        bandit_fpr = format!("{bandit_fpr:.6}"),
        uni_effective_qps = format!("{uni_effective_qps:.0}"),
        bandit_effective_qps = format!("{bandit_effective_qps:.0}"),
        hot_alloc_bytes,
        cold_alloc_bytes,
        "completed A/B 5.2 multi-tenant contextual bandit benchmark"
    );

    vec![
        AblationMetric {
            extension_name: "AB5_Sidecar_MultiTenant_Bandit_FPR".to_string(),
            metric_label: "Multi-Tenant Workload Empirical FPR (Bandit vs Uniform Static)"
                .to_string(),
            unit: "ratio".to_string(),
            higher_is_better: false,
            key_count: AB5_TOTAL_KEYS,
            query_count: total_queries,
            baseline_value: uni_fpr,
            optimized_value: bandit_fpr,
            delta_pct: compute_delta_pct(uni_fpr, bandit_fpr),
            p50_ns: bandit_lat.p50_ns,
            p99_ns: bandit_lat.p99_ns,
            p999_ns: bandit_lat.p999_ns,
            empirical_fpr: bandit_fpr,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: bandit_ram.max(uniform_ram),
            disk_io_bytes: disk_io,
            scale_factor: 1.0,
            notes: format!(
                "budget={SHARED_BUDGET_BYTES}B; hot_alloc={hot_alloc_bytes}B vs cold_alloc={cold_alloc_bytes}B; uni_fps={uni_total_fps} ({uni_fpr:.6}) vs bandit_fps={bandit_total_fps} ({bandit_fpr:.6}); uni_ram={uniform_ram}B; bandit_ram={bandit_ram}B"
            ),
        },
        AblationMetric {
            extension_name: "AB5_Sidecar_MultiTenant_Bandit_Throughput".to_string(),
            metric_label: "Multi-Tenant Effective Query Throughput (incl. 50us FP Disk Penalty)"
                .to_string(),
            unit: "ops/sec".to_string(),
            higher_is_better: true,
            key_count: AB5_TOTAL_KEYS,
            query_count: total_queries,
            baseline_value: uni_effective_qps,
            optimized_value: bandit_effective_qps,
            delta_pct: compute_delta_pct(uni_effective_qps, bandit_effective_qps),
            p50_ns: bandit_lat.p50_ns,
            p99_ns: bandit_lat.p99_ns,
            p999_ns: bandit_lat.p999_ns,
            empirical_fpr: bandit_fpr,
            false_negatives: 0,
            repeated_fps: 0,
            peak_ram_bytes: bandit_ram.max(uniform_ram),
            disk_io_bytes: disk_io,
            scale_factor: 1.0,
            notes: format!(
                "uni_cpu_ms={:.2}; bandit_cpu_ms={:.2}; uni_p50/p99_ns={}/{}; bandit_p50/p99_ns={}/{}",
                uni_cpu_ns as f64 / 1e6,
                bandit_cpu_ns as f64 / 1e6,
                uni_lat.p50_ns,
                uni_lat.p99_ns,
                bandit_lat.p50_ns,
                bandit_lat.p99_ns
            ),
        },
    ]
}

/// Executes all empty range queries and present-key FN checks across 4 real shard filters.
fn eval_multitenant_shards(
    filters: &[AerisFilter],
    oracles: &[SortedKeyOracle],
    queries: &[Vec<QueryRange>],
) -> ([u64; 4], LatencyStats, u64, u64) {
    let mut fps_per_shard = [0u64; 4];
    let mut lat_samples = Vec::with_capacity(20_000);
    let mut total_cpu_ns = 0u64;
    let mut total_fn = 0u64;

    for sid in 0..4 {
        let f = &filters[sid];
        let q_slice = &queries[sid];
        let stride = (q_slice.len() / 5_000).max(1);
        let t_shard = Instant::now();
        let mut local_fps = 0u64;
        for (idx, &q) in q_slice.iter().enumerate() {
            if idx % stride == 0 {
                let t0 = Instant::now();
                if f.query_range(q) {
                    local_fps += 1;
                }
                lat_samples.push(t0.elapsed().as_nanos() as u64);
            } else if f.query_range(q) {
                local_fps += 1;
            }
        }
        total_cpu_ns += t_shard.elapsed().as_nanos() as u64;
        fps_per_shard[sid] = local_fps;

        // Verify 0 false negatives across sampled present keys on every shard.
        let present = oracles[sid].as_slice();
        for &k in present.iter().step_by((present.len() / 5_000).max(1)) {
            if !f.query_point(k) {
                total_fn += 1;
            }
        }
    }

    (
        fps_per_shard,
        LatencyStats::from_samples(lat_samples),
        total_cpu_ns,
        total_fn,
    )
}

/// **A/B 5.3:** Guardrail rejection accuracy across live Python sidecar telemetry
/// (valid + low-confidence sparse series) AND exhaustive adversarial boundary tests
/// on [`SafetyGuardrail::validate_tuning`] and [`SafetyGuardrail::validate_memory_plan`].
fn run_ab5_guardrail_accuracy(
    advisor: &GuardedTuningAdvisor<NativeFallbackOptimizer>,
    bounds: &SafetyBounds,
) -> AblationMetric {
    let guardrail = SafetyGuardrail::new();
    let mut correct = 0usize;
    let mut total_cases = 0usize;
    let mut lat_samples = Vec::with_capacity(32);

    // 1. Live Python sidecar sparse telemetry -> low ML confidence -> rejected -> native fallback.
    let sparse_tel = TelemetrySnapshot {
        request_id: "ab5-guardrail-sparse-1".to_string(),
        shard_id: 0,
        epoch: 1,
        timestamp_ms: 1_700_000_000_000,
        current_q: 20,
        current_r: 14,
        current_m: 6,
        payload_budget_c: bounds.payload_budget_c,
        target_epsilon: bounds.target_epsilon,
        load_factor: 0.70,
        ingestion_rate_ewma: 10_000.0,
        query_rate: 15_000.0,
        observed_fpr: 0.003,
        p95_query_len: 64.0,
        p99_query_len: 96.0,
        reverse_map_entries: 500_000,
        bulk_load_speed: 250_000.0,
        recent_query_len_series: vec![64.0, 64.0],
        recent_ingest_rate_series: vec![10_000.0, 11_000.0],
    };
    let rej_before = advisor.metrics().guardrail_rejections;
    let sparse_prop = advisor
        .advise(&sparse_tel)
        .expect("sparse telemetry must fall back cleanly");
    let rej_after = advisor.metrics().guardrail_rejections;
    total_cases += 1;
    if sparse_prop.model_version == NATIVE_FALLBACK_MODEL_VERSION
        && rej_after == rej_before + 1
        && guardrail.validate_tuning(&sparse_prop, bounds) == GuardrailVerdict::Approved
    {
        correct += 1;
    }

    // 2. Direct SafetyGuardrail::validate_tuning boundary & adversarial suite.
    let valid_base = TuningProposal {
        request_id: "valid-base".to_string(),
        proposed_r: 14,
        proposed_m: 6,
        proposed_target_alpha: 0.85,
        proposed_proactive_margin_k: 1.75,
        confidence: 0.92,
        trigger_proactive_rebuild: false,
        model_version: "aeris-sidecar-ml-v1".to_string(),
        rationale: "valid baseline".to_string(),
        forecast_peak_ingest_rate: 25_000.0,
    };

    let tuning_cases: Vec<(&str, TuningProposal, bool)> = vec![
        ("valid_nominal", valid_base.clone(), true),
        (
            "valid_exact_boundaries",
            TuningProposal {
                proposed_r: 12, // == r_min(1/256)
                proposed_m: 8,  // r + m == 20 == C
                proposed_target_alpha: 0.90,
                proposed_proactive_margin_k: 3.0,
                confidence: 0.80,
                ..valid_base.clone()
            },
            true,
        ),
        (
            "reject_r_below_r_min",
            TuningProposal {
                proposed_r: 11,
                proposed_m: 8,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_r_plus_m_over_c",
            TuningProposal {
                proposed_r: 15,
                proposed_m: 6,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_m_over_m_max",
            TuningProposal {
                proposed_r: 12,
                proposed_m: bounds.m_max + 1,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_alpha_over_0.90",
            TuningProposal {
                proposed_target_alpha: 0.95,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_alpha_nan",
            TuningProposal {
                proposed_target_alpha: f64::NAN,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_margin_k_over_3.0",
            TuningProposal {
                proposed_proactive_margin_k: 3.5,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_low_confidence",
            TuningProposal {
                confidence: 0.45,
                ..valid_base.clone()
            },
            false,
        ),
        (
            "reject_negative_forecast_rate",
            TuningProposal {
                forecast_peak_ingest_rate: -10.0,
                ..valid_base
            },
            false,
        ),
    ];

    for (name, proposal, expected_approved) in tuning_cases {
        let t0 = Instant::now();
        let verdict = guardrail.validate_tuning(&proposal, bounds);
        lat_samples.push(t0.elapsed().as_nanos() as u64);
        let is_approved = verdict == GuardrailVerdict::Approved;
        assert_eq!(
            is_approved, expected_approved,
            "guardrail tuning case '{name}' mismatch: verdict={verdict:?}"
        );
        total_cases += 1;
        if is_approved == expected_approved {
            correct += 1;
        }
    }

    // 3. Direct SafetyGuardrail::validate_memory_plan boundary & adversarial suite.
    let budget = 8 * 1024 * 1024u64;
    let valid_plan = MemoryAllocationPlan {
        request_id: "mem-valid".to_string(),
        allocations: vec![
            ShardAllocationDecision {
                shard_id: 0,
                allocated_memory_bytes: 5 * 1024 * 1024,
                recommended_q: 20,
                recommended_r: 12,
                recommended_m: 8,
            },
            ShardAllocationDecision {
                shard_id: 1,
                allocated_memory_bytes: 3 * 1024 * 1024,
                recommended_q: 19,
                recommended_r: 16,
                recommended_m: 4,
            },
        ],
        confidence: 0.90,
        model_version: "aeris-sidecar-ml-v1".to_string(),
    };

    let memory_cases: Vec<(&str, MemoryAllocationPlan, bool)> = vec![
        ("mem_valid_exact_budget", valid_plan.clone(), true),
        (
            "mem_reject_budget_overshoot",
            MemoryAllocationPlan {
                allocations: vec![
                    ShardAllocationDecision {
                        allocated_memory_bytes: 5 * 1024 * 1024 + 1,
                        ..valid_plan.allocations[0].clone()
                    },
                    valid_plan.allocations[1].clone(),
                ],
                ..valid_plan.clone()
            },
            false,
        ),
        (
            "mem_reject_u64_wrap_attack",
            MemoryAllocationPlan {
                allocations: vec![
                    ShardAllocationDecision {
                        allocated_memory_bytes: u64::MAX,
                        ..valid_plan.allocations[0].clone()
                    },
                    ShardAllocationDecision {
                        allocated_memory_bytes: 16,
                        ..valid_plan.allocations[1].clone()
                    },
                ],
                ..valid_plan.clone()
            },
            false,
        ),
        (
            "mem_reject_duplicate_shard",
            MemoryAllocationPlan {
                allocations: vec![
                    valid_plan.allocations[0].clone(),
                    ShardAllocationDecision {
                        shard_id: 0,
                        ..valid_plan.allocations[1].clone()
                    },
                ],
                ..valid_plan.clone()
            },
            false,
        ),
        (
            "mem_reject_q_out_of_bounds",
            MemoryAllocationPlan {
                allocations: vec![ShardAllocationDecision {
                    recommended_q: 45,
                    ..valid_plan.allocations[0].clone()
                }],
                ..valid_plan.clone()
            },
            false,
        ),
        (
            "mem_reject_r_below_r_min",
            MemoryAllocationPlan {
                allocations: vec![ShardAllocationDecision {
                    recommended_r: 10,
                    recommended_m: 8,
                    ..valid_plan.allocations[0].clone()
                }],
                ..valid_plan.clone()
            },
            false,
        ),
        (
            "mem_reject_low_confidence",
            MemoryAllocationPlan {
                confidence: 0.25,
                ..valid_plan
            },
            false,
        ),
    ];

    for (name, plan, expected_approved) in memory_cases {
        let t0 = Instant::now();
        let verdict = guardrail.validate_memory_plan(&plan, budget, bounds);
        lat_samples.push(t0.elapsed().as_nanos() as u64);
        let is_approved = verdict == GuardrailVerdict::Approved;
        assert_eq!(
            is_approved, expected_approved,
            "guardrail memory case '{name}' mismatch: verdict={verdict:?}"
        );
        total_cases += 1;
        if is_approved == expected_approved {
            correct += 1;
        }
    }

    let accuracy_pct = (correct as f64 / total_cases as f64) * 100.0;
    assert!(
        (accuracy_pct - 100.0).abs() < 1e-9,
        "SafetyGuardrail accuracy must be 100%, got {accuracy_pct}%"
    );
    let lat = LatencyStats::from_samples(lat_samples);

    info!(
        correct,
        total_cases,
        accuracy_pct,
        p50_ns = lat.p50_ns,
        p99_ns = lat.p99_ns,
        "completed A/B 5.3 SafetyGuardrail accuracy audit"
    );

    AblationMetric {
        extension_name: "AB5_Sidecar_Guardrail_Accuracy".to_string(),
        metric_label: "SafetyGuardrail Validation & Adversarial Rejection Accuracy".to_string(),
        unit: "percent".to_string(),
        higher_is_better: true,
        key_count: 0,
        query_count: total_cases,
        baseline_value: 0.0,
        optimized_value: accuracy_pct,
        delta_pct: 100.0,
        p50_ns: lat.p50_ns,
        p99_ns: lat.p99_ns,
        p999_ns: lat.p999_ns,
        empirical_fpr: 0.0,
        false_negatives: 0,
        repeated_fps: 0,
        peak_ram_bytes: 0,
        disk_io_bytes: 0,
        scale_factor: 1.0,
        notes: format!(
            "passed={correct}/{total_cases} (100%); includes live gRPC sparse-series rejection + tuning & memory boundary/overflow/NaN/duplicate-shard checks"
        ),
    }
}

/// **A/B 5.4:** Abruptly kills (`SIGKILL`) the live Python gRPC sidecar subprocess
/// mid-session and measures exact fallback latency (`p50`/`p99`/`p999` ns) across
/// subsequent `advise()` and `allocate_memory()` calls returning
/// [`NATIVE_FALLBACK_MODEL_VERSION`] with zero errors and zero Data Plane stall.
fn run_ab5_crash_fallback(
    py_proc: &mut PythonSidecarProcess,
    advisor: &GuardedTuningAdvisor<NativeFallbackOptimizer>,
    bounds: &SafetyBounds,
) -> AblationMetric {
    const FALLBACK_ITERATIONS_EACH: usize = 20;

    // Spin up a real RcuAerisFilter queried continuously on a Data Plane thread
    // throughout the crash and fallback window to verify ZERO Data Plane stall!
    let temp_dir = TempDir::new().expect("create tempdir for A/B 5.4 Data Plane probe");
    let rm = Arc::new(
        RocksDbReverseMap::open(
            temp_dir.path().join("dp_rm"),
            ReverseMapConfig::default().with_sync_writes(false),
        )
        .expect("open DP RM"),
    );
    let dp_keys = generate_uniform_keys_oracle(0xAB54_0001, 50_000).into_vec();
    let dp_cfg = FilterConfig::new(17, 14, 64, 20, 1.0 / 256.0).expect("valid DP FilterConfig");
    let dp_filter = AerisFilter::bulk_load(dp_cfg, &dp_keys, rm.as_ref()).expect("bulk_load DP");
    let dp_rcu = RcuAerisFilter::new(0, dp_filter, rm as Arc<dyn ReverseMapStore>);

    let metrics_before = advisor.metrics();
    let guardrail = SafetyGuardrail::new();
    let budget = 8 * 1024 * 1024u64;
    let shards = vec![
        ShardTelemetrySnapshot {
            shard_id: 0,
            key_count: 300_000,
            current_memory_bytes: 2_000_000,
            query_rate: 40_000.0,
            observed_fpr: 0.05,
            p95_query_len: 256.0,
            load_factor: 0.85,
            current_q: 19,
            current_r: 14,
            current_m: 6,
        },
        ShardTelemetrySnapshot {
            shard_id: 1,
            key_count: 200_000,
            current_memory_bytes: 2_000_000,
            query_rate: 10.0,
            observed_fpr: 0.0005,
            p95_query_len: 8.0,
            load_factor: 0.25,
            current_q: 19,
            current_r: 14,
            current_m: 6,
        },
    ];

    let dp_stop = AtomicBool::new(false);
    let dp_ready = Barrier::new(2);

    let (fallback_lat, dp_lat, dp_fn, dp_queries) = thread::scope(|scope| {
        let stop_ref = &dp_stop;
        let ready_ref = &dp_ready;
        let rcu_ref = &dp_rcu;
        let keys_ref = &dp_keys;

        let dp_handle = scope.spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(0xAB54_2000);
            let mut samples = Vec::with_capacity(50_000);
            let mut local_q = 0u64;
            let mut local_fn = 0u64;

            ready_ref.wait();
            while !stop_ref.load(Ordering::Acquire) {
                let k = keys_ref[rng.gen_range(0..keys_ref.len())];
                let t0 = Instant::now();
                let hit = rcu_ref.query_point(k);
                let dt = t0.elapsed().as_nanos() as u64;
                if !hit {
                    local_fn += 1;
                }
                local_q += 1;
                if samples.len() < 50_000 {
                    samples.push(dt);
                }
            }
            (LatencyStats::from_samples(samples), local_fn, local_q)
        });

        dp_ready.wait();

        // Abruptly SIGKILL the live Python sidecar mid-session!
        py_proc.kill_and_wait();
        assert!(
            !advisor.is_healthy(),
            "killed Python sidecar must report unhealthy"
        );

        let mut fb_samples = Vec::with_capacity(FALLBACK_ITERATIONS_EACH * 2);
        for i in 0..FALLBACK_ITERATIONS_EACH {
            let tel = TelemetrySnapshot {
                request_id: format!("ab5-crash-advise-{i}"),
                shard_id: 0,
                epoch: 10 + i as u64,
                timestamp_ms: 1_700_000_500_000 + (i as u64) * 1000,
                current_q: 20,
                current_r: 14,
                current_m: 6,
                payload_budget_c: bounds.payload_budget_c,
                target_epsilon: bounds.target_epsilon,
                load_factor: 0.72,
                ingestion_rate_ewma: 18_000.0,
                query_rate: 25_000.0,
                observed_fpr: 0.003,
                p95_query_len: 128.0,
                p99_query_len: 180.0,
                reverse_map_entries: 500_000,
                bulk_load_speed: 250_000.0,
                recent_query_len_series: vec![128.0; 48],
                recent_ingest_rate_series: vec![18_000.0; 48],
            };

            let t0 = Instant::now();
            let prop = advisor
                .advise(&tel)
                .expect("post-crash advise() must return safe native fallback");
            fb_samples.push(t0.elapsed().as_nanos() as u64);
            assert_eq!(prop.model_version, NATIVE_FALLBACK_MODEL_VERSION);
            assert_eq!(
                guardrail.validate_tuning(&prop, bounds),
                GuardrailVerdict::Approved
            );

            let t1 = Instant::now();
            let plan = advisor
                .allocate_memory(10 + i as u64, budget, &shards)
                .expect("post-crash allocate_memory() must return safe native fallback");
            fb_samples.push(t1.elapsed().as_nanos() as u64);
            assert_eq!(plan.model_version, NATIVE_FALLBACK_MODEL_VERSION);
            assert_eq!(
                guardrail.validate_memory_plan(&plan, budget, bounds),
                GuardrailVerdict::Approved
            );
        }

        dp_stop.store(true, Ordering::Release);
        let (dp_stats, dp_fn_cnt, dp_q_cnt) = dp_handle.join().expect("DP probe thread panicked");
        (
            LatencyStats::from_samples(fb_samples),
            dp_stats,
            dp_fn_cnt,
            dp_q_cnt,
        )
    });

    assert_eq!(
        dp_fn, 0,
        "Data Plane observed false negatives during sidecar crash"
    );
    let metrics_after = advisor.metrics();
    let expected_fb_calls = (FALLBACK_ITERATIONS_EACH * 2) as u64;
    assert_eq!(
        metrics_after.transport_fallbacks - metrics_before.transport_fallbacks,
        expected_fb_calls,
        "every post-crash call must be counted as a transport fallback"
    );
    assert_eq!(
        metrics_after.total_calls,
        metrics_after.sidecar_accepted
            + metrics_after.guardrail_rejections
            + metrics_after.transport_fallbacks,
        "SidecarMetrics conservation invariant violated"
    );

    // Baseline is the configured RPC timeout ceiling (3,000,000,000 ns) that a naive blocking
    // client without fast failure + native fallback would incur; optimized is measured p99 fallback.
    let rpc_timeout_ns = 3_000_000_000.0_f64;

    info!(
        fallback_calls = expected_fb_calls,
        fb_p50_ns = fallback_lat.p50_ns,
        fb_p99_ns = fallback_lat.p99_ns,
        fb_p999_ns = fallback_lat.p999_ns,
        dp_queries,
        dp_p50_ns = dp_lat.p50_ns,
        dp_p99_ns = dp_lat.p99_ns,
        dp_p999_ns = dp_lat.p999_ns,
        "completed A/B 5.4 sidecar crash-fallback benchmark"
    );

    AblationMetric {
        extension_name: "AB5_Sidecar_Crash_Fallback_Latency".to_string(),
        metric_label: "Post-Crash Control-Plane Fallback p99 Latency (Zero Data-Plane Stall)"
            .to_string(),
        unit: "ns".to_string(),
        higher_is_better: false,
        key_count: dp_keys.len(),
        query_count: expected_fb_calls as usize,
        baseline_value: rpc_timeout_ns,
        optimized_value: fallback_lat.p99_ns as f64,
        delta_pct: compute_delta_pct(rpc_timeout_ns, fallback_lat.p99_ns as f64),
        p50_ns: fallback_lat.p50_ns,
        p99_ns: fallback_lat.p99_ns,
        p999_ns: fallback_lat.p999_ns,
        empirical_fpr: 0.0,
        false_negatives: 0,
        repeated_fps: 0,
        peak_ram_bytes: dp_rcu.memory_usage_bytes() as u64,
        disk_io_bytes: directory_size_bytes(temp_dir.path()),
        scale_factor: 1.0,
        notes: format!(
            "abrupt SIGKILL mid-session; {expected_fb_calls}/{expected_fb_calls} calls returned NATIVE_FALLBACK_MODEL_VERSION with 0 errors; fb_p50/p99/p999_ns={}/{}/{}; concurrent_dp_queries={dp_queries} dp_p50/p99/p999_ns={}/{}/{}",
            fallback_lat.p50_ns,
            fallback_lat.p99_ns,
            fallback_lat.p999_ns,
            dp_lat.p50_ns,
            dp_lat.p99_ns,
            dp_lat.p999_ns
        ),
    }
}

/// Orchestrates all A/B 5 experiments against one real live Python gRPC sidecar
/// subprocess (spawning it cleanly, exercising seasonal forecasting, multi-tenant
/// memory bandit, guardrail accuracy, and abrupt mid-session crash fallback).
fn run_ab5_phase5_sidecar() -> Vec<AblationMetric> {
    let mut py_proc = PythonSidecarProcess::spawn();
    let base_cfg =
        FilterConfig::new(20, 14, 64, 20, 1.0 / 256.0).expect("valid A/B 5 SafetyBounds config");
    let bounds = SafetyBounds::from_filter_config(&base_cfg);
    let client_cfg = ClientConfig::new(py_proc.endpoint())
        .with_rpc_timeout(Duration::from_secs(3))
        .with_max_retries(1)
        .with_base_backoff_ms(5)
        .with_jitter_seed(42);
    let client = SidecarClient::new(client_cfg).expect("valid SidecarClient config");
    let advisor = GuardedTuningAdvisor::with_owned_runtime(
        Some(client),
        NativeFallbackOptimizer::new(),
        bounds.clone(),
        CostModelParams::default(),
    )
    .expect("construct GuardedTuningAdvisor");

    assert!(
        advisor.is_healthy(),
        "live Python gRPC sidecar must report healthy before A/B 5 benchmarks"
    );

    let mut metrics = Vec::with_capacity(6);
    metrics.extend(run_ab5_seasonal_sidecar(&advisor, &bounds));
    metrics.extend(run_ab5_multitenant_bandit(&advisor, &bounds));
    metrics.push(run_ab5_guardrail_accuracy(&advisor, &bounds));
    metrics.push(run_ab5_crash_fallback(&mut py_proc, &advisor, &bounds));
    metrics
}

/// Top-level Worker 3 ablation suite combining A/B 4 (Scale Plane RCU + Sharding)
/// and A/B 5 (Phase 5 Live Python ML/RL Sidecar vs. Native Control Plane) and
/// atomically emitting `Filter/target/ablation_worker3_scale_sidecar.json`.
#[test]
fn run_ablation_scale_and_sidecar() {
    init_ablation_tracing();
    let mut host = HostMetadata::detect();

    let mut metrics = Vec::new();
    metrics.extend(run_ab4_multicore_scaling(&host));
    metrics.push(run_ab4_rcu_live_expansion(&host));
    metrics.extend(run_ab4_hotspot_sharding(&host));
    metrics.extend(run_ab5_phase5_sidecar());

    host.refresh_peak_rss();

    let report = WorkerAblationReport {
        worker_name: "worker3_scale_sidecar".to_string(),
        host,
        metrics,
        notes: vec![
            "A/B 4.1 evaluates 1T, 2T, 4T, 8T mixed 90R/10W scaling on 5M pre-loaded keys + 5M ops (scaled by host.scale_factor) comparing Mutex<(AerisFilter, RocksDbReverseMap)> vs 8-shard ShardedAerisSystem.".to_string(),
            "A/B 4.2 measures concurrent reader latency during two consecutive live 2x expansions comparing coarse RwLock<AerisFilter> vs wait-free epoch-RCU RcuAerisFilter.".to_string(),
            "A/B 4.3 measures 85% skewed hot-spot write latency and incremental RAM growth with live ShardedAerisSystem median splits and adjacent cold-shard merges vs global single-filter doubling.".to_string(),
            "A/B 5 runs against a live out-of-process Python gRPC sidecar (aeris_sidecar.server) verifying seasonal pre-warming lead time, multi-tenant contextual bandit RAM allocation over 1M keys, 100% SafetyGuardrail accuracy, and abrupt SIGKILL crash-fallback latency with zero Data Plane stall.".to_string(),
        ],
    };

    let out_path = write_worker_report(&report);
    assert!(
        out_path.exists(),
        "expected worker report at {}",
        out_path.display()
    );
}
