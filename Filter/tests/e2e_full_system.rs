//! End-to-End Multi-Plane Integration, SIGKILL Crash-Recovery,
//! Statistical Workload FPR, Adversarial Probing, and Deamortized Expansion
//! Verification Suite.
//!
//! Integrates all 4+1 architectural planes (`aeris-types`, `aeris-core`,
//! `aeris-storage`, `aeris-control`, `aeris-scale`, `aeris-sidecar`) over
//! seeded statistical distributions (`ChaCha8Rng`) with zero mocks/stubs and
//! zero `thread::sleep()`.

mod common;

use aeris_control::{
    ControlledAerisEngine, ControlledEngineConfig, DiscreteCostOptimizer, EstimatorConfig,
    FalsePositiveLog, FpLogConfig, IoThrottleConfig, RebuildReason,
};
use aeris_core::AerisFilter;
use aeris_scale::{ShardedAerisSystem, ShardingConfig};
use aeris_sidecar::{
    ClientConfig, GuardedTuningAdvisor, ReferenceSidecarServer, SafetyGuardrail, ServerBehavior,
    SidecarClient, SidecarEndpoint, NATIVE_FALLBACK_MODEL_VERSION, REFERENCE_SERVER_MODEL_VERSION,
};
use aeris_storage::{
    CoupledStorageEngine, KvStoreConfig, ReferenceKvStore, ReverseMapConfig, RocksDbReverseMap,
};
use aeris_types::{
    AerisError, CostModelParams, FilterConfig, GuardrailVerdict, Key, Lsn, QueryRange, RangeFilter,
    ReverseMapStore, SafetyBounds, TelemetrySnapshot, TuningAdvisor, WorkloadEstimator,
    MAX_QUOTIENT_BITS,
};
use common::{
    enclosing_range, generate_correlated_zipfian_empty_queries,
    generate_uncorrelated_zipfian_empty_queries, generate_uniform_empty_queries,
    generate_uniform_keys, init_test_tracing, is_range_empty, LatencySummary,
    DEFAULT_CORRELATION_D, DEFAULT_ZIPF_ALPHA, KEYSPACE_MASK,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tracing::info;

const E2E_SIGKILL_CHILD_ENV: &str = "AERIS_E2E_SIGKILL_CHILD";
const E2E_SIGKILL_DIR_ENV: &str = "AERIS_E2E_SIGKILL_DIR";
const E2E_SIGKILL_CYCLE_ENV: &str = "AERIS_E2E_SIGKILL_CYCLE";
const E2E_SIGKILL_CYCLES: u64 = 3;
const E2E_SIGKILL_ACKS_PER_CYCLE: usize = 30;
const E2E_SIGKILL_CHILD_CAP: usize = 40_000;

#[inline]
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sigkill_base_filter_config() -> FilterConfig {
    FilterConfig::new(8, 12, 32, 17, 1.0 / 256.0)
        .expect("valid sigkill base FilterConfig")
        .with_hash_seed(0xE2E5_1641)
}

fn sigkill_payload(key: Key) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(b"AERIS-E2E");
    out.extend_from_slice(&key.to_be_bytes());
    out.extend_from_slice(&(!key).to_le_bytes());
    out
}

fn insert_with_auto_expand(
    filter: &mut AerisFilter,
    reverse_map: &dyn ReverseMapStore,
    key: Key,
) -> aeris_types::Fingerprint {
    let mut attempts = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return fp,
            Err(AerisError::FilterFull { .. }) if attempts < MAX_QUOTIENT_BITS => {
                filter
                    .expand(reverse_map)
                    .expect("auto-expand during insert must succeed");
                attempts += 1;
            }
            Err(err) => panic!("unexpected filter insert error: {err}"),
        }
    }
}

fn adapt_with_auto_expand(
    filter: &mut AerisFilter,
    range: QueryRange,
    reverse_map: &dyn ReverseMapStore,
) {
    let mut attempts = 0u8;
    loop {
        match filter.adapt(range, reverse_map) {
            Ok(_) => return,
            Err(AerisError::FilterFull { .. }) if attempts < MAX_QUOTIENT_BITS => {
                filter
                    .expand(reverse_map)
                    .expect("auto-expand during adapt must succeed");
                attempts += 1;
            }
            Err(err) => panic!("unexpected filter adapt error: {err}"),
        }
    }
}

fn build_telemetry_snapshot(engine: &ControlledAerisEngine, request_id: &str) -> TelemetrySnapshot {
    let cfg = engine.filter().config();
    let stats = engine.estimator().snapshot();
    TelemetrySnapshot {
        request_id: request_id.to_string(),
        shard_id: 0,
        epoch: engine.epoch(),
        timestamp_ms: current_unix_ms(),
        current_q: cfg.q,
        current_r: cfg.r,
        current_m: cfg.m,
        payload_budget_c: cfg.payload_budget_c,
        target_epsilon: cfg.target_epsilon,
        load_factor: engine.filter().load_factor(),
        ingestion_rate_ewma: stats.ingestion_rate_ewma.max(25_000.0),
        query_rate: stats.query_rate.max(10_000.0),
        observed_fpr: stats.observed_fpr,
        p95_query_len: stats.p95_query_len as f64,
        p99_query_len: stats.p99_query_len as f64,
        reverse_map_entries: engine.reverse_map().entry_count_estimate(),
        bulk_load_speed: stats.bulk_load_speed,
        recent_query_len_series: vec![32.0, 32.0, 40.0, stats.p95_query_len as f64],
        recent_ingest_rate_series: vec![20_000.0, 22_000.0, 25_000.0, 28_000.0],
    }
}

/// Full-system E2E integration exercising ALL 4+1 architectural planes together:
/// - [`ControlledAerisEngine`] (Data + Storage + Deterministic Control Planes),
/// - [`ShardedAerisSystem`] (Scale Plane),
/// - [`GuardedTuningAdvisor<DiscreteCostOptimizer>`] against a live
///   [`ReferenceSidecarServer`] (Advisory Sidecar Plane).
///
/// Verifies:
/// 1. Seeded correlated Zipfian & uniform workloads discover & adapt real FPs into
///    `FalsePositiveLog`.
/// 2. Live sidecar advice approval (`Normal`), out-of-bounds guardrail rejection
///    fallback (`OutOfBoundsProposal`), and abrupt server crash fallback (`abort()`).
/// 3. Dynamic `R` shift (`32 -> 256`, `m: 5 -> 8`) triggering asynchronous background
///    rebuild with concurrent `DeltaBuffer` inserts + deletes.
/// 4. **0 false negatives** and **0 repeated FPs (100% monotonicity)** post-swap!
#[test]
fn e2e_all_planes_dynamic_r_monotonicity_and_sidecar_fallback() {
    init_test_tracing();

    let engine_dir = TempDir::new().expect("create temp dir for ControlledAerisEngine");
    let sharded_dir = TempDir::new().expect("create temp dir for ShardedAerisSystem");

    // eps = 1/32 -> r_min = 9; fixed payload budget C = 17 supports both
    // m = 5 (R = 32, r = 12) and m = 8 (R = 256, r = 9), keeping r >= 12 for
    // ReferenceSidecarServer memory allocation approval while generating real
    // false positives in Phase A for strict end-to-end monotonicity verification.
    let base_cfg =
        FilterConfig::new(11, 12, 32, 17, 1.0 / 32.0).expect("valid E2E base FilterConfig");
    let estimator_cfg = EstimatorConfig {
        window_size: 512,
        min_samples_for_clipping: 32,
        cooldown_queries: 64,
        required_sustained_evals: 2,
        hysteresis_ratio: 2.0,
        ..EstimatorConfig::default()
    };

    let cost_model = CostModelParams {
        disk_io_penalty_ns: 8_000.0,
        ..CostModelParams::default()
    };

    let mut engine_cfg = ControlledEngineConfig::new(base_cfg.clone());
    engine_cfg.estimator_config = estimator_cfg;
    engine_cfg.cost_model = cost_model.clone();
    engine_cfg.throttle_config = IoThrottleConfig::limited(25_000, 128);
    engine_cfg.fp_log_config = FpLogConfig {
        max_entries: 16_384,
        sync_on_append: false,
    };

    let mut engine = ControlledAerisEngine::open(engine_dir.path(), engine_cfg)
        .expect("open ControlledAerisEngine");

    let sharding_cfg = ShardingConfig::new(base_cfg.clone())
        .with_initial_shard_count(4)
        .with_fp_log_config(FpLogConfig {
            max_entries: 4_096,
            sync_on_append: false,
        });
    let sharded = ShardedAerisSystem::open(sharded_dir.path(), sharding_cfg)
        .expect("open ShardedAerisSystem");

    // 1. Insert 1,500 seeded keys (~73% occupancy at q=11) into both engines.
    let (initial_keys, mut live_key_set) = generate_uniform_keys(0xE2E0_1001, 1_500);
    for &k in &initial_keys {
        engine
            .insert(k, &k.to_be_bytes())
            .expect("ControlledAerisEngine initial insert");
        sharded
            .insert(k)
            .expect("ShardedAerisSystem initial insert");
    }
    engine.record_ingest_sample(initial_keys.len() as u64, 0.05);

    // 2. Run correlated Zipfian & uniform range queries (R = 32), collecting & adapting FPs.
    let uniform_queries = generate_uniform_empty_queries(0xE2E0_2001, &live_key_set, 32, 25_000);
    let corr_queries = generate_correlated_zipfian_empty_queries(
        0xE2E0_2002,
        &initial_keys,
        &live_key_set,
        32,
        DEFAULT_ZIPF_ALPHA,
        DEFAULT_CORRELATION_D,
        10_000,
    );

    let mut observed_fps: BTreeSet<QueryRange> = BTreeSet::new();
    for &q in uniform_queries.iter().chain(corr_queries.iter()) {
        let outcome = engine.query_range(q).expect("Phase A query_range failed");
        assert!(!outcome.non_empty, "empty query reported non-empty");
        if outcome.was_false_positive {
            assert!(
                observed_fps.insert(q),
                "REPEATED FP before rebuild on range [{}, {}]",
                q.start(),
                q.end()
            );
            assert!(outcome.adapted, "FP must trigger immediate adaptation");
            if sharded.query_range(q) {
                let _ = sharded
                    .adapt_range(q)
                    .expect("ShardedAerisSystem adapt_range failed");
            }
        }
    }

    assert!(
        !observed_fps.is_empty(),
        "expected >= 1 real FP during 10,000 Phase A queries"
    );
    assert_eq!(
        engine.fp_log().len(),
        observed_fps.len(),
        "FalsePositiveLog must hold every unique observed FP"
    );

    // Verify immediate 100% monotonicity before rebuild.
    for &fp_range in &observed_fps {
        let recheck = engine.query_range(fp_range).expect("pre-rebuild recheck");
        assert!(
            !recheck.filter_returned_positive && !recheck.was_false_positive,
            "pre-rebuild monotonicity violated on [{}, {}]",
            fp_range.start(),
            fp_range.end()
        );
    }

    // 3. Exercise live ReferenceSidecarServer + GuardedTuningAdvisor<DiscreteCostOptimizer>.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build Tokio runtime for live sidecar");

    let server = rt
        .block_on(ReferenceSidecarServer::start(
            SidecarEndpoint::loopback_v4(0),
            ServerBehavior::Normal,
        ))
        .expect("start ReferenceSidecarServer");

    let client_cfg = ClientConfig::new(server.endpoint().clone())
        .with_rpc_timeout(Duration::from_millis(250))
        .with_max_retries(1)
        .with_base_backoff_ms(10);
    let client = SidecarClient::new(client_cfg).expect("create SidecarClient");
    let bounds = SafetyBounds::from_filter_config(engine.filter().config());
    let advisor = GuardedTuningAdvisor::with_runtime_handle(
        Some(client),
        DiscreteCostOptimizer::new(),
        bounds.clone(),
        cost_model,
        rt.handle().clone(),
    )
    .expect("create GuardedTuningAdvisor");

    assert!(advisor.is_healthy(), "live sidecar must report healthy");

    // 3a. Valid sidecar advice approval + multi-tenant shard memory allocation.
    let telemetry_ok = build_telemetry_snapshot(&engine, "e2e-req-normal-1");
    let proposal_ok = advisor
        .advise(&telemetry_ok)
        .expect("valid sidecar advise must succeed");
    assert_eq!(proposal_ok.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&proposal_ok, &bounds),
        GuardrailVerdict::Approved
    );

    let shard_snaps = sharded.shard_telemetry_snapshots();
    let total_budget = 512 * 1024u64;
    let mem_plan = advisor
        .allocate_memory(engine.epoch(), total_budget, &shard_snaps)
        .expect("valid sidecar allocate_memory must succeed");
    assert_eq!(mem_plan.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert_eq!(mem_plan.allocations.len(), sharded.shard_count());
    let allocated_sum: u64 = mem_plan
        .allocations
        .iter()
        .map(|a| a.allocated_memory_bytes)
        .sum();
    assert!(allocated_sum <= total_budget);
    assert_eq!(advisor.metrics().sidecar_accepted, 2);

    // 3b. Out-of-bounds guardrail rejection -> deterministic native fallback.
    server.set_behavior(ServerBehavior::OutOfBoundsProposal);
    let telemetry_oob = build_telemetry_snapshot(&engine, "e2e-req-oob-2");
    let proposal_oob = advisor
        .advise(&telemetry_oob)
        .expect("guardrail rejection must fall back seamlessly");
    assert_eq!(proposal_oob.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&proposal_oob, &bounds),
        GuardrailVerdict::Approved
    );
    assert_eq!(advisor.metrics().guardrail_rejections, 1);

    // 3c. Abrupt server crash (abort) -> deterministic native fallback.
    rt.block_on(server.abort());
    let telemetry_crash = build_telemetry_snapshot(&engine, "e2e-req-crash-3");
    let proposal_crash = advisor
        .advise(&telemetry_crash)
        .expect("aborted sidecar must fall back seamlessly");
    assert_eq!(proposal_crash.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&proposal_crash, &bounds),
        GuardrailVerdict::Approved
    );
    assert_eq!(advisor.metrics().transport_fallbacks, 1);
    assert!(
        !advisor.is_healthy(),
        "aborted sidecar must report unhealthy"
    );

    // 4. Shift workload R from 32 (m = 5) to 256 (m = 8), triggering async background rebuild
    //    with concurrent DeltaBuffer inserts + deletes.
    let wide_queries = generate_uniform_empty_queries(0xE2E0_3003, &live_key_set, 256, 800);
    for &q in &wide_queries {
        let out = engine
            .query_range(q)
            .expect("wide R=256 query_range failed");
        assert!(!out.non_empty);
        if out.was_false_positive {
            observed_fps.insert(q);
        }
    }

    let decision = engine.evaluate_control_plane();
    assert!(
        decision.workload.effective_query_span >= 200,
        "expected shifted effective_query_span >= 200, got {}",
        decision.workload.effective_query_span
    );
    assert_eq!(
        decision.optimization.recommended_m, 8,
        "DiscreteCostOptimizer must select m=8 (R=256) under fixed budget C=17"
    );
    assert_eq!(decision.optimization.recommended_r, 9);
    assert!(
        decision.optimization.should_rebuild,
        "expected should_rebuild == true after R=32 -> R=256 shift"
    );

    let target_cfg = engine
        .filter()
        .config()
        .reconfigure_for_m(decision.optimization.recommended_m)
        .expect("reconfigure_for_m(8) must satisfy invariants");
    engine
        .trigger_async_rebuild(
            target_cfg,
            RebuildReason::DynamicRangeChange { old_m: 5, new_m: 8 },
        )
        .expect("trigger_async_rebuild must succeed");

    // Concurrent foreground mutations while background rebuild streams Reverse Map:
    // choose 120 brand-new delta keys disjoint from all observed_fps intervals so
    // every previously adapted empty interval remains genuinely empty!
    let mut delta_rng = ChaCha8Rng::seed_from_u64(0xDE17_A001);
    let mut delta_inserts = Vec::with_capacity(120);
    while delta_inserts.len() < 120 {
        let candidate = delta_rng.gen::<u64>() & KEYSPACE_MASK;
        if live_key_set.contains(&candidate) {
            continue;
        }
        if observed_fps.iter().any(|r| r.contains(candidate)) {
            continue;
        }
        live_key_set.insert(candidate);
        delta_inserts.push(candidate);
    }

    for &k in &delta_inserts {
        engine
            .insert(k, &k.to_be_bytes())
            .expect("concurrent delta insert failed");
    }

    let delta_deletes: Vec<Key> = initial_keys.iter().copied().take(40).collect();
    for &k in &delta_deletes {
        let deleted = engine.delete(k).expect("concurrent delta delete failed");
        assert!(deleted, "expected initial key {k} to be deleted");
        live_key_set.remove(&k);
    }

    // Complete background rebuild, draining DeltaBuffer + replaying FalsePositiveLog.
    // Concurrent foreground inserts that commit before the background worker opens its
    // RocksDB iterator snapshot are loaded in `keys_loaded` and idempotently deduplicated
    // during delta drain, preserving exact key conservation with zero duplicates!
    let rebuild_report = engine
        .finish_rebuild()
        .expect("finish_rebuild must succeed");
    assert_eq!(engine.filter().config().m, 8);
    assert_eq!(engine.filter().config().r, 9);
    assert_eq!(engine.filter().config().max_range_r, 256);
    assert_eq!(
        rebuild_report.keys_loaded + rebuild_report.delta_inserts_applied
            - rebuild_report.delta_deletes_applied,
        live_key_set.len(),
        "exact live key conservation violated across snapshot + DeltaBuffer drain"
    );
    assert_eq!(
        engine.filter().len(),
        live_key_set.len(),
        "rebuilt filter key cardinality must match exact surviving key set"
    );
    assert!(
        rebuild_report.delta_inserts_applied > 0,
        "expected concurrent foreground inserts to be drained from DeltaBuffer"
    );
    assert!(
        rebuild_report.delta_deletes_applied > 0,
        "expected concurrent foreground deletes to be drained from DeltaBuffer"
    );

    // 5. Verify 0 false negatives across 100% of surviving keys (initial - deleted + delta).
    for &k in &live_key_set {
        assert!(
            engine.filter().query_point(k),
            "false negative on query_point({k}) after dynamic-R rebuild"
        );
        assert!(
            engine.filter().query_range(enclosing_range(k, 32)),
            "false negative on enclosing range for {k} after dynamic-R rebuild"
        );
    }

    // 6. Verify 0 repeated FPs (100% monotonicity) across ALL previously observed FPs on F_new.
    for &fp_range in &observed_fps {
        assert!(
            is_range_empty(&live_key_set, fp_range),
            "test invariant bug: delta key overlapped observed FP"
        );
        let outcome = engine
            .query_range(fp_range)
            .expect("post-rebuild FP monotonicity recheck");
        assert!(
            !outcome.filter_returned_positive && !outcome.was_false_positive,
            "MONOTONICITY VIOLATED post dynamic-R swap on range [{}, {}]",
            fp_range.start(),
            fp_range.end()
        );
    }

    engine
        .filter()
        .verify_invariants()
        .expect("rebuilt AerisFilter invariants violated");
    sharded
        .verify_all_invariants()
        .expect("ShardedAerisSystem invariants violated");

    info!(
        test = "e2e_all_planes_dynamic_r_monotonicity_and_sidecar_fallback",
        surviving_keys = live_key_set.len(),
        observed_fps = observed_fps.len(),
        delta_inserts = rebuild_report.delta_inserts_applied,
        delta_deletes = rebuild_report.delta_deletes_applied,
        new_m = engine.filter().config().m,
        new_r = engine.filter().config().r,
        sidecar_accepted = advisor.metrics().sidecar_accepted,
        guardrail_rejections = advisor.metrics().guardrail_rejections,
        transport_fallbacks = advisor.metrics().transport_fallbacks,
        "all 4+1 planes E2E passed with 0 FNs and 100% FP monotonicity"
    );
}

/// Env-gated subprocess driver for [`e2e_full_stack_sigkill_crash_recovery`].
///
/// Under normal `cargo test`, returns immediately. When spawned by the parent
/// with `AERIS_E2E_SIGKILL_CHILD=1`, opens [`CoupledStorageEngine`], rebuilds a
/// real [`AerisFilter`], writes keys through BOTH `AerisFilter` and
/// `CoupledStorageEngine::put` (WAL-first `fsync`), and emits `ACK:<lsn>:<key>`
/// to stdout until violently terminated by `child.kill()`.
#[test]
fn e2e_sigkill_child_driver() {
    if env::var_os(E2E_SIGKILL_CHILD_ENV).is_none() {
        return;
    }

    let dir = env::var(E2E_SIGKILL_DIR_ENV).expect("AERIS_E2E_SIGKILL_DIR missing");
    let cycle: u64 = env::var(E2E_SIGKILL_CYCLE_ENV)
        .expect("AERIS_E2E_SIGKILL_CYCLE missing")
        .parse()
        .expect("AERIS_E2E_SIGKILL_CYCLE invalid u64");

    let (engine, _recovery) =
        CoupledStorageEngine::open_and_recover(&dir).expect("child open_and_recover failed");
    let mut filter = AerisFilter::new(sigkill_base_filter_config()).expect("child filter init");
    let _ = engine
        .rebuild_filter_and_reconcile(&mut filter)
        .expect("child rebuild_filter_and_reconcile failed");

    let mut rng = ChaCha8Rng::seed_from_u64(0xF011_57AC_u64.wrapping_add(cycle.wrapping_mul(131)));
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for _ in 0..E2E_SIGKILL_CHILD_CAP {
        let key: Key = ((cycle + 1) << 40) | (rng.gen::<u64>() & 0x0000_00FF_FFFF_FFFF);
        let payload = sigkill_payload(key);
        let fp = insert_with_auto_expand(&mut filter, engine.reverse_map(), key);
        let lsn = engine
            .put(key, &payload, &fp)
            .expect("child engine.put failed");

        writeln!(out, "ACK:{lsn}:{key}").expect("child ACK write failed");
        out.flush().expect("child ACK flush failed");
    }
}

/// Real subprocess `SIGKILL` / `TerminateProcess` (`child.kill()`) fault-injection
/// across 3 consecutive crash cycles through BOTH [`AerisFilter`] and
/// [`CoupledStorageEngine`].
///
/// Verifies after EVERY crash cycle:
/// - **0 lost keys** in [`ReferenceKvStore`] and [`RocksDbReverseMap`].
/// - **0 false negatives** on the recovered [`AerisFilter`] for both
///   `query_point(k)` and enclosing `query_range`.
#[test]
fn e2e_full_stack_sigkill_crash_recovery() {
    init_test_tracing();

    let dir = TempDir::new().expect("create temp dir for full-stack SIGKILL recovery");
    let exe = env::current_exe().expect("resolve current test binary");
    let mut all_acked: BTreeMap<Key, Lsn> = BTreeMap::new();
    let mut prev_lsn: Lsn = 0;

    for cycle in 0..E2E_SIGKILL_CYCLES {
        let mut cmd = Command::new(&exe);
        cmd.arg("--exact")
            .arg("e2e_sigkill_child_driver")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(E2E_SIGKILL_CHILD_ENV, "1")
            .env(E2E_SIGKILL_DIR_ENV, dir.path())
            .env(E2E_SIGKILL_CYCLE_ENV, cycle.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().expect("spawn full-stack SIGKILL child");
        let stdout = child.stdout.take().expect("child stdout missing");
        let mut reader = BufReader::new(stdout);

        let mut cycle_acks = 0usize;
        let mut line = String::new();
        while cycle_acks < E2E_SIGKILL_ACKS_PER_CYCLE {
            line.clear();
            let n = reader.read_line(&mut line).expect("read child ACK line");
            assert!(
                n > 0,
                "SIGKILL child exited prematurely on cycle {cycle} after {cycle_acks} ACKs"
            );
            if let Some((lsn, key)) = parse_ack_line(&line) {
                all_acked.insert(key, lsn);
                cycle_acks += 1;
            }
        }

        // Violently terminate child mid-write (`TerminateProcess` / `SIGKILL`).
        child.kill().expect("child.kill() failed");
        let status = child.wait().expect("child.wait() failed");
        assert!(
            !status.success(),
            "cycle {cycle}: killed child unexpectedly returned success {status:?}"
        );

        // Drain any complete ACK lines already flushed into the OS pipe before death.
        loop {
            line.clear();
            let n = reader.read_line(&mut line).expect("drain child stdout");
            if n == 0 {
                break;
            }
            if line.ends_with('\n') {
                if let Some((lsn, key)) = parse_ack_line(&line) {
                    all_acked.insert(key, lsn);
                }
            }
        }

        // Recover full storage engine and rebuild AerisFilter with Reverse Map reconciliation.
        let (engine, recovery) = CoupledStorageEngine::open_and_recover(dir.path())
            .unwrap_or_else(|err| panic!("cycle {cycle} open_and_recover failed: {err}"));
        assert!(
            recovery.last_lsn > prev_lsn,
            "cycle {cycle}: LSN did not advance ({prev_lsn} -> {})",
            recovery.last_lsn
        );
        prev_lsn = recovery.last_lsn;

        let mut recovered_filter =
            AerisFilter::new(sigkill_base_filter_config()).expect("create recovered filter");
        let loaded = engine
            .rebuild_filter_and_reconcile(&mut recovered_filter)
            .unwrap_or_else(|err| {
                panic!("cycle {cycle} rebuild_filter_and_reconcile failed: {err}")
            });

        let kv_keys = engine.kv_store().scan_all_keys().expect("scan kv_store");
        let rm_keys = engine
            .reverse_map()
            .scan_all_keys()
            .expect("scan reverse_map");
        assert_eq!(loaded, kv_keys.len());
        assert_eq!(
            kv_keys, rm_keys,
            "cycle {cycle}: KV and Reverse Map diverged"
        );
        let rm_set: BTreeSet<Key> = rm_keys.iter().copied().collect();

        // Verify 0 lost keys in ReferenceKvStore & RocksDbReverseMap AND 0 FNs on AerisFilter.
        for (&acked_key, &acked_lsn) in &all_acked {
            let expected_val = sigkill_payload(acked_key);
            let actual_val = engine
                .kv_store()
                .get(acked_key)
                .expect("kv_store.get after recovery");
            assert_eq!(
                actual_val.as_deref(),
                Some(expected_val.as_slice()),
                "cycle {cycle}: LOST key {acked_key} (LSN {acked_lsn}) in ReferenceKvStore"
            );
            assert!(
                rm_set.contains(&acked_key),
                "cycle {cycle}: LOST key {acked_key} (LSN {acked_lsn}) in RocksDbReverseMap"
            );
            assert!(
                recovered_filter.query_point(acked_key),
                "cycle {cycle}: FALSE NEGATIVE on recovered AerisFilter::query_point({acked_key})"
            );
            assert!(
                recovered_filter.query_range(enclosing_range(acked_key, 16)),
                "cycle {cycle}: FALSE NEGATIVE on recovered AerisFilter::query_range for {acked_key}"
            );
        }

        // Also check 0 FNs for any unacked-but-durable WAL keys recovered from disk.
        for &live_key in &kv_keys {
            assert!(recovered_filter.query_point(live_key));
            assert!(recovered_filter.query_range(enclosing_range(live_key, 16)));
        }
        recovered_filter
            .verify_invariants()
            .expect("recovered AerisFilter invariants violated");

        info!(
            test = "e2e_full_stack_sigkill_crash_recovery",
            cycle,
            acked_keys = all_acked.len(),
            live_keys = kv_keys.len(),
            replayed_records = recovery.replayed_records,
            truncated_tail_bytes = recovery.truncated_tail_bytes,
            last_lsn = recovery.last_lsn,
            filter_q = recovered_filter.config().q,
            "crash cycle recovered cleanly with 0 lost keys and 0 false negatives"
        );
    }

    assert!(all_acked.len() >= (E2E_SIGKILL_CYCLES as usize) * E2E_SIGKILL_ACKS_PER_CYCLE);
}

fn parse_ack_line(line: &str) -> Option<(Lsn, Key)> {
    let rest = line.trim().strip_prefix("ACK:")?;
    let (lsn_str, key_str) = rest.split_once(':')?;
    Some((lsn_str.parse().ok()?, key_str.parse().ok()?))
}

/// Evaluates seeded statistical workloads (`ChaCha8Rng`, `10,000` keys, `20,000`
/// empty queries per distribution) across:
/// 1. Uniform random (`R = 1`, `R = 32`, `R = 1024`),
/// 2. Zipfian correlated (`alpha = 1.5, D = 0.8` matching `Aeris Filters.pdf` Section 5.2),
/// 3. Zipfian uncorrelated (`alpha = 1.5`),
/// 4. Shifting `R` phases (`R = 16 -> 256 -> 32`, `30,000` empty queries total).
///
/// Uses real [`RocksDbReverseMap`] for all adaptations and asserts **0 false negatives**,
/// **0 repeated false positives**, and **`empirical_fpr <= target_epsilon`**.
#[test]
fn e2e_statistical_workloads_fpr_and_latency_report() {
    init_test_tracing();

    const NUM_KEYS: usize = 10_000;
    const QUERIES_PER_DIST: usize = 20_000;
    const TARGET_EPSILON: f64 = 1.0 / 256.0; // 0.00390625, r_min = 12

    let (keys, key_set) = generate_uniform_keys(0x57A7_0001, NUM_KEYS);

    // 1. Uniform random: R = 1 (m=0), R = 32 (m=5), R = 1024 (m=10).
    let uniform_specs: &[(&str, u64, u8, u64)] = &[
        ("uniform_r1", 1, 12, 0x57A7_1001),
        ("uniform_r32", 32, 17, 0x57A7_1002),
        ("uniform_r1024", 1024, 22, 0x57A7_1003),
    ];

    for &(label, range_len, budget_c, q_seed) in uniform_specs {
        let cfg = FilterConfig::new(14, 12, range_len, budget_c, TARGET_EPSILON)
            .expect("valid uniform FilterConfig");
        let queries = generate_uniform_empty_queries(q_seed, &key_set, range_len, QUERIES_PER_DIST);
        run_statistical_distribution_check(label, cfg, &keys, &queries);
    }

    // 2. Zipfian correlated (alpha = 1.5, D = 0.8, R = 32).
    {
        let cfg = FilterConfig::new(14, 12, 32, 17, TARGET_EPSILON)
            .expect("valid zipfian correlated FilterConfig");
        let queries = generate_correlated_zipfian_empty_queries(
            0x57A7_2001,
            &keys,
            &key_set,
            32,
            DEFAULT_ZIPF_ALPHA,
            DEFAULT_CORRELATION_D,
            QUERIES_PER_DIST,
        );
        run_statistical_distribution_check(
            "zipfian_correlated_a1p5_d0p8_r32",
            cfg,
            &keys,
            &queries,
        );
    }

    // 3. Zipfian uncorrelated (alpha = 1.5, R = 32).
    {
        let cfg = FilterConfig::new(14, 12, 32, 17, TARGET_EPSILON)
            .expect("valid zipfian uncorrelated FilterConfig");
        let queries = generate_uncorrelated_zipfian_empty_queries(
            0x57A7_3001,
            &key_set,
            32,
            DEFAULT_ZIPF_ALPHA,
            4_096,
            QUERIES_PER_DIST,
        );
        run_statistical_distribution_check("zipfian_uncorrelated_a1p5_r32", cfg, &keys, &queries);
    }

    // 4. Shifting R phases (R = 16 -> 256 -> 32) under fixed budget C = 20 with real FP Log replay.
    {
        let temp_dir = TempDir::new().expect("create temp dir for shifting-R statistical workload");
        let fp_log = FalsePositiveLog::open(
            temp_dir.path().join("shift_fp.log"),
            FpLogConfig {
                max_entries: 16_384,
                sync_on_append: false,
            },
        )
        .expect("open shift FalsePositiveLog");

        let mut current_cfg = FilterConfig::new(14, 16, 16, 20, TARGET_EPSILON)
            .expect("valid Phase 1 shifting-R FilterConfig");
        let mut rm = RocksDbReverseMap::open(
            temp_dir.path().join("rm_phase_0"),
            ReverseMapConfig::default(),
        )
        .expect("open phase 0 Reverse Map");
        let mut filter =
            AerisFilter::bulk_load(current_cfg.clone(), &keys, &rm).expect("bulk_load phase 0");

        let phases: &[(usize, u64, u8, u64)] = &[
            (0, 16, 4, 0x57A7_4001),
            (1, 256, 8, 0x57A7_4002),
            (2, 32, 5, 0x57A7_4003),
        ];
        let mut adapted_across_phases: HashSet<QueryRange> = HashSet::new();
        let mut total_queries = 0usize;
        let mut total_fps = 0usize;

        for &(phase_idx, phase_r, target_m, q_seed) in phases {
            if phase_idx > 0 {
                current_cfg = current_cfg
                    .reconfigure_for_m(target_m)
                    .expect("reconfigure_for_m must preserve budget C=20");
                let next_rm = RocksDbReverseMap::open(
                    temp_dir.path().join(format!("rm_phase_{phase_idx}")),
                    ReverseMapConfig::default(),
                )
                .expect("open phase Reverse Map");
                let mut rebuilt =
                    AerisFilter::bulk_load(current_cfg.clone(), &keys, &next_rm).expect("rebuild");
                let _ = fp_log
                    .replay_onto(&mut rebuilt, &next_rm, |r| Ok(is_range_empty(&key_set, r)))
                    .expect("FP Log replay across R phase transition");
                filter = rebuilt;
                rm = next_rm;

                // Confirm 100% monotonicity across phase boundary BEFORE running new queries.
                for &prev_fp in &adapted_across_phases {
                    assert!(
                        !filter.query_range(prev_fp),
                        "phase {phase_idx}: repeated FP after R shift on [{}, {}]",
                        prev_fp.start(),
                        prev_fp.end()
                    );
                }
            }

            let queries = generate_uniform_empty_queries(q_seed, &key_set, phase_r, 10_000);
            let mut phase_fps = 0usize;
            for &q in &queries {
                if filter.query_range(q) {
                    assert!(
                        !adapted_across_phases.contains(&q),
                        "phase {phase_idx}: repeated FP on [{}, {}]",
                        q.start(),
                        q.end()
                    );
                    fp_log
                        .record(q, current_unix_ms())
                        .expect("record shifting-R FP");
                    adapt_with_auto_expand(&mut filter, q, &rm);
                    assert!(!filter.query_range(q));
                    adapted_across_phases.insert(q);
                    phase_fps += 1;
                }
            }
            let phase_fpr = (phase_fps as f64) / (queries.len() as f64);
            assert!(
                phase_fpr <= TARGET_EPSILON,
                "shifting-R phase {phase_idx} (R={phase_r}) FPR {phase_fpr:.6} exceeded target {TARGET_EPSILON:.6}"
            );
            total_queries += queries.len();
            total_fps += phase_fps;
        }

        for &k in &keys {
            assert!(filter.query_point(k), "shifting-R FN on query_point({k})");
            assert!(filter.query_range(enclosing_range(k, 16)));
        }
        for &prev_fp in &adapted_across_phases {
            assert!(!filter.query_range(prev_fp), "shifting-R final repeated FP");
        }
        let overall_fpr = (total_fps as f64) / (total_queries as f64);
        assert!(overall_fpr <= TARGET_EPSILON);
        info!(
            workload = "shifting_r_16_256_32",
            keys = NUM_KEYS,
            queries = total_queries,
            false_positives = total_fps,
            empirical_fpr = format!("{overall_fpr:.6}"),
            target_epsilon = format!("{TARGET_EPSILON:.6}"),
            false_negatives = 0,
            repeated_fps = 0,
            "statistical shifting-R workload report"
        );
    }
}

fn run_statistical_distribution_check(
    label: &str,
    cfg: FilterConfig,
    keys: &[Key],
    empty_queries: &[QueryRange],
) {
    let temp_dir = TempDir::new().expect("create temp dir for statistical distribution");
    let rm = RocksDbReverseMap::open(temp_dir.path().join("rm"), ReverseMapConfig::default())
        .expect("open RocksDbReverseMap");
    let target_epsilon = cfg.target_epsilon;
    let mut filter = AerisFilter::bulk_load(cfg, keys, &rm).expect("bulk_load statistical filter");

    let mut adapted_set: HashSet<QueryRange> = HashSet::new();
    let mut false_positives = 0usize;
    let mut latencies_ns = Vec::with_capacity(empty_queries.len().min(4_096));
    let sample_stride = (empty_queries.len() / 4_096).max(1);

    let t0 = Instant::now();
    for (idx, &q) in empty_queries.iter().enumerate() {
        let q_start = if idx.is_multiple_of(sample_stride) {
            Some(Instant::now())
        } else {
            None
        };
        let hit = filter.query_range(q);
        if let Some(s) = q_start {
            latencies_ns.push(s.elapsed().as_nanos() as u64);
        }

        if hit {
            assert!(
                !adapted_set.contains(&q),
                "{label}: REPEATED FALSE POSITIVE on already-adapted range [{}, {}]",
                q.start(),
                q.end()
            );
            adapt_with_auto_expand(&mut filter, q, &rm);
            assert!(
                !filter.query_range(q),
                "{label}: adapted range immediately returned true"
            );
            adapted_set.insert(q);
            false_positives += 1;
        }
    }
    let elapsed_secs = t0.elapsed().as_secs_f64().max(1e-9);

    // Exhaustive 0-FN check across 100% of inserted keys.
    for &k in keys {
        assert!(
            filter.query_point(k),
            "{label}: false negative on query_point({k})"
        );
        assert!(
            filter.query_range(enclosing_range(k, 8)),
            "{label}: false negative on enclosing range for {k}"
        );
    }

    // Exhaustive 0-repeated-FP monotonicity check across all adapted intervals.
    for &fp_range in &adapted_set {
        assert!(
            !filter.query_range(fp_range),
            "{label}: post-run repeated FP on [{}, {}]",
            fp_range.start(),
            fp_range.end()
        );
    }

    let empirical_fpr = (false_positives as f64) / (empty_queries.len() as f64);
    assert!(
        empirical_fpr <= target_epsilon,
        "{label}: empirical FPR {empirical_fpr:.6} exceeded target epsilon {target_epsilon:.6} ({false_positives}/{})",
        empty_queries.len()
    );
    filter
        .verify_invariants()
        .unwrap_or_else(|e| panic!("{label}: filter invariants failed: {e}"));

    let lat = LatencySummary::from_samples(latencies_ns);
    let qps = (empty_queries.len() as f64) / elapsed_secs;
    let mem_bytes = filter.memory_usage_bytes();
    let bits_per_key = (mem_bytes as f64 * 8.0) / (keys.len() as f64);

    info!(
        workload = label,
        keys = keys.len(),
        queries = empty_queries.len(),
        false_positives,
        empirical_fpr = format!("{empirical_fpr:.6}"),
        target_epsilon = format!("{target_epsilon:.6}"),
        qps = format!("{qps:.0}"),
        p50_ns = lat.p50_ns,
        p99_ns = lat.p99_ns,
        memory_bytes = mem_bytes,
        bits_per_key = format!("{bits_per_key:.2}"),
        false_negatives = 0,
        repeated_fps = 0,
        "statistical workload report"
    );
}

/// Implements the exact two-phase adversarial latency-probing workload from
/// `Aeris Filters.pdf` Section 5.1:
/// - **Phase 1 (Warm-up):** Discovers genuine false positives against real
///   `AerisFilter` + `RocksDbReverseMap` + `ReferenceKvStore` vs a non-adaptive
///   `AerisFilter` baseline (queried without `adapt()`).
/// - **Phase 2 (Adversarial):** Injects 10% repeat queries of discovered FPs
///   alongside 90% fresh empty queries.
///
/// Asserts adaptive Aeris incurs **0 repeated FPs** during the adversarial phase
/// and performs dramatically fewer backing [`ReferenceKvStore`] disk checks than
/// the non-adaptive baseline!
#[test]
fn e2e_adversarial_latency_probing_stability() {
    init_test_tracing();

    const NUM_KEYS: usize = 10_000;
    const WARMUP_QUERIES: usize = 20_000;
    const ADVERSARIAL_QUERIES: usize = 20_000;
    const ADVERSARIAL_REPEAT_COUNT: usize = ADVERSARIAL_QUERIES / 10; // exact 10% = 2,000

    let temp_dir = TempDir::new().expect("create temp dir for adversarial test");
    let kv_store = ReferenceKvStore::open_with_config(
        temp_dir.path().join("kv_db"),
        KvStoreConfig::for_wal_coupled(),
    )
    .expect("open ReferenceKvStore");
    let rm = RocksDbReverseMap::open(
        temp_dir.path().join("reverse_map"),
        ReverseMapConfig::default(),
    )
    .expect("open RocksDbReverseMap");

    let (keys, key_set) = generate_uniform_keys(0xAD7E_5001, NUM_KEYS);
    for &k in &keys {
        kv_store
            .put(k, &k.to_be_bytes())
            .expect("populate ReferenceKvStore");
    }
    kv_store.flush().expect("flush ReferenceKvStore");

    // eps = 1/64 -> r_min = 10, m = 5 (R = 32), C = 15: yields rich real FPs in warm-up.
    let cfg =
        FilterConfig::new(14, 10, 32, 15, 1.0 / 64.0).expect("valid adversarial FilterConfig");
    let mut adaptive_filter =
        AerisFilter::bulk_load(cfg, &keys, &rm).expect("bulk_load adaptive filter");
    let baseline_filter = adaptive_filter.clone();

    // Phase 1: Warm-up discovers FPs.
    let warmup = generate_uniform_empty_queries(0xAD7E_5002, &key_set, 32, WARMUP_QUERIES);
    let mut discovered_fps: Vec<QueryRange> = Vec::new();
    let mut discovered_set: HashSet<QueryRange> = HashSet::new();

    for &q in &warmup {
        if baseline_filter.query_range(q) {
            let non_empty = kv_store.range_non_empty(q).expect("baseline warm-up check");
            assert!(!non_empty);
            if discovered_set.insert(q) {
                discovered_fps.push(q);
            }
        }
        if adaptive_filter.query_range(q) {
            let non_empty = kv_store.range_non_empty(q).expect("adaptive warm-up check");
            assert!(!non_empty);
            adapt_with_auto_expand(&mut adaptive_filter, q, &rm);
        }
    }

    assert!(
        discovered_fps.len() >= 10,
        "expected >= 10 real warm-up FPs for adversarial pool, got {}",
        discovered_fps.len()
    );

    // Phase 2: Build deterministic 20,000-query adversarial stream (10% repeat FPs, 90% fresh).
    let fresh = generate_uniform_empty_queries(
        0xAD7E_5003,
        &key_set,
        32,
        ADVERSARIAL_QUERIES - ADVERSARIAL_REPEAT_COUNT,
    );
    let mut adv_rng = ChaCha8Rng::seed_from_u64(0xAD7E_5004);
    let mut adversarial_stream: Vec<(QueryRange, bool)> = Vec::with_capacity(ADVERSARIAL_QUERIES);
    let mut fresh_idx = 0usize;
    for step in 0..ADVERSARIAL_QUERIES {
        if step.is_multiple_of(10) {
            let fp = discovered_fps[adv_rng.gen_range(0..discovered_fps.len())];
            adversarial_stream.push((fp, true));
        } else {
            adversarial_stream.push((fresh[fresh_idx], false));
            fresh_idx += 1;
        }
    }

    let mut adaptive_disk_checks = 0usize;
    let mut adaptive_repeated_fps = 0usize;
    let mut adaptive_new_fps = 0usize;
    let adaptive_t0 = Instant::now();
    for &(q, is_injected_repeat) in &adversarial_stream {
        if adaptive_filter.query_range(q) {
            adaptive_disk_checks += 1;
            let non_empty = kv_store.range_non_empty(q).expect("adaptive kv check");
            assert!(!non_empty);
            if is_injected_repeat {
                adaptive_repeated_fps += 1;
            } else {
                adaptive_new_fps += 1;
            }
            adapt_with_auto_expand(&mut adaptive_filter, q, &rm);
        }
    }
    let adaptive_ms = adaptive_t0.elapsed().as_secs_f64() * 1_000.0;

    let mut baseline_disk_checks = 0usize;
    let mut baseline_repeated_fps = 0usize;
    let baseline_t0 = Instant::now();
    for &(q, is_injected_repeat) in &adversarial_stream {
        if baseline_filter.query_range(q) {
            baseline_disk_checks += 1;
            let non_empty = kv_store.range_non_empty(q).expect("baseline kv check");
            assert!(!non_empty);
            if is_injected_repeat {
                baseline_repeated_fps += 1;
            }
        }
    }
    let baseline_ms = baseline_t0.elapsed().as_secs_f64() * 1_000.0;

    // Assert Section 5.1 adversarial stability guarantees.
    assert_eq!(
        adaptive_repeated_fps, 0,
        "adaptive Aeris must incur 0 repeated FPs during adversarial phase"
    );
    assert_eq!(
        baseline_repeated_fps, ADVERSARIAL_REPEAT_COUNT,
        "non-adaptive baseline should fail every injected warm-up FP"
    );
    assert!(
        adaptive_disk_checks * 10 < baseline_disk_checks,
        "expected >10x fewer backing ReferenceKvStore disk checks for adaptive Aeris (adaptive={adaptive_disk_checks}, baseline={baseline_disk_checks})"
    );

    // Exhaustive 0-FN check on adaptive Aeris.
    for &k in &keys {
        assert!(adaptive_filter.query_point(k));
    }

    let disk_reduction = (baseline_disk_checks as f64) / (adaptive_disk_checks.max(1) as f64);
    info!(
        test = "e2e_adversarial_latency_probing_stability",
        warmup_fps = discovered_fps.len(),
        adversarial_queries = ADVERSARIAL_QUERIES,
        injected_repeats = ADVERSARIAL_REPEAT_COUNT,
        adaptive_repeated_fps,
        adaptive_new_fps,
        adaptive_disk_checks,
        baseline_repeated_fps,
        baseline_disk_checks,
        disk_check_reduction = format!("{disk_reduction:.1}x"),
        adaptive_phase_ms = format!("{adaptive_ms:.2}"),
        baseline_phase_ms = format!("{baseline_ms:.2}"),
        "adversarial probing stability verified"
    );
}

/// Runs 8 consecutive `2x` expansions (`q = 7 -> 15`) over real [`RocksDbReverseMap`]
/// comparing [`AerisFilter::expand`] (InfiniFilter bit-splitting + selective
/// rejuvenation) against a full-rebuild baseline ([`AerisFilter::rebuild_from_keys`]
/// on every expansion).
///
/// Verifies:
/// - **0 false negatives** across all 8,192 keys.
/// - **0 repeated FPs** (100% monotonicity across 8 expansions including selective
///   rejuvenation of adapted boxes!).
/// - **Empirical `FPR <= target_epsilon`** after all 8 expansions.
/// - **`total_boxes_rejuvenated > 0`**.
/// - **Deamortized `expand` finishes strictly faster than full-rebuild baseline**
///   across the 8 cycles!
#[test]
fn e2e_deamortized_expansion_speedup_vs_full_rebuild() {
    init_test_tracing();

    // eps = 1/16 (0.0625) -> r_min = 7; after 7 expansions initial boxes reach age = 7 == r,
    // triggering real selective rejuvenation on expansion #8 (q = 14 -> 15).
    const TARGET_EPSILON: f64 = 1.0 / 16.0;
    const INITIAL_Q: u8 = 7;
    const EXPANSION_STEPS: usize = 8;

    let initial_cfg =
        FilterConfig::new(INITIAL_Q, 7, 32, 12, TARGET_EPSILON).expect("valid expansion config");

    // Pre-generate 8 deterministic key batches totaling 8,192 unique keys (filling to ~50% per q).
    let batch_sizes: [usize; EXPANSION_STEPS] = [64, 64, 128, 256, 512, 1_024, 2_048, 4_096];
    let total_keys: usize = batch_sizes.iter().sum();
    let (all_keys_sorted, all_key_set) = generate_uniform_keys(0xE9A0_8001, total_keys);

    // Shuffle deterministically so each expansion batch spans the full keyspace uniformly.
    let mut shuffled_keys = all_keys_sorted.clone();
    let mut shuffle_rng = ChaCha8Rng::seed_from_u64(0xE9A0_8002);
    for i in (1..shuffled_keys.len()).rev() {
        let j = shuffle_rng.gen_range(0..=i);
        shuffled_keys.swap(i, j);
    }

    let mut batches: Vec<&[Key]> = Vec::with_capacity(EXPANSION_STEPS);
    let mut offset = 0usize;
    for &sz in &batch_sizes {
        batches.push(&shuffled_keys[offset..offset + sz]);
        offset += sz;
    }

    // --- Path A: Deamortized InfiniFilter bit-splitting + selective rejuvenation (`expand`). ---
    let deamort_dir = TempDir::new().expect("create temp dir for deamortized expansion");
    let deamort_rm =
        RocksDbReverseMap::open(deamort_dir.path().join("rm"), ReverseMapConfig::default())
            .expect("open deamortized Reverse Map");
    let mut deamort_filter =
        AerisFilter::new(initial_cfg.clone()).expect("create deamortized filter");

    let fp_candidates = generate_uniform_empty_queries(0xE9A0_8003, &all_key_set, 32, 1_500);
    let mut adapted_fps: Vec<QueryRange> = Vec::new();
    let mut adapted_set: HashSet<QueryRange> = HashSet::new();

    let mut deamort_expand_ns = 0u64;
    let mut total_boxes_transferred = 0usize;
    let mut total_boxes_rejuvenated = 0usize;
    let mut total_keys_rejuvenated = 0usize;

    for (step, &batch) in batches.iter().enumerate() {
        let mut rm_batch = Vec::with_capacity(batch.len());
        for &k in batch {
            let fp = deamort_filter
                .insert(k)
                .expect("deamortized batch insert must fit before expansion");
            rm_batch.push((fp, k));
        }
        deamort_rm
            .append_batch(&rm_batch)
            .expect("deamortized Reverse Map append_batch");

        // Adapt real FPs at q = 7 (boxes will hit age == 7 and undergo selective rejuvenation
        // at step 7!) and at q = 10 to prove 100% monotonicity across rejuvenation.
        if step == 0 || step == 3 {
            let mut step_adapted = 0usize;
            for &q in &fp_candidates {
                if adapted_set.contains(&q) {
                    continue;
                }
                if deamort_filter.query_range(q) {
                    deamort_filter
                        .adapt(q, &deamort_rm)
                        .expect("adapt before expansion");
                    assert!(!deamort_filter.query_range(q));
                    adapted_set.insert(q);
                    adapted_fps.push(q);
                    step_adapted += 1;
                    if step_adapted >= 6 {
                        break;
                    }
                }
            }
        }

        let t0 = Instant::now();
        let rep = deamort_filter
            .expand(&deamort_rm)
            .expect("deamortized expand must succeed");
        deamort_expand_ns = deamort_expand_ns.saturating_add(t0.elapsed().as_nanos() as u64);

        total_boxes_transferred += rep.boxes_transferred;
        total_boxes_rejuvenated += rep.boxes_rejuvenated;
        total_keys_rejuvenated += rep.keys_rejuvenated;
    }

    assert_eq!(deamort_filter.config().q, INITIAL_Q + EXPANSION_STEPS as u8);
    assert_eq!(deamort_filter.expansion_cycles(), EXPANSION_STEPS as u64);
    assert!(
        total_boxes_rejuvenated > 0,
        "expected selective rejuvenation (boxes_rejuvenated > 0) across 8 expansions with r=7"
    );
    assert!(
        !adapted_fps.is_empty(),
        "expected at least 1 adapted FP prior to expansion cycles"
    );

    // Verify 0 false negatives across all 8,192 keys.
    for &k in &all_keys_sorted {
        assert!(
            deamort_filter.query_point(k),
            "false negative after 8 expansions on query_point({k})"
        );
        assert!(
            deamort_filter.query_range(enclosing_range(k, 16)),
            "false negative after 8 expansions on enclosing range for {k}"
        );
    }

    // Verify 0 repeated FPs across all adapted intervals (including rejuvenated ones!).
    for &fp_range in &adapted_fps {
        assert!(
            !deamort_filter.query_range(fp_range),
            "repeated FP after 8 expansions + selective rejuvenation on [{}, {}]",
            fp_range.start(),
            fp_range.end()
        );
    }

    // Verify post-expansion empirical FPR <= target_epsilon over 20,000 fresh empty queries.
    let post_queries = generate_uniform_empty_queries(0xE9A0_8004, &all_key_set, 32, 20_000);
    let post_fps = post_queries
        .iter()
        .filter(|&&q| deamort_filter.query_range(q))
        .count();
    let post_fpr = (post_fps as f64) / (post_queries.len() as f64);
    assert!(
        post_fpr <= TARGET_EPSILON,
        "post-expansion FPR {post_fpr:.6} exceeded target epsilon {TARGET_EPSILON:.6}"
    );
    deamort_filter
        .verify_invariants()
        .expect("deamortized filter invariants violated after 8 expansions");

    // --- Path B: Full-rebuild baseline (`scan_all_keys` + `rebuild_from_keys` on every 2x). ---
    let baseline_dir = TempDir::new().expect("create temp dir for full-rebuild baseline");
    let mut baseline_rm = RocksDbReverseMap::open(
        baseline_dir.path().join("rm_gen_0"),
        ReverseMapConfig::default(),
    )
    .expect("open baseline Reverse Map gen 0");
    let mut baseline_filter = AerisFilter::new(initial_cfg).expect("create baseline filter");
    let mut full_rebuild_ns = 0u64;

    for (step, &batch) in batches.iter().enumerate() {
        let mut rm_batch = Vec::with_capacity(batch.len());
        for &k in batch {
            let fp = baseline_filter
                .insert(k)
                .expect("baseline batch insert must fit before rebuild");
            rm_batch.push((fp, k));
        }
        baseline_rm
            .append_batch(&rm_batch)
            .expect("baseline Reverse Map append_batch");

        let next_cfg = baseline_filter
            .config()
            .expanded()
            .expect("next expanded config");
        let t0 = Instant::now();
        let persisted_keys = baseline_rm
            .scan_all_keys()
            .expect("baseline scan_all_keys failed");
        let next_rm = RocksDbReverseMap::open(
            baseline_dir.path().join(format!("rm_gen_{}", step + 1)),
            ReverseMapConfig::default(),
        )
        .expect("open next baseline Reverse Map");
        baseline_filter = AerisFilter::rebuild_from_keys(next_cfg, persisted_keys, &next_rm)
            .expect("baseline rebuild_from_keys failed");
        full_rebuild_ns = full_rebuild_ns.saturating_add(t0.elapsed().as_nanos() as u64);
        baseline_rm = next_rm;
    }

    assert_eq!(
        baseline_filter.config().q,
        INITIAL_Q + EXPANSION_STEPS as u8
    );
    assert_eq!(baseline_filter.len(), total_keys);
    assert!(
        deamort_expand_ns < full_rebuild_ns,
        "expected deamortized expand ({deamort_expand_ns} ns) to finish faster than full-rebuild baseline ({full_rebuild_ns} ns)"
    );

    let speedup = (full_rebuild_ns as f64) / (deamort_expand_ns.max(1) as f64);
    info!(
        test = "e2e_deamortized_expansion_speedup_vs_full_rebuild",
        initial_q = INITIAL_Q,
        final_q = deamort_filter.config().q,
        total_keys,
        total_boxes_transferred,
        total_boxes_rejuvenated,
        total_keys_rejuvenated,
        adapted_fps_verified = adapted_fps.len(),
        post_expansion_fpr = format!("{post_fpr:.6}"),
        target_epsilon = format!("{TARGET_EPSILON:.6}"),
        deamort_expand_ms = format!("{:.3}", deamort_expand_ns as f64 / 1e6),
        full_rebuild_ms = format!("{:.3}", full_rebuild_ns as f64 / 1e6),
        speedup = format!("{speedup:.2}x"),
        "deamortized 2x expansion speedup and selective rejuvenation verified"
    );
}
