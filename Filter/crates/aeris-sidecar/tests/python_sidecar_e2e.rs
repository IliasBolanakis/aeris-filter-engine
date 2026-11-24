//! End-to-end cross-language integration test:
//! Real Rust [`SidecarClient`] + [`GuardedTuningAdvisor`] <-> real Python
//! `aeris_sidecar.server` subprocess over local gRPC loopback TCP.
//!
//! Verifies:
//! 1. `HealthCheck` against live Python ML/RL sidecar.
//! 2. `ProposeTuning` with realistic seasonal telemetry approved by Rust [`SafetyGuardrail`].
//! 3. `ProposeTuning` with sparse/noisy series rejected by Rust [`SafetyGuardrail`]
//!    (low ML confidence) -> safe native fallback.
//! 4. `AllocateMemory` across hot high-FPR vs cold idle shards approved by Rust
//!    [`SafetyGuardrail`], conserving total budget and prioritizing the hot shard.
//! 5. Killing the Python subprocess mid-session -> seamless automatic fallback to
//!    the native deterministic optimizer with zero panics and accurate metrics.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use aeris_sidecar::{
    ClientConfig, GuardedTuningAdvisor, NativeFallbackOptimizer, SafetyGuardrail, SidecarClient,
    SidecarEndpoint, NATIVE_FALLBACK_MODEL_VERSION,
};
use aeris_types::{
    CostModelParams, FilterConfig, GuardrailVerdict, SafetyBounds, ShardTelemetrySnapshot,
    TelemetrySnapshot, TuningAdvisor,
};
use tempfile::TempDir;

/// RAII subprocess guard ensuring the spawned Python sidecar process is always
/// killed and reaped even if an assertion panics.
struct PythonSidecarProcess {
    child: Option<Child>,
    _tempdir: TempDir,
    port: u16,
}

impl PythonSidecarProcess {
    fn spawn() -> Self {
        let tempdir = tempfile::tempdir().expect("create tempdir for sidecar port file");
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
                    .expect("tempdir path must be valid UTF-8"),
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
            proc.child.as_mut().unwrap(),
            Duration::from_secs(20),
        );
        proc
    }

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

fn python_sidecar_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.join("../../python-sidecar");
    assert!(
        root.join("aeris_sidecar/server.py").exists(),
        "expected python-sidecar package at {}",
        root.display()
    );
    root
}

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

fn wait_for_port_file(port_file: &Path, child: &mut Child, timeout: Duration) -> u16 {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(contents) = std::fs::read_to_string(port_file) {
            let trimmed = contents.trim();
            if let Ok(port) = trimmed.parse::<u16>() {
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
        std::thread::yield_now();
    }
}

fn build_seasonal_telemetry(request_id: &str) -> TelemetrySnapshot {
    let mut ingest_series = Vec::with_capacity(48);
    let mut query_len_series = Vec::with_capacity(48);
    for step in 0..48usize {
        let t = step as f64;
        let angle = (2.0 * PI / 24.0) * t;
        let damped = (1.0 - 0.92_f64.powf(t)) / (1.0 - 0.92);
        let ingest = 12_000.0
            + 120.0 * damped
            + 5_500.0 * angle.sin()
            + 2_100.0 * (2.0 * angle).cos()
            + 900.0 * (3.0 * angle).sin();
        let log_qlen = 4.2 + 0.03 * damped + 1.1 * (angle - 0.4).sin() + 0.45 * (2.0 * angle).cos();
        ingest_series.push(ingest.max(50.0));
        query_len_series.push(log_qlen.exp_m1().max(1.0));
    }

    TelemetrySnapshot {
        request_id: request_id.to_string(),
        shard_id: 1,
        epoch: 7,
        timestamp_ms: 1_700_000_000_000,
        current_q: 16,
        current_r: 15,
        current_m: 5,
        payload_budget_c: 20,
        target_epsilon: 1.0 / 256.0,
        load_factor: 0.74,
        ingestion_rate_ewma: *ingest_series.last().unwrap(),
        query_rate: 20_000.0,
        observed_fpr: 0.003,
        p95_query_len: *query_len_series.last().unwrap(),
        p99_query_len: *query_len_series.last().unwrap() * 1.4,
        reverse_map_entries: 45_000,
        bulk_load_speed: 500_000.0,
        recent_query_len_series: query_len_series,
        recent_ingest_rate_series: ingest_series,
    }
}

#[test]
fn rust_client_interoperates_with_live_python_sidecar_and_survives_mid_session_kill() {
    let mut py_proc = PythonSidecarProcess::spawn();

    let cfg = FilterConfig::new(16, 15, 32, 20, 1.0 / 256.0).expect("valid FilterConfig");
    let bounds = SafetyBounds::from_filter_config(&cfg);
    let client_cfg = ClientConfig::new(py_proc.endpoint())
        .with_rpc_timeout(Duration::from_secs(3))
        .with_max_retries(1)
        .with_base_backoff_ms(15)
        .with_jitter_seed(42);
    let client = SidecarClient::new(client_cfg).expect("valid client config");

    let advisor = GuardedTuningAdvisor::with_owned_runtime(
        Some(client),
        NativeFallbackOptimizer::new(),
        bounds.clone(),
        CostModelParams::default(),
    )
    .expect("advisor builds");

    // 1. HealthCheck against real Python gRPC sidecar
    assert!(advisor.is_healthy(), "Python sidecar should report healthy");

    // 2. ProposeTuning with realistic diurnal seasonal series -> approved by SafetyGuardrail
    let seasonal_tel = build_seasonal_telemetry("py-e2e-tune-1");
    let proposal = advisor
        .advise(&seasonal_tel)
        .expect("ProposeTuning via Python sidecar succeeds");
    assert_eq!(proposal.model_version, "aeris-sidecar-ml-v1");
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&proposal, &bounds),
        GuardrailVerdict::Approved
    );
    assert!(proposal.confidence >= 0.80);
    assert!(proposal.forecast_peak_ingest_rate > 10_000.0);

    // 3. ProposeTuning with sparse series -> Python forecaster emits low confidence ->
    //    Rust SafetyGuardrail rejects and automatically falls back to native optimizer!
    let mut sparse_tel = seasonal_tel.clone();
    sparse_tel.request_id = "py-e2e-sparse-2".to_string();
    sparse_tel.recent_ingest_rate_series = vec![10_000.0, 11_000.0];
    sparse_tel.recent_query_len_series = vec![64.0, 64.0];
    let fallback_prop = advisor
        .advise(&sparse_tel)
        .expect("low-confidence proposal falls back cleanly");
    assert_eq!(fallback_prop.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&fallback_prop, &bounds),
        GuardrailVerdict::Approved
    );

    // 4. AllocateMemory across hot high-FPR shard vs cold idle shard -> approved by guardrail
    let budget = 16 * 1024 * 1024;
    let shards = vec![
        ShardTelemetrySnapshot {
            shard_id: 10,
            key_count: 200_000,
            current_memory_bytes: 2_000_000,
            query_rate: 45_000.0,
            observed_fpr: 0.06,
            p95_query_len: 256.0,
            load_factor: 0.88,
            current_q: 16,
            current_r: 14,
            current_m: 6,
        },
        ShardTelemetrySnapshot {
            shard_id: 20,
            key_count: 200_000,
            current_memory_bytes: 2_000_000,
            query_rate: 3.0,
            observed_fpr: 0.0002,
            p95_query_len: 4.0,
            load_factor: 0.20,
            current_q: 16,
            current_r: 14,
            current_m: 6,
        },
    ];
    let plan = advisor
        .allocate_memory(7, budget, &shards)
        .expect("AllocateMemory via Python sidecar succeeds");
    assert_eq!(plan.model_version, "aeris-sidecar-ml-v1");
    assert_eq!(
        SafetyGuardrail::new().validate_memory_plan(&plan, budget, &bounds),
        GuardrailVerdict::Approved
    );
    let hot_bytes = plan
        .allocations
        .iter()
        .find(|a| a.shard_id == 10)
        .expect("hot shard allocated")
        .allocated_memory_bytes;
    let cold_bytes = plan
        .allocations
        .iter()
        .find(|a| a.shard_id == 20)
        .expect("cold shard allocated")
        .allocated_memory_bytes;
    assert!(
        hot_bytes > cold_bytes,
        "contextual bandit should prioritize hot high-FPR shard ({hot_bytes}) over cold shard ({cold_bytes})"
    );

    // 5. Kill Python sidecar subprocess mid-session and verify seamless native fallback!
    py_proc.kill_and_wait();

    assert!(
        !advisor.is_healthy(),
        "killed Python sidecar must report unhealthy"
    );

    let post_kill_tel = build_seasonal_telemetry("py-e2e-post-kill-3");
    let post_kill_prop = advisor
        .advise(&post_kill_tel)
        .expect("post-kill advise falls back cleanly");
    assert_eq!(post_kill_prop.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&post_kill_prop, &bounds),
        GuardrailVerdict::Approved
    );

    let post_kill_plan = advisor
        .allocate_memory(8, budget, &shards)
        .expect("post-kill allocate_memory falls back cleanly");
    assert_eq!(post_kill_plan.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_memory_plan(&post_kill_plan, budget, &bounds),
        GuardrailVerdict::Approved
    );

    let metrics = advisor.metrics();
    assert_eq!(metrics.total_calls, 5);
    assert_eq!(metrics.sidecar_accepted, 2);
    assert_eq!(metrics.guardrail_rejections, 1);
    assert_eq!(metrics.transport_fallbacks, 2);
    assert_eq!(
        metrics.total_calls,
        metrics.sidecar_accepted + metrics.guardrail_rejections + metrics.transport_fallbacks
    );
}
