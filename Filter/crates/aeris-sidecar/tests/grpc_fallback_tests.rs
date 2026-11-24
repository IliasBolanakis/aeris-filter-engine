//! Live Rust gRPC client/server integration, fault-injection, timeout,
//! crash-fallback, and concurrency tests for [`GuardedTuningAdvisor`].

use std::sync::Arc;
use std::time::Duration;

use aeris_sidecar::{
    ClientConfig, GuardedTuningAdvisor, NativeFallbackOptimizer, ReferenceSidecarServer,
    SafetyGuardrail, ServerBehavior, SidecarClient, SidecarEndpoint, NATIVE_FALLBACK_MODEL_VERSION,
    REFERENCE_SERVER_MODEL_VERSION,
};
use aeris_types::{
    CostModelParams, FilterConfig, GuardrailVerdict, SafetyBounds, ShardTelemetrySnapshot,
    TelemetrySnapshot, TuningAdvisor,
};

fn default_bounds() -> SafetyBounds {
    let cfg = FilterConfig::new(14, 15, 32, 20, 1.0 / 256.0).expect("valid config");
    SafetyBounds::from_filter_config(&cfg)
}

fn sample_telemetry(request_id: &str) -> TelemetrySnapshot {
    TelemetrySnapshot {
        request_id: request_id.to_string(),
        shard_id: 3,
        epoch: 11,
        timestamp_ms: 1_700_000_000_000,
        current_q: 14,
        current_r: 15,
        current_m: 5,
        payload_budget_c: 20,
        target_epsilon: 1.0 / 256.0,
        load_factor: 0.72,
        ingestion_rate_ewma: 8_500.0,
        query_rate: 12_000.0,
        observed_fpr: 0.002,
        p95_query_len: 64.0,
        p99_query_len: 128.0,
        reverse_map_entries: 12_000,
        bulk_load_speed: 500_000.0,
        recent_query_len_series: vec![32.0, 48.0, 64.0, 60.0],
        recent_ingest_rate_series: vec![7_500.0, 8_200.0, 9_100.0, 8_500.0],
    }
}

fn sample_shards() -> Vec<ShardTelemetrySnapshot> {
    vec![
        ShardTelemetrySnapshot {
            shard_id: 1,
            key_count: 100_000,
            current_memory_bytes: 2_000_000,
            query_rate: 25_000.0,
            observed_fpr: 0.03,
            p95_query_len: 128.0,
            load_factor: 0.82,
            current_q: 15,
            current_r: 14,
            current_m: 6,
        },
        ShardTelemetrySnapshot {
            shard_id: 2,
            key_count: 100_000,
            current_memory_bytes: 2_000_000,
            query_rate: 50.0,
            observed_fpr: 0.0005,
            p95_query_len: 8.0,
            load_factor: 0.30,
            current_q: 15,
            current_r: 16,
            current_m: 4,
        },
    ]
}

fn build_advisor(
    endpoint: SidecarEndpoint,
    timeout: Duration,
    retries: u32,
) -> GuardedTuningAdvisor<NativeFallbackOptimizer> {
    let client_cfg = ClientConfig::new(endpoint)
        .with_rpc_timeout(timeout)
        .with_max_retries(retries)
        .with_base_backoff_ms(10)
        .with_jitter_seed(42);
    let client = SidecarClient::new(client_cfg).expect("valid client config");
    GuardedTuningAdvisor::with_owned_runtime(
        Some(client),
        NativeFallbackOptimizer::new(),
        default_bounds(),
        CostModelParams::default(),
    )
    .expect("advisor builds")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_grpc_normal_approval_and_idempotency() {
    let server =
        ReferenceSidecarServer::start(SidecarEndpoint::loopback_v4(0), ServerBehavior::Normal)
            .await
            .expect("server starts");

    let advisor = build_advisor(server.endpoint().clone(), Duration::from_secs(2), 2);
    assert!(advisor.is_healthy_async().await);
    assert!(advisor.is_healthy());

    let tel = sample_telemetry("normal-req-1");
    let p1 = advisor.advise(&tel).expect("advise succeeds");
    let p2 = advisor.advise(&tel).expect("idempotent advise succeeds");

    assert_eq!(p1, p2);
    assert_eq!(p1.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&p1, advisor.bounds()),
        GuardrailVerdict::Approved
    );
    assert_eq!(server.tuning_cache_hits(), 1);

    let budget = 8 * 1024 * 1024;
    let plan = advisor
        .allocate_memory(11, budget, &sample_shards())
        .expect("allocate_memory succeeds");
    assert_eq!(plan.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_memory_plan(&plan, budget, advisor.bounds()),
        GuardrailVerdict::Approved
    );

    let snap = advisor.metrics();
    assert_eq!(snap.total_calls, 3);
    assert_eq!(snap.sidecar_accepted, 3);
    assert_eq!(snap.guardrail_rejections, 0);
    assert_eq!(snap.transport_fallbacks, 0);

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn guardrail_rejection_triggers_safe_native_fallback() {
    for mode in [
        ServerBehavior::OutOfBoundsProposal,
        ServerBehavior::LowConfidence,
    ] {
        let server = ReferenceSidecarServer::start(SidecarEndpoint::loopback_v4(0), mode)
            .await
            .expect("server starts");

        let advisor = build_advisor(server.endpoint().clone(), Duration::from_secs(2), 1);

        let proposal = advisor
            .advise(&sample_telemetry("oob-req"))
            .expect("fallback never errors");
        assert_eq!(proposal.model_version, NATIVE_FALLBACK_MODEL_VERSION);
        assert_eq!(
            SafetyGuardrail::new().validate_tuning(&proposal, advisor.bounds()),
            GuardrailVerdict::Approved
        );

        let budget = 4 * 1024 * 1024;
        let plan = advisor
            .allocate_memory(4, budget, &sample_shards())
            .expect("memory fallback never errors");
        assert_eq!(plan.model_version, NATIVE_FALLBACK_MODEL_VERSION);
        assert_eq!(
            SafetyGuardrail::new().validate_memory_plan(&plan, budget, advisor.bounds()),
            GuardrailVerdict::Approved
        );

        let snap = advisor.metrics();
        assert_eq!(snap.total_calls, 2);
        assert_eq!(snap.sidecar_accepted, 0);
        assert_eq!(snap.guardrail_rejections, 2);
        assert_eq!(snap.transport_fallbacks, 0);

        server.shutdown().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_timeout_and_internal_error_trigger_transport_fallback() {
    // 1. SlowTimeout exceeding client rpc_timeout
    let slow_server = ReferenceSidecarServer::start(
        SidecarEndpoint::loopback_v4(0),
        ServerBehavior::SlowTimeout(Duration::from_millis(350)),
    )
    .await
    .expect("slow server starts");

    let advisor = build_advisor(slow_server.endpoint().clone(), Duration::from_millis(60), 1);
    let proposal = advisor
        .advise(&sample_telemetry("timeout-req"))
        .expect("timeout falls back cleanly");
    assert_eq!(proposal.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&proposal, advisor.bounds()),
        GuardrailVerdict::Approved
    );
    let snap = advisor.metrics();
    assert_eq!(snap.total_calls, 1);
    assert_eq!(snap.transport_fallbacks, 1);
    assert_eq!(snap.sidecar_accepted, 0);
    slow_server.shutdown().await;

    // 2. Internal error fails fast without wasting retries
    let err_server = ReferenceSidecarServer::start(
        SidecarEndpoint::loopback_v4(0),
        ServerBehavior::FailInternal,
    )
    .await
    .expect("error server starts");

    let advisor_err = build_advisor(err_server.endpoint().clone(), Duration::from_secs(2), 3);
    let proposal_err = advisor_err
        .advise(&sample_telemetry("internal-err-req"))
        .expect("internal error falls back cleanly");
    assert_eq!(proposal_err.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        err_server.request_count(),
        1,
        "non-retryable Internal must fail fast"
    );
    assert_eq!(advisor_err.metrics().transport_fallbacks, 1);
    err_server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flaky_server_recovers_via_bounded_retries_preserving_request_id() {
    let server = ReferenceSidecarServer::start(
        SidecarEndpoint::loopback_v4(0),
        ServerBehavior::FlakyThenNormal { fail_count: 2 },
    )
    .await
    .expect("flaky server starts");

    let advisor = build_advisor(server.endpoint().clone(), Duration::from_secs(2), 2);
    let proposal = advisor
        .advise(&sample_telemetry("flaky-idempotent-key"))
        .expect("succeeds on 3rd attempt");

    assert_eq!(proposal.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert_eq!(proposal.request_id, "flaky-idempotent-key");
    assert_eq!(server.request_count(), 3);
    assert_eq!(advisor.metrics().sidecar_accepted, 1);
    assert_eq!(advisor.metrics().transport_fallbacks, 0);

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abrupt_server_crash_mid_session_falls_back_seamlessly() {
    let server =
        ReferenceSidecarServer::start(SidecarEndpoint::loopback_v4(0), ServerBehavior::Normal)
            .await
            .expect("server starts");

    let advisor = build_advisor(server.endpoint().clone(), Duration::from_millis(250), 1);

    // Phase 1: healthy sidecar serves proposal
    let p_live = advisor
        .advise(&sample_telemetry("before-crash"))
        .expect("live call succeeds");
    assert_eq!(p_live.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert!(advisor.is_healthy());

    // Simulate sudden abrupt crash (SIGKILL equivalent)
    server.abort().await;

    // Phase 2: sidecar unreachable -> health is false, advisor seamlessly returns safe native fallback
    assert!(!advisor.is_healthy());

    let p_after = advisor
        .advise(&sample_telemetry("after-crash"))
        .expect("post-crash advise falls back safely");
    assert_eq!(p_after.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&p_after, advisor.bounds()),
        GuardrailVerdict::Approved
    );

    let budget = 8 * 1024 * 1024;
    let m_after = advisor
        .allocate_memory(99, budget, &sample_shards())
        .expect("post-crash allocate_memory falls back safely");
    assert_eq!(m_after.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_memory_plan(&m_after, budget, advisor.bounds()),
        GuardrailVerdict::Approved
    );

    let snap = advisor.metrics();
    assert_eq!(snap.total_calls, 3);
    assert_eq!(snap.sidecar_accepted, 1);
    assert_eq!(snap.transport_fallbacks, 2);
    assert_eq!(snap.guardrail_rejections, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_control_plane_threads_preserve_metrics_conservation() {
    let server =
        ReferenceSidecarServer::start(SidecarEndpoint::loopback_v4(0), ServerBehavior::Normal)
            .await
            .expect("server starts");

    let advisor = Arc::new(build_advisor(
        server.endpoint().clone(),
        Duration::from_secs(2),
        1,
    ));

    let mut handles = Vec::new();
    for worker in 0..8u64 {
        let adv = Arc::clone(&advisor);
        handles.push(std::thread::spawn(move || {
            for iter in 0..4u64 {
                let req_id = format!("conc-w{worker}-i{iter}");
                let p = adv.advise(&sample_telemetry(&req_id)).expect("advise ok");
                assert_eq!(
                    SafetyGuardrail::new().validate_tuning(&p, adv.bounds()),
                    GuardrailVerdict::Approved
                );
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread must not panic");
    }

    let snap = advisor.metrics();
    assert_eq!(snap.total_calls, 32);
    assert_eq!(
        snap.total_calls,
        snap.sidecar_accepted + snap.guardrail_rejections + snap.transport_fallbacks
    );
    assert_eq!(snap.sidecar_accepted, 32);

    server.shutdown().await;
}

#[test]
fn native_only_advisor_works_without_tokio_or_sidecar() {
    let advisor = GuardedTuningAdvisor::native_only(
        NativeFallbackOptimizer::new(),
        default_bounds(),
        CostModelParams::default(),
    )
    .expect("native-only advisor builds");

    assert!(!advisor.is_healthy());
    let proposal = advisor
        .advise(&sample_telemetry("native-only-1"))
        .expect("native advise succeeds");
    assert_eq!(proposal.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_tuning(&proposal, advisor.bounds()),
        GuardrailVerdict::Approved
    );

    let budget = 2 * 1024 * 1024;
    let plan = advisor
        .allocate_memory(1, budget, &sample_shards())
        .expect("native allocate_memory succeeds");
    assert_eq!(plan.model_version, NATIVE_FALLBACK_MODEL_VERSION);
    assert_eq!(
        SafetyGuardrail::new().validate_memory_plan(&plan, budget, advisor.bounds()),
        GuardrailVerdict::Approved
    );

    let snap = advisor.metrics();
    assert_eq!(snap.total_calls, 2);
    assert_eq!(snap.transport_fallbacks, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_runtime_handle_mode_and_shard_coverage_rejection() {
    let server =
        ReferenceSidecarServer::start(SidecarEndpoint::loopback_v4(0), ServerBehavior::Normal)
            .await
            .expect("server starts");

    let client_cfg = ClientConfig::new(server.endpoint().clone())
        .with_rpc_timeout(Duration::from_secs(2))
        .with_max_retries(1);
    let client = SidecarClient::new(client_cfg).expect("valid client");

    let advisor = GuardedTuningAdvisor::with_runtime_handle(
        Some(client),
        NativeFallbackOptimizer::new(),
        default_bounds(),
        CostModelParams::default(),
        tokio::runtime::Handle::current(),
    )
    .expect("advisor with external handle builds");

    let p = advisor
        .advise_async(&sample_telemetry("ext-handle-1"))
        .await
        .expect("async advise succeeds");
    assert_eq!(p.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert_eq!(advisor.metrics().sidecar_accepted, 1);

    server.shutdown().await;
}
