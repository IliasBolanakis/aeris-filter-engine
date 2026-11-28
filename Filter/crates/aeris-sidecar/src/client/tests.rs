//! Unit tests for [`super::SidecarClient`], protobuf codecs, RPC status
//! classification, and [`super::CircuitBreakerState`] transitions.

use std::time::{Duration, Instant};

use aeris_types::AerisError;
use tonic::Status;

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerInner};
use super::codec::{proto_to_memory_plan, proto_to_tuning_proposal};
use super::rpc::{is_retryable_status, map_grpc_status, should_reset_channel};
use super::*;
use crate::proto;
use crate::transport::{ClientConfig, SidecarEndpoint};

#[test]
fn proto_to_tuning_proposal_rejects_u8_overflow_and_request_id_mismatch() {
    let overflow_resp = proto::TuningResponse {
        request_id: "req-1".to_string(),
        proposed_r: 300,
        proposed_m: 4,
        proposed_target_alpha: 0.85,
        proposed_proactive_margin_k: 1.5,
        confidence: 0.9,
        trigger_proactive_rebuild: false,
        model_version: "m".to_string(),
        rationale: "r".to_string(),
        forecast_peak_ingest_rate: 100.0,
    };
    assert!(matches!(
        proto_to_tuning_proposal("req-1", overflow_resp),
        Err(AerisError::GuardrailRejected { .. })
    ));

    let mismatch_resp = proto::TuningResponse {
        request_id: "wrong-id".to_string(),
        proposed_r: 14,
        proposed_m: 6,
        proposed_target_alpha: 0.85,
        proposed_proactive_margin_k: 1.5,
        confidence: 0.9,
        trigger_proactive_rebuild: false,
        model_version: "m".to_string(),
        rationale: "r".to_string(),
        forecast_peak_ingest_rate: 100.0,
    };
    assert!(matches!(
        proto_to_tuning_proposal("req-1", mismatch_resp),
        Err(AerisError::SidecarUnavailable(_))
    ));

    // Empty response request_id inherits caller's expected request_id.
    let empty_id_resp = proto::TuningResponse {
        request_id: String::new(),
        proposed_r: 14,
        proposed_m: 6,
        proposed_target_alpha: 0.85,
        proposed_proactive_margin_k: 1.5,
        confidence: 0.9,
        trigger_proactive_rebuild: true,
        model_version: "m".to_string(),
        rationale: "r".to_string(),
        forecast_peak_ingest_rate: 100.0,
    };
    let parsed = proto_to_tuning_proposal("expected-1", empty_id_resp).unwrap();
    assert_eq!(parsed.request_id, "expected-1");
    assert_eq!(parsed.proposed_r, 14);
    assert_eq!(parsed.proposed_m, 6);
}

#[test]
fn proto_to_memory_plan_rejects_u8_overflow_and_request_id_mismatch() {
    let overflow_plan = proto::MemoryAllocationResponse {
        request_id: "mem-1".to_string(),
        allocations: vec![proto::ShardAllocation {
            shard_id: 1,
            allocated_memory_bytes: 1024,
            recommended_q: 300,
            recommended_r: 14,
            recommended_m: 6,
        }],
        confidence: 0.9,
        model_version: "m".to_string(),
    };
    assert!(matches!(
        proto_to_memory_plan("mem-1", overflow_plan),
        Err(AerisError::GuardrailRejected { .. })
    ));

    let mismatch_plan = proto::MemoryAllocationResponse {
        request_id: "other-mem".to_string(),
        allocations: vec![],
        confidence: 0.9,
        model_version: "m".to_string(),
    };
    assert!(matches!(
        proto_to_memory_plan("mem-1", mismatch_plan),
        Err(AerisError::SidecarUnavailable(_))
    ));
}

#[test]
fn grpc_status_classification_and_mapping() {
    assert!(is_retryable_status(&Status::unavailable("tmp")));
    assert!(is_retryable_status(&Status::deadline_exceeded("slow")));
    assert!(!is_retryable_status(&Status::invalid_argument("bad")));
    assert!(!is_retryable_status(&Status::internal("bug")));

    assert!(should_reset_channel(&Status::unavailable("down")));
    assert!(should_reset_channel(&Status::cancelled("drop")));
    assert!(!should_reset_channel(&Status::invalid_argument("bad")));

    assert!(matches!(
        map_grpc_status(Status::deadline_exceeded("t"), 250),
        AerisError::SidecarTimeout { timeout_ms: 250 }
    ));
    assert!(matches!(
        map_grpc_status(Status::internal("boom"), 250),
        AerisError::SidecarUnavailable(_)
    ));
}

#[test]
fn circuit_breaker_closed_open_half_open_closed_lifecycle() {
    let threshold = 2;
    let cooldown = Duration::from_secs(10);
    let t0 = Instant::now();
    let mut cb = CircuitBreakerInner::new();

    assert_eq!(
        cb.effective_state_at(t0, cooldown),
        CircuitBreakerState::Closed
    );
    assert!(!cb.try_fast_short_circuit_at(t0, cooldown));
    assert_eq!(cb.try_acquire_at(t0, cooldown), Some(false));

    // Failure #1 keeps circuit Closed (< threshold=2).
    cb.record_transport_failure_at(t0, threshold);
    assert_eq!(
        cb.effective_state_at(t0, cooldown),
        CircuitBreakerState::Closed
    );
    assert_eq!(cb.consecutive_failures, 1);
    assert_eq!(cb.trip_count, 0);

    // Failure #2 trips Closed -> Open.
    let t1 = t0 + Duration::from_millis(50);
    cb.record_transport_failure_at(t1, threshold);
    assert_eq!(
        cb.effective_state_at(t1, cooldown),
        CircuitBreakerState::Open
    );
    assert_eq!(cb.trip_count, 1);

    // Before cooldown elapses: fast short-circuit and try_acquire both reject immediately.
    let t_mid = t1 + Duration::from_secs(5);
    assert!(cb.try_fast_short_circuit_at(t_mid, cooldown));
    assert_eq!(cb.try_acquire_at(t_mid, cooldown), None);
    assert_eq!(cb.short_circuit_count, 2);

    // After cooldown elapses: effective state is HalfOpen, fast-short-circuit yields false,
    // and a single HalfOpen probe is granted.
    let t_after = t1 + cooldown;
    assert_eq!(
        cb.effective_state_at(t_after, cooldown),
        CircuitBreakerState::HalfOpen
    );
    assert!(!cb.try_fast_short_circuit_at(t_after, cooldown));
    assert_eq!(cb.try_acquire_at(t_after, cooldown), Some(true));
    assert_eq!(cb.state, CircuitBreakerState::HalfOpen);

    // Probe succeeds -> transitions HalfOpen -> Closed with reset failure counter.
    cb.record_success_at(t_after + Duration::from_millis(5));
    assert_eq!(cb.state, CircuitBreakerState::Closed);
    assert_eq!(cb.consecutive_failures, 0);
    assert!(!cb.try_fast_short_circuit_at(t_after + Duration::from_millis(10), cooldown));
}

#[test]
fn circuit_breaker_half_open_probe_failure_and_single_probe_gate() {
    let threshold = 2;
    let cooldown = Duration::from_secs(5);
    let t0 = Instant::now();
    let mut cb = CircuitBreakerInner::new();

    cb.record_transport_failure_at(t0, threshold);
    cb.record_transport_failure_at(t0, threshold);
    assert_eq!(cb.state, CircuitBreakerState::Open);
    assert_eq!(cb.trip_count, 1);

    // Cooldown expires: first caller acquires the sole HalfOpen probe.
    let t1 = t0 + cooldown;
    assert_eq!(cb.try_acquire_at(t1, cooldown), Some(true));

    // Concurrent second caller is short-circuited while the probe is in flight.
    assert!(cb.try_fast_short_circuit_at(t1, cooldown));
    assert_eq!(cb.try_acquire_at(t1, cooldown), None);

    // HalfOpen probe fails -> immediately re-opens for another full cooldown window.
    let t2 = t1 + Duration::from_millis(20);
    cb.record_transport_failure_at(t2, threshold);
    assert_eq!(cb.state, CircuitBreakerState::Open);
    assert_eq!(cb.trip_count, 2);
    assert!(cb.try_fast_short_circuit_at(t2 + Duration::from_secs(1), cooldown));
}

#[test]
fn circuit_breaker_aborted_probe_releases_slot() {
    let cb = CircuitBreaker::new(1, Duration::from_millis(1));
    {
        let permit = cb.try_acquire().expect("initial Closed permit");
        permit.record_transport_failure();
    }
    assert_eq!(cb.state(), CircuitBreakerState::Open);

    // Verify pure abort-release semantics directly without wall-clock sleeping.
    let t0 = Instant::now();
    let mut inner = CircuitBreakerInner::new();
    inner.record_transport_failure_at(t0, 1);
    let t1 = t0 + Duration::from_secs(1);
    assert_eq!(inner.try_acquire_at(t1, Duration::from_secs(1)), Some(true));
    assert_eq!(inner.try_acquire_at(t1, Duration::from_secs(1)), None);
    inner.release_aborted_probe();
    assert_eq!(inner.try_acquire_at(t1, Duration::from_secs(1)), Some(true));
}

#[test]
fn client_config_validates_circuit_breaker_parameters() {
    let ep = SidecarEndpoint::loopback_v4(50051);
    assert!(ClientConfig::new(ep.clone()).validate().is_ok());
    assert!(ClientConfig::new(ep.clone())
        .with_circuit_breaker_failure_threshold(0)
        .validate()
        .is_err());
    assert!(ClientConfig::new(ep.clone())
        .with_circuit_breaker_cooldown(Duration::ZERO)
        .validate()
        .is_err());
    assert!(ClientConfig::new(ep)
        .with_rpc_timeout(Duration::ZERO)
        .validate()
        .is_err());
}
