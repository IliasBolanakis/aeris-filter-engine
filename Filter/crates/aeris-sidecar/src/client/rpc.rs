//! Shared bounded retry, timeout, channel invalidation, and circuit-breaker
//! execution loop for [`super::SidecarClient`].

use std::future::Future;
use std::time::Instant;

use aeris_types::{AerisError, AerisResult};
use tonic::transport::Channel;
use tonic::Code;

use super::SidecarClient;
use crate::transport::compute_retry_delay;

impl SidecarClient {
    /// Executes an RPC operation protected by the circuit breaker, per-attempt
    /// timeout, channel reset on transport failure, and deterministic jittered retry.
    pub(super) async fn execute_rpc_with_retries<Resp, F, Fut>(
        &self,
        op_name: &'static str,
        request_id: &str,
        max_retries: u32,
        seq: u64,
        mut call: F,
    ) -> AerisResult<(Resp, u32, u64)>
    where
        F: FnMut(Channel) -> Fut,
        Fut: Future<Output = Result<Resp, tonic::Status>>,
    {
        let Some(permit) = self.circuit_breaker.try_acquire() else {
            return Err(AerisError::SidecarUnavailable(format!(
                "sidecar circuit breaker open; short-circuiting {op_name}"
            )));
        };

        let timeout_ms = self.config.rpc_timeout.as_millis() as u64;
        let mut last_err: Option<AerisError> = None;

        for attempt in 0..=max_retries {
            match self
                .execute_single_attempt(
                    op_name,
                    request_id,
                    attempt,
                    max_retries,
                    timeout_ms,
                    &mut call,
                )
                .await
            {
                Ok((resp, latency_ns)) => {
                    permit.record_success();
                    return Ok((resp, attempt, latency_ns));
                }
                Err(AttemptFailure::Terminal(err)) => {
                    permit.record_transport_failure();
                    return Err(err);
                }
                Err(AttemptFailure::Retryable(err)) => {
                    last_err = Some(err);
                }
            }

            let delay = compute_retry_delay(
                self.config.base_backoff_ms,
                attempt,
                self.config.jitter_seed,
                seq,
            );
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
        }

        permit.record_transport_failure();
        Err(last_err.unwrap_or_else(|| {
            AerisError::SidecarUnavailable(format!("{op_name} exhausted retry budget"))
        }))
    }

    async fn execute_single_attempt<Resp, F, Fut>(
        &self,
        op_name: &'static str,
        request_id: &str,
        attempt: u32,
        max_retries: u32,
        timeout_ms: u64,
        call: &mut F,
    ) -> Result<(Resp, u64), AttemptFailure>
    where
        F: FnMut(Channel) -> Fut,
        Fut: Future<Output = Result<Resp, tonic::Status>>,
    {
        let start = Instant::now();
        let rpc_future = async {
            let channel = self.get_or_connect_channel().await?;
            Ok::<_, AerisError>(call(channel).await)
        };

        let outcome = tokio::time::timeout(self.config.rpc_timeout, rpc_future).await;
        let latency_ns = start.elapsed().as_nanos() as u64;

        match outcome {
            Ok(Ok(Ok(resp))) => Ok((resp, latency_ns)),
            Ok(Ok(Err(status))) => {
                let retryable = is_retryable_status(&status);
                if should_reset_channel(&status) {
                    self.invalidate_channel().await;
                }
                tracing::warn!(
                    op = op_name,
                    request_id = %request_id,
                    attempt,
                    latency_ns,
                    grpc_code = ?status.code(),
                    retryable,
                    error = %status.message(),
                    "sidecar RPC returned gRPC error"
                );
                let err = map_grpc_status(status, timeout_ms);
                Err(classify_attempt_failure(
                    err,
                    retryable,
                    attempt,
                    max_retries,
                ))
            }
            Ok(Err(connect_err)) => {
                self.invalidate_channel().await;
                tracing::warn!(
                    op = op_name,
                    request_id = %request_id,
                    attempt,
                    latency_ns,
                    error = %connect_err,
                    "failed to establish sidecar channel"
                );
                Err(classify_attempt_failure(
                    connect_err,
                    true,
                    attempt,
                    max_retries,
                ))
            }
            Err(_elapsed) => {
                self.invalidate_channel().await;
                tracing::warn!(
                    op = op_name,
                    request_id = %request_id,
                    attempt,
                    latency_ns,
                    timeout_ms,
                    "sidecar RPC timed out"
                );
                let err = AerisError::SidecarTimeout { timeout_ms };
                Err(classify_attempt_failure(err, true, attempt, max_retries))
            }
        }
    }
}

enum AttemptFailure {
    Retryable(AerisError),
    Terminal(AerisError),
}

#[inline]
fn classify_attempt_failure(
    err: AerisError,
    retryable: bool,
    attempt: u32,
    max_retries: u32,
) -> AttemptFailure {
    if !retryable || attempt >= max_retries {
        AttemptFailure::Terminal(err)
    } else {
        AttemptFailure::Retryable(err)
    }
}

/// Classifies whether a gRPC [`tonic::Status`] is transient and eligible for bounded retry.
#[inline]
pub(super) fn is_retryable_status(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        Code::Unavailable
            | Code::DeadlineExceeded
            | Code::Aborted
            | Code::ResourceExhausted
            | Code::Unknown
    )
}

/// Determines whether the cached transport channel should be discarded before reconnecting.
#[inline]
pub(super) fn should_reset_channel(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        Code::Unavailable | Code::Unknown | Code::Cancelled | Code::DeadlineExceeded
    )
}

/// Maps a [`tonic::Status`] onto the unified [`AerisError`] hierarchy.
pub(super) fn map_grpc_status(status: tonic::Status, timeout_ms: u64) -> AerisError {
    if status.code() == Code::DeadlineExceeded {
        AerisError::SidecarTimeout { timeout_ms }
    } else {
        AerisError::SidecarUnavailable(format!("gRPC {:?}: {}", status.code(), status.message()))
    }
}
