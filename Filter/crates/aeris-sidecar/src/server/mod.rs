//! Reference Rust gRPC `ControlPlaneAdvisor` server supporting deterministic
//! fault injection over `SidecarEndpoint` (UDS on Unix, loopback TCP on Windows).
//!
//! Designed for exhaustive integration, guardrail, retry, timeout, and crash
//! fallback verification without requiring external infrastructure.

mod planner;
mod service;
#[cfg(test)]
mod tests;

pub use planner::REFERENCE_SERVER_MODEL_VERSION;

use std::net::SocketAddr;
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use aeris_types::{AerisError, AerisResult};
use service::{ReferenceAdvisorService, ServerState};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;

use crate::proto::control_plane_advisor_server::ControlPlaneAdvisorServer;
use crate::transport::SidecarEndpoint;

/// Configurable behavior mode for [`ReferenceSidecarServer`] enabling
/// deterministic fault-injection across every Control Plane failure path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerBehavior {
    /// Returns mathematically valid, guardrail-compliant proposals and
    /// budget-respecting memory allocation plans.
    Normal,
    /// Returns proposals that deliberately violate [`crate::guardrail::SafetyGuardrail`]
    /// (`r < r_min`, `alpha > 0.90`, `k > 3.0`, and memory allocations exceeding budget).
    OutOfBoundsProposal,
    /// Returns geometrically valid proposals with confidence (`0.25`) below the
    /// default guardrail minimum (`0.80`).
    LowConfidence,
    /// Delays every RPC response by the specified [`Duration`] to exercise
    /// client timeout and fallback paths.
    SlowTimeout(Duration),
    /// Rejects every RPC with `tonic::Status::internal`.
    FailInternal,
    /// Rejects every RPC with `tonic::Status::unavailable` (retryable).
    FailUnavailable,
    /// Returns `Unavailable` for the first `fail_count` requests, then serves
    /// [`Self::Normal`] responses to exercise bounded retries with idempotency.
    FlakyThenNormal {
        /// Number of initial requests that fail with `Unavailable`.
        fail_count: u32,
    },
}

/// Handle to a running [`ReferenceSidecarServer`] instance allowing dynamic
/// behavior mutation, call count inspection, graceful shutdown, or abrupt crash
/// simulation.
pub struct RunningServer {
    endpoint: SidecarEndpoint,
    state: Arc<ServerState>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl RunningServer {
    /// Returns the actual bound [`SidecarEndpoint`] (with ephemeral port `0`
    /// resolved to the OS-assigned loopback port).
    #[inline]
    pub fn endpoint(&self) -> &SidecarEndpoint {
        &self.endpoint
    }

    /// Dynamically updates the server's fault-injection [`ServerBehavior`]
    /// mid-session.
    pub fn set_behavior(&self, behavior: ServerBehavior) {
        *self.state.behavior.write() = behavior;
    }

    /// Returns the total number of `ProposeTuning` and `AllocateMemory` RPCs
    /// received by this server.
    #[inline]
    pub fn request_count(&self) -> u64 {
        self.state.request_count.load(Ordering::Relaxed)
    }

    /// Returns the number of `ProposeTuning` requests served from the bounded
    /// `request_id` idempotency cache.
    #[inline]
    pub fn tuning_cache_hits(&self) -> u64 {
        self.state.tuning_cache_hits.load(Ordering::Relaxed)
    }

    /// Gracefully stops the gRPC server and awaits listener teardown.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.await;
        }
        self.cleanup_unix_socket();
    }

    /// Abruptly terminates the server task without graceful drain to simulate
    /// an out-of-process crash mid-session.
    pub async fn abort(mut self) {
        let _ = self.shutdown_tx.take();
        if let Some(handle) = self.join_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        self.cleanup_unix_socket();
    }

    fn cleanup_unix_socket(&self) {
        #[cfg(unix)]
        if let SidecarEndpoint::UnixSocket(path) = &self.endpoint {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            handle.abort();
        }
        self.cleanup_unix_socket();
    }
}

/// Factory for starting a [`ReferenceSidecarServer`] bound to a local IPC
/// [`SidecarEndpoint`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceSidecarServer;

impl ReferenceSidecarServer {
    /// Starts the reference gRPC server on `endpoint` with the initial
    /// `behavior`.
    ///
    /// For `SidecarEndpoint::loopback_v4(0)`, the OS assigns an available
    /// ephemeral loopback port accessible via [`RunningServer::endpoint`].
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `endpoint` fails validation, or
    /// [`AerisError::SidecarUnavailable`] if binding the listener fails.
    pub async fn start(
        endpoint: SidecarEndpoint,
        behavior: ServerBehavior,
    ) -> AerisResult<RunningServer> {
        endpoint.validate()?;
        let state = Arc::new(ServerState::new(behavior));

        match endpoint {
            SidecarEndpoint::LoopbackTcp(addr) => start_loopback_tcp(addr, state).await,
            #[cfg(unix)]
            SidecarEndpoint::UnixSocket(path) => start_unix_socket(path, state),
        }
    }
}

async fn start_loopback_tcp(
    addr: SocketAddr,
    state: Arc<ServerState>,
) -> AerisResult<RunningServer> {
    let listener = TcpListener::bind(addr).await.map_err(|err| {
        AerisError::SidecarUnavailable(format!(
            "failed to bind loopback TCP sidecar on {addr}: {err}"
        ))
    })?;
    let bound_addr = listener.local_addr().map_err(|err| {
        AerisError::SidecarUnavailable(format!("failed to inspect bound sidecar address: {err}"))
    })?;
    let incoming = TcpListenerStream::new(listener);
    let bound_endpoint = SidecarEndpoint::LoopbackTcp(bound_addr);

    tracing::info!(
        endpoint = %bound_endpoint,
        "starting reference Rust gRPC sidecar server"
    );

    Ok(spawn_server_task(bound_endpoint, state, incoming))
}

#[cfg(unix)]
fn start_unix_socket(path: PathBuf, state: Arc<ServerState>) -> AerisResult<RunningServer> {
    if path.exists() {
        let _ = std::fs::remove_file(&path);
    }
    let listener = tokio::net::UnixListener::bind(&path).map_err(|err| {
        AerisError::SidecarUnavailable(format!(
            "failed to bind UDS sidecar at {}: {err}",
            path.display()
        ))
    })?;
    let incoming = tokio_stream::wrappers::UnixListenerStream::new(listener);
    let bound_endpoint = SidecarEndpoint::UnixSocket(path);

    tracing::info!(
        endpoint = %bound_endpoint,
        "starting reference Rust UDS gRPC sidecar server"
    );

    Ok(spawn_server_task(bound_endpoint, state, incoming))
}

fn spawn_server_task<I, IO, IE>(
    endpoint: SidecarEndpoint,
    state: Arc<ServerState>,
    incoming: I,
) -> RunningServer
where
    I: tokio_stream::Stream<Item = Result<IO, IE>> + Send + 'static,
    IO: tokio::io::AsyncRead
        + tokio::io::AsyncWrite
        + tonic::transport::server::Connected
        + Unpin
        + Send
        + 'static,
    IE: Into<Box<dyn std::error::Error + Send + Sync>> + Send + 'static,
{
    let svc = ReferenceAdvisorService {
        state: Arc::clone(&state),
    };
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(ControlPlaneAdvisorServer::new(svc))
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    RunningServer {
        endpoint,
        state,
        shutdown_tx: Some(shutdown_tx),
        join_handle: Some(handle),
    }
}
