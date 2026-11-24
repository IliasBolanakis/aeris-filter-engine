//! Cross-platform local IPC transport configuration for the Control
//! Plane gRPC sidecar.
//!
//! Supports:
//! - Native Unix Domain Sockets (`UnixSocket`) on Unix hosts via
//!   `tokio::net::UnixStream` wrapped in `hyper_util::rt::TokioIo`.
//! - Local loopback TCP (`LoopbackTcp`, e.g., `127.0.0.1:<port>`) on all
//!   platforms, serving as the primary live local IPC transport on Windows.
//!
//! Enforces strict loopback-only addressing, explicit per-attempt RPC timeouts,
//! and deterministic bounded exponential backoff with jitter.

use std::fmt;
use std::net::{Ipv4Addr, SocketAddr};
#[cfg(unix)]
use std::path::PathBuf;
use std::time::Duration;

use aeris_types::{AerisError, AerisResult, DEFAULT_HASH_SEED};
use tonic::transport::{Channel, Endpoint};

/// Default per-attempt gRPC deadline (`500 ms`).
pub const DEFAULT_RPC_TIMEOUT: Duration = Duration::from_millis(500);

/// Default number of retries after an initial transient RPC failure or timeout.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// Default base exponential backoff delay between retry attempts (`25 ms`).
pub const DEFAULT_BASE_BACKOFF_MS: u64 = 25;

/// Maximum cap on retry backoff delay in milliseconds (`5_000 ms`).
pub const MAX_RETRY_BACKOFF_MS: u64 = 5_000;

/// Default consecutive transport/timeout failures required to trip the sidecar
/// client circuit breaker from `Closed` to `Open`.
pub const DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD: u32 = 2;

/// Default cooldown window before an `Open` circuit breaker permits a single
/// `HalfOpen` probe (`10 s`).
pub const DEFAULT_CIRCUIT_BREAKER_COOLDOWN: Duration = Duration::from_secs(10);

/// Local IPC endpoint addressing an out-of-process Control Plane sidecar.
///
/// # Security Invariant
/// [`SidecarEndpoint::validate`] rejects non-loopback IP addresses so sidecar
/// control traffic remains strictly host-local.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SidecarEndpoint {
    /// Native Unix Domain Socket path (Unix platforms only).
    #[cfg(unix)]
    UnixSocket(PathBuf),
    /// Host-local loopback TCP address (`127.0.0.1:<port>` or `[::1]:<port>`).
    LoopbackTcp(SocketAddr),
}

impl SidecarEndpoint {
    /// Constructs a IPv4 loopback endpoint `127.0.0.1:<port>`.
    #[inline]
    pub const fn loopback_v4(port: u16) -> Self {
        Self::LoopbackTcp(SocketAddr::new(
            std::net::IpAddr::V4(Ipv4Addr::LOCALHOST),
            port,
        ))
    }

    /// Constructs a Unix Domain Socket endpoint from `path` (Unix only).
    #[cfg(unix)]
    #[inline]
    pub fn unix_socket(path: impl Into<PathBuf>) -> Self {
        Self::UnixSocket(path.into())
    }

    /// Validates that this endpoint represents a safe, well-formed local IPC target.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if a TCP endpoint is not a loopback
    /// address or if a Unix socket path is empty.
    pub fn validate(&self) -> AerisResult<()> {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => {
                if path.as_os_str().is_empty() {
                    return Err(AerisError::InvalidConfig(
                        "unix socket endpoint path must not be empty".to_string(),
                    ));
                }
                Ok(())
            }
            Self::LoopbackTcp(addr) => {
                if !addr.ip().is_loopback() {
                    return Err(AerisError::InvalidConfig(format!(
                        "sidecar TCP endpoint must be a loopback address, got {addr}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Establishes a `tonic` [`Channel`] to this sidecar endpoint with an explicit
    /// connection timeout.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if the endpoint is invalid, or
    /// [`AerisError::SidecarUnavailable`] if connection establishment fails or
    /// exceeds `connect_timeout`.
    pub async fn connect(&self, connect_timeout: Duration) -> AerisResult<Channel> {
        self.validate()?;
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => {
                let socket_path = path.clone();
                let endpoint = Endpoint::try_from("http://[::]:50051")
                    .map_err(|err| {
                        AerisError::InvalidConfig(format!(
                            "failed to construct UDS dummy URI: {err}"
                        ))
                    })?
                    .connect_timeout(connect_timeout);

                endpoint
                    .connect_with_connector(tower::service_fn(move |_: tonic::transport::Uri| {
                        let path = socket_path.clone();
                        async move {
                            let stream = tokio::net::UnixStream::connect(path).await?;
                            Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
                        }
                    }))
                    .await
                    .map_err(|err| {
                        AerisError::SidecarUnavailable(format!(
                            "failed to connect to UDS sidecar at {}: {err}",
                            path.display()
                        ))
                    })
            }
            Self::LoopbackTcp(addr) => {
                if addr.port() == 0 {
                    return Err(AerisError::InvalidConfig(
                        "cannot connect client to ephemeral port 0".to_string(),
                    ));
                }
                let uri = format!("http://{addr}");
                let endpoint = Endpoint::from_shared(uri.clone())
                    .map_err(|err| {
                        AerisError::InvalidConfig(format!("invalid sidecar URI '{uri}': {err}"))
                    })?
                    .connect_timeout(connect_timeout)
                    .tcp_nodelay(true);

                endpoint.connect().await.map_err(|err| {
                    AerisError::SidecarUnavailable(format!(
                        "failed to connect to loopback sidecar at {addr}: {err}"
                    ))
                })
            }
        }
    }
}

impl fmt::Display for SidecarEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => write!(f, "unix://{}", path.display()),
            Self::LoopbackTcp(addr) => write!(f, "http://{addr}"),
        }
    }
}

/// Configuration for [`crate::client::SidecarClient`] enforcing explicit RPC
/// timeouts, bounded retries, deterministic jittered exponential backoff, and
/// a fast-fail transport circuit breaker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientConfig {
    /// Local IPC endpoint of the out-of-process sidecar.
    pub endpoint: SidecarEndpoint,
    /// Explicit per-attempt RPC deadline. Must be strictly greater than zero.
    pub rpc_timeout: Duration,
    /// Maximum number of retry attempts after the initial attempt fails with a
    /// retryable transport error or timeout (`0` disables retries).
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential retry backoff.
    pub base_backoff_ms: u64,
    /// Deterministic 64-bit seed controlling bounded retry jitter.
    pub jitter_seed: u64,
    /// Number of consecutive RPC-level transport or timeout failures before the
    /// circuit breaker opens and short-circuits directly to native fallback (`>= 1`).
    pub circuit_breaker_failure_threshold: u32,
    /// Duration the circuit breaker remains `Open` before allowing a single
    /// `HalfOpen` probe to test sidecar recovery (`> Duration::ZERO`).
    pub circuit_breaker_cooldown: Duration,
}

impl ClientConfig {
    /// Constructs a [`ClientConfig`] with production-safe defaults (`500 ms`
    /// timeout, `2` retries, `25 ms` base backoff, circuit breaker threshold `2`,
    /// `10 s` cooldown, deterministic seed).
    pub const fn new(endpoint: SidecarEndpoint) -> Self {
        Self {
            endpoint,
            rpc_timeout: DEFAULT_RPC_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
            base_backoff_ms: DEFAULT_BASE_BACKOFF_MS,
            jitter_seed: DEFAULT_HASH_SEED,
            circuit_breaker_failure_threshold: DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            circuit_breaker_cooldown: DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
        }
    }

    /// Sets the explicit per-attempt RPC timeout.
    #[inline]
    pub const fn with_rpc_timeout(mut self, rpc_timeout: Duration) -> Self {
        self.rpc_timeout = rpc_timeout;
        self
    }

    /// Sets the maximum retry count after an initial transient failure.
    #[inline]
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the base backoff delay in milliseconds.
    #[inline]
    pub const fn with_base_backoff_ms(mut self, base_backoff_ms: u64) -> Self {
        self.base_backoff_ms = base_backoff_ms;
        self
    }

    /// Sets the deterministic jitter PRNG seed.
    #[inline]
    pub const fn with_jitter_seed(mut self, jitter_seed: u64) -> Self {
        self.jitter_seed = jitter_seed;
        self
    }

    /// Sets the consecutive transport/timeout failure threshold that trips the
    /// circuit breaker from `Closed` to `Open`.
    #[inline]
    pub const fn with_circuit_breaker_failure_threshold(mut self, threshold: u32) -> Self {
        self.circuit_breaker_failure_threshold = threshold;
        self
    }

    /// Sets the cooldown window an `Open` circuit breaker waits before allowing
    /// a single `HalfOpen` recovery probe.
    #[inline]
    pub const fn with_circuit_breaker_cooldown(mut self, cooldown: Duration) -> Self {
        self.circuit_breaker_cooldown = cooldown;
        self
    }

    /// Validates all transport and circuit-breaker configuration parameters.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if the endpoint is invalid,
    /// `rpc_timeout` is zero, `circuit_breaker_failure_threshold == 0`, or
    /// `circuit_breaker_cooldown` is zero.
    pub fn validate(&self) -> AerisResult<()> {
        self.endpoint.validate()?;
        if self.rpc_timeout.is_zero() {
            return Err(AerisError::InvalidConfig(
                "client rpc_timeout must be greater than zero".to_string(),
            ));
        }
        if self.circuit_breaker_failure_threshold == 0 {
            return Err(AerisError::InvalidConfig(
                "circuit_breaker_failure_threshold must be at least 1".to_string(),
            ));
        }
        if self.circuit_breaker_cooldown.is_zero() {
            return Err(AerisError::InvalidConfig(
                "circuit_breaker_cooldown must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Computes a deterministic, bounded exponential backoff delay with equal jitter.
///
/// For `attempt` (`0` = first retry), the capped exponential window is:
/// `cap = min(base_backoff_ms * 2^min(attempt, 10), MAX_RETRY_BACKOFF_MS)`.
/// Equal jitter selects a deterministic value in `[cap / 2, cap]` using a pure
/// SplitMix64 mixing function seeded by `(jitter_seed, request_seq, attempt)`.
///
/// # Complexity
/// `O(1)` time and `O(1)` space, zero allocations.
#[inline]
pub fn compute_retry_delay(
    base_backoff_ms: u64,
    attempt: u32,
    jitter_seed: u64,
    request_seq: u64,
) -> Duration {
    if base_backoff_ms == 0 {
        return Duration::ZERO;
    }
    let shift = attempt.min(10);
    let exp = base_backoff_ms
        .saturating_mul(1u64 << shift)
        .min(MAX_RETRY_BACKOFF_MS);
    let half = exp / 2;
    let span = exp - half + 1;
    let mixed = splitmix64(
        jitter_seed
            ^ request_seq.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(attempt).wrapping_mul(0xBF58_476D_1CE4_E5B9),
    );
    let delay_ms = half + (mixed % span);
    Duration::from_millis(delay_ms)
}

/// Pure SplitMix64 finalizer for deterministic, platform-independent jitter.
#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
