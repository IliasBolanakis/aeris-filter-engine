//! Token-bucket I/O rate limiter for background Reverse Map scans.

use aeris_types::{AerisError, AerisResult};
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::{Duration, Instant};

/// Default batch size (keys per pacing window) for [`IoThrottler`].
pub const DEFAULT_IO_THROTTLE_BATCH_SIZE: usize = 1024;

/// Configuration for token-bucket background scan rate limiting ([`IoThrottler`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoThrottleConfig {
    /// Maximum sustained Reverse Map scan throughput in keys/second (`None` = unthrottled).
    pub max_keys_per_sec: Option<u64>,
    /// Number of keys streamed per pacing check.
    pub batch_size: usize,
}

impl Default for IoThrottleConfig {
    fn default() -> Self {
        Self {
            max_keys_per_sec: None,
            batch_size: DEFAULT_IO_THROTTLE_BATCH_SIZE,
        }
    }
}

impl IoThrottleConfig {
    /// Constructs a throttled configuration capping scan speed at `max_keys_per_sec`.
    #[must_use]
    pub const fn limited(max_keys_per_sec: u64, batch_size: usize) -> Self {
        Self {
            max_keys_per_sec: Some(max_keys_per_sec),
            batch_size,
        }
    }

    /// Validates that `batch_size >= 1` and `max_keys_per_sec` (when `Some`) is `>= 1`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on invalid parameters.
    pub fn validate(&self) -> AerisResult<()> {
        if self.batch_size == 0 {
            return Err(AerisError::InvalidConfig(
                "IoThrottleConfig::batch_size must be at least 1".to_string(),
            ));
        }
        if self.max_keys_per_sec == Some(0) {
            return Err(AerisError::InvalidConfig(
                "IoThrottleConfig::max_keys_per_sec must be at least 1 when set".to_string(),
            ));
        }
        Ok(())
    }
}

/// Telemetry counters recorded by [`IoThrottler`] during background scanning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThrottleMetrics {
    /// Total keys paced through the throttler.
    pub keys_paced: u64,
    /// Number of batches that required sleeping to honor the rate limit.
    pub throttle_events: u64,
    /// Cumulative pacing delay applied in nanoseconds.
    pub total_throttled_ns: u64,
}

/// Token-bucket rate limiter pacing streaming [`aeris_types::ReverseMapStore::iter_all_keys`]
/// consumption during background rebuilds so disk I/O does not starve foreground traffic.
///
/// Separates pure virtual-clock math ([`Self::consume_at`]) from real wall-clock
/// sleeping ([`Self::pace`]) for deterministic unit testing and real I/O enforcement.
#[derive(Clone, Debug)]
pub struct IoThrottler {
    config: IoThrottleConfig,
    bucket_capacity: f64,
    tokens: f64,
    last_ns: u64,
    started_at: Instant,
    metrics: ThrottleMetrics,
}

impl IoThrottler {
    /// Creates a validated [`IoThrottler`] starting with an empty token bucket so
    /// even short scans strictly obey `max_keys_per_sec`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` is invalid.
    pub fn new(config: IoThrottleConfig) -> AerisResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            bucket_capacity: config.batch_size as f64,
            tokens: 0.0,
            last_ns: 0,
            started_at: Instant::now(),
            metrics: ThrottleMetrics::default(),
        })
    }

    /// Returns the configured [`IoThrottleConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> IoThrottleConfig {
        self.config
    }

    /// Returns the configured batch size in keys.
    #[inline]
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Returns accumulated [`ThrottleMetrics`].
    #[inline]
    #[must_use]
    pub const fn metrics(&self) -> ThrottleMetrics {
        self.metrics
    }

    /// Pure deterministic virtual-clock token-bucket step.
    ///
    /// Advances the virtual clock to `now_ns`, refills tokens at `max_keys_per_sec`
    /// up to `batch_size`, deducts `keys`, updates [`ThrottleMetrics`], and returns
    /// the exact required wait duration in nanoseconds (`0` if unthrottled or within budget).
    ///
    /// # Complexity
    /// `O(1)` time, `O(1)` space, zero system-clock calls.
    pub fn consume_at(&mut self, keys: u64, now_ns: u64) -> u64 {
        self.metrics.keys_paced = self.metrics.keys_paced.saturating_add(keys);
        let Some(rate_u64) = self.config.max_keys_per_sec else {
            self.last_ns = self.last_ns.max(now_ns);
            return 0;
        };
        if keys == 0 {
            self.last_ns = self.last_ns.max(now_ns);
            return 0;
        }

        let rate = rate_u64 as f64;
        let elapsed_ns = now_ns.saturating_sub(self.last_ns);
        let refilled = (elapsed_ns as f64) * rate / 1_000_000_000.0;
        self.tokens = (self.tokens + refilled).min(self.bucket_capacity);
        self.tokens -= keys as f64;

        if self.tokens < 0.0 {
            let deficit = -self.tokens;
            let wait_ns = ((deficit / rate) * 1_000_000_000.0).ceil() as u64;
            self.tokens = 0.0;
            self.last_ns = now_ns.saturating_add(wait_ns);
            self.metrics.throttle_events = self.metrics.throttle_events.saturating_add(1);
            self.metrics.total_throttled_ns =
                self.metrics.total_throttled_ns.saturating_add(wait_ns);
            wait_ns
        } else {
            self.last_ns = now_ns;
            0
        }
    }

    /// Enforces the configured rate limit in real wall-clock time for a completed
    /// batch of `keys`, sleeping only the exact token deficit duration when needed.
    pub fn pace(&mut self, keys: u64) {
        let now_ns = self.started_at.elapsed().as_nanos() as u64;
        let wait_ns = self.consume_at(keys, now_ns);
        if wait_ns > 0 {
            thread::sleep(Duration::from_nanos(wait_ns));
        }
    }
}
