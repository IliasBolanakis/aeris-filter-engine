//! Lock-brief transport circuit breaker guarding [`super::SidecarClient`] RPC paths.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Lifecycle state of the [`super::SidecarClient`] transport circuit breaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Normal operation: RPCs are permitted and consecutive failures are counted.
    Closed,
    /// Sidecar recently failed repeatedly: RPCs short-circuit immediately to
    /// native fallback without touching any socket until `cooldown` elapses.
    Open,
    /// Cooldown elapsed: a single probe RPC is permitted to verify recovery.
    /// Success transitions back to [`Self::Closed`]; failure re-opens [`Self::Open`].
    HalfOpen,
}

/// Immutable observability snapshot of the [`super::SidecarClient`] circuit breaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitBreakerStats {
    /// Current circuit breaker state (`Closed`, `Open`, or `HalfOpen`).
    pub state: CircuitBreakerState,
    /// Current streak of consecutive transport/timeout RPC failures.
    pub consecutive_failures: u32,
    /// Cumulative number of transitions into [`CircuitBreakerState::Open`].
    pub trip_count: u64,
    /// Cumulative number of calls short-circuited without attempting an RPC.
    pub short_circuit_count: u64,
}

/// Pure inner state machine for the sidecar transport circuit breaker.
#[derive(Debug)]
pub(super) struct CircuitBreakerInner {
    pub(super) state: CircuitBreakerState,
    pub(super) consecutive_failures: u32,
    opened_at: Option<Instant>,
    half_open_probe_in_flight: bool,
    pub(super) trip_count: u64,
    pub(super) short_circuit_count: u64,
}

impl CircuitBreakerInner {
    pub(super) const fn new() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            consecutive_failures: 0,
            opened_at: None,
            half_open_probe_in_flight: false,
            trip_count: 0,
            short_circuit_count: 0,
        }
    }

    /// Returns `true` if a caller should immediately short-circuit to native fallback
    /// without starting an RPC or consuming a `HalfOpen` probe.
    pub(super) fn try_fast_short_circuit_at(&mut self, now: Instant, cooldown: Duration) -> bool {
        match self.state {
            CircuitBreakerState::Closed => false,
            CircuitBreakerState::Open => {
                let elapsed = self
                    .opened_at
                    .map_or(cooldown, |t| now.saturating_duration_since(t));
                if elapsed < cooldown {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                if self.half_open_probe_in_flight {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Attempts to acquire permission to execute an RPC at `now`.
    ///
    /// Returns `Some(is_half_open_probe)` if permitted, or `None` if short-circuited.
    pub(super) fn try_acquire_at(&mut self, now: Instant, cooldown: Duration) -> Option<bool> {
        match self.state {
            CircuitBreakerState::Closed => Some(false),
            CircuitBreakerState::Open => {
                let elapsed = self
                    .opened_at
                    .map_or(cooldown, |t| now.saturating_duration_since(t));
                if elapsed >= cooldown {
                    self.state = CircuitBreakerState::HalfOpen;
                    self.half_open_probe_in_flight = true;
                    tracing::info!(
                        cooldown_ms = cooldown.as_millis() as u64,
                        trip_count = self.trip_count,
                        "sidecar circuit breaker transitioning Open -> HalfOpen for recovery probe"
                    );
                    Some(true)
                } else {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    None
                }
            }
            CircuitBreakerState::HalfOpen => {
                if !self.half_open_probe_in_flight {
                    self.half_open_probe_in_flight = true;
                    Some(true)
                } else {
                    self.short_circuit_count = self.short_circuit_count.saturating_add(1);
                    None
                }
            }
        }
    }

    /// Records a successful transport round-trip at `now`, closing the breaker.
    pub(super) fn record_success_at(&mut self, _now: Instant) {
        let prev = self.state;
        self.state = CircuitBreakerState::Closed;
        self.consecutive_failures = 0;
        self.opened_at = None;
        self.half_open_probe_in_flight = false;
        if prev != CircuitBreakerState::Closed {
            tracing::info!(
                previous_state = ?prev,
                trip_count = self.trip_count,
                "sidecar circuit breaker recovered and closed"
            );
        }
    }

    /// Records an RPC-level transport or timeout failure at `now`, tripping to `Open`
    /// once `consecutive_failures >= failure_threshold` or immediately on `HalfOpen` failure.
    pub(super) fn record_transport_failure_at(&mut self, now: Instant, failure_threshold: u32) {
        self.half_open_probe_in_flight = false;
        match self.state {
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.opened_at = Some(now);
                self.trip_count = self.trip_count.saturating_add(1);
                tracing::warn!(
                    trip_count = self.trip_count,
                    "sidecar HalfOpen probe failed; re-opening circuit breaker"
                );
            }
            CircuitBreakerState::Closed => {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                if self.consecutive_failures >= failure_threshold {
                    self.state = CircuitBreakerState::Open;
                    self.opened_at = Some(now);
                    self.trip_count = self.trip_count.saturating_add(1);
                    tracing::warn!(
                        consecutive_failures = self.consecutive_failures,
                        failure_threshold,
                        trip_count = self.trip_count,
                        "sidecar circuit breaker tripped Closed -> Open"
                    );
                }
            }
            CircuitBreakerState::Open => {
                self.opened_at = Some(now);
            }
        }
    }

    /// Releases an in-flight `HalfOpen` probe slot if the caller was cancelled before completion.
    pub(super) fn release_aborted_probe(&mut self) {
        if self.state == CircuitBreakerState::HalfOpen {
            self.half_open_probe_in_flight = false;
        }
    }

    /// Computes effective state at `now` without mutating internal counters.
    pub(super) fn effective_state_at(
        &self,
        now: Instant,
        cooldown: Duration,
    ) -> CircuitBreakerState {
        match self.state {
            CircuitBreakerState::Open => {
                let elapsed = self
                    .opened_at
                    .map_or(cooldown, |t| now.saturating_duration_since(t));
                if elapsed >= cooldown {
                    CircuitBreakerState::HalfOpen
                } else {
                    CircuitBreakerState::Open
                }
            }
            other => other,
        }
    }

    pub(super) fn stats_at(&self, now: Instant, cooldown: Duration) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.effective_state_at(now, cooldown),
            consecutive_failures: self.consecutive_failures,
            trip_count: self.trip_count,
            short_circuit_count: self.short_circuit_count,
        }
    }
}

/// Thread-safe, lock-brief circuit breaker guarding [`super::SidecarClient`] RPC paths.
///
/// # Lock Ordering
/// `inner` is a synchronous leaf `std::sync::Mutex` never held across `.await` points.
#[derive(Debug)]
pub(super) struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    inner: Mutex<CircuitBreakerInner>,
}

impl CircuitBreaker {
    pub(super) fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            inner: Mutex::new(CircuitBreakerInner::new()),
        }
    }

    pub(super) fn try_fast_short_circuit(&self) -> bool {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.try_fast_short_circuit_at(Instant::now(), self.cooldown)
    }

    pub(super) fn try_acquire(&self) -> Option<CircuitBreakerPermit<'_>> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner
            .try_acquire_at(Instant::now(), self.cooldown)
            .map(|is_half_open_probe| CircuitBreakerPermit {
                breaker: self,
                is_half_open_probe,
                completed: false,
            })
    }

    pub(super) fn state(&self) -> CircuitBreakerState {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.effective_state_at(Instant::now(), self.cooldown)
    }

    pub(super) fn stats(&self) -> CircuitBreakerStats {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.stats_at(Instant::now(), self.cooldown)
    }
}

/// RAII permit guaranteeing that a cancelled `HalfOpen` probe never strands the
/// circuit breaker in an un-probeable state.
pub(super) struct CircuitBreakerPermit<'a> {
    breaker: &'a CircuitBreaker,
    is_half_open_probe: bool,
    completed: bool,
}

impl CircuitBreakerPermit<'_> {
    pub(super) fn record_success(mut self) {
        self.completed = true;
        let mut inner = self
            .breaker
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.record_success_at(Instant::now());
    }

    pub(super) fn record_transport_failure(mut self) {
        self.completed = true;
        let mut inner = self
            .breaker
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.record_transport_failure_at(Instant::now(), self.breaker.failure_threshold);
    }
}

impl Drop for CircuitBreakerPermit<'_> {
    fn drop(&mut self) {
        if !self.completed && self.is_half_open_probe {
            let mut inner = self
                .breaker
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.release_aborted_probe();
        }
    }
}
