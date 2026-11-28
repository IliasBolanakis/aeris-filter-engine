//! Lock-free atomic observability metrics for [`super::GuardedTuningAdvisor`].

use std::sync::atomic::{AtomicU64, Ordering};

use aeris_types::AerisError;

/// Lock-free atomic observability counters for [`super::GuardedTuningAdvisor`].
///
/// # Conservation Invariant
/// After every completed `advise` or `allocate_memory` call:
/// `total_calls == sidecar_accepted + guardrail_rejections + transport_fallbacks`.
#[derive(Debug, Default)]
pub struct SidecarMetrics {
    /// Total `advise` and `allocate_memory` calls served by the advisor.
    pub total_calls: AtomicU64,
    /// Proposals returned by the sidecar and approved by [`crate::SafetyGuardrail`].
    pub sidecar_accepted: AtomicU64,
    /// Proposals returned by the sidecar that failed [`crate::SafetyGuardrail`] and
    /// triggered native fallback.
    pub guardrail_rejections: AtomicU64,
    /// Calls that fell back to the native optimizer because the sidecar was
    /// unconfigured, unreachable, crashed, errored, or timed out.
    pub transport_fallbacks: AtomicU64,
}

impl SidecarMetrics {
    /// Captures a plain value snapshot of all atomic counters.
    #[inline]
    pub fn snapshot(&self) -> SidecarMetricsSnapshot {
        SidecarMetricsSnapshot {
            total_calls: self.total_calls.load(Ordering::Relaxed),
            sidecar_accepted: self.sidecar_accepted.load(Ordering::Relaxed),
            guardrail_rejections: self.guardrail_rejections.load(Ordering::Relaxed),
            transport_fallbacks: self.transport_fallbacks.load(Ordering::Relaxed),
        }
    }

    #[inline]
    pub(super) fn record_rpc_error(&self, err: &AerisError) {
        match err {
            AerisError::GuardrailRejected { .. } => {
                self.guardrail_rejections.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.transport_fallbacks.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Immutable point-in-time snapshot of [`SidecarMetrics`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SidecarMetricsSnapshot {
    /// Total `advise` and `allocate_memory` calls served.
    pub total_calls: u64,
    /// Sidecar responses approved by [`crate::SafetyGuardrail`].
    pub sidecar_accepted: u64,
    /// Sidecar responses rejected by [`crate::SafetyGuardrail`] (fell back to native).
    pub guardrail_rejections: u64,
    /// Sidecar transport/timeout/disabled fallbacks to native optimizer.
    pub transport_fallbacks: u64,
}
