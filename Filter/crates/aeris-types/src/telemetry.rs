//! Operational reports, workload statistics, optimization decisions, and
//! Advisory telemetry and guardrail DTOs.

use crate::range::{Epoch, ShardId};
use serde::{Deserialize, Serialize};

/// Diagnostic report produced after adapting an Aeris Filter to a false-positive range query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptationReport {
    /// Number of partition subqueries inspected for the false-positive range.
    pub subqueries_checked: usize,
    /// Number of keepsake boxes split or extended.
    pub boxes_split: usize,
    /// Total keys retrieved from the Reverse Map and rehashed.
    pub keys_rehashed: usize,
    /// Total extension slots added across all adapted keepsake boxes.
    pub extension_slots_added: usize,
    /// Number of aged keepsake boxes opportunistically rejuvenated to age 0.
    pub rejuvenated_boxes: usize,
}

/// Diagnostic report produced after a `2x` InfiniFilter expansion (`q -> q + 1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpansionReport {
    /// Quotient bit width before expansion.
    pub old_q: u8,
    /// Quotient bit width after expansion (`old_q + 1`).
    pub new_q: u8,
    /// Number of keepsake boxes transferred in-memory via bit-splitting.
    pub boxes_transferred: usize,
    /// Number of oldest keepsake boxes selectively rejuvenated via the Reverse Map.
    pub boxes_rejuvenated: usize,
    /// Total keys read from the Reverse Map during selective rejuvenation.
    pub keys_rejuvenated: usize,
    /// Wall-clock duration of the expansion in nanoseconds.
    pub duration_ns: u64,
}

/// Sliding-window workload summary produced by a [`crate::traits::WorkloadEstimator`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkloadStats {
    /// Total queries observed in the active sliding window.
    pub window_query_count: u64,
    /// Observed empirical false positive rate within the active window.
    pub observed_fpr: f64,
    /// Median (`p50`) range query length in the sliding window (after outlier clipping).
    pub p50_query_len: u64,
    /// 95th percentile (`p95`) range query length in the sliding window (after outlier clipping).
    pub p95_query_len: u64,
    /// 99th percentile (`p99`) range query length in the sliding window (after outlier clipping).
    pub p99_query_len: u64,
    /// Hysteresis-stabilized effective target range length `R_eff`.
    pub effective_query_span: u64,
    /// Number of extreme outlier queries clipped by the estimator.
    pub clipped_outliers: u64,
    /// EWMA key ingestion rate (keys/second).
    pub ingestion_rate_ewma: f64,
    /// Observed bulk-load speed (keys/second) used in `T_rebuild` estimation.
    pub bulk_load_speed: f64,
    /// Observed query arrival rate (queries/second).
    pub query_rate: f64,
    /// Average number of keys per non-empty partition (`mu`).
    pub avg_keys_per_partition: f64,
}

impl Default for WorkloadStats {
    fn default() -> Self {
        Self {
            window_query_count: 0,
            observed_fpr: 0.0,
            p50_query_len: 1,
            p95_query_len: 1,
            p99_query_len: 1,
            effective_query_span: 1,
            clipped_outliers: 0,
            ingestion_rate_ewma: 0.0,
            bulk_load_speed: 500_000.0,
            query_rate: 0.0,
            avg_keys_per_partition: 1.0,
        }
    }
}

/// Output of the deterministic Discrete Cost Optimizer evaluating `m = log2(P)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationDecision {
    /// Optimal memento bit width `m* = log2(P*)` selected from discrete candidates.
    pub recommended_m: u8,
    /// Corresponding remainder bit width `r* = C - m*`.
    pub recommended_r: u8,
    /// Expected query cost (nanoseconds) under the recommended `(r*, m*)`.
    pub estimated_cost_ns: f64,
    /// Expected query cost (nanoseconds) under the current `(r, m)`.
    pub current_cost_ns: f64,
    /// `true` if bypassing the range filter is cheaper than probing it for the
    /// current query span (e.g., extremely wide full-table range scans).
    pub should_bypass_filter: bool,
    /// `true` if switching from current `m` to `recommended_m` exceeds the
    /// configured hysteresis benefit threshold.
    pub should_rebuild: bool,
}

/// Snapshot of runtime telemetry sent asynchronously from the Control Plane to
/// the optional ML/RL Advisory Sidecar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Unique idempotency key for RPC deduplication across bounded retries.
    pub request_id: String,
    /// Identifier of the target shard (0 for single-instance filters).
    pub shard_id: ShardId,
    /// Current configuration epoch.
    pub epoch: Epoch,
    /// Snapshot wall-clock timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
    /// Current quotient bit width `q`.
    pub current_q: u8,
    /// Current remainder bit width `r`.
    pub current_r: u8,
    /// Current memento bit width `m`.
    pub current_m: u8,
    /// Fixed slot payload budget `C`.
    pub payload_budget_c: u8,
    /// Target false positive rate `epsilon`.
    pub target_epsilon: f64,
    /// Current load factor `alpha`.
    pub load_factor: f64,
    /// EWMA ingestion rate (keys/sec).
    pub ingestion_rate_ewma: f64,
    /// Recent query rate (queries/sec).
    pub query_rate: f64,
    /// Observed empirical FPR.
    pub observed_fpr: f64,
    /// Observed `p95` query length.
    pub p95_query_len: f64,
    /// Observed `p99` query length.
    pub p99_query_len: f64,
    /// Estimated total entries in the Reverse Map.
    pub reverse_map_entries: u64,
    /// Observed bulk-load speed (keys/sec).
    pub bulk_load_speed: f64,
    /// Recent time-series of query span observations for seasonal forecasting.
    pub recent_query_len_series: Vec<f64>,
    /// Recent time-series of ingestion rate observations for seasonal forecasting.
    pub recent_ingest_rate_series: Vec<f64>,
}

/// Parameter tuning proposal returned by a [`crate::traits::TuningAdvisor`].
///
/// Every proposal MUST be validated by the deterministic Rust `SafetyGuardrail`
/// before it can mutate any filter configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningProposal {
    /// Echoed idempotency key from the triggering [`TelemetrySnapshot`].
    pub request_id: String,
    /// Proposed remainder bit width `r`.
    pub proposed_r: u8,
    /// Proposed memento bit width `m = log2(P)`.
    pub proposed_m: u8,
    /// Proposed target load factor ceiling `alpha` (must lie in `[0.50, 0.90]`).
    pub proposed_target_alpha: f64,
    /// Proposed proactive rebuild margin multiplier `k` (must lie in `[1.0, 3.0]`).
    pub proposed_proactive_margin_k: f64,
    /// Model confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Whether the advisor recommends proactively launching a background rebuild.
    pub trigger_proactive_rebuild: bool,
    /// Identifier/version of the producing model or fallback optimizer.
    pub model_version: String,
    /// Human-readable rationale for observability logs.
    pub rationale: String,
    /// Forecasted peak ingestion rate (keys/sec) over the upcoming seasonal window.
    pub forecast_peak_ingest_rate: f64,
}

/// Per-shard telemetry used for multi-tenant memory allocation via offline contextual bandits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardTelemetrySnapshot {
    /// Target shard identifier.
    pub shard_id: ShardId,
    /// Number of keys currently stored in the shard.
    pub key_count: u64,
    /// Current RAM footprint of the shard in bytes.
    pub current_memory_bytes: u64,
    /// Recent query arrival rate (queries/sec) routed to this shard.
    pub query_rate: f64,
    /// Observed empirical FPR on this shard.
    pub observed_fpr: f64,
    /// Observed `p95` range query length on this shard.
    pub p95_query_len: f64,
    /// Current load factor `alpha` of this shard.
    pub load_factor: f64,
    /// Current quotient width `q`.
    pub current_q: u8,
    /// Current remainder width `r`.
    pub current_r: u8,
    /// Current memento width `m`.
    pub current_m: u8,
}

/// Memory budget recommendation for a single shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardAllocationDecision {
    /// Target shard identifier.
    pub shard_id: ShardId,
    /// Allocated RAM budget in bytes.
    pub allocated_memory_bytes: u64,
    /// Recommended power-of-two quotient width `q_i`.
    pub recommended_q: u8,
    /// Recommended remainder width `r_i`.
    pub recommended_r: u8,
    /// Recommended memento width `m_i`.
    pub recommended_m: u8,
}

/// Multi-tenant RAM allocation plan across all active range shards.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryAllocationPlan {
    /// Echoed idempotency key.
    pub request_id: String,
    /// Per-shard allocations summing to at most the requested total budget.
    pub allocations: Vec<ShardAllocationDecision>,
    /// Model confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Identifier/version of the producing bandit model or deterministic fallback.
    pub model_version: String,
}

/// Outcome of validating a [`TuningProposal`] against [`crate::config::SafetyBounds`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardrailVerdict {
    /// Proposal satisfied every deterministic invariant and may be applied.
    Approved,
    /// Proposal violated one or more deterministic invariants and was rejected.
    Rejected {
        /// Ordered list of all violated guardrail constraints.
        reasons: Vec<String>,
    },
}
