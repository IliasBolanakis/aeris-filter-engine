//! Deterministic, zero-I/O safety guardrail for validating Phase 5 ML/RL
//! control-plane tuning proposals and multi-tenant memory allocation plans.
//!
//! Every proposal returned by an out-of-process advisor MUST pass
//! [`SafetyGuardrail::validate_tuning`] or [`SafetyGuardrail::validate_memory_plan`]
//! before any filter configuration or shard allocation may be mutated.
//!
//! # Invariants Enforced
//! For macro-level tuning proposals (`validate_tuning`):
//! - Monotonic adaptivity floor: `proposed_r >= FilterConfig::min_remainder_bits(bounds.target_epsilon)`.
//! - Memento upper bound: `proposed_m <= bounds.m_max`.
//! - Fixed per-slot payload budget: `(proposed_r as u16) + (proposed_m as u16) <= (bounds.payload_budget_c as u16)`.
//! - Target load factor ceiling: `proposed_target_alpha` is finite and in `[bounds.min_alpha, bounds.max_alpha]`.
//! - Proactive rebuild safety margin: `proposed_proactive_margin_k` is finite and in
//!   `[bounds.min_proactive_margin_k, bounds.max_proactive_margin_k]`.
//! - Model confidence: `confidence` is finite and in `[bounds.min_confidence, 1.0]`.
//! - Forecasted peak ingestion rate: `forecast_peak_ingest_rate` is finite and `>= 0.0`.
//!
//! For multi-tenant RAM allocation plans (`validate_memory_plan`):
//! - Total allocated bytes across all shards `<= total_budget_bytes` (checked with `u128` arithmetic).
//! - No duplicate shard identifiers in the allocation list.
//! - Every shard's `recommended_q` lies in `[MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS]` (`[6, 40]`).
//! - Every shard's `(recommended_r, recommended_m)` satisfies `r >= r_min(epsilon)`,
//!   `m <= bounds.m_max`, and `r + m <= bounds.payload_budget_c`.
//! - Plan `confidence` is finite and in `[bounds.min_confidence, 1.0]`.

use std::collections::HashSet;

use aeris_types::{
    FilterConfig, GuardrailVerdict, MemoryAllocationPlan, SafetyBounds, ShardAllocationDecision,
    ShardId, TuningProposal, MAX_QUOTIENT_BITS, MIN_QUOTIENT_BITS,
};

/// Pure, stateless, deterministic validator enforcing all mathematical and
/// operational safety constraints on out-of-process ML/RL proposals.
///
/// Contains zero I/O, zero locks, and zero mutable state so it can be invoked
/// deterministically in production and under simulation testing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SafetyGuardrail;

impl SafetyGuardrail {
    /// Constructs a new [`SafetyGuardrail`] instance.
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Validates a macro-level [`TuningProposal`] against [`SafetyBounds`].
    ///
    /// Evaluates every single guardrail invariant without short-circuiting so
    /// operators and diagnostics observe the complete set of violations in
    /// [`GuardrailVerdict::Rejected`]. Logs a structured `tracing::warn!` event
    /// whenever a proposal is rejected.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space (bounded vector of at most 8 violation messages).
    pub fn validate_tuning(
        &self,
        proposal: &TuningProposal,
        bounds: &SafetyBounds,
    ) -> GuardrailVerdict {
        let mut reasons = Vec::new();

        validate_geometry_bits(
            proposal.proposed_r,
            proposal.proposed_m,
            bounds,
            None,
            &mut reasons,
        );
        validate_bounded_f64(
            "proposed_target_alpha",
            proposal.proposed_target_alpha,
            bounds.min_alpha,
            bounds.max_alpha,
            &mut reasons,
        );
        validate_bounded_f64(
            "proposed_proactive_margin_k",
            proposal.proposed_proactive_margin_k,
            bounds.min_proactive_margin_k,
            bounds.max_proactive_margin_k,
            &mut reasons,
        );
        validate_confidence(proposal.confidence, bounds.min_confidence, &mut reasons);
        validate_non_negative_f64(
            "forecast_peak_ingest_rate",
            proposal.forecast_peak_ingest_rate,
            &mut reasons,
        );

        if reasons.is_empty() {
            tracing::debug!(
                request_id = %proposal.request_id,
                proposed_r = proposal.proposed_r,
                proposed_m = proposal.proposed_m,
                alpha = proposal.proposed_target_alpha,
                proactive_margin_k = proposal.proposed_proactive_margin_k,
                confidence = proposal.confidence,
                model_version = %proposal.model_version,
                "safety guardrail approved tuning proposal"
            );
            GuardrailVerdict::Approved
        } else {
            tracing::warn!(
                request_id = %proposal.request_id,
                model_version = %proposal.model_version,
                violation_count = reasons.len(),
                reasons = ?reasons,
                "safety guardrail rejected tuning proposal"
            );
            GuardrailVerdict::Rejected { reasons }
        }
    }

    /// Validates a multi-tenant [`MemoryAllocationPlan`] against the global
    /// `total_budget_bytes` cap and per-shard [`SafetyBounds`].
    ///
    /// Uses widening `u128` summation to prevent `u64` wraparound from bypassing
    /// the global RAM budget invariant. Collects all violations into
    /// [`GuardrailVerdict::Rejected`] and logs `tracing::warn!` on rejection.
    ///
    /// # Complexity
    /// `O(|allocations|)` time and `O(|allocations|)` space for duplicate shard detection.
    pub fn validate_memory_plan(
        &self,
        plan: &MemoryAllocationPlan,
        total_budget_bytes: u64,
        bounds: &SafetyBounds,
    ) -> GuardrailVerdict {
        let mut reasons = Vec::new();

        validate_confidence(plan.confidence, bounds.min_confidence, &mut reasons);

        let r_min = match FilterConfig::min_remainder_bits(bounds.target_epsilon) {
            Ok(v) => Some(v),
            Err(err) => {
                reasons.push(format!("invalid safety bounds target_epsilon: {err}"));
                None
            }
        };

        let mut total_allocated: u128 = 0;
        let mut seen_shards: HashSet<ShardId> = HashSet::with_capacity(plan.allocations.len());

        for alloc in &plan.allocations {
            total_allocated += u128::from(alloc.allocated_memory_bytes);
            if !seen_shards.insert(alloc.shard_id) {
                reasons.push(format!(
                    "duplicate shard_id {} in memory allocation plan",
                    alloc.shard_id
                ));
            }
            validate_shard_allocation(alloc, bounds, r_min, &mut reasons);
        }

        if total_allocated > u128::from(total_budget_bytes) {
            reasons.push(format!(
                "total allocated memory ({total_allocated} bytes) exceeds total_budget_bytes ({total_budget_bytes} bytes)"
            ));
        }

        if reasons.is_empty() {
            tracing::debug!(
                request_id = %plan.request_id,
                shard_count = plan.allocations.len(),
                total_allocated_bytes = total_allocated,
                total_budget_bytes,
                confidence = plan.confidence,
                model_version = %plan.model_version,
                "safety guardrail approved memory allocation plan"
            );
            GuardrailVerdict::Approved
        } else {
            tracing::warn!(
                request_id = %plan.request_id,
                model_version = %plan.model_version,
                violation_count = reasons.len(),
                reasons = ?reasons,
                "safety guardrail rejected memory allocation plan"
            );
            GuardrailVerdict::Rejected { reasons }
        }
    }
}

/// Validates `(r, m)` bit allocation invariants against `bounds`.
fn validate_geometry_bits(
    r: u8,
    m: u8,
    bounds: &SafetyBounds,
    shard_id: Option<ShardId>,
    reasons: &mut Vec<String>,
) {
    let prefix = shard_prefix(shard_id);
    match FilterConfig::min_remainder_bits(bounds.target_epsilon) {
        Ok(r_min) => {
            if r < r_min {
                reasons.push(format!(
                    "{prefix}remainder bits r ({r}) < r_min(epsilon={}) ({r_min})",
                    bounds.target_epsilon
                ));
            }
        }
        Err(err) => {
            reasons.push(format!(
                "{prefix}invalid safety bounds target_epsilon: {err}"
            ));
        }
    }

    if m > bounds.m_max {
        reasons.push(format!(
            "{prefix}memento bits m ({m}) > m_max ({})",
            bounds.m_max
        ));
    }

    let sum_bits = u16::from(r) + u16::from(m);
    let budget_c = u16::from(bounds.payload_budget_c);
    if sum_bits > budget_c {
        reasons.push(format!(
            "{prefix}r ({r}) + m ({m}) = {sum_bits} exceeds payload_budget_c ({budget_c})"
        ));
    }
}

/// Validates a single shard's `(q, r, m)` allocation inside a memory plan.
fn validate_shard_allocation(
    alloc: &ShardAllocationDecision,
    bounds: &SafetyBounds,
    r_min: Option<u8>,
    reasons: &mut Vec<String>,
) {
    let prefix = shard_prefix(Some(alloc.shard_id));

    if alloc.recommended_q < MIN_QUOTIENT_BITS || alloc.recommended_q > MAX_QUOTIENT_BITS {
        reasons.push(format!(
            "{prefix}recommended_q ({}) must be in [{MIN_QUOTIENT_BITS}, {MAX_QUOTIENT_BITS}]",
            alloc.recommended_q
        ));
    }

    if let Some(floor) = r_min {
        if alloc.recommended_r < floor {
            reasons.push(format!(
                "{prefix}recommended_r ({}) < r_min(epsilon={}) ({floor})",
                alloc.recommended_r, bounds.target_epsilon
            ));
        }
    }

    if alloc.recommended_m > bounds.m_max {
        reasons.push(format!(
            "{prefix}recommended_m ({}) > m_max ({})",
            alloc.recommended_m, bounds.m_max
        ));
    }

    let sum_bits = u16::from(alloc.recommended_r) + u16::from(alloc.recommended_m);
    let budget_c = u16::from(bounds.payload_budget_c);
    if sum_bits > budget_c {
        reasons.push(format!(
            "{prefix}recommended_r ({}) + recommended_m ({}) = {sum_bits} exceeds payload_budget_c ({budget_c})",
            alloc.recommended_r, alloc.recommended_m
        ));
    }
}

/// Validates that `value` is finite and lies in `[min_val, max_val]`.
fn validate_bounded_f64(
    field_name: &str,
    value: f64,
    min_val: f64,
    max_val: f64,
    reasons: &mut Vec<String>,
) {
    if !min_val.is_finite() || !max_val.is_finite() || min_val > max_val {
        reasons.push(format!(
            "invalid safety bounds for {field_name}: [{min_val}, {max_val}]"
        ));
        return;
    }
    if !value.is_finite() || value < min_val || value > max_val {
        reasons.push(format!(
            "{field_name} ({value}) must be finite and in [{min_val}, {max_val}]"
        ));
    }
}

/// Validates model confidence is finite and in `[min_confidence, 1.0]`.
fn validate_confidence(confidence: f64, min_confidence: f64, reasons: &mut Vec<String>) {
    if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
        reasons.push(format!(
            "invalid safety bounds min_confidence ({min_confidence}) outside [0.0, 1.0]"
        ));
        return;
    }
    if !confidence.is_finite() || confidence < min_confidence || confidence > 1.0 {
        reasons.push(format!(
            "confidence ({confidence}) must be finite and in [{min_confidence}, 1.0]"
        ));
    }
}

/// Validates that a non-negative floating-point quantity is finite and `>= 0.0`.
fn validate_non_negative_f64(field_name: &str, value: f64, reasons: &mut Vec<String>) {
    if !value.is_finite() || value < 0.0 {
        reasons.push(format!("{field_name} ({value}) must be finite and >= 0.0"));
    }
}

#[inline]
fn shard_prefix(shard_id: Option<ShardId>) -> String {
    match shard_id {
        Some(id) => format!("shard {id}: "),
        None => String::new(),
    }
}
