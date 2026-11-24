//! # `aeris-types`
//!
//! Shared domain types, mathematical invariants, canonical variable-length
//! fingerprints, bit-layout constants, and cross-plane trait contracts for the
//! **Aeris-Based Optimized Range Filtering System**.
//!
//! This crate contains zero I/O, zero network/IPC dependencies, and zero
//! `unsafe` code, establishing the Milestone M0 contract boundary consumed by
//! all other workspace crates.

#![deny(missing_docs)]

pub mod config;
pub mod constants;
pub mod error;
pub mod fingerprint;
pub mod hash;
pub mod range;
pub mod telemetry;
pub mod traits;
pub mod wal;

pub use config::{CostModelParams, FilterConfig, SafetyBounds, ShardBounds};
pub use constants::{
    KeepsakeSlotKind, DEFAULT_HASH_SEED, DEFAULT_MIN_CONFIDENCE, DEFAULT_TARGET_LOAD_FACTOR,
    KEY_BITS, MAX_CHUNK_OFFSET, MAX_GUARDRAIL_ALPHA, MAX_GUARDRAIL_PROACTIVE_MARGIN,
    MAX_LOAD_FACTOR, MAX_MEMENTO_BITS, MAX_QUOTIENT_BITS, METADATA_BITS_PER_SLOT,
    MIN_GUARDRAIL_ALPHA, MIN_GUARDRAIL_PROACTIVE_MARGIN, MIN_QUOTIENT_BITS, SLOTS_PER_CHUNK,
    SLOTS_PER_CHUNK_LOG2,
};
pub use error::{AerisError, AerisResult};
pub use fingerprint::Fingerprint;
pub use hash::PartitionHasher;
pub use range::{
    Epoch, Key, Lsn, Memento, PartitionId, PartitionSubqueryIter, QueryRange, ShardId, SlotIndex,
};
pub use telemetry::{
    AdaptationReport, ExpansionReport, GuardrailVerdict, MemoryAllocationPlan,
    OptimizationDecision, ShardAllocationDecision, ShardTelemetrySnapshot, TelemetrySnapshot,
    TuningProposal, WorkloadStats,
};
pub use traits::{
    CostOptimizer, RangeFilter, ReverseMapStore, TuningAdvisor, WalEngine, WorkloadEstimator,
};
pub use wal::{FpLogEntry, WalOp, WalRecord};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_enforces_mathematical_invariants() {
        // eps = 2^-8 -> log2(1/eps) = 8, log2(8) = 3, r_min = 8 + 3 + 1 = 12
        let r_min = FilterConfig::min_remainder_bits(1.0 / 256.0).unwrap();
        assert_eq!(r_min, 12);

        // R = 32 -> m = 5, P = 32, C = 17
        let cfg = FilterConfig::new(10, 12, 32, 17, 1.0 / 256.0).unwrap();
        assert_eq!(cfg.num_slots(), 1024);
        assert_eq!(cfg.num_chunks(), 16);
        assert_eq!(cfg.partition_size(), 32);
        assert_eq!(cfg.m, 5);

        // Reject r below r_min
        assert!(FilterConfig::new(10, 11, 32, 16, 1.0 / 256.0).is_err());
        // Reject r + m != C
        assert!(FilterConfig::new(10, 12, 32, 18, 1.0 / 256.0).is_err());
        // Reject q < 6
        assert!(FilterConfig::new(5, 12, 32, 17, 1.0 / 256.0).is_err());
        // Reject R == 0
        assert!(FilterConfig::new(10, 12, 0, 12, 1.0 / 256.0).is_err());
        // Reject epsilon >= 0.5 or <= 0.0
        assert!(FilterConfig::min_remainder_bits(0.5).is_err());
        assert!(FilterConfig::min_remainder_bits(0.0).is_err());
        assert!(FilterConfig::min_remainder_bits(f64::NAN).is_err());
    }

    #[test]
    fn config_reconfigure_preserves_budget() {
        let cfg = FilterConfig::new(10, 15, 32, 20, 1.0 / 256.0).unwrap();
        let next = cfg.reconfigure_for_m(8).unwrap();
        assert_eq!(next.m, 8);
        assert_eq!(next.r, 12);
        assert_eq!(next.max_range_r, 256);
        assert_eq!(next.payload_budget_c, 20);
        // m=9 -> r=11 < r_min(2^-8)=12 -> rejected
        assert!(cfg.reconfigure_for_m(9).is_err());
    }

    #[test]
    fn fingerprint_prefix_and_expansion_invariance() {
        let hasher = PartitionHasher::new(42);
        let part = 0x1234_5678_9ABC_u64;

        let fp_short = Fingerprint::from_hasher(&hasher, part, 22);
        let fp_long = Fingerprint::from_hasher(&hasher, part, 175);
        assert!(fp_short.is_prefix_of(&fp_long));
        assert!(!fp_long.is_prefix_of(&fp_short));

        // Simulate q=10, r=12 -> expand to q=11, r_eff=11: canonical fingerprint is identical!
        let (q10, r12) = hasher.quotient_and_remainder(part, 10, 12);
        let fp_before = Fingerprint::from_parts(q10, 10, r12, 12, &[]);

        let msb_r = (r12 >> 11) & 1;
        let q11 = (q10 << 1) | msb_r;
        let r11 = r12 & ((1 << 11) - 1);
        let fp_after = Fingerprint::from_parts(q11, 11, r11, 11, &[]);

        assert_eq!(fp_before, fp_short);
        assert_eq!(fp_before, fp_after);

        // Byte round-trip for both inline (<=128) and spilled (>128) fingerprints
        let bytes_s = fp_short.to_bytes();
        assert_eq!(Fingerprint::from_bytes(&bytes_s).unwrap(), fp_short);
        let bytes_l = fp_long.to_bytes();
        assert_eq!(Fingerprint::from_bytes(&bytes_l).unwrap(), fp_long);
    }

    #[test]
    fn query_range_partition_subqueries_cover_endpoints_and_max_u64() {
        let r = QueryRange::new(30, 35).unwrap();
        let subs: Vec<_> = r.partition_subqueries(5).collect();
        assert_eq!(subs, vec![(0, 30, 31), (1, 0, 3)]);

        let max_r = QueryRange::new(u64::MAX - 2, u64::MAX).unwrap();
        assert_eq!(max_r.len_u128(), 3);
        let subs_max: Vec<_> = max_r.partition_subqueries(5).collect();
        assert_eq!(subs_max.len(), 1);
        assert_eq!(subs_max[0].1, 29);
        assert_eq!(subs_max[0].2, 31);

        let full = QueryRange::new(0, u64::MAX).unwrap();
        assert_eq!(full.len_u128(), (u64::MAX as u128) + 1);
        assert_eq!(full.len_saturated(), u64::MAX);
    }

    #[test]
    fn keepsake_slot_kind_round_trip() {
        for (re, ext, kind) in [
            (false, false, KeepsakeSlotKind::PayloadInterior),
            (false, true, KeepsakeSlotKind::Extension),
            (true, false, KeepsakeSlotKind::BoxEnd),
            (true, true, KeepsakeSlotKind::BoxAndRunEnd),
        ] {
            assert_eq!(KeepsakeSlotKind::from_bits(re, ext), kind);
            assert_eq!(kind.to_bits(), (re, ext));
        }
    }
}
