//! Structural metadata invariant verification for [`QuotientTable`].

use super::{QuotientTable, TRAILING_GUARD_BYTES};
use aeris_types::{AerisError, AerisResult, KeepsakeSlotKind, SLOTS_PER_CHUNK_LOG2};

impl QuotientTable {
    /// Verifies all structural Quotient Filter metadata invariants:
    /// 1. Tail chunks have `occupied == 0`.
    /// 2. Total `occupied` bits == total `qrun_end` bits.
    /// 3. Every chunk's `offset` exactly equals the canonical `E_prev` spillover formula.
    /// 4. Every `occupied` home slot's run starts at `>= home` and has `BoxAndRunEnd` at `run_end`.
    /// 5. No slot outside occupied runs has stray `runend` or `extension` bits.
    /// 6. Trailing 8 guard bytes remain strictly zero.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] describing the first violation found.
    pub fn verify_metadata_invariants(&self) -> AerisResult<()> {
        self.verify_buffer_and_tail_invariants()?;
        let (occ_slots, qrun_ends) = self.collect_occupied_and_qrun_ends()?;
        self.verify_chunk_offsets(&occ_slots, &qrun_ends)?;
        self.verify_runs_and_gaps(&occ_slots, &qrun_ends)
    }

    fn verify_buffer_and_tail_invariants(&self) -> AerisResult<()> {
        let expected_len = self.total_chunks * self.block_bytes + TRAILING_GUARD_BYTES;
        if self.blocks.len() != expected_len {
            return Err(AerisError::InternalInvariant(format!(
                "AoS block buffer length mismatch: actual={}, expected={expected_len}",
                self.blocks.len()
            )));
        }
        let guard_start = self.total_chunks * self.block_bytes;
        if self.blocks[guard_start..].iter().any(|&b| b != 0) {
            return Err(AerisError::InternalInvariant(
                "non-zero byte detected in trailing AoS guard region".to_string(),
            ));
        }
        for c in self.canonical_chunks..self.total_chunks {
            let occ = self.block_occupied(c);
            if occ != 0 {
                return Err(AerisError::InternalInvariant(format!(
                    "tail chunk {c} has non-zero occupied mask {occ:#x}"
                )));
            }
        }
        Ok(())
    }

    fn collect_occupied_and_qrun_ends(&self) -> AerisResult<(Vec<usize>, Vec<usize>)> {
        let mut occ_slots = Vec::new();
        for c in 0..self.canonical_chunks {
            let mut w = self.block_occupied(c);
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                occ_slots.push((c << SLOTS_PER_CHUNK_LOG2) + bit);
                w &= w - 1;
            }
        }

        let mut qrun_ends = Vec::new();
        for c in 0..self.total_chunks {
            let mut w = self.block_qrun_end_mask(c);
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                qrun_ends.push((c << SLOTS_PER_CHUNK_LOG2) + bit);
                w &= w - 1;
            }
        }

        if occ_slots.len() != qrun_ends.len() {
            return Err(AerisError::InternalInvariant(format!(
                "occupied count ({}) != qrun_end count ({})",
                occ_slots.len(),
                qrun_ends.len()
            )));
        }
        Ok((occ_slots, qrun_ends))
    }

    fn verify_chunk_offsets(&self, occ_slots: &[usize], qrun_ends: &[usize]) -> AerisResult<()> {
        for c in 0..self.total_chunks {
            let base = c << SLOTS_PER_CHUNK_LOG2;
            let count_before = occ_slots.partition_point(|&h| h < base);
            let expected_offset = if count_before == 0 {
                0
            } else {
                let e_prev = qrun_ends[count_before - 1];
                (e_prev + 1).saturating_sub(base)
            };
            let actual_offset = self.chunk_offset(c);
            if actual_offset != expected_offset {
                return Err(AerisError::InternalInvariant(format!(
                    "chunk {c} offset mismatch: actual={actual_offset}, expected={expected_offset}"
                )));
            }
        }
        Ok(())
    }

    fn verify_runs_and_gaps(&self, occ_slots: &[usize], qrun_ends: &[usize]) -> AerisResult<()> {
        let mut prev_end_plus_one = 0usize;
        for (idx, (&home, &expected_end)) in occ_slots.iter().zip(qrun_ends.iter()).enumerate() {
            let expected_start = home.max(prev_end_plus_one);
            if expected_end < expected_start {
                return Err(AerisError::InternalInvariant(format!(
                    "run #{idx} for home {home}: end {expected_end} < start {expected_start}"
                )));
            }
            for gap_slot in prev_end_plus_one..expected_start {
                if self.slot_kind(gap_slot) != KeepsakeSlotKind::PayloadInterior {
                    return Err(AerisError::InternalInvariant(format!(
                        "stray metadata {:?} at empty gap slot {gap_slot}",
                        self.slot_kind(gap_slot)
                    )));
                }
            }
            let located = self.locate_run(home);
            if located != Some((expected_start, expected_end)) {
                return Err(AerisError::InternalInvariant(format!(
                    "locate_run({home}) returned {located:?}, expected Some(({expected_start}, {expected_end}))"
                )));
            }
            prev_end_plus_one = expected_end + 1;
        }

        for tail_slot in prev_end_plus_one..self.physical_slots() {
            if self.slot_kind(tail_slot) != KeepsakeSlotKind::PayloadInterior {
                return Err(AerisError::InternalInvariant(format!(
                    "stray metadata {:?} at empty tail slot {tail_slot}",
                    self.slot_kind(tail_slot)
                )));
            }
        }
        Ok(())
    }
}
