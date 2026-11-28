//! SIMD rank/select run navigation and cluster boundary search for [`QuotientTable`].

use super::{
    read_u64_le, QuotientTable, BLOCK_EXTENSION_POS, BLOCK_OCCUPIED_POS, BLOCK_OFFSET_POS,
    BLOCK_RUNEND_POS,
};
use crate::simd;
use aeris_types::{MAX_CHUNK_OFFSET, SLOTS_PER_CHUNK_LOG2};

impl QuotientTable {
    /// Locates the inclusive physical slot range `(run_start, run_end)` of
    /// canonical `home`'s quotient run, or `None` if `home` is not occupied.
    ///
    /// # Complexity
    /// `O(1)` expected time via SIMD `POPCNT` + `PDEP` rank/select; zero heap allocations.
    #[inline]
    pub fn locate_run(&self, home: usize) -> Option<(usize, usize)> {
        debug_assert!(home < self.canonical_slots);
        let chunk_idx = home >> SLOTS_PER_CHUNK_LOG2;
        let local = (home & 63) as u32;
        let block_base = self.block_base(chunk_idx);
        let occupied = read_u64_le(&self.blocks, block_base + BLOCK_OCCUPIED_POS);
        if ((occupied >> local) & 1) == 0 {
            return None;
        }

        let base_slot = chunk_idx << SLOTS_PER_CHUNK_LOG2;
        let raw_offset = self.blocks[block_base + BLOCK_OFFSET_POS];
        let offset = if raw_offset < MAX_CHUNK_OFFSET {
            raw_offset as usize
        } else {
            self.offset_overflow
                .get(&chunk_idx)
                .copied()
                .unwrap_or(MAX_CHUNK_OFFSET as u32) as usize
        };
        let first_slot = base_slot + offset;
        let rank = simd::rank64(occupied, local) as usize; // >= 1

        if rank == 1 {
            let run_end = self.select_qrun_end(first_slot, 0);
            Some((home.max(first_slot), run_end))
        } else {
            let (prev_end, run_end) = self.select_adjacent_qrun_ends(first_slot, rank - 2);
            Some((home.max(prev_end + 1), run_end))
        }
    }

    /// Finds the next slot index `>= slot` whose `runend` bit is `1` (ending the
    /// keepsake box containing `slot`).
    ///
    /// # Complexity
    /// `O(1)` expected time via `trailing_zeros` on block `runend` bitmaps.
    #[inline]
    pub fn find_box_end(&self, slot: usize) -> usize {
        let mut chunk_idx = slot >> SLOTS_PER_CHUNK_LOG2;
        let bit_off = (slot & 63) as u32;
        let masked = self.block_runend(chunk_idx) & (u64::MAX << bit_off);
        if masked != 0 {
            return (chunk_idx << SLOTS_PER_CHUNK_LOG2) + (masked.trailing_zeros() as usize);
        }
        chunk_idx += 1;
        while chunk_idx < self.total_chunks {
            let w = self.block_runend(chunk_idx);
            if w != 0 {
                return (chunk_idx << SLOTS_PER_CHUNK_LOG2) + (w.trailing_zeros() as usize);
            }
            chunk_idx += 1;
        }
        unreachable!("every keepsake box must terminate with runend=1")
    }

    /// Returns the physical slot index of the final `qrun_end` belonging to
    /// runs with `home_slot <= slot`, or `None` if no occupied slot exists in `0..=slot`.
    pub fn end_of_runs_up_to(&self, slot: usize) -> Option<usize> {
        let chunk_idx = slot >> SLOTS_PER_CHUNK_LOG2;
        if chunk_idx >= self.total_chunks {
            return self.last_qrun_end_overall();
        }
        let local = (slot & 63) as u32;
        let base_slot = chunk_idx << SLOTS_PER_CHUNK_LOG2;
        let offset = self.chunk_offset(chunk_idx);
        let occ_count = simd::rank64(self.block_occupied(chunk_idx), local) as usize;

        if occ_count > 0 {
            return Some(self.select_qrun_end(base_slot + offset, occ_count - 1));
        }
        if offset > 0 {
            return Some(base_slot + offset - 1);
        }
        // No occupied bits in `base_slot..=slot` and no runs from earlier chunks reach `base_slot`.
        // Scan backward for the most recent chunk with any `qrun_end` bit.
        let mut prev_chunk = chunk_idx;
        while prev_chunk > 0 {
            prev_chunk -= 1;
            let qrun = self.block_qrun_end_mask(prev_chunk);
            if qrun != 0 {
                let hi_bit = 63 - qrun.leading_zeros() as usize;
                return Some((prev_chunk << SLOTS_PER_CHUNK_LOG2) + hi_bit);
            }
        }
        None
    }

    #[inline]
    pub(super) fn select_qrun_end(&self, start_slot: usize, mut rank: usize) -> usize {
        let mut chunk_idx = start_slot >> SLOTS_PER_CHUNK_LOG2;
        let first_bit = (start_slot & 63) as u32;
        let first_mask = self.block_qrun_end_mask(chunk_idx) & (u64::MAX << first_bit);
        let first_cnt = simd::popcnt64(first_mask) as usize;
        if rank < first_cnt {
            return (chunk_idx << SLOTS_PER_CHUNK_LOG2)
                + (simd::select64(first_mask, rank as u32) as usize);
        }
        rank -= first_cnt;
        chunk_idx += 1;

        while chunk_idx < self.total_chunks {
            let qrun = self.block_qrun_end_mask(chunk_idx);
            let cnt = simd::popcnt64(qrun) as usize;
            if rank < cnt {
                return (chunk_idx << SLOTS_PER_CHUNK_LOG2)
                    + (simd::select64(qrun, rank as u32) as usize);
            }
            rank -= cnt;
            chunk_idx += 1;
        }
        unreachable!("occupied/qrun_end bijection guarantees selected qrun_end exists")
    }

    /// Selects both the `rank`-th and `(rank + 1)`-th `qrun_end` bits at or after
    /// `start_slot` in a single chunk pass.
    #[inline]
    fn select_adjacent_qrun_ends(&self, start_slot: usize, mut rank: usize) -> (usize, usize) {
        let mut chunk_idx = start_slot >> SLOTS_PER_CHUNK_LOG2;
        let first_bit = (start_slot & 63) as u32;
        let mut qrun = self.block_qrun_end_mask(chunk_idx) & (u64::MAX << first_bit);

        loop {
            let cnt = simd::popcnt64(qrun) as usize;
            if rank < cnt {
                let first_local = simd::select64(qrun, rank as u32);
                let first_global = (chunk_idx << SLOTS_PER_CHUNK_LOG2) + (first_local as usize);
                if first_local < 63 {
                    let rest = qrun & (u64::MAX << (first_local + 1));
                    if rest != 0 {
                        let second_global =
                            (chunk_idx << SLOTS_PER_CHUNK_LOG2) + (rest.trailing_zeros() as usize);
                        return (first_global, second_global);
                    }
                }
                chunk_idx += 1;
                while chunk_idx < self.total_chunks {
                    let next_qrun = self.block_qrun_end_mask(chunk_idx);
                    if next_qrun != 0 {
                        let second_global = (chunk_idx << SLOTS_PER_CHUNK_LOG2)
                            + (next_qrun.trailing_zeros() as usize);
                        return (first_global, second_global);
                    }
                    chunk_idx += 1;
                }
                unreachable!("occupied/qrun_end bijection guarantees adjacent qrun_end exists");
            }
            rank -= cnt;
            chunk_idx += 1;
            qrun = self.block_qrun_end_mask(chunk_idx);
        }
    }

    fn last_qrun_end_overall(&self) -> Option<usize> {
        for c in (0..self.total_chunks).rev() {
            let qrun = self.block_qrun_end_mask(c);
            if qrun != 0 {
                let hi_bit = 63 - qrun.leading_zeros() as usize;
                return Some((c << SLOTS_PER_CHUNK_LOG2) + hi_bit);
            }
        }
        None
    }

    /// Counts set `occupied` bits in the inclusive canonical slot range `[lo ..= hi]`.
    #[inline]
    fn count_occupied_inclusive(&self, lo: usize, hi: usize) -> usize {
        if lo > hi || lo >= self.canonical_slots {
            return 0;
        }
        let hi = hi.min(self.canonical_slots - 1);
        let c_lo = lo >> SLOTS_PER_CHUNK_LOG2;
        let c_hi = hi >> SLOTS_PER_CHUNK_LOG2;
        let b_lo = (lo & 63) as u32;
        let b_hi = (hi & 63) as u32;
        if c_lo == c_hi {
            let mask = (u64::MAX << b_lo) & (u64::MAX >> (63 - b_hi));
            return simd::popcnt64(self.block_occupied(c_lo) & mask) as usize;
        }
        let mut total = simd::popcnt64(self.block_occupied(c_lo) & (u64::MAX << b_lo)) as usize;
        for c in (c_lo + 1)..c_hi {
            total += simd::popcnt64(self.block_occupied(c)) as usize;
        }
        total += simd::popcnt64(self.block_occupied(c_hi) & (u64::MAX >> (63 - b_hi))) as usize;
        total
    }

    /// Finds the first empty physical slot `>= start`, dynamically extending
    /// tail overflow chunks if a cluster reaches the physical end of the table.
    ///
    /// Uses a strictly forward monotonic scan over `(occupied, qrun_end)` bitmaps
    /// (`#occupied(0..=t) == #qrun_end(0..t)` iff slot `t` is empty) so no slot is
    /// ever rescanned from the chunk base during high-load-factor clusters.
    pub(super) fn find_first_empty_slot(&mut self, start: usize) -> usize {
        let mut pos = start;
        loop {
            if pos >= self.physical_slots() {
                self.ensure_physical_slot(pos + 1);
                return pos;
            }
            let c = pos >> SLOTS_PER_CHUNK_LOG2;
            let base = c << SLOTS_PER_CHUNK_LOG2;
            let offset = self.chunk_offset(c);
            if pos < base + offset {
                // Slots `[pos .. base + offset)` are occupied by runs originating in `< base`.
                pos = base + offset;
                continue;
            }

            let pending = self.pending_runs_at(c, pos, offset);
            if pending == 0 {
                return pos;
            }
            return self.jump_over_cluster(pos, pending);
        }
    }

    #[inline]
    fn pending_runs_at(&self, chunk_idx: usize, pos: usize, offset: usize) -> usize {
        let local = (pos & 63) as u32;
        let block_base = self.block_base(chunk_idx);
        let occ_w = read_u64_le(&self.blocks, block_base + BLOCK_OCCUPIED_POS);
        let occ_cnt = simd::popcnt64(occ_w & (u64::MAX >> (63 - local))) as usize;
        let ended_cnt = if local == (offset as u32) {
            0usize
        } else {
            let qrun_w = read_u64_le(&self.blocks, block_base + BLOCK_RUNEND_POS)
                & read_u64_le(&self.blocks, block_base + BLOCK_EXTENSION_POS);
            let mask = (u64::MAX << (offset as u32)) & ((1u64 << local) - 1);
            simd::popcnt64(qrun_w & mask) as usize
        };
        debug_assert!(occ_cnt >= ended_cnt);
        occ_cnt - ended_cnt
    }

    #[inline]
    fn jump_over_cluster(&mut self, mut pos: usize, mut pending: usize) -> usize {
        // Strictly forward cluster jump: `pending` runs with `home <= pos` remain
        // unterminated at `>= pos`, so the `pending`-th `qrun_end` at `>= pos`
        // accounts for all runs with `home <= pos` and zero runs with `home > pos`.
        loop {
            let e = self.select_qrun_end(pos, pending - 1);
            let next_pos = e + 1;
            let new_occ = self.count_occupied_inclusive(pos + 1, next_pos);
            if new_occ == 0 {
                if next_pos >= self.physical_slots() {
                    self.ensure_physical_slot(next_pos + 1);
                }
                return next_pos;
            }
            pending = new_occ;
            pos = next_pos;
        }
    }

    /// Computes the trailing slot that becomes vacated when deleting 1 slot from a
    /// run currently ending at `run_end`.
    pub(super) fn find_vacated_slot(&self, run_end: usize) -> usize {
        let mut cur_end = run_end;
        loop {
            match self.end_of_runs_up_to(cur_end) {
                Some(e) if e > cur_end => cur_end = e,
                _ => return cur_end,
            }
        }
    }
}
