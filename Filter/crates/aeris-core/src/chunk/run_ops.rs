//! Quotient run insertion, expansion, replacement, and deletion primitives.

use super::{read_u64_le, write_u64_le, QuotientTable, BLOCK_OCCUPIED_POS, BLOCK_OFFSET_POS};
use crate::simd;
use aeris_types::{KeepsakeSlotKind, MAX_CHUNK_OFFSET, SLOTS_PER_CHUNK_LOG2};

impl QuotientTable {
    /// Inserts a single-slot brand-new quotient run for unoccupied canonical
    /// `home` carrying `payload` and terminal marker [`KeepsakeSlotKind::BoxAndRunEnd`].
    ///
    /// Never performs backward chunk scans: when `occ_count == 0 && offset == 0`
    /// in `home`'s block, `home` is guaranteed empty and written in-place within
    /// the single already-loaded AoS block.
    ///
    /// # Preconditions
    /// - `home < self.canonical_slots()` and `!self.is_occupied(home)`.
    ///
    /// # Complexity
    /// `O(1)` expected time, zero heap allocations.
    #[inline]
    pub fn insert_single_slot_new_run(&mut self, home: usize, payload: u64) {
        debug_assert!(home < self.canonical_slots);
        debug_assert!(!self.is_occupied(home));
        let chunk_idx = home >> SLOTS_PER_CHUNK_LOG2;
        let local = (home & 63) as u32;
        let base_slot = chunk_idx << SLOTS_PER_CHUNK_LOG2;
        let block_base = self.block_base(chunk_idx);
        let occ = read_u64_le(&self.blocks, block_base + BLOCK_OCCUPIED_POS);
        let raw_offset = self.blocks[block_base + BLOCK_OFFSET_POS];
        let offset = if raw_offset < MAX_CHUNK_OFFSET {
            raw_offset as usize
        } else {
            self.offset_overflow
                .get(&chunk_idx)
                .copied()
                .unwrap_or(MAX_CHUNK_OFFSET as u32) as usize
        };
        let occ_count = simd::rank64(occ, local) as usize;

        // Ultra-fast path: zero spillover and no earlier occupied slots in this block ->
        // `home` itself is guaranteed empty; mutate solely within `block_base`.
        if occ_count == 0 && offset == 0 {
            write_u64_le(
                &mut self.blocks,
                block_base + BLOCK_OCCUPIED_POS,
                occ | (1u64 << local),
            );
            self.write_payload_at_base(block_base, local as usize, payload);
            self.write_kind_at_base(block_base, local, KeepsakeSlotKind::BoxAndRunEnd);
            return;
        }

        let start_pos = if occ_count > 0 {
            home.max(self.select_qrun_end(base_slot + offset, occ_count - 1) + 1)
        } else {
            home.max(base_slot + offset)
        };
        let empty = self.find_first_empty_slot(start_pos);
        self.ensure_physical_slot(empty + 1);
        if empty > start_pos {
            self.shift_right_one(start_pos, empty);
        }
        if (home >> SLOTS_PER_CHUNK_LOG2) < (empty >> SLOTS_PER_CHUNK_LOG2) {
            self.increment_offsets(home, empty);
        }
        let occ_pos = self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS;
        let cur_occ = read_u64_le(&self.blocks, occ_pos);
        write_u64_le(&mut self.blocks, occ_pos, cur_occ | (1u64 << local));
        let dst_base = self.block_base(start_pos >> SLOTS_PER_CHUNK_LOG2);
        let dst_local = start_pos & 63;
        self.write_payload_at_base(dst_base, dst_local, payload);
        self.write_kind_at_base(dst_base, dst_local as u32, KeepsakeSlotKind::BoxAndRunEnd);
    }

    /// Inserts a single-slot keepsake box carrying `payload` at `insert_pos`
    /// inside an already-occupied `home`'s quotient run ending at `run_end`,
    /// avoiding a redundant `locate_run(home)` call.
    ///
    /// # Preconditions
    /// - `self.is_occupied(home)`.
    /// - `run_start <= insert_pos <= run_end + 1`, where `run_end` is the current
    ///   inclusive end of `home`'s quotient run.
    ///
    /// # Complexity
    /// `O(1)` expected time, zero heap allocations.
    #[inline]
    pub fn insert_single_slot_in_existing_run(
        &mut self,
        home: usize,
        insert_pos: usize,
        run_end: usize,
        payload: u64,
    ) {
        debug_assert!(home < self.canonical_slots);
        debug_assert!(self.is_occupied(home));
        debug_assert!(insert_pos <= run_end + 1);

        let empty = self.find_first_empty_slot(insert_pos);
        self.ensure_physical_slot(empty + 1);
        if empty > insert_pos {
            self.shift_right_one(insert_pos, empty);
        }
        if (home >> SLOTS_PER_CHUNK_LOG2) < (empty >> SLOTS_PER_CHUNK_LOG2) {
            self.increment_offsets(home, empty);
        }

        let base = self.block_base(insert_pos >> SLOTS_PER_CHUNK_LOG2);
        let local = insert_pos & 63;
        self.write_payload_at_base(base, local, payload);
        if insert_pos <= run_end {
            self.write_kind_at_base(base, local as u32, KeepsakeSlotKind::BoxEnd);
        } else {
            self.set_slot_kind(run_end, KeepsakeSlotKind::BoxEnd);
            self.write_kind_at_base(base, local as u32, KeepsakeSlotKind::BoxAndRunEnd);
        }
    }

    /// Expands a single-slot unextended keepsake box at `box_start` inside
    /// `home`'s quotient run into a two-slot box `(slot0, PayloadInterior)`
    /// followed by `(slot1, terminal_kind)`, without re-locating `home`'s run.
    ///
    /// # Preconditions
    /// - `self.is_occupied(home)` and `box_start` currently holds a 1-slot box
    ///   whose [`KeepsakeSlotKind`] is `terminal_kind` (`BoxEnd` or `BoxAndRunEnd`).
    ///
    /// # Complexity
    /// `O(1)` expected time, zero heap allocations.
    #[inline]
    pub fn expand_single_slot_to_two(
        &mut self,
        home: usize,
        box_start: usize,
        slot0: u64,
        slot1: u64,
        terminal_kind: KeepsakeSlotKind,
    ) {
        debug_assert!(home < self.canonical_slots);
        debug_assert!(self.is_occupied(home));
        debug_assert!(terminal_kind.is_box_end());

        self.insert_blank_slot_interior(home, box_start);

        let base0 = self.block_base(box_start >> SLOTS_PER_CHUNK_LOG2);
        let local0 = box_start & 63;
        self.write_payload_at_base(base0, local0, slot0);
        self.write_kind_at_base(base0, local0 as u32, KeepsakeSlotKind::PayloadInterior);

        let slot1_idx = box_start + 1;
        let base1 = self.block_base(slot1_idx >> SLOTS_PER_CHUNK_LOG2);
        let local1 = slot1_idx & 63;
        self.write_payload_at_base(base1, local1, slot1);
        self.write_kind_at_base(base1, local1 as u32, terminal_kind);
    }

    /// Replaces the physical slot region `[region_start .. region_start + old_len)`
    /// inside `home`'s quotient run with `new_slots`.
    ///
    /// Handles all lifecycle transitions atomically while preserving every
    /// Quotient Filter metadata and chunk-offset invariant:
    /// - Creating a brand-new run (`!is_occupied(home)`, `old_len == 0`, `new_slots.len() >= 1`).
    /// - Growing or shrinking a region inside an existing run (including appending
    ///   at `run_end + 1` or truncating the final box).
    /// - Deleting an entire run (`new_slots.is_empty()` and `old_len == run_len`).
    ///
    /// # Preconditions
    /// - Caller ensures `new_slots` carries accurate [`KeepsakeSlotKind`] markers
    ///   consistent with its position in `home`'s run (in particular, the final slot
    ///   of `home`'s run after the edit must be [`KeepsakeSlotKind::BoxAndRunEnd`]).
    /// - If `!is_occupied(home)`, `old_len` must be `0` and `new_slots` must be non-empty.
    pub fn replace_region(
        &mut self,
        home: usize,
        region_start: usize,
        old_len: usize,
        new_slots: &[(u64, KeepsakeSlotKind)],
    ) {
        debug_assert!(home < self.canonical_slots);
        let new_len = new_slots.len();

        if !self.is_occupied(home) {
            debug_assert_eq!(old_len, 0);
            debug_assert!(new_len > 0);
            let start_pos = match self.end_of_runs_up_to(home) {
                Some(prev_end) => home.max(prev_end + 1),
                None => home,
            };
            self.insert_first_slot_of_new_run(home, start_pos);
            for _ in 1..new_len {
                self.insert_blank_slot_interior(home, start_pos);
            }
            self.write_slots(start_pos, new_slots);
            return;
        }

        let (run_start, run_end) = self
            .locate_run(home)
            .expect("home is occupied so run must exist");
        self.replace_region_known_run(home, run_start, run_end, region_start, old_len, new_slots);
    }

    /// Replaces the physical slot region `[region_start .. region_start + old_len)`
    /// inside an already-located occupied `home`'s quotient run `[run_start ..= run_end]`
    /// with `new_slots`, avoiding a redundant [`Self::locate_run`] lookup.
    ///
    /// # Preconditions
    /// - `self.is_occupied(home)` and `(run_start, run_end) == self.locate_run(home).unwrap()`.
    /// - `run_start <= region_start <= run_end + 1` and `region_start + old_len <= run_end + 1`.
    pub fn replace_region_known_run(
        &mut self,
        home: usize,
        run_start: usize,
        run_end: usize,
        region_start: usize,
        old_len: usize,
        new_slots: &[(u64, KeepsakeSlotKind)],
    ) {
        debug_assert!(home < self.canonical_slots);
        debug_assert!(self.is_occupied(home));
        debug_assert!(region_start >= run_start && region_start <= run_end + 1);
        debug_assert!(region_start + old_len <= run_end + 1);
        let new_len = new_slots.len();

        if new_len == old_len {
            self.write_slots(region_start, new_slots);
        } else if new_len > old_len {
            self.grow_run_region(home, run_end, region_start, new_len - old_len, new_slots);
        } else {
            self.shrink_run_region(home, run_start, run_end, region_start, old_len, new_slots);
        }
    }

    fn grow_run_region(
        &mut self,
        home: usize,
        mut run_end: usize,
        region_start: usize,
        delta: usize,
        new_slots: &[(u64, KeepsakeSlotKind)],
    ) {
        if region_start <= run_end {
            for _ in 0..delta {
                self.insert_blank_slot_interior(home, region_start);
            }
        } else {
            for _ in 0..delta {
                self.append_blank_slot_after_run(home, run_end);
                run_end += 1;
            }
        }
        self.write_slots(region_start, new_slots);
    }

    fn shrink_run_region(
        &mut self,
        home: usize,
        run_start: usize,
        mut run_end: usize,
        region_start: usize,
        old_len: usize,
        new_slots: &[(u64, KeepsakeSlotKind)],
    ) {
        let new_len = new_slots.len();
        let delta = old_len - new_len;
        let run_len = run_end - run_start + 1;

        if new_len == 0 && old_len == run_len {
            for _ in 0..(run_len - 1) {
                self.remove_slot_interior(home, run_start, run_end);
                run_end -= 1;
            }
            self.remove_sole_slot_of_run(home, run_start);
            return;
        }

        let removes_tail_of_run = region_start + old_len - 1 == run_end && new_len == 0;
        if removes_tail_of_run {
            // Remove interior slots first so `run_end` keeps `BoxAndRunEnd`, then
            // remove the terminal slot and promote `region_start - 1`.
            for _ in 0..(delta - 1) {
                self.remove_slot_interior(home, region_start, run_end);
                run_end -= 1;
            }
            self.remove_terminal_slot_of_multi_slot_run(home, run_end);
        } else {
            for _ in 0..delta {
                self.remove_slot_interior(home, region_start, run_end);
                run_end -= 1;
            }
            self.write_slots(region_start, new_slots);
        }
    }

    /// Inserts the sole initial slot of a brand-new run for unoccupied `home` at `pos`.
    fn insert_first_slot_of_new_run(&mut self, home: usize, pos: usize) {
        let empty = self.find_first_empty_slot(pos);
        self.ensure_physical_slot(empty + 1);
        self.shift_right_one(pos, empty);
        self.increment_offsets(home, empty);
        self.set_block_occupied_bit(home >> SLOTS_PER_CHUNK_LOG2, (home & 63) as u32);
        self.set_slot_kind(pos, KeepsakeSlotKind::BoxAndRunEnd);
    }

    /// Inserts 1 blank slot at `pos <= run_end` inside an existing run for `home`.
    fn insert_blank_slot_interior(&mut self, home: usize, pos: usize) {
        let empty = self.find_first_empty_slot(pos);
        self.ensure_physical_slot(empty + 1);
        self.shift_right_one(pos, empty);
        self.increment_offsets(home, empty);
    }

    /// Appends 1 blank slot immediately after `run_end` of an existing run for `home`,
    /// transferring the `BoxAndRunEnd` marker from `run_end` to `run_end + 1`.
    fn append_blank_slot_after_run(&mut self, home: usize, run_end: usize) {
        let pos = run_end + 1;
        let empty = self.find_first_empty_slot(pos);
        self.ensure_physical_slot(empty + 1);
        self.shift_right_one(pos, empty);
        self.increment_offsets(home, empty);
        self.set_slot_kind(run_end, KeepsakeSlotKind::BoxEnd);
        self.set_slot_kind(pos, KeepsakeSlotKind::BoxAndRunEnd);
    }

    /// Deletes 1 slot at `pos < run_end` inside an existing run for `home`.
    fn remove_slot_interior(&mut self, home: usize, pos: usize, run_end: usize) {
        debug_assert!(pos < run_end);
        let vacated = self.find_vacated_slot(run_end);
        self.shift_left_one(pos, vacated);
        self.decrement_offsets(home, vacated);
    }

    /// Deletes the terminal slot `run_end` of a multi-slot run (`run_end > run_start`)
    /// and promotes `run_end - 1` to `BoxAndRunEnd`.
    fn remove_terminal_slot_of_multi_slot_run(&mut self, home: usize, run_end: usize) {
        let vacated = self.find_vacated_slot(run_end);
        self.shift_left_one(run_end, vacated);
        self.decrement_offsets(home, vacated);
        self.set_slot_kind(run_end - 1, KeepsakeSlotKind::BoxAndRunEnd);
    }

    /// Deletes the sole remaining slot `pos` of `home`'s run and clears `occupied[home]`.
    fn remove_sole_slot_of_run(&mut self, home: usize, pos: usize) {
        let vacated = self.find_vacated_slot(pos);
        self.shift_left_one(pos, vacated);
        self.decrement_offsets(home, vacated);
        self.clear_block_occupied_bit(home >> SLOTS_PER_CHUNK_LOG2, (home & 63) as u32);
    }

    /// Increments chunk offsets for all chunks `c` with `home < 64 * c <= empty`.
    #[inline]
    fn increment_offsets(&mut self, home: usize, empty: usize) {
        let first_c = (home >> SLOTS_PER_CHUNK_LOG2) + 1;
        let last_c = empty >> SLOTS_PER_CHUNK_LOG2;
        for c in first_c..=last_c {
            let cur = self.chunk_offset(c);
            self.set_chunk_offset(c, cur + 1);
        }
    }

    /// Decrements chunk offsets for all chunks `c` with `home < 64 * c <= vacated`.
    #[inline]
    fn decrement_offsets(&mut self, home: usize, vacated: usize) {
        let first_c = (home >> SLOTS_PER_CHUNK_LOG2) + 1;
        let last_c = vacated >> SLOTS_PER_CHUNK_LOG2;
        for c in first_c..=last_c {
            let cur = self.chunk_offset(c);
            debug_assert!(cur > 0, "chunk {c} offset underflow during shift-left");
            self.set_chunk_offset(c, cur - 1);
        }
    }
}
