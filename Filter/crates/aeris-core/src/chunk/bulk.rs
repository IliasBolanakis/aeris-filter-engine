//! Sequential bulk-append helpers and stack-buffered block writer for [`QuotientTable`].

use super::{
    write_u64_le, QuotientTable, BLOCK_EXTENSION_POS, BLOCK_HEADER_BYTES, BLOCK_OCCUPIED_POS,
    BLOCK_RUNEND_POS,
};
use aeris_types::{KeepsakeSlotKind, SLOTS_PER_CHUNK_LOG2};

impl QuotientTable {
    /// Writes an entire quotient run sequentially during sorted bulk construction
    /// (`bulk_load` or `expand`), advancing `write_cursor` in `O(run.len())` time
    /// without any Robin Hood shifting.
    ///
    /// # Preconditions
    /// - Calls MUST supply strictly ascending `home` indices in `0..canonical_slots`.
    /// - `run_slots` MUST be non-empty and end with [`KeepsakeSlotKind::BoxAndRunEnd`].
    /// - `write_cursor` starts at `0` on an empty table and is updated in place.
    pub fn bulk_append_run(
        &mut self,
        home: usize,
        run_slots: &[(u64, KeepsakeSlotKind)],
        write_cursor: &mut usize,
    ) {
        debug_assert!(home < self.canonical_slots);
        debug_assert!(!run_slots.is_empty());
        debug_assert!(run_slots.last().is_some_and(|&(_, k)| k.is_run_end()));

        let start = home.max(*write_cursor);
        let end_exclusive = start + run_slots.len();
        self.ensure_physical_slot(end_exclusive);

        self.set_block_occupied_bit(home >> SLOTS_PER_CHUNK_LOG2, (home & 63) as u32);
        self.write_slots(start, run_slots);

        let first_chunk_after_home = (home >> SLOTS_PER_CHUNK_LOG2) + 1;
        let last_spilled_chunk = (end_exclusive - 1) >> SLOTS_PER_CHUNK_LOG2;
        self.update_spilled_chunk_offsets(
            first_chunk_after_home,
            last_spilled_chunk,
            end_exclusive,
        );

        *write_cursor = end_exclusive;
    }

    /// Creates a block-buffered [`BulkTableWriter`] for streaming sorted runs into
    /// an empty table in strictly ascending `home` order without per-slot DRAM
    /// read-modify-write overhead.
    pub fn bulk_writer(&mut self) -> BulkTableWriter<'_> {
        BulkTableWriter {
            table: self,
            write_cursor: 0,
            occ_chunk: 0,
            occ_word: 0,
            cur_chunk: 0,
            cur_runend: 0,
            cur_extension: 0,
            cur_payload_words: [0u64; 64],
            cur_dirty: false,
        }
    }

    #[inline]
    fn update_spilled_chunk_offsets(
        &mut self,
        first_chunk_after_home: usize,
        last_spilled_chunk: usize,
        end_exclusive: usize,
    ) {
        if first_chunk_after_home <= last_spilled_chunk {
            for c in first_chunk_after_home..=last_spilled_chunk {
                let base = c << SLOTS_PER_CHUNK_LOG2;
                self.set_chunk_offset(c, end_exclusive - base);
            }
        }
    }
}

/// Sequential block-buffered writer for sorted `bulk_load` construction.
///
/// Accumulates `occupied`, `runend`, `extension`, and packed payload `u64` words
/// in stack/register buffers and flushes each 64-slot AoS block sequentially once
/// the physical write cursor leaves it, eliminating per-slot read-modify-write
/// traffic on DRAM-resident tables with zero `unsafe` code.
pub struct BulkTableWriter<'a> {
    table: &'a mut QuotientTable,
    write_cursor: usize,
    occ_chunk: usize,
    occ_word: u64,
    cur_chunk: usize,
    cur_runend: u64,
    cur_extension: u64,
    cur_payload_words: [u64; 64],
    cur_dirty: bool,
}

impl BulkTableWriter<'_> {
    /// Appends a single-slot quotient run for `home` carrying `payload` and
    /// [`KeepsakeSlotKind::BoxAndRunEnd`].
    #[inline]
    pub fn append_single_slot_run(&mut self, home: usize, payload: u64) {
        let hc = self.mark_home_occupied(home);
        let start = home.max(self.write_cursor);
        self.push_slot(start, payload, KeepsakeSlotKind::BoxAndRunEnd);
        let end_exclusive = start + 1;
        self.table.update_spilled_chunk_offsets(
            hc + 1,
            start >> SLOTS_PER_CHUNK_LOG2,
            end_exclusive,
        );
        self.write_cursor = end_exclusive;
    }

    /// Appends a multi-slot quotient run for `home` in strictly ascending `home` order.
    #[inline]
    pub fn append_run(&mut self, home: usize, run_slots: &[(u64, KeepsakeSlotKind)]) {
        debug_assert!(!run_slots.is_empty());
        debug_assert!(run_slots.last().is_some_and(|&(_, k)| k.is_run_end()));

        let hc = self.mark_home_occupied(home);
        let start = home.max(self.write_cursor);
        for (i, &(payload, kind)) in run_slots.iter().enumerate() {
            self.push_slot(start + i, payload, kind);
        }
        let end_exclusive = start + run_slots.len();
        self.table.update_spilled_chunk_offsets(
            hc + 1,
            (end_exclusive - 1) >> SLOTS_PER_CHUNK_LOG2,
            end_exclusive,
        );
        self.write_cursor = end_exclusive;
    }

    /// Flushes all buffered block headers and payloads into the underlying [`QuotientTable`].
    pub fn finish(mut self) {
        self.flush_occ();
        self.flush_cur_block();
    }

    #[inline]
    fn mark_home_occupied(&mut self, home: usize) -> usize {
        debug_assert!(home < self.table.canonical_slots);
        let hc = home >> SLOTS_PER_CHUNK_LOG2;
        if hc != self.occ_chunk {
            self.flush_occ();
            self.occ_chunk = hc;
            self.occ_word = 0;
        }
        self.occ_word |= 1u64 << ((home & 63) as u32);
        hc
    }

    #[inline]
    fn push_slot(&mut self, slot: usize, payload: u64, kind: KeepsakeSlotKind) {
        let chunk = slot >> SLOTS_PER_CHUNK_LOG2;
        if chunk != self.cur_chunk {
            self.flush_cur_block();
            self.cur_chunk = chunk;
            self.cur_runend = 0;
            self.cur_extension = 0;
            let words = self.table.slot_bits as usize;
            self.cur_payload_words[..words].fill(0);
            self.table
                .ensure_physical_slot((chunk + 1) << SLOTS_PER_CHUNK_LOG2);
        }
        self.cur_dirty = true;
        let local = slot & 63;
        let (re, ext) = kind.to_bits();
        self.cur_runend |= u64::from(re) << (local as u32);
        self.cur_extension |= u64::from(ext) << (local as u32);

        let masked = payload & self.table.slot_mask;
        let slot_bits = self.table.slot_bits as usize;
        let bit_pos = local * slot_bits;
        let w_idx = bit_pos >> 6;
        let b_off = (bit_pos & 63) as u32;
        self.cur_payload_words[w_idx] |= masked << b_off;
        if b_off + (slot_bits as u32) > 64 {
            self.cur_payload_words[w_idx + 1] |= masked >> (64 - b_off);
        }
    }

    #[inline]
    fn flush_occ(&mut self) {
        if self.occ_word != 0 {
            let pos = self.table.block_base(self.occ_chunk) + BLOCK_OCCUPIED_POS;
            write_u64_le(&mut self.table.blocks, pos, self.occ_word);
            self.occ_word = 0;
        }
    }

    #[inline]
    fn flush_cur_block(&mut self) {
        if !self.cur_dirty {
            return;
        }
        let words = self.table.slot_bits as usize;
        let base = self.table.block_base(self.cur_chunk);
        let blk = &mut self.table.blocks[base..(base + BLOCK_HEADER_BYTES + words * 8)];
        write_u64_le(blk, BLOCK_RUNEND_POS, self.cur_runend);
        write_u64_le(blk, BLOCK_EXTENSION_POS, self.cur_extension);
        for (i, &w) in self.cur_payload_words[..words].iter().enumerate() {
            write_u64_le(blk, BLOCK_HEADER_BYTES + i * 8, w);
        }
        self.cur_dirty = false;
    }
}
