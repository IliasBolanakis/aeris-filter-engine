//! Word-level AoS block payload and metadata 1-slot shift primitives.

use super::{
    read_u64_le, write_u64_le, QuotientTable, BLOCK_EXTENSION_POS, BLOCK_HEADER_BYTES,
    BLOCK_RUNEND_POS,
};
use aeris_types::SLOTS_PER_CHUNK_LOG2;

impl QuotientTable {
    /// Shifts payloads and `(runend, extension)` metadata for `[start .. end)`
    /// right by 1 slot into `[start + 1 .. end + 1)` and clears slot `start`.
    pub(super) fn shift_right_one(&mut self, start: usize, end: usize) {
        self.shift_payloads_right_one(start, end);
        shift_metadata_right_one(&mut self.blocks, self.block_bytes, start, end);
    }

    /// Shifts payloads and `(runend, extension)` metadata for `(start ..= end]`
    /// left by 1 slot into `[start .. end)` and clears slot `end`.
    pub(super) fn shift_left_one(&mut self, start: usize, end: usize) {
        self.shift_payloads_left_one(start, end);
        shift_metadata_left_one(&mut self.blocks, self.block_bytes, start, end);
    }

    /// Shifts all 64 packed payloads of block `chunk_idx` right by 1 slot in
    /// word-level `u64` steps, inserting `carry` into slot `0`.
    #[inline]
    fn shift_full_block_payload_right(&mut self, chunk_idx: usize, carry: u64) {
        let payload_base = self.block_base(chunk_idx) + BLOCK_HEADER_BYTES;
        let words = self.slot_bits as usize;
        if self.slot_bits == 64 {
            self.blocks
                .copy_within(payload_base..(payload_base + 63 * 8), payload_base + 8);
            write_u64_le(&mut self.blocks, payload_base, carry);
            return;
        }
        let s = self.slot_bits as u32;
        let inv_s = 64 - s;
        for i in (1..words).rev() {
            let cur = read_u64_le(&self.blocks, payload_base + i * 8);
            let prev = read_u64_le(&self.blocks, payload_base + (i - 1) * 8);
            write_u64_le(
                &mut self.blocks,
                payload_base + i * 8,
                (cur << s) | (prev >> inv_s),
            );
        }
        let w0 = read_u64_le(&self.blocks, payload_base);
        write_u64_le(
            &mut self.blocks,
            payload_base,
            (w0 << s) | (carry & self.slot_mask),
        );
    }

    /// Shifts all 64 packed payloads of block `chunk_idx` left by 1 slot in
    /// word-level `u64` steps, inserting `carry` into slot `63`.
    #[inline]
    fn shift_full_block_payload_left(&mut self, chunk_idx: usize, carry: u64) {
        let payload_base = self.block_base(chunk_idx) + BLOCK_HEADER_BYTES;
        let words = self.slot_bits as usize;
        if self.slot_bits == 64 {
            self.blocks
                .copy_within((payload_base + 8)..(payload_base + 64 * 8), payload_base);
            write_u64_le(&mut self.blocks, payload_base + 63 * 8, carry);
            return;
        }
        let s = self.slot_bits as u32;
        let inv_s = 64 - s;
        for i in 0..(words - 1) {
            let cur = read_u64_le(&self.blocks, payload_base + i * 8);
            let next = read_u64_le(&self.blocks, payload_base + (i + 1) * 8);
            write_u64_le(
                &mut self.blocks,
                payload_base + i * 8,
                (cur >> s) | (next << inv_s),
            );
        }
        let last_off = payload_base + (words - 1) * 8;
        let last = read_u64_le(&self.blocks, last_off);
        write_u64_le(
            &mut self.blocks,
            last_off,
            (last >> s) | ((carry & self.slot_mask) << inv_s),
        );
    }

    /// Shifts bit-packed slot payloads inside block `chunk_idx` for `[lo .. hi)`
    /// right by 1 slot into `[lo + 1 ..= hi]` and writes `carry` into slot `lo`.
    #[inline]
    fn shift_partial_block_payload_right(
        &mut self,
        chunk_idx: usize,
        lo: usize,
        hi: usize,
        carry: u64,
    ) {
        debug_assert!(lo <= hi && hi < 64);
        let payload_base = self.block_base(chunk_idx) + BLOCK_HEADER_BYTES;
        if lo == hi {
            self.write_payload_at_base(self.block_base(chunk_idx), lo, carry);
            return;
        }
        if lo == 0 && hi == 63 {
            self.shift_full_block_payload_right(chunk_idx, carry);
            return;
        }
        let s = self.slot_bits as usize;
        if s == 64 {
            let p = &mut self.blocks[payload_base..(payload_base + 64 * 8)];
            p.copy_within((lo * 8)..(hi * 8), (lo + 1) * 8);
            write_u64_le(p, lo * 8, carry);
            return;
        }

        let start_bit = lo * s;
        let end_bit = (hi + 1) * s;
        let span_bits = end_bit - start_bit;
        let bit_in_byte = (start_bit & 7) as u32;
        let masked_carry = carry & self.slot_mask;

        if bit_in_byte + (span_bits as u32) <= 64 {
            shift_single_word_right(
                &mut self.blocks,
                payload_base + (start_bit >> 3),
                bit_in_byte,
                span_bits,
                (hi - lo) * s,
                s as u32,
                masked_carry,
            );
        } else {
            let p = &mut self.blocks[payload_base..(payload_base + s * 8)];
            shift_multi_word_right(p, start_bit, end_bit, s as u32, masked_carry);
        }
    }

    #[inline]
    fn shift_partial_block_payload_left(&mut self, chunk_idx: usize, lo: usize, hi: usize) {
        let base = self.block_base(chunk_idx);
        let s = self.slot_bits as usize;
        let start_bit = lo * s;
        let span_bits = (hi - lo + 1) * s;
        let bit_in_byte = (start_bit & 7) as u32;
        if bit_in_byte + (span_bits as u32) <= 64 {
            let byte_off = base + BLOCK_HEADER_BYTES + (start_bit >> 3);
            let word = read_u64_le(&self.blocks, byte_off);
            let span_mask = if span_bits == 64 {
                u64::MAX
            } else {
                ((1u64 << span_bits) - 1) << bit_in_byte
            };
            let move_bits = (hi - lo) * s;
            let move_mask = ((1u64 << move_bits) - 1) << (bit_in_byte + (s as u32));
            let updated = (word & !span_mask) | ((word & move_mask) >> (s as u32));
            write_u64_le(&mut self.blocks, byte_off, updated);
            return;
        }
        for local in lo..hi {
            let v = self.read_payload_at_base(base, local + 1);
            self.write_payload_at_base(base, local, v);
        }
        self.write_payload_at_base(base, hi, 0);
    }

    /// Shifts bit-packed slot payloads for `[start .. end)` right by 1 slot into
    /// `[start + 1 .. end + 1)` and zeroes slot `start`.
    pub(super) fn shift_payloads_right_one(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end && end < self.physical_slots());
        if start == end {
            self.set_payload(start, 0);
            return;
        }
        let c_start = start >> SLOTS_PER_CHUNK_LOG2;
        let c_end = end >> SLOTS_PER_CHUNK_LOG2;

        if c_start == c_end {
            self.shift_partial_block_payload_right(c_start, start & 63, end & 63, 0);
            return;
        }

        let end_carry = self.read_payload_at_base(self.block_base(c_end - 1), 63);
        self.shift_partial_block_payload_right(c_end, 0, end & 63, end_carry);

        for c in ((c_start + 1)..c_end).rev() {
            let carry = self.read_payload_at_base(self.block_base(c - 1), 63);
            self.shift_full_block_payload_right(c, carry);
        }

        self.shift_partial_block_payload_right(c_start, start & 63, 63, 0);
    }

    /// Shifts bit-packed slot payloads for `(start ..= end]` left by 1 slot into
    /// `[start .. end)` and zeroes slot `end`.
    pub(super) fn shift_payloads_left_one(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end && end < self.physical_slots());
        if start == end {
            self.set_payload(end, 0);
            return;
        }
        let c_start = start >> SLOTS_PER_CHUNK_LOG2;
        let c_end = end >> SLOTS_PER_CHUNK_LOG2;

        if c_start == c_end {
            self.shift_partial_block_payload_left(c_start, start & 63, end & 63);
            return;
        }

        let start_base = self.block_base(c_start);
        for local in (start & 63)..63 {
            let v = self.read_payload_at_base(start_base, local + 1);
            self.write_payload_at_base(start_base, local, v);
        }
        let start_carry = self.read_payload_at_base(self.block_base(c_start + 1), 0);
        self.write_payload_at_base(start_base, 63, start_carry);

        for c in (c_start + 1)..c_end {
            let carry = self.read_payload_at_base(self.block_base(c + 1), 0);
            self.shift_full_block_payload_left(c, carry);
        }

        let b_end = end & 63;
        let end_base = self.block_base(c_end);
        for local in 0..b_end {
            let v = self.read_payload_at_base(end_base, local + 1);
            self.write_payload_at_base(end_base, local, v);
        }
        self.write_payload_at_base(end_base, b_end, 0);
    }
}

#[inline(always)]
fn shift_single_word_right(
    blocks: &mut [u8],
    byte_off: usize,
    bit_in_byte: u32,
    span_bits: usize,
    move_bits: usize,
    s_u32: u32,
    masked_carry: u64,
) {
    let word = read_u64_le(blocks, byte_off);
    let span_mask = if span_bits == 64 {
        u64::MAX
    } else {
        ((1u64 << span_bits) - 1) << bit_in_byte
    };
    let move_mask = ((1u64 << move_bits) - 1) << bit_in_byte;
    let updated =
        (word & !span_mask) | ((word & move_mask) << s_u32) | (masked_carry << bit_in_byte);
    write_u64_le(blocks, byte_off, updated);
}

#[inline(always)]
fn shift_multi_word_right(
    p: &mut [u8],
    start_bit: usize,
    end_bit: usize,
    s_u32: u32,
    masked_carry: u64,
) {
    let w_start = start_bit >> 6;
    let b_start = (start_bit & 63) as u32;
    let w_end = (end_bit - 1) >> 6;
    let end_mod = (end_bit & 63) as u32;
    let inv_s = 64 - s_u32;

    let orig_start = read_u64_le(p, w_start * 8);
    let orig_end = read_u64_le(p, w_end * 8);
    let keep_lo = if b_start == 0 {
        0u64
    } else {
        (1u64 << b_start) - 1
    };
    let keep_hi = if end_mod == 0 {
        0u64
    } else {
        u64::MAX << end_mod
    };
    let start_movable = orig_start & !keep_lo;

    for i in ((w_start + 1)..=w_end).rev() {
        let cur = read_u64_le(p, i * 8);
        let prev = if i == w_start + 1 {
            start_movable
        } else {
            read_u64_le(p, (i - 1) * 8)
        };
        let mut shifted = (cur << s_u32) | (prev >> inv_s);
        if i == w_end && keep_hi != 0 {
            shifted = (orig_end & keep_hi) | (shifted & !keep_hi);
        }
        write_u64_le(p, i * 8, shifted);
    }

    let mut new_start = (orig_start & keep_lo) | (start_movable << s_u32);
    if masked_carry != 0 {
        new_start |= masked_carry << b_start;
        if b_start + s_u32 > 64 {
            let next_w = read_u64_le(p, (w_start + 1) * 8);
            write_u64_le(
                p,
                (w_start + 1) * 8,
                next_w | (masked_carry >> (64 - b_start)),
            );
        }
    }
    write_u64_le(p, w_start * 8, new_start);
}

#[inline(always)]
fn clear_metadata_slot(blocks: &mut [u8], block_bytes: usize, slot: usize) {
    let base = (slot >> SLOTS_PER_CHUNK_LOG2) * block_bytes;
    let bit = !(1u64 << (slot & 63));
    let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS) & bit;
    let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS) & bit;
    write_u64_le(blocks, base + BLOCK_RUNEND_POS, re);
    write_u64_le(blocks, base + BLOCK_EXTENSION_POS, ext);
}

#[inline(always)]
fn shift_metadata_right_word_masked(
    blocks: &mut [u8],
    base: usize,
    keep_mask: u64,
    move_mask: u64,
    re_carry: u64,
    ext_carry: u64,
) {
    let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS);
    let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS);
    write_u64_le(
        blocks,
        base + BLOCK_RUNEND_POS,
        (re & keep_mask) | ((re & move_mask) << 1) | re_carry,
    );
    write_u64_le(
        blocks,
        base + BLOCK_EXTENSION_POS,
        (ext & keep_mask) | ((ext & move_mask) << 1) | ext_carry,
    );
}

#[inline(always)]
fn shift_metadata_left_word_masked(
    blocks: &mut [u8],
    base: usize,
    keep_mask: u64,
    move_mask: u64,
    re_carry: u64,
    ext_carry: u64,
) {
    let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS);
    let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS);
    write_u64_le(
        blocks,
        base + BLOCK_RUNEND_POS,
        (re & keep_mask) | ((re & move_mask) >> 1) | re_carry,
    );
    write_u64_le(
        blocks,
        base + BLOCK_EXTENSION_POS,
        (ext & keep_mask) | ((ext & move_mask) >> 1) | ext_carry,
    );
}

/// Shifts both `runend` and `extension` bitmaps across AoS blocks for `[start .. end)` ->
/// `[start + 1 .. end + 1)`, clearing bit `start`.
pub(super) fn shift_metadata_right_one(
    blocks: &mut [u8],
    block_bytes: usize,
    start: usize,
    end: usize,
) {
    if start == end {
        clear_metadata_slot(blocks, block_bytes, start);
        return;
    }

    let c_start = start >> SLOTS_PER_CHUNK_LOG2;
    let b_start = (start & 63) as u32;
    let c_end = end >> SLOTS_PER_CHUNK_LOG2;
    let b_end = (end & 63) as u32;
    let lo_keep = (1u64 << b_start) - 1;
    let hi_keep = if b_end == 63 {
        0
    } else {
        !((1u64 << (b_end + 1)) - 1)
    };

    if c_start == c_end {
        let move_mask = ((1u64 << b_end) - 1) & !lo_keep;
        shift_metadata_right_word_masked(
            blocks,
            c_start * block_bytes,
            lo_keep | hi_keep,
            move_mask,
            0,
            0,
        );
        return;
    }

    let end_base = c_end * block_bytes;
    let prev_base = (c_end - 1) * block_bytes;
    let prev_re_carry = read_u64_le(blocks, prev_base + BLOCK_RUNEND_POS) >> 63;
    let prev_ext_carry = read_u64_le(blocks, prev_base + BLOCK_EXTENSION_POS) >> 63;
    shift_metadata_right_word_masked(
        blocks,
        end_base,
        hi_keep,
        (1u64 << b_end) - 1,
        prev_re_carry,
        prev_ext_carry,
    );

    for c in ((c_start + 1)..c_end).rev() {
        let p_base = (c - 1) * block_bytes;
        let re_carry = read_u64_le(blocks, p_base + BLOCK_RUNEND_POS) >> 63;
        let ext_carry = read_u64_le(blocks, p_base + BLOCK_EXTENSION_POS) >> 63;
        shift_metadata_right_word_masked(blocks, c * block_bytes, 0, u64::MAX, re_carry, ext_carry);
    }

    shift_metadata_right_word_masked(blocks, c_start * block_bytes, lo_keep, !lo_keep, 0, 0);
}

/// Shifts both `runend` and `extension` bitmaps across AoS blocks for `(start ..= end]` ->
/// `[start .. end)`, clearing bit `end`.
pub(super) fn shift_metadata_left_one(
    blocks: &mut [u8],
    block_bytes: usize,
    start: usize,
    end: usize,
) {
    if start == end {
        clear_metadata_slot(blocks, block_bytes, end);
        return;
    }

    let c_start = start >> SLOTS_PER_CHUNK_LOG2;
    let b_start = (start & 63) as u32;
    let c_end = end >> SLOTS_PER_CHUNK_LOG2;
    let b_end = (end & 63) as u32;
    let lo_keep = (1u64 << b_start) - 1;
    let hi_keep = if b_end == 63 {
        0
    } else {
        !((1u64 << (b_end + 1)) - 1)
    };
    let up_to_end = u64::MAX >> (63 - b_end);
    let up_to_start = u64::MAX >> (63 - b_start);

    if c_start == c_end {
        shift_metadata_left_word_masked(
            blocks,
            c_start * block_bytes,
            lo_keep | hi_keep,
            up_to_end & !up_to_start,
            0,
            0,
        );
        return;
    }

    let next_base = (c_start + 1) * block_bytes;
    let next_re_carry = (read_u64_le(blocks, next_base + BLOCK_RUNEND_POS) & 1) << 63;
    let next_ext_carry = (read_u64_le(blocks, next_base + BLOCK_EXTENSION_POS) & 1) << 63;
    shift_metadata_left_word_masked(
        blocks,
        c_start * block_bytes,
        lo_keep,
        !up_to_start,
        next_re_carry,
        next_ext_carry,
    );

    for c in (c_start + 1)..c_end {
        let n_base = (c + 1) * block_bytes;
        let re_carry = (read_u64_le(blocks, n_base + BLOCK_RUNEND_POS) & 1) << 63;
        let ext_carry = (read_u64_le(blocks, n_base + BLOCK_EXTENSION_POS) & 1) << 63;
        shift_metadata_left_word_masked(blocks, c * block_bytes, 0, u64::MAX, re_carry, ext_carry);
    }

    shift_metadata_left_word_masked(
        blocks,
        c_end * block_bytes,
        hi_keep,
        up_to_end & !1u64,
        0,
        0,
    );
}
