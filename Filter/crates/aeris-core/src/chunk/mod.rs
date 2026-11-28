//! 64-slot Quotient Filter AoS blocks, SIMD rank/select run navigation, and
//! Robin Hood slot displacement with tail overflow blocks.
//!
//! # Architectural Invariants
//! 1. **Canonical Domain:** Canonical home slots lie strictly in `0..N`, where
//!    `N = 2^q` is a multiple of [`SLOTS_PER_CHUNK`] (`64`).
//! 2. **Tail Overflow (No Wraparound):** Clusters near `N - 1` spill seamlessly
//!    into dynamically expandable tail blocks (`occupied == 0`) beyond `N`,
//!    eliminating circular wraparound and block-0 offset corruption.
//! 3. **Occupied / Run-End Bijection:** Every set `occupied` bit in `0..N`
//!    corresponds 1-to-1, in ascending order, to a physical slot with
//!    `qrun_end = runend & extension == 1` ([`KeepsakeSlotKind::BoxAndRunEnd`]).
//! 4. **Chunk Offset Invariant:** For every block `c` (`base_slot = 64 * c`), let
//!    `E_prev` be the physical slot index of the final `qrun_end` belonging to
//!    runs with `home_slot < base_slot` (if any). Then:
//!    `offset(c) = if E_prev >= base_slot { E_prev + 1 - base_slot } else { 0 }`.
//!    Values `< 255` are stored inline at byte `0` of the block; values `>= 255`
//!    store sentinel `255` inline and the exact `u32` offset in the cold
//!    `offset_overflow` `BTreeMap`.
//! 5. **Interleaved 64-Slot AoS Block Layout:** Each 64-slot block occupies
//!    `block_bytes = 25 + 8 * slot_bits` contiguous bytes with zero alignment
//!    padding:
//!    - `base + 0`: `offset: u8` (`0..=254` inline; `255` overflow sentinel)
//!    - `base + 1 .. base + 9`: `occupied: u64` (little-endian)
//!    - `base + 9 .. base + 17`: `runend: u64` (little-endian)
//!    - `base + 17 .. base + 25`: `extension: u64` (little-endian)
//!    - `base + 25 .. base + 25 + 8 * slot_bits`: 64 bit-packed payloads
//!      (`slot_bits = r + 1 + m` bits each, preserving the 1-bit InfiniFilter
//!      unary age sentinel).
//!
//!    Eight trailing zero guard bytes follow the final block so unaligned 8-byte
//!    safe loads/stores near the end of the table are always in bounds.

mod bulk;
mod invariants;
mod metadata;
mod navigation;
mod run_ops;
mod shift;
#[cfg(test)]
mod tests;

pub use bulk::BulkTableWriter;
pub use metadata::ChunkMetadata;

use aeris_types::{KeepsakeSlotKind, MAX_CHUNK_OFFSET, SLOTS_PER_CHUNK, SLOTS_PER_CHUNK_LOG2};
use std::collections::BTreeMap;

/// Number of extra 64-slot blocks appended whenever the tail overflow region grows.
const TAIL_GROWTH_CHUNKS: usize = 4;

/// Minimum initial tail overflow blocks allocated beyond the `N = 2^q` canonical slots.
const MIN_INITIAL_TAIL_CHUNKS: usize = 4;

/// Maximum initial tail overflow blocks pre-allocated on construction.
const MAX_INITIAL_TAIL_CHUNKS: usize = 16;

/// Byte offset of `offset: u8` within a 64-slot AoS block.
pub(super) const BLOCK_OFFSET_POS: usize = 0;

/// Byte offset of `occupied: u64` (LE) within a 64-slot AoS block.
pub(super) const BLOCK_OCCUPIED_POS: usize = 1;

/// Byte offset of `runend: u64` (LE) within a 64-slot AoS block.
pub(super) const BLOCK_RUNEND_POS: usize = 9;

/// Byte offset of `extension: u64` (LE) within a 64-slot AoS block.
pub(super) const BLOCK_EXTENSION_POS: usize = 17;

/// Exact header byte width (`1 + 8 + 8 + 8 = 25`) preceding the 64 packed payloads.
pub(super) const BLOCK_HEADER_BYTES: usize = 25;

/// Trailing zero guard bytes appended to `QuotientTable::blocks` so 8-byte LE
/// reads/writes on the final block never cross the buffer end.
pub(super) const TRAILING_GUARD_BYTES: usize = 8;

/// Reads a little-endian `u64` from `buf[off..off + 8]` without `unsafe`.
#[inline]
pub(super) fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

/// Writes a little-endian `u64` into `buf[off..off + 8]` without `unsafe`.
#[inline]
pub(super) fn write_u64_le(buf: &mut [u8], off: usize, val: u64) {
    buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

/// Quotient Filter physical table storing interleaved 64-slot AoS blocks
/// (`25`-byte metadata header + `8 * slot_bits` packed payload bytes per block)
/// together with cold offset overflow storage.
///
/// # Memory Layout Invariants
/// - `1 <= slot_bits <= 64`.
/// - `block_bytes == 25 + 8 * (slot_bits as usize)`.
/// - `blocks.len() == total_chunks * block_bytes + 8`.
/// - Trailing 8 guard bytes in `blocks` remain strictly zero.
#[derive(Clone, Debug)]
pub struct QuotientTable {
    pub(super) canonical_slots: usize,
    pub(super) canonical_chunks: usize,
    pub(super) total_chunks: usize,
    pub(super) slot_bits: u8,
    pub(super) slot_mask: u64,
    pub(super) block_bytes: usize,
    pub(super) blocks: Vec<u8>,
    pub(super) offset_overflow: BTreeMap<usize, u32>,
}

impl QuotientTable {
    /// Creates an empty [`QuotientTable`] with `canonical_slots = 2^q` slots
    /// and fixed payload width `slot_bits` (`1..=64`).
    ///
    /// # Panics
    /// Panics if `canonical_slots == 0`, `canonical_slots % 64 != 0`, or
    /// `slot_bits` is outside `1..=64`.
    ///
    /// # Complexity
    /// `O(canonical_slots * (25 + 8 * slot_bits) / 64)` time and space.
    pub fn new(canonical_slots: usize, slot_bits: u8) -> Self {
        assert!(
            canonical_slots >= SLOTS_PER_CHUNK && canonical_slots.is_multiple_of(SLOTS_PER_CHUNK),
            "canonical_slots ({canonical_slots}) must be a positive multiple of {SLOTS_PER_CHUNK}"
        );
        assert!(
            (1..=64).contains(&slot_bits),
            "slot_bits must be in 1..=64, got {slot_bits}"
        );

        let canonical_chunks = canonical_slots >> SLOTS_PER_CHUNK_LOG2;
        let initial_tail =
            (canonical_chunks / 8).clamp(MIN_INITIAL_TAIL_CHUNKS, MAX_INITIAL_TAIL_CHUNKS);
        let total_chunks = canonical_chunks + initial_tail;
        let slot_mask = if slot_bits == 64 {
            u64::MAX
        } else {
            (1u64 << slot_bits) - 1
        };
        let block_bytes = BLOCK_HEADER_BYTES + 8 * (slot_bits as usize);
        let total_bytes = total_chunks * block_bytes + TRAILING_GUARD_BYTES;

        Self {
            canonical_slots,
            canonical_chunks,
            total_chunks,
            slot_bits,
            slot_mask,
            block_bytes,
            blocks: vec![0u8; total_bytes],
            offset_overflow: BTreeMap::new(),
        }
    }

    /// Returns canonical slot capacity `N = 2^q`.
    #[inline]
    pub const fn canonical_slots(&self) -> usize {
        self.canonical_slots
    }

    /// Returns canonical 64-slot chunk count `N / 64`.
    #[inline]
    pub const fn canonical_chunks(&self) -> usize {
        self.canonical_chunks
    }

    /// Returns total physical slots including tail overflow chunks.
    #[inline]
    pub const fn physical_slots(&self) -> usize {
        self.total_chunks << SLOTS_PER_CHUNK_LOG2
    }

    /// Returns total allocated 64-slot chunks (canonical + tail).
    #[inline]
    pub const fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Returns the bit width of each slot payload (`1..=64`).
    #[inline]
    pub const fn slot_bits(&self) -> u8 {
        self.slot_bits
    }

    /// Returns the exact heap memory usage of the interleaved AoS block buffer
    /// (including trailing guard bytes) plus cold offset overflow entries in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        let overflow_bytes = self.offset_overflow.len()
            * (core::mem::size_of::<usize>() + core::mem::size_of::<u32>());
        self.blocks.len() + overflow_bytes
    }

    /// Returns `true` if canonical slot `home` has a non-empty quotient run.
    #[inline]
    pub fn is_occupied(&self, home: usize) -> bool {
        debug_assert!(home < self.canonical_slots);
        let chunk_idx = home >> SLOTS_PER_CHUNK_LOG2;
        let local = (home & 63) as u32;
        ((self.block_occupied(chunk_idx) >> local) & 1) != 0
    }

    /// Reads the `slot_bits`-wide payload at physical index `slot`.
    ///
    /// # Complexity
    /// `O(1)` time, zero heap allocations.
    #[inline]
    pub fn get_payload(&self, slot: usize) -> u64 {
        debug_assert!(slot < self.physical_slots());
        let base = self.block_base(slot >> SLOTS_PER_CHUNK_LOG2);
        self.read_payload_at_base(base, slot & 63)
    }

    /// Writes `value` into the payload at physical index `slot`.
    ///
    /// Any bits above `slot_bits` in `value` are masked off.
    ///
    /// # Complexity
    /// `O(1)` time, zero heap allocations.
    #[inline]
    pub fn set_payload(&mut self, slot: usize, value: u64) {
        debug_assert!(slot < self.physical_slots());
        let base = self.block_base(slot >> SLOTS_PER_CHUNK_LOG2);
        self.write_payload_at_base(base, slot & 63, value);
    }

    /// Returns the [`KeepsakeSlotKind`] of physical slot `slot`.
    #[inline]
    pub fn slot_kind(&self, slot: usize) -> KeepsakeSlotKind {
        debug_assert!(slot < self.physical_slots());
        let base = self.block_base(slot >> SLOTS_PER_CHUNK_LOG2);
        self.read_kind_at_base(base, (slot & 63) as u32)
    }

    /// Sets the [`KeepsakeSlotKind`] of physical slot `slot`.
    #[inline]
    pub fn set_slot_kind(&mut self, slot: usize, kind: KeepsakeSlotKind) {
        debug_assert!(slot < self.physical_slots());
        let base = self.block_base(slot >> SLOTS_PER_CHUNK_LOG2);
        self.write_kind_at_base(base, (slot & 63) as u32, kind);
    }

    /// Reads both the `slot_bits`-wide payload and the [`KeepsakeSlotKind`] of
    /// physical slot `slot` from its single contiguous 64-slot AoS block.
    ///
    /// # Complexity
    /// `O(1)` time, zero heap allocations.
    #[inline]
    pub fn get_payload_and_kind(&self, slot: usize) -> (u64, KeepsakeSlotKind) {
        debug_assert!(slot < self.physical_slots());
        let base = self.block_base(slot >> SLOTS_PER_CHUNK_LOG2);
        let local = slot & 63;
        let payload = self.read_payload_at_base(base, local);
        let kind = self.read_kind_at_base(base, local as u32);
        (payload, kind)
    }

    /// Returns the `runend` bitmap word for chunk `chunk_idx`.
    #[inline]
    pub fn chunk_runend_word(&self, chunk_idx: usize) -> u64 {
        debug_assert!(chunk_idx < self.total_chunks);
        self.block_runend(chunk_idx)
    }

    /// Returns the exact right-shift offset of chunk `chunk_idx`.
    #[inline]
    pub fn chunk_offset(&self, chunk_idx: usize) -> usize {
        debug_assert!(chunk_idx < self.total_chunks);
        let raw = self.blocks[self.block_base(chunk_idx) + BLOCK_OFFSET_POS];
        if raw < MAX_CHUNK_OFFSET {
            raw as usize
        } else {
            self.offset_overflow
                .get(&chunk_idx)
                .copied()
                .unwrap_or(MAX_CHUNK_OFFSET as u32) as usize
        }
    }

    /// Sets the exact offset of chunk `chunk_idx`, spilling to `offset_overflow`
    /// only if `offset >= 255`.
    #[inline]
    pub fn set_chunk_offset(&mut self, chunk_idx: usize, offset: usize) {
        debug_assert!(chunk_idx < self.total_chunks);
        let pos = self.block_base(chunk_idx) + BLOCK_OFFSET_POS;
        if offset < MAX_CHUNK_OFFSET as usize {
            if self.blocks[pos] == MAX_CHUNK_OFFSET {
                self.offset_overflow.remove(&chunk_idx);
            }
            self.blocks[pos] = offset as u8;
        } else {
            self.blocks[pos] = MAX_CHUNK_OFFSET;
            self.offset_overflow.insert(chunk_idx, offset as u32);
        }
    }

    /// Returns an iterator yielding all occupied canonical home slots in ascending order.
    pub fn occupied_homes(&self) -> OccupiedHomesIter<'_> {
        let current_word = if self.canonical_chunks > 0 {
            self.block_occupied(0)
        } else {
            0
        };
        OccupiedHomesIter {
            blocks: &self.blocks,
            block_bytes: self.block_bytes,
            canonical_chunks: self.canonical_chunks,
            chunk_idx: 0,
            current_word,
        }
    }

    // -------------------------------------------------------------------------
    // Internal AoS block primitive accessors
    // -------------------------------------------------------------------------

    #[inline]
    pub(super) const fn block_base(&self, chunk_idx: usize) -> usize {
        chunk_idx * self.block_bytes
    }

    #[inline]
    pub(super) fn block_occupied(&self, chunk_idx: usize) -> u64 {
        read_u64_le(
            &self.blocks,
            self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS,
        )
    }

    #[inline]
    pub(super) fn set_block_occupied_bit(&mut self, chunk_idx: usize, local: u32) {
        let pos = self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS;
        let updated = read_u64_le(&self.blocks, pos) | (1u64 << local);
        write_u64_le(&mut self.blocks, pos, updated);
    }

    #[inline]
    pub(super) fn clear_block_occupied_bit(&mut self, chunk_idx: usize, local: u32) {
        let pos = self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS;
        let updated = read_u64_le(&self.blocks, pos) & !(1u64 << local);
        write_u64_le(&mut self.blocks, pos, updated);
    }

    #[inline]
    pub(super) fn block_runend(&self, chunk_idx: usize) -> u64 {
        read_u64_le(&self.blocks, self.block_base(chunk_idx) + BLOCK_RUNEND_POS)
    }

    #[inline]
    pub(super) fn block_qrun_end_mask(&self, chunk_idx: usize) -> u64 {
        let base = self.block_base(chunk_idx);
        read_u64_le(&self.blocks, base + BLOCK_RUNEND_POS)
            & read_u64_le(&self.blocks, base + BLOCK_EXTENSION_POS)
    }

    #[inline]
    pub(super) fn read_kind_at_base(&self, base: usize, local: u32) -> KeepsakeSlotKind {
        let re = ((read_u64_le(&self.blocks, base + BLOCK_RUNEND_POS) >> local) & 1) != 0;
        let ext = ((read_u64_le(&self.blocks, base + BLOCK_EXTENSION_POS) >> local) & 1) != 0;
        KeepsakeSlotKind::from_bits(re, ext)
    }

    #[inline]
    pub(super) fn write_kind_at_base(&mut self, base: usize, local: u32, kind: KeepsakeSlotKind) {
        let bit = 1u64 << local;
        let (re, ext) = kind.to_bits();
        let re_pos = base + BLOCK_RUNEND_POS;
        let ext_pos = base + BLOCK_EXTENSION_POS;
        let mut re_word = read_u64_le(&self.blocks, re_pos);
        let mut ext_word = read_u64_le(&self.blocks, ext_pos);
        if re {
            re_word |= bit;
        } else {
            re_word &= !bit;
        }
        if ext {
            ext_word |= bit;
        } else {
            ext_word &= !bit;
        }
        write_u64_le(&mut self.blocks, re_pos, re_word);
        write_u64_le(&mut self.blocks, ext_pos, ext_word);
    }

    #[inline]
    pub(super) fn read_payload_at_base(&self, base: usize, local: usize) -> u64 {
        let bit_pos = local * (self.slot_bits as usize);
        let byte_off = base + BLOCK_HEADER_BYTES + (bit_pos >> 3);
        let bit_in_byte = (bit_pos & 7) as u32;
        let word = read_u64_le(&self.blocks, byte_off);
        if bit_in_byte + (self.slot_bits as u32) <= 64 {
            (word >> bit_in_byte) & self.slot_mask
        } else {
            // `bit_in_byte` is strictly in 1..=7 and `slot_bits >= 58`.
            let lo = word >> bit_in_byte;
            let hi = (self.blocks[byte_off + 8] as u64) << (64 - bit_in_byte);
            (lo | hi) & self.slot_mask
        }
    }

    #[inline]
    pub(super) fn write_payload_at_base(&mut self, base: usize, local: usize, value: u64) {
        let masked = value & self.slot_mask;
        let bit_pos = local * (self.slot_bits as usize);
        let byte_off = base + BLOCK_HEADER_BYTES + (bit_pos >> 3);
        let bit_in_byte = (bit_pos & 7) as u32;
        if bit_in_byte + (self.slot_bits as u32) <= 64 {
            // Preserve neighboring bits in `[0, bit_in_byte)` and `[bit_in_byte + slot_bits, 64)`.
            let clear = !(self.slot_mask << bit_in_byte);
            let cur = read_u64_le(&self.blocks, byte_off);
            write_u64_le(
                &mut self.blocks,
                byte_off,
                (cur & clear) | (masked << bit_in_byte),
            );
        } else {
            // `bit_in_byte` is strictly in 1..=7; `hi_bits` is strictly in 1..=6.
            let lo_bits = 64 - bit_in_byte;
            let hi_bits = (self.slot_bits as u32) - lo_bits;
            let lo_keep_mask = (1u64 << bit_in_byte) - 1;
            let cur_lo = read_u64_le(&self.blocks, byte_off);
            write_u64_le(
                &mut self.blocks,
                byte_off,
                (cur_lo & lo_keep_mask) | (masked << bit_in_byte),
            );
            let hi_mask = (1u8 << hi_bits) - 1;
            let cur_hi = self.blocks[byte_off + 8];
            self.blocks[byte_off + 8] =
                (cur_hi & !hi_mask) | (((masked >> lo_bits) as u8) & hi_mask);
        }
    }

    #[inline]
    pub(super) fn write_slots(&mut self, start: usize, slots: &[(u64, KeepsakeSlotKind)]) {
        for (i, &(payload, kind)) in slots.iter().enumerate() {
            let idx = start + i;
            let base = self.block_base(idx >> SLOTS_PER_CHUNK_LOG2);
            let local = idx & 63;
            self.write_payload_at_base(base, local, payload);
            self.write_kind_at_base(base, local as u32, kind);
        }
    }

    pub(super) fn ensure_physical_slot(&mut self, required_slot_exclusive: usize) {
        if required_slot_exclusive <= self.physical_slots() {
            return;
        }
        let needed_chunks = required_slot_exclusive.div_ceil(SLOTS_PER_CHUNK);
        let new_total_chunks = needed_chunks.max(self.total_chunks + TAIL_GROWTH_CHUNKS);
        self.total_chunks = new_total_chunks;
        self.blocks.resize(
            new_total_chunks * self.block_bytes + TRAILING_GUARD_BYTES,
            0u8,
        );
    }
}

/// Iterator over occupied canonical home slots of a [`QuotientTable`] in ascending order.
pub struct OccupiedHomesIter<'a> {
    blocks: &'a [u8],
    block_bytes: usize,
    canonical_chunks: usize,
    chunk_idx: usize,
    current_word: u64,
}

impl Iterator for OccupiedHomesIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.chunk_idx += 1;
            if self.chunk_idx >= self.canonical_chunks {
                return None;
            }
            self.current_word = read_u64_le(
                self.blocks,
                self.chunk_idx * self.block_bytes + BLOCK_OCCUPIED_POS,
            );
        }
        let bit = self.current_word.trailing_zeros() as usize;
        self.current_word &= self.current_word - 1;
        Some((self.chunk_idx << SLOTS_PER_CHUNK_LOG2) + bit)
    }
}
