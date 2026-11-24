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

use crate::simd;
use aeris_types::{
    AerisError, AerisResult, KeepsakeSlotKind, MAX_CHUNK_OFFSET, SLOTS_PER_CHUNK,
    SLOTS_PER_CHUNK_LOG2,
};
use std::collections::BTreeMap;

/// Number of extra 64-slot blocks appended whenever the tail overflow region grows.
const TAIL_GROWTH_CHUNKS: usize = 4;

/// Minimum initial tail overflow blocks allocated beyond the `N = 2^q` canonical slots.
const MIN_INITIAL_TAIL_CHUNKS: usize = 4;

/// Maximum initial tail overflow blocks pre-allocated on construction.
const MAX_INITIAL_TAIL_CHUNKS: usize = 16;

/// Byte offset of `offset: u8` within a 64-slot AoS block.
const BLOCK_OFFSET_POS: usize = 0;

/// Byte offset of `occupied: u64` (LE) within a 64-slot AoS block.
const BLOCK_OCCUPIED_POS: usize = 1;

/// Byte offset of `runend: u64` (LE) within a 64-slot AoS block.
const BLOCK_RUNEND_POS: usize = 9;

/// Byte offset of `extension: u64` (LE) within a 64-slot AoS block.
const BLOCK_EXTENSION_POS: usize = 17;

/// Exact header byte width (`1 + 8 + 8 + 8 = 25`) preceding the 64 packed payloads.
const BLOCK_HEADER_BYTES: usize = 25;

/// Trailing zero guard bytes appended to `QuotientTable::blocks` so 8-byte LE
/// reads/writes on the final block never cross the buffer end.
const TRAILING_GUARD_BYTES: usize = 8;

/// Reads a little-endian `u64` from `buf[off..off + 8]` without `unsafe`.
#[inline]
fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

/// Writes a little-endian `u64` into `buf[off..off + 8]` without `unsafe`.
#[inline]
fn write_u64_le(buf: &mut [u8], off: usize, val: u64) {
    buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

/// Metadata block governing 64 contiguous Quotient Filter slots.
///
/// Packs three 64-bit bitmaps (`occupied`, `runend`, `extension`) and an 8-bit
/// inline offset (`25 bytes` logical), enabling `O(1)` SIMD `POPCNT`/`PDEP`
/// run navigation. Retained as a public value type for downstream callers and
/// inspection utilities; [`QuotientTable`] stores these exact fields interleaved
/// with each block's 64 bit-packed payloads in a single zero-padding byte array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkMetadata {
    /// Bit `i` is `1` iff canonical home slot `64 * chunk + i` has a non-empty quotient run.
    pub occupied: u64,
    /// Bit `i` is `1` iff physical slot `64 * chunk + i` terminates a keepsake box.
    pub runend: u64,
    /// Bit `i` is `1` iff physical slot `64 * chunk + i` is an extension slot (`runend = 0`)
    /// or terminates a quotient run (`runend = 1`).
    pub extension: u64,
    /// Number of leading slots in this chunk consumed by runs originating in earlier chunks.
    /// A value of [`MAX_CHUNK_OFFSET`] (`255`) signals an exact `u32` entry in
    /// `QuotientTable::offset_overflow`.
    pub offset: u8,
}

impl ChunkMetadata {
    /// Returns `true` if canonical slot `local_idx` (`0..64`) is occupied.
    #[inline]
    pub const fn is_occupied(&self, local_idx: u32) -> bool {
        ((self.occupied >> local_idx) & 1) != 0
    }

    /// Sets the `occupied` bit for `local_idx` (`0..64`).
    #[inline]
    pub fn set_occupied(&mut self, local_idx: u32) {
        self.occupied |= 1u64 << local_idx;
    }

    /// Clears the `occupied` bit for `local_idx` (`0..64`).
    #[inline]
    pub fn clear_occupied(&mut self, local_idx: u32) {
        self.occupied &= !(1u64 << local_idx);
    }

    /// Returns the 64-bit bitmask of slots that terminate a quotient run
    /// (`runend & extension`).
    #[inline]
    pub const fn qrun_end_mask(&self) -> u64 {
        self.runend & self.extension
    }

    /// Decodes the [`KeepsakeSlotKind`] of physical slot `local_idx` (`0..64`).
    #[inline]
    pub const fn slot_kind(&self, local_idx: u32) -> KeepsakeSlotKind {
        let re = ((self.runend >> local_idx) & 1) != 0;
        let ext = ((self.extension >> local_idx) & 1) != 0;
        KeepsakeSlotKind::from_bits(re, ext)
    }

    /// Encodes `kind` into the `(runend, extension)` bits of `local_idx` (`0..64`).
    #[inline]
    pub fn set_slot_kind(&mut self, local_idx: u32, kind: KeepsakeSlotKind) {
        let bit = 1u64 << local_idx;
        let (re, ext) = kind.to_bits();
        if re {
            self.runend |= bit;
        } else {
            self.runend &= !bit;
        }
        if ext {
            self.extension |= bit;
        } else {
            self.extension &= !bit;
        }
    }
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
    canonical_slots: usize,
    canonical_chunks: usize,
    total_chunks: usize,
    slot_bits: u8,
    slot_mask: u64,
    block_bytes: usize,
    blocks: Vec<u8>,
    offset_overflow: BTreeMap<usize, u32>,
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
    /// Computing the block base address once allows hot-path query scanning and
    /// keepsake box decoding to fetch both metadata bits and payload bits from
    /// the same cache-local block.
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
        mut run_end: usize,
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
            return;
        }

        if new_len > old_len {
            let delta = new_len - old_len;
            if region_start <= run_end {
                for _ in 0..delta {
                    self.insert_blank_slot_interior(home, region_start);
                }
            } else {
                // Appending strictly after `run_end`.
                for _ in 0..delta {
                    self.append_blank_slot_after_run(home, run_end);
                    run_end += 1;
                }
            }
            self.write_slots(region_start, new_slots);
            return;
        }

        // Shrinking: `new_len < old_len`.
        let delta = old_len - new_len;
        let run_len = run_end - run_start + 1;

        if new_len == 0 && old_len == run_len {
            // Deleting the entire run.
            for _ in 0..(run_len - 1) {
                self.remove_slot_interior(home, run_start, run_end);
                run_end -= 1;
            }
            self.remove_sole_slot_of_run(home, run_start);
            return;
        }

        // At least 1 slot remains in `home`'s run after shrinking.
        let removes_tail_of_run = region_start + old_len - 1 == run_end && new_len == 0;
        if removes_tail_of_run {
            // Remove the first `delta - 1` slots interior to the run so `run_end`
            // stays marked `BoxAndRunEnd`, then remove the terminal slot and promote
            // `region_start - 1` to `BoxAndRunEnd`.
            for _ in 0..(delta - 1) {
                self.remove_slot_interior(home, region_start, run_end);
                run_end -= 1;
            }
            self.remove_terminal_slot_of_multi_slot_run(home, run_end);
        } else {
            // `region_start + delta <= run_end`, so deleting `delta` slots at
            // `region_start` leaves the `BoxAndRunEnd` at `run_end` shifting left cleanly.
            for _ in 0..delta {
                self.remove_slot_interior(home, region_start, run_end);
                run_end -= 1;
            }
            self.write_slots(region_start, new_slots);
        }
    }

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

        // Update chunk offsets for every chunk boundary `base` with `home < base < end_exclusive`.
        let first_chunk_after_home = (home >> SLOTS_PER_CHUNK_LOG2) + 1;
        let last_spilled_chunk = (end_exclusive - 1) >> SLOTS_PER_CHUNK_LOG2;
        for c in first_chunk_after_home..=last_spilled_chunk {
            let base = c << SLOTS_PER_CHUNK_LOG2;
            self.set_chunk_offset(c, end_exclusive - base);
        }

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

        // Verify exact chunk offsets from first principles.
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

        // Verify every run's start/end, locate_run agreement, and no stray metadata in gaps.
        let mut prev_end_plus_one = 0usize;
        for (idx, (&home, &expected_end)) in occ_slots.iter().zip(qrun_ends.iter()).enumerate() {
            let expected_start = home.max(prev_end_plus_one);
            if expected_end < expected_start {
                return Err(AerisError::InternalInvariant(format!(
                    "run #{idx} for home {home}: end {expected_end} < start {expected_start}"
                )));
            }
            // Check empty gap before this run has zero metadata bits.
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

        // Check trailing slots after last run have zero metadata bits.
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

    // -------------------------------------------------------------------------
    // Internal AoS block primitive accessors
    // -------------------------------------------------------------------------

    #[inline]
    const fn block_base(&self, chunk_idx: usize) -> usize {
        chunk_idx * self.block_bytes
    }

    #[inline]
    fn block_occupied(&self, chunk_idx: usize) -> u64 {
        read_u64_le(
            &self.blocks,
            self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS,
        )
    }

    #[inline]
    fn set_block_occupied_bit(&mut self, chunk_idx: usize, local: u32) {
        let pos = self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS;
        let updated = read_u64_le(&self.blocks, pos) | (1u64 << local);
        write_u64_le(&mut self.blocks, pos, updated);
    }

    #[inline]
    fn clear_block_occupied_bit(&mut self, chunk_idx: usize, local: u32) {
        let pos = self.block_base(chunk_idx) + BLOCK_OCCUPIED_POS;
        let updated = read_u64_le(&self.blocks, pos) & !(1u64 << local);
        write_u64_le(&mut self.blocks, pos, updated);
    }

    #[inline]
    fn block_runend(&self, chunk_idx: usize) -> u64 {
        read_u64_le(&self.blocks, self.block_base(chunk_idx) + BLOCK_RUNEND_POS)
    }

    #[inline]
    fn block_qrun_end_mask(&self, chunk_idx: usize) -> u64 {
        let base = self.block_base(chunk_idx);
        read_u64_le(&self.blocks, base + BLOCK_RUNEND_POS)
            & read_u64_le(&self.blocks, base + BLOCK_EXTENSION_POS)
    }

    #[inline]
    fn read_kind_at_base(&self, base: usize, local: u32) -> KeepsakeSlotKind {
        let re = ((read_u64_le(&self.blocks, base + BLOCK_RUNEND_POS) >> local) & 1) != 0;
        let ext = ((read_u64_le(&self.blocks, base + BLOCK_EXTENSION_POS) >> local) & 1) != 0;
        KeepsakeSlotKind::from_bits(re, ext)
    }

    #[inline]
    fn write_kind_at_base(&mut self, base: usize, local: u32, kind: KeepsakeSlotKind) {
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
    fn read_payload_at_base(&self, base: usize, local: usize) -> u64 {
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
    fn write_payload_at_base(&mut self, base: usize, local: usize, value: u64) {
        let masked = value & self.slot_mask;
        let bit_pos = local * (self.slot_bits as usize);
        let byte_off = base + BLOCK_HEADER_BYTES + (bit_pos >> 3);
        let bit_in_byte = (bit_pos & 7) as u32;
        if bit_in_byte + (self.slot_bits as u32) <= 64 {
            // Only bits in `[bit_in_byte, bit_in_byte + slot_bits)` are modified;
            // any trailing bytes belonging to the next block or guard region are preserved bit-for-bit.
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

    // -------------------------------------------------------------------------
    // Internal navigation & 1-slot shift primitives
    // -------------------------------------------------------------------------

    #[inline]
    fn write_slots(&mut self, start: usize, slots: &[(u64, KeepsakeSlotKind)]) {
        for (i, &(payload, kind)) in slots.iter().enumerate() {
            let idx = start + i;
            let base = self.block_base(idx >> SLOTS_PER_CHUNK_LOG2);
            let local = idx & 63;
            self.write_payload_at_base(base, local, payload);
            self.write_kind_at_base(base, local as u32, kind);
        }
    }

    #[inline]
    fn select_qrun_end(&self, start_slot: usize, mut rank: usize) -> usize {
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
                // Fast check for next qrun_end within the same chunk.
                if first_local < 63 {
                    let rest = qrun & (u64::MAX << (first_local + 1));
                    if rest != 0 {
                        let second_global =
                            (chunk_idx << SLOTS_PER_CHUNK_LOG2) + (rest.trailing_zeros() as usize);
                        return (first_global, second_global);
                    }
                }
                // Next qrun_end lies in a subsequent chunk.
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
    fn find_first_empty_slot(&mut self, start: usize) -> usize {
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

            // Now `pos >= base + offset`, so `offset <= (pos & 63) < 64` and every run
            // originating in `< base` has already terminated strictly before `base + offset`.
            let local = (pos & 63) as u32;
            let block_base = self.block_base(c);
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
            let mut pending = occ_cnt - ended_cnt;
            if pending == 0 {
                return pos;
            }

            // Strictly forward cluster jump: at entry, `pending` runs with `home <= pos`
            // remain unterminated at `>= pos`, so the `pending`-th `qrun_end` at `>= pos`
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
    }

    /// Computes the trailing slot that becomes vacated when deleting 1 slot from a
    /// run currently ending at `run_end`.
    fn find_vacated_slot(&self, run_end: usize) -> usize {
        let mut cur_end = run_end;
        loop {
            match self.end_of_runs_up_to(cur_end) {
                Some(e) if e > cur_end => cur_end = e,
                _ => return cur_end,
            }
        }
    }

    fn ensure_physical_slot(&mut self, required_slot_exclusive: usize) {
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
        // `pos` is now PayloadInterior (`runend=0, extension=0`); the existing
        // `BoxAndRunEnd` of `home`'s run shifted right by 1 automatically.
    }

    /// Appends 1 blank slot immediately after `run_end` of an existing run for `home`,
    /// transferring the `BoxAndRunEnd` marker from `run_end` to `run_end + 1`.
    fn append_blank_slot_after_run(&mut self, home: usize, run_end: usize) {
        let pos = run_end + 1;
        let empty = self.find_first_empty_slot(pos);
        self.ensure_physical_slot(empty + 1);
        self.shift_right_one(pos, empty);
        self.increment_offsets(home, empty);
        // Demote old run_end to BoxEnd and mark the newly appended slot as BoxAndRunEnd.
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

    /// Shifts payloads and `(runend, extension)` metadata for `[start .. end)`
    /// right by 1 slot into `[start + 1 .. end + 1)` and clears slot `start`.
    fn shift_right_one(&mut self, start: usize, end: usize) {
        self.shift_payloads_right_one(start, end);
        shift_metadata_right_one(&mut self.blocks, self.block_bytes, start, end);
    }

    /// Shifts payloads and `(runend, extension)` metadata for `(start ..= end]`
    /// left by 1 slot into `[start .. end)` and clears slot `end`.
    fn shift_left_one(&mut self, start: usize, end: usize) {
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
            // 64 exact `u64` words: shift words `0..63` up by 8 bytes.
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

    /// Shifts bit-packed slot payloads inside a single block `chunk_idx` for
    /// `[lo .. hi)` right by 1 slot into `[lo + 1 ..= hi]` and writes `carry`
    /// into slot `lo`, operating in 64-bit word steps over a single block slice
    /// so LLVM eliminates per-slot bounds checks.
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
            // Fast single-word intra-block right shift (covers 1–4 slot Robin Hood shifts).
            let byte_off = payload_base + (start_bit >> 3);
            let word = read_u64_le(&self.blocks, byte_off);
            let span_mask = if span_bits == 64 {
                u64::MAX
            } else {
                ((1u64 << span_bits) - 1) << bit_in_byte
            };
            let move_bits = (hi - lo) * s;
            let move_mask = ((1u64 << move_bits) - 1) << bit_in_byte;
            let updated = (word & !span_mask)
                | ((word & move_mask) << (s as u32))
                | (masked_carry << bit_in_byte);
            write_u64_le(&mut self.blocks, byte_off, updated);
            return;
        }

        // Multi-word partial block right shift: 64-bit word steps with a single slice check.
        let words = s;
        let p = &mut self.blocks[payload_base..(payload_base + words * 8)];
        let w_start = start_bit >> 6;
        let b_start = (start_bit & 63) as u32;
        let w_end = (end_bit - 1) >> 6;
        let end_mod = (end_bit & 63) as u32;
        let s_u32 = s as u32;
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

    /// Shifts bit-packed slot payloads for `[start .. end)` right by 1 slot into
    /// `[start + 1 .. end + 1)` and zeroes slot `start`.
    fn shift_payloads_right_one(&mut self, start: usize, end: usize) {
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

        // Trailing block `c_end`.
        let b_end = end & 63;
        let end_carry = self.read_payload_at_base(self.block_base(c_end - 1), 63);
        self.shift_partial_block_payload_right(c_end, 0, b_end, end_carry);

        // Intermediate full blocks `(c_start + 1)..c_end` using word-level shifts.
        for c in ((c_start + 1)..c_end).rev() {
            let carry = self.read_payload_at_base(self.block_base(c - 1), 63);
            self.shift_full_block_payload_right(c, carry);
        }

        // Starting block `c_start`.
        let b_start = start & 63;
        self.shift_partial_block_payload_right(c_start, b_start, 63, 0);
    }

    /// Shifts bit-packed slot payloads for `(start ..= end]` left by 1 slot into
    /// `[start .. end)` and zeroes slot `end`.
    fn shift_payloads_left_one(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end && end < self.physical_slots());
        if start == end {
            self.set_payload(end, 0);
            return;
        }
        let c_start = start >> SLOTS_PER_CHUNK_LOG2;
        let c_end = end >> SLOTS_PER_CHUNK_LOG2;

        if c_start == c_end {
            let base = self.block_base(c_start);
            let lo = start & 63;
            let hi = end & 63;
            let s = self.slot_bits as usize;
            let start_bit = lo * s;
            let span_bits = (hi - lo + 1) * s;
            let bit_in_byte = (start_bit & 7) as u32;
            if bit_in_byte + (span_bits as u32) <= 64 {
                // Fast single-word intra-block left shift.
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
            return;
        }

        // Starting block `c_start`.
        let b_start = start & 63;
        let start_base = self.block_base(c_start);
        for local in b_start..63 {
            let v = self.read_payload_at_base(start_base, local + 1);
            self.write_payload_at_base(start_base, local, v);
        }
        let start_carry = self.read_payload_at_base(self.block_base(c_start + 1), 0);
        self.write_payload_at_base(start_base, 63, start_carry);

        // Intermediate full blocks `(c_start + 1)..c_end` using word-level shifts.
        for c in (c_start + 1)..c_end {
            let carry = self.read_payload_at_base(self.block_base(c + 1), 0);
            self.shift_full_block_payload_left(c, carry);
        }

        // Trailing block `c_end`.
        let b_end = end & 63;
        let end_base = self.block_base(c_end);
        for local in 0..b_end {
            let v = self.read_payload_at_base(end_base, local + 1);
            self.write_payload_at_base(end_base, local, v);
        }
        self.write_payload_at_base(end_base, b_end, 0);
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
        debug_assert!(home < self.table.canonical_slots);
        let hc = home >> SLOTS_PER_CHUNK_LOG2;
        if hc != self.occ_chunk {
            self.flush_occ();
            self.occ_chunk = hc;
            self.occ_word = 0;
        }
        self.occ_word |= 1u64 << ((home & 63) as u32);

        let start = home.max(self.write_cursor);
        self.push_slot(start, payload, KeepsakeSlotKind::BoxAndRunEnd);
        let end_exclusive = start + 1;

        let first_chunk_after_home = hc + 1;
        let last_spilled_chunk = start >> SLOTS_PER_CHUNK_LOG2;
        if first_chunk_after_home <= last_spilled_chunk {
            for c in first_chunk_after_home..=last_spilled_chunk {
                let base = c << SLOTS_PER_CHUNK_LOG2;
                self.table.set_chunk_offset(c, end_exclusive - base);
            }
        }
        self.write_cursor = end_exclusive;
    }

    /// Appends a multi-slot quotient run for `home` in strictly ascending `home` order.
    #[inline]
    pub fn append_run(&mut self, home: usize, run_slots: &[(u64, KeepsakeSlotKind)]) {
        debug_assert!(home < self.table.canonical_slots);
        debug_assert!(!run_slots.is_empty());
        debug_assert!(run_slots.last().is_some_and(|&(_, k)| k.is_run_end()));

        let hc = home >> SLOTS_PER_CHUNK_LOG2;
        if hc != self.occ_chunk {
            self.flush_occ();
            self.occ_chunk = hc;
            self.occ_word = 0;
        }
        self.occ_word |= 1u64 << ((home & 63) as u32);

        let start = home.max(self.write_cursor);
        for (i, &(payload, kind)) in run_slots.iter().enumerate() {
            self.push_slot(start + i, payload, kind);
        }
        let end_exclusive = start + run_slots.len();

        let first_chunk_after_home = hc + 1;
        let last_spilled_chunk = (end_exclusive - 1) >> SLOTS_PER_CHUNK_LOG2;
        if first_chunk_after_home <= last_spilled_chunk {
            for c in first_chunk_after_home..=last_spilled_chunk {
                let base = c << SLOTS_PER_CHUNK_LOG2;
                self.table.set_chunk_offset(c, end_exclusive - base);
            }
        }
        self.write_cursor = end_exclusive;
    }

    /// Flushes all buffered block headers and payloads into the underlying [`QuotientTable`].
    pub fn finish(mut self) {
        self.flush_occ();
        self.flush_cur_block();
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

/// Shifts both `runend` and `extension` bitmaps across AoS blocks for `[start .. end)` ->
/// `[start + 1 .. end + 1)`, clearing bit `start`.
fn shift_metadata_right_one(blocks: &mut [u8], block_bytes: usize, start: usize, end: usize) {
    if start == end {
        let base = (start >> SLOTS_PER_CHUNK_LOG2) * block_bytes;
        let bit = !(1u64 << (start & 63));
        let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS) & bit;
        let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS) & bit;
        write_u64_le(blocks, base + BLOCK_RUNEND_POS, re);
        write_u64_le(blocks, base + BLOCK_EXTENSION_POS, ext);
        return;
    }

    let c_start = start >> SLOTS_PER_CHUNK_LOG2;
    let b_start = (start & 63) as u32;
    let c_end = end >> SLOTS_PER_CHUNK_LOG2;
    let b_end = (end & 63) as u32;

    if c_start == c_end {
        let lo_keep = (1u64 << b_start) - 1;
        let hi_keep = if b_end == 63 {
            0
        } else {
            !((1u64 << (b_end + 1)) - 1)
        };
        let keep = lo_keep | hi_keep;
        let move_mask = ((1u64 << b_end) - 1) & !lo_keep;
        let base = c_start * block_bytes;
        let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS);
        let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS);
        write_u64_le(
            blocks,
            base + BLOCK_RUNEND_POS,
            (re & keep) | ((re & move_mask) << 1),
        );
        write_u64_le(
            blocks,
            base + BLOCK_EXTENSION_POS,
            (ext & keep) | ((ext & move_mask) << 1),
        );
        return;
    }

    // Trailing chunk `c_end`.
    let hi_keep = if b_end == 63 {
        0
    } else {
        !((1u64 << (b_end + 1)) - 1)
    };
    let end_move_mask = (1u64 << b_end) - 1;
    let end_base = c_end * block_bytes;
    let prev_base = (c_end - 1) * block_bytes;
    let prev_re_carry = read_u64_le(blocks, prev_base + BLOCK_RUNEND_POS) >> 63;
    let prev_ext_carry = read_u64_le(blocks, prev_base + BLOCK_EXTENSION_POS) >> 63;
    let end_re = read_u64_le(blocks, end_base + BLOCK_RUNEND_POS);
    let end_ext = read_u64_le(blocks, end_base + BLOCK_EXTENSION_POS);
    write_u64_le(
        blocks,
        end_base + BLOCK_RUNEND_POS,
        (end_re & hi_keep) | ((end_re & end_move_mask) << 1) | prev_re_carry,
    );
    write_u64_le(
        blocks,
        end_base + BLOCK_EXTENSION_POS,
        (end_ext & hi_keep) | ((end_ext & end_move_mask) << 1) | prev_ext_carry,
    );

    // Intermediate full chunks.
    for c in ((c_start + 1)..c_end).rev() {
        let cur_base = c * block_bytes;
        let p_base = (c - 1) * block_bytes;
        let re_carry = read_u64_le(blocks, p_base + BLOCK_RUNEND_POS) >> 63;
        let ext_carry = read_u64_le(blocks, p_base + BLOCK_EXTENSION_POS) >> 63;
        let cur_re = read_u64_le(blocks, cur_base + BLOCK_RUNEND_POS);
        let cur_ext = read_u64_le(blocks, cur_base + BLOCK_EXTENSION_POS);
        write_u64_le(
            blocks,
            cur_base + BLOCK_RUNEND_POS,
            (cur_re << 1) | re_carry,
        );
        write_u64_le(
            blocks,
            cur_base + BLOCK_EXTENSION_POS,
            (cur_ext << 1) | ext_carry,
        );
    }

    // Starting chunk `c_start`.
    let lo_keep = (1u64 << b_start) - 1;
    let start_move_mask = !lo_keep;
    let start_base = c_start * block_bytes;
    let start_re = read_u64_le(blocks, start_base + BLOCK_RUNEND_POS);
    let start_ext = read_u64_le(blocks, start_base + BLOCK_EXTENSION_POS);
    write_u64_le(
        blocks,
        start_base + BLOCK_RUNEND_POS,
        (start_re & lo_keep) | ((start_re & start_move_mask) << 1),
    );
    write_u64_le(
        blocks,
        start_base + BLOCK_EXTENSION_POS,
        (start_ext & lo_keep) | ((start_ext & start_move_mask) << 1),
    );
}

/// Shifts both `runend` and `extension` bitmaps across AoS blocks for `(start ..= end]` ->
/// `[start .. end)`, clearing bit `end`.
fn shift_metadata_left_one(blocks: &mut [u8], block_bytes: usize, start: usize, end: usize) {
    if start == end {
        let base = (end >> SLOTS_PER_CHUNK_LOG2) * block_bytes;
        let bit = !(1u64 << (end & 63));
        let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS) & bit;
        let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS) & bit;
        write_u64_le(blocks, base + BLOCK_RUNEND_POS, re);
        write_u64_le(blocks, base + BLOCK_EXTENSION_POS, ext);
        return;
    }

    let c_start = start >> SLOTS_PER_CHUNK_LOG2;
    let b_start = (start & 63) as u32;
    let c_end = end >> SLOTS_PER_CHUNK_LOG2;
    let b_end = (end & 63) as u32;

    if c_start == c_end {
        let lo_keep = (1u64 << b_start) - 1;
        let hi_keep = if b_end == 63 {
            0
        } else {
            !((1u64 << (b_end + 1)) - 1)
        };
        let keep = lo_keep | hi_keep;
        let up_to_end = u64::MAX >> (63 - b_end);
        let up_to_start = u64::MAX >> (63 - b_start);
        let move_mask = up_to_end & !up_to_start;
        let base = c_start * block_bytes;
        let re = read_u64_le(blocks, base + BLOCK_RUNEND_POS);
        let ext = read_u64_le(blocks, base + BLOCK_EXTENSION_POS);
        write_u64_le(
            blocks,
            base + BLOCK_RUNEND_POS,
            (re & keep) | ((re & move_mask) >> 1),
        );
        write_u64_le(
            blocks,
            base + BLOCK_EXTENSION_POS,
            (ext & keep) | ((ext & move_mask) >> 1),
        );
        return;
    }

    // Starting chunk `c_start`.
    let lo_keep = (1u64 << b_start) - 1;
    let start_move_mask = !(u64::MAX >> (63 - b_start));
    let start_base = c_start * block_bytes;
    let next_base = (c_start + 1) * block_bytes;
    let next_re_carry = (read_u64_le(blocks, next_base + BLOCK_RUNEND_POS) & 1) << 63;
    let next_ext_carry = (read_u64_le(blocks, next_base + BLOCK_EXTENSION_POS) & 1) << 63;
    let start_re = read_u64_le(blocks, start_base + BLOCK_RUNEND_POS);
    let start_ext = read_u64_le(blocks, start_base + BLOCK_EXTENSION_POS);
    write_u64_le(
        blocks,
        start_base + BLOCK_RUNEND_POS,
        (start_re & lo_keep) | ((start_re & start_move_mask) >> 1) | next_re_carry,
    );
    write_u64_le(
        blocks,
        start_base + BLOCK_EXTENSION_POS,
        (start_ext & lo_keep) | ((start_ext & start_move_mask) >> 1) | next_ext_carry,
    );

    // Intermediate full chunks.
    for c in (c_start + 1)..c_end {
        let cur_base = c * block_bytes;
        let n_base = (c + 1) * block_bytes;
        let re_carry = (read_u64_le(blocks, n_base + BLOCK_RUNEND_POS) & 1) << 63;
        let ext_carry = (read_u64_le(blocks, n_base + BLOCK_EXTENSION_POS) & 1) << 63;
        let cur_re = read_u64_le(blocks, cur_base + BLOCK_RUNEND_POS);
        let cur_ext = read_u64_le(blocks, cur_base + BLOCK_EXTENSION_POS);
        write_u64_le(
            blocks,
            cur_base + BLOCK_RUNEND_POS,
            (cur_re >> 1) | re_carry,
        );
        write_u64_le(
            blocks,
            cur_base + BLOCK_EXTENSION_POS,
            (cur_ext >> 1) | ext_carry,
        );
    }

    // Trailing chunk `c_end`.
    let hi_keep = if b_end == 63 {
        0
    } else {
        !((1u64 << (b_end + 1)) - 1)
    };
    let end_move_mask = (u64::MAX >> (63 - b_end)) & !1u64;
    let end_base = c_end * block_bytes;
    let end_re = read_u64_le(blocks, end_base + BLOCK_RUNEND_POS);
    let end_ext = read_u64_le(blocks, end_base + BLOCK_EXTENSION_POS);
    write_u64_le(
        blocks,
        end_base + BLOCK_RUNEND_POS,
        (end_re & hi_keep) | ((end_re & end_move_mask) >> 1),
    );
    write_u64_le(
        blocks,
        end_base + BLOCK_EXTENSION_POS,
        (end_ext & hi_keep) | ((end_ext & end_move_mask) >> 1),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn metadata_shift_matches_naive_bit_reference_across_boundaries() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC40C_0001);
        let num_chunks = 5;
        let total_bits = num_chunks * 64;
        let slot_bits = 13usize;
        let block_bytes = BLOCK_HEADER_BYTES + 8 * slot_bits;

        let boundary_pairs = [
            (0, 0),
            (0, 1),
            (0, 63),
            (63, 63),
            (63, 64),
            (0, 64),
            (0, 128),
            (62, 129),
            (63, 192),
            (1, 255),
            (64, 127),
            (64, 128),
            (127, 191),
            (0, total_bits - 1),
        ];

        for &(start, end) in &boundary_pairs {
            let mut blocks = vec![0u8; num_chunks * block_bytes + TRAILING_GUARD_BYTES];
            let mut init_offsets = vec![0u8; num_chunks];
            let mut init_occupied = vec![0u64; num_chunks];
            let mut init_payloads = vec![vec![0u8; 8 * slot_bits]; num_chunks];

            for c in 0..num_chunks {
                let base = c * block_bytes;
                init_offsets[c] = rng.gen();
                init_occupied[c] = rng.next_u64();
                blocks[base + BLOCK_OFFSET_POS] = init_offsets[c];
                write_u64_le(&mut blocks, base + BLOCK_OCCUPIED_POS, init_occupied[c]);
                write_u64_le(&mut blocks, base + BLOCK_RUNEND_POS, rng.next_u64());
                write_u64_le(&mut blocks, base + BLOCK_EXTENSION_POS, rng.next_u64());
                rng.fill_bytes(&mut init_payloads[c]);
                blocks[(base + BLOCK_HEADER_BYTES)..(base + block_bytes)]
                    .copy_from_slice(&init_payloads[c]);
            }

            // Reference right shift.
            let mut ref_re: Vec<bool> = (0..total_bits)
                .map(|i| {
                    let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_RUNEND_POS);
                    ((w >> (i % 64)) & 1) != 0
                })
                .collect();
            let mut ref_ext: Vec<bool> = (0..total_bits)
                .map(|i| {
                    let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_EXTENSION_POS);
                    ((w >> (i % 64)) & 1) != 0
                })
                .collect();

            let mut right_blocks = blocks.clone();
            shift_metadata_right_one(&mut right_blocks, block_bytes, start, end);
            for i in ((start + 1)..=end).rev() {
                ref_re[i] = ref_re[i - 1];
                ref_ext[i] = ref_ext[i - 1];
            }
            ref_re[start] = false;
            ref_ext[start] = false;

            for i in 0..total_bits {
                let base = (i / 64) * block_bytes;
                let got_re =
                    ((read_u64_le(&right_blocks, base + BLOCK_RUNEND_POS) >> (i % 64)) & 1) != 0;
                let got_ext =
                    ((read_u64_le(&right_blocks, base + BLOCK_EXTENSION_POS) >> (i % 64)) & 1) != 0;
                assert_eq!(
                    got_re, ref_re[i],
                    "right shift runend at {i} for ({start},{end})"
                );
                assert_eq!(
                    got_ext, ref_ext[i],
                    "right shift ext at {i} for ({start},{end})"
                );
            }
            // Verify offset, occupied, payloads, and guard bytes are untouched.
            for c in 0..num_chunks {
                let base = c * block_bytes;
                assert_eq!(right_blocks[base + BLOCK_OFFSET_POS], init_offsets[c]);
                assert_eq!(
                    read_u64_le(&right_blocks, base + BLOCK_OCCUPIED_POS),
                    init_occupied[c]
                );
                assert_eq!(
                    &right_blocks[(base + BLOCK_HEADER_BYTES)..(base + block_bytes)],
                    init_payloads[c].as_slice()
                );
            }
            assert!(right_blocks[(num_chunks * block_bytes)..]
                .iter()
                .all(|&b| b == 0));

            // Reference left shift.
            let mut ref_re: Vec<bool> = (0..total_bits)
                .map(|i| {
                    let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_RUNEND_POS);
                    ((w >> (i % 64)) & 1) != 0
                })
                .collect();
            let mut ref_ext: Vec<bool> = (0..total_bits)
                .map(|i| {
                    let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_EXTENSION_POS);
                    ((w >> (i % 64)) & 1) != 0
                })
                .collect();

            let mut left_blocks = blocks.clone();
            shift_metadata_left_one(&mut left_blocks, block_bytes, start, end);
            for i in start..end {
                ref_re[i] = ref_re[i + 1];
                ref_ext[i] = ref_ext[i + 1];
            }
            ref_re[end] = false;
            ref_ext[end] = false;

            for i in 0..total_bits {
                let base = (i / 64) * block_bytes;
                let got_re =
                    ((read_u64_le(&left_blocks, base + BLOCK_RUNEND_POS) >> (i % 64)) & 1) != 0;
                let got_ext =
                    ((read_u64_le(&left_blocks, base + BLOCK_EXTENSION_POS) >> (i % 64)) & 1) != 0;
                assert_eq!(
                    got_re, ref_re[i],
                    "left shift runend at {i} for ({start},{end})"
                );
                assert_eq!(
                    got_ext, ref_ext[i],
                    "left shift ext at {i} for ({start},{end})"
                );
            }
            for c in 0..num_chunks {
                let base = c * block_bytes;
                assert_eq!(left_blocks[base + BLOCK_OFFSET_POS], init_offsets[c]);
                assert_eq!(
                    read_u64_le(&left_blocks, base + BLOCK_OCCUPIED_POS),
                    init_occupied[c]
                );
                assert_eq!(
                    &left_blocks[(base + BLOCK_HEADER_BYTES)..(base + block_bytes)],
                    init_payloads[c].as_slice()
                );
            }
            assert!(left_blocks[(num_chunks * block_bytes)..]
                .iter()
                .all(|&b| b == 0));
        }
    }

    #[test]
    fn payload_round_trip_across_edge_slot_bits_without_metadata_bleed() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xA05B_0001);
        let edge_widths = [1u8, 2, 7, 10, 13, 18, 32, 57, 58, 63, 64];

        for slot_bits in edge_widths {
            let mut table = QuotientTable::new(256, slot_bits);
            let total_slots = table.physical_slots();
            let total_chunks = table.total_chunks();
            let expected_block_bytes = BLOCK_HEADER_BYTES + 8 * (slot_bits as usize);
            assert_eq!(table.block_bytes, expected_block_bytes);
            assert_eq!(
                table.memory_usage_bytes(),
                total_chunks * expected_block_bytes + TRAILING_GUARD_BYTES
            );

            let mask = if slot_bits == 64 {
                u64::MAX
            } else {
                (1u64 << slot_bits) - 1
            };

            // Seed non-zero metadata headers on every block to catch any payload-to-metadata bleed.
            let mut exp_offsets = Vec::with_capacity(total_chunks);
            let mut exp_occupied = Vec::with_capacity(total_chunks);
            let mut exp_runend = Vec::with_capacity(total_chunks);
            let mut exp_extension = Vec::with_capacity(total_chunks);

            for c in 0..total_chunks {
                let off = rng.gen_range(1u8..250u8);
                let occ = rng.next_u64() | 0x8000_0000_0000_0001;
                let re = rng.next_u64() | 0xA5A5_5A5A_0000_0001;
                let ext = rng.next_u64() | 0x5A5A_A5A5_8000_0000;
                let base = c * table.block_bytes;
                table.blocks[base + BLOCK_OFFSET_POS] = off;
                write_u64_le(&mut table.blocks, base + BLOCK_OCCUPIED_POS, occ);
                write_u64_le(&mut table.blocks, base + BLOCK_RUNEND_POS, re);
                write_u64_le(&mut table.blocks, base + BLOCK_EXTENSION_POS, ext);
                exp_offsets.push(off);
                exp_occupied.push(occ);
                exp_runend.push(re);
                exp_extension.push(ext);
            }

            // Pass 1: forward write across ALL physical slots (including slot 63 of the final block).
            let mut expected = vec![0u64; total_slots];
            for (slot, exp) in expected.iter_mut().enumerate() {
                let raw = if slot % 17 == 0 {
                    u64::MAX
                } else if slot % 19 == 0 {
                    0
                } else {
                    rng.next_u64()
                };
                table.set_payload(slot, raw);
                *exp = raw & mask;
            }

            for (slot, &exp) in expected.iter().enumerate() {
                let got = table.get_payload(slot);
                assert_eq!(got, exp, "pass1 width={slot_bits} slot={slot}");
                let (got2, kind) = table.get_payload_and_kind(slot);
                assert_eq!(got2, exp, "pass1 co-located width={slot_bits} slot={slot}");
                assert_eq!(kind, table.slot_kind(slot));
            }

            // Pass 2: reverse overwrite pass to ensure zero neighbor bit bleed in either direction.
            for (slot, exp) in expected.iter_mut().enumerate().rev() {
                let raw = rng.next_u64();
                table.set_payload(slot, raw);
                *exp = raw & mask;
            }

            for (slot, &exp) in expected.iter().enumerate() {
                let (got, kind) = table.get_payload_and_kind(slot);
                assert_eq!(got, exp, "pass2 width={slot_bits} slot={slot}");
                assert_eq!(kind, table.slot_kind(slot));
            }

            // Verify every block's metadata headers AND trailing guard bytes are 100% untouched.
            for c in 0..total_chunks {
                let base = c * table.block_bytes;
                assert_eq!(
                    table.blocks[base + BLOCK_OFFSET_POS],
                    exp_offsets[c],
                    "offset bled width={slot_bits} chunk={c}"
                );
                assert_eq!(
                    read_u64_le(&table.blocks, base + BLOCK_OCCUPIED_POS),
                    exp_occupied[c],
                    "occupied bled width={slot_bits} chunk={c}"
                );
                assert_eq!(
                    read_u64_le(&table.blocks, base + BLOCK_RUNEND_POS),
                    exp_runend[c],
                    "runend bled width={slot_bits} chunk={c}"
                );
                assert_eq!(
                    read_u64_le(&table.blocks, base + BLOCK_EXTENSION_POS),
                    exp_extension[c],
                    "extension bled width={slot_bits} chunk={c}"
                );
            }
            let guard_start = total_chunks * table.block_bytes;
            assert!(
                table.blocks[guard_start..].iter().all(|&b| b == 0),
                "guard bytes corrupted for width={slot_bits}"
            );
        }
    }

    #[test]
    fn payload_shift_right_and_left_across_edge_slot_bits_and_blocks() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xA05B_0002);
        let edge_widths = [1u8, 2, 7, 10, 13, 18, 32, 57, 58, 63, 64];
        let pairs = [
            (0usize, 0usize),
            (0, 1),
            (1, 2),
            (10, 13),
            (5, 20),
            (60, 63),
            (62, 63),
            (0, 63),
            (63, 64),
            (10, 135),
            (0, 255),
            (62, 259),
        ];

        for slot_bits in edge_widths {
            let mask = if slot_bits == 64 {
                u64::MAX
            } else {
                (1u64 << slot_bits) - 1
            };

            for &(start, end) in &pairs {
                let mut table = QuotientTable::new(256, slot_bits);
                let total_slots = table.physical_slots();
                let mut model = vec![0u64; total_slots];
                for (i, m) in model.iter_mut().enumerate() {
                    let v = rng.next_u64() & mask;
                    table.set_payload(i, v);
                    *m = v;
                }

                // Shift right: `[start .. end)` -> `[start + 1 .. end + 1)`, slot `start` zeroed.
                table.shift_payloads_right_one(start, end);
                for i in ((start + 1)..=end).rev() {
                    model[i] = model[i - 1];
                }
                model[start] = 0;
                for (i, &exp) in model.iter().enumerate() {
                    assert_eq!(
                        table.get_payload(i),
                        exp,
                        "shift_right width={slot_bits} pair=({start},{end}) slot={i}"
                    );
                }

                // Shift left: `(start ..= end]` -> `[start .. end)`, slot `end` zeroed.
                table.shift_payloads_left_one(start, end);
                for i in start..end {
                    model[i] = model[i + 1];
                }
                model[end] = 0;
                for (i, &exp) in model.iter().enumerate() {
                    assert_eq!(
                        table.get_payload(i),
                        exp,
                        "shift_left width={slot_bits} pair=({start},{end}) slot={i}"
                    );
                }

                let guard_start = table.total_chunks() * table.block_bytes;
                assert!(table.blocks[guard_start..].iter().all(|&b| b == 0));
            }
        }
    }

    #[test]
    fn exact_memory_usage_and_bits_per_key_at_90_percent_load_factor() {
        let q = 20u8;
        let n = 1usize << q; // 1,048,576 canonical slots
        let keys_90_pct = (n as f64) * 0.90;
        let r = 7u8;

        for (m, max_bpk) in [(2u8, 14.70f64), (5u8, 18.00f64), (10u8, 23.60f64)] {
            // Preserve the 1-bit InfiniFilter unary age sentinel: `slot_bits = r + 1 + m`.
            let slot_bits = r + 1 + m;
            let table = QuotientTable::new(n, slot_bits);
            let expected_block_bytes = 25 + 8 * (slot_bits as usize);
            let expected_bytes = table.total_chunks() * expected_block_bytes + TRAILING_GUARD_BYTES;
            assert_eq!(table.memory_usage_bytes(), expected_bytes);

            let bpk = (table.memory_usage_bytes() as f64) * 8.0 / keys_90_pct;
            assert!(
                bpk <= max_bpk,
                "r={r}, m={m} (slot_bits={slot_bits}): bpk={bpk:.4} exceeded {max_bpk:.2}"
            );
            if m == 5 {
                // Confirm <= 1.25 bpk above C++ baseline's 16.85 bpk while keeping the 1-bit age sentinel.
                assert!(
                    bpk - 16.85 <= 1.25,
                    "m=5 delta over C++ 16.85 bpk was {:.4} > 1.25",
                    bpk - 16.85
                );
            }
        }
    }

    #[test]
    fn random_run_edits_preserve_table_invariants_and_handle_offset_overflow() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC40C_0002);
        let n = 128;
        let mut table = QuotientTable::new(n, 20);
        let mut model: BTreeMap<usize, Vec<(u64, KeepsakeSlotKind)>> = BTreeMap::new();

        // Force chunk-1 offset overflow (> 255) by building a 300-slot run at home 63!
        let mut big_run = Vec::with_capacity(300);
        for i in 0..299 {
            big_run.push((i as u64 + 1, KeepsakeSlotKind::PayloadInterior));
        }
        big_run.push((300, KeepsakeSlotKind::BoxAndRunEnd));
        table.replace_region(63, 0, 0, &big_run);
        model.insert(63, big_run);
        assert!(
            table.chunk_offset(1) >= 255,
            "expected offset overflow on chunk 1"
        );
        assert!(table.verify_metadata_invariants().is_ok());

        // Perform randomized run creations, expansions, contractions, and deletions.
        for _ in 0..400 {
            let home = rng.gen_range(0..n);
            let op = rng.gen_range(0..4);
            let existing = model.get(&home).cloned();
            match (existing, op) {
                (None, _) => {
                    let len = rng.gen_range(1..=6);
                    let mut slots: Vec<_> = (0..len)
                        .map(|i| {
                            let kind = if i + 1 == len {
                                KeepsakeSlotKind::BoxAndRunEnd
                            } else if i % 2 == 0 {
                                KeepsakeSlotKind::Extension
                            } else {
                                KeepsakeSlotKind::PayloadInterior
                            };
                            (rng.gen::<u64>() & ((1 << 20) - 1), kind)
                        })
                        .collect();
                    slots.last_mut().unwrap().1 = KeepsakeSlotKind::BoxAndRunEnd;
                    table.replace_region(home, 0, 0, &slots);
                    model.insert(home, slots);
                }
                (Some(old), 0) => {
                    // Delete entire run.
                    let (rs, re) = table.locate_run(home).unwrap();
                    assert_eq!(re - rs + 1, old.len());
                    table.replace_region(home, rs, old.len(), &[]);
                    model.remove(&home);
                }
                (Some(old), _) => {
                    // Replace entire run with new length (grow, shrink, or same).
                    let (rs, re) = table.locate_run(home).unwrap();
                    assert_eq!(re - rs + 1, old.len());
                    let new_len = rng.gen_range(1..=10);
                    let mut slots: Vec<_> = (0..new_len)
                        .map(|i| {
                            let kind = if i + 1 == new_len {
                                KeepsakeSlotKind::BoxAndRunEnd
                            } else {
                                KeepsakeSlotKind::BoxEnd
                            };
                            (rng.gen::<u64>() & ((1 << 20) - 1), kind)
                        })
                        .collect();
                    slots.last_mut().unwrap().1 = KeepsakeSlotKind::BoxAndRunEnd;
                    table.replace_region(home, rs, old.len(), &slots);
                    model.insert(home, slots);
                }
            }

            table.verify_metadata_invariants().unwrap();
            let occupied_list: Vec<usize> = table.occupied_homes().collect();
            let expected_homes: Vec<usize> = model.keys().copied().collect();
            assert_eq!(occupied_list, expected_homes);

            for (&h, expected_slots) in &model {
                let (rs, re) = table.locate_run(h).unwrap();
                assert_eq!(re - rs + 1, expected_slots.len());
                for (i, &(exp_p, exp_k)) in expected_slots.iter().enumerate() {
                    assert_eq!(table.get_payload(rs + i), exp_p);
                    assert_eq!(table.slot_kind(rs + i), exp_k);
                    assert_eq!(table.get_payload_and_kind(rs + i), (exp_p, exp_k));
                }
            }
        }
    }
}
