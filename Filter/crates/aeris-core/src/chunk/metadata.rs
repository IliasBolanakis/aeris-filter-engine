//! 64-slot Quotient Filter chunk metadata value type.

use aeris_types::KeepsakeSlotKind;

/// Metadata block governing 64 contiguous Quotient Filter slots.
///
/// Packs three 64-bit bitmaps (`occupied`, `runend`, `extension`) and an 8-bit
/// inline offset (`25 bytes` logical), enabling `O(1)` SIMD `POPCNT`/`PDEP`
/// run navigation. Retained as a public value type for downstream callers and
/// inspection utilities; [`super::QuotientTable`] stores these exact fields
/// interleaved with each block's 64 bit-packed payloads in a single zero-padding
/// byte array.
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
    /// A value of [`aeris_types::MAX_CHUNK_OFFSET`] (`255`) signals an exact `u32` entry in
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
