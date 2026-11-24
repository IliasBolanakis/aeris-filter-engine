//! Deterministic, extendable partition hash stream `h(p(k))`.
//!
//! Generates an arbitrarily long, MSB-first bit stream from `(seed, partition)`
//! with zero heap allocation on the query hot path. The first 64 bits (`block 0`)
//! are computed in a few integer instructions using a bijective SplitMix64 mixer,
//! and subsequent 64-bit blocks are derived deterministically on demand when
//! keepsake boxes are extended beyond 64 bits.

use crate::constants::DEFAULT_HASH_SEED;
use crate::range::PartitionId;
use serde::{Deserialize, Serialize};

/// Golden-ratio odd constant used to separate independent 64-bit hash blocks.
const BLOCK_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Secondary domain-separation constant for multi-word hashing.
const DOMAIN_MIX: u64 = 0xD1B5_4A32_D192_ED03;

/// Deterministic extendable hash generator for partition prefixes `p(k)`.
///
/// # Bit Ordering Invariant
/// Bits are indexed `0, 1, 2, ...` from most-significant bit (MSB) to
/// least-significant bit (LSB):
/// - Bit `0` is bit 63 (MSB) of `word(partition, 0)`.
/// - Bit `63` is bit 0 (LSB) of `word(partition, 0)`.
/// - Bit `64` is bit 63 (MSB) of `word(partition, 1)`.
///
/// Because every shorter fingerprint is an exact prefix of every longer
/// fingerprint, InfiniFilter expansion (`q -> q + 1`) and Aeris fingerprint
/// extension preserve prefix consistency without altering prior bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionHasher {
    seed: u64,
}

impl Default for PartitionHasher {
    #[inline]
    fn default() -> Self {
        Self::new(DEFAULT_HASH_SEED)
    }
}

impl PartitionHasher {
    /// Creates a new [`PartitionHasher`] with the provided 64-bit `seed`.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Returns the hash seed.
    #[inline]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Computes the `block_index`-th 64-bit word of the hash stream `h(partition)`.
    ///
    /// # Complexity
    /// `O(1)` time (~2 ns) and `O(1)` space, zero heap allocation.
    #[inline]
    pub const fn word(&self, partition: PartitionId, block_index: u32) -> u64 {
        let mixed = partition
            .wrapping_add(self.seed)
            .wrapping_add((block_index as u64).wrapping_mul(BLOCK_STRIDE));
        if block_index == 0 {
            splitmix64(mixed)
        } else {
            splitmix64(
                mixed ^ splitmix64((block_index as u64).wrapping_mul(DOMAIN_MIX) ^ self.seed),
            )
        }
    }

    /// Extracts `bit_len` consecutive MSB-first bits starting at `bit_offset`
    /// from `h(partition)`, returned right-aligned in the low `bit_len` bits of a `u64`.
    ///
    /// # Panics
    /// Panics in debug builds if `bit_len > 64`. If `bit_len == 0`, returns `0`.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space, zero heap allocation.
    #[inline]
    pub const fn extract_bits(&self, partition: PartitionId, bit_offset: u16, bit_len: u8) -> u64 {
        debug_assert!(bit_len <= 64);
        if bit_len == 0 {
            return 0;
        }
        let block = (bit_offset / 64) as u32;
        let inner_offset = (bit_offset % 64) as u8;
        let available_in_first = 64 - inner_offset;
        let w0 = self.word(partition, block);

        if bit_len <= available_in_first {
            // All requested bits reside within `w0`.
            let shift_right = available_in_first - bit_len;
            let shifted = w0 >> shift_right;
            if bit_len == 64 {
                shifted
            } else {
                shifted & ((1u64 << bit_len) - 1)
            }
        } else {
            // Spans `w0` and `w1`.
            let hi_bits = available_in_first;
            let lo_bits = bit_len - hi_bits;
            let hi_mask = if hi_bits == 64 {
                u64::MAX
            } else {
                (1u64 << hi_bits) - 1
            };
            let hi = w0 & hi_mask;
            let w1 = self.word(partition, block + 1);
            let lo = w1 >> (64 - lo_bits);
            (hi << lo_bits) | lo
        }
    }

    /// Fast hot-path extraction of `(quotient, remainder)` when `q + r <= 64`.
    ///
    /// Falls back cleanly to [`Self::extract_bits`] if `q + r > 64`.
    ///
    /// # Complexity
    /// `O(1)` time (single `word(partition, 0)` evaluation in the common case).
    #[inline]
    pub const fn quotient_and_remainder(&self, partition: PartitionId, q: u8, r: u8) -> (u64, u64) {
        if (q as u16) + (r as u16) <= 64 {
            let w0 = self.word(partition, 0);
            let quotient = if q == 0 {
                0
            } else if q == 64 {
                w0
            } else {
                w0 >> (64 - q)
            };
            let remainder = if r == 0 {
                0
            } else {
                let shift = 64 - (q + r);
                let mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
                (w0 >> shift) & mask
            };
            (quotient, remainder)
        } else {
            (
                self.extract_bits(partition, 0, q),
                self.extract_bits(partition, q as u16, r),
            )
        }
    }
}

/// Bijective 64-bit SplitMix64 avalanche finalizer.
#[inline]
const fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
