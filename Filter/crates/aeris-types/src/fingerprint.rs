//! Variable-length canonical fingerprints shared between Data Plane and Reverse Map.
//!
//! # Key Architectural Invariant
//! A [`Fingerprint`] stores the exact MSB-first prefix of `h(p(k))` of length
//! `bit_len = q + r_eff + ext_bits`. Because InfiniFilter expansion shifts one
//! bit from `r_eff` into `q`, the total length `q + r_eff + ext_bits` and every
//! bit value remain **strictly invariant** across non-rejuvenating expansions.
//! Consequently, Reverse Map entries require zero rewrites during non-rejuvenating
//! expansions and can be retrieved identically during adaptation or selective
//! rejuvenation.

use crate::error::{AerisError, AerisResult};
use crate::hash::PartitionHasher;
use crate::range::PartitionId;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Canonical variable-length fingerprint representing the top `bit_len` MSB bits
/// of `h(p(k))`.
///
/// Up to 128 bits are stored completely inline (`head: [u64; 2]`) with zero heap
/// allocations; extensions beyond 128 bits spill transparently into `tail`.
///
/// # Invariants
/// - All trailing bits beyond `bit_len` are strictly zeroed, ensuring canonical
///   equality, ordering, hashing, and byte serialization.
/// - `tail` contains only the minimal number of extra 64-bit words required beyond
///   128 bits (`tail.is_empty()` whenever `bit_len <= 128`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fingerprint {
    bit_len: u16,
    head: [u64; 2],
    tail: Vec<u64>,
}

impl Fingerprint {
    /// Constructs an empty (0-bit) fingerprint.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            bit_len: 0,
            head: [0, 0],
            tail: Vec::new(),
        }
    }

    /// Constructs a canonical `bit_len`-bit (`0..=64`) [`Fingerprint`] directly
    /// from the top `bit_len` MSB bits of a precomputed 64-bit hash word `word0`
    /// with zero panics and zero heap allocations.
    ///
    /// Bit-for-bit identical to [`Self::from_parts`] for any `(q, r)` with
    /// `q + r <= 64` and empty extensions, while allowing LLVM to dead-code
    /// eliminate unused return values on hot insert paths.
    #[inline]
    pub const fn from_u64_prefix(word0: u64, bit_len: u8) -> Self {
        let masked = if bit_len == 0 {
            0
        } else if bit_len >= 64 {
            word0
        } else {
            word0 & !((1u64 << (64 - bit_len)) - 1)
        };
        Self {
            bit_len: bit_len as u16,
            head: [masked, 0],
            tail: Vec::new(),
        }
    }

    /// Constructs a canonical `bit_len`-bit fingerprint directly from `hasher`
    /// and `partition`.
    ///
    /// # Complexity
    /// `O(ceil(bit_len / 64))` time; zero heap allocation for `bit_len <= 128`.
    pub fn from_hasher(hasher: &PartitionHasher, partition: PartitionId, bit_len: u16) -> Self {
        if bit_len == 0 {
            return Self::empty();
        }
        let num_words = (bit_len as usize).div_ceil(64);
        let mut head = [0u64; 2];
        let mut tail = if num_words > 2 {
            Vec::with_capacity(num_words - 2)
        } else {
            Vec::new()
        };

        for (w_idx, slot) in head.iter_mut().enumerate().take(num_words) {
            let mut w = hasher.word(partition, w_idx as u32);
            let bits_before = w_idx * 64;
            let remaining = (bit_len as usize) - bits_before;
            if remaining < 64 {
                let mask = !((1u64 << (64 - remaining)) - 1);
                w &= mask;
            }
            *slot = w;
        }
        for w_idx in 2..num_words {
            let mut w = hasher.word(partition, w_idx as u32);
            let bits_before = w_idx * 64;
            let remaining = (bit_len as usize) - bits_before;
            if remaining < 64 {
                // Keep only top `remaining` MSB bits; zero trailing bits.
                let mask = !((1u64 << (64 - remaining)) - 1);
                w &= mask;
            }
            tail.push(w);
        }

        Self {
            bit_len,
            head,
            tail,
        }
    }

    /// Appends `len` right-aligned bits (`1 <= len <= 64`) to the MSB-first bit stream
    /// of this fingerprint.
    ///
    /// # Panics
    /// Panics if `len > 64` or if `self.bit_len as usize + len as usize > u16::MAX as usize`.
    pub fn push_bits(&mut self, bits: u64, len: u8) {
        assert!(len <= 64, "cannot push more than 64 bits at once");
        if len == 0 {
            return;
        }
        let new_len_usize = (self.bit_len as usize)
            .checked_add(len as usize)
            .expect("fingerprint bit length overflow");
        assert!(
            new_len_usize <= u16::MAX as usize,
            "fingerprint exceeds u16::MAX bits"
        );

        let masked = if len == 64 {
            bits
        } else {
            bits & ((1u64 << len) - 1)
        };

        let start_word = (self.bit_len / 64) as usize;
        let inner_offset = (self.bit_len % 64) as u8;
        let space_in_word = 64 - inner_offset;

        if len <= space_in_word {
            let shift = space_in_word - len;
            self.or_word(start_word, masked << shift);
        } else {
            let hi_len = space_in_word;
            let lo_len = len - hi_len;
            let hi = masked >> lo_len;
            let lo = masked << (64 - lo_len);
            self.or_word(start_word, hi);
            self.or_word(start_word + 1, lo);
        }

        self.bit_len = new_len_usize as u16;
    }

    /// Builds a canonical [`Fingerprint`] from Quotient Filter slot components:
    /// `(quotient, q_bits)`, `(remainder, rem_bits)`, and an ordered slice of
    /// `(extension_bits, ext_bit_len)` segments.
    ///
    /// # Complexity
    /// `O(ext_segments.len())` time; zero heap allocation up to 128 total bits.
    pub fn from_parts(
        quotient: u64,
        q_bits: u8,
        remainder: u64,
        rem_bits: u8,
        ext_segments: &[(u64, u8)],
    ) -> Self {
        let mut fp = Self::empty();
        fp.push_bits(quotient, q_bits);
        fp.push_bits(remainder, rem_bits);
        for &(ext_val, ext_len) in ext_segments {
            fp.push_bits(ext_val, ext_len);
        }
        fp
    }

    /// Returns the total length of this fingerprint in bits (`q + r_eff + ext_bits`).
    #[inline]
    pub const fn bit_len(&self) -> u16 {
        self.bit_len
    }

    /// Extracts `len` consecutive MSB-first bits starting at `bit_offset`,
    /// returned right-aligned in the low `len` bits of a `u64`.
    ///
    /// # Panics
    /// Panics if `len > 64` or `bit_offset as usize + len as usize > self.bit_len as usize`.
    pub fn extract_bits(&self, bit_offset: u16, len: u8) -> u64 {
        assert!(len <= 64, "cannot extract more than 64 bits");
        assert!(
            (bit_offset as usize) + (len as usize) <= (self.bit_len as usize),
            "bit slice out of fingerprint bounds"
        );
        if len == 0 {
            return 0;
        }
        let block = (bit_offset / 64) as usize;
        let inner_offset = (bit_offset % 64) as u8;
        let avail = 64 - inner_offset;
        let w0 = self.get_word(block);

        if len <= avail {
            let shifted = w0 >> (avail - len);
            if len == 64 {
                shifted
            } else {
                shifted & ((1u64 << len) - 1)
            }
        } else {
            let hi_bits = avail;
            let lo_bits = len - hi_bits;
            let hi_mask = if hi_bits == 64 {
                u64::MAX
            } else {
                (1u64 << hi_bits) - 1
            };
            let hi = w0 & hi_mask;
            let w1 = self.get_word(block + 1);
            let lo = w1 >> (64 - lo_bits);
            (hi << lo_bits) | lo
        }
    }

    /// Returns `true` if `self` is an exact bit-prefix of `longer`
    /// (requiring `self.bit_len <= longer.bit_len`).
    pub fn is_prefix_of(&self, longer: &Self) -> bool {
        if self.bit_len > longer.bit_len {
            return false;
        }
        if self.bit_len == 0 {
            return true;
        }
        let full_words = (self.bit_len / 64) as usize;
        for i in 0..full_words {
            if self.get_word(i) != longer.get_word(i) {
                return false;
            }
        }
        let rem_bits = (self.bit_len % 64) as u8;
        if rem_bits > 0 {
            let mask = !((1u64 << (64 - rem_bits)) - 1);
            if (self.get_word(full_words) & mask) != (longer.get_word(full_words) & mask) {
                return false;
            }
        }
        true
    }

    /// Encodes the fingerprint into a canonical, self-delimiting byte vector:
    /// `[bit_len: u16 big-endian][ceil(bit_len / 8) MSB-first payload bytes]`.
    ///
    /// Used as the storage key in RocksDB (`ReverseMapStore`) and in WAL frames.
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload_bytes = (self.bit_len as usize).div_ceil(8);
        let mut out = Vec::with_capacity(2 + payload_bytes);
        out.extend_from_slice(&self.bit_len.to_be_bytes());
        let mut written = 0usize;
        let mut word_idx = 0usize;
        while written < payload_bytes {
            let word_bytes = self.get_word(word_idx).to_be_bytes();
            let take = (payload_bytes - written).min(8);
            out.extend_from_slice(&word_bytes[..take]);
            written += take;
            word_idx += 1;
        }
        out
    }

    /// Decodes a [`Fingerprint`] from bytes previously produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`AerisError::EncodingError`] if the byte slice is truncated,
    /// has trailing garbage bytes, or has non-zero trailing padding bits.
    pub fn from_bytes(bytes: &[u8]) -> AerisResult<Self> {
        if bytes.len() < 2 {
            return Err(AerisError::EncodingError(
                "fingerprint buffer shorter than 2-byte length prefix".to_string(),
            ));
        }
        let bit_len = u16::from_be_bytes([bytes[0], bytes[1]]);
        let payload_bytes = (bit_len as usize).div_ceil(8);
        if bytes.len() != 2 + payload_bytes {
            return Err(AerisError::EncodingError(format!(
                "fingerprint length mismatch: expected {} bytes for {} bits, got {}",
                2 + payload_bytes,
                bit_len,
                bytes.len()
            )));
        }
        if bit_len == 0 {
            return Ok(Self::empty());
        }

        // Verify trailing padding bits in the final byte are zeroed.
        let rem_bits_in_last_byte = (bit_len % 8) as u8;
        if rem_bits_in_last_byte != 0 {
            let last_byte = bytes[2 + payload_bytes - 1];
            let pad_mask = (1u8 << (8 - rem_bits_in_last_byte)) - 1;
            if (last_byte & pad_mask) != 0 {
                return Err(AerisError::EncodingError(
                    "fingerprint contains non-zero trailing padding bits".to_string(),
                ));
            }
        }

        let num_words = (bit_len as usize).div_ceil(64);
        let mut head = [0u64; 2];
        let mut tail = if num_words > 2 {
            Vec::with_capacity(num_words - 2)
        } else {
            Vec::new()
        };

        let payload = &bytes[2..];
        for (w_idx, slot) in head.iter_mut().enumerate().take(num_words) {
            let mut buf = [0u8; 8];
            let start = w_idx * 8;
            let end = (start + 8).min(payload.len());
            buf[..(end - start)].copy_from_slice(&payload[start..end]);
            *slot = u64::from_be_bytes(buf);
        }
        for w_idx in 2..num_words {
            let mut buf = [0u8; 8];
            let start = w_idx * 8;
            let end = (start + 8).min(payload.len());
            buf[..(end - start)].copy_from_slice(&payload[start..end]);
            tail.push(u64::from_be_bytes(buf));
        }

        Ok(Self {
            bit_len,
            head,
            tail,
        })
    }

    #[inline]
    fn get_word(&self, idx: usize) -> u64 {
        if idx < 2 {
            self.head[idx]
        } else {
            self.tail.get(idx - 2).copied().unwrap_or(0)
        }
    }

    #[inline]
    fn or_word(&mut self, idx: usize, val: u64) {
        if idx < 2 {
            self.head[idx] |= val;
        } else {
            let tail_idx = idx - 2;
            if self.tail.len() <= tail_idx {
                self.tail.resize(tail_idx + 1, 0);
            }
            self.tail[tail_idx] |= val;
        }
    }
}

impl PartialEq for Fingerprint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bit_len == other.bit_len && self.head == other.head && self.tail == other.tail
    }
}

impl Eq for Fingerprint {}

impl PartialOrd for Fingerprint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fingerprint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.bit_len
            .cmp(&other.bit_len)
            .then_with(|| self.head.cmp(&other.head))
            .then_with(|| self.tail.cmp(&other.tail))
    }
}

impl Hash for Fingerprint {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bit_len.hash(state);
        self.head.hash(state);
        self.tail.hash(state);
    }
}
