//! Keepsake box encoding, InfiniFilter unary age counters, and zero-allocation
//! hot-path query matching.
//!
//! # Bit-Layout Invariants
//! Every physical slot has width `slot_bits = r + 1 + m` (`<= 64` bits):
//! - **Slot 0 high `r + 1` bits:** Unary age counter followed by `r - age` hash
//!   remainder bits (`age` leading `0`s, sentinel `1`, then `r - age` bits).
//!   When `age == r`, `0` remainder bits remain (`high == 1`), triggering
//!   selective rejuvenation on the next `2x` expansion.
//! - **3-Bit Metadata Encoding ([`KeepsakeSlotKind`]):**
//!   - `Extension` (`runend = 0, extension = 1`): holds extension bits and is
//!     **never** the terminal slot of a keepsake box.
//!   - `PayloadInterior` (`runend = 0, extension = 0`): non-terminal payload or
//!     memento continuation slot.
//!   - `BoxEnd` (`runend = 1, extension = 0`): terminates a keepsake box inside
//!     a quotient run.
//!   - `BoxAndRunEnd` (`runend = 1, extension = 1`): terminates both a keepsake
//!     box and its enclosing quotient run (`qrun_end = runend & extension`).
//! - **Mementos (`m > 0`):** Stored in sorted ascending order as a multiset
//!   (preserving duplicates from hash-colliding partitions). Continuation slots
//!   pack `B >= 1` mementos alongside an exact count header `(count - 1)` so
//!   memento `0` and duplicate values have zero ambiguity.
//! - **Point Mode (`m == 0`):** Unextended multiplicity-1 boxes occupy 1 slot;
//!   multiplicity `>= 2` or extended boxes terminate with a non-extension slot
//!   carrying key multiplicity so `Extension` is never terminal.

mod codec;
mod hot_match;
#[cfg(test)]
mod tests;

pub(crate) use codec::encode_unextended_fresh_box_into;
pub use hot_match::box_matches_query_hot;
pub(crate) use hot_match::{box_matches_partition_hot, box_matches_query_hot_with_age_hint};

use aeris_types::{FilterConfig, Fingerprint, Memento, PartitionHasher, PartitionId};

/// Precomputed bit-layout geometry derived once from [`FilterConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeepsakeGeometry {
    /// Remainder bit width `r` at fresh insertion (`age = 0`).
    pub r: u8,
    /// Memento bit width `m = ceil(log2(R))`.
    pub m: u8,
    /// Physical slot bit width `r + 1 + m = C + 1` (`2..=64`).
    pub slot_bits: u8,
    /// Bit mask for extracting a single `m`-bit memento (`0` when `m == 0`).
    pub m_mask: u64,
    /// Bit width of each subsequent extension slot (`r + m`).
    pub subsequent_ext_bits: u8,
    /// Bit mask for each subsequent extension slot.
    pub subsequent_ext_mask: u64,
    /// Maximum mementos packed into one continuation slot when `m > 0` (`0` when `m == 0`).
    pub cont_mementos_per_slot: usize,
    /// Header bit width encoding `count - 1` inside each continuation slot (`0` if `B <= 1`).
    pub cont_header_bits: u8,
    /// Bit mask for extracting `count - 1` from the low bits of a continuation slot.
    pub cont_header_mask: u64,
}

impl KeepsakeGeometry {
    /// Computes exact keepsake box packing geometry from a validated [`FilterConfig`].
    ///
    /// # Complexity
    /// `O(slot_bits)` time (runs once per filter construction/reconfiguration), `O(1)` space.
    pub fn from_config(config: &FilterConfig) -> Self {
        let r = config.r;
        let m = config.m;
        let slot_bits = config.physical_slot_bits();
        let m_mask = config.memento_mask();
        let subsequent_ext_bits = config.subsequent_extension_bits();
        let subsequent_ext_mask = mask_for_bits(subsequent_ext_bits);
        let (cont_mementos_per_slot, cont_header_bits) = compute_continuation_packing(slot_bits, m);
        let cont_header_mask = mask_for_bits(cont_header_bits);

        Self {
            r,
            m,
            slot_bits,
            m_mask,
            subsequent_ext_bits,
            subsequent_ext_mask,
            cont_mementos_per_slot,
            cont_header_bits,
            cont_header_mask,
        }
    }

    /// Returns total extension bit width for `ext_level` extension increments.
    #[inline]
    pub const fn ext_bits_for_level(&self, ext_level: usize) -> u16 {
        ext_bits_for_level(self.r, self.m, ext_level)
    }

    /// Returns the smallest `ext_level` supplying at least `min_ext_bits` extension bits.
    #[inline]
    pub const fn min_ext_level_for_bits(&self, min_ext_bits: u16) -> usize {
        min_ext_level_for_bits(self.r, self.m, min_ext_bits)
    }

    /// Computes the exact physical slot count of an unextended (`ext_level == 0`)
    /// keepsake box containing `key_count >= 1` keys under this geometry.
    #[inline]
    pub(crate) fn unextended_box_slot_len(&self, key_count: usize) -> usize {
        debug_assert!(key_count >= 1);
        if key_count == 1 {
            return 1;
        }
        if self.m > 0 {
            1 + (key_count - 1).div_ceil(self.cont_mementos_per_slot)
        } else {
            1 + codec::encoded_point_count_slots(key_count, self.slot_bits)
        }
    }
}

/// Computes total extension bits for `ext_level` increments under `(r, m)`.
///
/// - `ext_level == 0`: `0` bits.
/// - `ext_level >= 1` and `m > 0`: `m + (ext_level - 1) * (r + m)`.
/// - `ext_level >= 1` and `m == 0`: `ext_level * r`.
#[inline]
pub const fn ext_bits_for_level(r: u8, m: u8, ext_level: usize) -> u16 {
    if ext_level == 0 {
        0
    } else if m > 0 {
        (m as u16) + ((ext_level - 1) as u16) * ((r as u16) + (m as u16))
    } else {
        (ext_level as u16) * (r as u16)
    }
}

/// Computes the minimum extension level yielding at least `min_ext_bits` bits.
#[inline]
pub const fn min_ext_level_for_bits(r: u8, m: u8, min_ext_bits: u16) -> usize {
    if min_ext_bits == 0 {
        return 0;
    }
    if m > 0 {
        if min_ext_bits <= m as u16 {
            1
        } else {
            let rem = (min_ext_bits - (m as u16)) as usize;
            let stride = (r as usize) + (m as usize);
            1 + rem.div_ceil(stride)
        }
    } else {
        (min_ext_bits as usize).div_ceil(r as usize)
    }
}

/// Encodes `(age, remainder)` into the `r + 1`-bit InfiniFilter unary age field:
/// `age` leading zero bits, sentinel `1` at bit `r - age`, followed by the
/// `r - age` hash remainder bits.
///
/// # Panics
/// Panics in debug builds if `age > r`.
#[inline]
pub const fn encode_age_remainder(r: u8, age: u8, remainder: u64) -> u64 {
    debug_assert!(age <= r, "age cannot exceed remainder width r");
    let rem_bits = r - age;
    let rem_mask = mask_for_bits(rem_bits);
    (1u64 << rem_bits) | (remainder & rem_mask)
}

/// Decodes an `r + 1`-bit unary age field into `(age, rem_bits, remainder)`.
///
/// # Preconditions
/// `high_bits` MUST be non-zero with its sentinel bit at position `<= r`.
#[inline]
pub const fn decode_age_remainder(r: u8, high_bits: u64) -> (u8, u8, u64) {
    debug_assert!(high_bits != 0, "unary age field must carry sentinel 1 bit");
    let rem_bits = (63 - high_bits.leading_zeros()) as u8;
    debug_assert!(rem_bits <= r, "decoded rem_bits exceeds r");
    let age = r - rem_bits;
    let remainder = high_bits & mask_for_bits(rem_bits);
    (age, rem_bits, remainder)
}

/// Decoded payload carried by a keepsake box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoxPayload {
    /// Sorted ascending multiset of `m`-bit mementos (`m > 0`, always non-empty).
    Mementos(Vec<Memento>),
    /// Positive multiplicity of keys stored in this box (`m == 0`, always `>= 1`).
    PointCount(usize),
}

impl BoxPayload {
    /// Returns the total number of keys represented by this payload.
    #[inline]
    pub fn key_count(&self) -> usize {
        match self {
            Self::Mementos(v) => v.len(),
            Self::PointCount(c) => *c,
        }
    }

    /// Returns `true` if `self` represents zero keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.key_count() == 0
    }

    /// Inserts `memento` while preserving sorted ascending multiset order.
    pub fn insert_memento(&mut self, memento: Memento) {
        match self {
            Self::Mementos(v) => {
                let pos = v.partition_point(|&x| x < memento);
                v.insert(pos, memento);
            }
            Self::PointCount(c) => {
                *c += 1;
            }
        }
    }

    /// Merges another payload into `self`, preserving sorted ascending multiset order.
    ///
    /// # Panics
    /// Panics if `self` and `other` have mismatched variants (`Mementos` vs `PointCount`).
    pub fn merge_from(&mut self, other: Self) {
        match (self, other) {
            (Self::Mementos(a), Self::Mementos(b)) => {
                a.extend(b);
                a.sort_unstable();
            }
            (Self::PointCount(a), Self::PointCount(b)) => {
                *a += b;
            }
            _ => unreachable!("cannot merge Mementos and PointCount payloads"),
        }
    }

    /// Removes a single occurrence of `memento`, returning `true` if found and removed.
    pub fn remove_one(&mut self, memento: Memento) -> bool {
        match self {
            Self::Mementos(v) => {
                if let Ok(idx) = v.binary_search(&memento) {
                    v.remove(idx);
                    true
                } else {
                    false
                }
            }
            Self::PointCount(c) => {
                if *c > 0 {
                    *c -= 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Returns `true` if at least one stored memento falls within `[low_m, high_m]`.
    pub fn contains_in_range(&self, low_m: Memento, high_m: Memento) -> bool {
        match self {
            Self::Mementos(v) => {
                let idx = v.partition_point(|&x| x < low_m);
                idx < v.len() && v[idx] <= high_m
            }
            Self::PointCount(c) => *c > 0,
        }
    }
}

/// Decoded in-memory representation of a single Aeris keepsake box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeepsakeBox {
    /// InfiniFilter age (`0 <= age <= r`); effective remainder width is `r - age`.
    pub age: u8,
    /// Remaining `r - age` hash remainder bits (right-aligned).
    pub remainder: u64,
    /// Ordered extension segments `(bits, bit_len)` appended beyond `q + (r - age)`.
    pub ext_segments: Vec<(u64, u8)>,
    /// Stored mementos (`m > 0`) or key multiplicity (`m == 0`).
    pub payload: BoxPayload,
}

impl KeepsakeBox {
    /// Builds a fresh `age = 0` keepsake box for `partition` with `ext_level`
    /// extension increments and initial `payload`.
    pub fn new_fresh(
        hasher: &PartitionHasher,
        partition: PartitionId,
        q: u8,
        geom: &KeepsakeGeometry,
        ext_level: usize,
        payload: BoxPayload,
    ) -> Self {
        let (_, rem) = hasher.quotient_and_remainder(partition, q, geom.r);
        let ext_segments = extract_extension_segments(
            hasher,
            partition,
            (q as u16) + (geom.r as u16),
            geom,
            ext_level,
        );
        Self {
            age: 0,
            remainder: rem,
            ext_segments,
            payload,
        }
    }

    /// Returns the number of effective remainder bits (`r - age`).
    #[inline]
    pub const fn rem_bits(&self, r: u8) -> u8 {
        r - self.age
    }

    /// Returns the total number of extension bits carried by this box.
    #[inline]
    pub fn ext_bits(&self) -> u16 {
        self.ext_segments.iter().map(|&(_, len)| len as u16).sum()
    }

    /// Returns the total canonical fingerprint bit length `q + (r - age) + ext_bits`.
    #[inline]
    pub fn total_fp_bits(&self, q: u8, r: u8) -> u16 {
        (q as u16) + (self.rem_bits(r) as u16) + self.ext_bits()
    }

    /// Returns the extension level (`0` = unextended, `>= 1` = number of extension increments).
    #[inline]
    pub fn ext_level(&self) -> usize {
        self.ext_segments.len()
    }

    /// Reconstructs the canonical [`Fingerprint`] for this box given its `home_slot` and `(q, r)`.
    ///
    /// # Invariant
    /// Across non-rejuvenating `2x` expansions (`q -> q + 1`, `age -> age + 1`),
    /// this method produces the bit-for-bit identical [`Fingerprint`].
    pub fn canonical_fingerprint(&self, home: usize, q: u8, r: u8) -> Fingerprint {
        Fingerprint::from_parts(
            home as u64,
            q,
            self.remainder,
            self.rem_bits(r),
            &self.ext_segments,
        )
    }

    /// Returns `true` if `partition`'s hash stream matches this box's remainder
    /// and all extension segments (assuming `partition` already maps to this box's `home_slot`).
    pub fn matches_partition(
        &self,
        hasher: &PartitionHasher,
        partition: PartitionId,
        q: u8,
        r: u8,
    ) -> bool {
        let rem_bits = self.rem_bits(r);
        let expected_rem = hasher.extract_bits(partition, q as u16, rem_bits);
        if self.remainder != expected_rem {
            return false;
        }
        let mut bit_off = (q as u16) + (rem_bits as u16);
        for &(seg_val, seg_len) in &self.ext_segments {
            if hasher.extract_bits(partition, bit_off, seg_len) != seg_val {
                return false;
            }
            bit_off += seg_len as u16;
        }
        true
    }

    /// Performs in-memory InfiniFilter bit-splitting for `q -> q + 1`.
    ///
    /// - If `self.age < r`, shifts the MSB of `remainder` into `new_home = (old_home << 1) | b0`,
    ///   increments `age` by `1`, preserves exact extensions and payload, and returns
    ///   `Some((new_home, split_box))`.
    /// - If `self.age == r` (`0` remainder bits left), returns `None`, signaling that
    ///   this box requires selective rejuvenation via the Reverse Map.
    pub fn split_for_expansion(&self, old_home: usize, r: u8) -> Option<(usize, Self)> {
        if self.age >= r {
            return None;
        }
        let rem_bits = r - self.age;
        let b0 = ((self.remainder >> (rem_bits - 1)) & 1) as usize;
        let new_home = (old_home << 1) | b0;
        let new_rem_bits = rem_bits - 1;
        let new_rem = self.remainder & mask_for_bits(new_rem_bits);
        Some((
            new_home,
            Self {
                age: self.age + 1,
                remainder: new_rem,
                ext_segments: self.ext_segments.clone(),
                payload: self.payload.clone(),
            },
        ))
    }

    /// Deterministic ordering key for keepsake boxes within the same quotient run
    /// (`age = 0` fresh boxes first, then ascending remainder and extension bits).
    pub fn sort_key(&self) -> (u8, u64, usize) {
        (self.age, self.remainder, self.ext_segments.len())
    }
}

/// Extracts `ext_level` ordered extension segments from `hasher` starting at `start_bit_offset`.
pub fn extract_extension_segments(
    hasher: &PartitionHasher,
    partition: PartitionId,
    start_bit_offset: u16,
    geom: &KeepsakeGeometry,
    ext_level: usize,
) -> Vec<(u64, u8)> {
    if ext_level == 0 {
        return Vec::new();
    }
    let mut segs = Vec::with_capacity(ext_level);
    let mut off = start_bit_offset;
    if geom.m > 0 {
        segs.push((hasher.extract_bits(partition, off, geom.m), geom.m));
        off += geom.m as u16;
        for _ in 1..ext_level {
            segs.push((
                hasher.extract_bits(partition, off, geom.subsequent_ext_bits),
                geom.subsequent_ext_bits,
            ));
            off += geom.subsequent_ext_bits as u16;
        }
    } else {
        for _ in 0..ext_level {
            segs.push((hasher.extract_bits(partition, off, geom.r), geom.r));
            off += geom.r as u16;
        }
    }
    segs
}

#[inline]
pub(super) const fn mask_for_bits(bits: u8) -> u64 {
    if bits == 0 {
        0
    } else if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

const fn compute_continuation_packing(slot_bits: u8, m: u8) -> (usize, u8) {
    if m == 0 {
        return (0, 0);
    }
    let w = slot_bits as usize;
    let m_usize = m as usize;
    let max_possible_b = w / m_usize;
    let mut best_b = 1usize;
    let mut best_h = 0u8;
    let mut b = 1usize;
    while b <= max_possible_b {
        let h = if b <= 1 {
            0u8
        } else {
            (usize::BITS - (b - 1).leading_zeros()) as u8
        };
        if (h as usize) + b * m_usize <= w {
            best_b = b;
            best_h = h;
        }
        b += 1;
    }
    (best_b, best_h)
}
