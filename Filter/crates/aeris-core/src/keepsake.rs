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

use crate::chunk::QuotientTable;
use aeris_types::{
    FilterConfig, Fingerprint, KeepsakeSlotKind, Memento, PartitionHasher, PartitionId,
};

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
            1 + encoded_point_count_slots(key_count, self.slot_bits)
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

    /// Computes the exact number of physical Quotient Filter slots occupied by
    /// this box when encoded under `geom`.
    pub fn encoded_slot_len(&self, geom: &KeepsakeGeometry) -> usize {
        match &self.payload {
            BoxPayload::Mementos(mems) => {
                debug_assert!(geom.m > 0 && !mems.is_empty());
                if self.ext_segments.is_empty() {
                    if mems.len() == 1 {
                        1
                    } else {
                        1 + (mems.len() - 1).div_ceil(geom.cont_mementos_per_slot)
                    }
                } else {
                    // Slot 0 carries first extension; remaining `ext_segments.len() - 1`
                    // carry subsequent extensions; all mementos reside in continuation slots.
                    self.ext_segments.len() + mems.len().div_ceil(geom.cont_mementos_per_slot)
                }
            }
            BoxPayload::PointCount(count) => {
                debug_assert!(geom.m == 0 && *count >= 1);
                if self.ext_segments.is_empty() {
                    if *count == 1 {
                        1
                    } else {
                        1 + encoded_point_count_slots(*count, geom.slot_bits)
                    }
                } else {
                    // Slot 0 marks extension; `ext_segments.len()` slots carry `r`-bit extensions;
                    // trailing slot(s) carry multiplicity.
                    1 + self.ext_segments.len() + encoded_point_count_slots(*count, geom.slot_bits)
                }
            }
        }
    }

    /// Encodes this keepsake box into physical `(payload, KeepsakeSlotKind)` slots.
    ///
    /// # Panics
    /// Panics if the box payload is empty (`key_count() == 0`) or violates `geom`.
    pub fn encode(
        &self,
        geom: &KeepsakeGeometry,
        is_run_end: bool,
    ) -> Vec<(u64, KeepsakeSlotKind)> {
        let mut out = Vec::with_capacity(self.encoded_slot_len(geom));
        self.encode_into(geom, is_run_end, &mut out);
        out
    }

    /// Appends encoded `(payload, KeepsakeSlotKind)` slots for this box into `out`.
    pub fn encode_into(
        &self,
        geom: &KeepsakeGeometry,
        is_run_end: bool,
        out: &mut Vec<(u64, KeepsakeSlotKind)>,
    ) {
        let terminal = if is_run_end {
            KeepsakeSlotKind::BoxAndRunEnd
        } else {
            KeepsakeSlotKind::BoxEnd
        };
        let high = encode_age_remainder(geom.r, self.age, self.remainder);

        match &self.payload {
            BoxPayload::Mementos(mems) => {
                assert!(
                    geom.m > 0 && !mems.is_empty(),
                    "m > 0 keepsake box must be non-empty"
                );
                if self.ext_segments.is_empty() {
                    let slot0 = (high << geom.m) | (mems[0] & geom.m_mask);
                    if mems.len() == 1 {
                        out.push((slot0, terminal));
                    } else {
                        out.push((slot0, KeepsakeSlotKind::PayloadInterior));
                        encode_memento_continuations(&mems[1..], geom, terminal, out);
                    }
                } else {
                    debug_assert_eq!(self.ext_segments[0].1, geom.m);
                    let slot0 = (high << geom.m) | (self.ext_segments[0].0 & geom.m_mask);
                    out.push((slot0, KeepsakeSlotKind::Extension));
                    for &(seg_val, seg_len) in &self.ext_segments[1..] {
                        debug_assert_eq!(seg_len, geom.subsequent_ext_bits);
                        out.push((
                            seg_val & geom.subsequent_ext_mask,
                            KeepsakeSlotKind::Extension,
                        ));
                    }
                    encode_memento_continuations(mems, geom, terminal, out);
                }
            }
            BoxPayload::PointCount(count) => {
                assert!(
                    geom.m == 0 && *count >= 1,
                    "m == 0 keepsake box must have count >= 1"
                );
                let slot0 = high;
                if self.ext_segments.is_empty() {
                    if *count == 1 {
                        out.push((slot0, terminal));
                    } else {
                        out.push((slot0, KeepsakeSlotKind::PayloadInterior));
                        encode_point_count_slots(*count, geom.slot_bits, terminal, out);
                    }
                } else {
                    out.push((slot0, KeepsakeSlotKind::Extension));
                    for &(seg_val, seg_len) in &self.ext_segments {
                        debug_assert_eq!(seg_len, geom.r);
                        out.push((
                            seg_val & geom.subsequent_ext_mask,
                            KeepsakeSlotKind::Extension,
                        ));
                    }
                    encode_point_count_slots(*count, geom.slot_bits, terminal, out);
                }
            }
        }
    }

    /// Decodes a keepsake box starting at physical slot `box_start` inside `table`.
    ///
    /// Returns `(decoded_box, slot_count)`.
    pub fn decode_from_table(
        table: &QuotientTable,
        geom: &KeepsakeGeometry,
        box_start: usize,
    ) -> (Self, usize) {
        let (slot0, kind0) = table.get_payload_and_kind(box_start);
        let high = slot0 >> geom.m;
        let (age, _rem_bits, remainder) = decode_age_remainder(geom.r, high);

        if geom.m > 0 {
            if kind0.is_box_end() {
                let bx = Self {
                    age,
                    remainder,
                    ext_segments: Vec::new(),
                    payload: BoxPayload::Mementos(vec![slot0 & geom.m_mask]),
                };
                return (bx, 1);
            }

            let mut ext_segments = Vec::new();
            let mut mementos = Vec::new();
            let mut idx = box_start + 1;

            let (mut cur_slot, mut cur_kind) = if kind0 == KeepsakeSlotKind::Extension {
                ext_segments.push((slot0 & geom.m_mask, geom.m));
                loop {
                    let (s, k) = table.get_payload_and_kind(idx);
                    if k != KeepsakeSlotKind::Extension {
                        break (s, k);
                    }
                    ext_segments.push((s & geom.subsequent_ext_mask, geom.subsequent_ext_bits));
                    idx += 1;
                }
            } else {
                // `PayloadInterior`: unextended multi-memento box; Slot 0 holds `m_0`.
                mementos.push(slot0 & geom.m_mask);
                table.get_payload_and_kind(idx)
            };

            loop {
                decode_memento_continuation_slot(cur_slot, geom, &mut mementos);
                if cur_kind.is_box_end() {
                    let bx = Self {
                        age,
                        remainder,
                        ext_segments,
                        payload: BoxPayload::Mementos(mementos),
                    };
                    return (bx, idx - box_start + 1);
                }
                idx += 1;
                (cur_slot, cur_kind) = table.get_payload_and_kind(idx);
            }
        } else {
            // Point-filter mode (`m == 0`).
            if kind0.is_box_end() {
                let bx = Self {
                    age,
                    remainder,
                    ext_segments: Vec::new(),
                    payload: BoxPayload::PointCount(1),
                };
                return (bx, 1);
            }

            let mut ext_segments = Vec::new();
            let mut idx = box_start + 1;

            let (mut cur_slot, mut cur_kind) = if kind0 == KeepsakeSlotKind::Extension {
                loop {
                    let (s, k) = table.get_payload_and_kind(idx);
                    if k != KeepsakeSlotKind::Extension {
                        break (s, k);
                    }
                    ext_segments.push((s & geom.subsequent_ext_mask, geom.r));
                    idx += 1;
                }
            } else {
                table.get_payload_and_kind(idx)
            };

            let mut count = 0usize;
            let mut shift = 0u32;
            loop {
                count |= (cur_slot as usize) << shift;
                shift += geom.slot_bits as u32;
                if cur_kind.is_box_end() {
                    let bx = Self {
                        age,
                        remainder,
                        ext_segments,
                        payload: BoxPayload::PointCount(count),
                    };
                    return (bx, idx - box_start + 1);
                }
                idx += 1;
                (cur_slot, cur_kind) = table.get_payload_and_kind(idx);
            }
        }
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

/// Zero-allocation hot-path check of whether the keepsake box starting at
/// `box_start` matches `(partition, w0, q, rem_full)` AND contains at least
/// one memento within `[low_m, high_m]`.
///
/// Returns `(matched, next_box_start)` so the caller can advance directly to
/// the following keepsake box in the quotient run without decoding payloads.
/// When `has_aged` is `false` (no expansions have occurred) and the box is a
/// fresh (`age == 0`) box whose remainder strictly exceeds `rem_full`, returns
/// `(false, usize::MAX)` to signal immediate run termination because fresh
/// boxes are stored in ascending remainder order.
///
/// # Complexity
/// `O(box_slots)` time, strictly zero heap allocations.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn box_matches_query_hot(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    hasher: &PartitionHasher,
    partition: PartitionId,
    w0: u64,
    q: u8,
    rem_full: u64,
    low_m: Memento,
    high_m: Memento,
    box_start: usize,
) -> (bool, usize) {
    box_matches_query_hot_inner(
        table, geom, hasher, partition, w0, q, rem_full, low_m, high_m, box_start, true,
    )
}

/// Internal variant of [`box_matches_query_hot`] that accepts `has_aged` so
/// callers with `expansion_cycles == 0` can terminate run scans early once a
/// fresh box's remainder exceeds `rem_full`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn box_matches_query_hot_with_age_hint(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    hasher: &PartitionHasher,
    partition: PartitionId,
    w0: u64,
    q: u8,
    rem_full: u64,
    low_m: Memento,
    high_m: Memento,
    box_start: usize,
    has_aged: bool,
) -> (bool, usize) {
    box_matches_query_hot_inner(
        table, geom, hasher, partition, w0, q, rem_full, low_m, high_m, box_start, has_aged,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn box_matches_query_hot_inner(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    hasher: &PartitionHasher,
    partition: PartitionId,
    w0: u64,
    q: u8,
    rem_full: u64,
    low_m: Memento,
    high_m: Memento,
    box_start: usize,
    has_aged: bool,
) -> (bool, usize) {
    let (slot0, kind0) = table.get_payload_and_kind(box_start);
    let high = slot0 >> geom.m;

    let rem_bits = if !has_aged {
        // 100%-predictable branch when no expansions have occurred: all boxes have `age == 0`,
        // so `high == (1 << r) | rem_full` replaces `LZCNT` + shift + mask with 1 compare.
        let expected_fresh_high = (1u64 << geom.r) | rem_full;
        if high != expected_fresh_high {
            let next = if kind0.is_box_end() {
                box_start + 1
            } else {
                table.find_box_end(box_start + 1) + 1
            };
            return (false, next);
        }
        geom.r
    } else {
        let (age, rem_bits, box_rem) = decode_age_remainder(geom.r, high);
        if box_rem != (rem_full >> age) {
            let next = if kind0.is_box_end() {
                box_start + 1
            } else {
                table.find_box_end(box_start + 1) + 1
            };
            return (false, next);
        }
        rem_bits
    };

    // Fast path: single-slot unextended keepsake box (>90% of lookups).
    if kind0.is_box_end() {
        if geom.m == 0 {
            return (true, box_start + 1);
        }
        let m0 = slot0 & geom.m_mask;
        return (m0 >= low_m && m0 <= high_m, box_start + 1);
    }

    let mut idx = box_start + 1;
    let mut prefetched_cont: Option<(u64, KeepsakeSlotKind)> = None;
    if kind0 == KeepsakeSlotKind::Extension {
        let mut bit_off = (q as u16) + (rem_bits as u16);
        if geom.m > 0 {
            let expected_ext0 = extract_bits_cached(hasher, partition, w0, bit_off, geom.m);
            if (slot0 & geom.m_mask) != expected_ext0 {
                return (false, table.find_box_end(idx) + 1);
            }
            bit_off += geom.m as u16;
        }
        let seg_len = geom.subsequent_ext_bits;
        let seg_mask = geom.subsequent_ext_mask;
        loop {
            let (slot, kind) = table.get_payload_and_kind(idx);
            if kind != KeepsakeSlotKind::Extension {
                if geom.m == 0 {
                    let next = if kind.is_box_end() {
                        idx + 1
                    } else {
                        table.find_box_end(idx + 1) + 1
                    };
                    return (true, next);
                }
                prefetched_cont = Some((slot, kind));
                break;
            }
            let expected = extract_bits_cached(hasher, partition, w0, bit_off, seg_len);
            if (slot & seg_mask) != expected {
                return (false, table.find_box_end(idx + 1) + 1);
            }
            bit_off += seg_len as u16;
            idx += 1;
        }
    } else {
        // Unextended multi-slot keepsake box (`PayloadInterior` on Slot 0).
        if geom.m == 0 {
            return (true, table.find_box_end(idx) + 1);
        }
        let m0 = slot0 & geom.m_mask;
        if m0 >= low_m && m0 <= high_m {
            return (true, table.find_box_end(idx) + 1);
        }
        if m0 > high_m {
            return (false, table.find_box_end(idx) + 1);
        }
    }

    // Scan sorted memento continuation slots from `idx` until `is_box_end()`.
    loop {
        let (slot, kind) = match prefetched_cont.take() {
            Some(pair) => pair,
            None => table.get_payload_and_kind(idx),
        };
        let count = if geom.cont_header_bits == 0 {
            1usize
        } else {
            ((slot & geom.cont_header_mask) as usize) + 1
        };
        let mut packed = slot >> geom.cont_header_bits;
        for _ in 0..count {
            let mem = packed & geom.m_mask;
            if mem >= low_m && mem <= high_m {
                let next = if kind.is_box_end() {
                    idx + 1
                } else {
                    table.find_box_end(idx + 1) + 1
                };
                return (true, next);
            }
            if mem > high_m {
                let next = if kind.is_box_end() {
                    idx + 1
                } else {
                    table.find_box_end(idx + 1) + 1
                };
                return (false, next);
            }
            packed >>= geom.m;
        }
        if kind.is_box_end() {
            return (false, idx + 1);
        }
        idx += 1;
    }
}

/// Zero-allocation check of whether the keepsake box starting at `box_start`
/// matches `partition`'s hash prefix (remainder + all extension segments),
/// ignoring memento values.
///
/// Returns `(matched_partition, next_box_start, slot0_high, kind0)` so the
/// in-place `adapt()` fast path can simultaneously detect peer remainder
/// collisions and extended aged boxes in a single scan.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn box_matches_partition_hot(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    hasher: &PartitionHasher,
    partition: PartitionId,
    w0: u64,
    q: u8,
    rem_full: u64,
    box_start: usize,
) -> (bool, usize, u64, KeepsakeSlotKind) {
    let (slot0, kind0) = table.get_payload_and_kind(box_start);
    let high = slot0 >> geom.m;
    let fresh_sentinel = 1u64 << geom.r;
    let expected_fresh_high = fresh_sentinel | rem_full;

    let rem_bits = if high == expected_fresh_high {
        geom.r
    } else if high >= fresh_sentinel {
        let next = if kind0.is_box_end() {
            box_start + 1
        } else {
            table.find_box_end(box_start + 1) + 1
        };
        return (false, next, high, kind0);
    } else {
        let (age, rem_bits, box_rem) = decode_age_remainder(geom.r, high);
        if box_rem != (rem_full >> age) {
            let next = if kind0.is_box_end() {
                box_start + 1
            } else {
                table.find_box_end(box_start + 1) + 1
            };
            return (false, next, high, kind0);
        }
        rem_bits
    };

    if kind0.is_box_end() {
        return (true, box_start + 1, high, kind0);
    }

    let mut idx = box_start + 1;
    if kind0 == KeepsakeSlotKind::Extension {
        let mut bit_off = (q as u16) + (rem_bits as u16);
        if geom.m > 0 {
            let expected_ext0 = extract_bits_cached(hasher, partition, w0, bit_off, geom.m);
            if (slot0 & geom.m_mask) != expected_ext0 {
                return (false, table.find_box_end(idx) + 1, high, kind0);
            }
            bit_off += geom.m as u16;
        }
        let seg_len = geom.subsequent_ext_bits;
        let seg_mask = geom.subsequent_ext_mask;
        loop {
            let (slot, kind) = table.get_payload_and_kind(idx);
            if kind != KeepsakeSlotKind::Extension {
                let next = if kind.is_box_end() {
                    idx + 1
                } else {
                    table.find_box_end(idx + 1) + 1
                };
                return (true, next, high, kind0);
            }
            let expected = extract_bits_cached(hasher, partition, w0, bit_off, seg_len);
            if (slot & seg_mask) != expected {
                return (false, table.find_box_end(idx + 1) + 1, high, kind0);
            }
            bit_off += seg_len as u16;
            idx += 1;
        }
    }

    (true, table.find_box_end(idx) + 1, high, kind0)
}

/// Extracts `len` bits at `bit_offset`, serving bits within `0..64` directly
/// from the pre-hashed `w0` word without re-invoking SplitMix64.
#[inline]
fn extract_bits_cached(
    hasher: &PartitionHasher,
    partition: PartitionId,
    w0: u64,
    bit_offset: u16,
    len: u8,
) -> u64 {
    if len == 0 {
        return 0;
    }
    if (bit_offset as u32) + (len as u32) <= 64 {
        let shift = 64 - (bit_offset as u32) - (len as u32);
        (w0 >> shift) & mask_for_bits(len)
    } else {
        hasher.extract_bits(partition, bit_offset, len)
    }
}

#[inline]
const fn mask_for_bits(bits: u8) -> u64 {
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

/// Appends encoded `(payload, KeepsakeSlotKind)` slots for a fresh (`age = 0`),
/// unextended keepsake box directly into `out` without allocating intermediate
/// [`KeepsakeBox`] or [`BoxPayload`] structures.
///
/// # Preconditions
/// - `key_count >= 1`.
/// - When `geom.m > 0`, `mementos` MUST be sorted ascending with `mementos.len() == key_count`.
#[inline]
pub(crate) fn encode_unextended_fresh_box_into(
    geom: &KeepsakeGeometry,
    remainder: u64,
    mementos: &[Memento],
    key_count: usize,
    is_run_end: bool,
    out: &mut Vec<(u64, KeepsakeSlotKind)>,
) {
    debug_assert!(key_count >= 1);
    let terminal = if is_run_end {
        KeepsakeSlotKind::BoxAndRunEnd
    } else {
        KeepsakeSlotKind::BoxEnd
    };
    let high = encode_age_remainder(geom.r, 0, remainder);

    if geom.m > 0 {
        debug_assert_eq!(mementos.len(), key_count);
        let slot0 = (high << geom.m) | (mementos[0] & geom.m_mask);
        if key_count == 1 {
            out.push((slot0, terminal));
        } else {
            out.push((slot0, KeepsakeSlotKind::PayloadInterior));
            encode_memento_continuations(&mementos[1..], geom, terminal, out);
        }
    } else if key_count == 1 {
        out.push((high, terminal));
    } else {
        out.push((high, KeepsakeSlotKind::PayloadInterior));
        encode_point_count_slots(key_count, geom.slot_bits, terminal, out);
    }
}

fn encode_memento_continuations(
    mems: &[Memento],
    geom: &KeepsakeGeometry,
    terminal: KeepsakeSlotKind,
    out: &mut Vec<(u64, KeepsakeSlotKind)>,
) {
    debug_assert!(!mems.is_empty());
    let b = geom.cont_mementos_per_slot;
    let total_chunks = mems.len().div_ceil(b);
    for (chunk_idx, chunk) in mems.chunks(b).enumerate() {
        let mut word = if geom.cont_header_bits == 0 {
            0u64
        } else {
            (chunk.len() - 1) as u64
        };
        let mut bit_off = geom.cont_header_bits as u32;
        for &mem in chunk {
            word |= (mem & geom.m_mask) << bit_off;
            bit_off += geom.m as u32;
        }
        let kind = if chunk_idx + 1 == total_chunks {
            terminal
        } else {
            KeepsakeSlotKind::PayloadInterior
        };
        out.push((word, kind));
    }
}

fn decode_memento_continuation_slot(slot: u64, geom: &KeepsakeGeometry, out: &mut Vec<Memento>) {
    let count = if geom.cont_header_bits == 0 {
        1usize
    } else {
        ((slot & geom.cont_header_mask) as usize) + 1
    };
    let mut packed = slot >> geom.cont_header_bits;
    for _ in 0..count {
        out.push(packed & geom.m_mask);
        packed >>= geom.m;
    }
}

fn encoded_point_count_slots(count: usize, slot_bits: u8) -> usize {
    debug_assert!(count >= 1);
    if slot_bits >= 64 {
        return 1;
    }
    let mut rem = count;
    let mut slots = 0usize;
    while rem > 0 {
        slots += 1;
        rem >>= slot_bits;
    }
    slots
}

fn encode_point_count_slots(
    count: usize,
    slot_bits: u8,
    terminal: KeepsakeSlotKind,
    out: &mut Vec<(u64, KeepsakeSlotKind)>,
) {
    let total = encoded_point_count_slots(count, slot_bits);
    let mask = mask_for_bits(slot_bits);
    let mut rem = count;
    for i in 0..total {
        let limb = (rem as u64) & mask;
        if slot_bits < 64 {
            rem >>= slot_bits;
        } else {
            rem = 0;
        }
        let kind = if i + 1 == total {
            terminal
        } else {
            KeepsakeSlotKind::PayloadInterior
        };
        out.push((limb, kind));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_age_counter_round_trips_for_all_ages() {
        for r in [4u8, 8, 12, 20, 40] {
            for age in 0..=r {
                let rem_bits = r - age;
                let max_rem = mask_for_bits(rem_bits);
                for rem in [0u64, 1 & max_rem, max_rem >> 1, max_rem] {
                    let encoded = encode_age_remainder(r, age, rem);
                    assert_ne!(encoded, 0);
                    let (dec_age, dec_bits, dec_rem) = decode_age_remainder(r, encoded);
                    assert_eq!(dec_age, age);
                    assert_eq!(dec_bits, rem_bits);
                    assert_eq!(dec_rem, rem);
                }
            }
        }
    }

    #[test]
    fn keepsake_box_round_trip_and_hot_match_for_m_positive_and_zero() {
        let hasher = PartitionHasher::new(0x1234_5678_9ABC_DEF0);

        for &(r, m, max_range) in &[(12u8, 0u8, 1u64), (12, 1, 2), (12, 5, 32), (12, 10, 1024)] {
            let cfg = FilterConfig::new(8, r, max_range, r + m, 1.0 / 256.0).unwrap();
            let geom = KeepsakeGeometry::from_config(&cfg);
            let mut table = QuotientTable::new(cfg.num_slots(), geom.slot_bits);

            let partition = 0xDEAD_BEEF_u64;
            let (home_u64, rem_full) = hasher.quotient_and_remainder(partition, cfg.q, cfg.r);
            let home = home_u64 as usize;
            let w0 = hasher.word(partition, 0);

            for ext_level in 0..=3 {
                let payloads: Vec<BoxPayload> = if m == 0 {
                    vec![
                        BoxPayload::PointCount(1),
                        BoxPayload::PointCount(2),
                        BoxPayload::PointCount(500),
                    ]
                } else {
                    let max_m = geom.m_mask;
                    vec![
                        BoxPayload::Mementos(vec![0]),
                        BoxPayload::Mementos(vec![max_m]),
                        BoxPayload::Mementos(vec![0, 0, 1 & max_m, max_m, max_m]),
                        BoxPayload::Mementos((0..=max_m.min(11)).collect()),
                    ]
                };

                for payload in payloads {
                    let bx = KeepsakeBox::new_fresh(
                        &hasher, partition, cfg.q, &geom, ext_level, payload,
                    );
                    let encoded = bx.encode(&geom, true);
                    assert_eq!(encoded.len(), bx.encoded_slot_len(&geom));
                    // Extension slot must NEVER be last slot of a keepsake box.
                    assert!(!encoded.last().unwrap().1.is_extension_slot());
                    assert!(encoded.last().unwrap().1.is_run_end());

                    let old_len = table.locate_run(home).map(|(s, e)| e - s + 1).unwrap_or(0);
                    let start = table.locate_run(home).map(|(s, _)| s).unwrap_or(0);
                    table.replace_region(home, start, old_len, &encoded);
                    let (rs, re) = table.locate_run(home).unwrap();
                    assert_eq!(re - rs + 1, encoded.len());

                    let (decoded, consumed) = KeepsakeBox::decode_from_table(&table, &geom, rs);
                    assert_eq!(consumed, encoded.len());
                    assert_eq!(decoded, bx);
                    assert_eq!(
                        decoded.canonical_fingerprint(home, cfg.q, cfg.r),
                        Fingerprint::from_hasher(
                            &hasher,
                            partition,
                            (cfg.q as u16) + (cfg.r as u16) + geom.ext_bits_for_level(ext_level)
                        )
                    );

                    // Verify hot-path query matches exact decoded payload.
                    let (hit_full, next_slot) = box_matches_query_hot(
                        &table,
                        &geom,
                        &hasher,
                        partition,
                        w0,
                        cfg.q,
                        rem_full,
                        0,
                        geom.m_mask,
                        rs,
                    );
                    assert!(hit_full);
                    assert_eq!(next_slot, re + 1);

                    // Non-matching partition must not match (except vanishingly unlikely collision).
                    let other_part = partition ^ 0xFFFF_FFFF;
                    let (_, other_rem) = hasher.quotient_and_remainder(other_part, cfg.q, cfg.r);
                    let (other_hit, other_next) = box_matches_query_hot(
                        &table,
                        &geom,
                        &hasher,
                        other_part,
                        hasher.word(other_part, 0),
                        cfg.q,
                        other_rem,
                        0,
                        geom.m_mask,
                        rs,
                    );
                    assert_eq!(other_next, re + 1);
                    assert_eq!(
                        other_hit,
                        bx.matches_partition(&hasher, other_part, cfg.q, cfg.r)
                    );
                }
            }
        }
    }

    #[test]
    fn expansion_bit_split_preserves_canonical_fingerprint_until_rejuvenation() {
        let hasher = PartitionHasher::new(999);
        let cfg = FilterConfig::new(8, 6, 32, 11, 1.0 / 8.0)
            .or_else(|_| FilterConfig::with_minimum_bits(8, 32, 0.05))
            .unwrap();
        let geom = KeepsakeGeometry::from_config(&cfg);
        let partition = 42_424_242u64;
        let (home0, _) = hasher.quotient_and_remainder(partition, cfg.q, cfg.r);

        let mut bx = KeepsakeBox::new_fresh(
            &hasher,
            partition,
            cfg.q,
            &geom,
            2,
            BoxPayload::Mementos(vec![3, 7, 19]),
        );
        let orig_fp = bx.canonical_fingerprint(home0 as usize, cfg.q, cfg.r);

        let mut cur_home = home0 as usize;
        let mut cur_q = cfg.q;
        for step in 0..cfg.r {
            let (next_home, next_box) = bx
                .split_for_expansion(cur_home, cfg.r)
                .expect("age < r must split in memory");
            cur_home = next_home;
            cur_q += 1;
            bx = next_box;
            assert_eq!(bx.age, step + 1);
            assert_eq!(bx.canonical_fingerprint(cur_home, cur_q, cfg.r), orig_fp);
            assert!(bx.matches_partition(&hasher, partition, cur_q, cfg.r));
        }

        // At age == r, next split returns None -> triggers selective rejuvenation!
        assert_eq!(bx.age, cfg.r);
        assert!(bx.split_for_expansion(cur_home, cfg.r).is_none());
    }

    #[test]
    fn point_mode_multi_limb_count_round_trips_for_narrow_and_64bit_slots() {
        let hasher = PartitionHasher::new(77);
        for &(r, eps) in &[(7u8, 0.1), (63u8, 1.0 / 256.0)] {
            let cfg = FilterConfig::new(7, r, 1, r, eps).unwrap();
            let geom = KeepsakeGeometry::from_config(&cfg);
            let mut table = QuotientTable::new(cfg.num_slots(), geom.slot_bits);
            let part = 12345u64;
            let (home_u64, _) = hasher.quotient_and_remainder(part, cfg.q, cfg.r);
            let home = home_u64 as usize;

            for ext_level in [0usize, 1, 2] {
                for count in [1usize, 2, 255, 256, 70_000] {
                    let bx = KeepsakeBox::new_fresh(
                        &hasher,
                        part,
                        cfg.q,
                        &geom,
                        ext_level,
                        BoxPayload::PointCount(count),
                    );
                    let enc = bx.encode(&geom, true);
                    assert_eq!(enc.len(), bx.encoded_slot_len(&geom));
                    assert!(!enc.last().unwrap().1.is_extension_slot());
                    let (start, old_len) = table
                        .locate_run(home)
                        .map(|(s, e)| (s, e - s + 1))
                        .unwrap_or((0, 0));
                    table.replace_region(home, start, old_len, &enc);
                    let (rs, _) = table.locate_run(home).unwrap();
                    let (dec, consumed) = KeepsakeBox::decode_from_table(&table, &geom, rs);
                    assert_eq!(consumed, enc.len());
                    assert_eq!(dec, bx);
                }
            }
        }
    }
}
