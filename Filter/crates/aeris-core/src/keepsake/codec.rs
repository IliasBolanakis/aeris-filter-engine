//! Serialization and deserialization of [`KeepsakeBox`] records into physical
//! [`QuotientTable`] slots.

use super::{
    decode_age_remainder, encode_age_remainder, mask_for_bits, BoxPayload, KeepsakeBox,
    KeepsakeGeometry,
};
use crate::chunk::QuotientTable;
use aeris_types::{KeepsakeSlotKind, Memento};

impl KeepsakeBox {
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
    ///
    /// # Panics
    /// Panics if the box payload is empty or inconsistent with `geom.m`.
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
                encode_memento_box_into(high, &self.ext_segments, mems, geom, terminal, out);
            }
            BoxPayload::PointCount(count) => {
                encode_point_box_into(high, &self.ext_segments, *count, geom, terminal, out);
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
            decode_memento_box(table, geom, box_start, age, remainder, slot0, kind0)
        } else {
            decode_point_box(table, geom, box_start, age, remainder, kind0)
        }
    }
}

fn encode_memento_box_into(
    high: u64,
    ext_segments: &[(u64, u8)],
    mems: &[Memento],
    geom: &KeepsakeGeometry,
    terminal: KeepsakeSlotKind,
    out: &mut Vec<(u64, KeepsakeSlotKind)>,
) {
    assert!(
        geom.m > 0 && !mems.is_empty(),
        "m > 0 keepsake box must be non-empty"
    );
    if ext_segments.is_empty() {
        let slot0 = (high << geom.m) | (mems[0] & geom.m_mask);
        if mems.len() == 1 {
            out.push((slot0, terminal));
        } else {
            out.push((slot0, KeepsakeSlotKind::PayloadInterior));
            encode_memento_continuations(&mems[1..], geom, terminal, out);
        }
    } else {
        debug_assert_eq!(ext_segments[0].1, geom.m);
        let slot0 = (high << geom.m) | (ext_segments[0].0 & geom.m_mask);
        out.push((slot0, KeepsakeSlotKind::Extension));
        for &(seg_val, seg_len) in &ext_segments[1..] {
            debug_assert_eq!(seg_len, geom.subsequent_ext_bits);
            out.push((
                seg_val & geom.subsequent_ext_mask,
                KeepsakeSlotKind::Extension,
            ));
        }
        encode_memento_continuations(mems, geom, terminal, out);
    }
}

fn encode_point_box_into(
    high: u64,
    ext_segments: &[(u64, u8)],
    count: usize,
    geom: &KeepsakeGeometry,
    terminal: KeepsakeSlotKind,
    out: &mut Vec<(u64, KeepsakeSlotKind)>,
) {
    assert!(
        geom.m == 0 && count >= 1,
        "m == 0 keepsake box must have count >= 1"
    );
    if ext_segments.is_empty() {
        if count == 1 {
            out.push((high, terminal));
        } else {
            out.push((high, KeepsakeSlotKind::PayloadInterior));
            encode_point_count_slots(count, geom.slot_bits, terminal, out);
        }
    } else {
        out.push((high, KeepsakeSlotKind::Extension));
        for &(seg_val, seg_len) in ext_segments {
            debug_assert_eq!(seg_len, geom.r);
            out.push((
                seg_val & geom.subsequent_ext_mask,
                KeepsakeSlotKind::Extension,
            ));
        }
        encode_point_count_slots(count, geom.slot_bits, terminal, out);
    }
}

fn collect_extension_slots(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    start_idx: usize,
    seg_bits: u8,
    ext_segments: &mut Vec<(u64, u8)>,
) -> (usize, u64, KeepsakeSlotKind) {
    let mut idx = start_idx;
    loop {
        let (s, k) = table.get_payload_and_kind(idx);
        if k != KeepsakeSlotKind::Extension {
            return (idx, s, k);
        }
        ext_segments.push((s & geom.subsequent_ext_mask, seg_bits));
        idx += 1;
    }
}

fn decode_memento_box(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    box_start: usize,
    age: u8,
    remainder: u64,
    slot0: u64,
    kind0: KeepsakeSlotKind,
) -> (KeepsakeBox, usize) {
    if kind0.is_box_end() {
        let bx = KeepsakeBox {
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
        let (next_idx, s, k) = collect_extension_slots(
            table,
            geom,
            idx,
            geom.subsequent_ext_bits,
            &mut ext_segments,
        );
        idx = next_idx;
        (s, k)
    } else {
        // `PayloadInterior`: unextended multi-memento box; Slot 0 holds `m_0`.
        mementos.push(slot0 & geom.m_mask);
        table.get_payload_and_kind(idx)
    };

    loop {
        decode_memento_continuation_slot(cur_slot, geom, &mut mementos);
        if cur_kind.is_box_end() {
            let bx = KeepsakeBox {
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
}

fn decode_point_box(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    box_start: usize,
    age: u8,
    remainder: u64,
    kind0: KeepsakeSlotKind,
) -> (KeepsakeBox, usize) {
    if kind0.is_box_end() {
        let bx = KeepsakeBox {
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
        let (next_idx, s, k) = collect_extension_slots(table, geom, idx, geom.r, &mut ext_segments);
        idx = next_idx;
        (s, k)
    } else {
        table.get_payload_and_kind(idx)
    };

    let mut count = 0usize;
    let mut shift = 0u32;
    loop {
        count |= (cur_slot as usize) << shift;
        shift += geom.slot_bits as u32;
        if cur_kind.is_box_end() {
            let bx = KeepsakeBox {
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

pub(super) fn encoded_point_count_slots(count: usize, slot_bits: u8) -> usize {
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
