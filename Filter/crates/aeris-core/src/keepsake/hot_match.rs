//! Zero-allocation hot-path keepsake box matching for queries and in-place adaptations.

use super::{decode_age_remainder, mask_for_bits, KeepsakeGeometry};
use crate::chunk::QuotientTable;
use aeris_types::{KeepsakeSlotKind, Memento, PartitionHasher, PartitionId};

/// Result of matching extension slots inside [`box_matches_query_hot_inner`].
#[derive(Clone, Copy)]
enum ExtQueryMatch {
    Mismatch(usize),
    MatchedPoint(usize),
    MatchedMementoCont {
        slot: u64,
        kind: KeepsakeSlotKind,
        idx: usize,
    },
}

/// Zero-allocation hot-path check of whether the keepsake box starting at
/// `box_start` matches `(partition, w0, q, rem_full)` AND contains at least
/// one memento within `[low_m, high_m]`.
///
/// Returns `(matched, next_box_start)` so the caller can advance directly to
/// the following keepsake box in the quotient run without decoding payloads.
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
/// callers with `expansion_cycles == 0` can terminate unaged remainder checks
/// with a single equality comparison instead of `LZCNT`.
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

#[inline(always)]
fn skip_box_after(table: &QuotientTable, next_slot: usize, kind: KeepsakeSlotKind) -> usize {
    if kind.is_box_end() {
        next_slot
    } else {
        table.find_box_end(next_slot) + 1
    }
}

#[inline(always)]
fn check_query_slot0_remainder(
    geom: &KeepsakeGeometry,
    high: u64,
    rem_full: u64,
    has_aged: bool,
) -> Option<u8> {
    if !has_aged {
        // 100%-predictable branch when no expansions have occurred: all boxes have `age == 0`,
        // so `high == (1 << r) | rem_full` replaces `LZCNT` + shift + mask with 1 compare.
        let expected_fresh_high = (1u64 << geom.r) | rem_full;
        if high == expected_fresh_high {
            Some(geom.r)
        } else {
            None
        }
    } else {
        let (age, rem_bits, box_rem) = decode_age_remainder(geom.r, high);
        if box_rem == (rem_full >> age) {
            Some(rem_bits)
        } else {
            None
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn match_query_extensions_hot(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    hasher: &PartitionHasher,
    partition: PartitionId,
    w0: u64,
    q: u8,
    rem_bits: u8,
    slot0: u64,
    mut idx: usize,
) -> ExtQueryMatch {
    let mut bit_off = (q as u16) + (rem_bits as u16);
    if geom.m > 0 {
        let expected_ext0 = extract_bits_cached(hasher, partition, w0, bit_off, geom.m);
        if (slot0 & geom.m_mask) != expected_ext0 {
            return ExtQueryMatch::Mismatch(table.find_box_end(idx) + 1);
        }
        bit_off += geom.m as u16;
    }
    let seg_len = geom.subsequent_ext_bits;
    let seg_mask = geom.subsequent_ext_mask;
    loop {
        let (slot, kind) = table.get_payload_and_kind(idx);
        if kind != KeepsakeSlotKind::Extension {
            if geom.m == 0 {
                return ExtQueryMatch::MatchedPoint(skip_box_after(table, idx + 1, kind));
            }
            return ExtQueryMatch::MatchedMementoCont { slot, kind, idx };
        }
        let expected = extract_bits_cached(hasher, partition, w0, bit_off, seg_len);
        if (slot & seg_mask) != expected {
            return ExtQueryMatch::Mismatch(table.find_box_end(idx + 1) + 1);
        }
        bit_off += seg_len as u16;
        idx += 1;
    }
}

#[inline(always)]
fn scan_memento_continuations_hot(
    table: &QuotientTable,
    geom: &KeepsakeGeometry,
    low_m: Memento,
    high_m: Memento,
    mut idx: usize,
    mut prefetched: Option<(u64, KeepsakeSlotKind)>,
) -> (bool, usize) {
    loop {
        let (slot, kind) = match prefetched.take() {
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
                return (true, skip_box_after(table, idx + 1, kind));
            }
            if mem > high_m {
                return (false, skip_box_after(table, idx + 1, kind));
            }
            packed >>= geom.m;
        }
        if kind.is_box_end() {
            return (false, idx + 1);
        }
        idx += 1;
    }
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

    let Some(rem_bits) = check_query_slot0_remainder(geom, high, rem_full, has_aged) else {
        return (false, skip_box_after(table, box_start + 1, kind0));
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
    let prefetched_cont = if kind0 == KeepsakeSlotKind::Extension {
        match match_query_extensions_hot(
            table, geom, hasher, partition, w0, q, rem_bits, slot0, idx,
        ) {
            ExtQueryMatch::Mismatch(next) => return (false, next),
            ExtQueryMatch::MatchedPoint(next) => return (true, next),
            ExtQueryMatch::MatchedMementoCont {
                slot,
                kind,
                idx: next_idx,
            } => {
                idx = next_idx;
                Some((slot, kind))
            }
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
        None
    };

    scan_memento_continuations_hot(table, geom, low_m, high_m, idx, prefetched_cont)
}

#[inline(always)]
fn check_partition_slot0_high(geom: &KeepsakeGeometry, high: u64, rem_full: u64) -> Option<u8> {
    let fresh_sentinel = 1u64 << geom.r;
    let expected_fresh_high = fresh_sentinel | rem_full;
    if high == expected_fresh_high {
        Some(geom.r)
    } else if high >= fresh_sentinel {
        None
    } else {
        let (age, rem_bits, box_rem) = decode_age_remainder(geom.r, high);
        if box_rem == (rem_full >> age) {
            Some(rem_bits)
        } else {
            None
        }
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

    let Some(rem_bits) = check_partition_slot0_high(geom, high, rem_full) else {
        return (
            false,
            skip_box_after(table, box_start + 1, kind0),
            high,
            kind0,
        );
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
                let next = skip_box_after(table, idx + 1, kind);
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
