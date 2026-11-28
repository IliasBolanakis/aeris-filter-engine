//! Data Plane [`AerisFilter`] implementing [`RangeFilter`] over 64-slot
//! Quotient Filter chunks, keepsake boxes, and InfiniFilter unary age counters.
//!
//! # Key Invariants
//! 1. **Zero False Negatives:** Every inserted, non-deleted key returns `true`
//!    for point queries and enclosing range queries across all adaptations,
//!    expansions, and selective/opportunistic rejuvenations.
//! 2. **Monotonic Adaptivity:** Once [`RangeFilter::adapt`] completes for an
//!    empty range `[a, b]`, `query_range([a, b])` returns `false` across all
//!    subsequent adaptations and `2x` expansions (until a key in `[a, b]` is
//!    inserted).
//! 3. **Global Fingerprint Uniqueness & Prefix-Freeness:** Every keepsake box
//!    carries a unique canonical [`Fingerprint`], and all boxes with
//!    `bit_len >= q + r` in the same quotient run are mutually prefix-free.
//! 4. **Zero-Allocation Query Hot Path:** [`RangeFilter::query_point`] and
//!    [`RangeFilter::query_range`] perform zero heap allocations and inline
//!    SIMD rank/select and cached 64-bit hash extraction.

mod adapt;
mod adapt_fast;
mod adapt_planner;
mod bulk_load;
mod expand;
mod insert;
mod invariants;
mod query;
mod radix;
#[cfg(test)]
mod tests;

use crate::chunk::QuotientTable;
use crate::keepsake::{BoxPayload, KeepsakeBox, KeepsakeGeometry};
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, ExpansionReport, FilterConfig, Fingerprint,
    KeepsakeSlotKind, Key, Memento, PartitionHasher, PartitionId, QueryRange, RangeFilter,
    ReverseMapStore,
};

/// Production Data Plane Aeris Range Filter residing in RAM.
///
/// Combines a 64-slot chunked Quotient Filter with SIMD `POPCNT`/`PDEP`
/// navigation, Aeris keepsake box encoding, extendable fingerprints, and
/// InfiniFilter unary age counters supporting unbounded `2x` expansions.
#[derive(Clone, Debug)]
pub struct AerisFilter {
    pub(super) config: FilterConfig,
    pub(super) hasher: PartitionHasher,
    pub(super) geom: KeepsakeGeometry,
    pub(super) table: QuotientTable,
    pub(super) num_keys: usize,
    pub(super) occupied_slots: usize,
    pub(super) max_occupied_slots: usize,
    pub(super) expansion_cycles: u64,
}

impl AerisFilter {
    /// Constructs an empty [`AerisFilter`] from a validated [`FilterConfig`].
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` violates any formal
    /// mathematical invariant (`N = 2^q`, `r + m = C`, `r >= r_min(epsilon)`,
    /// `max_load_factor <= 0.95`).
    ///
    /// # Complexity
    /// `O(2^q)` time and space.
    pub fn new(config: FilterConfig) -> AerisResult<Self> {
        config.validate()?;
        let hasher = PartitionHasher::new(config.hash_seed);
        let geom = KeepsakeGeometry::from_config(&config);
        let table = QuotientTable::new(config.num_slots(), geom.slot_bits);
        let max_occupied_slots =
            ((config.num_slots() as f64) * config.max_load_factor).floor() as usize;
        tracing::debug!(
            q = config.q,
            r = config.r,
            m = config.m,
            capacity = config.num_slots(),
            slot_bits = geom.slot_bits,
            "initialized empty AerisFilter"
        );
        Ok(Self {
            config,
            hasher,
            geom,
            table,
            num_keys: 0,
            occupied_slots: 0,
            max_occupied_slots,
            expansion_cycles: 0,
        })
    }

    /// Returns the total heap memory footprint of the filter's metadata chunks,
    /// overflow offset map, and bit-packed slot payload array in bytes.
    #[inline]
    pub fn memory_usage_bytes(&self) -> usize {
        self.table.memory_usage_bytes()
    }

    /// Returns the number of canonical 64-slot Quotient Filter chunks (`N / 64`).
    #[inline]
    pub fn num_chunks(&self) -> usize {
        self.table.canonical_chunks()
    }

    /// Returns the total number of physical Quotient Filter slots currently occupied
    /// by keepsake boxes (including extension and continuation slots).
    #[inline]
    pub const fn occupied_slots(&self) -> usize {
        self.occupied_slots
    }

    /// Returns the deterministic [`PartitionHasher`] bound to this filter.
    #[inline]
    pub const fn hasher(&self) -> PartitionHasher {
        self.hasher
    }

    #[inline]
    pub(super) fn singleton_payload(&self, mem: Memento) -> BoxPayload {
        if self.config.m > 0 {
            BoxPayload::Mementos(vec![mem])
        } else {
            BoxPayload::PointCount(1)
        }
    }

    #[inline]
    pub(super) fn check_can_add_slots(&self, delta_slots: isize) -> AerisResult<()> {
        if delta_slots <= 0 {
            return Ok(());
        }
        let projected = self.occupied_slots + (delta_slots as usize);
        if projected <= self.max_occupied_slots {
            return Ok(());
        }
        let projected_alpha = (projected as f64) / (self.capacity() as f64);
        if projected_alpha > self.config.max_load_factor {
            tracing::warn!(
                current_alpha = self.load_factor(),
                projected_alpha,
                max_load_factor = self.config.max_load_factor,
                capacity = self.capacity(),
                delta_slots,
                "mutation rejected: filter load factor ceiling reached"
            );
            return Err(AerisError::FilterFull {
                load_factor: self.load_factor(),
                capacity: self.capacity(),
            });
        }
        Ok(())
    }

    pub(super) fn decode_run(&self, home: usize) -> Vec<(usize, usize, KeepsakeBox)> {
        let Some((rs, re)) = self.table.locate_run(home) else {
            return Vec::new();
        };
        let mut boxes = Vec::new();
        let mut cursor = rs;
        while cursor <= re {
            let (bx, len) = KeepsakeBox::decode_from_table(&self.table, &self.geom, cursor);
            boxes.push((cursor, len, bx));
            cursor += len;
        }
        boxes
    }

    pub(super) fn encode_run_boxes(&self, boxes: &[KeepsakeBox]) -> Vec<(u64, KeepsakeSlotKind)> {
        let total_slots: usize = boxes.iter().map(|b| b.encoded_slot_len(&self.geom)).sum();
        let mut out = Vec::with_capacity(total_slots);
        for (idx, bx) in boxes.iter().enumerate() {
            let is_run_end = idx + 1 == boxes.len();
            bx.encode_into(&self.geom, is_run_end, &mut out);
        }
        out
    }

    /// Extracts `(home_quotient, remainder)` directly from `w0` when `q + r <= 64`,
    /// falling back to multi-word hash extraction for wider fingerprints.
    #[inline]
    pub(super) fn quotient_and_remainder_hot(&self, partition: PartitionId) -> (u64, u64) {
        let q = self.config.q;
        let r = self.config.r;
        if (q as u16) + (r as u16) <= 64 {
            let w0 = self.hasher.word(partition, 0);
            let h = w0 >> (64 - q);
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (h, (w0 >> rem_shift) & rem_mask)
        } else {
            self.hasher.quotient_and_remainder(partition, q, r)
        }
    }

    pub(super) fn payload_from_key_slice(&self, keys: &[Key]) -> BoxPayload {
        if self.config.m > 0 {
            let mut mems: Vec<Memento> = keys.iter().map(|&k| self.config.memento_of(k)).collect();
            if mems.len() > 1 {
                mems.sort_unstable();
            }
            BoxPayload::Mementos(mems)
        } else {
            BoxPayload::PointCount(keys.len())
        }
    }
}

impl RangeFilter for AerisFilter {
    #[inline]
    fn config(&self) -> &FilterConfig {
        &self.config
    }

    fn insert(&mut self, key: Key) -> AerisResult<Fingerprint> {
        self.insert_key_internal(key, 0, true)
    }

    fn delete(&mut self, key: Key) -> AerisResult<Option<Fingerprint>> {
        self.delete_internal(key)
    }

    #[inline]
    fn query_point(&self, key: Key) -> bool {
        self.query_point_hot(key)
    }

    #[inline]
    fn query_range(&self, range: QueryRange) -> bool {
        self.query_range_hot(range)
    }

    fn locate_matching_fingerprints(&self, range: QueryRange) -> Vec<Fingerprint> {
        self.locate_matching_fingerprints_internal(range)
    }

    fn adapt(
        &mut self,
        range: QueryRange,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<AdaptationReport> {
        self.adapt_internal(range, reverse_map)
    }

    fn expand(&mut self, reverse_map: &dyn ReverseMapStore) -> AerisResult<ExpansionReport> {
        self.expand_internal(reverse_map)
    }

    #[inline]
    fn len(&self) -> usize {
        self.num_keys
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.canonical_slots()
    }

    #[inline]
    fn load_factor(&self) -> f64 {
        (self.occupied_slots as f64) / (self.capacity() as f64)
    }

    #[inline]
    fn expansion_cycles(&self) -> u64 {
        self.expansion_cycles
    }
}
