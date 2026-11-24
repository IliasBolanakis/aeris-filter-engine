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

use crate::chunk::QuotientTable;
use crate::keepsake::{
    box_matches_partition_hot, box_matches_query_hot_with_age_hint,
    encode_unextended_fresh_box_into, BoxPayload, KeepsakeBox, KeepsakeGeometry,
};
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, ExpansionReport, FilterConfig, Fingerprint,
    KeepsakeSlotKind, Key, Memento, PartitionHasher, PartitionId, QueryRange, RangeFilter,
    ReverseMapStore,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// Maximum `(Fingerprint, Key)` entries buffered per streaming Reverse Map batch
/// during [`AerisFilter::bulk_load`] (`~3 MB` constant peak memory).
const REVERSE_MAP_STREAM_CHUNK: usize = 65_536;

/// Bits per LSD radix sort pass (`8`-bit byte passes -> `256` histogram buckets).
const RADIX_BITS: usize = 8;

/// Number of histogram buckets per 8-bit LSD radix pass (`2^8 = 256`).
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// Production Data Plane Aeris Range Filter residing in RAM.
///
/// Combines a 64-slot chunked Quotient Filter with SIMD `POPCNT`/`PDEP`
/// navigation, Aeris keepsake box encoding, extendable fingerprints, and
/// InfiniFilter unary age counters supporting unbounded `2x` expansions.
#[derive(Clone, Debug)]
pub struct AerisFilter {
    config: FilterConfig,
    hasher: PartitionHasher,
    geom: KeepsakeGeometry,
    table: QuotientTable,
    num_keys: usize,
    occupied_slots: usize,
    max_occupied_slots: usize,
    expansion_cycles: u64,
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

    /// Constructs an [`AerisFilter`] via linear-time `O(K + N)` 8-bit LSD radix
    /// sorting and single-pass streaming AoS block emission, persisting all
    /// `(fingerprint, key)` associations to `reverse_map` in bounded-memory chunks.
    ///
    /// # Errors
    /// - Returns [`AerisError::InvalidConfig`] if `config` is invalid.
    /// - Returns [`AerisError::FilterFull`] if the encoded keepsake boxes exceed
    ///   `config.max_load_factor` (before allocating the table or mutating `reverse_map`).
    /// - Returns [`AerisError::StorageError`] if `reverse_map.append_batch` fails.
    ///
    /// # Complexity
    /// `O(K * ceil((q + r + m) / 8) + N)` time (`K = keys.len()`, `N = 2^q`),
    /// peak auxiliary memory `16 B/key` during radix sort (dropped before table allocation),
    /// and constant `~3 MB` Reverse Map streaming buffer (`65,536` entries/batch).
    pub fn bulk_load(
        config: FilterConfig,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let start = Instant::now();
        config.validate()?;
        if keys.is_empty() {
            return Self::new(config);
        }

        let total_bits = (config.q as u16) + (config.r as u16) + (config.m as u16);
        let filter = if total_bits <= 64 {
            Self::bulk_load_radix_streaming(config, keys, reverse_map)?
        } else {
            Self::bulk_load_wide_fallback(config, keys, reverse_map)?
        };

        tracing::debug!(
            keys = keys.len(),
            occupied_slots = filter.occupied_slots,
            load_factor = filter.load_factor(),
            latency_ns = start.elapsed().as_nanos() as u64,
            "completed bulk_load"
        );
        Ok(filter)
    }

    /// Fast streaming `bulk_load` path when `q + r + m <= 64`: packs
    /// `(home, rem, mem)` into `Vec<u64>`, sorts via single-prepass 8-bit LSD
    /// radix sort, validates load factor in a read-only pass, streams directly
    /// into the AoS [`QuotientTable`], and streams Reverse Map writes in
    /// constant-memory chunks.
    fn bulk_load_radix_streaming(
        config: FilterConfig,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let q = config.q;
        let r = config.r;
        let m = config.m;
        let hasher = PartitionHasher::new(config.hash_seed);
        let geom = KeepsakeGeometry::from_config(&config);

        let q_plus_r = q + r;
        let rem_shift = 64 - (q_plus_r as u32);
        let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
        let m_mask = config.memento_mask();
        let m_shift = m as u32;
        let rm_shift = (r + m) as u32;

        // Stage 1: Pack `(home, rem, mem)` into 8 B/key `u64` keys via trusted-len iterator
        // (eliminating per-key `push` capacity checks and redundant home/rem masks).
        let mut packed: Vec<u64> = Vec::with_capacity(keys.len());
        if m > 0 {
            packed.extend(keys.iter().map(|&key| {
                let p = key >> m_shift;
                let mem = key & m_mask;
                let w0 = hasher.word(p, 0);
                ((w0 >> rem_shift) << m_shift) | mem
            }));
        } else {
            packed.extend(keys.iter().map(|&key| {
                let w0 = hasher.word(key, 0);
                w0 >> rem_shift
            }));
        }

        // Stage 2: 8-bit LSD radix sort with stack write-combining buffers; temporary
        // scratch buffer is dropped immediately before allocating the QuotientTable.
        let total_bits = (q as usize) + (r as usize) + (m as usize);
        radix_sort_u64(&mut packed, total_bits);

        // Stage 3: Read-only slot counting pass (zero allocations) & load ceiling enforcement
        // BEFORE allocating the QuotientTable or mutating `reverse_map`.
        let mut total_slots = 0usize;
        let mut i = 0usize;
        let n_packed = packed.len();
        while i < n_packed {
            let hr = packed[i] >> m_shift;
            let group_start = i;
            i += 1;
            while i < n_packed && (packed[i] >> m_shift) == hr {
                i += 1;
            }
            let cnt = i - group_start;
            total_slots += if cnt == 1 {
                1
            } else {
                geom.unextended_box_slot_len(cnt)
            };
        }

        let capacity = config.num_slots();
        let projected_alpha = (total_slots as f64) / (capacity as f64);
        if projected_alpha > config.max_load_factor {
            tracing::warn!(
                projected_alpha,
                max_load_factor = config.max_load_factor,
                capacity,
                keys = keys.len(),
                "bulk_load rejected: projected load factor exceeds ceiling"
            );
            return Err(AerisError::FilterFull {
                load_factor: projected_alpha,
                capacity,
            });
        }

        // Stage 4: Single-pass block-sequential AoS emission with stack-buffered
        // header/payload accumulation (zero per-slot DRAM read-modify-write).
        let mut filter = Self::new(config)?;
        let fresh_sentinel = 1u64 << r;
        let mut run_buf: Vec<(u64, KeepsakeSlotKind)> = Vec::with_capacity(16);
        let mut mem_buf: Vec<Memento> = Vec::with_capacity(16);

        {
            let mut writer = filter.table.bulk_writer();
            let mut i = 0usize;
            while i < n_packed {
                let first = packed[i];
                let home = (first >> rm_shift) as usize;

                // Fast path (>90% of runs): single-key, single-slot run for `home`.
                if i + 1 == n_packed || ((packed[i + 1] >> rm_shift) as usize) != home {
                    let rem = (first >> m_shift) & rem_mask;
                    let high = fresh_sentinel | rem;
                    let slot0 = if m > 0 {
                        (high << m_shift) | (first & m_mask)
                    } else {
                        high
                    };
                    writer.append_single_slot_run(home, slot0);
                    i += 1;
                    continue;
                }

                run_buf.clear();
                while i < n_packed && ((packed[i] >> rm_shift) as usize) == home {
                    let hr = packed[i] >> m_shift;
                    let rem = hr & rem_mask;
                    let box_start = i;
                    if m > 0 {
                        mem_buf.clear();
                        while i < n_packed && (packed[i] >> m_shift) == hr {
                            mem_buf.push(packed[i] & m_mask);
                            i += 1;
                        }
                    } else {
                        while i < n_packed && (packed[i] >> m_shift) == hr {
                            i += 1;
                        }
                    }
                    let count = i - box_start;
                    let is_run_end = i == n_packed || ((packed[i] >> rm_shift) as usize) != home;
                    encode_unextended_fresh_box_into(
                        &filter.geom,
                        rem,
                        if m > 0 { &mem_buf } else { &[] },
                        count,
                        is_run_end,
                        &mut run_buf,
                    );
                }

                writer.append_run(home, &run_buf);
            }
            writer.finish();
        }
        filter.occupied_slots = total_slots;
        filter.num_keys = keys.len();
        drop(packed);

        // Stage 5: Constant-memory chunked Reverse Map streaming (~3 MB peak buffer),
        // completely bypassed when `reverse_map.is_noop()` is `true`.
        if !reverse_map.is_noop() {
            let chunk_cap = REVERSE_MAP_STREAM_CHUNK.min(keys.len());
            let mut rm_chunk: Vec<(Fingerprint, Key)> = Vec::with_capacity(chunk_cap);
            for chunk in keys.chunks(REVERSE_MAP_STREAM_CHUNK) {
                rm_chunk.clear();
                for &key in chunk {
                    let p = filter.config.partition_of(key);
                    let w0 = filter.hasher.word(p, 0);
                    rm_chunk.push((Fingerprint::from_u64_prefix(w0, q_plus_r), key));
                }
                reverse_map.append_batch(&rm_chunk)?;
            }
        }

        Ok(filter)
    }

    /// General fallback `bulk_load` path for wide configurations where `q + r + m > 64`.
    fn bulk_load_wide_fallback(
        config: FilterConfig,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let q = config.q;
        let r = config.r;
        let m = config.m;
        let hasher = PartitionHasher::new(config.hash_seed);
        let geom = KeepsakeGeometry::from_config(&config);

        let mut projected: Vec<(usize, u64, Memento)> = Vec::with_capacity(keys.len());
        for &key in keys {
            let p = config.partition_of(key);
            let mem = config.memento_of(key);
            let (home_u64, rem) = hasher.quotient_and_remainder(p, q, r);
            projected.push((home_u64 as usize, rem, mem));
        }
        projected.sort_unstable();

        // Read-only slot counting pass & load ceiling check BEFORE table allocation or RM writes.
        let mut total_slots = 0usize;
        let mut i = 0usize;
        while i < projected.len() {
            let (home, rem, _) = projected[i];
            let start_i = i;
            i += 1;
            while i < projected.len() && projected[i].0 == home && projected[i].1 == rem {
                i += 1;
            }
            total_slots += geom.unextended_box_slot_len(i - start_i);
        }

        let capacity = config.num_slots();
        let projected_alpha = (total_slots as f64) / (capacity as f64);
        if projected_alpha > config.max_load_factor {
            tracing::warn!(
                projected_alpha,
                max_load_factor = config.max_load_factor,
                capacity,
                keys = keys.len(),
                "bulk_load rejected: projected load factor exceeds ceiling"
            );
            return Err(AerisError::FilterFull {
                load_factor: projected_alpha,
                capacity,
            });
        }

        let mut filter = Self::new(config)?;
        let mut run_buf: Vec<(u64, KeepsakeSlotKind)> = Vec::with_capacity(16);
        let mut mem_buf: Vec<Memento> = Vec::with_capacity(16);
        let mut write_cursor = 0usize;

        let mut i = 0usize;
        while i < projected.len() {
            let home = projected[i].0;
            run_buf.clear();

            while i < projected.len() && projected[i].0 == home {
                let rem = projected[i].1;
                let box_start = i;
                if m > 0 {
                    mem_buf.clear();
                    while i < projected.len() && projected[i].0 == home && projected[i].1 == rem {
                        mem_buf.push(projected[i].2);
                        i += 1;
                    }
                } else {
                    while i < projected.len() && projected[i].0 == home && projected[i].1 == rem {
                        i += 1;
                    }
                }
                let count = i - box_start;
                let is_run_end = i == projected.len() || projected[i].0 != home;
                encode_unextended_fresh_box_into(
                    &filter.geom,
                    rem,
                    if m > 0 { &mem_buf } else { &[] },
                    count,
                    is_run_end,
                    &mut run_buf,
                );
            }

            filter
                .table
                .bulk_append_run(home, &run_buf, &mut write_cursor);
        }
        filter.occupied_slots = total_slots;
        filter.num_keys = keys.len();
        drop(projected);

        if !reverse_map.is_noop() {
            let chunk_cap = REVERSE_MAP_STREAM_CHUNK.min(keys.len());
            let mut rm_chunk: Vec<(Fingerprint, Key)> = Vec::with_capacity(chunk_cap);
            for chunk in keys.chunks(REVERSE_MAP_STREAM_CHUNK) {
                rm_chunk.clear();
                for &key in chunk {
                    let p = filter.config.partition_of(key);
                    let (home_u64, rem) = filter.hasher.quotient_and_remainder(p, q, r);
                    rm_chunk.push((Fingerprint::from_parts(home_u64, q, rem, r, &[]), key));
                }
                reverse_map.append_batch(&rm_chunk)?;
            }
        }

        Ok(filter)
    }

    /// Rebuilds an [`AerisFilter`] from an arbitrary key iterator under `config`,
    /// populating `reverse_map` with the rebuilt `(fingerprint, key)` mappings.
    ///
    /// # Errors
    /// Propagates validation, capacity (`FilterFull`), or storage errors from [`Self::bulk_load`].
    pub fn rebuild_from_keys(
        config: FilterConfig,
        keys: impl IntoIterator<Item = Key>,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let collected: Vec<Key> = keys.into_iter().collect();
        Self::bulk_load(config, &collected, reverse_map)
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

    /// Verifies all structural Quotient Filter metadata invariants, keepsake box
    /// encoding invariants, key/slot accounting, and global fingerprint uniqueness.
    ///
    /// Intended for automated testing, fuzzing, and diagnostic audits.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on the first invariant violation.
    pub fn verify_invariants(&self) -> AerisResult<()> {
        self.table.verify_metadata_invariants()?;

        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);

        let mut counted_slots = 0usize;
        let mut counted_keys = 0usize;
        let mut seen_fps: BTreeSet<Fingerprint> = BTreeSet::new();

        for home in self.table.occupied_homes() {
            let (rs, re) = self.table.locate_run(home).ok_or_else(|| {
                AerisError::InternalInvariant(format!("occupied home {home} missing run"))
            })?;
            let mut cursor = rs;
            let mut full_length_fps: Vec<Fingerprint> = Vec::new();

            while cursor <= re {
                let (bx, consumed) =
                    KeepsakeBox::decode_from_table(&self.table, &self.geom, cursor);
                if consumed == 0 || cursor + consumed - 1 > re {
                    return Err(AerisError::InternalInvariant(format!(
                        "keepsake box at slot {cursor} overran run end {re}"
                    )));
                }
                // Verify 3-bit invariant: extension slot never ends a keepsake box.
                let end_slot = cursor + consumed - 1;
                let end_kind = self.table.slot_kind(end_slot);
                if end_kind.is_extension_slot() || !end_kind.is_box_end() {
                    return Err(AerisError::InternalInvariant(format!(
                        "terminal slot {end_slot} of keepsake box has invalid kind {end_kind:?}"
                    )));
                }
                if end_slot == re && !end_kind.is_run_end() {
                    return Err(AerisError::InternalInvariant(format!(
                        "run end slot {re} is not BoxAndRunEnd"
                    )));
                }
                if end_slot < re && end_kind.is_run_end() {
                    return Err(AerisError::InternalInvariant(format!(
                        "interior box end slot {end_slot} prematurely marked BoxAndRunEnd"
                    )));
                }
                if bx.age > r {
                    return Err(AerisError::InternalInvariant(format!(
                        "keepsake box age {} exceeds r={r}",
                        bx.age
                    )));
                }
                if bx.payload.is_empty() {
                    return Err(AerisError::InternalInvariant(format!(
                        "empty keepsake box at home {home}, slot {cursor}"
                    )));
                }
                if let BoxPayload::Mementos(ref mems) = bx.payload {
                    if !mems.windows(2).all(|w| w[0] <= w[1]) {
                        return Err(AerisError::InternalInvariant(format!(
                            "unsorted mementos in keepsake box at home {home}, slot {cursor}"
                        )));
                    }
                }

                let fp = bx.canonical_fingerprint(home, q, r);
                if !seen_fps.insert(fp.clone()) {
                    return Err(AerisError::InternalInvariant(format!(
                        "duplicate canonical fingerprint at home {home}"
                    )));
                }
                if fp.bit_len() >= fresh_base {
                    full_length_fps.push(fp);
                }

                counted_slots += consumed;
                counted_keys += bx.payload.key_count();
                cursor += consumed;
            }

            // Verify pairwise prefix-freeness among all boxes with `bit_len >= q + r` in this run.
            for i in 0..full_length_fps.len() {
                for j in (i + 1)..full_length_fps.len() {
                    if full_length_fps[i].is_prefix_of(&full_length_fps[j])
                        || full_length_fps[j].is_prefix_of(&full_length_fps[i])
                    {
                        return Err(AerisError::InternalInvariant(format!(
                            "prefix conflict between full-length boxes in home {home}"
                        )));
                    }
                }
            }
        }

        if counted_slots != self.occupied_slots {
            return Err(AerisError::InternalInvariant(format!(
                "occupied_slots accounting mismatch: tracked={}, counted={counted_slots}",
                self.occupied_slots
            )));
        }
        if counted_keys != self.num_keys {
            return Err(AerisError::InternalInvariant(format!(
                "num_keys accounting mismatch: tracked={}, counted={counted_keys}",
                self.num_keys
            )));
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Internal hot-path query helper (strictly zero heap allocation)
    // -------------------------------------------------------------------------

    /// Evaluates a single partition subquery `(partition, low_m, high_m)` with
    /// zero heap allocation.
    #[inline]
    fn query_partition_hot(&self, partition: PartitionId, low_m: Memento, high_m: Memento) -> bool {
        let q = self.config.q;
        let r = self.config.r;
        let w0 = self.hasher.word(partition, 0);

        let (home_u64, rem_full) = if (q as u16) + (r as u16) <= 64 {
            let h = w0 >> (64 - q);
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (h, (w0 >> rem_shift) & rem_mask)
        } else {
            self.hasher.quotient_and_remainder(partition, q, r)
        };

        let home = home_u64 as usize;
        let Some((run_start, run_end)) = self.table.locate_run(home) else {
            return false;
        };

        let has_aged = self.expansion_cycles > 0;
        let mut cursor = run_start;
        while cursor <= run_end {
            let (matched, next_slot) = box_matches_query_hot_with_age_hint(
                &self.table,
                &self.geom,
                &self.hasher,
                partition,
                w0,
                q,
                rem_full,
                low_m,
                high_m,
                cursor,
                has_aged,
            );
            if matched {
                return true;
            }
            cursor = next_slot;
        }
        false
    }

    // -------------------------------------------------------------------------
    // Internal mutation & planning helpers
    // -------------------------------------------------------------------------

    #[inline]
    fn singleton_payload(&self, mem: Memento) -> BoxPayload {
        if self.config.m > 0 {
            BoxPayload::Mementos(vec![mem])
        } else {
            BoxPayload::PointCount(1)
        }
    }

    #[inline]
    fn check_can_add_slots(&self, delta_slots: isize) -> AerisResult<()> {
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

    fn decode_run(&self, home: usize) -> Vec<(usize, usize, KeepsakeBox)> {
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

    fn encode_run_boxes(&self, boxes: &[KeepsakeBox]) -> Vec<(u64, KeepsakeSlotKind)> {
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
    fn quotient_and_remainder_hot(&self, partition: PartitionId) -> (u64, u64) {
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

    /// Inserts `key` with a minimum canonical fingerprint width `min_fp_bits`
    /// (`0` for normal inserts; `old_fp.bit_len()` during selective rejuvenation).
    fn insert_key_internal(
        &mut self,
        key: Key,
        min_fp_bits: u16,
        check_load_limit: bool,
    ) -> AerisResult<Fingerprint> {
        let q = self.config.q;
        let r = self.config.r;
        let m = self.config.m;
        let p = self.config.partition_of(key);
        let mem = self.config.memento_of(key);
        let single_word = (q as u16) + (r as u16) <= 64;
        let (w0, home_u64, rem_full) = if single_word {
            let w0 = self.hasher.word(p, 0);
            let h = w0 >> (64 - q);
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (w0, h, (w0 >> rem_shift) & rem_mask)
        } else {
            let (h, rem) = self.hasher.quotient_and_remainder(p, q, r);
            (0, h, rem)
        };
        let home = home_u64 as usize;

        if min_fp_bits == 0 {
            let fresh_sentinel = 1u64 << r;
            let high = fresh_sentinel | rem_full;
            let slot0 = if m > 0 {
                (high << m) | (mem & self.geom.m_mask)
            } else {
                high
            };

            // Fast Path A: Unoccupied home slot (zero heap allocations, zero backward scans).
            let Some((run_start, run_end)) = self.table.locate_run(home) else {
                if check_load_limit {
                    self.check_can_add_slots(1)?;
                }
                self.table.insert_single_slot_new_run(home, slot0);
                self.occupied_slots += 1;
                self.num_keys += 1;
                return Ok(if single_word {
                    Fingerprint::from_u64_prefix(w0, q + r)
                } else {
                    Fingerprint::from_parts(home_u64, q, rem_full, r, &[])
                });
            };

            // Fast Path B: Occupied home slot — zero-allocation linear run scan comparing
            // `box_high` directly against `high = (1 << r) | rem_full` (zero `LZCNT`).
            let mut cursor = run_start;
            let mut has_extension = false;
            let mut insert_pos: Option<usize> = None;
            let mut matched_box: Option<(usize, u64, KeepsakeSlotKind)> = None;

            while cursor <= run_end {
                let (box_slot0, kind0) = self.table.get_payload_and_kind(cursor);
                if kind0 == KeepsakeSlotKind::Extension {
                    has_extension = true;
                    break;
                }
                let box_high = box_slot0 >> m;
                if box_high == high {
                    matched_box = Some((cursor, box_slot0, kind0));
                } else if insert_pos.is_none() && (box_high < fresh_sentinel || box_high > high) {
                    insert_pos = Some(cursor);
                }
                cursor = if kind0.is_box_end() {
                    cursor + 1
                } else {
                    self.table.find_box_end(cursor + 1) + 1
                };
            }

            if !has_extension {
                match matched_box {
                    // Subcase B1 (>99% of occupied-home inserts): brand-new unextended box.
                    None => {
                        if check_load_limit {
                            self.check_can_add_slots(1)?;
                        }
                        let pos = insert_pos.unwrap_or(run_end + 1);
                        self.table
                            .insert_single_slot_in_existing_run(home, pos, run_end, slot0);
                        self.occupied_slots += 1;
                        self.num_keys += 1;
                        return Ok(if single_word {
                            Fingerprint::from_u64_prefix(w0, q + r)
                        } else {
                            Fingerprint::from_parts(home_u64, q, rem_full, r, &[])
                        });
                    }
                    // Subcase B2: exact match with single-slot unextended age=0 box (`m > 0`).
                    Some((box_start, existing_slot0, kind0)) if m > 0 && kind0.is_box_end() => {
                        if check_load_limit {
                            self.check_can_add_slots(1)?;
                        }
                        let m0 = existing_slot0 & self.geom.m_mask;
                        let mem_masked = mem & self.geom.m_mask;
                        let (lo, hi) = if m0 <= mem_masked {
                            (m0, mem_masked)
                        } else {
                            (mem_masked, m0)
                        };
                        let new_slot0 = (high << m) | lo;
                        let new_slot1 = hi << self.geom.cont_header_bits;
                        self.table.expand_single_slot_to_two(
                            home, box_start, new_slot0, new_slot1, kind0,
                        );
                        self.occupied_slots += 1;
                        self.num_keys += 1;
                        return Ok(if single_word {
                            Fingerprint::from_u64_prefix(w0, q + r)
                        } else {
                            Fingerprint::from_parts(home_u64, q, rem_full, r, &[])
                        });
                    }
                    // Subcase B3: multi-slot continuation box or point mode (`m == 0`) count >= 2.
                    Some(_) => {}
                }
            }
        }

        let fresh_base = (q as u16) + (r as u16);
        let min_req = fresh_base.max(min_fp_bits);

        if !self.table.is_occupied(home) {
            let ext_level = self
                .geom
                .min_ext_level_for_bits(min_req.saturating_sub(fresh_base));
            let bx = KeepsakeBox::new_fresh(
                &self.hasher,
                p,
                q,
                &self.geom,
                ext_level,
                self.singleton_payload(mem),
            );
            let needed = bx.encoded_slot_len(&self.geom);
            if check_load_limit {
                self.check_can_add_slots(needed as isize)?;
            }
            let encoded = bx.encode(&self.geom, true);
            self.table.replace_region(home, 0, 0, &encoded);
            self.occupied_slots += needed;
            self.num_keys += 1;
            return Ok(bx.canonical_fingerprint(home, q, r));
        }

        let boxes = self.decode_run(home);
        let mut peer_full_fps: Vec<Fingerprint> = Vec::new();
        let mut exact_match_idx: Option<usize> = None;

        for (idx, (_, _, bx)) in boxes.iter().enumerate() {
            if bx.total_fp_bits(q, r) >= fresh_base {
                if bx.matches_partition(&self.hasher, p, q, r) {
                    debug_assert!(
                        bx.total_fp_bits(q, r) >= min_req,
                        "existing matching full-length box ({} bits) shorter than min_req ({min_req})",
                        bx.total_fp_bits(q, r)
                    );
                    exact_match_idx = Some(idx);
                    break;
                }
                let peer_fp = bx.canonical_fingerprint(home, q, r);
                if peer_fp.extract_bits(q as u16, r) == rem_full {
                    peer_full_fps.push(peer_fp);
                }
            }
        }

        if let Some(idx) = exact_match_idx {
            let (box_start, old_len, mut target) = boxes[idx].clone();
            let is_run_end = idx + 1 == boxes.len();
            target.payload.insert_memento(mem);
            let new_len = target.encoded_slot_len(&self.geom);
            let delta = (new_len as isize) - (old_len as isize);
            if check_load_limit {
                self.check_can_add_slots(delta)?;
            }
            let encoded = target.encode(&self.geom, is_run_end);
            self.table
                .replace_region(home, box_start, old_len, &encoded);
            self.occupied_slots = ((self.occupied_slots as isize) + delta) as usize;
            self.num_keys += 1;
            return Ok(target.canonical_fingerprint(home, q, r));
        }

        // No existing full-length box matched `p`; compute minimal prefix-free `ext_level`.
        let mut ext_level = self
            .geom
            .min_ext_level_for_bits(min_req.saturating_sub(fresh_base));
        if !peer_full_fps.is_empty() {
            loop {
                let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
                let cand_fp = Fingerprint::from_hasher(&self.hasher, p, bit_len);
                let conflicts = peer_full_fps
                    .iter()
                    .any(|peer| cand_fp.is_prefix_of(peer) || peer.is_prefix_of(&cand_fp));
                if !conflicts {
                    break;
                }
                ext_level += 1;
            }
        }

        let new_box = KeepsakeBox::new_fresh(
            &self.hasher,
            p,
            q,
            &self.geom,
            ext_level,
            self.singleton_payload(mem),
        );
        let needed = new_box.encoded_slot_len(&self.geom);
        if check_load_limit {
            self.check_can_add_slots(needed as isize)?;
        }

        let new_sort_key = new_box.sort_key();
        let insert_idx = boxes.partition_point(|(_, _, b)| b.sort_key() <= new_sort_key);
        if insert_idx < boxes.len() {
            let insert_pos = boxes[insert_idx].0;
            let encoded = new_box.encode(&self.geom, false);
            self.table.replace_region(home, insert_pos, 0, &encoded);
        } else {
            let (_, run_end) = self
                .table
                .locate_run(home)
                .expect("home is occupied so run must exist");
            let encoded = new_box.encode(&self.geom, true);
            self.table.replace_region(home, run_end + 1, 0, &encoded);
        }

        self.occupied_slots += needed;
        self.num_keys += 1;
        Ok(new_box.canonical_fingerprint(home, q, r))
    }

    /// Fast in-place adaptation path for the >95% common case: a single matching
    /// subquery `(p_q, low_m, high_m)` hitting a single fresh (`age == 0`) box
    /// whose keys all belong to one colliding partition `p_other != p_q`, with no
    /// peer remainder collisions or extended aged boxes in the same quotient run.
    ///
    /// Returns `Ok(Some(report))` on fast-path completion, `Ok(None)` when a
    /// rarer multi-partition / aged / peer-collision case requires the general
    /// overlay planner, or `Err(...)` on `FilterFull` / `StorageError` (leaving
    /// both RAM table and `reverse_map` untouched).
    fn try_adapt_single_box_fast(
        &mut self,
        p_q: PartitionId,
        home: usize,
        reverse_map: &dyn ReverseMapStore,
        report: &mut AdaptationReport,
    ) -> AerisResult<Option<()>> {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        let Some((run_start, run_end)) = self.table.locate_run(home) else {
            return Ok(Some(()));
        };

        let w0_q = self.hasher.word(p_q, 0);
        let rem_q = if fresh_base <= 64 {
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (w0_q >> rem_shift) & rem_mask
        } else {
            self.hasher.quotient_and_remainder(p_q, q, r).1
        };
        let fresh_sentinel = 1u64 << r;
        let expected_fresh_high = fresh_sentinel | rem_q;

        // Single zero-allocation scan of `home`'s run to locate the sole matching box
        // and verify zero potential peer conflicts or extended aged boxes.
        let mut matched_box: Option<(usize, usize)> = None;
        let mut cursor = run_start;
        while cursor <= run_end {
            let (is_match, next_slot, high, kind0) = box_matches_partition_hot(
                &self.table,
                &self.geom,
                &self.hasher,
                p_q,
                w0_q,
                q,
                rem_q,
                cursor,
            );
            if is_match {
                if matched_box.is_some() || high < fresh_sentinel {
                    // Multiple matching boxes or an aged box (`age > 0` changes sort order).
                    return Ok(None);
                }
                matched_box = Some((cursor, next_slot - cursor));
            } else if high == expected_fresh_high
                || (high < fresh_sentinel && kind0 == KeepsakeSlotKind::Extension)
            {
                // Peer box with identical remainder or extended aged box: use general planner.
                return Ok(None);
            }
            cursor = next_slot;
        }

        let Some((box_start, box_len)) = matched_box else {
            return Ok(Some(()));
        };

        // Decode ONLY the single colliding box.
        let (bx, decoded_len) = KeepsakeBox::decode_from_table(&self.table, &self.geom, box_start);
        debug_assert_eq!(decoded_len, box_len);
        debug_assert_eq!(bx.age, 0);

        let old_fp = bx.canonical_fingerprint(home, q, r);
        let old_bits = old_fp.bit_len();
        let mut keys = reverse_map.get_keys(&old_fp)?;
        if keys.is_empty() {
            return Err(AerisError::StorageError(format!(
                "reverse map returned 0 keys for active keepsake box fingerprint ({old_bits} bits) at home {home}"
            )));
        }
        if keys.len() > 1 {
            keys.sort_unstable();
            keys.dedup();
        }

        let p_other = self.config.partition_of(keys[0]);
        if p_other == p_q {
            return Ok(None);
        }
        for &k in &keys[1..] {
            if self.config.partition_of(k) != p_other {
                return Ok(None);
            }
        }

        // Compute minimal `ext_level` diverging from `p_q` (peer-freeness is already guaranteed).
        let min_req = fresh_base.max(old_bits);
        let mut ext_level = self
            .geom
            .min_ext_level_for_bits(min_req.saturating_sub(fresh_base));
        let w0_other = self.hasher.word(p_other, 0);
        let diff_w0 = w0_other ^ w0_q;
        loop {
            let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
            let diverges = if bit_len <= 64 {
                (diff_w0 >> (64 - bit_len)) != 0
            } else {
                Fingerprint::from_hasher(&self.hasher, p_other, bit_len)
                    != Fingerprint::from_hasher(&self.hasher, p_q, bit_len)
            };
            if diverges {
                break;
            }
            ext_level += 1;
        }

        let payload = self.payload_from_key_slice(&keys);
        let new_box =
            KeepsakeBox::new_fresh(&self.hasher, p_other, q, &self.geom, ext_level, payload);
        let new_len = new_box.encoded_slot_len(&self.geom);
        let slot_delta = (new_len as isize) - (box_len as isize);
        let key_delta = (new_box.payload.key_count() as isize) - (bx.payload.key_count() as isize);

        // Enforce `max_load_factor` atomically BEFORE any Reverse Map or RAM mutation.
        self.check_can_add_slots(slot_delta)?;

        let new_fp = new_box.canonical_fingerprint(home, q, r);
        if keys.len() == 1 {
            let entry = [(new_fp, keys[0])];
            reverse_map.replace_fingerprint(&old_fp, &entry)?;
        } else {
            let new_rm_entries: Vec<(Fingerprint, Key)> =
                keys.iter().map(|&k| (new_fp.clone(), k)).collect();
            reverse_map.replace_fingerprint(&old_fp, &new_rm_entries)?;
        }

        let is_run_end = box_start + box_len - 1 == run_end;
        let encoded = new_box.encode(&self.geom, is_run_end);
        self.table
            .replace_region_known_run(home, run_start, run_end, box_start, box_len, &encoded);

        self.occupied_slots = ((self.occupied_slots as isize) + slot_delta) as usize;
        self.num_keys = ((self.num_keys as isize) + key_delta) as usize;
        report.boxes_split += 1;
        report.keys_rehashed += keys.len();
        report.extension_slots_added += new_len.saturating_sub(box_len);
        Ok(Some(()))
    }

    /// Plans adaptation for a single matching subquery `(p_q, low_m, high_m)` on
    /// flat `Vec`-backed overlay structures without mutating `self` or `reverse_map`.
    #[allow(clippy::too_many_arguments)]
    fn plan_subquery_adaptation(
        &self,
        p_q: PartitionId,
        low_m: Memento,
        high_m: Memento,
        reverse_map: &dyn ReverseMapStore,
        overlay_runs: &mut Vec<(usize, usize, usize, Vec<KeepsakeBox>)>,
        overlay_keys: &mut Vec<(Fingerprint, Vec<Key>)>,
        fresh_overlay_fps: &mut Vec<Fingerprint>,
        tombstoned_disk_fps: &mut Vec<Fingerprint>,
        report: &mut AdaptationReport,
    ) -> AerisResult<()> {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        let (home_u64, _) = self.quotient_and_remainder_hot(p_q);
        let home = home_u64 as usize;

        let run_idx = if let Some(pos) = overlay_runs.iter().position(|e| e.0 == home) {
            pos
        } else {
            if !self.table.is_occupied(home) {
                return Ok(());
            }
            let decoded_full = self.decode_run(home);
            let old_slots: usize = decoded_full.iter().map(|&(_, len, _)| len).sum();
            let old_keys: usize = decoded_full
                .iter()
                .map(|(_, _, bx)| bx.payload.key_count())
                .sum();
            let boxes: Vec<KeepsakeBox> = decoded_full.into_iter().map(|(_, _, bx)| bx).collect();
            overlay_runs.push((home, old_slots, old_keys, boxes));
            overlay_runs.len() - 1
        };

        let current_boxes = &overlay_runs[run_idx].3;
        let mut matched_indices: Vec<usize> = Vec::with_capacity(2);
        let mut has_memento_hit = false;

        for (idx, bx) in current_boxes.iter().enumerate() {
            if bx.matches_partition(&self.hasher, p_q, q, r) {
                matched_indices.push(idx);
                if bx.payload.contains_in_range(low_m, high_m) {
                    has_memento_hit = true;
                }
            }
        }

        if !has_memento_hit {
            return Ok(());
        }

        // Separate unmatched boxes from matched boxes.
        let mut remaining_boxes: Vec<KeepsakeBox> =
            Vec::with_capacity(current_boxes.len().saturating_sub(matched_indices.len()) + 2);
        let mut matched_boxes: Vec<KeepsakeBox> = Vec::with_capacity(matched_indices.len());
        let mut next_match_ptr = 0usize;
        for (idx, bx) in current_boxes.iter().enumerate() {
            if next_match_ptr < matched_indices.len() && matched_indices[next_match_ptr] == idx {
                matched_boxes.push(bx.clone());
                next_match_ptr += 1;
            } else {
                remaining_boxes.push(bx.clone());
            }
        }

        // Retrieve live keys for every matched box, recording per-partition monotonicity floors.
        let mut colliding_parts: Vec<(PartitionId, u16, Vec<Key>)> = Vec::with_capacity(2);
        let mut query_part_min_bits: u16 = 0;
        let mut query_part_keys: Vec<Key> = Vec::new();
        let mut old_matched_slots = 0usize;

        for bx in &matched_boxes {
            old_matched_slots += bx.encoded_slot_len(&self.geom);
            if bx.age > 0 {
                report.rejuvenated_boxes += 1;
            }
            let old_fp = bx.canonical_fingerprint(home, q, r);
            let old_bits = old_fp.bit_len();

            let keys_for_box: Vec<Key> = if let Some(fpos) =
                fresh_overlay_fps.iter().position(|f| f == &old_fp)
            {
                fresh_overlay_fps.swap_remove(fpos);
                if let Some(kpos) = overlay_keys.iter().position(|(f, _)| f == &old_fp) {
                    overlay_keys.swap_remove(kpos).1
                } else {
                    Vec::new()
                }
            } else {
                let mut combined = reverse_map.get_keys(&old_fp)?;
                if let Some(kpos) = overlay_keys.iter().position(|(f, _)| f == &old_fp) {
                    combined.extend(overlay_keys.swap_remove(kpos).1);
                }
                if combined.len() > 1 {
                    combined.sort_unstable();
                    combined.dedup();
                }
                if combined.is_empty() {
                    return Err(AerisError::StorageError(format!(
                            "reverse map returned 0 keys for active keepsake box fingerprint ({old_bits} bits) at home {home}"
                        )));
                }
                if !tombstoned_disk_fps.iter().any(|f| f == &old_fp) {
                    tombstoned_disk_fps.push(old_fp);
                }
                combined
            };

            report.keys_rehashed += keys_for_box.len();
            for key in keys_for_box {
                let p = self.config.partition_of(key);
                if p == p_q {
                    query_part_min_bits = query_part_min_bits.max(old_bits);
                    query_part_keys.push(key);
                } else if let Some(entry) = colliding_parts.iter_mut().find(|e| e.0 == p) {
                    entry.1 = entry.1.max(old_bits);
                    entry.2.push(key);
                } else {
                    colliding_parts.push((p, fresh_base.max(old_bits), vec![key]));
                }
            }
        }

        // Deduplicate per-partition key vectors (matching exact BTreeSet semantics).
        for (_, _, ks) in &mut colliding_parts {
            if ks.len() > 1 {
                ks.sort_unstable();
                ks.dedup();
            }
        }
        if query_part_keys.len() > 1 {
            query_part_keys.sort_unstable();
            query_part_keys.dedup();
        }

        // Step 1: Resolve colliding partitions `p' != p_q` FIRST.
        let mut new_colliding_specs: Vec<(PartitionId, usize, Vec<Key>)> =
            Vec::with_capacity(colliding_parts.len());

        for (p_other, min_bits, keys_vec) in colliding_parts {
            let merge_target = remaining_boxes.iter().position(|u| {
                u.total_fp_bits(q, r) >= fresh_base.max(min_bits)
                    && u.matches_partition(&self.hasher, p_other, q, r)
            });

            if let Some(u_idx) = merge_target {
                let target_box = &mut remaining_boxes[u_idx];
                for &k in &keys_vec {
                    target_box.payload.insert_memento(self.config.memento_of(k));
                }
                let u_fp = target_box.canonical_fingerprint(home, q, r);
                overlay_keys_extend(overlay_keys, u_fp, keys_vec);
                continue;
            }

            let min_req = fresh_base.max(min_bits);
            let mut ext_level = self
                .geom
                .min_ext_level_for_bits(min_req.saturating_sub(fresh_base));

            loop {
                let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
                let fp_other = Fingerprint::from_hasher(&self.hasher, p_other, bit_len);
                let fp_q = Fingerprint::from_hasher(&self.hasher, p_q, bit_len);

                let diverges_from_q = fp_other != fp_q;
                let conflicts_remaining = remaining_boxes.iter().any(|u| {
                    if u.total_fp_bits(q, r) < fresh_base {
                        return false;
                    }
                    let u_fp = u.canonical_fingerprint(home, q, r);
                    fp_other.is_prefix_of(&u_fp) || u_fp.is_prefix_of(&fp_other)
                });

                if diverges_from_q && !conflicts_remaining {
                    break;
                }
                ext_level += 1;
            }

            new_colliding_specs.push((p_other, ext_level, keys_vec));
        }

        // Normalize pairwise prefix-freeness among all newly created colliding boxes.
        self.normalize_colliding_specs(home, &remaining_boxes, &mut new_colliding_specs);

        // Group colliding partitions that share the exact same final fingerprint into one box.
        let mut grouped_colliding: Vec<(Fingerprint, PartitionId, usize, Vec<Key>)> =
            Vec::with_capacity(new_colliding_specs.len());
        for (p_other, ext_level, keys_vec) in new_colliding_specs {
            let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
            let fp = Fingerprint::from_hasher(&self.hasher, p_other, bit_len);
            if let Some((_, _, _, existing_keys)) =
                grouped_colliding.iter_mut().find(|(f, _, _, _)| f == &fp)
            {
                existing_keys.extend(keys_vec);
                existing_keys.sort_unstable();
                existing_keys.dedup();
            } else {
                grouped_colliding.push((fp, p_other, ext_level, keys_vec));
            }
        }

        let mut created_boxes: Vec<(Fingerprint, KeepsakeBox, Vec<Key>)> =
            Vec::with_capacity(grouped_colliding.len() + 1);
        for (fp, p_rep, ext_level, keys_vec) in grouped_colliding {
            let payload = self.payload_from_key_slice(&keys_vec);
            let bx = KeepsakeBox::new_fresh(&self.hasher, p_rep, q, &self.geom, ext_level, payload);
            debug_assert_eq!(bx.canonical_fingerprint(home, q, r), fp);
            created_boxes.push((fp, bx, keys_vec));
        }

        // Step 2 (Query-Partition Edge Case): Extend `p_q` LAST once every colliding box's
        // final length is known so `p_q` is prefix-free against all colliding & remaining boxes.
        if !query_part_keys.is_empty() {
            let min_req_q = fresh_base.max(query_part_min_bits);
            let mut ext_level_q = self
                .geom
                .min_ext_level_for_bits(min_req_q.saturating_sub(fresh_base));

            loop {
                let bit_len_q = fresh_base + self.geom.ext_bits_for_level(ext_level_q);
                let fp_q = Fingerprint::from_hasher(&self.hasher, p_q, bit_len_q);

                let conflicts_remaining = remaining_boxes.iter().any(|u| {
                    if u.total_fp_bits(q, r) < fresh_base {
                        return false;
                    }
                    let u_fp = u.canonical_fingerprint(home, q, r);
                    fp_q.is_prefix_of(&u_fp) || u_fp.is_prefix_of(&fp_q)
                });
                let conflicts_colliding = created_boxes
                    .iter()
                    .any(|(c_fp, _, _)| fp_q.is_prefix_of(c_fp) || c_fp.is_prefix_of(&fp_q));

                if !conflicts_remaining && !conflicts_colliding {
                    break;
                }
                ext_level_q += 1;
            }

            let payload_q = self.payload_from_key_slice(&query_part_keys);
            let bx_q =
                KeepsakeBox::new_fresh(&self.hasher, p_q, q, &self.geom, ext_level_q, payload_q);
            let fp_q = bx_q.canonical_fingerprint(home, q, r);
            created_boxes.push((fp_q, bx_q, query_part_keys));
        }

        let mut new_created_slots = 0usize;
        for (fp, bx, keys_vec) in created_boxes {
            new_created_slots += bx.encoded_slot_len(&self.geom);
            if !fresh_overlay_fps.iter().any(|f| f == &fp) {
                fresh_overlay_fps.push(fp.clone());
            }
            overlay_keys_extend(overlay_keys, fp, keys_vec);
            remaining_boxes.push(bx);
        }

        remaining_boxes.sort_by_key(KeepsakeBox::sort_key);
        overlay_runs[run_idx].3 = remaining_boxes;

        report.boxes_split += matched_boxes.len();
        report.extension_slots_added += new_created_slots.saturating_sub(old_matched_slots);
        Ok(())
    }

    /// Ensures no two newly generated colliding boxes are strict prefixes of each other.
    fn normalize_colliding_specs(
        &self,
        home: usize,
        remaining_boxes: &[KeepsakeBox],
        specs: &mut [(PartitionId, usize, Vec<Key>)],
    ) {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);

        loop {
            let mut changed = false;
            for i in 0..specs.len() {
                let len_i = fresh_base + self.geom.ext_bits_for_level(specs[i].1);
                let fp_i = Fingerprint::from_hasher(&self.hasher, specs[i].0, len_i);
                for j in 0..specs.len() {
                    if i == j {
                        continue;
                    }
                    let len_j = fresh_base + self.geom.ext_bits_for_level(specs[j].1);
                    if len_i < len_j {
                        let fp_j = Fingerprint::from_hasher(&self.hasher, specs[j].0, len_j);
                        if fp_i.is_prefix_of(&fp_j) {
                            specs[i].1 = specs[j].1;
                            // Ensure promoted `specs[i]` also remains prefix-free against remaining_boxes.
                            loop {
                                let new_len = fresh_base + self.geom.ext_bits_for_level(specs[i].1);
                                let new_fp =
                                    Fingerprint::from_hasher(&self.hasher, specs[i].0, new_len);
                                let conflicts = remaining_boxes.iter().any(|u| {
                                    if u.total_fp_bits(q, r) < fresh_base {
                                        return false;
                                    }
                                    let u_fp = u.canonical_fingerprint(home, q, r);
                                    new_fp.is_prefix_of(&u_fp) || u_fp.is_prefix_of(&new_fp)
                                });
                                if !conflicts {
                                    break;
                                }
                                specs[i].1 += 1;
                            }
                            changed = true;
                            break;
                        }
                    }
                }
                if changed {
                    break;
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn payload_from_key_slice(&self, keys: &[Key]) -> BoxPayload {
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
        if self.num_keys == 0 {
            return Ok(None);
        }
        let q = self.config.q;
        let r = self.config.r;
        let p = self.config.partition_of(key);
        let mem = self.config.memento_of(key);
        let (home_u64, _) = self.hasher.quotient_and_remainder(p, q, r);
        let home = home_u64 as usize;

        if !self.table.is_occupied(home) {
            return Ok(None);
        }

        let boxes = self.decode_run(home);
        // Prefer the longest matching fingerprint (and youngest age on tie) containing `mem`.
        let mut best_idx: Option<(usize, u16, u8)> = None;
        for (idx, (_, _, bx)) in boxes.iter().enumerate() {
            if bx.matches_partition(&self.hasher, p, q, r) && bx.payload.contains_in_range(mem, mem)
            {
                let fp_bits = bx.total_fp_bits(q, r);
                let better = match best_idx {
                    None => true,
                    Some((_, best_bits, best_age)) => {
                        fp_bits > best_bits || (fp_bits == best_bits && bx.age < best_age)
                    }
                };
                if better {
                    best_idx = Some((idx, fp_bits, bx.age));
                }
            }
        }

        let Some((idx, _, _)) = best_idx else {
            return Ok(None);
        };

        let (box_start, old_len, mut target) = boxes[idx].clone();
        let is_run_end = idx + 1 == boxes.len();
        let removed_fp = target.canonical_fingerprint(home, q, r);
        let removed = target.payload.remove_one(mem);
        debug_assert!(removed, "contains_in_range(mem, mem) guaranteed presence");

        if target.payload.is_empty() {
            self.table.replace_region(home, box_start, old_len, &[]);
            self.occupied_slots -= old_len;
        } else {
            let new_len = target.encoded_slot_len(&self.geom);
            let encoded = target.encode(&self.geom, is_run_end);
            self.table
                .replace_region(home, box_start, old_len, &encoded);
            self.occupied_slots =
                ((self.occupied_slots as isize) + (new_len as isize) - (old_len as isize)) as usize;
        }
        self.num_keys -= 1;

        Ok(Some(removed_fp))
    }

    #[inline]
    fn query_point(&self, key: Key) -> bool {
        if self.num_keys == 0 {
            return false;
        }
        let p = self.config.partition_of(key);
        let mem = self.config.memento_of(key);
        self.query_partition_hot(p, mem, mem)
    }

    #[inline]
    fn query_range(&self, range: QueryRange) -> bool {
        if self.num_keys == 0 {
            return false;
        }
        if range.start() == 0 && range.end() == u64::MAX {
            return true;
        }
        let m = self.config.m.min(63);
        let mask = if m == 0 { 0u64 } else { (1u64 << m) - 1 };
        let start_part = range.start() >> m;
        let end_part = range.end() >> m;
        let start_m = range.start() & mask;
        let end_m = range.end() & mask;
        if start_part == end_part {
            return self.query_partition_hot(start_part, start_m, end_m);
        }
        if start_part + 1 == end_part {
            return self.query_partition_hot(start_part, start_m, mask)
                || self.query_partition_hot(end_part, 0, end_m);
        }
        for (partition, low_m, high_m) in range.partition_subqueries(self.config.m) {
            if self.query_partition_hot(partition, low_m, high_m) {
                return true;
            }
        }
        false
    }

    fn locate_matching_fingerprints(&self, range: QueryRange) -> Vec<Fingerprint> {
        if self.num_keys == 0 {
            return Vec::new();
        }
        let q = self.config.q;
        let r = self.config.r;
        let mut matched_fps = BTreeSet::new();

        for (partition, low_m, high_m) in range.partition_subqueries(self.config.m) {
            let (home_u64, _) = self.hasher.quotient_and_remainder(partition, q, r);
            let home = home_u64 as usize;
            if !self.table.is_occupied(home) {
                continue;
            }
            for (_, _, bx) in self.decode_run(home) {
                if bx.matches_partition(&self.hasher, partition, q, r)
                    && bx.payload.contains_in_range(low_m, high_m)
                {
                    matched_fps.insert(bx.canonical_fingerprint(home, q, r));
                }
            }
        }

        matched_fps.into_iter().collect()
    }

    fn adapt(
        &mut self,
        range: QueryRange,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<AdaptationReport> {
        let start = Instant::now();
        let mut report = AdaptationReport::default();
        if self.num_keys == 0 {
            return Ok(report);
        }

        // Pre-filter subqueries with zero allocations, keeping only partitions that
        // currently return a positive match on `[low_m, high_m]`.
        let mut stack_hits = [(0u64, 0u64, 0u64); 8];
        let mut stack_len = 0usize;
        let mut overflow_hits: Vec<(PartitionId, Memento, Memento)> = Vec::new();

        for (p_q, low_m, high_m) in range.partition_subqueries(self.config.m) {
            report.subqueries_checked += 1;
            if self.query_partition_hot(p_q, low_m, high_m) {
                if stack_len < stack_hits.len() {
                    stack_hits[stack_len] = (p_q, low_m, high_m);
                    stack_len += 1;
                } else {
                    if overflow_hits.is_empty() {
                        overflow_hits.extend_from_slice(&stack_hits);
                    }
                    overflow_hits.push((p_q, low_m, high_m));
                }
            }
        }

        if stack_len == 0 {
            return Ok(report);
        }

        // Fast Path A (>95% of adaptations): single matching subquery hitting a single
        // fresh colliding box with one colliding partition and no peer remainder conflict.
        if stack_len == 1 && overflow_hits.is_empty() {
            let (p_q, _, _) = stack_hits[0];
            let (home_u64, _) = self.quotient_and_remainder_hot(p_q);
            if self
                .try_adapt_single_box_fast(p_q, home_u64 as usize, reverse_map, &mut report)?
                .is_some()
            {
                tracing::debug!(
                    range_start = range.start(),
                    range_end = range.end(),
                    subqueries_checked = report.subqueries_checked,
                    boxes_split = report.boxes_split,
                    keys_rehashed = report.keys_rehashed,
                    extension_slots_added = report.extension_slots_added,
                    rejuvenated_boxes = report.rejuvenated_boxes,
                    load_factor = self.load_factor(),
                    latency_ns = start.elapsed().as_nanos() as u64,
                    "completed AerisFilter adaptation (in-place fast path)"
                );
                return Ok(report);
            }
        }

        // Phase 1 (General Planner): Flat Vec-backed overlay preserving exact all-or-nothing
        // atomicity on `FilterFull` and multi-partition / query-partition-last semantics.
        let matching_slice: &[(PartitionId, Memento, Memento)] = if overflow_hits.is_empty() {
            &stack_hits[..stack_len]
        } else {
            &overflow_hits
        };
        let mut overlay_runs: Vec<(usize, usize, usize, Vec<KeepsakeBox>)> =
            Vec::with_capacity(matching_slice.len());
        let mut overlay_keys: Vec<(Fingerprint, Vec<Key>)> = Vec::with_capacity(4);
        let mut fresh_overlay_fps: Vec<Fingerprint> = Vec::with_capacity(4);
        let mut tombstoned_disk_fps: Vec<Fingerprint> = Vec::with_capacity(4);

        for &(p_q, low_m, high_m) in matching_slice {
            self.plan_subquery_adaptation(
                p_q,
                low_m,
                high_m,
                reverse_map,
                &mut overlay_runs,
                &mut overlay_keys,
                &mut fresh_overlay_fps,
                &mut tombstoned_disk_fps,
                &mut report,
            )?;
        }

        if tombstoned_disk_fps.is_empty() && overlay_keys.is_empty() {
            return Ok(report);
        }

        // Phase 2: Compute net slot and key deltas using cached `(old_slots, old_keys)`
        // (zero redundant run decoding) and enforce `max_load_factor` atomically.
        let mut encoded_updates: Vec<(usize, Vec<(u64, KeepsakeSlotKind)>)> =
            Vec::with_capacity(overlay_runs.len());
        let mut total_slot_delta: isize = 0;
        let mut total_key_delta: isize = 0;

        for (home, old_slots, old_keys, new_boxes) in &overlay_runs {
            let new_encoded = self.encode_run_boxes(new_boxes);
            let new_keys: usize = new_boxes.iter().map(|b| b.payload.key_count()).sum();

            total_slot_delta += (new_encoded.len() as isize) - (*old_slots as isize);
            total_key_delta += (new_keys as isize) - (*old_keys as isize);
            encoded_updates.push((*home, new_encoded));
        }

        self.check_can_add_slots(total_slot_delta)?;

        // Phase 3: Commit Reverse Map updates atomically (tombstone all old disk FPs
        // before appending replacement entries so overlapping FPs are never wiped).
        let total_rm: usize = overlay_keys.iter().map(|(_, ks)| ks.len()).sum();
        let mut new_rm_entries: Vec<(Fingerprint, Key)> = Vec::with_capacity(total_rm);
        for (fp, keys) in overlay_keys {
            for k in keys {
                new_rm_entries.push((fp.clone(), k));
            }
        }

        if !tombstoned_disk_fps.is_empty() {
            for old_fp in &tombstoned_disk_fps[1..] {
                reverse_map.tombstone(old_fp)?;
            }
            reverse_map.replace_fingerprint(&tombstoned_disk_fps[0], &new_rm_entries)?;
        } else if !new_rm_entries.is_empty() {
            reverse_map.append_batch(&new_rm_entries)?;
        }

        // Phase 4: Commit RAM Quotient Filter runs, re-locating each run because
        // modifying an earlier home may Robin-Hood shift subsequent runs rightward.
        for (home, new_encoded) in encoded_updates {
            let (cur_start, cur_len) = self
                .table
                .locate_run(home)
                .map(|(s, e)| (s, e - s + 1))
                .unwrap_or((0, 0));
            self.table
                .replace_region(home, cur_start, cur_len, &new_encoded);
        }
        self.occupied_slots = ((self.occupied_slots as isize) + total_slot_delta) as usize;
        self.num_keys = ((self.num_keys as isize) + total_key_delta) as usize;

        tracing::debug!(
            range_start = range.start(),
            range_end = range.end(),
            subqueries_checked = report.subqueries_checked,
            boxes_split = report.boxes_split,
            keys_rehashed = report.keys_rehashed,
            extension_slots_added = report.extension_slots_added,
            rejuvenated_boxes = report.rejuvenated_boxes,
            load_factor = self.load_factor(),
            latency_ns = start.elapsed().as_nanos() as u64,
            "completed AerisFilter adaptation"
        );
        Ok(report)
    }

    fn expand(&mut self, reverse_map: &dyn ReverseMapStore) -> AerisResult<ExpansionReport> {
        let start = Instant::now();
        let old_q = self.config.q;
        let new_config = self.config.expanded()?;
        let new_q = new_config.q;
        let r = self.config.r;

        let mut new_table = QuotientTable::new(new_config.num_slots(), self.geom.slot_bits);
        let mut write_cursor = 0usize;
        let mut boxes_transferred = 0usize;
        let mut new_occupied_slots = 0usize;
        let mut new_num_keys = 0usize;
        let mut rejuvenate_fps: Vec<Fingerprint> = Vec::new();

        // Stage 1: Linear `O(N)` in-memory InfiniFilter bit-splitting for all boxes with `age < r`
        // (zero Reverse Map I/O). Collect `age == r` boxes for Stage 2 selective rejuvenation.
        for old_home in self.table.occupied_homes() {
            let boxes = self.decode_run(old_home);
            let mut split_0: Vec<KeepsakeBox> = Vec::new();
            let mut split_1: Vec<KeepsakeBox> = Vec::new();

            for (_, _, bx) in boxes {
                if let Some((new_home, split_box)) = bx.split_for_expansion(old_home, r) {
                    if (new_home & 1) == 0 {
                        split_0.push(split_box);
                    } else {
                        split_1.push(split_box);
                    }
                    boxes_transferred += 1;
                } else {
                    rejuvenate_fps.push(bx.canonical_fingerprint(old_home, old_q, r));
                }
            }

            if !split_0.is_empty() {
                split_0.sort_by_key(KeepsakeBox::sort_key);
                let run_keys: usize = split_0.iter().map(|b| b.payload.key_count()).sum();
                let run_slots = self.encode_run_boxes(&split_0);
                new_occupied_slots += run_slots.len();
                new_num_keys += run_keys;
                new_table.bulk_append_run(old_home << 1, &run_slots, &mut write_cursor);
            }

            if !split_1.is_empty() {
                split_1.sort_by_key(KeepsakeBox::sort_key);
                let run_keys: usize = split_1.iter().map(|b| b.payload.key_count()).sum();
                let run_slots = self.encode_run_boxes(&split_1);
                new_occupied_slots += run_slots.len();
                new_num_keys += run_keys;
                new_table.bulk_append_run((old_home << 1) | 1, &run_slots, &mut write_cursor);
            }
        }

        let boxes_rejuvenated = rejuvenate_fps.len();
        let mut keys_rejuvenated = 0usize;

        // Stage 2: Selective rejuvenation of oldest (`age == r`) boxes via Reverse Map,
        // preserving `new_fp.bit_len() >= old_fp.bit_len()` for monotonic adaptivity.
        let new_max_occupied_slots =
            ((new_config.num_slots() as f64) * new_config.max_load_factor).floor() as usize;
        let mut new_filter = Self {
            config: new_config,
            hasher: self.hasher,
            geom: self.geom,
            table: new_table,
            num_keys: new_num_keys,
            occupied_slots: new_occupied_slots,
            max_occupied_slots: new_max_occupied_slots,
            expansion_cycles: self.expansion_cycles + 1,
        };

        if !rejuvenate_fps.is_empty() {
            // Read all keys from Reverse Map BEFORE modifying Reverse Map so storage errors
            // leave `self` completely intact.
            let mut key_min_bits: BTreeMap<Key, u16> = BTreeMap::new();
            for old_fp in &rejuvenate_fps {
                let keys = reverse_map.get_keys(old_fp)?;
                keys_rejuvenated += keys.len();
                let floor_bits = old_fp.bit_len();
                for k in keys {
                    key_min_bits
                        .entry(k)
                        .and_modify(|cur| *cur = (*cur).max(floor_bits))
                        .or_insert(floor_bits);
                }
            }

            // Insert descending by `min_fp_bits` so longer monotonic floors are established
            // before shorter unextended boxes at the same `(new_home, new_rem)`.
            let mut ordered_rejuv: Vec<(u16, Key)> = key_min_bits
                .into_iter()
                .map(|(k, bits)| (bits, k))
                .collect();
            ordered_rejuv.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

            let mut new_rm_entries: Vec<(Fingerprint, Key)> =
                Vec::with_capacity(ordered_rejuv.len());
            for (min_bits, key) in ordered_rejuv {
                let new_fp = new_filter.insert_key_internal(key, min_bits, false)?;
                new_rm_entries.push((new_fp, key));
            }

            // Tombstone all old `age == r` fingerprints first, then insert replacement entries
            // atomically so an unchanged `old_fp == new_fp` is never wiped.
            for old_fp in &rejuvenate_fps[1..] {
                reverse_map.tombstone(old_fp)?;
            }
            reverse_map.replace_fingerprint(&rejuvenate_fps[0], &new_rm_entries)?;
        }

        *self = new_filter;
        let duration_ns = start.elapsed().as_nanos() as u64;

        tracing::info!(
            old_q,
            new_q,
            boxes_transferred,
            boxes_rejuvenated,
            keys_rejuvenated,
            load_factor = self.load_factor(),
            latency_ns = duration_ns,
            expansion_cycles = self.expansion_cycles,
            "completed 2x InfiniFilter expansion"
        );

        Ok(ExpansionReport {
            old_q,
            new_q,
            boxes_transferred,
            boxes_rejuvenated,
            keys_rejuvenated,
            duration_ns,
        })
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

/// Helper extending a flat `(Fingerprint, Vec<Key>)` overlay list while maintaining
/// deduplicated key sets without `BTreeMap`/`BTreeSet` heap overhead.
#[inline]
fn overlay_keys_extend(
    overlay_keys: &mut Vec<(Fingerprint, Vec<Key>)>,
    fp: Fingerprint,
    keys: Vec<Key>,
) {
    if let Some((_, existing)) = overlay_keys.iter_mut().find(|(f, _)| f == &fp) {
        existing.extend(keys);
        if existing.len() > 1 {
            existing.sort_unstable();
            existing.dedup();
        }
    } else {
        overlay_keys.push((fp, keys));
    }
}

/// In-place 8-bit LSD radix sort for packed `u64` keys spanning the low
/// `total_bits` (`0..=64`) bits.
///
/// Computes all `ceil(total_bits / 8)` byte histograms in a single pre-pass,
/// skips trivial passes where all elements share a single byte value, allocates
/// at most one temporary `Vec<u64>` buffer (`8 B/key`), swaps vectors in `O(1)`
/// if the final sorted output lands in `tmp`, and drops `tmp` immediately before
/// returning.
///
/// # Complexity
/// `O(n * ceil(total_bits / 8))` time, `8 * n` bytes transient scratch space.
fn radix_sort_u64(data: &mut Vec<u64>, total_bits: usize) {
    const WC_BUF_LEN: usize = 8;
    const WC_THRESHOLD: usize = 65_536;

    let n = data.len();
    if n <= 1 || total_bits == 0 {
        return;
    }

    let num_passes = total_bits.min(64).div_ceil(RADIX_BITS);
    let mut counts = [[0usize; RADIX_BUCKETS]; 8];
    match num_passes {
        5 => {
            for &val in data.iter() {
                counts[0][(val & 0xFF) as usize] += 1;
                counts[1][((val >> 8) & 0xFF) as usize] += 1;
                counts[2][((val >> 16) & 0xFF) as usize] += 1;
                counts[3][((val >> 24) & 0xFF) as usize] += 1;
                counts[4][((val >> 32) & 0xFF) as usize] += 1;
            }
        }
        6 => {
            for &val in data.iter() {
                counts[0][(val & 0xFF) as usize] += 1;
                counts[1][((val >> 8) & 0xFF) as usize] += 1;
                counts[2][((val >> 16) & 0xFF) as usize] += 1;
                counts[3][((val >> 24) & 0xFF) as usize] += 1;
                counts[4][((val >> 32) & 0xFF) as usize] += 1;
                counts[5][((val >> 40) & 0xFF) as usize] += 1;
            }
        }
        _ => {
            for &val in data.iter() {
                for (pass, hist) in counts.iter_mut().enumerate().take(num_passes) {
                    let byte = ((val >> (pass * RADIX_BITS)) & 0xFF) as usize;
                    hist[byte] += 1;
                }
            }
        }
    }

    let has_active_pass = counts
        .iter()
        .take(num_passes)
        .any(|hist| !hist.contains(&n));
    if !has_active_pass {
        return;
    }

    let mut tmp: Vec<u64> = vec![0u64; n];
    let mut in_tmp = false;

    for (pass, hist) in counts.iter().enumerate().take(num_passes) {
        if hist.contains(&n) {
            continue;
        }
        let mut offsets = [0usize; RADIX_BUCKETS];
        let mut sum = 0usize;
        for (off, &c) in offsets.iter_mut().zip(hist.iter()) {
            *off = sum;
            sum += c;
        }

        let shift = pass * RADIX_BITS;
        let (src, dst): (&[u64], &mut [u64]) = if in_tmp {
            (&tmp, data.as_mut_slice())
        } else {
            (data.as_slice(), &mut tmp)
        };
        if n >= WC_THRESHOLD {
            // 16 KiB L1-resident write-combining buffer (256 buckets * 64 B cache line)
            // reduces DRAM TLB thrashing and bounds checks by 8x on large inputs.
            let mut wc_buf = [[0u64; WC_BUF_LEN]; RADIX_BUCKETS];
            let mut wc_len = [0u8; RADIX_BUCKETS];
            for &val in src {
                let bucket = ((val >> shift) & 0xFF) as usize;
                let len = wc_len[bucket] as usize;
                wc_buf[bucket][len] = val;
                let next_len = len + 1;
                if next_len == WC_BUF_LEN {
                    let pos = offsets[bucket];
                    dst[pos..(pos + WC_BUF_LEN)].copy_from_slice(&wc_buf[bucket]);
                    offsets[bucket] = pos + WC_BUF_LEN;
                    wc_len[bucket] = 0;
                } else {
                    wc_len[bucket] = next_len as u8;
                }
            }
            for bucket in 0..RADIX_BUCKETS {
                let rem = wc_len[bucket] as usize;
                if rem > 0 {
                    let pos = offsets[bucket];
                    dst[pos..(pos + rem)].copy_from_slice(&wc_buf[bucket][..rem]);
                }
            }
        } else {
            for &val in src {
                let bucket = ((val >> shift) & 0xFF) as usize;
                let pos = offsets[bucket];
                dst[pos] = val;
                offsets[bucket] = pos + 1;
            }
        }
        in_tmp = !in_tmp;
    }

    if in_tmp {
        std::mem::swap(data, &mut tmp);
    }
    drop(tmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// In-memory `ReverseMapStore` mock that tracks `append_batch` write counts
    /// for verifying `bulk_load` atomicity and full `adapt`/`expand` workflows.
    #[derive(Default)]
    struct CountingReverseMap {
        map: Mutex<BTreeMap<Fingerprint, Vec<Key>>>,
        append_batch_calls: AtomicUsize,
        total_appended_entries: AtomicUsize,
    }

    impl CountingReverseMap {
        fn batch_calls(&self) -> usize {
            self.append_batch_calls.load(Ordering::SeqCst)
        }

        fn appended_entries(&self) -> usize {
            self.total_appended_entries.load(Ordering::SeqCst)
        }
    }

    impl ReverseMapStore for CountingReverseMap {
        fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
            self.map
                .lock()
                .unwrap()
                .entry(fp.clone())
                .or_default()
                .push(key);
            Ok(())
        }

        fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
            self.append_batch_calls.fetch_add(1, Ordering::SeqCst);
            self.total_appended_entries
                .fetch_add(entries.len(), Ordering::SeqCst);
            let mut guard = self.map.lock().unwrap();
            for (fp, key) in entries {
                guard.entry(fp.clone()).or_default().push(*key);
            }
            Ok(())
        }

        fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
            let mut guard = self.map.lock().unwrap();
            if let Some(list) = guard.get_mut(fp) {
                if let Some(pos) = list.iter().position(|&k| k == key) {
                    list.swap_remove(pos);
                }
            }
            Ok(())
        }

        fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
            Ok(self
                .map
                .lock()
                .unwrap()
                .get(fp)
                .cloned()
                .unwrap_or_default())
        }

        fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
            self.map.lock().unwrap().remove(fp);
            Ok(())
        }

        fn replace_fingerprint(
            &self,
            old_fp: &Fingerprint,
            new_entries: &[(Fingerprint, Key)],
        ) -> AerisResult<()> {
            let mut guard = self.map.lock().unwrap();
            guard.remove(old_fp);
            for (fp, key) in new_entries {
                guard.entry(fp.clone()).or_default().push(*key);
            }
            Ok(())
        }

        fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
            let guard = self.map.lock().unwrap();
            let mut set = BTreeSet::new();
            for list in guard.values() {
                set.extend(list.iter().copied());
            }
            Ok(set.into_iter().collect())
        }

        fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
            let keys = self.scan_all_keys()?;
            Ok(Box::new(keys.into_iter().map(Ok)))
        }

        fn entry_count_estimate(&self) -> u64 {
            self.map
                .lock()
                .unwrap()
                .values()
                .map(|v| v.len() as u64)
                .sum()
        }

        fn flush(&self) -> AerisResult<()> {
            Ok(())
        }

        fn compact(&self) -> AerisResult<()> {
            Ok(())
        }
    }

    #[test]
    fn radix_sort_u64_handles_all_edge_cases() {
        // Empty.
        let mut empty: Vec<u64> = Vec::new();
        radix_sort_u64(&mut empty, 64);
        assert!(empty.is_empty());

        // Single element.
        let mut single = vec![0xDEAD_BEEF_CAFE_BABEu64];
        radix_sort_u64(&mut single, 64);
        assert_eq!(single, vec![0xDEAD_BEEF_CAFE_BABEu64]);

        // All identical (skips all passes without allocating tmp).
        let mut identical = vec![0x0123_4567_89AB_CDEFu64; 512];
        radix_sort_u64(&mut identical, 64);
        assert!(identical.iter().all(|&x| x == 0x0123_4567_89AB_CDEF));

        // Already sorted.
        let mut sorted: Vec<u64> = (0..1000).collect();
        radix_sort_u64(&mut sorted, 16);
        assert_eq!(sorted, (0..1000).collect::<Vec<u64>>());

        // Reverse sorted (odd pass count: 3 passes for 24 bits -> tests `in_tmp` swap).
        let mut rev: Vec<u64> = (0..2048).rev().collect();
        radix_sort_u64(&mut rev, 24);
        assert_eq!(rev, (0..2048).collect::<Vec<u64>>());

        // Full 64-bit random keys (8 passes) and narrower bit widths.
        let mut rng = ChaCha8Rng::seed_from_u64(0x2AD1_0001);
        for total_bits in [8usize, 13, 25, 32, 42, 57, 64] {
            let mask = if total_bits == 64 {
                u64::MAX
            } else {
                (1u64 << total_bits) - 1
            };
            let mut vals: Vec<u64> = (0..4096).map(|_| rng.next_u64() & mask).collect();
            let mut expected = vals.clone();
            expected.sort_unstable();
            radix_sort_u64(&mut vals, total_bits);
            assert_eq!(vals, expected, "mismatch for total_bits={total_bits}");
        }
    }

    #[test]
    fn radix_bulk_load_matches_sequential_insert_across_m_modes_and_collisions() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xB01C_0001);

        for &(m, max_range) in &[(0u8, 1u64), (2, 4), (5, 32), (10, 1024)] {
            let q = 10u8;
            let r = 8u8;
            let cfg = FilterConfig::new(q, r, max_range, r + m, 0.05).unwrap();

            // Generate keys including deliberate same-partition collisions & duplicates.
            let mut keys: Vec<Key> = Vec::with_capacity(600);
            for _ in 0..450 {
                keys.push(rng.next_u64() & 0x000F_FFFF);
            }
            // Add multiple keys inside the exact same partition (triggering multi-memento
            // continuation slots / point-mode multiplicity >= 2 and Subcase B2/B3).
            let hot_part_base = 42u64 << m;
            for offset in 0..16u64 {
                let mem = offset & cfg.memento_mask();
                keys.push(hot_part_base | mem);
            }
            // Also duplicate an existing key explicitly.
            keys.push(hot_part_base);
            keys.push(hot_part_base);

            let rm_bulk = CountingReverseMap::default();
            let bulk = AerisFilter::bulk_load(cfg.clone(), &keys, &rm_bulk).unwrap();
            bulk.verify_invariants().unwrap();
            assert_eq!(rm_bulk.appended_entries(), keys.len());

            let mut seq = AerisFilter::new(cfg.clone()).unwrap();
            let rm_seq = CountingReverseMap::default();
            for &k in &keys {
                let fp = seq.insert(k).unwrap();
                rm_seq.append_key(&fp, k).unwrap();
            }
            seq.verify_invariants().unwrap();

            assert_eq!(bulk.len(), seq.len());
            assert_eq!(bulk.occupied_slots(), seq.occupied_slots());

            // Verify bit-for-bit identical decoded keepsake boxes across every canonical home.
            let bulk_homes: Vec<usize> = bulk.table.occupied_homes().collect();
            let seq_homes: Vec<usize> = seq.table.occupied_homes().collect();
            assert_eq!(bulk_homes, seq_homes, "occupied homes mismatch for m={m}");

            for home in bulk_homes {
                let b_boxes: Vec<KeepsakeBox> = bulk
                    .decode_run(home)
                    .into_iter()
                    .map(|(_, _, bx)| bx)
                    .collect();
                let s_boxes: Vec<KeepsakeBox> = seq
                    .decode_run(home)
                    .into_iter()
                    .map(|(_, _, bx)| bx)
                    .collect();
                assert_eq!(
                    b_boxes, s_boxes,
                    "decoded run mismatch at home={home} m={m}"
                );
            }

            // Zero FN & identical query answers across inserted keys and probes.
            for &k in &keys {
                assert!(bulk.query_point(k), "bulk FN for m={m} key={k}");
                assert!(seq.query_point(k), "seq FN for m={m} key={k}");
            }
            for _ in 0..500 {
                let probe = rng.next_u64() & 0x000F_FFFF;
                assert_eq!(bulk.query_point(probe), seq.query_point(probe));
                let range = QueryRange::with_len(probe, max_range.max(1)).unwrap();
                assert_eq!(bulk.query_range(range), seq.query_range(range));
            }
        }
    }

    #[test]
    fn bulk_load_filter_full_is_atomic_with_zero_reverse_map_writes() {
        // Test fast radix path (`q + r + m <= 64`) overflow rejection.
        let cfg = FilterConfig::new(6, 8, 32, 13, 0.05).unwrap(); // capacity = 64 slots
        let keys: Vec<Key> = (0..120u64).map(|i| i << 5).collect();
        let rm = CountingReverseMap::default();
        let err = AerisFilter::bulk_load(cfg, &keys, &rm).unwrap_err();
        assert!(
            matches!(err, AerisError::FilterFull { .. }),
            "expected FilterFull, got {err:?}"
        );
        assert_eq!(
            rm.batch_calls(),
            0,
            "reverse_map must receive 0 writes on FilterFull"
        );
        assert_eq!(rm.appended_entries(), 0);

        // Test wide fallback path (`q + r + m > 64`) overflow rejection.
        let wide_cfg = FilterConfig::new(6, 55, 32, 60, 0.05).unwrap(); // 6 + 55 + 5 = 66 > 64
        let rm_wide = CountingReverseMap::default();
        let err_wide = AerisFilter::bulk_load(wide_cfg, &keys, &rm_wide).unwrap_err();
        assert!(
            matches!(err_wide, AerisError::FilterFull { .. }),
            "expected FilterFull on wide path, got {err_wide:?}"
        );
        assert_eq!(rm_wide.batch_calls(), 0);
        assert_eq!(rm_wide.appended_entries(), 0);
    }

    #[test]
    fn zero_alloc_insert_fast_paths_handle_aged_boxes_extensions_and_point_mode() {
        // `r = 7 >= r_min(0.08) = 7`, `m = 5`, `q = 7` (128 slots).
        let cfg = FilterConfig::new(7, 7, 32, 12, 0.08).unwrap();
        let rm = CountingReverseMap::default();
        let mut filter = AerisFilter::new(cfg).unwrap();

        // Phase 1: Insert keys hitting Fast Path A, Subcase B1, Subcase B2 (1->2 mementos),
        // and Subcase B3 (3+ mementos in same partition).
        let mut inserted = Vec::new();
        for i in 0..45u64 {
            let k = i.wrapping_mul(0x9E37_79B9) & 0x00FF_FFFF;
            let fp = filter.insert(k).unwrap();
            rm.append_key(&fp, k).unwrap();
            inserted.push(k);
        }
        // Explicitly exercise Subcase B2 (2nd memento with lo > hi and duplicate) and Subcase B3.
        let p_a = 100u64 << 5;
        for m_val in [20u64, 5, 20, 31, 0] {
            let k = p_a | m_val;
            let fp = filter.insert(k).unwrap();
            rm.append_key(&fp, k).unwrap();
            inserted.push(k);
        }
        filter.verify_invariants().unwrap();

        // Phase 2: Trigger `expand()` so boxes acquire `age = 1`, then insert fresh `age = 0` keys
        // into the same occupied runs (verifying sort order: age=0 before age>0).
        filter.expand(&rm).unwrap();
        filter.verify_invariants().unwrap();
        for i in 0..40u64 {
            let k = (i + 500).wrapping_mul(0x517C_C1B7) & 0x00FF_FFFF;
            let fp = filter.insert(k).unwrap();
            rm.append_key(&fp, k).unwrap();
            inserted.push(k);
        }
        filter.verify_invariants().unwrap();

        // Phase 3: Trigger an `adapt()` to introduce `Extension` boxes, then keep inserting
        // into runs containing `Extension` boxes (verifying fallback to full prefix-free path).
        let inserted_set: BTreeSet<Key> = inserted.iter().copied().collect();
        let mut adapted_any = false;
        for probe in 0..20_000u64 {
            let range = QueryRange::with_len(probe * 32, 32).unwrap();
            let truly_empty = !inserted_set
                .range(range.start()..=range.end())
                .any(|_| true);
            if truly_empty && filter.query_range(range) {
                let rep = filter.adapt(range, &rm).unwrap();
                if rep.boxes_split > 0 {
                    adapted_any = true;
                    assert!(!filter.query_range(range));
                    break;
                }
            }
        }
        assert!(adapted_any, "expected at least one FP adaptation");

        for i in 0..30u64 {
            let k = (i + 2000).wrapping_mul(0x2722_0A95) & 0x00FF_FFFF;
            let fp = filter.insert(k).unwrap();
            rm.append_key(&fp, k).unwrap();
            inserted.push(k);
        }
        filter.verify_invariants().unwrap();
        for &k in &inserted {
            assert!(filter.query_point(k), "zero-FN violation for key {k}");
        }

        // Phase 4: Point mode (`m = 0`) sequential inserts with duplicate multiplicities
        // (exercises Fast Path A, Subcase B1, and Subcase B3 fallback for count >= 2).
        let pt_cfg = FilterConfig::new(7, 8, 1, 8, 0.05).unwrap();
        let mut pt = AerisFilter::new(pt_cfg).unwrap();
        for k in [10u64, 20, 10, 10, 30, 20, 40, 50] {
            pt.insert(k).unwrap();
        }
        pt.verify_invariants().unwrap();
        assert_eq!(pt.len(), 8);
        for k in [10u64, 20, 30, 40, 50] {
            assert!(pt.query_point(k));
        }

        // Phase 5: Wide fallback `bulk_load` (`q + r + m > 64`) matches sequential `insert`.
        let wide_cfg = FilterConfig::new(8, 55, 32, 60, 0.05).unwrap(); // 68 > 64
        let wide_keys: Vec<Key> = (0..60u64).map(|i| (i / 2) << 5 | (i % 32)).collect();
        let rm_wide = CountingReverseMap::default();
        let wide_bulk = AerisFilter::bulk_load(wide_cfg.clone(), &wide_keys, &rm_wide).unwrap();
        wide_bulk.verify_invariants().unwrap();
        let mut wide_seq = AerisFilter::new(wide_cfg).unwrap();
        for &k in &wide_keys {
            wide_seq.insert(k).unwrap();
        }
        wide_seq.verify_invariants().unwrap();
        assert_eq!(wide_bulk.occupied_slots(), wide_seq.occupied_slots());
        for &k in &wide_keys {
            assert!(wide_bulk.query_point(k));
            assert!(wide_seq.query_point(k));
        }
    }

    #[test]
    fn noop_reverse_map_skips_stage5_and_matches_u64_prefix_fingerprints() {
        struct LocalNoopRm {
            calls: AtomicUsize,
        }
        impl ReverseMapStore for LocalNoopRm {
            fn append_key(&self, _: &Fingerprint, _: Key) -> AerisResult<()> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn append_batch(&self, _: &[(Fingerprint, Key)]) -> AerisResult<()> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn remove_key(&self, _: &Fingerprint, _: Key) -> AerisResult<()> {
                Ok(())
            }
            fn get_keys(&self, _: &Fingerprint) -> AerisResult<Vec<Key>> {
                Ok(Vec::new())
            }
            fn tombstone(&self, _: &Fingerprint) -> AerisResult<()> {
                Ok(())
            }
            fn replace_fingerprint(
                &self,
                _: &Fingerprint,
                _: &[(Fingerprint, Key)],
            ) -> AerisResult<()> {
                Ok(())
            }
            fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
                Ok(Vec::new())
            }
            fn iter_all_keys(
                &self,
            ) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
                Ok(Box::new(std::iter::empty()))
            }
            fn entry_count_estimate(&self) -> u64 {
                0
            }
            fn flush(&self) -> AerisResult<()> {
                Ok(())
            }
            fn compact(&self) -> AerisResult<()> {
                Ok(())
            }
            fn is_noop(&self) -> bool {
                true
            }
        }

        // Verify `is_noop()` skips Stage 5 completely (0 `append_batch` calls) while
        // producing bit-for-bit identical filter state above the radix WC threshold (`> 65_536`).
        let cfg = FilterConfig::new(17, 7, 32, 12, 0.08).unwrap(); // 131,072 slots
        let mut rng = ChaCha8Rng::seed_from_u64(0x900B_0001);
        let keys: Vec<Key> = (0..80_000).map(|_| rng.next_u64()).collect();
        let noop = LocalNoopRm {
            calls: AtomicUsize::new(0),
        };
        let filter = AerisFilter::bulk_load(cfg.clone(), &keys, &noop).unwrap();
        assert_eq!(noop.calls.load(Ordering::SeqCst), 0);
        filter.verify_invariants().unwrap();
        for &k in &keys {
            assert!(filter.query_point(k), "FN after WC-buffered bulk_load");
        }

        // Verify `Fingerprint::from_u64_prefix` matches `Fingerprint::from_parts` across widths.
        let hasher = PartitionHasher::new(cfg.hash_seed);
        for &(q, r) in &[(6u8, 7u8), (16, 12), (24, 7), (30, 7), (32, 32)] {
            for p in [0u64, 1, 42, 0xDEAD_BEEF_CAFE_BABE, u64::MAX >> 5] {
                let w0 = hasher.word(p, 0);
                let (h, rem) = hasher.quotient_and_remainder(p, q, r);
                assert_eq!(
                    Fingerprint::from_u64_prefix(w0, q + r),
                    Fingerprint::from_parts(h, q, rem, r, &[]),
                    "from_u64_prefix mismatch for q={q}, r={r}, p={p}"
                );
            }
        }
    }

    #[test]
    fn adapt_challenge_query_partition_outside_range_multi_ext_expand_and_filter_full_atomicity() {
        // 1. Multi-partition hash collision where keys exist in `p_q` OUTSIDE `[a, b]`:
        // colliding partitions MUST be extended first, and `p_q` extended last with 0 FN.
        let cfg = FilterConfig::new(6, 7, 32, 12, 0.08).unwrap(); // q=6 (64 slots), r=7, m=5
        let hasher = PartitionHasher::new(cfg.hash_seed);
        let p_q: PartitionId = 100;
        let (target_home, target_rem) = hasher.quotient_and_remainder(p_q, cfg.q, cfg.r);

        // Find two distinct colliding partitions `p_c1`, `p_c2` sharing `(target_home, target_rem)`.
        let mut colliding_parts = Vec::new();
        for cand in 101..500_000u64 {
            if hasher.quotient_and_remainder(cand, cfg.q, cfg.r) == (target_home, target_rem) {
                colliding_parts.push(cand);
                if colliding_parts.len() == 2 {
                    break;
                }
            }
        }
        assert_eq!(colliding_parts.len(), 2, "must find 2 colliding partitions");

        let rm = CountingReverseMap::default();
        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let k_q_outside_lo = (p_q << 5) | 1; // in `p_q`, outside query [10, 20]
        let k_q_outside_hi = (p_q << 5) | 28; // in `p_q`, outside query [10, 20]
        let k_c1_inside = (colliding_parts[0] << 5) | 15; // colliding partition inside [10, 20]
        let k_c2_inside = (colliding_parts[1] << 5) | 18; // 2nd colliding partition inside [10, 20]
        let all_keys = [k_q_outside_lo, k_q_outside_hi, k_c1_inside, k_c2_inside];
        for &k in &all_keys {
            let fp = filter.insert(k).unwrap();
            rm.append_key(&fp, k).unwrap();
        }

        let fp_range = QueryRange::new((p_q << 5) | 10, (p_q << 5) | 20).unwrap();
        assert!(filter.query_range(fp_range), "expected FP before adapt");
        let rep = filter.adapt(fp_range, &rm).unwrap();
        assert!(rep.boxes_split >= 1);
        assert!(
            !filter.query_range(fp_range),
            "FP must be resolved after adapt"
        );
        filter.verify_invariants().unwrap();
        for &k in &all_keys {
            assert!(filter.query_point(k), "zero-FN violated for key {k}");
        }

        // 2. Repeated multi-slot extensions (`1x`, `2x`, `3x`) interleaved with `expand()` & `delete()`,
        // plus adversarial reprobing verifying 0 FN and 0 repeated FPs.
        let cfg2 = FilterConfig::new(7, 7, 32, 12, 0.08).unwrap(); // 128 slots
        let rm2 = CountingReverseMap::default();
        let mut f2 = AerisFilter::new(cfg2).unwrap();
        let mut live_keys = BTreeSet::new();
        let mut rng = ChaCha8Rng::seed_from_u64(0xAD07_2026);
        for _ in 0..80 {
            let k = rng.next_u64() & 0x000F_FFFF;
            let fp = f2.insert(k).unwrap();
            rm2.append_key(&fp, k).unwrap();
            live_keys.insert(k);
        }

        let mut adapted_ranges = Vec::new();
        for probe in 0..30_000u64 {
            let r = QueryRange::with_len(probe.wrapping_mul(37) & 0x000F_FFFF, 32).unwrap();
            if live_keys.range(r.start()..=r.end()).next().is_none()
                && f2.query_range(r)
                && f2.adapt(r, &rm2).is_ok()
            {
                assert!(!f2.query_range(r));
                adapted_ranges.push(r);
                if adapted_ranges.len() >= 12 {
                    break;
                }
            }
        }
        assert!(!adapted_ranges.is_empty());
        f2.verify_invariants().unwrap();

        // Interleave `expand()` and verify monotonic adaptivity (0 repeated FPs) and 0 FN.
        f2.expand(&rm2).unwrap();
        f2.verify_invariants().unwrap();
        for &r in &adapted_ranges {
            assert!(
                !f2.query_range(r),
                "monotonic adaptivity violated across expand() for {r:?}"
            );
        }
        // Delete 10 keys, adapt again on aged boxes, expand a second time, and reprobe.
        let to_delete: Vec<Key> = live_keys.iter().copied().take(10).collect();
        for k in to_delete {
            if let Some(fp) = f2.delete(k).unwrap() {
                rm2.remove_key(&fp, k).unwrap();
            }
            live_keys.remove(&k);
        }
        for probe in 30_000..60_000u64 {
            let r = QueryRange::with_len(probe.wrapping_mul(53) & 0x000F_FFFF, 32).unwrap();
            if live_keys.range(r.start()..=r.end()).next().is_none()
                && f2.query_range(r)
                && f2.adapt(r, &rm2).is_ok()
            {
                assert!(!f2.query_range(r));
                adapted_ranges.push(r);
                if adapted_ranges.len() >= 20 {
                    break;
                }
            }
        }
        f2.expand(&rm2).unwrap();
        f2.verify_invariants().unwrap();
        for &r in &adapted_ranges {
            assert!(
                !f2.query_range(r),
                "adversarial reprobe repeated FP after second expand for {r:?}"
            );
        }
        for &k in &live_keys {
            assert!(
                f2.query_point(k),
                "FN after interleaved adapt/expand/delete"
            );
        }

        // 3. `FilterFull` rollback atomicity near `alpha = 94.9%` during `adapt()`:
        // both the RAM filter and ReverseMap must remain completely untouched on error.
        let full_cfg = FilterConfig::new(7, 7, 32, 12, 0.08).unwrap(); // 128 slots, max 121 slots (94.53%)
        let rm_full = CountingReverseMap::default();
        let mut f_full = AerisFilter::new(full_cfg).unwrap();
        let mut full_keys = BTreeSet::new();
        let mut next_part = 1u64;
        while f_full.occupied_slots() < f_full.max_occupied_slots {
            let k = (next_part << 5) | (next_part & 31);
            next_part += 1;
            if let Ok(fp) = f_full.insert(k) {
                rm_full.append_key(&fp, k).unwrap();
                full_keys.insert(k);
            } else {
                break;
            }
        }
        assert_eq!(f_full.occupied_slots(), f_full.max_occupied_slots);
        let rm_snapshot_before = rm_full.map.lock().unwrap().clone();
        let slots_before = f_full.occupied_slots();
        let keys_before = f_full.len();

        let mut hit_filter_full = false;
        for probe in 100_000..200_000u64 {
            let r = QueryRange::with_len(probe << 5, 32).unwrap();
            if full_keys.range(r.start()..=r.end()).next().is_none() && f_full.query_range(r) {
                match f_full.adapt(r, &rm_full) {
                    Err(AerisError::FilterFull { .. }) => {
                        hit_filter_full = true;
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => panic!("unexpected adapt error: {e}"),
                }
            }
        }
        assert!(
            hit_filter_full,
            "expected FilterFull rollback during adapt at ceiling"
        );
        assert_eq!(f_full.occupied_slots(), slots_before);
        assert_eq!(f_full.len(), keys_before);
        assert_eq!(
            *rm_full.map.lock().unwrap(),
            rm_snapshot_before,
            "ReverseMap mutated on FilterFull rollback"
        );
        f_full.verify_invariants().unwrap();
        for &k in &full_keys {
            assert!(f_full.query_point(k), "FN after FilterFull rollback");
        }
    }
}
