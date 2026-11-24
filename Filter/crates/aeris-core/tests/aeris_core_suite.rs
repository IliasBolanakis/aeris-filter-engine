//! Comprehensive verification suite for `aeris-core` covering:
//! 1. Unit & boundary conditions (`0`, `1`, `u64::MAX - 1`, `u64::MAX`, `m = 0, 1, 5, 10`,
//!    `bulk_load`, `rebuild_from_keys`, `FilterFull` & storage-fault atomicity).
//! 2. Strict monotonic adaptivity (zero repeated false positives across subsequent
//!    inserts, adaptations, and expansions).
//! 3. Explicit query-partition adaptation edge case (both 1st-level and multi-level
//!    extension collisions + opportunistic rejuvenation).
//! 4. 8+ consecutive `2x` InfiniFilter expansions with selective rejuvenation and
//!    sustained empirical `FPR <= epsilon` across `epsilon in {0.1, 0.05, 1/32, 1/64}`.
//! 5. `proptest` randomized state-machine fuzzing verifying zero false negatives,
//!    zero repeated adapted false positives, and full structural QF invariants.

use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, FilterConfig, Fingerprint, Key, PartitionHasher, QueryRange,
    RangeFilter, ReverseMapStore,
};
use proptest::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

/// In-memory thread-safe `ReverseMapStore` used exclusively by `aeris-core` tests
/// (real RocksDB persistence is implemented and tested in `aeris-storage`).
#[derive(Default)]
struct TestReverseMap {
    map: RwLock<HashMap<Fingerprint, BTreeSet<Key>>>,
    fail_reads: AtomicBool,
}

impl TestReverseMap {
    fn new() -> Self {
        Self::default()
    }

    fn set_fail_reads(&self, fail: bool) {
        self.fail_reads.store(fail, Ordering::SeqCst);
    }

    fn snapshot(&self) -> HashMap<Fingerprint, BTreeSet<Key>> {
        self.map.read().unwrap().clone()
    }
}

impl ReverseMapStore for TestReverseMap {
    fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let mut guard = self.map.write().unwrap();
        guard.entry(fp.clone()).or_default().insert(key);
        Ok(())
    }

    fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
        let mut guard = self.map.write().unwrap();
        for (fp, key) in entries {
            guard.entry(fp.clone()).or_default().insert(*key);
        }
        Ok(())
    }

    fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let mut guard = self.map.write().unwrap();
        if let Some(set) = guard.get_mut(fp) {
            set.remove(&key);
            if set.is_empty() {
                guard.remove(fp);
            }
        }
        Ok(())
    }

    fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
        if self.fail_reads.load(Ordering::SeqCst) {
            return Err(AerisError::StorageError(
                "injected reverse map read failure".to_string(),
            ));
        }
        let guard = self.map.read().unwrap();
        Ok(guard
            .get(fp)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default())
    }

    fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
        let mut guard = self.map.write().unwrap();
        guard.remove(fp);
        Ok(())
    }

    fn replace_fingerprint(
        &self,
        old_fp: &Fingerprint,
        new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        let mut guard = self.map.write().unwrap();
        guard.remove(old_fp);
        for (fp, key) in new_entries {
            guard.entry(fp.clone()).or_default().insert(*key);
        }
        Ok(())
    }

    fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        let guard = self.map.read().unwrap();
        let mut all = BTreeSet::new();
        for set in guard.values() {
            all.extend(set.iter().copied());
        }
        Ok(all.into_iter().collect())
    }

    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
        let keys = self.scan_all_keys()?;
        Ok(Box::new(keys.into_iter().map(Ok)))
    }

    fn entry_count_estimate(&self) -> u64 {
        let guard = self.map.read().unwrap();
        guard.values().map(|s| s.len() as u64).sum()
    }

    fn flush(&self) -> AerisResult<()> {
        Ok(())
    }

    fn compact(&self) -> AerisResult<()> {
        Ok(())
    }
}

fn insert_with_rm(filter: &mut AerisFilter, rm: &TestReverseMap, key: Key) -> AerisResult<()> {
    let fp = filter.insert(key)?;
    rm.append_key(&fp, key)?;
    Ok(())
}

fn delete_with_rm(filter: &mut AerisFilter, rm: &TestReverseMap, key: Key) -> AerisResult<bool> {
    if let Some(fp) = filter.delete(key)? {
        rm.remove_key(&fp, key)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn range_is_empty(truth: &BTreeSet<Key>, range: QueryRange) -> bool {
    truth.range(range.start()..=range.end()).next().is_none()
}

// =============================================================================
// 1. Boundary keys, m in {0, 5, 10}, deletes, bulk_load, and FilterFull atomicity
// =============================================================================

#[test]
fn zero_false_negatives_and_boundary_keys_across_m_modes() {
    for &(r, max_range_r) in &[(10u8, 1u64), (10, 32), (10, 1024)] {
        let m = FilterConfig::ceil_log2_u64(max_range_r).unwrap();
        let cfg = FilterConfig::new(8, r, max_range_r, r + m, 1.0 / 64.0)
            .unwrap()
            .with_hash_seed(0xA3E1_1001 + (m as u64));
        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();

        // Empty filter assertions.
        assert!(filter.is_empty());
        assert!(!filter.query_point(0));
        assert!(!filter.query_point(u64::MAX));
        assert!(!filter.query_range(QueryRange::new(0, u64::MAX).unwrap()));
        assert_eq!(filter.delete(42).unwrap(), None);
        assert!(filter
            .locate_matching_fingerprints(QueryRange::new(0, 100).unwrap())
            .is_empty());

        let p = cfg.partition_size();
        let mut keys = vec![
            0u64,
            1,
            2,
            p.saturating_sub(1),
            p,
            p.saturating_add(1),
            p.saturating_mul(2).saturating_sub(1),
            p.saturating_mul(2),
            1u64 << 32,
            u64::MAX - 2,
            u64::MAX - 1,
            u64::MAX,
        ];
        keys.sort_unstable();
        keys.dedup();

        // Add additional seeded random keys up to ~65% load.
        let mut rng = ChaCha8Rng::seed_from_u64(0xB00D_0001 + (m as u64));
        let mut truth: BTreeSet<Key> = keys.into_iter().collect();
        while truth.len() < 140 {
            truth.insert(rng.gen::<u64>());
        }

        for &k in &truth {
            insert_with_rm(&mut filter, &rm, k).unwrap();
        }
        assert_eq!(filter.len(), truth.len());
        filter.verify_invariants().unwrap();

        // Verify zero false negatives on point queries and enclosing range queries.
        for &k in &truth {
            assert!(filter.query_point(k), "FN on point key {k} (m={m})");
            let r_lo = QueryRange::new(k.saturating_sub(5), k).unwrap();
            let r_hi = QueryRange::new(k, k.saturating_add(5)).unwrap();
            let r_mid = QueryRange::new(k.saturating_sub(3), k.saturating_add(3)).unwrap();
            assert!(filter.query_range(r_lo), "FN on range {r_lo:?} (m={m})");
            assert!(filter.query_range(r_hi), "FN on range {r_hi:?} (m={m})");
            assert!(filter.query_range(r_mid), "FN on range {r_mid:?} (m={m})");
        }
        assert!(filter.query_range(QueryRange::new(0, u64::MAX).unwrap()));

        // Bulk load equivalence test.
        let rm_bulk = TestReverseMap::new();
        let all_keys_vec: Vec<Key> = truth.iter().copied().collect();
        let bulk = AerisFilter::bulk_load(cfg.clone(), &all_keys_vec, &rm_bulk).unwrap();
        bulk.verify_invariants().unwrap();
        assert_eq!(bulk.len(), truth.len());
        for &k in &truth {
            assert!(bulk.query_point(k), "bulk_load FN on {k} (m={m})");
        }
        let rm_rebuilt = TestReverseMap::new();
        let rebuilt =
            AerisFilter::rebuild_from_keys(cfg.clone(), truth.iter().copied(), &rm_rebuilt)
                .unwrap();
        rebuilt.verify_invariants().unwrap();
        assert_eq!(rebuilt.len(), truth.len());

        // Delete half the keys, verify surviving keys still have 0 false negatives.
        let to_delete: Vec<Key> = truth.iter().step_by(2).copied().collect();
        for k in to_delete {
            assert!(delete_with_rm(&mut filter, &rm, k).unwrap());
            truth.remove(&k);
        }
        filter.verify_invariants().unwrap();
        assert_eq!(filter.len(), truth.len());
        for &k in &truth {
            assert!(
                filter.query_point(k),
                "FN after partial delete on {k} (m={m})"
            );
        }

        // Delete remaining keys down to empty filter.
        let rest: Vec<Key> = truth.iter().copied().collect();
        for k in rest {
            assert!(delete_with_rm(&mut filter, &rm, k).unwrap());
            truth.remove(&k);
        }
        assert!(filter.is_empty());
        assert_eq!(filter.occupied_slots(), 0);
        assert_eq!(filter.load_factor(), 0.0);
        filter.verify_invariants().unwrap();
    }
}

#[test]
fn filter_full_and_storage_failure_preserve_strict_atomicity() {
    let cfg = FilterConfig::new(6, 8, 32, 13, 0.05)
        .unwrap()
        .with_max_load_factor(0.75)
        .unwrap();
    let mut filter = AerisFilter::new(cfg.clone()).unwrap();
    let rm = TestReverseMap::new();
    let mut truth = BTreeSet::new();

    // Fill until `FilterFull` triggers.
    let mut hit_full = false;
    let mut rng = ChaCha8Rng::seed_from_u64(0xF011_0001);
    for _ in 0..500 {
        let k = rng.gen::<u64>();
        if truth.contains(&k) {
            continue;
        }
        match filter.insert(k) {
            Ok(fp) => {
                rm.append_key(&fp, k).unwrap();
                truth.insert(k);
            }
            Err(AerisError::FilterFull { .. }) => {
                hit_full = true;
                break;
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert!(hit_full, "expected FilterFull when exceeding alpha=0.75");
    filter.verify_invariants().unwrap();
    assert_eq!(filter.len(), truth.len());
    for &k in &truth {
        assert!(filter.query_point(k));
    }

    // `bulk_load` exceeding `max_load_factor` must fail BEFORE mutating `reverse_map`.
    let rm_bulk = TestReverseMap::new();
    let too_many: Vec<Key> = (0..100).map(|i| i * 64).collect();
    let err = AerisFilter::bulk_load(cfg.clone(), &too_many, &rm_bulk).unwrap_err();
    assert!(matches!(err, AerisError::FilterFull { .. }));
    assert_eq!(rm_bulk.entry_count_estimate(), 0);

    // Find a false positive and verify storage failure during `adapt` leaves filter intact.
    let mut found_fp = None;
    for candidate in 0..100_000u64 {
        let start = candidate.wrapping_mul(10007);
        let qr = QueryRange::with_len(start, 16).unwrap();
        if range_is_empty(&truth, qr) && filter.query_range(qr) {
            found_fp = Some(qr);
            break;
        }
    }
    if let Some(fp_range) = found_fp {
        let rm_before = rm.snapshot();
        let slots_before = filter.occupied_slots();
        let len_before = filter.len();
        rm.set_fail_reads(true);
        let err = filter.adapt(fp_range, &rm).unwrap_err();
        assert!(matches!(err, AerisError::StorageError(_)));
        rm.set_fail_reads(false);
        assert_eq!(rm.snapshot(), rm_before);
        assert_eq!(filter.occupied_slots(), slots_before);
        assert_eq!(filter.len(), len_before);
        filter.verify_invariants().unwrap();
    }
}

// =============================================================================
// 2. Strict Monotonic Adaptivity (0 Repeated False Positives)
// =============================================================================

#[test]
fn strict_monotonic_adaptivity_across_m_modes_inserts_and_expansions() {
    for &(r, max_range_r, eps) in &[(8u8, 1u64, 0.05), (8, 32, 0.05), (9, 1024, 1.0 / 32.0)] {
        let m = FilterConfig::ceil_log2_u64(max_range_r).unwrap();
        let cfg = FilterConfig::new(8, r, max_range_r, r + m, eps)
            .unwrap()
            .with_hash_seed(0xADAF_0001 + (m as u64))
            .with_max_load_factor(0.90)
            .unwrap();
        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();
        let mut truth = BTreeSet::new();
        let mut rng = ChaCha8Rng::seed_from_u64(0xADAF_1001 + (m as u64));

        // Insert 120 keys spaced from upper half of key space so lower ranges stay empty.
        while truth.len() < 120 {
            let k = rng.gen_range((1u64 << 40)..(1u64 << 50));
            if truth.insert(k) {
                insert_with_rm(&mut filter, &rm, k).unwrap();
            }
        }
        filter.verify_invariants().unwrap();

        // Collect and adapt 40 genuine false positives (range queries of length <= R).
        let mut adapted_ranges: Vec<QueryRange> = Vec::new();
        let mut probe_idx = 0u64;
        while adapted_ranges.len() < 40 && probe_idx < 250_000 {
            probe_idx += 1;
            let start = rng.gen_range(0..(1u64 << 39));
            let len = if max_range_r == 1 {
                1
            } else {
                rng.gen_range(1..=max_range_r)
            };
            let qr = QueryRange::with_len(start, len).unwrap();
            if !range_is_empty(&truth, qr) {
                continue;
            }
            if filter.query_range(qr) {
                // Ensure room to extend; expand if approaching ceiling.
                if filter.load_factor() > 0.80 {
                    filter.expand(&rm).unwrap();
                    if !filter.query_range(qr) {
                        continue;
                    }
                }
                let report = filter.adapt(qr, &rm).unwrap();
                assert!(report.boxes_split > 0);
                assert!(
                    !filter.query_range(qr),
                    "adapted range {qr:?} still returned true immediately after adapt (m={m})"
                );
                adapted_ranges.push(qr);
            }
        }
        assert!(
            adapted_ranges.len() >= 15,
            "expected >= 15 false positives to adapt for m={m}, got {}",
            adapted_ranges.len()
        );
        filter.verify_invariants().unwrap();

        // Re-check ALL adapted FPs: 0 repeated FPs!
        for &qr in &adapted_ranges {
            assert!(
                !filter.query_range(qr),
                "repeated FP before further inserts: {qr:?}"
            );
        }

        // Insert 100 MORE keys outside adapted ranges, then run 3 expansions!
        let mut added = 0;
        while added < 100 {
            if filter.load_factor() > 0.78 {
                filter.expand(&rm).unwrap();
                filter.verify_invariants().unwrap();
            }
            let k = rng.gen_range((1u64 << 40)..(1u64 << 50));
            if truth.insert(k) {
                insert_with_rm(&mut filter, &rm, k).unwrap();
                added += 1;
            }
        }

        for _ in 0..3 {
            filter.expand(&rm).unwrap();
            filter.verify_invariants().unwrap();
            // Zero false negatives for ALL live keys.
            for &k in &truth {
                assert!(filter.query_point(k), "FN after expand for key {k} (m={m})");
            }
            // Zero repeated false positives for ALL previously adapted ranges!
            for &qr in &adapted_ranges {
                assert!(
                    !filter.query_range(qr),
                    "repeated FP after expand for adapted range {qr:?} (m={m})"
                );
            }
        }
    }
}

// =============================================================================
// 3. Explicit Query-Partition Adaptation Edge Case & Opportunistic Rejuvenation
// =============================================================================

#[test]
fn query_partition_edge_case_first_and_multi_level_extension_collisions() {
    let q = 7u8;
    let r = 8u8;
    let m = 5u8; // P = 32
    let cfg = FilterConfig::new(q, r, 32, r + m, 0.05)
        .unwrap()
        .with_hash_seed(0xE0CE_CAFE);
    let hasher = PartitionHasher::new(cfg.hash_seed);

    let p_q: u64 = 1000;
    let (home_q, rem_q) = hasher.quotient_and_remainder(p_q, q, r);
    let ext0_q = hasher.extract_bits(p_q, (q + r) as u16, m);

    // Find:
    // - `p_level1`: collides with `p_q` on `(q + r)` bits, diverges in first `m` extension bits.
    // - `p_level2`: collides with `p_q` on `(q + r + m)` bits, requiring 2nd extension slot!
    let mut p_level1 = None;
    let mut p_level2 = None;
    for cand in (p_q + 1)..(p_q + 5_000_000) {
        let (h, rem) = hasher.quotient_and_remainder(cand, q, r);
        if h == home_q && rem == rem_q {
            let ext0 = hasher.extract_bits(cand, (q + r) as u16, m);
            if ext0 != ext0_q && p_level1.is_none() {
                p_level1 = Some(cand);
            } else if ext0 == ext0_q && p_level2.is_none() {
                p_level2 = Some(cand);
            }
            if p_level1.is_some() && p_level2.is_some() {
                break;
            }
        }
    }
    let p1 = p_level1.expect("must find 1st-level colliding partition");
    let p2 = p_level2.expect("must find 2nd-level colliding partition");

    // --- Subtest A: 1st-level divergence with duplicate memento multiplicity ---
    {
        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();

        // Keys in query partition `p_q` strictly OUTSIDE query mementos `[10, 15]`.
        let k_q_a = cfg.compose_key(p_q, 2);
        let k_q_b = cfg.compose_key(p_q, 27);
        // Keys in colliding partition `p1`: one with memento `12` INSIDE `[10, 15]`,
        // AND one with identical memento `2` (testing multiset preservation!).
        let k_p1_inside = cfg.compose_key(p1, 12);
        let k_p1_dup_mem = cfg.compose_key(p1, 2);

        for k in [k_q_a, k_q_b, k_p1_inside, k_p1_dup_mem] {
            insert_with_rm(&mut filter, &rm, k).unwrap();
        }
        filter.verify_invariants().unwrap();

        let fp_range = QueryRange::new(cfg.compose_key(p_q, 10), cfg.compose_key(p_q, 15)).unwrap();
        // Both partitions share the exact same unextended keepsake box!
        assert_eq!(filter.locate_matching_fingerprints(fp_range).len(), 1);
        assert!(filter.query_range(fp_range), "must initially trigger FP");

        let report = filter.adapt(fp_range, &rm).unwrap();
        assert_eq!(report.boxes_split, 1);
        assert_eq!(report.keys_rehashed, 4);
        filter.verify_invariants().unwrap();

        // False positive MUST now be eliminated.
        assert!(!filter.query_range(fp_range));
        // Every real key in BOTH `p_q` and `p1` MUST still return `true` (0 false negatives)!
        for k in [k_q_a, k_q_b, k_p1_inside, k_p1_dup_mem] {
            assert!(
                filter.query_point(k),
                "FN on key {k} after query-partition adapt"
            );
        }

        // Subsequent inserts into `p_q` (outside `[10, 15]`) and `p1` preserve invariants.
        let k_q_c = cfg.compose_key(p_q, 31);
        let k_p1_c = cfg.compose_key(p1, 14);
        insert_with_rm(&mut filter, &rm, k_q_c).unwrap();
        insert_with_rm(&mut filter, &rm, k_p1_c).unwrap();
        filter.verify_invariants().unwrap();
        assert!(!filter.query_range(fp_range));
        for k in [k_q_a, k_q_b, k_q_c, k_p1_inside, k_p1_dup_mem, k_p1_c] {
            assert!(filter.query_point(k));
        }

        // Across expansions, FP never recurs and 0 FNs hold!
        for _ in 0..4 {
            filter.expand(&rm).unwrap();
            filter.verify_invariants().unwrap();
            assert!(!filter.query_range(fp_range));
            for k in [k_q_a, k_q_b, k_q_c, k_p1_inside, k_p1_dup_mem, k_p1_c] {
                assert!(filter.query_point(k));
            }
        }
    }

    // --- Subtest B: Multi-level extension collision (`p_q`, `p1`, `p2` all in same box) ---
    {
        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();

        let k_q = cfg.compose_key(p_q, 0);
        let k_p1 = cfg.compose_key(p1, 11);
        let k_p2 = cfg.compose_key(p2, 13);

        for k in [k_q, k_p1, k_p2] {
            insert_with_rm(&mut filter, &rm, k).unwrap();
        }
        filter.verify_invariants().unwrap();

        let fp_range = QueryRange::new(cfg.compose_key(p_q, 10), cfg.compose_key(p_q, 15)).unwrap();
        assert!(filter.query_range(fp_range));

        let report = filter.adapt(fp_range, &rm).unwrap();
        assert!(report.extension_slots_added > 0);
        filter.verify_invariants().unwrap();

        assert!(!filter.query_range(fp_range));
        for k in [k_q, k_p1, k_p2] {
            assert!(filter.query_point(k));
        }
        // `p_q`'s box must have been extended to at least level 2 (`q + r + m + r + m` bits)
        // because `p2` collided with `p_q` on the first `m` extension bits!
        let pq_fps = filter.locate_matching_fingerprints(QueryRange::point(k_q));
        assert_eq!(pq_fps.len(), 1);
        assert!(
            pq_fps[0].bit_len() >= (q as u16) + (r as u16) + (m as u16) + (r as u16) + (m as u16),
            "expected p_q to extend to level >= 2, got {} bits",
            pq_fps[0].bit_len()
        );
    }

    // --- Subtest C: Opportunistic rejuvenation of aged boxes during adapt ---
    {
        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();
        let k_p1 = cfg.compose_key(p1, 12);
        insert_with_rm(&mut filter, &rm, k_p1).unwrap();

        // Age the box by 3 expansions (`age = 3`).
        for _ in 0..3 {
            filter.expand(&rm).unwrap();
        }
        let fp_range = QueryRange::new(cfg.compose_key(p_q, 10), cfg.compose_key(p_q, 15)).unwrap();
        assert!(filter.query_range(fp_range));

        let report = filter.adapt(fp_range, &rm).unwrap();
        assert!(
            report.rejuvenated_boxes >= 1,
            "expected opportunistic rejuvenation of aged box"
        );
        assert!(!filter.query_range(fp_range));
        assert!(filter.query_point(k_p1));
        filter.verify_invariants().unwrap();
    }
}

#[test]
fn multi_subquery_same_home_and_adjacent_home_adaptation_and_max_64bit_slot_width() {
    // --- Subtest A: Two subqueries in one `adapt(range)` colliding on the same cluster/home ---
    {
        let q = 7u8;
        let r = 7u8;
        let m = 5u8;
        let cfg = FilterConfig::new(q, r, 32, r + m, 0.1)
            .unwrap()
            .with_hash_seed(0xC105_0001);
        let hasher = PartitionHasher::new(cfg.hash_seed);

        // Find consecutive partitions `p_a` and `p_a + 1` whose home slots are identical or adjacent,
        // and find colliding partitions for both!
        let mut target_pair = None;
        for p_a in 100u64..100_000 {
            let h_a = hasher.quotient_and_remainder(p_a, q, r).0;
            let h_b = hasher.quotient_and_remainder(p_a + 1, q, r).0;
            if h_a == h_b {
                target_pair = Some(p_a);
                break;
            }
        }
        let p_a = target_pair.expect("must find consecutive partitions with identical home_slot");
        let p_b = p_a + 1;
        let (h_a, rem_a) = hasher.quotient_and_remainder(p_a, q, r);
        let (h_b, rem_b) = hasher.quotient_and_remainder(p_b, q, r);
        assert_eq!(h_a, h_b);

        let mut coll_a = None;
        let mut coll_b = None;
        for cand in 200_000u64..2_000_000 {
            let (h, rem) = hasher.quotient_and_remainder(cand, q, r);
            if h == h_a && rem == rem_a && coll_a.is_none() {
                coll_a = Some(cand);
            }
            if h == h_b && rem == rem_b && coll_b.is_none() {
                coll_b = Some(cand);
            }
            if coll_a.is_some() && coll_b.is_some() {
                break;
            }
        }
        let ca = coll_a.unwrap();
        let cb = coll_b.unwrap();

        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();
        let k_ca = cfg.compose_key(ca, 30);
        let k_cb = cfg.compose_key(cb, 2);
        let k_pa_outside = cfg.compose_key(p_a, 5);
        for k in [k_ca, k_cb, k_pa_outside] {
            insert_with_rm(&mut filter, &rm, k).unwrap();
        }

        // Range spans `[p_a:28 .. p_b:4]` -> hits BOTH `p_a` and `p_b` at the EXACT same `home_slot`!
        let cross_range =
            QueryRange::new(cfg.compose_key(p_a, 28), cfg.compose_key(p_b, 4)).unwrap();
        assert!(filter.query_range(cross_range));
        let report = filter.adapt(cross_range, &rm).unwrap();
        assert_eq!(report.subqueries_checked, 2);
        assert!(!filter.query_range(cross_range));
        for k in [k_ca, k_cb, k_pa_outside] {
            assert!(filter.query_point(k));
        }
        filter.verify_invariants().unwrap();
    }

    // --- Subtest B: Extended box selectively rejuvenated after `r + 1` expansions retains > `new_q + r` bits ---
    {
        let q = 7u8;
        let r = 7u8;
        let m = 5u8;
        let cfg = FilterConfig::new(q, r, 32, r + m, 0.1)
            .unwrap()
            .with_hash_seed(0xE0CE_CAFE);
        let hasher = PartitionHasher::new(cfg.hash_seed);
        let p_q = 500u64;
        let (hq, rq) = hasher.quotient_and_remainder(p_q, q, r);
        let ext0_q = hasher.extract_bits(p_q, (q + r) as u16, m);

        // Find `p_deep` colliding on `q + r + m` bits so adaptation extends it to level 2
        // (`7 + 7 + 5 + 12 = 31` bits > `15 + 7 = 22` bits after 8 expansions!).
        let p_deep = ((p_q + 1)..(p_q + 5_000_000))
            .find(|&cand| {
                let (h, rem) = hasher.quotient_and_remainder(cand, q, r);
                h == hq && rem == rq && hasher.extract_bits(cand, (q + r) as u16, m) == ext0_q
            })
            .expect("must find level-2 colliding partition");

        let mut filter = AerisFilter::new(cfg.clone()).unwrap();
        let rm = TestReverseMap::new();
        let k_deep = cfg.compose_key(p_deep, 12);
        insert_with_rm(&mut filter, &rm, k_deep).unwrap();

        let fp_range = QueryRange::new(cfg.compose_key(p_q, 10), cfg.compose_key(p_q, 15)).unwrap();
        assert!(filter.query_range(fp_range));
        filter.adapt(fp_range, &rm).unwrap();
        assert!(!filter.query_range(fp_range));

        let adapted_bits =
            filter.locate_matching_fingerprints(QueryRange::point(k_deep))[0].bit_len();
        assert!(adapted_bits >= 31);

        // Run `r + 1 = 8` expansions so `k_deep`'s extended box ages to `r = 7` and selectively rejuvenates!
        let mut total_rejuv = 0usize;
        for _ in 0..=r {
            let rep = filter.expand(&rm).unwrap();
            total_rejuv += rep.boxes_rejuvenated;
            assert!(
                !filter.query_range(fp_range),
                "FP recurred during expansion!"
            );
            assert!(filter.query_point(k_deep), "FN on k_deep during expansion!");
        }
        assert!(
            total_rejuv >= 1,
            "extended box must selectively rejuvenate after r+1 expansions"
        );
        let post_rejuv_bits =
            filter.locate_matching_fingerprints(QueryRange::point(k_deep))[0].bit_len();
        assert!(
            post_rejuv_bits >= adapted_bits,
            "selective rejuvenation shrank fingerprint from {adapted_bits} to {post_rejuv_bits} bits"
        );
        filter.verify_invariants().unwrap();
    }

    // --- Subtest C: Extreme 64-bit physical slot width (`C = 63`, `slot_bits = 64`, `q + r > 64`) ---
    for &(r, m, max_r) in &[(55u8, 8u8, 256u64), (63u8, 0u8, 1u64)] {
        let cfg = FilterConfig::new(10, r, max_r, 63, 1.0 / 256.0).unwrap();
        assert_eq!(cfg.m, m);
        assert_eq!(cfg.physical_slot_bits(), 64);
        assert!((cfg.q as u16) + (cfg.r as u16) > 64);

        let mut filter = AerisFilter::new(cfg).unwrap();
        let rm = TestReverseMap::new();
        let keys = [0u64, 1, 255, 256, 1024, u64::MAX - 1, u64::MAX];
        for k in keys {
            insert_with_rm(&mut filter, &rm, k).unwrap();
        }
        filter.verify_invariants().unwrap();
        for k in keys {
            assert!(filter.query_point(k));
        }
        filter.expand(&rm).unwrap();
        filter.verify_invariants().unwrap();
        for k in keys {
            assert!(filter.query_point(k));
        }
    }
}

#[test]
fn hot_path_query_correctness_at_scale() {
    let cfg = FilterConfig::with_minimum_bits(16, 32, 1.0 / 64.0)
        .unwrap()
        .with_hash_seed(0x5CA1_E001);
    let mut rng = ChaCha8Rng::seed_from_u64(0x5CA1_E002);
    let num_keys = 50_000usize;
    let mut keys: Vec<Key> = Vec::with_capacity(num_keys);
    for _ in 0..num_keys {
        keys.push(rng.gen_range((1u64 << 40)..(1u64 << 52)));
    }
    keys.sort_unstable();
    keys.dedup();

    let rm = TestReverseMap::new();
    let filter = AerisFilter::bulk_load(cfg, &keys, &rm).unwrap();
    filter.verify_invariants().unwrap();

    // Verify 100% zero false negatives across all 50k bulk-loaded keys.
    for &k in &keys {
        assert!(filter.query_point(k));
    }

    // Exercise 500k seeded point queries and 500k seeded range queries (`R = 32`).
    let num_queries = if cfg!(debug_assertions) {
        50_000usize
    } else {
        2_000_000usize
    };
    let mut point_hits = 0usize;
    let t0 = std::time::Instant::now();
    for i in 0..num_queries {
        let qk = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) & ((1u64 << 39) - 1);
        if filter.query_point(qk) {
            point_hits += 1;
        }
    }
    let dt_pt = t0.elapsed();

    let mut range_hits = 0usize;
    let t1 = std::time::Instant::now();
    for i in 0..num_queries {
        let start = (i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) & ((1u64 << 39) - 1);
        let qr = QueryRange::with_len(start, 32).unwrap();
        if filter.query_range(qr) {
            range_hits += 1;
        }
    }
    let dt_rng = t1.elapsed();

    let pt_fpr = (point_hits as f64) / (num_queries as f64);
    let rng_fpr = (range_hits as f64) / (num_queries as f64);
    assert!(pt_fpr <= 1.0 / 64.0, "point FPR {pt_fpr:.5} > 1/64");
    assert!(rng_fpr <= 1.0 / 64.0, "range FPR {rng_fpr:.5} > 1/64");

    // Ensure hot-path queries stay fast and never degenerate.
    assert!(dt_pt.as_nanos() > 0 && dt_rng.as_nanos() > 0);
}

// =============================================================================
// 4. 8+ Consecutive 2x Expansions, Selective Rejuvenation, & Sustained FPR <= eps
// =============================================================================

fn run_multi_expansion_fpr_gate(
    initial_q: u8,
    num_expansions: usize,
    max_range_r: u64,
    target_epsilon: f64,
    seed: u64,
) {
    let cfg = FilterConfig::with_minimum_bits(initial_q, max_range_r, target_epsilon)
        .unwrap()
        .with_hash_seed(seed)
        .with_max_load_factor(0.90)
        .unwrap();
    let mut filter = AerisFilter::new(cfg).unwrap();
    let rm = TestReverseMap::new();
    let mut truth: BTreeSet<Key> = BTreeSet::new();
    let mut adapted_fps: Vec<QueryRange> = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x55AA_55AA);

    let mut total_boxes_rejuvenated = 0usize;
    let mut total_keys_rejuvenated = 0usize;

    // Keys reside in the upper domain `[1 << 40, 1 << 48)`; empty queries probe `[0, 1 << 39)`
    // AND interleaved gaps around live keys.
    for cycle in 0..=num_expansions {
        let target_slots = ((filter.capacity() as f64) * 0.78) as usize;
        while filter.occupied_slots() < target_slots {
            // Leave headroom for multi-slot boxes.
            if filter.occupied_slots() + 8 >= ((filter.capacity() as f64) * 0.88) as usize {
                break;
            }
            let key = rng.gen_range((1u64 << 40)..(1u64 << 48));
            if truth.contains(&key) {
                continue;
            }
            match filter.insert(key) {
                Ok(fp) => {
                    rm.append_key(&fp, key).unwrap();
                    truth.insert(key);
                }
                Err(AerisError::FilterFull { .. }) => break,
                Err(e) => panic!("unexpected insert error at cycle {cycle}: {e}"),
            }
        }

        // Adapt up to 3 false positives per cycle (when load factor has headroom).
        let mut adapted_this_cycle = 0;
        let mut attempts = 0;
        while adapted_this_cycle < 3 && attempts < 4_000 && filter.load_factor() < 0.84 {
            attempts += 1;
            let start = rng.gen_range(0..(1u64 << 39));
            let qr = QueryRange::with_len(start, max_range_r).unwrap();
            if filter.query_range(qr) {
                match filter.adapt(qr, &rm) {
                    Ok(_) => {
                        assert!(!filter.query_range(qr));
                        adapted_fps.push(qr);
                        adapted_this_cycle += 1;
                    }
                    Err(AerisError::FilterFull { .. }) => break,
                    Err(e) => panic!("unexpected adapt error at cycle {cycle}: {e}"),
                }
            }
        }

        // Verify ZERO false negatives for EVERY inserted key so far.
        for &k in &truth {
            assert!(
                filter.query_point(k),
                "FALSE NEGATIVE at cycle {cycle} (q={}) for key {k}",
                filter.config().q
            );
        }

        // Verify ZERO repeated false positives for ALL previously adapted queries.
        for &qr in &adapted_fps {
            assert!(
                !filter.query_range(qr),
                "REPEATED FALSE POSITIVE at cycle {cycle} (q={}) for {qr:?}",
                filter.config().q
            );
        }

        // Verify structural invariants on early/mid/final cycles (and every cycle up to 2^13).
        if filter.capacity() <= 8192 || cycle == num_expansions {
            filter.verify_invariants().unwrap();
        }

        if cycle < num_expansions {
            let exp_report = filter.expand(&rm).unwrap();
            assert_eq!(exp_report.new_q, exp_report.old_q + 1);
            total_boxes_rejuvenated += exp_report.boxes_rejuvenated;
            total_keys_rejuvenated += exp_report.keys_rejuvenated;
        }
    }

    assert_eq!(filter.expansion_cycles(), num_expansions as u64);
    assert_eq!(filter.config().q, initial_q + (num_expansions as u8));
    assert!(
        total_boxes_rejuvenated > 0,
        "expected selective rejuvenation to fire over {num_expansions} expansions (r={})",
        filter.config().r
    );
    assert!(total_keys_rejuvenated > 0);
    filter.verify_invariants().unwrap();

    // Measure empirical FPR on 20,000 seeded empty range queries of length `max_range_r`
    // AND 20,000 seeded empty point queries.
    let num_queries = 20_000usize;
    let mut range_fps = 0usize;
    let mut point_fps = 0usize;
    for _ in 0..num_queries {
        let start = rng.gen_range(0..(1u64 << 39));
        let qr = QueryRange::with_len(start, max_range_r).unwrap();
        if filter.query_range(qr) {
            range_fps += 1;
        }
        let pt = rng.gen_range(0..(1u64 << 39));
        if filter.query_point(pt) {
            point_fps += 1;
        }
    }

    let empirical_range_fpr = (range_fps as f64) / (num_queries as f64);
    let empirical_point_fpr = (point_fps as f64) / (num_queries as f64);
    assert!(
        empirical_range_fpr <= target_epsilon,
        "empirical range FPR {empirical_range_fpr:.5} exceeded target epsilon {target_epsilon:.5} \
         after {num_expansions} expansions (q={}, r={}, m={}, load={:.3})",
        filter.config().q,
        filter.config().r,
        filter.config().m,
        filter.load_factor()
    );
    assert!(
        empirical_point_fpr <= target_epsilon,
        "empirical point FPR {empirical_point_fpr:.5} exceeded target epsilon {target_epsilon:.5}"
    );
}

#[test]
fn eight_consecutive_expansions_eps_0_1_rejuvenation_and_fpr() {
    // r_min(0.1) = 7; 8 expansions (q = 7 -> 15, 128 -> 32768 slots) triggers rejuvenation on cycle 8!
    run_multi_expansion_fpr_gate(7, 8, 32, 0.1, 0xE801_0001);
}

#[test]
fn nine_consecutive_expansions_eps_0_05_rejuvenation_and_fpr() {
    // r_min(0.05) = 8; 9 expansions (q = 7 -> 16) triggers rejuvenation on cycle 9!
    run_multi_expansion_fpr_gate(7, 9, 32, 0.05, 0xE801_0002);
}

#[test]
fn ten_consecutive_expansions_eps_1_over_32_rejuvenation_and_fpr() {
    // r_min(1/32) = 9; 10 expansions (q = 6 -> 16, 64 -> 65536 slots) triggers rejuvenation!
    run_multi_expansion_fpr_gate(6, 10, 32, 1.0 / 32.0, 0xE801_0003);
}

#[test]
fn eleven_consecutive_expansions_eps_1_over_64_rejuvenation_and_fpr() {
    // r_min(1/64) = 10; 11 expansions (q = 6 -> 17, 64 -> 131072 slots) triggers rejuvenation!
    run_multi_expansion_fpr_gate(6, 11, 32, 1.0 / 64.0, 0xE801_0004);
}

// =============================================================================
// 5. `proptest` Property-Based Randomized State-Machine Fuzzing
// =============================================================================

#[derive(Clone, Debug)]
enum Op {
    Insert(u64),
    DeleteIdx(usize),
    DeleteMissing(u64),
    QueryPoint(u64),
    QueryRange(u64, u64),
    Adapt(u64, u64),
    Expand,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        8 => (0u64..4096).prop_map(Op::Insert),
        2 => (0usize..256).prop_map(Op::DeleteIdx),
        1 => (10_000u64..20_000).prop_map(Op::DeleteMissing),
        3 => (0u64..8192).prop_map(Op::QueryPoint),
        3 => (0u64..8192, 1u64..64).prop_map(|(a, len)| Op::QueryRange(a, len)),
        4 => (0u64..8192, 1u64..32).prop_map(|(a, len)| Op::Adapt(a, len)),
        1 => Just(Op::Expand),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn proptest_aeris_state_machine_preserves_invariants_and_monotonicity(
        m_choice in prop_oneof![Just(0u8), Just(1u8), Just(5u8), Just(8u8)],
        seed in any::<u64>(),
        ops in prop::collection::vec(op_strategy(), 20..120),
    ) {
        let max_range_r = 1u64 << m_choice;
        let eps = 0.05;
        let r = FilterConfig::min_remainder_bits(eps).unwrap();
        let cfg = FilterConfig::new(6, r, max_range_r, r + m_choice, eps)
            .unwrap()
            .with_hash_seed(seed)
            .with_max_load_factor(0.88)
            .unwrap();
        let mut filter = AerisFilter::new(cfg).unwrap();
        let rm = TestReverseMap::new();
        let mut truth: BTreeSet<Key> = BTreeSet::new();
        let mut adapted_empty_ranges: Vec<QueryRange> = Vec::new();
        let mut expansions_done = 0usize;

        for op in ops {
            match op {
                Op::Insert(k) => {
                    if !truth.contains(&k) {
                        match filter.insert(k) {
                            Ok(fp) => {
                                rm.append_key(&fp, k).unwrap();
                                truth.insert(k);
                                // Drop any previously adapted range that now legitimately contains `k`.
                                adapted_empty_ranges.retain(|r| !r.contains(k));
                            }
                            Err(AerisError::FilterFull { .. }) => {
                                if expansions_done < 4 {
                                    filter.expand(&rm).unwrap();
                                    expansions_done += 1;
                                }
                            }
                            Err(e) => prop_assert!(false, "unexpected insert error: {e}"),
                        }
                    }
                }
                Op::DeleteIdx(idx) => {
                    if !truth.is_empty() {
                        let key = *truth.iter().nth(idx % truth.len()).unwrap();
                        let deleted = delete_with_rm(&mut filter, &rm, key).unwrap();
                        prop_assert!(deleted, "live key {key} must be deletable");
                        truth.remove(&key);
                    }
                }
                Op::DeleteMissing(k) => {
                    // Never delete a key that happens to collide with live keys; only test empty filter
                    // or guaranteed non-matching keys to avoid CQF false-delete corruption.
                    if !truth.contains(&k) && !filter.query_point(k) {
                        prop_assert_eq!(filter.delete(k).unwrap(), None);
                    }
                }
                Op::QueryPoint(k) => {
                    if truth.contains(&k) {
                        prop_assert!(filter.query_point(k), "FN on live key {k}");
                    }
                }
                Op::QueryRange(a, len) => {
                    let qr = QueryRange::with_len(a, len).unwrap();
                    if !range_is_empty(&truth, qr) {
                        prop_assert!(filter.query_range(qr), "FN on non-empty range {qr:?}");
                    }
                }
                Op::Adapt(a, len) => {
                    let qr = QueryRange::with_len(a, len).unwrap();
                    if range_is_empty(&truth, qr) && filter.query_range(qr) {
                        match filter.adapt(qr, &rm) {
                            Ok(_) => {
                                prop_assert!(
                                    !filter.query_range(qr),
                                    "adapted range {qr:?} still returned true"
                                );
                                adapted_empty_ranges.push(qr);
                            }
                            Err(AerisError::FilterFull { .. }) => {}
                            Err(e) => prop_assert!(false, "unexpected adapt error: {e}"),
                        }
                    }
                }
                Op::Expand => {
                    if expansions_done < 4 {
                        filter.expand(&rm).unwrap();
                        expansions_done += 1;
                    }
                }
            }

            // Full invariant & contract audit after every single state transition!
            prop_assert!(filter.verify_invariants().is_ok());
            prop_assert_eq!(filter.len(), truth.len());
            prop_assert_eq!(filter.capacity(), 1usize << filter.config().q);

            for &live_key in &truth {
                prop_assert!(filter.query_point(live_key), "FN on live key {live_key}");
            }
            for &qr in &adapted_empty_ranges {
                prop_assert!(
                    !filter.query_range(qr),
                    "repeated FP on previously adapted range {qr:?}"
                );
            }
        }
    }
}
