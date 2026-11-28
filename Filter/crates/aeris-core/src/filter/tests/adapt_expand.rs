use super::super::*;
use super::CountingReverseMap;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
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

    let cfg = FilterConfig::new(17, 7, 32, 12, 0.08).unwrap();
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
    let cfg = FilterConfig::new(6, 7, 32, 12, 0.08).unwrap();
    let hasher = PartitionHasher::new(cfg.hash_seed);
    let p_q: PartitionId = 100;
    let (target_home, target_rem) = hasher.quotient_and_remainder(p_q, cfg.q, cfg.r);

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
    let k_q_outside_lo = (p_q << 5) | 1;
    let k_q_outside_hi = (p_q << 5) | 28;
    let k_c1_inside = (colliding_parts[0] << 5) | 15;
    let k_c2_inside = (colliding_parts[1] << 5) | 18;
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

    let cfg2 = FilterConfig::new(7, 7, 32, 12, 0.08).unwrap();
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

    f2.expand(&rm2).unwrap();
    f2.verify_invariants().unwrap();
    for &r in &adapted_ranges {
        assert!(
            !f2.query_range(r),
            "monotonic adaptivity violated across expand() for {r:?}"
        );
    }
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

    let full_cfg = FilterConfig::new(7, 7, 32, 12, 0.08).unwrap();
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

#[test]
fn delete_rebuild_and_locate_matching_fingerprints_edge_cases() {
    let cfg = FilterConfig::new(6, 7, 32, 12, 0.08).unwrap();
    let rm = CountingReverseMap::default();
    let mut filter = AerisFilter::new(cfg.clone()).unwrap();

    // Deleting from empty filter returns Ok(None).
    assert_eq!(filter.delete(42).unwrap(), None);
    assert!(filter
        .locate_matching_fingerprints(QueryRange::new(0, 10).unwrap())
        .is_empty());

    let keys = vec![100u64, 105, 1000, 2000];
    filter = AerisFilter::rebuild_from_keys(cfg, keys.clone(), &rm).unwrap();
    assert_eq!(filter.len(), 4);
    assert_ne!(
        filter
            .locate_matching_fingerprints(QueryRange::new(100, 105).unwrap())
            .len(),
        0
    );

    assert!(filter.delete(100).unwrap().is_some());
    assert_eq!(filter.delete(100).unwrap(), None);
    assert_eq!(filter.len(), 3);
    assert!(filter.query_point(105));
    filter.verify_invariants().unwrap();
}
