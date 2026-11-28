mod adapt_expand;

use super::radix::radix_sort_u64;
use super::*;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// In-memory `ReverseMapStore` mock tracking `append_batch` write counts
/// for verifying `bulk_load` atomicity and full `adapt`/`expand` workflows.
#[derive(Default)]
pub(super) struct CountingReverseMap {
    pub(super) map: Mutex<BTreeMap<Fingerprint, Vec<Key>>>,
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
    let mut empty: Vec<u64> = Vec::new();
    radix_sort_u64(&mut empty, 64);
    assert!(empty.is_empty());

    let mut single = vec![0xDEAD_BEEF_CAFE_BABEu64];
    radix_sort_u64(&mut single, 64);
    assert_eq!(single, vec![0xDEAD_BEEF_CAFE_BABEu64]);

    let mut identical = vec![0x0123_4567_89AB_CDEFu64; 512];
    radix_sort_u64(&mut identical, 64);
    assert!(identical.iter().all(|&x| x == 0x0123_4567_89AB_CDEF));

    let mut sorted: Vec<u64> = (0..1000).collect();
    radix_sort_u64(&mut sorted, 16);
    assert_eq!(sorted, (0..1000).collect::<Vec<u64>>());

    let mut rev: Vec<u64> = (0..2048).rev().collect();
    radix_sort_u64(&mut rev, 24);
    assert_eq!(rev, (0..2048).collect::<Vec<u64>>());

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

        let mut keys: Vec<Key> = Vec::with_capacity(600);
        for _ in 0..450 {
            keys.push(rng.next_u64() & 0x000F_FFFF);
        }
        let hot_part_base = 42u64 << m;
        for offset in 0..16u64 {
            let mem = offset & cfg.memento_mask();
            keys.push(hot_part_base | mem);
        }
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
    let cfg = FilterConfig::new(6, 8, 32, 13, 0.05).unwrap();
    let keys: Vec<Key> = (0..120u64).map(|i| i << 5).collect();
    let rm = CountingReverseMap::default();
    let err = AerisFilter::bulk_load(cfg, &keys, &rm).unwrap_err();
    assert!(
        matches!(err, AerisError::FilterFull { .. }),
        "expected FilterFull, got {err:?}"
    );
    assert_eq!(rm.batch_calls(), 0);
    assert_eq!(rm.appended_entries(), 0);

    let wide_cfg = FilterConfig::new(6, 55, 32, 60, 0.05).unwrap();
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
    let cfg = FilterConfig::new(7, 7, 32, 12, 0.08).unwrap();
    let rm = CountingReverseMap::default();
    let mut filter = AerisFilter::new(cfg).unwrap();

    let mut inserted = Vec::new();
    for i in 0..45u64 {
        let k = i.wrapping_mul(0x9E37_79B9) & 0x00FF_FFFF;
        let fp = filter.insert(k).unwrap();
        rm.append_key(&fp, k).unwrap();
        inserted.push(k);
    }
    let p_a = 100u64 << 5;
    for m_val in [20u64, 5, 20, 31, 0] {
        let k = p_a | m_val;
        let fp = filter.insert(k).unwrap();
        rm.append_key(&fp, k).unwrap();
        inserted.push(k);
    }
    filter.verify_invariants().unwrap();

    filter.expand(&rm).unwrap();
    filter.verify_invariants().unwrap();
    for i in 0..40u64 {
        let k = (i + 500).wrapping_mul(0x517C_C1B7) & 0x00FF_FFFF;
        let fp = filter.insert(k).unwrap();
        rm.append_key(&fp, k).unwrap();
        inserted.push(k);
    }
    filter.verify_invariants().unwrap();

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

    let wide_cfg = FilterConfig::new(8, 55, 32, 60, 0.05).unwrap();
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
