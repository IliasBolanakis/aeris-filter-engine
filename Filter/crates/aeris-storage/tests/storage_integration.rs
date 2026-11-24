//! Module, integration, reconciliation, and concurrency tests for `aeris-storage`.

use aeris_storage::{
    CoupledStorageEngine, CoupledWal, ReverseMapConfig, RocksDbReverseMap, StorageConfig,
    KV_DB_DIR_NAME, REVERSE_MAP_DB_DIR_NAME, WAL_FILE_NAME,
};
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, ExpansionReport, FilterConfig, Fingerprint, Key,
    PartitionHasher, QueryRange, RangeFilter, ReverseMapStore, WalEngine,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

/// Minimal contract-compliant `RangeFilter` used strictly to test
/// `CoupledStorageEngine::rebuild_filter_and_reconcile` against the locked M0 trait
/// without coupling to any concrete Data Plane internals.
struct ContractTestFilter {
    config: FilterConfig,
    hasher: PartitionHasher,
    /// Maps each live key to its current canonical fingerprint (mirroring InfiniFilter
    /// expansion invariance and selective rejuvenation).
    keys: BTreeMap<Key, Fingerprint>,
    expansions: u64,
}

impl ContractTestFilter {
    fn new(config: FilterConfig) -> Self {
        let hasher = PartitionHasher::new(config.hash_seed);
        Self {
            config,
            hasher,
            keys: BTreeMap::new(),
            expansions: 0,
        }
    }

    fn fresh_fp(&self, key: Key) -> Fingerprint {
        let part = self.config.partition_of(key);
        let bit_len = (self.config.q as u16) + (self.config.r as u16);
        Fingerprint::from_hasher(&self.hasher, part, bit_len)
    }

    fn current_fp(&self, key: Key) -> Option<&Fingerprint> {
        self.keys.get(&key)
    }
}

impl RangeFilter for ContractTestFilter {
    fn config(&self) -> &FilterConfig {
        &self.config
    }

    fn insert(&mut self, key: Key) -> AerisResult<Fingerprint> {
        let next_load = (self.keys.len() + 1) as f64 / (self.config.num_slots() as f64);
        if next_load > self.config.max_load_factor {
            return Err(AerisError::FilterFull {
                load_factor: self.load_factor(),
                capacity: self.capacity(),
            });
        }
        let fp = self.fresh_fp(key);
        self.keys.insert(key, fp.clone());
        Ok(fp)
    }

    fn delete(&mut self, key: Key) -> AerisResult<Option<Fingerprint>> {
        Ok(self.keys.remove(&key))
    }

    fn query_point(&self, key: Key) -> bool {
        self.keys.contains_key(&key)
    }

    fn query_range(&self, range: QueryRange) -> bool {
        self.keys
            .range(range.start()..=range.end())
            .next()
            .is_some()
    }

    fn locate_matching_fingerprints(&self, range: QueryRange) -> Vec<Fingerprint> {
        self.keys
            .range(range.start()..=range.end())
            .map(|(_, fp)| fp.clone())
            .collect()
    }

    fn adapt(
        &mut self,
        _range: QueryRange,
        _reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<AdaptationReport> {
        Ok(AdaptationReport::default())
    }

    fn expand(&mut self, reverse_map: &dyn ReverseMapStore) -> AerisResult<ExpansionReport> {
        let old_q = self.config.q;
        self.config = self.config.expanded()?;
        self.expansions += 1;

        // Simulate selective rejuvenation of the oldest fingerprint box via ReverseMapStore,
        // while non-rejuvenated boxes retain invariant canonical fingerprints (InfiniFilter).
        let mut boxes_rejuvenated = 0usize;
        let mut keys_rejuvenated = 0usize;
        if let Some((&oldest_key, oldest_fp)) = self.keys.iter().next() {
            let old_fp = oldest_fp.clone();
            let rm_keys = reverse_map.get_keys(&old_fp)?;
            assert!(
                rm_keys.contains(&oldest_key),
                "expand() rejuvenation expected reverse_map to hold key {oldest_key}"
            );
            let mut new_entries = Vec::with_capacity(rm_keys.len());
            for k in rm_keys {
                let new_fp = self.fresh_fp(k);
                self.keys.insert(k, new_fp.clone());
                new_entries.push((new_fp, k));
                keys_rejuvenated += 1;
            }
            reverse_map.replace_fingerprint(&old_fp, &new_entries)?;
            boxes_rejuvenated = 1;
        }

        Ok(ExpansionReport {
            old_q,
            new_q: self.config.q,
            boxes_transferred: self.keys.len(),
            boxes_rejuvenated,
            keys_rejuvenated,
            duration_ns: 0,
        })
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn capacity(&self) -> usize {
        self.config.num_slots()
    }

    fn load_factor(&self) -> f64 {
        (self.keys.len() as f64) / (self.config.num_slots() as f64)
    }

    fn expansion_cycles(&self) -> u64 {
        self.expansions
    }
}

fn fp_for(seed: u64, key: Key, bits: u16) -> Fingerprint {
    let hasher = PartitionHasher::new(seed);
    Fingerprint::from_hasher(&hasher, key >> 5, bits)
}

#[test]
fn coupled_engine_put_delete_adapt_checkpoint_and_reopen() {
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xA3E1_5001);

    let mut expected_kv: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
    let mut expected_rm: BTreeMap<Fingerprint, BTreeSet<Key>> = BTreeMap::new();

    let (checkpoint_lsn, pre_reopen_lsn);
    {
        let (engine, initial_report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        assert_eq!(initial_report.replayed_records, 0);
        assert_eq!(initial_report.live_key_count, 0);

        // Phase 1: Insert 120 statistical keys before checkpoint
        for i in 0..120u64 {
            let key: Key = rng.gen::<u64>() & 0x000F_FFFF_FFFF;
            let val = format!("v1-{i}-{key}").into_bytes();
            let fp = fp_for(7, key, 22);
            engine.put(key, &val, &fp).unwrap();
            expected_kv.insert(key, val);
            expected_rm.entry(fp).or_default().insert(key);
        }

        // Checkpoint flushes SSTables and persists durable WalOp::Checkpoint
        checkpoint_lsn = engine.checkpoint().unwrap();
        assert_eq!(checkpoint_lsn, 120);

        // Phase 2: Post-checkpoint puts (20), deletes (15), and adapt (1)
        let existing_keys: Vec<Key> = expected_kv.keys().copied().collect();
        for &del_key in existing_keys.iter().take(15) {
            let fp = fp_for(7, del_key, 22);
            engine.delete(del_key, &fp).unwrap();
            expected_kv.remove(&del_key);
            if let Some(set) = expected_rm.get_mut(&fp) {
                set.remove(&del_key);
                if set.is_empty() {
                    expected_rm.remove(&fp);
                }
            }
        }

        for i in 0..20u64 {
            let key: Key = (rng.gen::<u64>() & 0x000F_FFFF_FFFF) | 0x0010_0000_0000;
            let val = format!("v2-{i}-{key}").into_bytes();
            let fp = fp_for(7, key, 22);
            engine.put(key, &val, &fp).unwrap();
            expected_kv.insert(key, val);
            expected_rm.entry(fp).or_default().insert(key);
        }

        // Adapt one live fingerprint to an extended 34-bit fingerprint
        let (adapt_old_fp, adapt_keys) = expected_rm.iter().next().unwrap();
        let adapt_old_fp = adapt_old_fp.clone();
        let adapt_keys: Vec<Key> = adapt_keys.iter().copied().collect();
        let mut new_entries = Vec::new();
        for &k in &adapt_keys {
            let ext_fp = fp_for(7, k, 34);
            new_entries.push((ext_fp, k));
        }
        engine
            .record_adapt(
                vec![adapt_old_fp.clone()],
                new_entries.clone(),
                QueryRange::new(10, 50).unwrap(),
            )
            .unwrap();
        expected_rm.remove(&adapt_old_fp);
        for (ext_fp, k) in new_entries {
            expected_rm.entry(ext_fp).or_default().insert(k);
        }

        pre_reopen_lsn = engine.last_lsn();
        assert_eq!(pre_reopen_lsn, checkpoint_lsn + 1 + 15 + 20 + 1);
    }

    // Reopen and verify exact recovery report & 100% key parity
    let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    assert_eq!(report.recovered_puts, 20);
    assert_eq!(report.recovered_deletes, 15);
    assert_eq!(report.recovered_adapts, 1);
    assert_eq!(report.replayed_records, 36);
    assert_eq!(report.truncated_tail_bytes, 0);
    assert_eq!(report.last_lsn, pre_reopen_lsn);
    assert_eq!(report.live_key_count, expected_kv.len());

    for (&key, val) in &expected_kv {
        assert_eq!(engine.get(key).unwrap().as_deref(), Some(val.as_slice()));
        assert!(engine.range_non_empty(QueryRange::point(key)).unwrap());
    }
    for (fp, keys) in &expected_rm {
        let stored = engine.reverse_map().get_keys(fp).unwrap();
        let expected_vec: Vec<Key> = keys.iter().copied().collect();
        assert_eq!(stored, expected_vec);
    }
    assert_eq!(
        engine.kv_store().scan_all_keys().unwrap(),
        engine.reverse_map().scan_all_keys().unwrap()
    );
}

#[test]
fn wal_replay_idempotency_across_repeated_recoveries() {
    let dir = TempDir::new().unwrap();
    let fp_a = fp_for(13, 1000, 22);
    let fp_b = fp_for(13, 1000, 34);

    {
        let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        engine.put(1000, b"first", &fp_a).unwrap();
        engine.put(1000, b"second", &fp_a).unwrap();
        engine.put(1001, b"keep", &fp_a).unwrap();
        engine.put(1002, b"gone", &fp_a).unwrap();
        engine.delete(1002, &fp_a).unwrap();
        engine
            .record_adapt(
                vec![fp_a.clone()],
                vec![(fp_b.clone(), 1000), (fp_b.clone(), 1001)],
                QueryRange::new(500, 900).unwrap(),
            )
            .unwrap();
        // Deliberately NO checkpoint so every open replays the full WAL from LSN 1
    }

    // Re-run recovery 3 times without checkpointing, including once after wiping both RocksDB dirs
    for pass in 0..3 {
        if pass == 2 {
            fs::remove_dir_all(dir.path().join(KV_DB_DIR_NAME)).unwrap();
            fs::remove_dir_all(dir.path().join(REVERSE_MAP_DB_DIR_NAME)).unwrap();
        }
        let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        assert_eq!(report.recovered_puts, 4);
        assert_eq!(report.recovered_deletes, 1);
        assert_eq!(report.recovered_adapts, 1);
        assert_eq!(report.live_key_count, 2);

        assert_eq!(
            engine.get(1000).unwrap().as_deref(),
            Some(b"second".as_slice())
        );
        assert_eq!(
            engine.get(1001).unwrap().as_deref(),
            Some(b"keep".as_slice())
        );
        assert_eq!(engine.get(1002).unwrap(), None);
        assert!(engine.reverse_map().get_keys(&fp_a).unwrap().is_empty());
        assert_eq!(
            engine.reverse_map().get_keys(&fp_b).unwrap(),
            vec![1000, 1001]
        );
    }
}

#[test]
fn checkpoint_and_compact_wal_preserves_state_and_lsns() {
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xC049_AC71);
    let mut all_keys = BTreeSet::new();

    let (compacted_lsn, final_lsn);
    {
        let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        for _ in 0..300 {
            let k: Key = rng.gen();
            let fp = fp_for(21, k, 22);
            engine.put(k, &k.to_le_bytes(), &fp).unwrap();
            all_keys.insert(k);
        }

        let wal_size_before = engine.wal().file_size_bytes();
        let (cp_lsn, reclaimed) = engine.checkpoint_and_compact_wal().unwrap();
        compacted_lsn = cp_lsn;
        assert!(reclaimed > 0);
        assert!(engine.wal().file_size_bytes() < wal_size_before / 5);

        for _ in 0..40 {
            let k: Key = rng.gen();
            let fp = fp_for(21, k, 22);
            engine.put(k, &k.to_le_bytes(), &fp).unwrap();
            all_keys.insert(k);
        }
        final_lsn = engine.last_lsn();
        assert_eq!(final_lsn, compacted_lsn + 1 + 40);
    }

    let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    assert_eq!(report.replayed_records, 40);
    assert_eq!(report.recovered_puts, 40);
    assert_eq!(report.last_lsn, final_lsn);
    assert_eq!(report.live_key_count, all_keys.len());
    assert_eq!(
        engine.kv_store().scan_all_keys().unwrap(),
        all_keys.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        engine.reverse_map().scan_all_keys().unwrap(),
        all_keys.iter().copied().collect::<Vec<_>>()
    );
}

#[test]
fn reconcile_recovers_missing_reverse_map_and_removes_orphans() {
    let dir = TempDir::new().unwrap();
    let fp1 = fp_for(31, 100, 22);
    let fp2 = fp_for(31, 200, 22);
    let orphan_fp = fp_for(31, 9999, 22);

    {
        let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        engine.put(100, b"a", &fp1).unwrap();
        engine.put(200, b"b", &fp2).unwrap();
        engine.checkpoint().unwrap();

        // Inject an orphan key directly into Reverse Map that does not exist in KV store
        engine
            .reverse_map()
            .append_key(&orphan_fp, 999_999)
            .unwrap();
        engine.reverse_map().flush().unwrap();
    }

    // Recovery detects and removes the orphan key
    {
        let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        assert_eq!(report.live_key_count, 2);
        assert_eq!(
            engine.reverse_map().scan_all_keys().unwrap(),
            vec![100, 200]
        );
        assert!(engine
            .reverse_map()
            .get_keys(&orphan_fp)
            .unwrap()
            .is_empty());
    }

    // Wipe reverse_map_db completely while WAL history is still intact
    fs::remove_dir_all(dir.path().join(REVERSE_MAP_DB_DIR_NAME)).unwrap();
    let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    assert_eq!(report.live_key_count, 2);
    assert_eq!(
        engine.reverse_map().scan_all_keys().unwrap(),
        vec![100, 200]
    );
    assert_eq!(engine.reverse_map().get_keys(&fp1).unwrap(), vec![100]);
    assert_eq!(engine.reverse_map().get_keys(&fp2).unwrap(), vec![200]);
}

#[test]
fn rebuild_filter_and_reconcile_zero_false_negatives_and_auto_expand() {
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xF117_E200);
    let cfg = FilterConfig::with_minimum_bits(6, 32, 1.0 / 256.0)
        .unwrap()
        .with_hash_seed(77);

    let mut live_keys = BTreeSet::new();
    {
        let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        let hasher = PartitionHasher::new(cfg.hash_seed);
        let bit_len = (cfg.q as u16) + (cfg.r as u16);

        for _ in 0..200 {
            let k: Key = rng.gen::<u64>() & 0x0000_FFFF_FFFF;
            let fp = Fingerprint::from_hasher(&hasher, cfg.partition_of(k), bit_len);
            engine.put(k, &k.to_be_bytes(), &fp).unwrap();
            live_keys.insert(k);
        }

        // Simulate prior adaptation leaving a longer extended fingerprint in Reverse Map
        let sample_key = *live_keys.iter().next().unwrap();
        let old_fp = Fingerprint::from_hasher(&hasher, cfg.partition_of(sample_key), bit_len);
        let ext_fp = Fingerprint::from_hasher(&hasher, cfg.partition_of(sample_key), bit_len + 16);
        engine
            .record_adapt(
                vec![old_fp],
                vec![(ext_fp, sample_key)],
                QueryRange::point(sample_key.saturating_add(1)),
            )
            .unwrap();
        engine.checkpoint().unwrap();
    }

    let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    // Fresh small filter (capacity 64 at q=6, load cap 0.85 -> 54 keys before expand)
    let mut rebuilt = ContractTestFilter::new(cfg.clone());
    let loaded = engine.rebuild_filter_and_reconcile(&mut rebuilt).unwrap();
    assert_eq!(loaded, live_keys.len());
    assert!(
        rebuilt.expansion_cycles() >= 2,
        "expected >= 2 automatic expansions for {} keys starting at q=6, got {}",
        live_keys.len(),
        rebuilt.expansion_cycles()
    );

    // Zero false negatives on every live key AND exact Reverse Map alignment
    for &k in &live_keys {
        assert!(rebuilt.query_point(k), "false negative for key {k}");
        assert!(rebuilt.query_range(QueryRange::point(k)));
        let expected_fp = rebuilt.current_fp(k).expect("key missing from filter map");
        let rm_keys = engine.reverse_map().get_keys(expected_fp).unwrap();
        assert!(
            rm_keys.contains(&k),
            "reconciled Reverse Map missing key {k} under rebuilt fingerprint"
        );
    }
    assert_eq!(
        engine.reverse_map().scan_all_keys().unwrap(),
        live_keys.iter().copied().collect::<Vec<_>>()
    );

    // Also verify the non-expanding rebuild path (both reconciliation diff and no-op steady state)
    let large_cfg = FilterConfig::with_minimum_bits(10, 32, 1.0 / 256.0)
        .unwrap()
        .with_hash_seed(77);
    let mut non_expanding = ContractTestFilter::new(large_cfg.clone());
    assert_eq!(
        engine
            .rebuild_filter_and_reconcile(&mut non_expanding)
            .unwrap(),
        live_keys.len()
    );
    assert_eq!(non_expanding.expansion_cycles(), 0);
    for &k in &live_keys {
        assert!(non_expanding.query_point(k));
        let fp = non_expanding.current_fp(k).unwrap();
        assert!(engine.reverse_map().get_keys(fp).unwrap().contains(&k));
    }

    // Second rebuild with identical config exercises the zero-diff no-op path
    let mut steady_filter = ContractTestFilter::new(large_cfg);
    assert_eq!(
        engine
            .rebuild_filter_and_reconcile(&mut steady_filter)
            .unwrap(),
        live_keys.len()
    );
    assert_eq!(steady_filter.expansion_cycles(), 0);
}

#[test]
fn concurrent_writers_maintain_linearizable_wal_and_storage_consistency() {
    let dir = TempDir::new().unwrap();
    let (engine, _) =
        CoupledStorageEngine::open_and_recover_with_config(dir.path(), StorageConfig::default())
            .unwrap();
    let engine = Arc::new(engine);

    const WRITERS: usize = 6;
    const OPS_PER_WRITER: u64 = 40;
    let barrier = Arc::new(Barrier::new(WRITERS + 2));

    let mut handles = Vec::new();
    for tid in 0..WRITERS {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(0x5555_0000 + tid as u64);
            bar.wait();
            for step in 0..OPS_PER_WRITER {
                // Deliberately overlap key space across threads to test linearizability
                let key = (rng.gen::<u64>() % 50) + 1;
                let fp = fp_for(99, key, 22);
                if step % 4 == 3 {
                    eng.delete(key, &fp).unwrap();
                } else {
                    let val = format!("t{tid}-s{step}").into_bytes();
                    eng.put(key, &val, &fp).unwrap();
                }
            }
        }));
    }

    // Concurrent readers querying KV store and Reverse Map while writers contend
    for _ in 0..2 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            for probe in 1..=50u64 {
                let _ = eng.get(probe).unwrap();
                let _ = eng
                    .range_non_empty(QueryRange::new(probe, probe + 5).unwrap())
                    .unwrap();
                let fp = fp_for(99, probe, 22);
                let _ = eng.reverse_map().get_keys(&fp).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let live_kv_keys = engine.kv_store().scan_all_keys().unwrap();
    let live_rm_keys = engine.reverse_map().scan_all_keys().unwrap();
    assert_eq!(live_kv_keys, live_rm_keys);

    // Copy ONLY wal.log into a brand-new directory and recover from scratch;
    // recovered state MUST match live state bit-for-bit (proving WAL linearizability!).
    let replay_dir = TempDir::new().unwrap();
    fs::copy(
        dir.path().join(WAL_FILE_NAME),
        replay_dir.path().join(WAL_FILE_NAME),
    )
    .unwrap();

    // Verify WAL monotonicity directly
    let wal = CoupledWal::open(replay_dir.path().join(WAL_FILE_NAME)).unwrap();
    let records = wal.replay_from(1).unwrap();
    assert_eq!(records.len(), WRITERS * (OPS_PER_WRITER as usize));
    for (i, rec) in records.iter().enumerate() {
        assert_eq!(rec.lsn, (i as u64) + 1);
    }
    drop(wal);

    let (replayed, report) = CoupledStorageEngine::open_and_recover(replay_dir.path()).unwrap();
    assert_eq!(report.replayed_records, records.len());
    assert_eq!(replayed.kv_store().scan_all_keys().unwrap(), live_kv_keys);
    assert_eq!(
        replayed.reverse_map().scan_all_keys().unwrap(),
        live_rm_keys
    );
    for &k in &live_kv_keys {
        assert_eq!(replayed.get(k).unwrap(), engine.get(k).unwrap());
    }
}

#[test]
fn reverse_map_multi_flush_compaction_and_streaming_scale() {
    let dir = TempDir::new().unwrap();
    let rm = RocksDbReverseMap::open(
        dir.path(),
        ReverseMapConfig::default().with_write_buffer_size(64 * 1024),
    )
    .unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(0xD00D_F00D);
    let mut oracle: BTreeMap<Fingerprint, BTreeSet<Key>> = BTreeMap::new();
    let fps: Vec<Fingerprint> = (0..120u64).map(|p| fp_for(5, p, 24)).collect();

    for round in 0..4 {
        for _ in 0..500 {
            let fp = &fps[rng.gen_range(0..fps.len())];
            let key: Key = rng.gen_range(0..10_000);
            if rng.gen_bool(0.75) {
                rm.append_key(fp, key).unwrap();
                oracle.entry(fp.clone()).or_default().insert(key);
            } else {
                rm.remove_key(fp, key).unwrap();
                if let Some(set) = oracle.get_mut(fp) {
                    set.remove(&key);
                }
            }
        }
        // Tombstone 5 fingerprints per round
        for fp in &fps[(round * 5)..((round + 1) * 5)] {
            rm.tombstone(fp).unwrap();
            oracle.remove(fp);
        }
        rm.flush().unwrap();
    }
    rm.compact().unwrap();

    let mut expected_all_keys = BTreeSet::new();
    for fp in &fps {
        let expected: Vec<Key> = oracle
            .get(fp)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        assert_eq!(rm.get_keys(fp).unwrap(), expected);
        expected_all_keys.extend(expected);
    }

    let scanned = rm.scan_all_keys().unwrap();
    assert_eq!(
        scanned,
        expected_all_keys.iter().copied().collect::<Vec<_>>()
    );

    let mut streamed: Vec<Key> = rm
        .iter_all_keys()
        .unwrap()
        .collect::<AerisResult<Vec<_>>>()
        .unwrap();
    streamed.sort_unstable();
    assert_eq!(streamed, scanned);
}
