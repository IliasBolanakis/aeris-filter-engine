//! Unit tests for [`super::CoupledStorageEngine`] and recovery reconciliation helpers.

use super::recovery::{compute_reverse_map_diff, reconcile_kv_and_reverse_map};
use super::*;
use aeris_types::{Fingerprint, Key, PartitionHasher, QueryRange, ReverseMapStore};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use tempfile::TempDir;

fn test_fp(key: Key, bits: u16) -> Fingerprint {
    let hasher = PartitionHasher::new(0x570A_6E01);
    Fingerprint::from_hasher(&hasher, key >> 5, bits)
}

#[test]
fn default_config_uses_group_commit_1000_and_rejects_zero_batch_size() {
    let default_cfg = StorageConfig::default();
    assert_eq!(
        default_cfg.wal_sync_policy,
        WalSyncPolicy::GroupCommit {
            batch_size: DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE
        }
    );
    assert_eq!(DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE, 1_000);
    assert!(default_cfg.wal_sync_policy.validate().is_ok());

    let strict_cfg = StorageConfig::strict_per_op_fsync();
    assert_eq!(strict_cfg.wal_sync_policy, WalSyncPolicy::PerOpFsync);
    assert!(strict_cfg.wal_sync_policy.validate().is_ok());

    let invalid_cfg =
        StorageConfig::default().with_wal_sync_policy(WalSyncPolicy::GroupCommit { batch_size: 0 });
    assert!(matches!(
        invalid_cfg.wal_sync_policy.validate(),
        Err(AerisError::InvalidConfig(_))
    ));

    let dir = TempDir::new().unwrap();
    assert!(matches!(
        CoupledStorageEngine::open_and_recover_with_config(dir.path(), invalid_cfg),
        Err(AerisError::InvalidConfig(_))
    ));
}

#[test]
fn group_commit_batch_boundary_flush_checkpoint_and_drop_sync() {
    let dir = TempDir::new().unwrap();
    let cfg =
        StorageConfig::default().with_wal_sync_policy(WalSyncPolicy::GroupCommit { batch_size: 4 });
    {
        let (engine, _) =
            CoupledStorageEngine::open_and_recover_with_config(dir.path(), cfg).unwrap();
        assert_eq!(
            engine.wal_sync_policy(),
            WalSyncPolicy::GroupCommit { batch_size: 4 }
        );
        assert_eq!(engine.pending_sync_ops(), 0);

        // Ops 1, 2, 3 accumulate pending_sync_ops; op 4 hits batch_size == 4 and syncs.
        let fp1 = test_fp(10, 22);
        let fp2 = test_fp(20, 22);
        let fp3 = test_fp(30, 22);
        let fp4 = test_fp(40, 22);

        engine.put(10, b"v10", &fp1).unwrap();
        assert_eq!(engine.pending_sync_ops(), 1);
        engine.put(20, b"v20", &fp2).unwrap();
        assert_eq!(engine.pending_sync_ops(), 2);
        engine.put(30, b"v30", &fp3).unwrap();
        assert_eq!(engine.pending_sync_ops(), 3);
        engine.put(40, b"v40", &fp4).unwrap();
        assert_eq!(engine.pending_sync_ops(), 0);

        // Delete and adapt also increment pending_sync_ops; flush_wal resets to 0.
        engine.delete(40, &fp4).unwrap();
        assert_eq!(engine.pending_sync_ops(), 1);
        let ext_fp3 = test_fp(30, 34);
        engine
            .record_adapt(
                vec![fp3.clone()],
                vec![(ext_fp3.clone(), 30)],
                QueryRange::new(100, 131).unwrap(),
            )
            .unwrap();
        assert_eq!(engine.pending_sync_ops(), 2);

        engine.flush_wal().unwrap();
        assert_eq!(engine.pending_sync_ops(), 0);

        // Checkpoint resets pending_sync_ops to 0.
        let fp5 = test_fp(50, 22);
        engine.put(50, b"v50", &fp5).unwrap();
        assert_eq!(engine.pending_sync_ops(), 1);
        engine.checkpoint().unwrap();
        assert_eq!(engine.pending_sync_ops(), 0);

        // Leave 1 pending op to exercise best-effort sync on Drop.
        let fp6 = test_fp(60, 22);
        engine.put(60, b"v60", &fp6).unwrap();
        assert_eq!(engine.pending_sync_ops(), 1);
    }

    // Reopen and confirm all operations (including post-checkpoint op synced on Drop) survived.
    let (reopened, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    assert_eq!(report.recovered_puts, 1);
    assert_eq!(report.live_key_count, 5);
    assert_eq!(
        reopened.get(10).unwrap().as_deref(),
        Some(b"v10".as_slice())
    );
    assert_eq!(reopened.get(40).unwrap(), None);
    assert_eq!(
        reopened.get(50).unwrap().as_deref(),
        Some(b"v50".as_slice())
    );
    assert_eq!(
        reopened.get(60).unwrap().as_deref(),
        Some(b"v60".as_slice())
    );
    assert_eq!(
        reopened.reverse_map().get_keys(&test_fp(30, 34)).unwrap(),
        vec![30]
    );
}

#[test]
fn put_batch_persists_atomically_with_single_sync_and_recovers_from_wal() {
    let dir = TempDir::new().unwrap();
    let fps: Vec<Fingerprint> = (100..150u64).map(|k| test_fp(k, 22)).collect();
    let vals: Vec<Vec<u8>> = (100..150u64)
        .map(|k| format!("batch-val-{k}").into_bytes())
        .collect();

    {
        let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();

        // Empty batch returns current last_lsn without advancing LSN.
        assert_eq!(engine.put_batch(&[]).unwrap(), 0);
        assert_eq!(engine.last_lsn(), 0);

        // Seed one unsynced put so we verify put_batch resets pending_sync_ops to 0.
        let seed_fp = test_fp(1, 22);
        let lsn1 = engine.put(1, b"seed", &seed_fp).unwrap();
        assert_eq!(lsn1, 1);
        assert_eq!(engine.pending_sync_ops(), 1);

        let entries: Vec<(Key, &[u8], &Fingerprint)> = (0..50usize)
            .map(|i| (100 + i as u64, vals[i].as_slice(), &fps[i]))
            .collect();
        let highest_lsn = engine.put_batch(&entries).unwrap();
        assert_eq!(highest_lsn, 51);
        assert_eq!(engine.last_lsn(), 51);
        assert_eq!(engine.pending_sync_ops(), 0);

        // Empty batch after writes returns current last_lsn.
        assert_eq!(engine.put_batch(&[]).unwrap(), 51);

        for (i, &(k, v, fp)) in entries.iter().enumerate() {
            assert_eq!(engine.get(k).unwrap().as_deref(), Some(v));
            assert!(engine.reverse_map().get_keys(fp).unwrap().contains(&k));
            let _ = i;
        }
    }

    // Wipe both RocksDB directories and prove full WAL replay reconstructs every batched key!
    fs::remove_dir_all(dir.path().join(KV_DB_DIR_NAME)).unwrap();
    fs::remove_dir_all(dir.path().join(REVERSE_MAP_DB_DIR_NAME)).unwrap();

    let (reopened, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    assert_eq!(report.recovered_puts, 51);
    assert_eq!(report.live_key_count, 51);
    assert_eq!(
        reopened.get(1).unwrap().as_deref(),
        Some(b"seed".as_slice())
    );
    for i in 0..50usize {
        let k = 100 + i as u64;
        assert_eq!(
            reopened.get(k).unwrap().as_deref(),
            Some(vals[i].as_slice())
        );
        assert!(reopened
            .reverse_map()
            .get_keys(&fps[i])
            .unwrap()
            .contains(&k));
    }
}

#[test]
fn strict_per_op_fsync_keeps_pending_sync_ops_at_zero() {
    let dir = TempDir::new().unwrap();
    let (engine, _) = CoupledStorageEngine::open_and_recover_with_config(
        dir.path(),
        StorageConfig::strict_per_op_fsync(),
    )
    .unwrap();
    assert_eq!(engine.wal_sync_policy(), WalSyncPolicy::PerOpFsync);

    let fp1 = test_fp(7, 22);
    let fp2 = test_fp(7, 34);
    engine.put(7, b"strict", &fp1).unwrap();
    assert_eq!(engine.pending_sync_ops(), 0);
    engine
        .record_adapt(
            vec![fp1.clone()],
            vec![(fp2.clone(), 7)],
            QueryRange::point(8),
        )
        .unwrap();
    assert_eq!(engine.pending_sync_ops(), 0);
    engine.delete(7, &fp2).unwrap();
    assert_eq!(engine.pending_sync_ops(), 0);
}

#[test]
fn reconciliation_helpers_compute_minimal_diff_and_heal_orphans() {
    let fp_a = test_fp(10, 22);
    let fp_b = test_fp(100, 22);
    assert_ne!(fp_a, fp_b);

    let mut existing = BTreeMap::new();
    existing.insert(fp_a.clone(), BTreeSet::from([10, 999]));
    let mut desired = BTreeMap::new();
    desired.insert(fp_a.clone(), BTreeSet::from([10, 11]));
    desired.insert(fp_b.clone(), BTreeSet::from([100]));

    let (tombstone, merge) = compute_reverse_map_diff(&existing, &desired);
    assert_eq!(tombstone, vec![fp_a.clone()]);
    assert_eq!(
        merge,
        vec![(fp_a.clone(), 10), (fp_a.clone(), 11), (fp_b.clone(), 100)]
    );

    // Inject an orphan key into Reverse Map without KV backing and verify reconciliation removes it.
    let dir = TempDir::new().unwrap();
    let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
    engine.put(10, b"v10", &fp_a).unwrap();
    engine.flush_wal().unwrap();
    engine.reverse_map().append_key(&fp_b, 999).unwrap();

    assert!(
        reconcile_kv_and_reverse_map(engine.wal(), engine.kv_store(), engine.reverse_map())
            .unwrap()
    );
    assert_eq!(engine.reverse_map().scan_all_keys().unwrap(), vec![10]);
}
