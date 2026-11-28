//! Unit tests for [`super::RocksDbReverseMap`] and streaming iteration.

use super::*;
use aeris_types::{PartitionHasher, ReverseMapStore};
use tempfile::TempDir;

fn make_fp(seed: u64, partition: u64, bits: u16) -> Fingerprint {
    let hasher = PartitionHasher::new(seed);
    Fingerprint::from_hasher(&hasher, partition, bits)
}

#[test]
fn append_remove_idempotency_and_reopen() {
    let dir = TempDir::new().unwrap();
    let fp1 = make_fp(7, 1, 22);
    let fp2 = make_fp(7, 2, 22);
    let fp_long = make_fp(7, 3, 160);

    {
        let rm = RocksDbReverseMap::open(dir.path(), ReverseMapConfig::default()).unwrap();
        assert_eq!(rm.entry_count_estimate(), 0);

        // Out-of-place appends including duplicates (simulating WAL replay idempotency)
        rm.append_key(&fp1, 300).unwrap();
        rm.append_key(&fp1, 100).unwrap();
        rm.append_key(&fp1, 200).unwrap();
        rm.append_key(&fp1, 100).unwrap();

        rm.append_batch(&[
            (fp2.clone(), 500),
            (fp_long.clone(), 900),
            (fp2.clone(), 400),
        ])
        .unwrap();

        // Flush to force partial_merge + L0 SSTable creation, then apply more merges
        rm.flush().unwrap();
        rm.remove_key(&fp1, 200).unwrap();
        rm.remove_key(&fp1, 999_999).unwrap(); // absent key remove is a no-op
        rm.append_key(&fp1, 250).unwrap();

        assert_eq!(rm.get_keys(&fp1).unwrap(), vec![100, 250, 300]);
        assert_eq!(rm.get_keys(&fp2).unwrap(), vec![400, 500]);
        assert_eq!(rm.get_keys(&fp_long).unwrap(), vec![900]);
        assert_eq!(
            rm.scan_all_keys().unwrap(),
            vec![100, 250, 300, 400, 500, 900]
        );
        rm.compact().unwrap();
        assert_eq!(rm.entry_count_estimate(), 6);
    }

    // Reopen existing DB and verify accurate entry count initialization & data persistence
    let rm = RocksDbReverseMap::open(dir.path(), ReverseMapConfig::default()).unwrap();
    assert_eq!(rm.entry_count_estimate(), 6);
    assert_eq!(rm.get_keys(&fp1).unwrap(), vec![100, 250, 300]);
    assert_eq!(rm.get_keys(&fp2).unwrap(), vec![400, 500]);
    assert_eq!(rm.get_keys(&fp_long).unwrap(), vec![900]);
}

#[test]
fn tombstone_replace_and_streaming_iterator_deduplicate() {
    let dir = TempDir::new().unwrap();
    let rm = RocksDbReverseMap::open(dir.path(), ReverseMapConfig::default()).unwrap();

    let fp_old = make_fp(11, 10, 22);
    let fp_ext_a = make_fp(11, 10, 34);
    let fp_ext_b = make_fp(11, 11, 34);

    rm.append_batch(&[
        (fp_old.clone(), 10),
        (fp_old.clone(), 20),
        (fp_old.clone(), 30),
    ])
    .unwrap();
    rm.flush().unwrap();

    // Replace old_fp atomically with extended fingerprints (plus intentional duplicate key 20 across boxes)
    rm.replace_fingerprint(
        &fp_old,
        &[
            (fp_ext_a.clone(), 10),
            (fp_ext_a.clone(), 20),
            (fp_ext_b.clone(), 20),
            (fp_ext_b.clone(), 30),
        ],
    )
    .unwrap();

    assert!(rm.get_keys(&fp_old).unwrap().is_empty());
    assert_eq!(rm.get_keys(&fp_ext_a).unwrap(), vec![10, 20]);
    assert_eq!(rm.get_keys(&fp_ext_b).unwrap(), vec![20, 30]);

    // Streaming iterator yields deduplicated live keys without materializing upfront
    let mut streamed: Vec<Key> = rm
        .iter_all_keys()
        .unwrap()
        .collect::<AerisResult<Vec<_>>>()
        .unwrap();
    streamed.sort_unstable();
    assert_eq!(streamed, vec![10, 20, 30]);

    // Tombstone fp_ext_a, compact, verify physical reclamation and reuse of fp_ext_a
    rm.tombstone(&fp_ext_a).unwrap();
    rm.compact().unwrap();
    assert!(rm.get_keys(&fp_ext_a).unwrap().is_empty());
    assert_eq!(rm.scan_all_keys().unwrap(), vec![20, 30]);
    assert_eq!(rm.entry_count_estimate(), 2);

    // Appending after tombstone must NOT resurrect pre-tombstone keys
    rm.append_key(&fp_ext_a, 777).unwrap();
    assert_eq!(rm.get_keys(&fp_ext_a).unwrap(), vec![777]);

    // Removing all keys leaves an empty CompactedSnapshot with 0 live keys across reopen
    rm.remove_key(&fp_ext_a, 777).unwrap();
    rm.remove_key(&fp_ext_b, 20).unwrap();
    rm.remove_key(&fp_ext_b, 30).unwrap();
    rm.flush().unwrap();
    rm.compact().unwrap();
    assert!(rm.get_keys(&fp_ext_a).unwrap().is_empty());
    assert!(rm.scan_all_keys().unwrap().is_empty());
    assert_eq!(rm.entry_count_estimate(), 0);
    drop(rm);

    let reopened = RocksDbReverseMap::open(dir.path(), ReverseMapConfig::default()).unwrap();
    assert_eq!(reopened.entry_count_estimate(), 0);
    assert!(reopened.scan_all_keys().unwrap().is_empty());
}

#[test]
fn config_builders_and_empty_batch_are_safe() {
    let coupled = ReverseMapConfig::for_wal_coupled();
    assert!(!coupled.sync_writes);
    assert!(coupled.enable_lz4_compression);

    let custom = ReverseMapConfig::standalone_durable()
        .with_write_buffer_size(8 * 1024 * 1024)
        .with_sync_writes(false)
        .with_lz4_compression(false);
    assert_eq!(custom.write_buffer_size, 8 * 1024 * 1024);
    assert!(!custom.sync_writes);
    assert!(!custom.enable_lz4_compression);

    let dir = TempDir::new().unwrap();
    let rm = RocksDbReverseMap::open(dir.path(), custom).unwrap();
    rm.replace_fingerprints_batch(&[], &[]).unwrap();
    rm.append_batch(&[]).unwrap();
    assert_eq!(rm.refresh_entry_count().unwrap(), 0);
    assert_eq!(rm.path(), dir.path());
}
