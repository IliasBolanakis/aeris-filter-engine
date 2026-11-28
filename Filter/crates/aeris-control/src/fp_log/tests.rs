//! Unit tests for [`super::FalsePositiveLog`], frame codec, and recovery helpers.

use super::recovery::compaction_tmp_path;
use super::*;
use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
use aeris_types::FilterConfig;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use tempfile::TempDir;

#[test]
fn frame_codec_round_trips_and_detects_corruption() {
    let entry = FpLogEntry {
        seq: 42,
        range: QueryRange::new(100, 255).unwrap(),
        timestamp_ms: 1_700_000_000_123,
    };
    let frame = encode_fp_frame(&entry);
    assert_eq!(frame.len(), FP_LOG_FRAME_LEN);
    assert_eq!(&frame[0..4], b"AFPL");
    assert_eq!(decode_fp_frame(&frame).unwrap(), entry);

    // Corrupt magic
    let mut bad_magic = frame;
    bad_magic[0] = b'X';
    assert!(matches!(
        decode_fp_frame(&bad_magic),
        Err(AerisError::WalCorruption { .. })
    ));

    // Corrupt payload byte (breaks CRC32)
    let mut bad_crc = frame;
    bad_crc[15] ^= 0xFF;
    assert!(matches!(
        decode_fp_frame(&bad_crc),
        Err(AerisError::WalCorruption { .. })
    ));
}

#[test]
fn deduplication_bounded_rotation_and_persistence_across_reopen() {
    let dir = TempDir::new().unwrap();
    let log_path = dir.path().join("fp.log");
    let cfg = FpLogConfig {
        max_entries: 5,
        sync_on_append: true,
    };

    {
        let log = FalsePositiveLog::open(&log_path, cfg).unwrap();
        let r1 = QueryRange::new(10, 20).unwrap();

        // Record r1 twice -> second is deduplicated.
        let e1 = log.record(r1, 1000).unwrap().unwrap();
        assert_eq!(e1.seq, 1);
        assert!(log.record(r1, 1001).unwrap().is_none());
        assert_eq!(log.len(), 1);

        // Record 7 distinct ranges -> exceeds max_entries=5 and triggers FIFO rotation.
        for i in 2..=8u64 {
            let r = QueryRange::new(i * 100, i * 100 + 31).unwrap();
            let entry = log.record(r, 1000 + i).unwrap().unwrap();
            assert_eq!(entry.seq, i);
        }
        assert_eq!(log.len(), 5);
        assert_eq!(log.compactions(), 3);
        assert!(!log.contains_range(r1));
        assert_eq!(log.last_seq(), 8);
    }

    // Reopen and verify exact retained entries, seq monotonicity, and file size.
    let reopened = FalsePositiveLog::open(&log_path, cfg).unwrap();
    assert_eq!(reopened.len(), 5);
    assert_eq!(reopened.last_seq(), 8);
    assert_eq!(
        fs::metadata(&log_path).unwrap().len(),
        (5 * FP_LOG_FRAME_LEN) as u64
    );

    let (_, entries) = reopened.snapshot_entries();
    let seqs: Vec<u64> = entries.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, vec![4, 5, 6, 7, 8]);

    // Next record gets seq 9.
    let e9 = reopened
        .record(QueryRange::new(9000, 9031).unwrap(), 2000)
        .unwrap()
        .unwrap();
    assert_eq!(e9.seq, 9);
}

#[test]
fn torn_tail_is_truncated_cleanly_and_mid_log_corruption_errors() {
    let dir = TempDir::new().unwrap();
    let log_path = dir.path().join("fp.log");
    let cfg = FpLogConfig::default();

    {
        let log = FalsePositiveLog::open(&log_path, cfg).unwrap();
        for i in 0..4u64 {
            log.record(QueryRange::new(i * 50, i * 50 + 10).unwrap(), 100 + i)
                .unwrap();
        }
    }

    // Append 17 junk bytes simulating crash mid-write.
    {
        let mut f = OpenOptions::new().append(true).open(&log_path).unwrap();
        f.write_all(&[
            0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        ])
        .unwrap();
        f.sync_all().unwrap();
    }

    {
        let recovered = FalsePositiveLog::open(&log_path, cfg).unwrap();
        assert_eq!(recovered.len(), 4);
        assert_eq!(recovered.truncated_tail_bytes(), 17);
        assert_eq!(recovered.last_seq(), 4);
    }

    // Corrupt frame #2 (offset 40..80) while frames #3 and #4 remain valid -> mid-log corruption!
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&log_path)
            .unwrap();
        f.seek(SeekFrom::Start(50)).unwrap();
        f.write_all(&[0xFF, 0xEE]).unwrap();
        f.sync_all().unwrap();
    }

    let err = FalsePositiveLog::open(&log_path, cfg);
    assert!(matches!(err, Err(AerisError::WalCorruption { .. })));
}

#[test]
fn replay_onto_and_replay_since_eliminate_collisions_on_real_filter() {
    let dir = TempDir::new().unwrap();
    let rm = RocksDbReverseMap::open(dir.path().join("rm"), ReverseMapConfig::default()).unwrap();
    let log = FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap();

    // Build a real AerisFilter with dense keys so collisions arise naturally.
    let cfg = FilterConfig::new(8, 12, 32, 17, 1.0 / 256.0).unwrap();
    let keys: Vec<u64> = (0..180u64).map(|i| i * 500).collect();
    let mut filter = AerisFilter::bulk_load(cfg, &keys, &rm).unwrap();

    // Discover genuine empty false-positive intervals and log them.
    let mut found_fps = Vec::new();
    for probe in 0..40_000u64 {
        let start = probe * 500 + 100;
        let range = QueryRange::new(start, start + 31).unwrap();
        if filter.query_range(range) {
            log.record(range, 1000 + probe).unwrap();
            found_fps.push(range);
            if found_fps.len() >= 10 {
                break;
            }
        }
    }
    assert!(
        !found_fps.is_empty(),
        "expected at least one natural FP on 40k probes"
    );

    // Also log a range that actually contains key 0 (must be skipped by verify_empty!).
    let non_empty_range = QueryRange::new(0, 10).unwrap();
    log.record(non_empty_range, 9999).unwrap();

    // Replay onto the real filter: all empty FPs must be resolved, non-empty skipped.
    let key_set: std::collections::BTreeSet<u64> = keys.iter().copied().collect();
    let report = log
        .replay_onto(&mut filter, &rm, |r| {
            Ok(key_set.range(r.start()..=r.end()).next().is_none())
        })
        .unwrap();

    assert_eq!(report.collisions_resolved, found_fps.len());
    assert_eq!(report.skipped_non_empty, 1);

    // Verify 100% monotonicity: every logged empty FP now returns false!
    for &fp_range in &found_fps {
        assert!(
            !filter.query_range(fp_range),
            "replayed FP range {:?} still returned true",
            fp_range
        );
    }
    // Verify zero false negatives across all original keys!
    for &k in &keys {
        assert!(filter.query_point(k), "false negative for key {k}");
    }
}

#[test]
fn config_validation_invalid_frame_range_and_stale_tmp_cleanup_work() {
    assert!(matches!(
        FpLogConfig {
            max_entries: 0,
            sync_on_append: false,
        }
        .validate(),
        Err(AerisError::InvalidConfig(_))
    ));

    // Craft frame with start > end and matching CRC32 -> decode_fp_frame rejects it.
    let mut bad_range_frame = [0u8; FP_LOG_FRAME_LEN];
    bad_range_frame[0..4].copy_from_slice(&FP_LOG_MAGIC);
    bad_range_frame[4..12].copy_from_slice(&1u64.to_le_bytes());
    bad_range_frame[12..20].copy_from_slice(&200u64.to_le_bytes());
    bad_range_frame[20..28].copy_from_slice(&100u64.to_le_bytes());
    bad_range_frame[28..36].copy_from_slice(&0u64.to_le_bytes());
    let crc = crc32fast::hash(&bad_range_frame[4..36]);
    bad_range_frame[36..40].copy_from_slice(&crc.to_le_bytes());
    assert!(matches!(
        decode_fp_frame(&bad_range_frame),
        Err(AerisError::WalCorruption { lsn: 1, .. })
    ));

    // Stale .compact.tmp is removed on open and non-sync append + explicit sync works.
    let dir = TempDir::new().unwrap();
    let log_path = dir.path().join("fp.log");
    let stale_tmp = compaction_tmp_path(&log_path);
    fs::write(&stale_tmp, b"stale").unwrap();

    let log = FalsePositiveLog::open(
        &log_path,
        FpLogConfig {
            max_entries: 10,
            sync_on_append: false,
        },
    )
    .unwrap();
    assert!(!stale_tmp.exists());
    assert!(log.is_empty());
    assert_eq!(log.path(), log_path);
    log.record(QueryRange::point(7), 1).unwrap();
    log.record(QueryRange::point(8), 2).unwrap();
    log.sync().unwrap();
    let (last_seq, since) = log.entries_since(1);
    assert_eq!(last_seq, 2);
    assert_eq!(since.len(), 1);
    assert_eq!(since[0].seq, 2);
}
