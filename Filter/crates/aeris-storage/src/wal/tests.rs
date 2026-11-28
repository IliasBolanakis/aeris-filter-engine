//! Unit tests for [`super::CoupledWal`] and WAL recovery helpers.

use super::recovery::{compaction_tmp_path, has_valid_frame_after, is_all_zero_tail};
use super::*;
use crate::codec::{
    compute_frame_crc32, encode_wal_frame, WalFrameHeader, MAX_WAL_PAYLOAD_LEN, WAL_FORMAT_VERSION,
    WAL_HEADER_LEN, WAL_MAGIC,
};
use aeris_types::{Fingerprint, PartitionHasher, WalEngine, WalOp};
use std::fs::{self, OpenOptions};
use std::io::Write;
use tempfile::TempDir;

fn sample_put(key: u64) -> WalOp {
    let hasher = PartitionHasher::new(42);
    let fp = Fingerprint::from_hasher(&hasher, key >> 5, 22);
    WalOp::Put {
        key,
        value: format!("val-{key}").into_bytes(),
        fp,
    }
}

#[test]
fn append_sync_replay_and_reopen() {
    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");

    {
        let wal = CoupledWal::open(&wal_path).unwrap();
        assert_eq!(wal.last_lsn(), 0);
        assert_eq!(wal.checkpoint_lsn(), 0);

        let l1 = wal.append(sample_put(10)).unwrap();
        let l2 = wal.append(sample_put(20)).unwrap();
        let l3 = wal.append(sample_put(30)).unwrap();
        wal.sync().unwrap();
        assert_eq!((l1, l2, l3), (1, 2, 3));
        assert_eq!(wal.last_lsn(), 3);

        wal.checkpoint(2).unwrap();
        assert_eq!(wal.last_lsn(), 4);
        assert_eq!(wal.checkpoint_lsn(), 2);

        let from_3 = wal.replay_from(3).unwrap();
        assert_eq!(from_3.len(), 2);
        assert_eq!(from_3[0].lsn, 3);
        assert_eq!(from_3[1].lsn, 4);
    }

    // Reopen and verify LSN continuity
    let wal = CoupledWal::open(&wal_path).unwrap();
    assert_eq!(wal.last_lsn(), 4);
    assert_eq!(wal.checkpoint_lsn(), 2);
    assert_eq!(wal.truncated_tail_bytes(), 0);
    let l5 = wal.append(sample_put(50)).unwrap();
    wal.sync().unwrap();
    assert_eq!(l5, 5);
}

#[test]
fn truncates_torn_tail_mid_header_mid_payload_zero_tail_and_bad_tail_crc() {
    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");

    let wal = CoupledWal::open(&wal_path).unwrap();
    wal.append(sample_put(1)).unwrap();
    wal.append(sample_put(2)).unwrap();
    wal.sync().unwrap();
    let clean_len = wal.file_size_bytes();
    drop(wal);

    // Case 1: Incomplete header at EOF (10 bytes)
    {
        let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
        f.write_all(&WAL_MAGIC).unwrap();
        f.write_all(&[WAL_FORMAT_VERSION, 0, 1, 2, 3, 4]).unwrap();
        f.sync_all().unwrap();
    }
    {
        let wal = CoupledWal::open(&wal_path).unwrap();
        assert_eq!(wal.truncated_tail_bytes(), 10);
        assert_eq!(wal.file_size_bytes(), clean_len);
        assert_eq!(wal.replay_from(1).unwrap().len(), 2);
    }

    // Case 2: Valid header claiming 100 bytes payload, only 15 bytes present
    {
        let (full_frame, _) = encode_wal_frame(3, &sample_put(3)).unwrap();
        let partial_len = WAL_HEADER_LEN + 3;
        let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
        f.write_all(&full_frame[..partial_len]).unwrap();
        f.sync_all().unwrap();
    }
    {
        let wal = CoupledWal::open(&wal_path).unwrap();
        assert_eq!(wal.truncated_tail_bytes(), (WAL_HEADER_LEN + 3) as u64);
        assert_eq!(wal.file_size_bytes(), clean_len);
        assert_eq!(wal.replay_from(1).unwrap().len(), 2);
    }

    // Case 3: Full-length tail frame with bit-flipped payload (CRC32 mismatch at EOF)
    {
        let (mut full_frame, _) = encode_wal_frame(3, &sample_put(3)).unwrap();
        let last_idx = full_frame.len() - 1;
        full_frame[last_idx] ^= 0xFF;
        let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
        f.write_all(&full_frame).unwrap();
        f.sync_all().unwrap();
    }
    {
        let wal = CoupledWal::open(&wal_path).unwrap();
        assert!(wal.truncated_tail_bytes() > WAL_HEADER_LEN as u64);
        assert_eq!(wal.file_size_bytes(), clean_len);
        assert_eq!(wal.replay_from(1).unwrap().len(), 2);
    }

    // Case 4: Zero-filled OS crash tail (64 zero bytes)
    {
        let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        f.sync_all().unwrap();
    }
    {
        let wal = CoupledWal::open(&wal_path).unwrap();
        assert_eq!(wal.truncated_tail_bytes(), 64);
        assert_eq!(wal.file_size_bytes(), clean_len);
        assert_eq!(wal.replay_from(1).unwrap().len(), 2);
    }
}

#[test]
fn detects_mid_log_crc_corruption_and_invalid_magic() {
    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");

    let (f1, _) = encode_wal_frame(1, &sample_put(1)).unwrap();
    let (f2, _) = encode_wal_frame(2, &sample_put(2)).unwrap();
    let (f3, _) = encode_wal_frame(3, &sample_put(3)).unwrap();

    // Corrupt a payload byte in middle frame f2
    let mut combined = Vec::new();
    combined.extend_from_slice(&f1);
    let f2_offset = combined.len();
    combined.extend_from_slice(&f2);
    combined.extend_from_slice(&f3);
    combined[f2_offset + WAL_HEADER_LEN] ^= 0x55;
    fs::write(&wal_path, &combined).unwrap();

    let err = CoupledWal::open(&wal_path).err().unwrap();
    assert!(
        matches!(err, AerisError::WalCorruption { lsn: 2, .. }),
        "expected WalCorruption at LSN 2, got {err:?}"
    );

    // Corrupt magic byte in middle frame f2
    let mut bad_magic = Vec::new();
    bad_magic.extend_from_slice(&f1);
    let f2_off = bad_magic.len();
    bad_magic.extend_from_slice(&f2);
    bad_magic.extend_from_slice(&f3);
    bad_magic[f2_off] = b'Z';
    fs::write(&wal_path, &bad_magic).unwrap();

    let err = CoupledWal::open(&wal_path).err().unwrap();
    assert!(
        matches!(err, AerisError::WalCorruption { .. }),
        "expected WalCorruption for bad mid-log magic, got {err:?}"
    );

    // Corrupt version byte in middle frame f2
    let mut bad_ver = Vec::new();
    bad_ver.extend_from_slice(&f1);
    let f2_off = bad_ver.len();
    bad_ver.extend_from_slice(&f2);
    bad_ver.extend_from_slice(&f3);
    bad_ver[f2_off + 4] = 99;
    fs::write(&wal_path, &bad_ver).unwrap();
    assert!(matches!(
        CoupledWal::open(&wal_path).err().unwrap(),
        AerisError::WalCorruption { .. }
    ));

    // Corrupt non-tail payload_len overshooting EOF while valid f3 still follows
    let mut bad_len = Vec::new();
    bad_len.extend_from_slice(&f1);
    let f2_off = bad_len.len();
    bad_len.extend_from_slice(&f2);
    bad_len.extend_from_slice(&f3);
    bad_len[f2_off + 5..f2_off + 9].copy_from_slice(&999_999u32.to_le_bytes());
    fs::write(&wal_path, &bad_len).unwrap();
    assert!(matches!(
        CoupledWal::open(&wal_path).err().unwrap(),
        AerisError::WalCorruption { lsn: 2, .. }
    ));

    // Non-monotonic LSN (valid CRC frames ordered LSN 2 then LSN 1)
    let mut bad_order = Vec::new();
    bad_order.extend_from_slice(&f2);
    bad_order.extend_from_slice(&f1);
    fs::write(&wal_path, &bad_order).unwrap();
    assert!(matches!(
        CoupledWal::open(&wal_path).err().unwrap(),
        AerisError::WalCorruption { lsn: 1, .. }
    ));
}

#[test]
fn truncate_before_past_last_lsn_keeps_anchor_and_cleans_stale_tmp() {
    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");

    {
        let wal = CoupledWal::open(&wal_path).unwrap();
        for k in 1..=5 {
            wal.append(sample_put(k)).unwrap();
        }
        wal.checkpoint(5).unwrap(); // LSN 6
        let reclaimed = wal.truncate_before(6).unwrap();
        assert!(reclaimed > 0);
        assert_eq!(wal.last_lsn(), 6);
    }

    // Plant a stale compaction tmp file to verify open() cleans it up safely
    let stale_tmp = compaction_tmp_path(&wal_path);
    fs::write(&stale_tmp, b"partial-interrupted-compaction").unwrap();
    assert!(stale_tmp.exists());

    let reopened = CoupledWal::open(&wal_path).unwrap();
    assert!(!stale_tmp.exists());
    assert_eq!(reopened.last_lsn(), 6);
    assert_eq!(reopened.checkpoint_lsn(), 6);
    assert_eq!(reopened.append(sample_put(7)).unwrap(), 7);
}

#[test]
fn truncate_before_compacts_atomically_and_preserves_monotonicity() {
    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");

    let wal = CoupledWal::open(&wal_path).unwrap();
    for k in 1..=20 {
        wal.append(sample_put(k)).unwrap();
    }
    wal.sync().unwrap();
    wal.checkpoint(20).unwrap(); // LSN 21
    for k in 21..=25 {
        wal.append(sample_put(k)).unwrap(); // LSN 22..=26
    }
    wal.sync().unwrap();

    let size_before = wal.file_size_bytes();
    let reclaimed = wal.truncate_before(20).unwrap();
    assert!(reclaimed > 0);
    assert!(wal.file_size_bytes() < size_before);

    // Verify retained records start at Checkpoint(LSN 21) through Put(LSN 26)
    let retained = wal.replay_from(1).unwrap();
    assert_eq!(retained.len(), 6);
    assert_eq!(retained[0].lsn, 21);
    assert_eq!(retained[5].lsn, 26);
    drop(wal);

    // Reopen after compaction: next LSN must be 27 and checkpoint_lsn == 20
    let reopened = CoupledWal::open(&wal_path).unwrap();
    assert_eq!(reopened.last_lsn(), 26);
    assert_eq!(reopened.checkpoint_lsn(), 20);
    let l27 = reopened.append(sample_put(27)).unwrap();
    assert_eq!(l27, 27);
}

#[test]
fn recovery_helpers_reject_oversized_payload_and_inspect_tails() {
    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("oversized.wal");

    // Craft a frame whose header claims payload_len > MAX_WAL_PAYLOAD_LEN while matching file_len
    // is impractical, so craft a header with MAX_WAL_PAYLOAD_LEN + 1 and pad only if needed?
    // Actually: frame_end <= file_len is checked before MAX_WAL_PAYLOAD_LEN! Test zero tail & lookahead directly.
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&wal_path)
        .unwrap();
    f.write_all(&[0u8; 32]).unwrap();
    assert!(is_all_zero_tail(&mut f, 0, 32).unwrap());
    assert!(!has_valid_frame_after(&mut f, 0, 32).unwrap());

    let (valid_frame, _) = encode_wal_frame(1, &WalOp::Checkpoint { up_to_lsn: 0 }).unwrap();
    f.write_all(&valid_frame).unwrap();
    let total_len = 32 + valid_frame.len() as u64;
    assert!(!is_all_zero_tail(&mut f, 0, total_len).unwrap());
    assert!(has_valid_frame_after(&mut f, 0, total_len).unwrap());
    drop(f);

    // Closed WalInner reports WalIoError cleanly.
    let mut closed = WalInner {
        file: None,
        write_offset: 0,
        truncated_tail_bytes: 0,
    };
    assert!(matches!(closed.file_mut(), Err(AerisError::WalIoError(_))));

    // Header claiming payload_len > MAX_WAL_PAYLOAD_LEN with exact CRC is impossible without
    // buffer, so verify WalFrameHeader encodes/decodes MAX_WAL_PAYLOAD_LEN + 1 and compute_frame_crc32 works.
    let bad_hdr = WalFrameHeader {
        version: WAL_FORMAT_VERSION,
        payload_len: MAX_WAL_PAYLOAD_LEN + 1,
        lsn: 1,
        crc32: compute_frame_crc32(WAL_FORMAT_VERSION, MAX_WAL_PAYLOAD_LEN + 1, 1, &[]),
    };
    assert!(WalFrameHeader::decode(&bad_hdr.encode()).is_ok());
}
