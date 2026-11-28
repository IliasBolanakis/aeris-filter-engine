//! Unit tests for the binary WAL and Reverse Map codecs.

use super::reverse_map_codec::try_decode_exact_snapshot;
use super::wal_codec::WAL_OP_TAG_ADAPT;
use super::*;
use aeris_types::{Fingerprint, PartitionHasher, QueryRange, WalOp};
use std::collections::BTreeSet;

#[test]
fn wal_op_round_trip_all_variants() {
    let hasher = PartitionHasher::new(99);
    let fp_short = Fingerprint::from_hasher(&hasher, 17, 22);
    let fp_long = Fingerprint::from_hasher(&hasher, 42, 165);
    let fp_empty = Fingerprint::empty();

    let ops = [
        WalOp::Put {
            key: 0,
            value: Vec::new(),
            fp: fp_empty.clone(),
        },
        WalOp::Put {
            key: u64::MAX,
            value: b"aeris-storage-payload-v1".to_vec(),
            fp: fp_long.clone(),
        },
        WalOp::Delete {
            key: 123_456_789,
            fp: fp_short.clone(),
        },
        WalOp::Adapt {
            old_fps: vec![fp_short.clone(), fp_empty],
            new_entries: vec![(fp_long, 555), (fp_short, 777)],
            range: QueryRange::new(100, 999).unwrap(),
        },
        WalOp::Checkpoint { up_to_lsn: 42_000 },
    ];

    for (idx, op) in ops.iter().enumerate() {
        let encoded = encode_wal_op(op).unwrap();
        let decoded = decode_wal_op(&encoded).unwrap();
        assert_eq!(&decoded, op);

        let lsn = (idx as u64) + 1;
        let (frame, crc) = encode_wal_frame(lsn, op).unwrap();
        let header = WalFrameHeader::decode(&frame).unwrap();
        assert_eq!(header.version, WAL_FORMAT_VERSION);
        assert_eq!(header.lsn, lsn);
        assert_eq!(header.payload_len as usize, encoded.len());
        assert_eq!(header.crc32, crc);
        let recomputed = compute_frame_crc32(
            header.version,
            header.payload_len,
            header.lsn,
            &frame[WAL_HEADER_LEN..],
        );
        assert_eq!(recomputed, crc);
    }
}

#[test]
fn wal_op_decode_rejects_malformed_inputs() {
    // Empty payload
    assert!(decode_wal_op(&[]).is_err());
    // Unknown tag
    assert!(decode_wal_op(&[0xFF]).is_err());
    // Trailing bytes on Checkpoint
    let mut cp = encode_wal_op(&WalOp::Checkpoint { up_to_lsn: 7 }).unwrap();
    cp.push(0x00);
    assert!(decode_wal_op(&cp).is_err());
    // Truncated Put
    let hasher = PartitionHasher::new(1);
    let fp = Fingerprint::from_hasher(&hasher, 1, 20);
    let put = encode_wal_op(&WalOp::Put {
        key: 1,
        value: vec![1, 2, 3],
        fp,
    })
    .unwrap();
    assert!(decode_wal_op(&put[..put.len() - 1]).is_err());
    // Invalid Adapt range (start > end)
    let mut bad_adapt = vec![WAL_OP_TAG_ADAPT];
    bad_adapt.extend_from_slice(&50u64.to_le_bytes());
    bad_adapt.extend_from_slice(&10u64.to_le_bytes());
    bad_adapt.extend_from_slice(&0u32.to_le_bytes());
    bad_adapt.extend_from_slice(&0u32.to_le_bytes());
    assert!(decode_wal_op(&bad_adapt).is_err());
    // LSN 0 rejected on frame encode
    assert!(encode_wal_frame(0, &WalOp::Checkpoint { up_to_lsn: 1 }).is_err());
}

#[test]
fn wal_header_rejects_bad_magic_version_and_truncation() {
    assert!(WalFrameHeader::decode(&[0u8; WAL_HEADER_LEN - 1]).is_err());

    let (mut frame, _) = encode_wal_frame(1, &WalOp::Checkpoint { up_to_lsn: 0 }).unwrap();
    frame[0] = b'X';
    assert!(WalFrameHeader::decode(&frame).is_err());

    frame[0] = b'A';
    frame[4] = 99;
    assert!(WalFrameHeader::decode(&frame).is_err());
}

#[test]
fn reverse_map_merge_algebra_is_idempotent_and_associative() {
    let a10 = encode_add_key_operand(10);
    let a20 = encode_add_key_operand(20);
    let a30 = encode_add_key_operand(30);
    let d10 = encode_del_key_operand(10);
    let d99 = encode_del_key_operand(99);

    // Full merge without base
    let merged = fold_full_merge(
        None,
        [
            a20.as_slice(),
            a10.as_slice(),
            a10.as_slice(), // duplicate add -> idempotent
            d99.as_slice(), // delete absent key -> no-op
            a30.as_slice(),
            d10.as_slice(),
        ],
    )
    .unwrap();
    assert_eq!(decode_key_list(&merged).unwrap(), vec![20, 30]);
    assert_eq!(count_keys_in_value(&merged).unwrap(), 2);

    // Associativity check: partial_merge([d10, a30, a10, d20]) applied to base {10, 20}
    let mut base_set = BTreeSet::new();
    base_set.insert(10);
    base_set.insert(20);
    let base_snap = encode_compacted_snapshot(&base_set).unwrap();

    let d20 = encode_del_key_operand(20);
    let ops = [
        d10.as_slice(),
        a30.as_slice(),
        a10.as_slice(),
        d20.as_slice(),
    ];

    let direct = fold_full_merge(Some(&base_snap), ops).unwrap();
    let partial = fold_partial_merge(ops).unwrap();
    let via_partial = fold_full_merge(Some(&base_snap), [partial.as_slice()]).unwrap();
    assert_eq!(direct, via_partial);
    assert_eq!(decode_key_list(&via_partial).unwrap(), vec![10, 30]);

    // Snapshot inside partial_merge supersedes earlier deltas
    let snap_partial =
        fold_partial_merge([a30.as_slice(), base_snap.as_slice(), d10.as_slice()]).unwrap();
    assert_eq!(snap_partial[0], RM_TAG_COMPACTED_SNAPSHOT);
    assert_eq!(decode_key_list(&snap_partial).unwrap(), vec![20]);
}

#[test]
fn reverse_map_codec_rejects_corrupt_operands() {
    assert!(fold_full_merge(None, [&[0x99u8][..]]).is_err());
    assert!(fold_full_merge(None, [&[RM_TAG_ADD_KEY, 1, 2][..]]).is_err());
    assert!(fold_partial_merge([&[RM_TAG_DEL_KEY, 0][..]]).is_err());
    // Truncated CompactedSnapshot claiming 5 keys with 0 bytes
    let bad_snap = [RM_TAG_COMPACTED_SNAPSHOT, 5, 0, 0, 0];
    assert!(decode_key_list(&bad_snap).is_err());
    assert!(fold_partial_merge([bad_snap.as_slice()]).is_err());

    // Empty boundary checks: empty slice, empty CompactedSnapshot, empty partial_merge
    assert!(decode_key_list(&[]).unwrap().is_empty());
    assert_eq!(count_keys_in_value(&[]).unwrap(), 0);
    let empty_snap = encode_compacted_snapshot(&BTreeSet::new()).unwrap();
    assert_eq!(empty_snap, vec![RM_TAG_COMPACTED_SNAPSHOT, 0, 0, 0, 0]);
    assert!(decode_key_list(&empty_snap).unwrap().is_empty());
    assert_eq!(count_keys_in_value(&empty_snap).unwrap(), 0);
    let empty_partial = fold_partial_merge(std::iter::empty::<&[u8]>()).unwrap();
    assert!(empty_partial.is_empty());
}

#[test]
fn snapshot_fast_path_and_adapt_count_bounds_handle_edge_cases() {
    // Snapshot followed by an AddKey operand must fall back to full stream folding.
    let mut base = BTreeSet::new();
    base.insert(1);
    let mut combined = encode_compacted_snapshot(&base).unwrap();
    combined.extend_from_slice(&encode_add_key_operand(2));
    assert_eq!(try_decode_exact_snapshot(&combined).unwrap(), None);
    assert_eq!(decode_key_list(&combined).unwrap(), vec![1, 2]);
    assert_eq!(count_keys_in_value(&combined).unwrap(), 2);

    // Truncated snapshot header (< 5 bytes) fails.
    assert!(try_decode_exact_snapshot(&[RM_TAG_COMPACTED_SNAPSHOT, 0]).is_err());

    // Inflated Adapt old_fps count is rejected before OOM allocation.
    let mut inflated_adapt = vec![WAL_OP_TAG_ADAPT];
    inflated_adapt.extend_from_slice(&1u64.to_le_bytes());
    inflated_adapt.extend_from_slice(&2u64.to_le_bytes());
    inflated_adapt.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_wal_op(&inflated_adapt).is_err());
}
