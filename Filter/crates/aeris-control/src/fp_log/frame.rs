//! Binary 40-byte frame encoding and CRC32 validation for [`super::FalsePositiveLog`].
//!
//! # Frame Layout (`40` bytes)
//! `[magic: b"AFPL" (4B)][seq: u64 LE][start: u64 LE][end: u64 LE][timestamp_ms: u64 LE][crc32: u32 LE]`

use aeris_types::{AerisError, AerisResult, FpLogEntry, QueryRange};

/// 4-byte binary frame magic header for [`super::FalsePositiveLog`] records.
pub const FP_LOG_MAGIC: [u8; 4] = *b"AFPL";

/// Fixed on-disk byte length of a single encoded [`FpLogEntry`] frame.
pub const FP_LOG_FRAME_LEN: usize = 40;

/// Encodes a single [`FpLogEntry`] into its canonical 40-byte binary frame.
///
/// # Complexity
/// `O(1)` time and `O(1)` stack space, zero heap allocations.
#[must_use]
pub fn encode_fp_frame(entry: &FpLogEntry) -> [u8; FP_LOG_FRAME_LEN] {
    let mut buf = [0u8; FP_LOG_FRAME_LEN];
    buf[0..4].copy_from_slice(&FP_LOG_MAGIC);
    buf[4..12].copy_from_slice(&entry.seq.to_le_bytes());
    buf[12..20].copy_from_slice(&entry.range.start().to_le_bytes());
    buf[20..28].copy_from_slice(&entry.range.end().to_le_bytes());
    buf[28..36].copy_from_slice(&entry.timestamp_ms.to_le_bytes());
    let crc = crc32fast::hash(&buf[4..36]);
    buf[36..40].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Decodes and checksum-verifies a single 40-byte binary [`FpLogEntry`] frame.
///
/// # Errors
/// Returns [`AerisError::WalCorruption`] if the magic header, IEEE CRC32 checksum,
/// or closed-interval invariant (`start <= end`) is invalid.
///
/// # Complexity
/// `O(1)` time and `O(1)` space.
pub fn decode_fp_frame(raw: &[u8; FP_LOG_FRAME_LEN]) -> AerisResult<FpLogEntry> {
    if raw[0..4] != FP_LOG_MAGIC {
        return Err(AerisError::WalCorruption {
            lsn: 0,
            reason: format!(
                "invalid FP log magic: expected {:?}, got {:?}",
                FP_LOG_MAGIC,
                &raw[0..4]
            ),
        });
    }

    let seq = u64::from_le_bytes(raw[4..12].try_into().expect("8-byte slice"));
    let start = u64::from_le_bytes(raw[12..20].try_into().expect("8-byte slice"));
    let end = u64::from_le_bytes(raw[20..28].try_into().expect("8-byte slice"));
    let timestamp_ms = u64::from_le_bytes(raw[28..36].try_into().expect("8-byte slice"));
    let stored_crc = u32::from_le_bytes(raw[36..40].try_into().expect("4-byte slice"));

    let computed_crc = crc32fast::hash(&raw[4..36]);
    if stored_crc != computed_crc {
        return Err(AerisError::WalCorruption {
            lsn: seq,
            reason: format!(
                "FP log CRC32 mismatch at seq {seq}: stored=0x{stored_crc:08x}, computed=0x{computed_crc:08x}"
            ),
        });
    }

    let range = QueryRange::new(start, end).map_err(|_| AerisError::WalCorruption {
        lsn: seq,
        reason: format!("FP log entry at seq {seq} has invalid range [{start}, {end}]"),
    })?;

    Ok(FpLogEntry {
        seq,
        range,
        timestamp_ms,
    })
}
