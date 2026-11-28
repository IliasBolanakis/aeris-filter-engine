//! WAL crash-recovery scanner, torn-tail truncation, and mid-log corruption detection.

use crate::codec::{
    compute_frame_crc32, decode_wal_op, WalFrameHeader, MAX_WAL_PAYLOAD_LEN, WAL_FORMAT_VERSION,
    WAL_HEADER_LEN, WAL_MAGIC,
};
use aeris_types::{AerisError, AerisResult, Lsn, WalOp, WalRecord};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use tracing::warn;

/// Maximum number of bytes scanned ahead when checking whether a corrupted frame
/// is followed by intact records (mid-log corruption vs. torn tail).
const CORRUPTION_LOOKAHEAD_CAP_BYTES: usize = 4 * 1024 * 1024;

/// Summary produced by scanning and validating a WAL file on open or replay.
pub(super) struct ScanOutcome {
    pub(super) records: Vec<WalRecord>,
    pub(super) valid_end: u64,
    pub(super) last_valid_lsn: Lsn,
    pub(super) max_checkpoint_lsn: Lsn,
    pub(super) truncated_tail_bytes: u64,
}

enum FrameStep {
    Record { record: WalRecord, next_offset: u64 },
    TruncatedTail(u64),
}

/// Scans `file` from offset 0 validating magic, version, length bounds, CRC32,
/// and LSN monotonicity. Truncates and `fsync`s any torn tail cleanly while
/// returning [`AerisError::WalCorruption`] on mid-log corruption.
pub(super) fn scan_and_recover_file(file: &mut File, path: &Path) -> AerisResult<ScanOutcome> {
    let file_len = file
        .metadata()
        .map_err(|err| AerisError::WalIoError(format!("WAL stat failed: {err}")))?
        .len();
    file.seek(SeekFrom::Start(0))
        .map_err(|err| AerisError::WalIoError(format!("WAL rewind failed: {err}")))?;

    let mut records = Vec::new();
    let mut offset = 0u64;
    let mut last_valid_lsn: Lsn = 0;
    let mut max_checkpoint_lsn: Lsn = 0;
    let mut truncated_tail_bytes = 0u64;

    while offset < file_len {
        match read_and_validate_frame(file, path, offset, file_len, last_valid_lsn)? {
            FrameStep::Record {
                record,
                next_offset,
            } => {
                if let WalOp::Checkpoint { up_to_lsn } = &record.op {
                    max_checkpoint_lsn = max_checkpoint_lsn.max(*up_to_lsn);
                }
                last_valid_lsn = record.lsn;
                records.push(record);
                offset = next_offset;
            }
            FrameStep::TruncatedTail(truncated) => {
                truncated_tail_bytes = truncated;
                break;
            }
        }
    }

    file.seek(SeekFrom::Start(offset))
        .map_err(|err| AerisError::WalIoError(format!("WAL seek to valid_end failed: {err}")))?;

    Ok(ScanOutcome {
        records,
        valid_end: offset,
        last_valid_lsn,
        max_checkpoint_lsn,
        truncated_tail_bytes,
    })
}

fn read_and_validate_frame(
    file: &mut File,
    path: &Path,
    offset: u64,
    file_len: u64,
    last_valid_lsn: Lsn,
) -> AerisResult<FrameStep> {
    let remaining = file_len - offset;
    if remaining < WAL_HEADER_LEN as u64 {
        let truncated = truncate_torn_tail(
            file,
            path,
            offset,
            file_len,
            last_valid_lsn,
            "incomplete WAL header at EOF",
        )?;
        return Ok(FrameStep::TruncatedTail(truncated));
    }

    let Some(header) = read_header_or_zero_tail(file, path, offset, file_len, last_valid_lsn)?
    else {
        let truncated = truncate_torn_tail(
            file,
            path,
            offset,
            file_len,
            last_valid_lsn,
            "zero-filled torn tail at EOF",
        )?;
        return Ok(FrameStep::TruncatedTail(truncated));
    };

    let frame_len = (WAL_HEADER_LEN as u64) + (header.payload_len as u64);
    let frame_end = offset.saturating_add(frame_len);
    let Some(payload) = read_bounded_payload_or_tail(
        file,
        path,
        offset,
        frame_end,
        file_len,
        &header,
        last_valid_lsn,
    )?
    else {
        let truncated = truncate_torn_tail(
            file,
            path,
            offset,
            file_len,
            last_valid_lsn,
            "truncated WAL payload at EOF",
        )?;
        return Ok(FrameStep::TruncatedTail(truncated));
    };

    if !verify_crc_or_tail(
        file,
        path,
        offset,
        frame_end,
        file_len,
        &header,
        &payload,
        last_valid_lsn,
    )? {
        let truncated = file_len.saturating_sub(offset);
        return Ok(FrameStep::TruncatedTail(truncated));
    }

    if header.lsn == 0 || header.lsn <= last_valid_lsn {
        return Err(AerisError::WalCorruption {
            lsn: header.lsn,
            reason: format!(
                "non-monotonic WAL LSN {} following {} at byte offset {offset}",
                header.lsn, last_valid_lsn
            ),
        });
    }

    let op = decode_wal_op(&payload).map_err(|err| AerisError::WalCorruption {
        lsn: header.lsn,
        reason: format!("invalid WalOp payload at byte offset {offset}: {err}"),
    })?;

    Ok(FrameStep::Record {
        record: WalRecord {
            lsn: header.lsn,
            checksum: header.crc32,
            op,
        },
        next_offset: frame_end,
    })
}

/// Reads and validates a 21-byte WAL header at `offset`.
/// Returns `Ok(None)` when the remaining tail bytes are all `0x00`.
fn read_header_or_zero_tail(
    file: &mut File,
    _path: &Path,
    offset: u64,
    file_len: u64,
    _last_valid_lsn: Lsn,
) -> AerisResult<Option<WalFrameHeader>> {
    let mut header_buf = [0u8; WAL_HEADER_LEN];
    read_exact_at(file, offset, &mut header_buf)?;

    if header_buf[0..4] != WAL_MAGIC {
        if is_all_zero_tail(file, offset, file_len)? {
            return Ok(None);
        }
        return Err(AerisError::WalCorruption {
            lsn: 0,
            reason: format!(
                "invalid WAL frame magic {:?} at byte offset {offset}",
                &header_buf[0..4]
            ),
        });
    }

    if header_buf[4] != WAL_FORMAT_VERSION {
        return Err(AerisError::WalCorruption {
            lsn: 0,
            reason: format!(
                "unsupported WAL version {} at byte offset {offset}",
                header_buf[4]
            ),
        });
    }

    let header = WalFrameHeader::decode(&header_buf).map_err(|err| AerisError::WalCorruption {
        lsn: 0,
        reason: format!("header decode failed at byte offset {offset}: {err}"),
    })?;
    Ok(Some(header))
}

/// Validates payload length bounds and reads the payload bytes, or returns `Ok(None)`
/// if `frame_end > file_len` with no valid trailing frame (torn tail).
fn read_bounded_payload_or_tail(
    file: &mut File,
    _path: &Path,
    offset: u64,
    frame_end: u64,
    file_len: u64,
    header: &WalFrameHeader,
    _last_valid_lsn: Lsn,
) -> AerisResult<Option<Vec<u8>>> {
    if frame_end > file_len {
        if has_valid_frame_after(file, offset + WAL_HEADER_LEN as u64, file_len)? {
            return Err(AerisError::WalCorruption {
                lsn: header.lsn,
                reason: format!(
                    "corrupt payload_len {} at non-tail byte offset {offset}",
                    header.payload_len
                ),
            });
        }
        return Ok(None);
    }

    if header.payload_len > MAX_WAL_PAYLOAD_LEN {
        return Err(AerisError::WalCorruption {
            lsn: header.lsn,
            reason: format!(
                "WAL payload length {} exceeds cap {} at byte offset {offset}",
                header.payload_len, MAX_WAL_PAYLOAD_LEN
            ),
        });
    }

    let mut payload = vec![0u8; header.payload_len as usize];
    read_exact_at(file, offset + WAL_HEADER_LEN as u64, &mut payload)?;
    Ok(Some(payload))
}

#[allow(clippy::too_many_arguments)]
fn verify_crc_or_tail(
    file: &mut File,
    path: &Path,
    offset: u64,
    frame_end: u64,
    file_len: u64,
    header: &WalFrameHeader,
    payload: &[u8],
    last_valid_lsn: Lsn,
) -> AerisResult<bool> {
    let actual_crc = compute_frame_crc32(header.version, header.payload_len, header.lsn, payload);
    if actual_crc == header.crc32 {
        return Ok(true);
    }
    if frame_end == file_len {
        let _ = truncate_torn_tail(
            file,
            path,
            offset,
            file_len,
            last_valid_lsn,
            "tail WAL frame failed CRC32 verification at EOF",
        )?;
        return Ok(false);
    }
    Err(AerisError::WalCorruption {
        lsn: header.lsn,
        reason: format!(
            "CRC32 mismatch at non-tail byte offset {offset}: expected 0x{:08X}, computed 0x{:08X}",
            header.crc32, actual_crc
        ),
    })
}

/// Truncates `file` cleanly to `valid_end`, flushes via `sync_all()`, and emits a
/// structured warning with truncated byte count and last intact LSN.
fn truncate_torn_tail(
    file: &mut File,
    path: &Path,
    valid_end: u64,
    original_len: u64,
    last_valid_lsn: Lsn,
    reason: &str,
) -> AerisResult<u64> {
    let truncated = original_len.saturating_sub(valid_end);
    file.set_len(valid_end).map_err(|err| {
        AerisError::WalIoError(format!(
            "failed to truncate torn WAL tail at offset {valid_end}: {err}"
        ))
    })?;
    file.sync_all()
        .map_err(|err| AerisError::WalIoError(format!("failed to fsync truncated WAL: {err}")))?;
    warn!(
        path = %path.display(),
        truncated_bytes = truncated,
        valid_end_offset = valid_end,
        last_intact_lsn = last_valid_lsn,
        reason,
        "truncated torn WAL tail frame during recovery"
    );
    Ok(truncated)
}

fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> AerisResult<()> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| AerisError::WalIoError(format!("WAL seek to {offset} failed: {err}")))?;
    file.read_exact(buf)
        .map_err(|err| AerisError::WalIoError(format!("WAL read at {offset} failed: {err}")))
}

/// Returns `true` iff every remaining byte in `[offset..file_len]` is `0x00`.
pub(super) fn is_all_zero_tail(file: &mut File, offset: u64, file_len: u64) -> AerisResult<bool> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| AerisError::WalIoError(format!("WAL seek failed: {err}")))?;
    let mut remaining = file_len.saturating_sub(offset);
    let mut buf = [0u8; 4096];
    while remaining > 0 {
        let to_read = (remaining as usize).min(buf.len());
        file.read_exact(&mut buf[..to_read])
            .map_err(|err| AerisError::WalIoError(format!("WAL tail read failed: {err}")))?;
        if buf[..to_read].iter().any(|&b| b != 0) {
            return Ok(false);
        }
        remaining -= to_read as u64;
    }
    Ok(true)
}

/// Scans forward from `search_start` up to `CORRUPTION_LOOKAHEAD_CAP_BYTES` to check
/// if any intact, CRC32-verified WAL frame follows an apparent truncation.
pub(super) fn has_valid_frame_after(
    file: &mut File,
    search_start: u64,
    file_len: u64,
) -> AerisResult<bool> {
    if search_start >= file_len {
        return Ok(false);
    }
    let scan_len = (file_len - search_start).min(CORRUPTION_LOOKAHEAD_CAP_BYTES as u64) as usize;
    if scan_len < WAL_HEADER_LEN {
        return Ok(false);
    }
    let mut buf = vec![0u8; scan_len];
    read_exact_at(file, search_start, &mut buf)?;

    for i in 0..=buf.len().saturating_sub(WAL_HEADER_LEN) {
        if buf[i..i + 4] != WAL_MAGIC || buf[i + 4] != WAL_FORMAT_VERSION {
            continue;
        }
        let Ok(hdr) = WalFrameHeader::decode(&buf[i..i + WAL_HEADER_LEN]) else {
            continue;
        };
        if hdr.payload_len > MAX_WAL_PAYLOAD_LEN || hdr.lsn == 0 {
            continue;
        }
        let end = i + WAL_HEADER_LEN + hdr.payload_len as usize;
        if end <= buf.len() {
            let crc = compute_frame_crc32(
                hdr.version,
                hdr.payload_len,
                hdr.lsn,
                &buf[i + WAL_HEADER_LEN..end],
            );
            if crc == hdr.crc32 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(super) fn compaction_tmp_path(wal_path: &Path) -> PathBuf {
    let mut os = wal_path.as_os_str().to_os_string();
    os.push(".compact.tmp");
    PathBuf::from(os)
}

pub(super) fn cleanup_stale_compaction_tmp(wal_path: &Path) {
    let tmp = compaction_tmp_path(wal_path);
    if tmp.exists() {
        match fs::remove_file(&tmp) {
            Ok(()) => warn!(
                tmp_path = %tmp.display(),
                "removed leftover WAL compaction temp file from interrupted compaction"
            ),
            Err(err) => warn!(
                tmp_path = %tmp.display(),
                error = %err,
                "failed to remove leftover WAL compaction temp file"
            ),
        }
    }
}
