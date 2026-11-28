//! Crash-recovery scanning, torn-tail truncation, and atomic rotation helpers
//! for [`super::FalsePositiveLog`].

use super::frame::{decode_fp_frame, encode_fp_frame, FP_LOG_FRAME_LEN};
use super::FpLogInner;
use aeris_types::{AerisError, AerisResult, FpLogEntry};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

/// Summary produced when scanning an on-disk FP log file during recovery.
pub(super) struct ScanOutcome {
    pub(super) entries: Vec<FpLogEntry>,
    pub(super) valid_end: u64,
    pub(super) last_seq: u64,
    pub(super) truncated_tail_bytes: u64,
}

enum FrameStep {
    Valid(FpLogEntry),
    TornTail,
}

/// Ensures the parent directory of `path` exists before opening the log file.
pub(super) fn ensure_parent_dir(path: &Path) -> AerisResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                AerisError::WalIoError(format!(
                    "failed to create FP log directory '{}': {err}",
                    parent.display()
                ))
            })?;
        }
    }
    Ok(())
}

/// Opens (or creates) the FP log file in read/write mode without `O_APPEND`
/// so explicit seeks and torn-tail truncation behave identically on Windows and Unix.
pub(super) fn open_fp_log_file(path: &Path) -> AerisResult<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to open FP log file '{}': {err}",
                path.display()
            ))
        })
}

/// Scans `file` from offset 0 validating all frames, truncating any torn tail
/// cleanly, and rejecting mid-log corruption with [`AerisError::WalCorruption`].
pub(super) fn scan_and_recover_file(file: &mut File, path: &Path) -> AerisResult<ScanOutcome> {
    let file_len = file
        .metadata()
        .map_err(|err| AerisError::WalIoError(format!("FP log stat failed: {err}")))?
        .len();

    file.seek(SeekFrom::Start(0))
        .map_err(|err| AerisError::WalIoError(format!("FP log seek(0) failed: {err}")))?;

    let mut raw_bytes = Vec::with_capacity(file_len as usize);
    file.read_to_end(&mut raw_bytes)
        .map_err(|err| AerisError::WalIoError(format!("FP log read failed: {err}")))?;

    let mut entries = Vec::new();
    let mut offset = 0usize;
    let mut last_seq = 0u64;

    while offset < raw_bytes.len() {
        match process_scanned_frame(&raw_bytes, offset, last_seq)? {
            FrameStep::Valid(entry) => {
                last_seq = entry.seq;
                entries.push(entry);
                offset += FP_LOG_FRAME_LEN;
            }
            FrameStep::TornTail => {
                return truncate_torn_tail(file, path, offset as u64, file_len, entries, last_seq);
            }
        }
    }

    Ok(ScanOutcome {
        entries,
        valid_end: offset as u64,
        last_seq,
        truncated_tail_bytes: 0,
    })
}

fn process_scanned_frame(raw_bytes: &[u8], offset: usize, last_seq: u64) -> AerisResult<FrameStep> {
    let remaining = raw_bytes.len() - offset;
    if remaining < FP_LOG_FRAME_LEN {
        return Ok(FrameStep::TornTail);
    }

    let frame: [u8; FP_LOG_FRAME_LEN] = raw_bytes[offset..offset + FP_LOG_FRAME_LEN]
        .try_into()
        .expect("exact 40-byte slice");
    let trailing = &raw_bytes[offset + FP_LOG_FRAME_LEN..];

    match decode_fp_frame(&frame) {
        Ok(entry) if entry.seq > last_seq => Ok(FrameStep::Valid(entry)),
        Ok(entry) => {
            if trailing_has_intact_frame(trailing, last_seq) {
                return Err(AerisError::WalCorruption {
                    lsn: entry.seq,
                    reason: format!(
                        "non-monotonic FP log seq {} after {} at offset {offset}",
                        entry.seq, last_seq
                    ),
                });
            }
            Ok(FrameStep::TornTail)
        }
        Err(err) => {
            if trailing_has_intact_frame(trailing, last_seq) {
                return Err(err);
            }
            Ok(FrameStep::TornTail)
        }
    }
}

/// Checks whether `trailing` contains at least one intact valid frame after a
/// corrupted offset (distinguishing mid-log corruption from a torn tail).
pub(super) fn trailing_has_intact_frame(trailing: &[u8], last_seq: u64) -> bool {
    let mut pos = 0usize;
    while pos + FP_LOG_FRAME_LEN <= trailing.len() {
        let candidate: [u8; FP_LOG_FRAME_LEN] = trailing[pos..pos + FP_LOG_FRAME_LEN]
            .try_into()
            .expect("exact 40-byte slice");
        if let Ok(entry) = decode_fp_frame(&candidate) {
            if entry.seq > last_seq {
                return true;
            }
        }
        pos += FP_LOG_FRAME_LEN;
    }
    false
}

/// Truncates `file` to `valid_end`, `fsync`s the truncated length, and logs a warning.
fn truncate_torn_tail(
    file: &mut File,
    path: &Path,
    valid_end: u64,
    file_len: u64,
    entries: Vec<FpLogEntry>,
    last_seq: u64,
) -> AerisResult<ScanOutcome> {
    let truncated = file_len.saturating_sub(valid_end);
    file.set_len(valid_end).map_err(|err| {
        AerisError::WalIoError(format!(
            "failed to truncate torn FP log tail at offset {valid_end}: {err}"
        ))
    })?;
    file.seek(SeekFrom::Start(valid_end)).map_err(|err| {
        AerisError::WalIoError(format!("FP log seek after truncate failed: {err}"))
    })?;
    file.sync_all().map_err(|err| {
        AerisError::WalIoError(format!("FP log sync after truncate failed: {err}"))
    })?;

    warn!(
        path = %path.display(),
        valid_end,
        truncated_tail_bytes = truncated,
        recovered_entries = entries.len(),
        last_seq,
        "truncated torn tail from FalsePositiveLog during recovery"
    );

    Ok(ScanOutcome {
        entries,
        valid_end,
        last_seq,
        truncated_tail_bytes: truncated,
    })
}

/// Atomically rewrites the active FP log file with only the retained `inner.entries`.
///
/// Drops `inner.file` prior to `fs::rename` so Windows does not fail with
/// `ERROR_SHARING_VIOLATION` while replacing an open file.
pub(super) fn compact_locked(path: &Path, inner: &mut FpLogInner) -> AerisResult<()> {
    let tmp_path = compaction_tmp_path(path);
    let new_size = write_compacted_tmp(&tmp_path, &inner.entries)?;

    inner.file.take();
    if let Err(rename_err) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        if let Ok(reopened) = open_fp_log_file(path) {
            inner.file = Some(reopened);
        }
        error!(
            path = %path.display(),
            error = %rename_err,
            "atomic FP log rotation failed"
        );
        return Err(AerisError::WalIoError(format!(
            "failed to atomically rotate FP log '{}': {rename_err}",
            path.display()
        )));
    }

    sync_parent_dir(path);

    let mut reopened = open_fp_log_file(path)?;
    reopened.seek(SeekFrom::Start(new_size)).map_err(|err| {
        AerisError::WalIoError(format!("FP log seek after rotation failed: {err}"))
    })?;
    inner.file = Some(reopened);
    inner.write_offset = new_size;
    inner.compactions = inner.compactions.saturating_add(1);

    info!(
        path = %path.display(),
        retained_entries = inner.entries.len(),
        new_bytes = new_size,
        compactions = inner.compactions,
        "atomically rotated bounded FalsePositiveLog"
    );
    Ok(())
}

/// Writes `entries` to `tmp_path` and calls `flush` + `sync_all`.
fn write_compacted_tmp(tmp_path: &Path, entries: &VecDeque<FpLogEntry>) -> AerisResult<u64> {
    let mut tmp_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp_path)
        .map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to create FP log compaction tmp '{}': {err}",
                tmp_path.display()
            ))
        })?;

    let mut total_bytes = 0u64;
    for entry in entries {
        let frame = encode_fp_frame(entry);
        tmp_file.write_all(&frame).map_err(|err| {
            AerisError::WalIoError(format!("failed to write FP log tmp frame: {err}"))
        })?;
        total_bytes += FP_LOG_FRAME_LEN as u64;
    }

    tmp_file
        .flush()
        .map_err(|err| AerisError::WalIoError(format!("FP log tmp flush failed: {err}")))?;
    tmp_file
        .sync_all()
        .map_err(|err| AerisError::WalIoError(format!("FP log tmp sync_all failed: {err}")))?;
    Ok(total_bytes)
}

/// Computes the sibling temporary path used during atomic bounded rotation.
pub(super) fn compaction_tmp_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".compact.tmp");
    PathBuf::from(os)
}

/// Cleans up any leftover `.compact.tmp` file from a prior crash mid-rotation.
pub(super) fn cleanup_stale_compaction_tmp(path: &Path) {
    let tmp = compaction_tmp_path(path);
    if tmp.exists() {
        warn!(
            tmp_path = %tmp.display(),
            "removing stale FP log compaction temporary file"
        );
        let _ = fs::remove_file(&tmp);
    }
}

/// Best-effort parent directory sync on Unix filesystems after atomic rename.
#[cfg(unix)]
fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

/// No-op parent directory sync on non-Unix platforms (Windows NTFS commits rename metadata).
#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) {}
