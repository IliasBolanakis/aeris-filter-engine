//! Crash-consistent binary-framed Write-Ahead Log (`WalEngine`).
//!
//! # On-Disk Frame Layout
//! ```text
//! +-----------+---------+-------------+---------+---------+-------------------+
//! | magic(4B) | ver(1B) | len(u32 LE) | lsn(8B) | crc(4B) | payload(len B)    |
//! +-----------+---------+-------------+---------+---------+-------------------+
//! |<--------------------- 21-byte header ---------------->|<---- payload ---->|
//! ```
//! - `magic` = `b"AWAL"`, `version` = `1`.
//! - `crc32` = IEEE CRC32 (`crc32fast`) over `[version, payload_len_le, lsn_le, payload]`.
//!
//! # Crash-Recovery & Concurrency Invariants
//! 1. **Monotonic LSN Ordering:** [`CoupledWal::append`] acquires `inner` before
//!    assigning the next LSN and writing the frame, guaranteeing that on-disk
//!    frame order strictly matches ascending LSN order even under concurrent callers.
//! 2. **Strict `fsync` Barrier:** [`CoupledWal::sync`] calls `flush()` followed by
//!    `sync_all()` (`fsync` / `FlushFileBuffers`).
//! 3. **Torn-Tail Truncation vs. Mid-Log Corruption:**
//!    - Incomplete headers (`< 21` bytes at EOF), zero-padded OS crash tails,
//!      truncated payloads at EOF, or tail frames failing CRC32 right at EOF are
//!      truncated cleanly to the last intact frame boundary and `fsync`-ed.
//!    - Non-tail CRC32 mismatches, invalid magic/version headers, non-monotonic
//!      LSNs, or mid-stream corrupted lengths return [`AerisError::WalCorruption`].
//! 4. **Atomic WAL Compaction:** [`CoupledWal::truncate_before`] writes retained
//!    frames to a sibling temporary file, calls `sync_all()`, and atomically renames
//!    it over the active WAL so a crash mid-compaction never loses committed frames.

use crate::codec::{
    compute_frame_crc32, decode_wal_op, encode_wal_frame, WalFrameHeader, MAX_WAL_PAYLOAD_LEN,
    WAL_FORMAT_VERSION, WAL_HEADER_LEN, WAL_MAGIC,
};
use aeris_types::{AerisError, AerisResult, Lsn, WalEngine, WalOp, WalRecord};
use parking_lot::Mutex;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

/// Maximum number of bytes scanned ahead when checking whether a corrupted frame
/// is followed by intact records (mid-log corruption vs. torn tail).
const CORRUPTION_LOOKAHEAD_CAP_BYTES: usize = 4 * 1024 * 1024;

/// Internal mutable file state guarded by [`CoupledWal::inner`].
struct WalInner {
    file: Option<File>,
    write_offset: u64,
    truncated_tail_bytes: u64,
}

impl WalInner {
    fn file_mut(&mut self) -> AerisResult<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            AerisError::WalIoError("WAL file handle is closed after failed compaction".to_string())
        })
    }
}

/// Summary produced by scanning and validating a WAL file on open or replay.
struct ScanOutcome {
    records: Vec<WalRecord>,
    valid_end: u64,
    last_valid_lsn: Lsn,
    max_checkpoint_lsn: Lsn,
    truncated_tail_bytes: u64,
}

/// Append-only, CRC32-framed, `fsync`-ordered Write-Ahead Log implementing [`WalEngine`].
///
/// # Lock Ordering
/// `CoupledWal` holds a single internal [`Mutex`] (`inner`). When wrapped by
/// [`crate::engine::CoupledStorageEngine`], the engine's `mutation_lock` is
/// always acquired before `CoupledWal::inner`.
pub struct CoupledWal {
    path: PathBuf,
    inner: Mutex<WalInner>,
    next_lsn: AtomicU64,
    checkpoint_lsn: AtomicU64,
}

impl CoupledWal {
    /// Opens an existing WAL file or creates a new one at `path`.
    ///
    /// Performs a full CRC32 validation pass over any existing frames, cleanly
    /// truncating and `fsync`-ing a torn tail if present, and restores `next_lsn`
    /// and `checkpoint_lsn` from durable disk state.
    ///
    /// # Errors
    /// Returns [`AerisError::WalCorruption`] on mid-log corruption or
    /// [`AerisError::WalIoError`] on filesystem errors.
    pub fn open(path: impl AsRef<Path>) -> AerisResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        if let Some(parent) = path_buf.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| {
                    AerisError::WalIoError(format!(
                        "failed to create WAL directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }

        cleanup_stale_compaction_tmp(&path_buf);

        let mut file = open_wal_file(&path_buf)?;
        let outcome = scan_and_recover_file(&mut file, &path_buf)?;

        let next_lsn = outcome.last_valid_lsn.saturating_add(1).max(1);
        info!(
            path = %path_buf.display(),
            records = outcome.records.len(),
            last_lsn = outcome.last_valid_lsn,
            checkpoint_lsn = outcome.max_checkpoint_lsn,
            truncated_tail_bytes = outcome.truncated_tail_bytes,
            "opened coupled WAL"
        );

        Ok(Self {
            path: path_buf,
            inner: Mutex::new(WalInner {
                file: Some(file),
                write_offset: outcome.valid_end,
                truncated_tail_bytes: outcome.truncated_tail_bytes,
            }),
            next_lsn: AtomicU64::new(next_lsn),
            checkpoint_lsn: AtomicU64::new(outcome.max_checkpoint_lsn),
        })
    }

    /// Returns the filesystem path of this WAL file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the highest durable `up_to_lsn` recorded via [`WalOp::Checkpoint`],
    /// or `0` if no checkpoint has been recorded yet.
    #[must_use]
    pub fn checkpoint_lsn(&self) -> Lsn {
        self.checkpoint_lsn.load(Ordering::Acquire)
    }

    /// Returns the number of torn-tail bytes truncated during the most recent
    /// open or replay recovery scan.
    #[must_use]
    pub fn truncated_tail_bytes(&self) -> u64 {
        self.inner.lock().truncated_tail_bytes
    }

    /// Returns the current valid byte size of the WAL file.
    #[must_use]
    pub fn file_size_bytes(&self) -> u64 {
        self.inner.lock().write_offset
    }

    /// Atomically compacts the WAL by discarding records with `lsn <= up_to_lsn`,
    /// returning the number of disk bytes reclaimed.
    ///
    /// Retains a durable [`WalOp::Checkpoint`] anchor if necessary so `next_lsn`
    /// and `checkpoint_lsn` remain strictly monotonic across subsequent restarts.
    /// Writes to a sibling temporary file, calls `sync_all()`, and atomically
    /// replaces the active WAL file so a crash mid-compaction cannot lose data.
    ///
    /// # Errors
    /// Returns [`AerisError::WalCorruption`] if the existing WAL is corrupt or
    /// [`AerisError::WalIoError`] on filesystem errors.
    pub fn truncate_before(&self, up_to_lsn: Lsn) -> AerisResult<u64> {
        let mut inner = self.inner.lock();
        let old_size = inner.write_offset;
        let outcome = {
            let file = inner.file_mut()?;
            scan_and_recover_file(file, &self.path)?
        };

        let retained = select_records_after_truncation(
            outcome.records,
            up_to_lsn,
            outcome.last_valid_lsn,
            self.checkpoint_lsn.load(Ordering::Acquire),
        );

        let tmp_path = compaction_tmp_path(&self.path);
        let new_size = write_compacted_tmp_file(&tmp_path, &retained)?;

        // Close active WAL handle before atomic rename (required on Windows to avoid
        // ERROR_SHARING_VIOLATION while replacing an open file).
        inner.file.take();
        if let Err(rename_err) = fs::rename(&tmp_path, &self.path) {
            // Reopen the untouched original WAL before returning the error.
            let _ = fs::remove_file(&tmp_path);
            if let Ok(reopened) = open_wal_file(&self.path) {
                inner.file = Some(reopened);
            }
            error!(
                path = %self.path.display(),
                error = %rename_err,
                "atomic WAL rename failed during truncate_before"
            );
            return Err(AerisError::WalIoError(format!(
                "failed to atomically replace WAL '{}': {rename_err}",
                self.path.display()
            )));
        }

        sync_parent_dir(&self.path);

        let mut reopened = open_wal_file(&self.path)?;
        reopened.seek(SeekFrom::Start(new_size)).map_err(|err| {
            AerisError::WalIoError(format!("WAL seek after compaction failed: {err}"))
        })?;
        inner.file = Some(reopened);
        inner.write_offset = new_size;

        let reclaimed = old_size.saturating_sub(new_size);
        info!(
            path = %self.path.display(),
            up_to_lsn,
            retained_records = retained.len(),
            old_bytes = old_size,
            new_bytes = new_size,
            reclaimed_bytes = reclaimed,
            "compacted coupled WAL"
        );
        Ok(reclaimed)
    }
}

impl WalEngine for CoupledWal {
    fn append(&self, op: WalOp) -> AerisResult<Lsn> {
        let mut inner = self.inner.lock();
        let lsn = self.next_lsn.load(Ordering::Relaxed);
        let (frame, _crc) = encode_wal_frame(lsn, &op)?;
        let write_offset = inner.write_offset;

        let file = inner.file_mut()?;
        file.seek(SeekFrom::Start(write_offset)).map_err(|err| {
            AerisError::WalIoError(format!("WAL seek to {write_offset} failed: {err}"))
        })?;

        if let Err(err) = file.write_all(&frame) {
            // Best-effort rollback of a partial write so live state never leaves a torn tail.
            let _ = file.set_len(write_offset);
            error!(
                path = %self.path.display(),
                lsn,
                error = %err,
                "WAL frame write failed"
            );
            return Err(AerisError::WalIoError(format!(
                "WAL write at LSN {lsn} failed: {err}"
            )));
        }

        inner.write_offset = write_offset + frame.len() as u64;
        self.next_lsn.store(lsn + 1, Ordering::Release);
        if let WalOp::Checkpoint { up_to_lsn } = op {
            self.checkpoint_lsn.fetch_max(up_to_lsn, Ordering::AcqRel);
        }
        debug!(lsn, frame_bytes = frame.len(), "appended WAL frame");
        Ok(lsn)
    }

    fn sync(&self) -> AerisResult<()> {
        let mut inner = self.inner.lock();
        let file = inner.file_mut()?;
        file.flush().map_err(|err| {
            error!(path = %self.path.display(), error = %err, "WAL flush failed");
            AerisError::WalIoError(format!("WAL flush failed: {err}"))
        })?;
        file.sync_all().map_err(|err| {
            error!(path = %self.path.display(), error = %err, "WAL sync_all failed");
            AerisError::WalIoError(format!("WAL sync_all (fsync) failed: {err}"))
        })?;
        Ok(())
    }

    fn last_lsn(&self) -> Lsn {
        self.next_lsn.load(Ordering::Acquire).saturating_sub(1)
    }

    fn replay_from(&self, start_lsn: Lsn) -> AerisResult<Vec<WalRecord>> {
        let mut inner = self.inner.lock();
        let outcome = {
            let file = inner.file_mut()?;
            scan_and_recover_file(file, &self.path)?
        };
        inner.write_offset = outcome.valid_end;
        if outcome.truncated_tail_bytes > 0 {
            inner.truncated_tail_bytes = outcome.truncated_tail_bytes;
        }
        if outcome.last_valid_lsn >= self.next_lsn.load(Ordering::Acquire) {
            self.next_lsn
                .store(outcome.last_valid_lsn.saturating_add(1), Ordering::Release);
        }
        self.checkpoint_lsn
            .fetch_max(outcome.max_checkpoint_lsn, Ordering::AcqRel);

        let filtered: Vec<WalRecord> = outcome
            .records
            .into_iter()
            .filter(|rec| rec.lsn >= start_lsn)
            .collect();
        debug!(
            path = %self.path.display(),
            start_lsn,
            returned_records = filtered.len(),
            "replayed WAL records"
        );
        Ok(filtered)
    }

    fn checkpoint(&self, up_to_lsn: Lsn) -> AerisResult<()> {
        let lsn = self.append(WalOp::Checkpoint { up_to_lsn })?;
        self.sync()?;
        info!(
            path = %self.path.display(),
            checkpoint_record_lsn = lsn,
            up_to_lsn,
            "persisted durable WAL checkpoint"
        );
        Ok(())
    }
}

/// Opens (or creates) the WAL file with read/write access without `O_APPEND`
/// so explicit seeks and torn-tail truncation work uniformly on Windows and Unix.
fn open_wal_file(path: &Path) -> AerisResult<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to open WAL file '{}': {err}",
                path.display()
            ))
        })
}

/// Scans `file` from offset 0 validating magic, version, length bounds, CRC32,
/// and LSN monotonicity. Truncates and `fsync`s any torn tail cleanly while
/// returning [`AerisError::WalCorruption`] on mid-log corruption.
fn scan_and_recover_file(file: &mut File, path: &Path) -> AerisResult<ScanOutcome> {
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
        let remaining = file_len - offset;
        if remaining < WAL_HEADER_LEN as u64 {
            truncated_tail_bytes = truncate_torn_tail(
                file,
                path,
                offset,
                file_len,
                last_valid_lsn,
                "incomplete WAL header at EOF",
            )?;
            break;
        }

        let mut header_buf = [0u8; WAL_HEADER_LEN];
        read_exact_at(file, offset, &mut header_buf)?;

        if header_buf[0..4] != WAL_MAGIC {
            if is_all_zero_tail(file, offset, file_len)? {
                truncated_tail_bytes = truncate_torn_tail(
                    file,
                    path,
                    offset,
                    file_len,
                    last_valid_lsn,
                    "zero-filled torn tail at EOF",
                )?;
                break;
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

        let header =
            WalFrameHeader::decode(&header_buf).map_err(|err| AerisError::WalCorruption {
                lsn: 0,
                reason: format!("header decode failed at byte offset {offset}: {err}"),
            })?;

        let frame_len = (WAL_HEADER_LEN as u64) + (header.payload_len as u64);
        let frame_end = offset.saturating_add(frame_len);

        if frame_end > file_len {
            // Distinguish a genuine torn tail mid-payload from a corrupted non-tail length.
            if has_valid_frame_after(file, offset + WAL_HEADER_LEN as u64, file_len)? {
                return Err(AerisError::WalCorruption {
                    lsn: header.lsn,
                    reason: format!(
                        "corrupt payload_len {} at non-tail byte offset {offset}",
                        header.payload_len
                    ),
                });
            }
            truncated_tail_bytes = truncate_torn_tail(
                file,
                path,
                offset,
                file_len,
                last_valid_lsn,
                "truncated WAL payload at EOF",
            )?;
            break;
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

        let actual_crc =
            compute_frame_crc32(header.version, header.payload_len, header.lsn, &payload);
        if actual_crc != header.crc32 {
            if frame_end == file_len {
                truncated_tail_bytes = truncate_torn_tail(
                    file,
                    path,
                    offset,
                    file_len,
                    last_valid_lsn,
                    "tail WAL frame failed CRC32 verification at EOF",
                )?;
                break;
            }
            return Err(AerisError::WalCorruption {
                lsn: header.lsn,
                reason: format!(
                    "CRC32 mismatch at non-tail byte offset {offset}: expected 0x{:08X}, computed 0x{:08X}",
                    header.crc32, actual_crc
                ),
            });
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

        if let WalOp::Checkpoint { up_to_lsn } = &op {
            max_checkpoint_lsn = max_checkpoint_lsn.max(*up_to_lsn);
        }

        records.push(WalRecord {
            lsn: header.lsn,
            checksum: header.crc32,
            op,
        });
        offset = frame_end;
        last_valid_lsn = header.lsn;
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
fn is_all_zero_tail(file: &mut File, offset: u64, file_len: u64) -> AerisResult<bool> {
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
fn has_valid_frame_after(file: &mut File, search_start: u64, file_len: u64) -> AerisResult<bool> {
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

/// Selects frames retained after `truncate_before(up_to_lsn)`, preserving a
/// durable checkpoint anchor so LSNs and `checkpoint_lsn` never regress.
fn select_records_after_truncation(
    records: Vec<WalRecord>,
    up_to_lsn: Lsn,
    last_valid_lsn: Lsn,
    known_checkpoint_lsn: Lsn,
) -> Vec<(Lsn, WalOp)> {
    let mut retained: Vec<(Lsn, WalOp)> = records
        .into_iter()
        .filter(|rec| rec.lsn > up_to_lsn)
        .map(|rec| (rec.lsn, rec.op))
        .collect();

    let effective_cp = up_to_lsn.max(known_checkpoint_lsn);
    let has_cp_at_or_above = retained
        .iter()
        .any(|(_, op)| matches!(op, WalOp::Checkpoint { up_to_lsn: cp } if *cp >= effective_cp));

    if retained.is_empty() && last_valid_lsn > 0 {
        retained.push((
            last_valid_lsn,
            WalOp::Checkpoint {
                up_to_lsn: effective_cp.min(last_valid_lsn),
            },
        ));
    } else if !has_cp_at_or_above && effective_cp > 0 {
        if let Some(&(first_lsn, _)) = retained.first() {
            if up_to_lsn >= 1 && up_to_lsn < first_lsn {
                retained.insert(
                    0,
                    (
                        up_to_lsn,
                        WalOp::Checkpoint {
                            up_to_lsn: effective_cp,
                        },
                    ),
                );
            }
        }
    }
    retained
}

/// Writes `records` into `tmp_path` and calls `flush()` + `sync_all()`.
fn write_compacted_tmp_file(tmp_path: &Path, records: &[(Lsn, WalOp)]) -> AerisResult<u64> {
    let mut tmp_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp_path)
        .map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to create compaction temp file '{}': {err}",
                tmp_path.display()
            ))
        })?;

    let mut total_bytes = 0u64;
    for (lsn, op) in records {
        let (frame, _crc) = encode_wal_frame(*lsn, op)?;
        tmp_file.write_all(&frame).map_err(|err| {
            AerisError::WalIoError(format!(
                "failed writing compacted WAL frame at LSN {lsn}: {err}"
            ))
        })?;
        total_bytes += frame.len() as u64;
    }

    tmp_file
        .flush()
        .map_err(|err| AerisError::WalIoError(format!("compacted WAL flush failed: {err}")))?;
    tmp_file
        .sync_all()
        .map_err(|err| AerisError::WalIoError(format!("compacted WAL sync_all failed: {err}")))?;
    Ok(total_bytes)
}

fn compaction_tmp_path(wal_path: &Path) -> PathBuf {
    let mut os = wal_path.as_os_str().to_os_string();
    os.push(".compact.tmp");
    PathBuf::from(os)
}

fn cleanup_stale_compaction_tmp(wal_path: &Path) {
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

#[cfg(unix)]
fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) {
    // Windows directory handles cannot be synced via File::open; MoveFileExW
    // metadata durability is handled by NTFS journal commits.
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeris_types::{Fingerprint, PartitionHasher};
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
}
