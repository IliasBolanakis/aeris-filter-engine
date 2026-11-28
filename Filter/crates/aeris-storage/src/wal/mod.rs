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

mod recovery;

#[cfg(test)]
mod tests;

use self::recovery::{cleanup_stale_compaction_tmp, compaction_tmp_path, scan_and_recover_file};
use crate::codec::encode_wal_frame;
use aeris_types::{AerisError, AerisResult, Lsn, WalEngine, WalOp, WalRecord};
use parking_lot::Mutex;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info};

/// Internal mutable file state guarded by [`CoupledWal::inner`].
pub(super) struct WalInner {
    pub(super) file: Option<File>,
    pub(super) write_offset: u64,
    pub(super) truncated_tail_bytes: u64,
}

impl WalInner {
    pub(super) fn file_mut(&mut self) -> AerisResult<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            AerisError::WalIoError("WAL file handle is closed after failed compaction".to_string())
        })
    }
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

/// Selects frames retained after `truncate_before(up_to_lsn)`, preserving a
/// durable checkpoint anchor so LSNs and `checkpoint_lsn` never regress.
pub(super) fn select_records_after_truncation(
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
