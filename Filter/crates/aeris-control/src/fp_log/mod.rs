//! Persistent append-only, bounded False-Positive Log and monotonic replay engine.
//!
//! # On-Disk Binary Frame Layout (`40` bytes / entry)
//! ```text
//! +------------+-----------+------------+----------+-----------------+-----------+
//! | magic (4B) | seq (8B)  | start (8B) | end (8B) | timestamp_ms(8B)| crc32(4B) |
//! +------------+-----------+------------+----------+-----------------+-----------+
//! |<-- b"AFPL" ->|<----------- 32-byte CRC32-covered payload ------->|<-- LE --->|
//! ```
//! - `magic`: fixed 4-byte marker `b"AFPL"`.
//! - `seq`, `start`, `end`, `timestamp_ms`: little-endian `u64` fields.
//! - `crc32`: little-endian IEEE CRC32 (`crc32fast`) over bytes `[4..36]`.
//!
//! # Crash-Recovery Invariants
//! 1. **Per-Frame Checksum Validation:** Every frame read during [`FalsePositiveLog::open`]
//!    validates `magic == b"AFPL"`, IEEE CRC32 over `seq..timestamp_ms`, `start <= end`,
//!    and strictly ascending `seq`.
//! 2. **Torn-Tail Truncation vs. Mid-Log Corruption:** Incomplete tail frames (`< 40` bytes
//!    at EOF), zero-padded OS crash tails, or a corrupted trailing frame with no valid
//!    successor frames are treated as interrupted crash writes, truncated cleanly to the
//!    last intact frame boundary, and `fsync`-ed with a structured `tracing::warn!`.
//!    Corrupted frames followed by intact valid frames return [`AerisError::WalCorruption`].
//! 3. **Atomic Bounded Rotation (`max_entries`):** Whenever deduplicated live entries exceed
//!    `config.max_entries`, the oldest entries are evicted and retained frames are written
//!    to a sibling `.compact.tmp` file, `fsync`-ed (`flush` + `sync_all`), and atomically
//!    renamed over the active log file (closing the active handle prior to rename for
//!    Windows compatibility).
//! 4. **Lock Ordering:** `FalsePositiveLog::inner` is strictly a leaf lock and is NEVER held
//!    while calling [`AerisFilter`], [`ReverseMapStore`], or caller `verify_empty` closures.

mod frame;
mod recovery;

#[cfg(test)]
mod tests;

pub use frame::{decode_fp_frame, encode_fp_frame, FP_LOG_FRAME_LEN, FP_LOG_MAGIC};

use self::recovery::{
    cleanup_stale_compaction_tmp, compact_locked, ensure_parent_dir, open_fp_log_file,
    scan_and_recover_file,
};
use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, FpLogEntry, QueryRange, RangeFilter, ReverseMapStore,
    MAX_QUOTIENT_BITS,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info};

/// Default maximum number of deduplicated false-positive intervals retained in the log.
pub const DEFAULT_FP_LOG_MAX_ENTRIES: usize = 65_536;

/// Configuration for [`FalsePositiveLog`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpLogConfig {
    /// Maximum number of deduplicated entries retained before atomic FIFO rotation.
    pub max_entries: usize,
    /// When `true`, every appended frame invokes `flush()` + `sync_all()` (`fsync`).
    pub sync_on_append: bool,
}

impl Default for FpLogConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_FP_LOG_MAX_ENTRIES,
            sync_on_append: true,
        }
    }
}

impl FpLogConfig {
    /// Validates [`FpLogConfig`] bounds (`max_entries >= 1`).
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `max_entries == 0`.
    pub fn validate(&self) -> AerisResult<()> {
        if self.max_entries == 0 {
            return Err(AerisError::InvalidConfig(
                "FpLogConfig::max_entries must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Diagnostic summary returned by [`FalsePositiveLog::replay_onto`] and
/// [`FalsePositiveLog::replay_since`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpReplayReport {
    /// Total log entries inspected during replay.
    pub replayed_entries: usize,
    /// Number of collisions detected on `F_new` and monotonically resolved via `adapt`.
    pub collisions_resolved: usize,
    /// Number of automatic `2x` expansions triggered by `FilterFull` during replay.
    pub expansions_during_replay: usize,
    /// Number of entries skipped because `verify_empty` reported the range is now non-empty.
    pub skipped_non_empty: usize,
    /// Highest `FpLogEntry::seq` inspected during this replay pass (`0` if none).
    pub last_replayed_seq: u64,
    /// Wall-clock duration of the replay pass in nanoseconds.
    pub latency_ns: u64,
}

/// Internal mutable state protected by [`FalsePositiveLog::inner`].
pub(super) struct FpLogInner {
    pub(super) file: Option<File>,
    pub(super) entries: VecDeque<FpLogEntry>,
    pub(super) dedup: HashSet<QueryRange>,
    pub(super) next_seq: u64,
    pub(super) write_offset: u64,
    pub(super) truncated_tail_bytes: u64,
    pub(super) compactions: u64,
}

impl FpLogInner {
    pub(super) fn file_mut(&mut self) -> AerisResult<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            AerisError::WalIoError(
                "FP log file handle is closed after failed compaction".to_string(),
            )
        })
    }
}

/// Persistent, bounded, deduplicating False-Positive Log guaranteeing 100%
/// monotonic false-positive elimination across dynamic `R` rebuilds (`F_old -> F_new`).
///
/// Thread-safe (`Send + Sync`): foreground query threads may concurrently call
/// [`FalsePositiveLog::record`] while background rebuild threads snapshot and replay
/// entries via [`FalsePositiveLog::replay_onto`] / [`FalsePositiveLog::replay_since`].
pub struct FalsePositiveLog {
    path: PathBuf,
    config: FpLogConfig,
    inner: Mutex<FpLogInner>,
}

impl FalsePositiveLog {
    /// Opens an existing [`FalsePositiveLog`] at `path` or creates a new one.
    ///
    /// Validates every frame via CRC32, cleanly truncates and `fsync`s any torn tail
    /// left by a crash, restores the in-memory deduplication mirror, and compacts the
    /// log if existing intact entries exceed `config.max_entries`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `config` is invalid,
    /// [`AerisError::WalCorruption`] on mid-log corruption, or
    /// [`AerisError::WalIoError`] on filesystem failure.
    pub fn open(path: impl AsRef<Path>, config: FpLogConfig) -> AerisResult<Self> {
        config.validate()?;
        let path_buf = path.as_ref().to_path_buf();
        ensure_parent_dir(&path_buf)?;
        cleanup_stale_compaction_tmp(&path_buf);

        let mut file = open_fp_log_file(&path_buf)?;
        let scan = scan_and_recover_file(&mut file, &path_buf)?;
        let (entries, dedup, needs_compaction) =
            deduplicate_and_bound_entries(scan.entries, config.max_entries);

        let next_seq = scan.last_seq.saturating_add(1).max(1);
        let log = Self {
            path: path_buf.clone(),
            config,
            inner: Mutex::new(FpLogInner {
                file: Some(file),
                entries,
                dedup,
                next_seq,
                write_offset: scan.valid_end,
                truncated_tail_bytes: scan.truncated_tail_bytes,
                compactions: 0,
            }),
        };

        if needs_compaction {
            let mut inner = log.inner.lock();
            compact_locked(&log.path, &mut inner)?;
        }

        info!(
            path = %path_buf.display(),
            live_entries = log.len(),
            last_seq = scan.last_seq,
            truncated_tail_bytes = scan.truncated_tail_bytes,
            max_entries = config.max_entries,
            "opened FalsePositiveLog"
        );
        Ok(log)
    }

    /// Returns the filesystem path of this log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the active [`FpLogConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> FpLogConfig {
        self.config
    }

    /// Returns the number of live deduplicated entries currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }

    /// Returns `true` if the log contains zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().entries.is_empty()
    }

    /// Returns the highest sequence number recorded so far (`0` if empty).
    #[must_use]
    pub fn last_seq(&self) -> u64 {
        self.inner.lock().next_seq.saturating_sub(1)
    }

    /// Returns the number of torn-tail bytes truncated during recovery open.
    #[must_use]
    pub fn truncated_tail_bytes(&self) -> u64 {
        self.inner.lock().truncated_tail_bytes
    }

    /// Returns the number of atomic bounded rotations/compactions performed.
    #[must_use]
    pub fn compactions(&self) -> u64 {
        self.inner.lock().compactions
    }

    /// Returns `true` if `range` is already present in the in-memory deduplication set.
    #[must_use]
    pub fn contains_range(&self, range: QueryRange) -> bool {
        self.inner.lock().dedup.contains(&range)
    }

    /// Records an observed false-positive empty interval `range` with `timestamp_ms`.
    ///
    /// If `range` is already present in the active log, deduplicates in `O(1)` RAM
    /// without performing disk I/O and returns `Ok(None)`. Otherwise appends a
    /// checksummed 40-byte frame (syncing when `config.sync_on_append` is `true`)
    /// and atomically rotates the file if `len > config.max_entries`.
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`] on filesystem write, sync, or rotation failure.
    pub fn record(&self, range: QueryRange, timestamp_ms: u64) -> AerisResult<Option<FpLogEntry>> {
        let mut inner = self.inner.lock();
        if inner.dedup.contains(&range) {
            return Ok(None);
        }

        let seq = inner.next_seq;
        let entry = FpLogEntry {
            seq,
            range,
            timestamp_ms,
        };
        write_and_sync_frame(&self.path, &mut inner, &entry, self.config.sync_on_append)?;

        inner.write_offset += FP_LOG_FRAME_LEN as u64;
        inner.next_seq = seq.saturating_add(1);
        inner.entries.push_back(entry);
        inner.dedup.insert(range);

        if inner.entries.len() > self.config.max_entries {
            evict_excess_entries(&mut inner, self.config.max_entries);
            compact_locked(&self.path, &mut inner)?;
        }

        debug!(
            seq,
            range_start = range.start(),
            range_end = range.end(),
            live_entries = inner.entries.len(),
            "recorded false-positive interval"
        );
        Ok(Some(entry))
    }

    /// Flushes OS buffers for the underlying log file to stable storage (`fsync`).
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`] if flush or `sync_all` fails.
    pub fn sync(&self) -> AerisResult<()> {
        let mut inner = self.inner.lock();
        let file = inner.file_mut()?;
        file.flush()
            .map_err(|err| AerisError::WalIoError(format!("FP log flush failed: {err}")))?;
        file.sync_all()
            .map_err(|err| AerisError::WalIoError(format!("FP log sync_all failed: {err}")))?;
        Ok(())
    }

    /// Captures a consistent snapshot `(last_seq, entries)` of all retained entries
    /// in ascending sequence order.
    #[must_use]
    pub fn snapshot_entries(&self) -> (u64, Vec<FpLogEntry>) {
        let inner = self.inner.lock();
        let last_seq = inner.next_seq.saturating_sub(1);
        let entries: Vec<FpLogEntry> = inner.entries.iter().copied().collect();
        (last_seq, entries)
    }

    /// Captures all retained entries with `entry.seq > min_seq_exclusive`, returning
    /// `(last_seq, entries)` in ascending sequence order.
    #[must_use]
    pub fn entries_since(&self, min_seq_exclusive: u64) -> (u64, Vec<FpLogEntry>) {
        let inner = self.inner.lock();
        let last_seq = inner.next_seq.saturating_sub(1);
        let entries: Vec<FpLogEntry> = inner
            .entries
            .iter()
            .copied()
            .filter(|e| e.seq > min_seq_exclusive)
            .collect();
        (last_seq, entries)
    }

    /// Replays every retained entry in the log onto `new_filter` (`F_new`),
    /// monotonically eliminating every collision that still matches an empty range.
    ///
    /// Equivalent to [`Self::replay_since`] with `min_seq = 0`.
    ///
    /// # Errors
    /// Propagates errors from `verify_empty`, [`AerisFilter::adapt`], or
    /// [`AerisFilter::expand`].
    pub fn replay_onto<F>(
        &self,
        new_filter: &mut AerisFilter,
        reverse_map: &dyn ReverseMapStore,
        verify_empty: F,
    ) -> AerisResult<FpReplayReport>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        self.replay_since(0, new_filter, reverse_map, verify_empty)
    }

    /// Replays all retained entries with `entry.seq > min_seq` onto `new_filter`.
    ///
    /// For each entry `[a, b]`:
    /// 1. Checks `new_filter.query_range(entry.range)` (`O(1)` in RAM).
    /// 2. If `true`, invokes `verify_empty(entry.range)?` to confirm no key has
    ///    subsequently been inserted inside `[a, b]`.
    /// 3. If empty, calls `new_filter.adapt(entry.range, reverse_map)`; on
    ///    [`AerisError::FilterFull`], automatically expands `new_filter` and retries.
    ///
    /// # Lock Ordering
    /// Snapshots entries under `self.inner` and immediately drops the mutex before
    /// touching `new_filter`, `reverse_map`, or `verify_empty`.
    ///
    /// # Errors
    /// Propagates errors from `verify_empty`, [`AerisFilter::adapt`], or
    /// [`AerisFilter::expand`].
    pub fn replay_since<F>(
        &self,
        min_seq: u64,
        new_filter: &mut AerisFilter,
        reverse_map: &dyn ReverseMapStore,
        verify_empty: F,
    ) -> AerisResult<FpReplayReport>
    where
        F: Fn(QueryRange) -> AerisResult<bool>,
    {
        let start = Instant::now();
        let (_, entries) = self.entries_since(min_seq);
        let mut report = FpReplayReport::default();

        for entry in entries {
            report.replayed_entries += 1;
            report.last_replayed_seq = entry.seq;

            if !new_filter.query_range(entry.range) {
                continue;
            }
            if !verify_empty(entry.range)? {
                report.skipped_non_empty += 1;
                continue;
            }

            adapt_with_auto_expand(
                new_filter,
                entry.range,
                reverse_map,
                &mut report.expansions_during_replay,
            )?;
            report.collisions_resolved += 1;
        }

        report.latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            path = %self.path.display(),
            min_seq,
            replayed_entries = report.replayed_entries,
            collisions_resolved = report.collisions_resolved,
            expansions_during_replay = report.expansions_during_replay,
            skipped_non_empty = report.skipped_non_empty,
            latency_ns = report.latency_ns,
            "completed FalsePositiveLog replay"
        );
        Ok(report)
    }
}

fn deduplicate_and_bound_entries(
    scanned: Vec<FpLogEntry>,
    max_entries: usize,
) -> (VecDeque<FpLogEntry>, HashSet<QueryRange>, bool) {
    let cap = max_entries.min(scanned.len() + 1);
    let mut entries = VecDeque::with_capacity(cap);
    let mut dedup = HashSet::with_capacity(cap);
    let mut needs_compaction = false;

    for entry in scanned {
        if dedup.insert(entry.range) {
            entries.push_back(entry);
        } else {
            needs_compaction = true;
        }
    }

    while entries.len() > max_entries {
        if let Some(evicted) = entries.pop_front() {
            dedup.remove(&evicted.range);
            needs_compaction = true;
        }
    }

    (entries, dedup, needs_compaction)
}

fn write_and_sync_frame(
    path: &Path,
    inner: &mut FpLogInner,
    entry: &FpLogEntry,
    sync_on_append: bool,
) -> AerisResult<()> {
    let frame = encode_fp_frame(entry);
    let write_offset = inner.write_offset;
    let file = inner.file_mut()?;
    file.seek(SeekFrom::Start(write_offset)).map_err(|err| {
        AerisError::WalIoError(format!("FP log seek to {write_offset} failed: {err}"))
    })?;
    if let Err(err) = file.write_all(&frame) {
        let _ = file.set_len(write_offset);
        error!(
            path = %path.display(),
            seq = entry.seq,
            error = %err,
            "FP log frame write failed"
        );
        return Err(AerisError::WalIoError(format!(
            "FP log write at seq {} failed: {err}",
            entry.seq
        )));
    }
    if sync_on_append {
        file.flush()
            .map_err(|err| AerisError::WalIoError(format!("FP log flush failed: {err}")))?;
        file.sync_all()
            .map_err(|err| AerisError::WalIoError(format!("FP log sync_all failed: {err}")))?;
    }
    Ok(())
}

fn evict_excess_entries(inner: &mut FpLogInner, max_entries: usize) {
    while inner.entries.len() > max_entries {
        if let Some(evicted) = inner.entries.pop_front() {
            inner.dedup.remove(&evicted.range);
        }
    }
}

/// Adapts `filter` to `range`, automatically calling `filter.expand(reverse_map)`
/// and retrying whenever [`AerisError::FilterFull`] is encountered.
fn adapt_with_auto_expand(
    filter: &mut AerisFilter,
    range: QueryRange,
    reverse_map: &dyn ReverseMapStore,
    expansions_counter: &mut usize,
) -> AerisResult<()> {
    let mut local_expansions = 0u8;
    loop {
        match filter.adapt(range, reverse_map) {
            Ok(_) => return Ok(()),
            Err(AerisError::FilterFull { .. }) if local_expansions < MAX_QUOTIENT_BITS => {
                filter.expand(reverse_map)?;
                local_expansions += 1;
                *expansions_counter += 1;
            }
            Err(err) => return Err(err),
        }
    }
}
