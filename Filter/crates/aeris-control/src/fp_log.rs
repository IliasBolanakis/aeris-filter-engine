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

use aeris_core::AerisFilter;
use aeris_types::{
    AerisError, AerisResult, FpLogEntry, QueryRange, RangeFilter, ReverseMapStore,
    MAX_QUOTIENT_BITS,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info, warn};

/// 4-byte binary frame magic header for [`FalsePositiveLog`] records.
pub const FP_LOG_MAGIC: [u8; 4] = *b"AFPL";

/// Fixed on-disk byte length of a single encoded [`FpLogEntry`] frame.
pub const FP_LOG_FRAME_LEN: usize = 40;

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

/// Internal mutable state protected by [`FalsePositiveLog::inner`].
struct FpLogInner {
    file: Option<File>,
    entries: VecDeque<FpLogEntry>,
    dedup: HashSet<QueryRange>,
    next_seq: u64,
    write_offset: u64,
    truncated_tail_bytes: u64,
    compactions: u64,
}

impl FpLogInner {
    fn file_mut(&mut self) -> AerisResult<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            AerisError::WalIoError(
                "FP log file handle is closed after failed compaction".to_string(),
            )
        })
    }
}

/// Summary produced when scanning an on-disk FP log file during recovery.
struct ScanOutcome {
    entries: Vec<FpLogEntry>,
    valid_end: u64,
    last_seq: u64,
    truncated_tail_bytes: u64,
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
        if let Some(parent) = path_buf.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| {
                    AerisError::WalIoError(format!(
                        "failed to create FP log directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }

        cleanup_stale_compaction_tmp(&path_buf);

        let mut file = open_fp_log_file(&path_buf)?;
        let scan = scan_and_recover_file(&mut file, &path_buf)?;

        let mut entries = VecDeque::with_capacity(config.max_entries.min(scan.entries.len() + 1));
        let mut dedup = HashSet::with_capacity(config.max_entries.min(scan.entries.len() + 1));
        let mut needs_compaction = false;

        for entry in scan.entries {
            if dedup.insert(entry.range) {
                entries.push_back(entry);
            } else {
                needs_compaction = true;
            }
        }

        while entries.len() > config.max_entries {
            if let Some(evicted) = entries.pop_front() {
                dedup.remove(&evicted.range);
                needs_compaction = true;
            }
        }

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

        let live_entries = log.len();
        info!(
            path = %path_buf.display(),
            live_entries,
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
        let frame = encode_fp_frame(&entry);
        let write_offset = inner.write_offset;

        {
            let file = inner.file_mut()?;
            file.seek(SeekFrom::Start(write_offset)).map_err(|err| {
                AerisError::WalIoError(format!("FP log seek to {write_offset} failed: {err}"))
            })?;
            if let Err(err) = file.write_all(&frame) {
                let _ = file.set_len(write_offset);
                error!(
                    path = %self.path.display(),
                    seq,
                    error = %err,
                    "FP log frame write failed"
                );
                return Err(AerisError::WalIoError(format!(
                    "FP log write at seq {seq} failed: {err}"
                )));
            }
            if self.config.sync_on_append {
                file.flush()
                    .map_err(|err| AerisError::WalIoError(format!("FP log flush failed: {err}")))?;
                file.sync_all().map_err(|err| {
                    AerisError::WalIoError(format!("FP log sync_all failed: {err}"))
                })?;
            }
        }

        inner.write_offset = write_offset + (FP_LOG_FRAME_LEN as u64);
        inner.next_seq = seq.saturating_add(1);
        inner.entries.push_back(entry);
        inner.dedup.insert(range);

        if inner.entries.len() > self.config.max_entries {
            while inner.entries.len() > self.config.max_entries {
                if let Some(evicted) = inner.entries.pop_front() {
                    inner.dedup.remove(&evicted.range);
                }
            }
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

/// Atomically rewrites the active FP log file with only the retained `inner.entries`.
///
/// Drops `inner.file` prior to `fs::rename` so Windows does not fail with
/// `ERROR_SHARING_VIOLATION` while replacing an open file.
fn compact_locked(path: &Path, inner: &mut FpLogInner) -> AerisResult<()> {
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

/// Opens (or creates) the FP log file in read/write mode without `O_APPEND`
/// so explicit seeks and torn-tail truncation behave identically on Windows and Unix.
fn open_fp_log_file(path: &Path) -> AerisResult<File> {
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
fn scan_and_recover_file(file: &mut File, path: &Path) -> AerisResult<ScanOutcome> {
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
        let remaining = raw_bytes.len() - offset;
        if remaining < FP_LOG_FRAME_LEN {
            return truncate_torn_tail(file, path, offset as u64, file_len, entries, last_seq);
        }

        let frame: [u8; FP_LOG_FRAME_LEN] = raw_bytes[offset..offset + FP_LOG_FRAME_LEN]
            .try_into()
            .expect("exact 40-byte slice");

        match decode_fp_frame(&frame) {
            Ok(entry) if entry.seq > last_seq => {
                last_seq = entry.seq;
                entries.push(entry);
                offset += FP_LOG_FRAME_LEN;
            }
            Ok(entry) => {
                // Non-monotonic sequence number indicates corruption.
                let trailing = &raw_bytes[offset + FP_LOG_FRAME_LEN..];
                if trailing_has_intact_frame(trailing, last_seq) {
                    return Err(AerisError::WalCorruption {
                        lsn: entry.seq,
                        reason: format!(
                            "non-monotonic FP log seq {} after {} at offset {offset}",
                            entry.seq, last_seq
                        ),
                    });
                }
                return truncate_torn_tail(file, path, offset as u64, file_len, entries, last_seq);
            }
            Err(err) => {
                let trailing = &raw_bytes[offset + FP_LOG_FRAME_LEN..];
                if trailing_has_intact_frame(trailing, last_seq) {
                    return Err(err);
                }
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

/// Checks whether `trailing` contains at least one intact valid frame after a
/// corrupted offset (distinguishing mid-log corruption from a torn tail).
fn trailing_has_intact_frame(trailing: &[u8], last_seq: u64) -> bool {
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

/// Computes the sibling temporary path used during atomic bounded rotation.
fn compaction_tmp_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".compact.tmp");
    PathBuf::from(os)
}

/// Cleans up any leftover `.compact.tmp` file from a prior crash mid-rotation.
fn cleanup_stale_compaction_tmp(path: &Path) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use aeris_storage::{ReverseMapConfig, RocksDbReverseMap};
    use aeris_types::FilterConfig;
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
        let rm =
            RocksDbReverseMap::open(dir.path().join("rm"), ReverseMapConfig::default()).unwrap();
        let log =
            FalsePositiveLog::open(dir.path().join("fp.log"), FpLogConfig::default()).unwrap();

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
}
