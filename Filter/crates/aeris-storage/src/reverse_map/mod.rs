//! Write-optimized RocksDB Reverse Map (`ReverseMapStore`).
//!
//! Maps variable-length canonical keepsake box [`Fingerprint`]s to their full
//! 64-bit unsigned [`Key`] lists using a custom RocksDB `MergeOperator`
//! (`"aeris_reverse_map_keylist_v1"`).
//!
//! # Write-Optimization & Idempotency Guarantees
//! - Every `append_key`, `append_batch`, and `remove_key` emits a 9-byte merge
//!   operand (`AddKey` / `DelKey`) directly into RocksDB's LSM memtable with
//!   **zero read-modify-write disk I/O**.
//! - Every `tombstone` issues a native RocksDB delete marker so obsolete
//!   fingerprints produced by adaptation or rejuvenation are physically
//!   reclaimed during compaction.
//! - The `full_merge` callback folds all operands chronologically into a
//!   `BTreeSet<Key>` and writes a canonical `CompactedSnapshot`, guaranteeing
//!   **strict WAL replay idempotency** even if the same `AddKey` or `DelKey`
//!   operands are replayed multiple times after a crash.

mod iterator;

#[cfg(test)]
mod tests;

use self::iterator::ReverseMapKeyIter;
use crate::codec::{
    count_keys_in_value, decode_key_list, encode_add_key_operand, encode_del_key_operand,
    fold_full_merge, fold_partial_merge,
};
use aeris_types::{AerisError, AerisResult, Fingerprint, Key, ReverseMapStore};
use rocksdb::{
    DBCompressionType, IteratorMode, MergeOperands, Options, WriteBatch, WriteOptions, DB,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, error, info};

/// Registered RocksDB `MergeOperator` identifier for the Aeris Reverse Map.
pub const MERGE_OPERATOR_NAME: &str = "aeris_reverse_map_keylist_v1";

/// Default RocksDB memtable write buffer size in bytes (16 MiB).
pub const DEFAULT_WRITE_BUFFER_SIZE_BYTES: usize = 16 * 1024 * 1024;

/// Default maximum number of concurrent memtable write buffers.
pub const DEFAULT_MAX_WRITE_BUFFER_NUMBER: i32 = 3;

/// Configuration parameters for [`RocksDbReverseMap`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReverseMapConfig {
    /// Per-memtable write buffer size in bytes.
    pub write_buffer_size: usize,
    /// Maximum number of immutable + active memtables kept in RAM before stalling.
    pub max_write_buffer_number: i32,
    /// Enables LZ4 SSTable compression when `true`; disables compression when `false`.
    pub enable_lz4_compression: bool,
    /// When `true`, every RocksDB write calls `fsync` directly.
    /// Set to `false` when coupled under [`crate::engine::CoupledStorageEngine`],
    /// where [`crate::wal::CoupledWal`] provides the strict write-ahead `fsync` barrier.
    pub sync_writes: bool,
}

impl Default for ReverseMapConfig {
    fn default() -> Self {
        Self {
            write_buffer_size: DEFAULT_WRITE_BUFFER_SIZE_BYTES,
            max_write_buffer_number: DEFAULT_MAX_WRITE_BUFFER_NUMBER,
            enable_lz4_compression: true,
            sync_writes: false,
        }
    }
}

impl ReverseMapConfig {
    /// Returns a configuration tuned for operation beneath [`crate::wal::CoupledWal`]
    /// (`sync_writes = false`, LZ4 compression enabled).
    #[must_use]
    pub fn for_wal_coupled() -> Self {
        Self {
            sync_writes: false,
            ..Self::default()
        }
    }

    /// Returns a configuration with synchronous RocksDB writes (`sync_writes = true`)
    /// for standalone durability without an external coupled WAL.
    #[must_use]
    pub fn standalone_durable() -> Self {
        Self {
            sync_writes: true,
            ..Self::default()
        }
    }

    /// Overrides the write buffer size in bytes.
    #[must_use]
    pub const fn with_write_buffer_size(mut self, bytes: usize) -> Self {
        self.write_buffer_size = bytes;
        self
    }

    /// Overrides `sync_writes`.
    #[must_use]
    pub const fn with_sync_writes(mut self, sync_writes: bool) -> Self {
        self.sync_writes = sync_writes;
        self
    }

    /// Overrides LZ4 compression.
    #[must_use]
    pub const fn with_lz4_compression(mut self, enable: bool) -> Self {
        self.enable_lz4_compression = enable;
        self
    }
}

/// Write-optimized RocksDB implementation of [`ReverseMapStore`].
///
/// # Concurrency & Thread Safety
/// `RocksDbReverseMap` is `Send + Sync` and supports lock-free concurrent `&self`
/// operations via RocksDB's internal thread-safe handle and an atomic entry-count
/// estimate.
pub struct RocksDbReverseMap {
    db: DB,
    path: PathBuf,
    sync_writes: bool,
    entry_count: AtomicU64,
}

impl RocksDbReverseMap {
    /// Opens or creates a [`RocksDbReverseMap`] at `path` using `config`.
    ///
    /// Automatically registers the custom [`MERGE_OPERATOR_NAME`] merge operator
    /// and initializes the live entry-count estimate accurately from any existing
    /// on-disk SSTables.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if RocksDB fails to open or if existing
    /// on-disk values cannot be decoded.
    pub fn open(path: impl AsRef<Path>, config: ReverseMapConfig) -> AerisResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let start = Instant::now();
        let opts = build_rocksdb_options(&config);

        let db = DB::open(&opts, &path_buf).map_err(|err| {
            error!(path = %path_buf.display(), error = %err, "failed to open RocksDB Reverse Map");
            AerisError::StorageError(format!(
                "failed to open Reverse Map at '{}': {err}",
                path_buf.display()
            ))
        })?;

        let initial_entries = count_live_entries(&db)?;
        let elapsed_us = start.elapsed().as_micros();
        info!(
            path = %path_buf.display(),
            initial_entries,
            sync_writes = config.sync_writes,
            lz4 = config.enable_lz4_compression,
            elapsed_us,
            "opened RocksDB Reverse Map"
        );

        Ok(Self {
            db,
            path: path_buf,
            sync_writes: config.sync_writes,
            entry_count: AtomicU64::new(initial_entries),
        })
    }

    /// Returns the filesystem directory path backing this Reverse Map.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Atomically tombstones every fingerprint in `old_fps` and appends every
    /// `(fingerprint, key)` pair in `new_entries` within a single RocksDB `WriteBatch`.
    ///
    /// Used by [`crate::engine::CoupledStorageEngine`] to apply multi-fingerprint
    /// `WalOp::Adapt` records and crash-recovery reconciliations atomically.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if the atomic batch write fails.
    pub fn replace_fingerprints_batch(
        &self,
        old_fps: &[Fingerprint],
        new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        if old_fps.is_empty() && new_entries.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for fp in old_fps {
            batch.delete(fp.to_bytes());
        }
        for (fp, key) in new_entries {
            batch.merge(fp.to_bytes(), encode_add_key_operand(*key));
        }
        self.db
            .write_opt(batch, &self.write_options())
            .map_err(|err| {
                error!(
                    path = %self.path.display(),
                    old_fp_count = old_fps.len(),
                    new_entry_count = new_entries.len(),
                    error = %err,
                    "failed atomic Reverse Map batch replacement"
                );
                AerisError::StorageError(format!(
                    "Reverse Map replace_fingerprints_batch failed: {err}"
                ))
            })?;
        // Adjust entry estimate only when adding brand-new entries with no tombstones;
        // replacement splits typically preserve total live key cardinality.
        if old_fps.is_empty() && !new_entries.is_empty() {
            self.entry_count
                .fetch_add(new_entries.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Scans all live `(Fingerprint, BTreeSet<Key>)` mappings currently stored in
    /// the Reverse Map.
    ///
    /// Used off the hot path by recovery reconciliation to identify orphaned keys
    /// or stale fingerprints left across a filter rebuild.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB iteration failure or
    /// [`AerisError::EncodingError`] if an on-disk fingerprint or value is malformed.
    pub fn scan_all_entries(&self) -> AerisResult<BTreeMap<Fingerprint, BTreeSet<Key>>> {
        let mut entries = BTreeMap::new();
        for item in self.db.iterator(IteratorMode::Start) {
            let (raw_fp, raw_val) = item.map_err(|err| {
                error!(path = %self.path.display(), error = %err, "Reverse Map entry scan error");
                AerisError::StorageError(format!("Reverse Map scan_all_entries failed: {err}"))
            })?;
            let keys = decode_key_list(&raw_val)?;
            if keys.is_empty() {
                continue;
            }
            let fp = Fingerprint::from_bytes(&raw_fp)?;
            entries.insert(fp, keys.into_iter().collect());
        }
        Ok(entries)
    }

    /// Recomputes and stores the exact live key-entry count from RocksDB.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if RocksDB iteration fails.
    pub fn refresh_entry_count(&self) -> AerisResult<u64> {
        let exact = count_live_entries(&self.db)?;
        self.entry_count.store(exact, Ordering::Relaxed);
        Ok(exact)
    }

    #[inline]
    fn write_options(&self) -> WriteOptions {
        let mut wo = WriteOptions::default();
        wo.set_sync(self.sync_writes);
        wo
    }
}

impl ReverseMapStore for RocksDbReverseMap {
    fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let operand = encode_add_key_operand(key);
        self.db
            .merge_opt(fp.to_bytes(), operand, &self.write_options())
            .map_err(|err| {
                error!(path = %self.path.display(), key, error = %err, "Reverse Map append_key failed");
                AerisError::StorageError(format!("Reverse Map append_key({key}) failed: {err}"))
            })?;
        self.entry_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for (fp, key) in entries {
            batch.merge(fp.to_bytes(), encode_add_key_operand(*key));
        }
        self.db
            .write_opt(batch, &self.write_options())
            .map_err(|err| {
                error!(
                    path = %self.path.display(),
                    batch_len = entries.len(),
                    error = %err,
                    "Reverse Map append_batch failed"
                );
                AerisError::StorageError(format!("Reverse Map append_batch failed: {err}"))
            })?;
        self.entry_count
            .fetch_add(entries.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let operand = encode_del_key_operand(key);
        self.db
            .merge_opt(fp.to_bytes(), operand, &self.write_options())
            .map_err(|err| {
                error!(path = %self.path.display(), key, error = %err, "Reverse Map remove_key failed");
                AerisError::StorageError(format!("Reverse Map remove_key({key}) failed: {err}"))
            })?;
        let _ = self
            .entry_count
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(1))
            });
        Ok(())
    }

    fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
        let raw = self.db.get(fp.to_bytes()).map_err(|err| {
            error!(path = %self.path.display(), error = %err, "Reverse Map get_keys failed");
            AerisError::StorageError(format!("Reverse Map get_keys failed: {err}"))
        })?;
        match raw {
            None => Ok(Vec::new()),
            Some(bytes) => decode_key_list(&bytes),
        }
    }

    fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
        self.db
            .delete_opt(fp.to_bytes(), &self.write_options())
            .map_err(|err| {
                error!(path = %self.path.display(), error = %err, "Reverse Map tombstone failed");
                AerisError::StorageError(format!("Reverse Map tombstone failed: {err}"))
            })?;
        let _ = self
            .entry_count
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(1))
            });
        Ok(())
    }

    fn replace_fingerprint(
        &self,
        old_fp: &Fingerprint,
        new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        let mut batch = WriteBatch::default();
        batch.delete(old_fp.to_bytes());
        for (new_fp, key) in new_entries {
            batch.merge(new_fp.to_bytes(), encode_add_key_operand(*key));
        }
        self.db
            .write_opt(batch, &self.write_options())
            .map_err(|err| {
                error!(
                    path = %self.path.display(),
                    new_entries = new_entries.len(),
                    error = %err,
                    "Reverse Map replace_fingerprint failed"
                );
                AerisError::StorageError(format!("Reverse Map replace_fingerprint failed: {err}"))
            })?;
        Ok(())
    }

    fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        let mut keys = BTreeSet::new();
        for item in self.db.iterator(IteratorMode::Start) {
            let (_raw_fp, raw_val) = item.map_err(|err| {
                error!(path = %self.path.display(), error = %err, "Reverse Map scan_all_keys failed");
                AerisError::StorageError(format!("Reverse Map scan_all_keys failed: {err}"))
            })?;
            for key in decode_key_list(&raw_val)? {
                keys.insert(key);
            }
        }
        self.entry_count.store(keys.len() as u64, Ordering::Relaxed);
        Ok(keys.into_iter().collect())
    }

    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
        let db_iter = self.db.iterator(IteratorMode::Start);
        Ok(Box::new(ReverseMapKeyIter::new(db_iter)))
    }

    fn entry_count_estimate(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }

    fn flush(&self) -> AerisResult<()> {
        let start = Instant::now();
        debug!(path = %self.path.display(), "flushing RocksDB Reverse Map memtables");
        self.db.flush().map_err(|err| {
            error!(path = %self.path.display(), error = %err, "Reverse Map flush failed");
            AerisError::StorageError(format!("Reverse Map flush failed: {err}"))
        })?;
        info!(
            path = %self.path.display(),
            elapsed_us = start.elapsed().as_micros(),
            "flushed RocksDB Reverse Map"
        );
        Ok(())
    }

    fn compact(&self) -> AerisResult<()> {
        let start = Instant::now();
        debug!(path = %self.path.display(), "starting RocksDB Reverse Map full compaction");
        self.db.compact_range::<&[u8], &[u8]>(None, None);
        let live_entries = self.refresh_entry_count()?;
        info!(
            path = %self.path.display(),
            live_entries,
            elapsed_us = start.elapsed().as_micros(),
            "compacted RocksDB Reverse Map"
        );
        Ok(())
    }
}

/// Builds RocksDB [`Options`] with the registered Reverse Map merge operator.
fn build_rocksdb_options(config: &ReverseMapConfig) -> Options {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(config.write_buffer_size);
    opts.set_max_write_buffer_number(config.max_write_buffer_number);
    opts.set_compression_type(if config.enable_lz4_compression {
        DBCompressionType::Lz4
    } else {
        DBCompressionType::None
    });
    opts.set_merge_operator(
        MERGE_OPERATOR_NAME,
        rocksdb_full_merge,
        rocksdb_partial_merge,
    );
    opts
}

/// RocksDB C-callback wrapper delegating `full_merge` to pure [`fold_full_merge`].
fn rocksdb_full_merge(
    _key: &[u8],
    existing_val: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    match fold_full_merge(existing_val, operands) {
        Ok(merged) => Some(merged),
        Err(err) => {
            error!(error = %err, "RocksDB Reverse Map full_merge encountered corrupt operand");
            None
        }
    }
}

/// RocksDB C-callback wrapper delegating `partial_merge` to pure [`fold_partial_merge`].
fn rocksdb_partial_merge(
    _key: &[u8],
    _existing_val: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    match fold_partial_merge(operands) {
        Ok(merged) => Some(merged),
        Err(err) => {
            error!(error = %err, "RocksDB Reverse Map partial_merge encountered corrupt operand");
            None
        }
    }
}

/// Scans an open Reverse Map DB to compute the initial live key-entry count.
fn count_live_entries(db: &DB) -> AerisResult<u64> {
    let mut total = 0u64;
    for item in db.iterator(IteratorMode::Start) {
        let (_k, v) = item.map_err(|err| {
            AerisError::StorageError(format!("Reverse Map initial count scan failed: {err}"))
        })?;
        total = total.saturating_add(count_keys_in_value(&v)? as u64);
    }
    Ok(total)
}
