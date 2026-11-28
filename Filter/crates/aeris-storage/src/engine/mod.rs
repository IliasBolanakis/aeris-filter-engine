//! Crash-consistent coupled storage engine coordinating [`CoupledWal`],
//! [`ReferenceKvStore`], and [`RocksDbReverseMap`].
//!
//! # Write-Ahead Ordering Protocol
//! Every mutating operation (`put`, `delete`, `record_adapt`, `put_batch`) acquires
//! the engine's `mutation_lock` and executes in strict order:
//! 1. Append checksummed [`WalOp`] frame(s) directly to [`CoupledWal`]'s OS file descriptor.
//! 2. Apply the configured [`WalSyncPolicy`] durability barrier (`fsync` on every op under
//!    [`WalSyncPolicy::PerOpFsync`], every `batch_size` ops under
//!    [`WalSyncPolicy::GroupCommit`], and once per [`CoupledStorageEngine::put_batch`]).
//! 3. Apply the mutation(s) to [`ReferenceKvStore`] and [`RocksDbReverseMap`].
//! 4. Return `Ok(lsn)` to the caller.
//!
//! # Lock Ordering
//! `CoupledStorageEngine::mutation_lock` -> `CoupledWal::inner` -> RocksDB internal locks.
//! Read-only paths (`get`, `range_non_empty`, `reverse_map().get_keys(...)`) do not
//! acquire `mutation_lock`.

mod config;
mod recovery;

#[cfg(test)]
mod tests;

pub use config::{
    RecoveryReport, StorageConfig, WalSyncPolicy, DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE,
};

use self::recovery::{
    insert_or_expand_during_rebuild, reconcile_kv_and_reverse_map, replay_wal_records,
    RebuildReconcileState,
};
use crate::kv_store::ReferenceKvStore;
use crate::reverse_map::RocksDbReverseMap;
use crate::wal::CoupledWal;
use aeris_types::{
    AerisError, AerisResult, Fingerprint, Key, Lsn, QueryRange, RangeFilter, ReverseMapStore,
    WalEngine, WalOp,
};
use parking_lot::Mutex;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info};

/// Filename for the append-only coupled Write-Ahead Log under the storage root directory.
pub const WAL_FILE_NAME: &str = "wal.log";

/// Subdirectory name for the RocksDB primary reference key-value store.
pub const KV_DB_DIR_NAME: &str = "kv_db";

/// Subdirectory name for the RocksDB write-optimized Reverse Map.
pub const REVERSE_MAP_DB_DIR_NAME: &str = "reverse_map_db";

/// Crash-consistent storage engine coupling [`CoupledWal`], [`ReferenceKvStore`],
/// and [`RocksDbReverseMap`] under a single directory.
///
/// # Directory Layout
/// - `<dir>/wal.log` — Binary CRC32-framed WAL (`CoupledWal`).
/// - `<dir>/kv_db/` — Big-endian sorted primary KV store (`ReferenceKvStore`).
/// - `<dir>/reverse_map_db/` — Write-optimized merge-operator Reverse Map (`RocksDbReverseMap`).
pub struct CoupledStorageEngine {
    root_dir: PathBuf,
    wal: CoupledWal,
    kv_store: ReferenceKvStore,
    reverse_map: RocksDbReverseMap,
    wal_sync_policy: WalSyncPolicy,
    pending_sync_ops: AtomicUsize,
    /// Serializes WAL LSN assignment with RocksDB application so WAL order strictly
    /// equals KV store and Reverse Map mutation order under concurrent writers.
    mutation_lock: Mutex<()>,
}

impl CoupledStorageEngine {
    /// Opens the storage directory `dir` with default WAL-coupled settings, replays
    /// all post-checkpoint WAL records idempotently into both `kv_store` and
    /// `reverse_map`, reconciles any discrepancy, and returns `(engine, RecoveryReport)`.
    ///
    /// # Errors
    /// Returns [`AerisError::WalCorruption`] on mid-log WAL corruption,
    /// [`AerisError::WalIoError`] on WAL filesystem failure, or
    /// [`AerisError::StorageError`] on RocksDB failure.
    pub fn open_and_recover(dir: impl AsRef<Path>) -> AerisResult<(Self, RecoveryReport)> {
        Self::open_and_recover_with_config(dir, StorageConfig::default())
    }

    /// Opens and recovers a [`CoupledStorageEngine`] at `dir` using explicit `config`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`], [`AerisError::WalCorruption`],
    /// [`AerisError::WalIoError`], or [`AerisError::StorageError`] on failure.
    pub fn open_and_recover_with_config(
        dir: impl AsRef<Path>,
        config: StorageConfig,
    ) -> AerisResult<(Self, RecoveryReport)> {
        config.wal_sync_policy.validate()?;
        let root_dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&root_dir).map_err(|err| {
            AerisError::WalIoError(format!(
                "failed to create storage root '{}': {err}",
                root_dir.display()
            ))
        })?;

        let wal = CoupledWal::open(root_dir.join(WAL_FILE_NAME))?;
        let truncated_tail_bytes = wal.truncated_tail_bytes();
        let kv_store =
            ReferenceKvStore::open_with_config(root_dir.join(KV_DB_DIR_NAME), config.kv_store)?;
        let reverse_map =
            RocksDbReverseMap::open(root_dir.join(REVERSE_MAP_DB_DIR_NAME), config.reverse_map)?;

        let replay_start_lsn = wal.checkpoint_lsn().saturating_add(1);
        let counts =
            replay_wal_records(wal.replay_from(replay_start_lsn)?, &kv_store, &reverse_map)?;
        let replayed_records = counts.total();
        let reconciled = reconcile_kv_and_reverse_map(&wal, &kv_store, &reverse_map)?;

        if replayed_records > 0 || reconciled {
            kv_store.flush()?;
            reverse_map.flush()?;
        }
        let _ = reverse_map.refresh_entry_count()?;

        let report = RecoveryReport {
            replayed_records,
            recovered_puts: counts.puts,
            recovered_deletes: counts.deletes,
            recovered_adapts: counts.adapts,
            truncated_tail_bytes,
            last_lsn: wal.last_lsn(),
            live_key_count: kv_store.len()?,
        };

        info!(
            dir = %root_dir.display(),
            replayed_records = report.replayed_records,
            recovered_puts = report.recovered_puts,
            recovered_deletes = report.recovered_deletes,
            recovered_adapts = report.recovered_adapts,
            truncated_tail_bytes = report.truncated_tail_bytes,
            last_lsn = report.last_lsn,
            live_key_count = report.live_key_count,
            "completed CoupledStorageEngine open_and_recover"
        );

        let engine = Self {
            root_dir,
            wal,
            kv_store,
            reverse_map,
            wal_sync_policy: config.wal_sync_policy,
            pending_sync_ops: AtomicUsize::new(0),
            mutation_lock: Mutex::new(()),
        };
        Ok((engine, report))
    }

    /// Executes the configured [`WalSyncPolicy`] durability barrier while the caller
    /// holds `self.mutation_lock`.
    #[inline]
    fn maybe_sync_wal_locked(&self) -> AerisResult<()> {
        match self.wal_sync_policy {
            WalSyncPolicy::PerOpFsync => {
                self.wal.sync()?;
                self.pending_sync_ops.store(0, Ordering::Release);
            }
            WalSyncPolicy::GroupCommit { batch_size } => {
                let pending = self.pending_sync_ops.fetch_add(1, Ordering::AcqRel) + 1;
                if pending >= batch_size {
                    self.wal.sync()?;
                    self.pending_sync_ops.store(0, Ordering::Release);
                }
            }
        }
        Ok(())
    }

    /// Durably persists a `(key, value)` pair along with its filter [`Fingerprint`].
    ///
    /// Strictly enforces WAL append (writing the CRC32 frame to the OS file descriptor)
    /// and the configured [`WalSyncPolicy`] barrier **before** applying the mutation to
    /// [`ReferenceKvStore`] and [`RocksDbReverseMap`].
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`], [`AerisError::EncodingError`], or
    /// [`AerisError::StorageError`] if WAL durability or RocksDB application fails.
    pub fn put(&self, key: Key, value: &[u8], fp: &Fingerprint) -> AerisResult<Lsn> {
        let _guard = self.mutation_lock.lock();
        let lsn = self.wal.append(WalOp::Put {
            key,
            value: value.to_vec(),
            fp: fp.clone(),
        })?;
        self.maybe_sync_wal_locked()?;
        self.kv_store.put(key, value)?;
        self.reverse_map.append_key(fp, key)?;
        Ok(lsn)
    }

    /// Durably persists a batch of `(key, value, fp)` mutations with a single
    /// group-commit WAL `fsync` barrier and a single batched [`RocksDbReverseMap`] write.
    ///
    /// Acquires `mutation_lock`, appends all [`WalOp::Put`] frames to [`CoupledWal`],
    /// executes a single `CoupledWal::sync()` barrier (resetting `pending_sync_ops = 0`),
    /// applies all writes to [`ReferenceKvStore`], and commits a single
    /// [`ReverseMapStore::append_batch`] call.
    ///
    /// Returns the highest assigned [`Lsn`], or [`Self::last_lsn`] if `entries` is empty.
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`], [`AerisError::EncodingError`], or
    /// [`AerisError::StorageError`] on failure.
    pub fn put_batch(&self, entries: &[(Key, &[u8], &Fingerprint)]) -> AerisResult<Lsn> {
        let _guard = self.mutation_lock.lock();
        if entries.is_empty() {
            return Ok(self.wal.last_lsn());
        }

        let mut highest_lsn = self.wal.last_lsn();
        for &(key, value, fp) in entries {
            highest_lsn = self.wal.append(WalOp::Put {
                key,
                value: value.to_vec(),
                fp: fp.clone(),
            })?;
            self.pending_sync_ops.fetch_add(1, Ordering::AcqRel);
        }
        self.wal.sync()?;
        self.pending_sync_ops.store(0, Ordering::Release);

        let mut rm_batch = Vec::with_capacity(entries.len());
        for &(key, value, fp) in entries {
            self.kv_store.put(key, value)?;
            rm_batch.push((fp.clone(), key));
        }
        self.reverse_map.append_batch(&rm_batch)?;
        Ok(highest_lsn)
    }

    /// Forces an immediate `fsync` durability barrier on [`CoupledWal`] and resets
    /// [`Self::pending_sync_ops`] to `0`.
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`] if flushing or syncing the WAL fails.
    pub fn flush_wal(&self) -> AerisResult<()> {
        let _guard = self.mutation_lock.lock();
        self.wal.sync()?;
        self.pending_sync_ops.store(0, Ordering::Release);
        Ok(())
    }

    /// Durably deletes `key` from both [`ReferenceKvStore`] and [`RocksDbReverseMap`].
    ///
    /// Strictly enforces WAL append and the configured [`WalSyncPolicy`] barrier
    /// **before** applying the deletion.
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`], [`AerisError::EncodingError`], or
    /// [`AerisError::StorageError`] on failure.
    pub fn delete(&self, key: Key, fp: &Fingerprint) -> AerisResult<Lsn> {
        let _guard = self.mutation_lock.lock();
        let lsn = self.wal.append(WalOp::Delete {
            key,
            fp: fp.clone(),
        })?;
        self.maybe_sync_wal_locked()?;
        self.kv_store.delete(key)?;
        self.reverse_map.remove_key(fp, key)?;
        Ok(lsn)
    }

    /// Durably records a false-positive fingerprint adaptation, atomically tombstoning
    /// `old_fps` and appending `new_entries` in [`RocksDbReverseMap`].
    ///
    /// Strictly enforces WAL append and the configured [`WalSyncPolicy`] barrier
    /// **before** mutating the Reverse Map.
    ///
    /// # Errors
    /// Returns [`AerisError::WalIoError`], [`AerisError::EncodingError`], or
    /// [`AerisError::StorageError`] on failure.
    pub fn record_adapt(
        &self,
        old_fps: Vec<Fingerprint>,
        new_entries: Vec<(Fingerprint, Key)>,
        range: QueryRange,
    ) -> AerisResult<Lsn> {
        let _guard = self.mutation_lock.lock();
        let lsn = self.wal.append(WalOp::Adapt {
            old_fps: old_fps.clone(),
            new_entries: new_entries.clone(),
            range,
        })?;
        self.maybe_sync_wal_locked()?;
        self.reverse_map
            .replace_fingerprints_batch(&old_fps, &new_entries)?;
        debug!(
            lsn,
            old_fp_count = old_fps.len(),
            new_entry_count = new_entries.len(),
            range_start = range.start(),
            range_end = range.end(),
            "persisted durable Adapt operation"
        );
        Ok(lsn)
    }

    /// Flushes both [`ReferenceKvStore`] and [`RocksDbReverseMap`] SSTables to disk
    /// and writes a synced [`WalOp::Checkpoint`] marker covering all LSNs up to
    /// `last_lsn`.
    ///
    /// Returns the checkpointed `up_to_lsn`.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if RocksDB flush fails or
    /// [`AerisError::WalIoError`] if WAL checkpoint sync fails.
    pub fn checkpoint(&self) -> AerisResult<Lsn> {
        let _guard = self.mutation_lock.lock();
        let up_to_lsn = self.wal.last_lsn();
        self.kv_store.flush()?;
        self.reverse_map.flush()?;
        self.wal.checkpoint(up_to_lsn)?;
        self.pending_sync_ops.store(0, Ordering::Release);
        info!(
            dir = %self.root_dir.display(),
            up_to_lsn,
            wal_last_lsn = self.wal.last_lsn(),
            "storage engine checkpoint completed"
        );
        Ok(up_to_lsn)
    }

    /// Executes [`Self::checkpoint`] and immediately compacts the WAL by removing
    /// all records at or before the newly checkpointed LSN.
    ///
    /// Returns `(checkpointed_lsn, reclaimed_wal_bytes)`.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`], [`AerisError::WalIoError`], or
    /// [`AerisError::WalCorruption`] on failure.
    pub fn checkpoint_and_compact_wal(&self) -> AerisResult<(Lsn, u64)> {
        let _guard = self.mutation_lock.lock();
        let up_to_lsn = self.wal.last_lsn();
        self.kv_store.flush()?;
        self.reverse_map.flush()?;
        self.wal.checkpoint(up_to_lsn)?;
        self.pending_sync_ops.store(0, Ordering::Release);
        let reclaimed = self.wal.truncate_before(up_to_lsn)?;
        info!(
            dir = %self.root_dir.display(),
            up_to_lsn,
            reclaimed_wal_bytes = reclaimed,
            "storage engine checkpoint and WAL compaction completed"
        );
        Ok((up_to_lsn, reclaimed))
    }

    /// Rebuilds an in-memory [`RangeFilter`] from every live key in [`ReferenceKvStore`],
    /// reconciles [`RocksDbReverseMap`] to match the rebuilt filter's canonical fingerprints,
    /// and verifies zero false negatives across all loaded keys.
    ///
    /// Automatically invokes [`RangeFilter::expand`] if the filter reaches its maximum
    /// load factor while ingesting the recovered key set.
    ///
    /// Returns the number of live keys loaded into `filter`.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB failure, any error surfaced by
    /// `filter.insert` / `filter.expand`, or [`AerisError::InternalInvariant`] if a
    /// false negative is detected after loading.
    pub fn rebuild_filter_and_reconcile<F: RangeFilter>(
        &self,
        filter: &mut F,
    ) -> AerisResult<usize> {
        let _guard = self.mutation_lock.lock();
        let live_keys = self.kv_store.scan_all_keys()?;
        let existing = self.reverse_map.scan_all_entries()?;

        let mut state = RebuildReconcileState::new(&existing);
        for &key in &live_keys {
            let fp = insert_or_expand_during_rebuild(filter, &self.reverse_map, &mut state, key)?;
            state.record_inserted(fp, key);
        }
        state.finalize(&self.reverse_map)?;

        for &key in &live_keys {
            if !filter.query_point(key) {
                return Err(AerisError::InternalInvariant(format!(
                    "rebuilt RangeFilter produced false negative for recovered live key {key}"
                )));
            }
        }

        info!(
            dir = %self.root_dir.display(),
            loaded_keys = live_keys.len(),
            expanded = state.did_expand,
            tombstoned_fps = state.tombstoned_count,
            merged_entries = state.merged_count,
            "rebuilt RangeFilter and reconciled Reverse Map with zero false negatives"
        );
        Ok(live_keys.len())
    }

    /// Returns the storage root directory path.
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Returns a shared reference to the underlying [`CoupledWal`].
    #[must_use]
    pub fn wal(&self) -> &CoupledWal {
        &self.wal
    }

    /// Returns a shared reference to the underlying [`ReferenceKvStore`].
    #[must_use]
    pub fn kv_store(&self) -> &ReferenceKvStore {
        &self.kv_store
    }

    /// Returns a shared reference to the underlying [`RocksDbReverseMap`].
    #[must_use]
    pub fn reverse_map(&self) -> &RocksDbReverseMap {
        &self.reverse_map
    }

    /// Retrieves the stored value for `key` from [`ReferenceKvStore`].
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB read failure.
    pub fn get(&self, key: Key) -> AerisResult<Option<Vec<u8>>> {
        self.kv_store.get(key)
    }

    /// Checks whether any live key exists in `range` via [`ReferenceKvStore::range_non_empty`].
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB read failure.
    pub fn range_non_empty(&self, range: QueryRange) -> AerisResult<bool> {
        self.kv_store.range_non_empty(range)
    }

    /// Returns the highest LSN recorded in [`CoupledWal`].
    #[must_use]
    pub fn last_lsn(&self) -> Lsn {
        self.wal.last_lsn()
    }

    /// Returns the configured [`WalSyncPolicy`].
    #[inline]
    #[must_use]
    pub const fn wal_sync_policy(&self) -> WalSyncPolicy {
        self.wal_sync_policy
    }

    /// Returns the number of WAL-appended mutations pending the next `fsync` barrier.
    #[inline]
    #[must_use]
    pub fn pending_sync_ops(&self) -> usize {
        self.pending_sync_ops.load(Ordering::Acquire)
    }
}

impl Drop for CoupledStorageEngine {
    fn drop(&mut self) {
        if *self.pending_sync_ops.get_mut() > 0 {
            let _ = self.wal.sync();
            *self.pending_sync_ops.get_mut() = 0;
        }
    }
}
