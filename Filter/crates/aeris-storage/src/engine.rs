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

use crate::kv_store::{KvStoreConfig, ReferenceKvStore};
use crate::reverse_map::{ReverseMapConfig, RocksDbReverseMap};
use crate::wal::CoupledWal;
use aeris_types::{
    AerisError, AerisResult, Fingerprint, Key, Lsn, QueryRange, RangeFilter, ReverseMapStore,
    WalEngine, WalOp, MAX_QUOTIENT_BITS,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info, warn};

/// Filename for the append-only coupled Write-Ahead Log under the storage root directory.
pub const WAL_FILE_NAME: &str = "wal.log";

/// Subdirectory name for the RocksDB primary reference key-value store.
pub const KV_DB_DIR_NAME: &str = "kv_db";

/// Subdirectory name for the RocksDB write-optimized Reverse Map.
pub const REVERSE_MAP_DB_DIR_NAME: &str = "reverse_map_db";

/// Default group-commit WAL batch size (`B = 1,000`) amortizing `fsync` latency
/// while preserving strict WAL-first OS kernel write ordering on every operation.
pub const DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE: usize = 1_000;

/// Durability barrier policy governing when [`CoupledStorageEngine`] invokes
/// `CoupledWal::sync()` (`fsync` / `FlushFileBuffers`) after writing WAL frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalSyncPolicy {
    /// Amortizes `fsync` across `batch_size` WAL-appended mutations (default
    /// [`DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE`] = `1_000`). Every mutation still
    /// writes its CRC32 frame to the OS file descriptor before RocksDB is mutated,
    /// and [`CoupledStorageEngine::put_batch`], [`CoupledStorageEngine::flush_wal`],
    /// [`CoupledStorageEngine::checkpoint`], and clean `Drop` force an immediate sync.
    GroupCommit {
        /// Number of mutations per automatic WAL `fsync` barrier (must be `>= 1`).
        batch_size: usize,
    },
    /// Strict synchronous `fsync` on every individual `put`, `delete`, and `record_adapt`.
    PerOpFsync,
}

impl Default for WalSyncPolicy {
    fn default() -> Self {
        Self::GroupCommit {
            batch_size: DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE,
        }
    }
}

impl WalSyncPolicy {
    /// Validates the WAL sync policy invariants.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `GroupCommit { batch_size: 0 }`.
    pub fn validate(&self) -> AerisResult<()> {
        if let Self::GroupCommit { batch_size: 0 } = self {
            return Err(AerisError::InvalidConfig(
                "WalSyncPolicy::GroupCommit::batch_size must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration bundle for [`CoupledStorageEngine`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageConfig {
    /// Configuration for the primary [`ReferenceKvStore`].
    pub kv_store: KvStoreConfig,
    /// Configuration for the write-optimized [`RocksDbReverseMap`].
    pub reverse_map: ReverseMapConfig,
    /// WAL durability barrier policy (defaults to group-commit batching with `B = 1_000`).
    pub wal_sync_policy: WalSyncPolicy,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            kv_store: KvStoreConfig::for_wal_coupled(),
            reverse_map: ReverseMapConfig::for_wal_coupled(),
            wal_sync_policy: WalSyncPolicy::default(),
        }
    }
}

impl StorageConfig {
    /// Constructs a [`StorageConfig`] enforcing strict synchronous `fsync` on every
    /// individual mutation ([`WalSyncPolicy::PerOpFsync`]).
    #[must_use]
    pub fn strict_per_op_fsync() -> Self {
        Self::default().with_wal_sync_policy(WalSyncPolicy::PerOpFsync)
    }

    /// Overrides [`Self::wal_sync_policy`].
    #[must_use]
    pub const fn with_wal_sync_policy(mut self, policy: WalSyncPolicy) -> Self {
        self.wal_sync_policy = policy;
        self
    }
}

/// Diagnostic summary returned by [`CoupledStorageEngine::open_and_recover`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Total number of mutation WAL records replayed (`Put` + `Delete` + `Adapt`).
    pub replayed_records: usize,
    /// Number of [`WalOp::Put`] records replayed during recovery.
    pub recovered_puts: usize,
    /// Number of [`WalOp::Delete`] records replayed during recovery.
    pub recovered_deletes: usize,
    /// Number of [`WalOp::Adapt`] records replayed during recovery.
    pub recovered_adapts: usize,
    /// Number of torn-tail WAL bytes truncated and `fsync`-ed during recovery.
    pub truncated_tail_bytes: u64,
    /// Highest intact LSN present in the WAL after recovery (`0` if empty).
    pub last_lsn: Lsn,
    /// Exact count of live keys present in [`ReferenceKvStore`] after recovery.
    pub live_key_count: usize,
}

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

        let checkpoint_lsn = wal.checkpoint_lsn();
        let replay_start_lsn = checkpoint_lsn.saturating_add(1);
        let records = wal.replay_from(replay_start_lsn)?;

        let mut recovered_puts = 0usize;
        let mut recovered_deletes = 0usize;
        let mut recovered_adapts = 0usize;

        for record in records {
            match record.op {
                WalOp::Put { key, value, fp } => {
                    kv_store.put(key, &value)?;
                    reverse_map.append_key(&fp, key)?;
                    recovered_puts += 1;
                }
                WalOp::Delete { key, fp } => {
                    kv_store.delete(key)?;
                    reverse_map.remove_key(&fp, key)?;
                    recovered_deletes += 1;
                }
                WalOp::Adapt {
                    old_fps,
                    new_entries,
                    ..
                } => {
                    reverse_map.replace_fingerprints_batch(&old_fps, &new_entries)?;
                    recovered_adapts += 1;
                }
                WalOp::Checkpoint { .. } => {}
            }
        }

        let replayed_records = recovered_puts + recovered_deletes + recovered_adapts;
        let reconciled = reconcile_kv_and_reverse_map(&wal, &kv_store, &reverse_map)?;

        if replayed_records > 0 || reconciled {
            kv_store.flush()?;
            reverse_map.flush()?;
        }
        let _ = reverse_map.refresh_entry_count()?;

        let live_key_count = kv_store.len()?;
        let last_lsn = wal.last_lsn();

        let report = RecoveryReport {
            replayed_records,
            recovered_puts,
            recovered_deletes,
            recovered_adapts,
            truncated_tail_bytes,
            last_lsn,
            live_key_count,
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

/// Incremental reconciliation state used by [`CoupledStorageEngine::rebuild_filter_and_reconcile`].
struct RebuildReconcileState<'a> {
    existing: &'a BTreeMap<Fingerprint, BTreeSet<Key>>,
    desired_before_expand: BTreeMap<Fingerprint, BTreeSet<Key>>,
    pending_after_expand: Vec<(Fingerprint, Key)>,
    did_expand: bool,
    tombstoned_count: usize,
    merged_count: usize,
}

impl<'a> RebuildReconcileState<'a> {
    fn new(existing: &'a BTreeMap<Fingerprint, BTreeSet<Key>>) -> Self {
        Self {
            existing,
            desired_before_expand: BTreeMap::new(),
            pending_after_expand: Vec::new(),
            did_expand: false,
            tombstoned_count: 0,
            merged_count: 0,
        }
    }

    fn record_inserted(&mut self, fp: Fingerprint, key: Key) {
        if self.did_expand {
            self.pending_after_expand.push((fp, key));
        } else {
            self.desired_before_expand
                .entry(fp)
                .or_default()
                .insert(key);
        }
    }

    fn flush_before_expand(&mut self, reverse_map: &RocksDbReverseMap) -> AerisResult<()> {
        if !self.did_expand {
            // Synchronize Reverse Map with keys inserted so far before the first expand()
            // so any selective rejuvenation invoked by expand() sees exact entries.
            let (to_tombstone, to_merge) =
                compute_reverse_map_diff(self.existing, &self.desired_before_expand);
            self.tombstoned_count += to_tombstone.len();
            self.merged_count += to_merge.len();
            reverse_map.replace_fingerprints_batch(&to_tombstone, &to_merge)?;
            self.did_expand = true;
            return Ok(());
        }
        if !self.pending_after_expand.is_empty() {
            self.merged_count += self.pending_after_expand.len();
            reverse_map.append_batch(&self.pending_after_expand)?;
            self.pending_after_expand.clear();
        }
        Ok(())
    }

    fn finalize(&mut self, reverse_map: &RocksDbReverseMap) -> AerisResult<()> {
        if self.did_expand {
            if !self.pending_after_expand.is_empty() {
                self.merged_count += self.pending_after_expand.len();
                reverse_map.append_batch(&self.pending_after_expand)?;
                self.pending_after_expand.clear();
            }
            reverse_map.flush()?;
            let _ = reverse_map.refresh_entry_count()?;
            return Ok(());
        }

        let (to_tombstone, to_merge) =
            compute_reverse_map_diff(self.existing, &self.desired_before_expand);
        self.tombstoned_count = to_tombstone.len();
        self.merged_count = to_merge.len();
        if !to_tombstone.is_empty() || !to_merge.is_empty() {
            reverse_map.replace_fingerprints_batch(&to_tombstone, &to_merge)?;
            reverse_map.flush()?;
            let _ = reverse_map.refresh_entry_count()?;
        }
        Ok(())
    }
}

/// Inserts `key` into `filter`, flushing pending Reverse Map state and invoking
/// `filter.expand` whenever `FilterFull` occurs (up to `MAX_QUOTIENT_BITS` times).
fn insert_or_expand_during_rebuild<F: RangeFilter>(
    filter: &mut F,
    reverse_map: &RocksDbReverseMap,
    state: &mut RebuildReconcileState<'_>,
    key: Key,
) -> AerisResult<Fingerprint> {
    let mut expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if expansions < MAX_QUOTIENT_BITS => {
                state.flush_before_expand(reverse_map)?;
                filter.expand(reverse_map)?;
                expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Computes the minimal `(old_fps_to_tombstone, new_entries_to_merge)` required to
/// bring `existing` Reverse Map entries into exact parity with `desired`.
fn compute_reverse_map_diff(
    existing: &BTreeMap<Fingerprint, BTreeSet<Key>>,
    desired: &BTreeMap<Fingerprint, BTreeSet<Key>>,
) -> (Vec<Fingerprint>, Vec<(Fingerprint, Key)>) {
    let mut to_tombstone = Vec::new();
    let mut tombstoned_set = BTreeSet::new();

    for (fp, existing_keys) in existing {
        let needs_tombstone = match desired.get(fp) {
            None => true,
            Some(desired_keys) => !existing_keys.is_subset(desired_keys),
        };
        if needs_tombstone {
            to_tombstone.push(fp.clone());
            tombstoned_set.insert(fp.clone());
        }
    }

    let mut to_merge = Vec::new();
    for (fp, desired_keys) in desired {
        let existing_keys = if tombstoned_set.contains(fp) {
            None
        } else {
            existing.get(fp)
        };
        for &key in desired_keys {
            let already_present = existing_keys.is_some_and(|set| set.contains(&key));
            if !already_present {
                to_merge.push((fp.clone(), key));
            }
        }
    }

    (to_tombstone, to_merge)
}

/// Reconciles discrepancies between [`ReferenceKvStore`] and [`RocksDbReverseMap`]
/// after WAL replay. Returns `Ok(true)` if any corrective writes were performed.
fn reconcile_kv_and_reverse_map(
    wal: &CoupledWal,
    kv_store: &ReferenceKvStore,
    reverse_map: &RocksDbReverseMap,
) -> AerisResult<bool> {
    let kv_keys: BTreeSet<Key> = kv_store.scan_all_keys()?.into_iter().collect();
    let rm_entries = reverse_map.scan_all_entries()?;
    let mut rm_keys = BTreeSet::new();
    for keys in rm_entries.values() {
        rm_keys.extend(keys.iter().copied());
    }

    let orphans: BTreeSet<Key> = rm_keys.difference(&kv_keys).copied().collect();
    let missing: BTreeSet<Key> = kv_keys.difference(&rm_keys).copied().collect();

    if orphans.is_empty() && missing.is_empty() {
        return Ok(false);
    }

    warn!(
        kv_count = kv_keys.len(),
        rm_count = rm_keys.len(),
        orphan_rm_keys = orphans.len(),
        missing_rm_keys = missing.len(),
        "reconciling discrepancy between ReferenceKvStore and RocksDbReverseMap"
    );

    let mut mutated = false;
    if !orphans.is_empty() {
        for (fp, keys) in &rm_entries {
            for &key in keys {
                if orphans.contains(&key) {
                    reverse_map.remove_key(fp, key)?;
                    mutated = true;
                }
            }
        }
    }

    if !missing.is_empty() {
        let recovered_fps = recover_latest_fps_from_wal(wal, &missing)?;
        let mut batch = Vec::with_capacity(recovered_fps.len());
        let mut unresolved = 0usize;
        for &key in &missing {
            if let Some(fp) = recovered_fps.get(&key) {
                batch.push((fp.clone(), key));
            } else {
                unresolved += 1;
            }
        }
        if !batch.is_empty() {
            reverse_map.append_batch(&batch)?;
            mutated = true;
        }
        if unresolved > 0 {
            warn!(
                unresolved_keys = unresolved,
                "some ReferenceKvStore keys lacked WAL fingerprint history; invoke rebuild_filter_and_reconcile to re-index"
            );
        }
    }

    Ok(mutated)
}

/// Scans retained WAL history to determine the latest [`Fingerprint`] associated with
/// each key in `target_keys`.
fn recover_latest_fps_from_wal(
    wal: &CoupledWal,
    target_keys: &BTreeSet<Key>,
) -> AerisResult<BTreeMap<Key, Fingerprint>> {
    let all_records = wal.replay_from(1)?;
    let mut latest: BTreeMap<Key, Fingerprint> = BTreeMap::new();
    for rec in all_records {
        match rec.op {
            WalOp::Put { key, fp, .. } if target_keys.contains(&key) => {
                latest.insert(key, fp);
            }
            WalOp::Delete { key, .. } if target_keys.contains(&key) => {
                latest.remove(&key);
            }
            WalOp::Adapt { new_entries, .. } => {
                for (fp, key) in new_entries {
                    if target_keys.contains(&key) {
                        latest.insert(key, fp);
                    }
                }
            }
            WalOp::Put { .. } | WalOp::Delete { .. } | WalOp::Checkpoint { .. } => {}
        }
    }
    Ok(latest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeris_types::PartitionHasher;
    use tempfile::TempDir;

    fn test_fp(key: Key, bits: u16) -> Fingerprint {
        let hasher = PartitionHasher::new(0x570A_6E01);
        Fingerprint::from_hasher(&hasher, key >> 5, bits)
    }

    #[test]
    fn default_config_uses_group_commit_1000_and_rejects_zero_batch_size() {
        let default_cfg = StorageConfig::default();
        assert_eq!(
            default_cfg.wal_sync_policy,
            WalSyncPolicy::GroupCommit {
                batch_size: DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE
            }
        );
        assert_eq!(DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE, 1_000);
        assert!(default_cfg.wal_sync_policy.validate().is_ok());

        let strict_cfg = StorageConfig::strict_per_op_fsync();
        assert_eq!(strict_cfg.wal_sync_policy, WalSyncPolicy::PerOpFsync);
        assert!(strict_cfg.wal_sync_policy.validate().is_ok());

        let invalid_cfg = StorageConfig::default()
            .with_wal_sync_policy(WalSyncPolicy::GroupCommit { batch_size: 0 });
        assert!(matches!(
            invalid_cfg.wal_sync_policy.validate(),
            Err(AerisError::InvalidConfig(_))
        ));

        let dir = TempDir::new().unwrap();
        assert!(matches!(
            CoupledStorageEngine::open_and_recover_with_config(dir.path(), invalid_cfg),
            Err(AerisError::InvalidConfig(_))
        ));
    }

    #[test]
    fn group_commit_batch_boundary_flush_checkpoint_and_drop_sync() {
        let dir = TempDir::new().unwrap();
        let cfg = StorageConfig::default()
            .with_wal_sync_policy(WalSyncPolicy::GroupCommit { batch_size: 4 });
        {
            let (engine, _) =
                CoupledStorageEngine::open_and_recover_with_config(dir.path(), cfg).unwrap();
            assert_eq!(
                engine.wal_sync_policy(),
                WalSyncPolicy::GroupCommit { batch_size: 4 }
            );
            assert_eq!(engine.pending_sync_ops(), 0);

            // Ops 1, 2, 3 accumulate pending_sync_ops; op 4 hits batch_size == 4 and syncs.
            let fp1 = test_fp(10, 22);
            let fp2 = test_fp(20, 22);
            let fp3 = test_fp(30, 22);
            let fp4 = test_fp(40, 22);

            engine.put(10, b"v10", &fp1).unwrap();
            assert_eq!(engine.pending_sync_ops(), 1);
            engine.put(20, b"v20", &fp2).unwrap();
            assert_eq!(engine.pending_sync_ops(), 2);
            engine.put(30, b"v30", &fp3).unwrap();
            assert_eq!(engine.pending_sync_ops(), 3);
            engine.put(40, b"v40", &fp4).unwrap();
            assert_eq!(engine.pending_sync_ops(), 0);

            // Delete and adapt also increment pending_sync_ops; flush_wal resets to 0.
            engine.delete(40, &fp4).unwrap();
            assert_eq!(engine.pending_sync_ops(), 1);
            let ext_fp3 = test_fp(30, 34);
            engine
                .record_adapt(
                    vec![fp3.clone()],
                    vec![(ext_fp3.clone(), 30)],
                    QueryRange::new(100, 131).unwrap(),
                )
                .unwrap();
            assert_eq!(engine.pending_sync_ops(), 2);

            engine.flush_wal().unwrap();
            assert_eq!(engine.pending_sync_ops(), 0);

            // Checkpoint resets pending_sync_ops to 0.
            let fp5 = test_fp(50, 22);
            engine.put(50, b"v50", &fp5).unwrap();
            assert_eq!(engine.pending_sync_ops(), 1);
            engine.checkpoint().unwrap();
            assert_eq!(engine.pending_sync_ops(), 0);

            // Leave 1 pending op to exercise best-effort sync on Drop.
            let fp6 = test_fp(60, 22);
            engine.put(60, b"v60", &fp6).unwrap();
            assert_eq!(engine.pending_sync_ops(), 1);
        }

        // Reopen and confirm all operations (including post-checkpoint op synced on Drop) survived.
        let (reopened, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        assert_eq!(report.recovered_puts, 1);
        assert_eq!(report.live_key_count, 5);
        assert_eq!(
            reopened.get(10).unwrap().as_deref(),
            Some(b"v10".as_slice())
        );
        assert_eq!(reopened.get(40).unwrap(), None);
        assert_eq!(
            reopened.get(50).unwrap().as_deref(),
            Some(b"v50".as_slice())
        );
        assert_eq!(
            reopened.get(60).unwrap().as_deref(),
            Some(b"v60".as_slice())
        );
        assert_eq!(
            reopened.reverse_map().get_keys(&test_fp(30, 34)).unwrap(),
            vec![30]
        );
    }

    #[test]
    fn put_batch_persists_atomically_with_single_sync_and_recovers_from_wal() {
        let dir = TempDir::new().unwrap();
        let fps: Vec<Fingerprint> = (100..150u64).map(|k| test_fp(k, 22)).collect();
        let vals: Vec<Vec<u8>> = (100..150u64)
            .map(|k| format!("batch-val-{k}").into_bytes())
            .collect();

        {
            let (engine, _) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();

            // Empty batch returns current last_lsn without advancing LSN.
            assert_eq!(engine.put_batch(&[]).unwrap(), 0);
            assert_eq!(engine.last_lsn(), 0);

            // Seed one unsynced put so we verify put_batch resets pending_sync_ops to 0.
            let seed_fp = test_fp(1, 22);
            let lsn1 = engine.put(1, b"seed", &seed_fp).unwrap();
            assert_eq!(lsn1, 1);
            assert_eq!(engine.pending_sync_ops(), 1);

            let entries: Vec<(Key, &[u8], &Fingerprint)> = (0..50usize)
                .map(|i| (100 + i as u64, vals[i].as_slice(), &fps[i]))
                .collect();
            let highest_lsn = engine.put_batch(&entries).unwrap();
            assert_eq!(highest_lsn, 51);
            assert_eq!(engine.last_lsn(), 51);
            assert_eq!(engine.pending_sync_ops(), 0);

            // Empty batch after writes returns current last_lsn.
            assert_eq!(engine.put_batch(&[]).unwrap(), 51);

            for (i, &(k, v, fp)) in entries.iter().enumerate() {
                assert_eq!(engine.get(k).unwrap().as_deref(), Some(v));
                assert!(engine.reverse_map().get_keys(fp).unwrap().contains(&k));
                let _ = i;
            }
        }

        // Wipe both RocksDB directories and prove full WAL replay reconstructs every batched key!
        fs::remove_dir_all(dir.path().join(KV_DB_DIR_NAME)).unwrap();
        fs::remove_dir_all(dir.path().join(REVERSE_MAP_DB_DIR_NAME)).unwrap();

        let (reopened, report) = CoupledStorageEngine::open_and_recover(dir.path()).unwrap();
        assert_eq!(report.recovered_puts, 51);
        assert_eq!(report.live_key_count, 51);
        assert_eq!(
            reopened.get(1).unwrap().as_deref(),
            Some(b"seed".as_slice())
        );
        for i in 0..50usize {
            let k = 100 + i as u64;
            assert_eq!(
                reopened.get(k).unwrap().as_deref(),
                Some(vals[i].as_slice())
            );
            assert!(reopened
                .reverse_map()
                .get_keys(&fps[i])
                .unwrap()
                .contains(&k));
        }
    }

    #[test]
    fn strict_per_op_fsync_keeps_pending_sync_ops_at_zero() {
        let dir = TempDir::new().unwrap();
        let (engine, _) = CoupledStorageEngine::open_and_recover_with_config(
            dir.path(),
            StorageConfig::strict_per_op_fsync(),
        )
        .unwrap();
        assert_eq!(engine.wal_sync_policy(), WalSyncPolicy::PerOpFsync);

        let fp1 = test_fp(7, 22);
        let fp2 = test_fp(7, 34);
        engine.put(7, b"strict", &fp1).unwrap();
        assert_eq!(engine.pending_sync_ops(), 0);
        engine
            .record_adapt(
                vec![fp1.clone()],
                vec![(fp2.clone(), 7)],
                QueryRange::point(8),
            )
            .unwrap();
        assert_eq!(engine.pending_sync_ops(), 0);
        engine.delete(7, &fp2).unwrap();
        assert_eq!(engine.pending_sync_ops(), 0);
    }
}
