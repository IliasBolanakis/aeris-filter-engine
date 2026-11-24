//! Persistent RocksDB-backed reference primary key-value store.
//!
//! Keys are stored as fixed 8-byte big-endian unsigned integers (`key.to_be_bytes()`),
//! making RocksDB's native lexicographical byte ordering strictly identical to numeric
//! `u64` ordering. Consequently, [`ReferenceKvStore::range_non_empty`] answers arbitrary
//! closed range emptiness checks `[a, b]` via a single `O(log N)` LSM seek without
//! scanning the full keyspace.

use aeris_types::{AerisError, AerisResult, Key, QueryRange};
use rocksdb::{DBCompressionType, Direction, IteratorMode, Options, WriteOptions, DB};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info};

/// Fixed byte length of a big-endian 64-bit unsigned integer key in RocksDB.
const KEY_BYTES_LEN: usize = 8;

/// Default memtable write buffer size for [`ReferenceKvStore`] (16 MiB).
pub const DEFAULT_KV_WRITE_BUFFER_SIZE_BYTES: usize = 16 * 1024 * 1024;

/// Configuration options for [`ReferenceKvStore`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvStoreConfig {
    /// Per-memtable write buffer size in bytes.
    pub write_buffer_size: usize,
    /// Enables LZ4 compression for SSTables when `true`.
    pub enable_lz4_compression: bool,
    /// When `true`, every `put`/`delete` invokes RocksDB `fsync`.
    /// Set to `false` when coupled under [`crate::engine::CoupledStorageEngine`],
    /// where [`crate::wal::CoupledWal`] enforces strict write-ahead `fsync`.
    pub sync_writes: bool,
}

impl Default for KvStoreConfig {
    fn default() -> Self {
        Self {
            write_buffer_size: DEFAULT_KV_WRITE_BUFFER_SIZE_BYTES,
            enable_lz4_compression: true,
            sync_writes: false,
        }
    }
}

impl KvStoreConfig {
    /// Configuration optimized for operation beneath [`crate::wal::CoupledWal`].
    #[must_use]
    pub fn for_wal_coupled() -> Self {
        Self {
            sync_writes: false,
            ..Self::default()
        }
    }

    /// Configuration with per-write synchronous RocksDB durability.
    #[must_use]
    pub fn standalone_durable() -> Self {
        Self {
            sync_writes: true,
            ..Self::default()
        }
    }
}

/// Persistent RocksDB primary key-value store backing false-positive verification
/// and crash-recovery ground truth.
///
/// # Key Ordering Invariant
/// Every `Key` (`u64`) is serialized in big-endian byte order (`u64::to_be_bytes`).
/// Therefore, for all `a, b: Key`, `a < b` if and only if `a.to_be_bytes() < b.to_be_bytes()`.
pub struct ReferenceKvStore {
    db: DB,
    path: PathBuf,
    sync_writes: bool,
}

impl ReferenceKvStore {
    /// Opens or creates a [`ReferenceKvStore`] at `path` with default [`KvStoreConfig`].
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if RocksDB fails to open `path`.
    pub fn open(path: impl AsRef<Path>) -> AerisResult<Self> {
        Self::open_with_config(path, KvStoreConfig::default())
    }

    /// Opens or creates a [`ReferenceKvStore`] at `path` using explicit `config`.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if RocksDB fails to open `path`.
    pub fn open_with_config(path: impl AsRef<Path>, config: KvStoreConfig) -> AerisResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let start = Instant::now();

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_write_buffer_size(config.write_buffer_size);
        opts.set_compression_type(if config.enable_lz4_compression {
            DBCompressionType::Lz4
        } else {
            DBCompressionType::None
        });

        let db = DB::open(&opts, &path_buf).map_err(|err| {
            error!(path = %path_buf.display(), error = %err, "failed to open ReferenceKvStore");
            AerisError::StorageError(format!(
                "failed to open ReferenceKvStore at '{}': {err}",
                path_buf.display()
            ))
        })?;

        info!(
            path = %path_buf.display(),
            sync_writes = config.sync_writes,
            lz4 = config.enable_lz4_compression,
            elapsed_us = start.elapsed().as_micros(),
            "opened ReferenceKvStore"
        );

        Ok(Self {
            db,
            path: path_buf,
            sync_writes: config.sync_writes,
        })
    }

    /// Returns the filesystem directory path backing this KV store.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Inserts or overwrites `key -> value`.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if the RocksDB write fails.
    ///
    /// # Complexity
    /// `O(log N)` amortized LSM memtable insertion.
    pub fn put(&self, key: Key, value: &[u8]) -> AerisResult<()> {
        self.db
            .put_opt(key.to_be_bytes(), value, &self.write_options())
            .map_err(|err| {
                error!(path = %self.path.display(), key, error = %err, "ReferenceKvStore put failed");
                AerisError::StorageError(format!("ReferenceKvStore put({key}) failed: {err}"))
            })
    }

    /// Deletes `key` if present (idempotent no-op if absent).
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if the RocksDB delete fails.
    ///
    /// # Complexity
    /// `O(log N)` amortized LSM tombstone write.
    pub fn delete(&self, key: Key) -> AerisResult<()> {
        self.db
            .delete_opt(key.to_be_bytes(), &self.write_options())
            .map_err(|err| {
                error!(path = %self.path.display(), key, error = %err, "ReferenceKvStore delete failed");
                AerisError::StorageError(format!("ReferenceKvStore delete({key}) failed: {err}"))
            })
    }

    /// Retrieves the stored value for `key`, or `Ok(None)` if absent.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB read failure.
    pub fn get(&self, key: Key) -> AerisResult<Option<Vec<u8>>> {
        self.db.get(key.to_be_bytes()).map_err(|err| {
            error!(path = %self.path.display(), key, error = %err, "ReferenceKvStore get failed");
            AerisError::StorageError(format!("ReferenceKvStore get({key}) failed: {err}"))
        })
    }

    /// Returns `Ok(true)` if at least one live key exists in the closed interval
    /// `range = [start, end]`, or `Ok(false)` if the interval is empty.
    ///
    /// Exploits big-endian key encoding to seek directly to `range.start()` and inspect
    /// only the first live key at or above `range.start()`.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB iteration failure or
    /// [`AerisError::EncodingError`] if a stored key is not 8 bytes wide.
    ///
    /// # Complexity
    /// `O(log N)` time (single RocksDB seek, zero full-table scan).
    pub fn range_non_empty(&self, range: QueryRange) -> AerisResult<bool> {
        let start_bytes = range.start().to_be_bytes();
        let mut iter = self
            .db
            .iterator(IteratorMode::From(&start_bytes, Direction::Forward));
        match iter.next() {
            None => Ok(false),
            Some(Ok((raw_key, _raw_val))) => {
                let first_key = decode_be_key(&raw_key)?;
                Ok(first_key <= range.end())
            }
            Some(Err(err)) => {
                error!(
                    path = %self.path.display(),
                    start = range.start(),
                    end = range.end(),
                    error = %err,
                    "ReferenceKvStore range_non_empty failed"
                );
                Err(AerisError::StorageError(format!(
                    "ReferenceKvStore range_non_empty([{start}, {end}]) failed: {err}",
                    start = range.start(),
                    end = range.end()
                )))
            }
        }
    }

    /// Scans and returns all live keys in strictly ascending numeric order.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB iteration failure or
    /// [`AerisError::EncodingError`] if any key is malformed.
    ///
    /// # Complexity
    /// `O(N)` time and `O(N)` space.
    pub fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        let mut keys = Vec::new();
        for item in self.db.iterator(IteratorMode::Start) {
            let (raw_key, _raw_val) = item.map_err(|err| {
                error!(path = %self.path.display(), error = %err, "ReferenceKvStore scan_all_keys failed");
                AerisError::StorageError(format!("ReferenceKvStore scan_all_keys failed: {err}"))
            })?;
            keys.push(decode_be_key(&raw_key)?);
        }
        Ok(keys)
    }

    /// Flushes in-memory RocksDB memtables to immutable SSTables on disk.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] if RocksDB flush fails.
    pub fn flush(&self) -> AerisResult<()> {
        let start = Instant::now();
        debug!(path = %self.path.display(), "flushing ReferenceKvStore memtables");
        self.db.flush().map_err(|err| {
            error!(path = %self.path.display(), error = %err, "ReferenceKvStore flush failed");
            AerisError::StorageError(format!("ReferenceKvStore flush failed: {err}"))
        })?;
        info!(
            path = %self.path.display(),
            elapsed_us = start.elapsed().as_micros(),
            "flushed ReferenceKvStore"
        );
        Ok(())
    }

    /// Returns the exact count of live keys currently persisted in the store.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB iteration failure.
    ///
    /// # Complexity
    /// `O(N)` key scan.
    pub fn len(&self) -> AerisResult<usize> {
        let mut count = 0usize;
        for item in self.db.iterator(IteratorMode::Start) {
            item.map_err(|err| {
                AerisError::StorageError(format!("ReferenceKvStore len failed: {err}"))
            })?;
            count += 1;
        }
        Ok(count)
    }

    /// Returns `Ok(true)` if the store contains zero live keys.
    ///
    /// # Errors
    /// Returns [`AerisError::StorageError`] on RocksDB iteration failure.
    ///
    /// # Complexity
    /// `O(1)` amortized (checks only the first iterator entry).
    pub fn is_empty(&self) -> AerisResult<bool> {
        let mut iter = self.db.iterator(IteratorMode::Start);
        match iter.next() {
            None => Ok(true),
            Some(Ok(_)) => Ok(false),
            Some(Err(err)) => Err(AerisError::StorageError(format!(
                "ReferenceKvStore is_empty failed: {err}"
            ))),
        }
    }

    #[inline]
    fn write_options(&self) -> WriteOptions {
        let mut wo = WriteOptions::default();
        wo.set_sync(self.sync_writes);
        wo
    }
}

fn decode_be_key(raw: &[u8]) -> AerisResult<Key> {
    if raw.len() != KEY_BYTES_LEN {
        return Err(AerisError::EncodingError(format!(
            "invalid ReferenceKvStore key length: expected {KEY_BYTES_LEN} bytes, got {}",
            raw.len()
        )));
    }
    Ok(u64::from_be_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn big_endian_ordering_and_range_non_empty_boundaries() {
        let dir = TempDir::new().unwrap();
        let kv = ReferenceKvStore::open(dir.path()).unwrap();
        assert!(kv.is_empty().unwrap());
        assert_eq!(kv.len().unwrap(), 0);
        assert!(!kv
            .range_non_empty(QueryRange::new(0, u64::MAX).unwrap())
            .unwrap());

        // Keys spanning u64 boundaries to verify big-endian numeric order (not little-endian!)
        let keys = [
            0u64,
            1,
            255,
            256,
            65_535,
            65_536,
            1 << 40,
            u64::MAX - 1,
            u64::MAX,
        ];
        for &k in keys.iter().rev() {
            kv.put(k, &k.to_le_bytes()).unwrap();
        }
        kv.flush().unwrap();

        assert!(!kv.is_empty().unwrap());
        assert_eq!(kv.len().unwrap(), keys.len());
        assert_eq!(kv.scan_all_keys().unwrap(), keys.to_vec());

        // Exact point checks
        for &k in &keys {
            assert_eq!(kv.get(k).unwrap(), Some(k.to_le_bytes().to_vec()));
            assert!(kv.range_non_empty(QueryRange::point(k)).unwrap());
        }

        // Empty gap checks
        assert!(!kv
            .range_non_empty(QueryRange::new(2, 254).unwrap())
            .unwrap());
        assert!(!kv
            .range_non_empty(QueryRange::new(257, 65_534).unwrap())
            .unwrap());
        assert!(!kv
            .range_non_empty(QueryRange::new((1 << 40) + 1, u64::MAX - 2).unwrap())
            .unwrap());

        // Overlapping range checks
        assert!(kv
            .range_non_empty(QueryRange::new(2, 255).unwrap())
            .unwrap());
        assert!(kv
            .range_non_empty(QueryRange::new(256, 500).unwrap())
            .unwrap());

        // Delete and verify absence
        kv.delete(255).unwrap();
        kv.delete(255).unwrap(); // idempotent delete
        assert_eq!(kv.get(255).unwrap(), None);
        assert!(!kv
            .range_non_empty(QueryRange::new(2, 255).unwrap())
            .unwrap());
        assert_eq!(kv.len().unwrap(), keys.len() - 1);
        kv.flush().unwrap();
        drop(kv);

        // Reopen and verify durability of surviving keys and deletion tombstone
        let reopened = ReferenceKvStore::open(dir.path()).unwrap();
        assert_eq!(reopened.len().unwrap(), keys.len() - 1);
        assert_eq!(reopened.get(255).unwrap(), None);
        assert_eq!(
            reopened.get(u64::MAX).unwrap(),
            Some(u64::MAX.to_le_bytes().to_vec())
        );
        assert!(reopened
            .range_non_empty(QueryRange::point(u64::MAX))
            .unwrap());
    }
}
