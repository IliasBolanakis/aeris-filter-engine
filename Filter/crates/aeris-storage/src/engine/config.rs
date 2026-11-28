//! Configuration and recovery diagnostic types for [`super::CoupledStorageEngine`].

use crate::kv_store::KvStoreConfig;
use crate::reverse_map::ReverseMapConfig;
use aeris_types::{AerisError, AerisResult, Lsn};

/// Default group-commit WAL batch size (`B = 1,000`) amortizing `fsync` latency
/// while preserving strict WAL-first OS kernel write ordering on every operation.
pub const DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE: usize = 1_000;

/// Durability barrier policy governing when [`super::CoupledStorageEngine`] invokes
/// `CoupledWal::sync()` (`fsync` / `FlushFileBuffers`) after writing WAL frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalSyncPolicy {
    /// Amortizes `fsync` across `batch_size` WAL-appended mutations (default
    /// [`DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE`] = `1_000`). Every mutation still
    /// writes its CRC32 frame to the OS file descriptor before RocksDB is mutated,
    /// and [`super::CoupledStorageEngine::put_batch`],
    /// [`super::CoupledStorageEngine::flush_wal`],
    /// [`super::CoupledStorageEngine::checkpoint`], and clean `Drop` force an immediate sync.
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

/// Configuration bundle for [`super::CoupledStorageEngine`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageConfig {
    /// Configuration for the primary [`crate::kv_store::ReferenceKvStore`].
    pub kv_store: KvStoreConfig,
    /// Configuration for the write-optimized [`crate::reverse_map::RocksDbReverseMap`].
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

/// Diagnostic summary returned by [`super::CoupledStorageEngine::open_and_recover`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Total number of mutation WAL records replayed (`Put` + `Delete` + `Adapt`).
    pub replayed_records: usize,
    /// Number of [`aeris_types::WalOp::Put`] records replayed during recovery.
    pub recovered_puts: usize,
    /// Number of [`aeris_types::WalOp::Delete`] records replayed during recovery.
    pub recovered_deletes: usize,
    /// Number of [`aeris_types::WalOp::Adapt`] records replayed during recovery.
    pub recovered_adapts: usize,
    /// Number of torn-tail WAL bytes truncated and `fsync`-ed during recovery.
    pub truncated_tail_bytes: u64,
    /// Highest intact LSN present in the WAL after recovery (`0` if empty).
    pub last_lsn: Lsn,
    /// Exact count of live keys present in [`crate::kv_store::ReferenceKvStore`] after recovery.
    pub live_key_count: usize,
}
