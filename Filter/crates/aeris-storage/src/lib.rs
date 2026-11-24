//! # `aeris-storage`
//!
//! Production Storage Plane for the **Aeris-Based Optimized Range Filtering System**,
//! providing:
//! - [`RocksDbReverseMap`] (`ReverseMapStore`): write-optimized disk-resident Reverse
//!   Map backed by RocksDB with a custom key-list `MergeOperator`
//!   (`"aeris_reverse_map_keylist_v1"`), native tombstones, compaction, and streaming
//!   deduplicated iteration.
//! - [`CoupledWal`] (`WalEngine`): binary-framed, CRC32-verified, `fsync`-ordered
//!   Write-Ahead Log with torn-tail truncation, mid-log corruption detection, and
//!   crash-safe atomic compaction (`truncate_before`).
//! - [`ReferenceKvStore`]: persistent big-endian RocksDB primary key-value store
//!   supporting `O(log N)` closed range emptiness queries.
//! - [`CoupledStorageEngine`]: unified WAL-first coordinator coupling `CoupledWal`,
//!   `ReferenceKvStore`, and `RocksDbReverseMap` with idempotent crash-recovery replay
//!   and zero-false-negative [`aeris_types::RangeFilter`] rebuild/reconciliation.
//!
//! # Crash-Recovery Invariants
//!
//! Every persistence path in this crate obeys five strict durability invariants:
//!
//! 1. **Strict WAL-First Ordering & Configurable `fsync` Policy:**
//!    Every acknowledged mutation (`put`, `delete`, `record_adapt`, `put_batch`) acquires
//!    the engine's serialization lock and writes a checksummed frame with a strictly
//!    monotonic LSN to [`CoupledWal`]'s OS file descriptor **before** mutating
//!    [`ReferenceKvStore`] or [`RocksDbReverseMap`]. Under the default
//!    [`WalSyncPolicy::GroupCommit`] (`B = DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE = 1_000`),
//!    `fsync` barriers are amortized across `B` mutations (and issued immediately on
//!    [`CoupledStorageEngine::put_batch`], [`CoupledStorageEngine::flush_wal`],
//!    [`CoupledStorageEngine::checkpoint`], and clean `Drop`). Callers requiring
//!    synchronous per-operation `fsync` across power loss may opt into
//!    [`StorageConfig::strict_per_op_fsync`] ([`WalSyncPolicy::PerOpFsync`]).
//!
//! 2. **CRC32 Frame Checksum Validation:**
//!    Every WAL frame begins with a 21-byte header
//!    `[magic: b"AWAL"][version: 1u8][payload_len: u32 LE][lsn: u64 LE][crc32: u32 LE]`
//!    followed by `payload_len` bytes. The IEEE CRC32 (`crc32fast`) covers
//!    `[version, payload_len_le, lsn_le, payload]`, detecting partial flushes,
//!    bit flips, and truncated writes.
//!
//! 3. **Torn-Tail Truncation vs. Mid-Log Corruption:**
//!    - Incomplete tail headers (`< 21` bytes at EOF), zero-filled OS preallocation
//!      tails, truncated tail payloads, and full tail frames failing CRC32 right at
//!      EOF are treated as interrupted writes from a sudden crash (`SIGKILL` /
//!      `TerminateProcess`), truncated cleanly to the last intact frame boundary,
//!      and `fsync`-ed with a structured `tracing::warn!`.
//!    - Mid-log CRC32 failures, corrupted magic/version bytes, non-monotonic LSNs,
//!      and corrupted non-tail length fields return [`aeris_types::AerisError::WalCorruption`]
//!      rather than silently discarding committed history.
//!
//! 4. **WAL Replay Idempotency:**
//!    - Replaying `WalOp::Put` overwrites the key in `ReferenceKvStore` and emits
//!      `AddKey(key)` into `RocksDbReverseMap`, where the custom `MergeOperator`
//!      folds all operands into a canonical `BTreeSet<Key>`.
//!    - Replaying `WalOp::Delete` tombstones the key in `ReferenceKvStore` and emits
//!      `DelKey(key)` into `RocksDbReverseMap` (no-op if absent).
//!    - Replaying `WalOp::Adapt` atomically tombstones `old_fps` and merges
//!      `new_entries` in a single RocksDB `WriteBatch`.
//!    - Repeated replays (including crashes mid-recovery) converge deterministically
//!      to the exact same durable state.
//!
//! 5. **RocksDB Durability, Checkpointing & Reconciliation:**
//!    - `checkpoint()` flushes both `ReferenceKvStore` and `RocksDbReverseMap`
//!      memtables to durable SSTables **before** appending and `fsync`-ing
//!      `WalOp::Checkpoint { up_to_lsn }`.
//!    - `truncate_before()` compacts WAL history via write-temp + `sync_all` +
//!      atomic rename while retaining a checkpoint anchor so LSNs never regress.
//!    - `open_and_recover()` replays all post-checkpoint WAL records and reconciles
//!      `ReferenceKvStore` and `RocksDbReverseMap`; `rebuild_filter_and_reconcile()`
//!      repopulates an in-memory `RangeFilter` and aligns Reverse Map fingerprints
//!      with zero lost keys and zero false negatives.
//!
//! # Lock Ordering
//! `CoupledStorageEngine::mutation_lock` -> `CoupledWal::inner` -> RocksDB internal locks.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod codec;
pub mod engine;
pub mod kv_store;
pub mod reverse_map;
pub mod wal;

pub use aeris_types;
pub use codec::{
    compute_frame_crc32, decode_key_list, decode_wal_op, encode_add_key_operand,
    encode_compacted_snapshot, encode_del_key_operand, encode_wal_frame, encode_wal_op,
    fold_full_merge, fold_partial_merge, WalFrameHeader, MAX_WAL_PAYLOAD_LEN, RM_TAG_ADD_KEY,
    RM_TAG_COMPACTED_SNAPSHOT, RM_TAG_DEL_KEY, WAL_FORMAT_VERSION, WAL_HEADER_LEN, WAL_MAGIC,
};
pub use engine::{
    CoupledStorageEngine, RecoveryReport, StorageConfig, WalSyncPolicy,
    DEFAULT_WAL_GROUP_COMMIT_BATCH_SIZE, KV_DB_DIR_NAME, REVERSE_MAP_DB_DIR_NAME, WAL_FILE_NAME,
};
pub use kv_store::{KvStoreConfig, ReferenceKvStore, DEFAULT_KV_WRITE_BUFFER_SIZE_BYTES};
pub use reverse_map::{
    ReverseMapConfig, RocksDbReverseMap, DEFAULT_MAX_WRITE_BUFFER_NUMBER,
    DEFAULT_WRITE_BUFFER_SIZE_BYTES, MERGE_OPERATOR_NAME,
};
pub use wal::CoupledWal;
