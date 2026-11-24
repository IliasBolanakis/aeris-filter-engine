//! Write-Ahead Log (WAL) record schemas and False-Positive Log entry types.
//!
//! # Crash-Recovery Ordering Contract
//! 1. Mutations (`Put`, `Delete`, `Adapt`) are serialized into checksummed WAL frames
//!    with strictly monotonic [`Lsn`] values and flushed with `fsync` **before**
//!    acknowledging writes to the caller or advancing the durable checkpoint.
//! 2. All WAL operations are replay-idempotent with respect to both the primary
//!    reference key-value store and the Reverse Map (`ReverseMapStore`).
//! 3. Corrupt or torn tail frames written during a `SIGKILL` crash fail CRC32
//!    verification and are truncated cleanly on recovery while preserving all
//!    previously synced prefixes.

use crate::fingerprint::Fingerprint;
use crate::range::{Key, Lsn, QueryRange};
use serde::{Deserialize, Serialize};

/// Atomic mutation coupled across the primary store, Reverse Map, and Aeris Filter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalOp {
    /// Insert or update a `(key, value)` pair along with its initial filter fingerprint.
    Put {
        /// 64-bit unsigned integer key.
        key: Key,
        /// Associated payload value stored in the primary reference KV store.
        value: Vec<u8>,
        /// Canonical fingerprint under which `key` was indexed in the Reverse Map.
        fp: Fingerprint,
    },
    /// Delete a key from the primary store and remove its Reverse Map association.
    Delete {
        /// 64-bit unsigned integer key to remove.
        key: Key,
        /// Canonical fingerprint under which `key` was indexed.
        fp: Fingerprint,
    },
    /// Record an atomic fingerprint split/adaptation produced by resolving a false positive.
    Adapt {
        /// Original colliding fingerprints tombstoned in the Reverse Map.
        old_fps: Vec<Fingerprint>,
        /// Extended `(fingerprint, key)` mappings inserted into the Reverse Map.
        new_entries: Vec<(Fingerprint, Key)>,
        /// False-positive query range that triggered the adaptation.
        range: QueryRange,
    },
    /// Durable checkpoint marker confirming all prior LSNs are persisted to storage.
    Checkpoint {
        /// Highest LSN guaranteed durable in both primary KV and Reverse Map.
        up_to_lsn: Lsn,
    },
}

/// Persisted WAL frame containing sequence number, CRC32 checksum, and operation payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalRecord {
    /// Strictly monotonic log sequence number starting at 1.
    pub lsn: Lsn,
    /// CRC32C/IEEE checksum covering the serialized `(lsn, op)` payload bytes.
    pub checksum: u32,
    /// Logged mutation payload.
    pub op: WalOp,
}

/// Append-only entry recorded in the False-Positive Log for monotonic replay across
/// asynchronous `R` rebuilds (`F_old -> F_new`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpLogEntry {
    /// Monotonic sequence number of the recorded false positive.
    pub seq: u64,
    /// Empty query interval `[a, b]` that triggered a false positive on `F_old`.
    pub range: QueryRange,
    /// Unix epoch timestamp in milliseconds when the false positive was observed.
    pub timestamp_ms: u64,
}
