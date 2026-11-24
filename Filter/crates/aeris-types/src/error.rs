//! Unified error hierarchy for all planes of the Aeris Range Filtering System.

use crate::range::{Key, Lsn, ShardId};
use thiserror::Error;

/// Result alias parameterized by [`AerisError`].
pub type AerisResult<T> = Result<T, AerisError>;

/// Error conditions surfaced across Data, Storage, Control, Scale, and Sidecar planes.
#[derive(Debug, Error)]
pub enum AerisError {
    /// A configuration invariant (e.g., `N = 2^q`, `r + log2(P) = C`, Goswami bound)
    /// was violated.
    #[error("invalid filter configuration: {0}")]
    InvalidConfig(String),

    /// A closed range query `[start, end]` had `start > end`.
    #[error("invalid query range: start ({start}) > end ({end})")]
    InvalidRange {
        /// Inclusive lower bound supplied by the caller.
        start: u64,
        /// Inclusive upper bound supplied by the caller.
        end: u64,
    },

    /// The Quotient Filter reached its maximum load factor and must expand before
    /// accepting additional slots.
    #[error("filter full: load factor {load_factor:.4} at capacity {capacity}")]
    FilterFull {
        /// Current slot occupancy ratio.
        load_factor: f64,
        /// Total slot capacity `N = 2^q`.
        capacity: usize,
    },

    /// A requested key was not present in the filter or store.
    #[error("key not found: {0}")]
    KeyNotFound(Key),

    /// Failure interacting with the underlying Reverse Map or key-value store.
    #[error("storage plane error: {0}")]
    StorageError(String),

    /// Corrupted WAL frame detected via checksum mismatch or invalid encoding.
    #[error("WAL corruption at LSN {lsn}: {reason}")]
    WalCorruption {
        /// Sequence number of the corrupted WAL record (or 0 if header was unreadable).
        lsn: Lsn,
        /// Diagnostic description of the corruption.
        reason: String,
    },

    /// Underlying filesystem I/O failure on WAL, FP Log, or checkpoint files.
    #[error("WAL/log I/O error: {0}")]
    WalIoError(String),

    /// An asynchronous rebuild or expansion is already in flight.
    #[error("background rebuild already in progress")]
    RebuildInProgress,

    /// An ML/RL advisory proposal violated one or more deterministic safety guardrails.
    #[error("safety guardrail rejected proposal: {reason}")]
    GuardrailRejected {
        /// Specific invariant violated by the proposal.
        reason: String,
    },

    /// The out-of-process gRPC sidecar could not be reached or returned an error.
    #[error("control plane sidecar unavailable: {0}")]
    SidecarUnavailable(String),

    /// An RPC to the out-of-process sidecar exceeded its configured deadline.
    #[error("control plane sidecar RPC timed out after {timeout_ms} ms")]
    SidecarTimeout {
        /// Configured RPC deadline in milliseconds.
        timeout_ms: u64,
    },

    /// A shard identifier was not found in the active range-sharded routing table.
    #[error("shard not found: {0}")]
    ShardNotFound(ShardId),

    /// Serialization or deserialization failure on canonical binary formats.
    #[error("encoding error: {0}")]
    EncodingError(String),

    /// An internal data-structure invariant was unexpectedly violated.
    #[error("internal invariant violated: {0}")]
    InternalInvariant(String),
}
