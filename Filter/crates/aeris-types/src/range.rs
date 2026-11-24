//! Core integer key domain and closed range query types.
//!
//! Per Goswami's lower bound (`Omega(log(R / epsilon))` bits/key), spatially robust
//! range filtering is strictly restricted to fixed-length 64-bit unsigned integer keys.

use crate::error::{AerisError, AerisResult};
use serde::{Deserialize, Serialize};

/// Fixed-length 64-bit unsigned integer key stored in the Aeris Range Filter.
pub type Key = u64;

/// Partition prefix identifier `p(k) = k >> m` for a key `k`.
pub type PartitionId = u64;

/// In-partition suffix `m(k) = k & ((1 << m) - 1)` for a key `k`.
pub type Memento = u64;

/// Unique identifier for a range shard in the Scale Plane.
pub type ShardId = u64;

/// Monotonic RCU / control-plane configuration epoch.
pub type Epoch = u64;

/// Strictly monotonic Write-Ahead Log Sequence Number (LSN).
pub type Lsn = u64;

/// Zero-based slot index into a Quotient Filter table of size `N = 2^q`.
pub type SlotIndex = usize;

/// Closed, inclusive range query `[start, end]` over 64-bit unsigned integer keys.
///
/// # Invariants
/// - `self.start <= self.end` always holds for any constructed `QueryRange`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QueryRange {
    start: Key,
    end: Key,
}

impl QueryRange {
    /// Constructs a validated closed interval `[start, end]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidRange`] if `start > end`.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    #[inline]
    pub const fn new(start: Key, end: Key) -> AerisResult<Self> {
        if start > end {
            return Err(AerisError::InvalidRange { start, end });
        }
        Ok(Self { start, end })
    }

    /// Constructs a point query `[key, key]` of length 1.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    #[inline]
    pub const fn point(key: Key) -> Self {
        Self {
            start: key,
            end: key,
        }
    }

    /// Constructs a closed interval `[start, start.saturating_add(len - 1)]`
    /// of exact integer length `len >= 1` (clamped at `u64::MAX`).
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `len == 0`.
    #[inline]
    pub fn with_len(start: Key, len: u64) -> AerisResult<Self> {
        if len == 0 {
            return Err(AerisError::InvalidConfig(
                "query range length must be at least 1".to_string(),
            ));
        }
        let end = start.saturating_add(len - 1);
        Ok(Self { start, end })
    }

    /// Returns the inclusive lower bound `a` of `[a, b]`.
    #[inline]
    pub const fn start(&self) -> Key {
        self.start
    }

    /// Returns the inclusive upper bound `b` of `[a, b]`.
    #[inline]
    pub const fn end(&self) -> Key {
        self.end
    }

    /// Returns the exact cardinality `end - start + 1` as a `u128` without overflow.
    #[inline]
    pub const fn len_u128(&self) -> u128 {
        (self.end as u128) - (self.start as u128) + 1
    }

    /// Returns the range length saturated to `u64::MAX`.
    #[inline]
    pub const fn len_saturated(&self) -> u64 {
        let len = self.len_u128();
        if len > u64::MAX as u128 {
            u64::MAX
        } else {
            len as u64
        }
    }

    /// Returns `true` if `key` lies within `[start, end]`.
    #[inline]
    pub const fn contains(&self, key: Key) -> bool {
        key >= self.start && key <= self.end
    }

    /// Returns `true` if `self` and `other` share at least one key.
    #[inline]
    pub const fn overlaps(&self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Splits the range into per-partition subqueries `(partition, low_memento, high_memento)`
    /// for a partition bit-width `m = log2(P)`.
    ///
    /// For any range of length `<= P = 2^m`, this yields at most 2 subqueries.
    ///
    /// # Panics
    /// Never panics; handles `m == 0` and `m >= 64` safely.
    ///
    /// # Complexity
    /// `O(ceil(len / P) + 1)` time and `O(1)` auxiliary space per yielded item.
    #[inline]
    pub fn partition_subqueries(&self, m: u8) -> PartitionSubqueryIter {
        let m_clamped = m.min(63);
        let mask = if m_clamped == 0 {
            0u64
        } else {
            (1u64 << m_clamped) - 1
        };
        let start_part = self.start >> m_clamped;
        let end_part = self.end >> m_clamped;
        PartitionSubqueryIter {
            current_part: start_part,
            end_part,
            start_part,
            start_memento: self.start & mask,
            end_memento: self.end & mask,
            max_memento: mask,
            done: false,
        }
    }
}

/// Zero-allocation iterator yielding `(PartitionId, low_memento, high_memento)`
/// subqueries covering a [`QueryRange`].
#[derive(Clone, Debug)]
pub struct PartitionSubqueryIter {
    current_part: PartitionId,
    end_part: PartitionId,
    start_part: PartitionId,
    start_memento: Memento,
    end_memento: Memento,
    max_memento: Memento,
    done: bool,
}

impl Iterator for PartitionSubqueryIter {
    type Item = (PartitionId, Memento, Memento);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let part = self.current_part;
        let low = if part == self.start_part {
            self.start_memento
        } else {
            0
        };
        let high = if part == self.end_part {
            self.end_memento
        } else {
            self.max_memento
        };
        if self.current_part == self.end_part {
            self.done = true;
        } else {
            self.current_part += 1;
        }
        Some((part, low, high))
    }
}
