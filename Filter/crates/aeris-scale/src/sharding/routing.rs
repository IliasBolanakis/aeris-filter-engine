//! Wait-free RCU routing snapshot ([`RoutingSnapshot`]) and contiguous key-space
//! boundary calculation helpers.

use super::shard::ShardInstance;
use aeris_types::{AerisError, AerisResult, Key, QueryRange, ShardBounds, ShardId};
use std::collections::HashSet;
use std::sync::Arc;

/// Threshold below which [`RoutingSnapshot::route_key_index`] executes a
/// branchless flat scan over 1–2 L1 cache lines (`<= 16` `u64` upper bounds),
/// avoiding branch mispredictions and `Arc` pointer chasing entirely.
const FLAT_LINEAR_SCAN_MAX_SHARDS: usize = 16;

/// Immutable, contiguous routing table snapshot published through [`crate::EpochCell`].
///
/// Stores a flat `max_keys: Box<[Key]>` array alongside `shards: Box<[Arc<ShardInstance>]>`
/// so point and range routing inspect 1–4 contiguous L1 cache lines with ZERO
/// `Arc` dereferences until the winning shard is selected.
///
/// # Invariants
/// 1. `shards` is non-empty and sorted strictly ascending by `bounds.min_key`.
/// 2. `max_keys.len() == shards.len()` and `max_keys[i] == shards[i].bounds.max_key`.
/// 3. `shards[0].bounds.min_key == 0` and `shards.last().unwrap().bounds.max_key == u64::MAX`.
/// 4. For every adjacent pair `(prev, next)`, `prev.bounds.max_key + 1 == next.bounds.min_key`
///    (zero gaps, zero overlaps).
/// 5. All `shard_id`s in `shards` are unique.
pub struct RoutingSnapshot {
    pub(super) max_keys: Box<[Key]>,
    pub(super) shards: Box<[Arc<ShardInstance>]>,
}

impl RoutingSnapshot {
    /// Constructs and validates a contiguous [`RoutingSnapshot`] covering `[0, u64::MAX]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] if any contiguity, ordering, or
    /// uniqueness invariant is violated.
    pub fn new(shards: Vec<Arc<ShardInstance>>) -> AerisResult<Self> {
        let max_keys: Box<[Key]> = shards.iter().map(|s| s.bounds.max_key).collect();
        let snap = Self {
            max_keys,
            shards: shards.into_boxed_slice(),
        };
        snap.verify_invariants()?;
        Ok(snap)
    }

    /// Returns all active shards in strict ascending `bounds.min_key` order.
    #[inline]
    #[must_use]
    pub fn shards(&self) -> &[Arc<ShardInstance>] {
        &self.shards
    }

    /// Wait-free index lookup on flat `max_keys` returning the unique shard index
    /// owning `key` with zero `Arc` pointer chasing.
    #[inline]
    #[must_use]
    pub(super) fn route_key_index(&self, key: Key) -> usize {
        let max_keys = &*self.max_keys;
        if max_keys.len() <= FLAT_LINEAR_SCAN_MAX_SHARDS {
            let mut idx = 0usize;
            for &mk in max_keys {
                idx += usize::from(mk < key);
            }
            idx
        } else {
            max_keys.partition_point(|&mk| mk < key)
        }
    }

    /// Wait-free `O(log S)` lookup returning the unique shard owning `key`.
    ///
    /// # Complexity
    /// `O(1)` L1 cache-line scan (`<= 16` shards) or `O(log S)` flat binary search,
    /// zero heap allocations, zero intermediate `Arc` dereferences.
    #[inline]
    #[must_use]
    pub fn route_key(&self, key: Key) -> &Arc<ShardInstance> {
        let idx = self.route_key_index(key);
        debug_assert!(
            idx < self.shards.len() && self.shards[idx].bounds.contains(key),
            "routing invariant violated: key {key} failed to resolve to a valid shard"
        );
        &self.shards[idx]
    }

    /// Wait-free `O(log S)` lookup returning the exact contiguous slice of
    /// shards intersecting `range`.
    ///
    /// # Complexity
    /// `O(log S)` time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn route_range(&self, range: QueryRange) -> &[Arc<ShardInstance>] {
        let first = self.route_key_index(range.start());
        if range.end() <= self.max_keys[first] {
            return &self.shards[first..=first];
        }
        let end_excl = self.route_range_end_excl(first, range.end());
        debug_assert!(
            first < end_excl && end_excl <= self.shards.len(),
            "routing invariant violated: range {range:?} yielded empty slice [{first}..{end_excl}]"
        );
        &self.shards[first..end_excl]
    }

    /// Computes the exclusive upper shard index for a multi-shard range whose
    /// `range_end > self.max_keys[first]`.
    #[inline]
    pub(super) fn route_range_end_excl(&self, first: usize, range_end: Key) -> usize {
        let tail = &self.max_keys[(first + 1)..];
        let offset = if tail.len() <= FLAT_LINEAR_SCAN_MAX_SHARDS {
            let mut idx = 0usize;
            for &mk in tail {
                idx += usize::from(mk < range_end);
            }
            idx
        } else {
            tail.partition_point(|&mk| mk < range_end)
        };
        first + offset + 2
    }

    /// Finds the positional index of `shard_id` in this routing snapshot.
    #[must_use]
    pub fn find_shard_index(&self, shard_id: ShardId) -> Option<usize> {
        self.shards.iter().position(|s| s.shard_id == shard_id)
    }

    /// Verifies complete contiguous `[0, u64::MAX]` coverage, zero gaps, zero
    /// overlaps, active (non-retired) status, and unique [`ShardId`]s.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on the first violation detected.
    pub fn verify_invariants(&self) -> AerisResult<()> {
        if self.shards.is_empty() {
            return Err(AerisError::InternalInvariant(
                "RoutingSnapshot contains zero shards".to_string(),
            ));
        }

        let first = &self.shards[0];
        if first.bounds.min_key != 0 {
            return Err(AerisError::InternalInvariant(format!(
                "first shard {} starts at {} instead of 0",
                first.shard_id, first.bounds.min_key
            )));
        }

        let last = self.shards.last().expect("verified non-empty shards above");
        if last.bounds.max_key != u64::MAX {
            return Err(AerisError::InternalInvariant(format!(
                "last shard {} ends at {} instead of u64::MAX",
                last.shard_id, last.bounds.max_key
            )));
        }

        let mut seen_ids = HashSet::with_capacity(self.shards.len());
        for (idx, shard) in self.shards.iter().enumerate() {
            let prev = if idx > 0 {
                Some(&self.shards[idx - 1])
            } else {
                None
            };
            verify_shard_entry(shard, prev, &mut seen_ids)?;
        }
        Ok(())
    }
}

fn verify_shard_entry(
    shard: &ShardInstance,
    prev: Option<&Arc<ShardInstance>>,
    seen_ids: &mut HashSet<ShardId>,
) -> AerisResult<()> {
    if !seen_ids.insert(shard.shard_id) {
        return Err(AerisError::InternalInvariant(format!(
            "duplicate shard_id {} in RoutingSnapshot",
            shard.shard_id
        )));
    }
    if shard.bounds.min_key > shard.bounds.max_key {
        return Err(AerisError::InternalInvariant(format!(
            "shard {} has inverted bounds [{}, {}]",
            shard.shard_id, shard.bounds.min_key, shard.bounds.max_key
        )));
    }
    if shard.is_retired() {
        return Err(AerisError::InternalInvariant(format!(
            "active RoutingSnapshot references retired shard {}",
            shard.shard_id
        )));
    }
    if let Some(prev) = prev {
        let expected_min = prev.bounds.max_key.checked_add(1).ok_or_else(|| {
            AerisError::InternalInvariant(format!(
                "shard {} ends at u64::MAX before final shard",
                prev.shard_id
            ))
        })?;
        if shard.bounds.min_key != expected_min {
            return Err(AerisError::InternalInvariant(format!(
                "routing discontinuity between shard {} [.., {}] and shard {} [{}, ..]",
                prev.shard_id, prev.bounds.max_key, shard.shard_id, shard.bounds.min_key
            )));
        }
    }
    Ok(())
}

/// Computes `shard_count` contiguous uniform [`ShardBounds`] partitioning `[0, u64::MAX]`
/// using overflow-free `u128` arithmetic.
///
/// # Errors
/// Returns [`AerisError::InvalidConfig`] if `shard_count == 0`.
pub fn compute_uniform_bounds(shard_count: usize) -> AerisResult<Vec<ShardBounds>> {
    if shard_count == 0 {
        return Err(AerisError::InvalidConfig(
            "shard_count must be at least 1".to_string(),
        ));
    }
    let universe: u128 = (u64::MAX as u128) + 1;
    let n = shard_count as u128;
    let mut out = Vec::with_capacity(shard_count);
    for i in 0..shard_count {
        let start = ((i as u128) * universe / n) as u64;
        let end_excl = (((i + 1) as u128) * universe) / n;
        let max_key = (end_excl - 1) as u64;
        out.push(ShardBounds::new(start, max_key)?);
    }
    Ok(out)
}

/// Builds contiguous [`ShardBounds`] covering `[0, u64::MAX]` from strictly
/// ascending inclusive split points `split_keys` (each `< u64::MAX`).
///
/// # Errors
/// Returns [`AerisError::InvalidConfig`] if `split_keys` is not strictly ascending
/// or contains `u64::MAX`.
pub fn compute_bounds_from_split_keys(split_keys: &[Key]) -> AerisResult<Vec<ShardBounds>> {
    let mut out = Vec::with_capacity(split_keys.len() + 1);
    let mut current_min = 0u64;
    for &split in split_keys {
        if split == u64::MAX {
            return Err(AerisError::InvalidConfig(
                "split boundary cannot equal u64::MAX".to_string(),
            ));
        }
        if split < current_min {
            return Err(AerisError::InvalidConfig(format!(
                "split boundaries must be strictly ascending: {split} < {current_min}"
            )));
        }
        out.push(ShardBounds::new(current_min, split)?);
        current_min = split + 1;
    }
    out.push(ShardBounds::new(current_min, u64::MAX)?);
    Ok(out)
}

/// Computes the data-driven median split boundary `split_key` for a shard spanning
/// `bounds` (with `bounds.min_key < bounds.max_key`) and holding strictly sorted
/// deduplicated `sorted_keys`.
///
/// Guarantees `bounds.min_key <= split_key < bounds.max_key` so both
/// `[bounds.min_key, split_key]` and `[split_key + 1, bounds.max_key]` are valid
/// non-empty key intervals without `u64` overflow.
#[inline]
#[must_use]
pub fn compute_median_split_key(bounds: ShardBounds, sorted_keys: &[Key]) -> Key {
    debug_assert!(
        bounds.min_key < bounds.max_key,
        "compute_median_split_key requires min_key < max_key"
    );
    if sorted_keys.len() >= 2 {
        let mid_idx = (sorted_keys.len() - 1) / 2;
        let candidate = sorted_keys[mid_idx];
        // Because `sorted_keys` is strictly increasing and lies inside `bounds`,
        // `candidate < sorted_keys[mid_idx + 1] <= bounds.max_key`.
        if candidate >= bounds.min_key && candidate < bounds.max_key {
            return candidate;
        }
    }
    bounds.min_key + (bounds.max_key - bounds.min_key) / 2
}
