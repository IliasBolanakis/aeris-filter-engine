//! Fine-grained cache-line-padded chunk and cluster striped locking ([`ChunkLockTable`]).
//!
//! Aligns lock stripes to 64-slot Quotient Filter chunks (`SLOTS_PER_CHUNK = 64`),
//! eliminating false sharing via [`crossbeam_utils::CachePadded`] and enforcing
//! **strict ascending stripe index acquisition** across all multi-chunk cluster
//! operations so overlapping Robin Hood clusters can NEVER deadlock.
//!
//! # Lock-Ordering Rule
//! Within a single [`ChunkLockTable`], multi-stripe guards ([`ClusterReadGuard`],
//! [`ClusterWriteGuard`]) deduplicate and sort target stripe indices in strict
//! ascending order `i_0 < i_1 < ... < i_{k-1}` before acquiring any lock.
//! Callers MUST NOT hold a guard from one [`ChunkLockTable`] while acquiring
//! another [`ChunkLockTable`], and MUST obey the crate-level lock hierarchy
//! documented in `aeris_scale`.

#![forbid(unsafe_code)]

mod guards;
#[cfg(test)]
mod tests;

pub use guards::{ChunkReadGuard, ChunkWriteGuard, ClusterReadGuard, ClusterWriteGuard};

use aeris_types::SLOTS_PER_CHUNK_LOG2;
use crossbeam_utils::CachePadded;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::fmt;

/// Default upper cap on cache-line-padded lock stripes per filter generation.
///
/// Each stripe is padded to a full cache line (`64`–`128` bytes), so `1024`
/// stripes caps metadata overhead at `<= 128 KiB` even for very large shards
/// while smaller shards size down to `num_chunks.next_power_of_two()`.
pub const DEFAULT_MAX_LOCK_STRIPES: usize = 1024;

/// Inline stack stripe capacity avoiding heap `Vec` allocations on the hot
/// single-key (`insert`/`delete`, 1–2 stripes) and small-range (`adapt`, <= 4
/// stripes) locking paths.
const INLINE_STRIPE_CAP: usize = 4;

/// Cache-line-padded striped reader-writer lock array aligned to 64-slot
/// Quotient Filter chunks.
///
/// # Invariants
/// - `self.stripes.len()` is always a power of two `>= 1`.
/// - `self.mask == self.stripes.len() - 1`, enabling branchless `O(1)` modulo
///   mapping (`chunk_idx & self.mask`).
/// - All multi-chunk acquisition helpers sort and deduplicate stripe indices
///   into strictly ascending order before locking, guaranteeing deadlock freedom.
pub struct ChunkLockTable {
    stripes: Box<[CachePadded<RwLock<()>>]>,
    mask: usize,
}

impl fmt::Debug for ChunkLockTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkLockTable")
            .field("num_stripes", &self.stripes.len())
            .finish()
    }
}

impl ChunkLockTable {
    /// Constructs a [`ChunkLockTable`] sized for `num_chunks` Quotient Filter
    /// chunks, capped at [`DEFAULT_MAX_LOCK_STRIPES`].
    ///
    /// # Complexity
    /// `O(num_stripes)` time and space.
    #[must_use]
    pub fn new(num_chunks: usize) -> Self {
        Self::with_max_stripes(num_chunks, DEFAULT_MAX_LOCK_STRIPES)
    }

    /// Constructs a [`ChunkLockTable`] sized for `num_chunks` chunks with an
    /// explicit upper cap `max_stripes` (rounded down to the largest power of
    /// two `<= max_stripes.max(1)`).
    ///
    /// # Complexity
    /// `O(num_stripes)` time and space.
    #[must_use]
    pub fn with_max_stripes(num_chunks: usize, max_stripes: usize) -> Self {
        let max_pow2 = floor_power_of_two(max_stripes.max(1));
        let desired = num_chunks
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(max_pow2);
        let num_stripes = desired.min(max_pow2).max(1);
        let stripes: Vec<CachePadded<RwLock<()>>> = (0..num_stripes)
            .map(|_| CachePadded::new(RwLock::new(())))
            .collect();
        Self {
            stripes: stripes.into_boxed_slice(),
            mask: num_stripes - 1,
        }
    }

    /// Returns the total number of power-of-two lock stripes in this table.
    #[inline]
    #[must_use]
    pub fn num_stripes(&self) -> usize {
        self.stripes.len()
    }

    /// Maps a 64-slot Quotient Filter `chunk_idx` to its owning stripe index.
    ///
    /// # Complexity
    /// `O(1)` bitwise AND.
    #[inline]
    #[must_use]
    pub const fn stripe_for_chunk(&self, chunk_idx: usize) -> usize {
        chunk_idx & self.mask
    }

    /// Maps a canonical Quotient Filter `slot_idx` to its owning stripe index.
    ///
    /// # Complexity
    /// `O(1)` shift and bitwise AND.
    #[inline]
    #[must_use]
    pub const fn stripe_for_slot(&self, slot_idx: usize) -> usize {
        (slot_idx >> SLOTS_PER_CHUNK_LOG2) & self.mask
    }

    /// Acquires a shared read lock on the stripe owning `chunk_idx`.
    ///
    /// # Complexity
    /// `O(1)` uncontended time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn read_chunk(&self, chunk_idx: usize) -> ChunkReadGuard<'_> {
        let stripe_idx = self.stripe_for_chunk(chunk_idx);
        ChunkReadGuard::new(stripe_idx, self.stripes[stripe_idx].read())
    }

    /// Acquires an exclusive write lock on the stripe owning `chunk_idx`.
    ///
    /// # Complexity
    /// `O(1)` uncontended time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn write_chunk(&self, chunk_idx: usize) -> ChunkWriteGuard<'_> {
        let stripe_idx = self.stripe_for_chunk(chunk_idx);
        ChunkWriteGuard::new(stripe_idx, self.stripes[stripe_idx].write())
    }

    /// Acquires exclusive write locks across all unique stripes covering the
    /// inclusive chunk interval `[start_chunk, end_chunk_inclusive]` in
    /// **strict ascending stripe index order**.
    ///
    /// If `start_chunk > end_chunk_inclusive`, the endpoints are normalized so
    /// the closed interval `[min, max]` is locked safely.
    ///
    /// # Deadlock Freedom
    /// Stripes are deduplicated and sorted in strict ascending order
    /// `s_0 < s_1 < ... < s_{k-1}` prior to acquisition, eliminating circular
    /// waits even when a cluster wraps across the stripe ring boundary.
    ///
    /// # Complexity
    /// `O(1)` with **zero heap allocations** for spans `<= 4` chunks (covering
    /// 100% of single-key `insert`/`delete` cluster writes); `O(k log k)` for
    /// wider spans where `k = min(span, num_stripes)`.
    #[inline]
    #[must_use]
    pub fn write_cluster_range(
        &self,
        start_chunk: usize,
        end_chunk_inclusive: usize,
    ) -> ClusterWriteGuard<'_> {
        let (lo, hi) = if start_chunk <= end_chunk_inclusive {
            (start_chunk, end_chunk_inclusive)
        } else {
            (end_chunk_inclusive, start_chunk)
        };

        let diff = hi - lo;
        // Fast path 1 (100% of insert/delete calls): 2-chunk cluster span `[c, c + 1]`.
        if diff == 1 {
            let s0 = self.stripe_for_chunk(lo);
            let s1 = self.stripe_for_chunk(hi);
            return if s0 == s1 {
                self.lock_write_inline_1(s0)
            } else if s0 < s1 {
                self.lock_write_inline_2(s0, s1)
            } else {
                self.lock_write_inline_2(s1, s0)
            };
        }

        // Fast path 2: single-chunk span `[c, c]`.
        if diff == 0 {
            return self.lock_write_inline_1(self.stripe_for_chunk(lo));
        }

        // Fast path 3: small 3- or 4-chunk span kept completely on the stack.
        if diff < INLINE_STRIPE_CAP {
            let mut buf = [0usize; INLINE_STRIPE_CAP];
            let mut raw_len = 0usize;
            for chunk in lo..=hi {
                buf[raw_len] = self.stripe_for_chunk(chunk);
                raw_len += 1;
            }
            let unique_len = sort_dedup_inline(&mut buf, raw_len);
            return self.lock_write_inline_slice(&buf, unique_len);
        }

        self.write_cluster_range_wide(lo, hi)
    }

    /// Acquires shared read locks across all unique stripes covering
    /// `chunk_indices` in **strict ascending stripe index order**.
    ///
    /// # Complexity
    /// Zero heap allocations when `chunk_indices.len() <= 4`; `O(k log k)` otherwise.
    #[must_use]
    pub fn read_chunks(&self, chunk_indices: &[usize]) -> ClusterReadGuard<'_> {
        if chunk_indices.is_empty() {
            return ClusterReadGuard::empty();
        }
        if chunk_indices.len() <= INLINE_STRIPE_CAP {
            let (buf, unique_len) = self.collect_inline_unique_stripes(chunk_indices);
            return self.lock_read_inline_slice(&buf, unique_len);
        }

        let stripe_indices = self.unique_sorted_slice_stripes(chunk_indices);
        self.lock_read_stripes_heap(stripe_indices)
    }

    /// Acquires exclusive write locks across all unique stripes covering
    /// `chunk_indices` in **strict ascending stripe index order**.
    ///
    /// # Complexity
    /// Zero heap allocations when `chunk_indices.len() <= 4`; `O(k log k)` otherwise.
    #[must_use]
    pub fn write_chunks(&self, chunk_indices: &[usize]) -> ClusterWriteGuard<'_> {
        if chunk_indices.is_empty() {
            return ClusterWriteGuard::empty();
        }
        if chunk_indices.len() <= INLINE_STRIPE_CAP {
            let (buf, unique_len) = self.collect_inline_unique_stripes(chunk_indices);
            return self.lock_write_inline_slice(&buf, unique_len);
        }

        let stripe_indices = self.unique_sorted_slice_stripes(chunk_indices);
        self.lock_write_stripes_heap(stripe_indices)
    }

    /// Acquires exclusive write locks on every stripe `0..num_stripes` in
    /// strict ascending order.
    ///
    /// Used by RCU swap / expansion barriers to quiesce writers on a retiring
    /// generation while leaving lock-free readers completely unblocked.
    ///
    /// # Complexity
    /// `O(num_stripes)` time and space (zero heap allocations when `num_stripes <= 4`).
    #[must_use]
    pub fn write_all(&self) -> ClusterWriteGuard<'_> {
        let num_stripes = self.stripes.len();
        if num_stripes <= INLINE_STRIPE_CAP {
            let mut buf = [0usize; INLINE_STRIPE_CAP];
            for (i, slot) in buf.iter_mut().enumerate().take(num_stripes) {
                *slot = i;
            }
            return self.lock_write_inline_slice(&buf, num_stripes);
        }
        let stripe_indices: Vec<usize> = (0..num_stripes).collect();
        self.lock_write_stripes_heap(stripe_indices)
    }

    /// Cold wide-span fallback for [`Self::write_cluster_range`] when `hi - lo >= INLINE_STRIPE_CAP`.
    #[cold]
    #[inline(never)]
    fn write_cluster_range_wide(&self, lo: usize, hi: usize) -> ClusterWriteGuard<'_> {
        let num_stripes = self.stripes.len();
        if hi.saturating_sub(lo) >= num_stripes - 1 {
            return self.write_all();
        }
        let span = (hi - lo) + 1;
        let mut indices = Vec::with_capacity(span);
        for chunk in lo..=hi {
            indices.push(self.stripe_for_chunk(chunk));
        }
        indices.sort_unstable();
        indices.dedup();
        self.lock_write_stripes_heap(indices)
    }

    /// Maps `chunk_indices` (`len <= INLINE_STRIPE_CAP`) into sorted, deduplicated
    /// stack-resident stripe indices.
    #[inline]
    fn collect_inline_unique_stripes(
        &self,
        chunk_indices: &[usize],
    ) -> ([usize; INLINE_STRIPE_CAP], usize) {
        let mut buf = [0usize; INLINE_STRIPE_CAP];
        for (dst, &c) in buf.iter_mut().zip(chunk_indices.iter()) {
            *dst = self.stripe_for_chunk(c);
        }
        let unique_len = sort_dedup_inline(&mut buf, chunk_indices.len());
        (buf, unique_len)
    }

    /// Maps an arbitrary chunk index slice (`len > INLINE_STRIPE_CAP`) to unique,
    /// strictly ascending stripe indices.
    fn unique_sorted_slice_stripes(&self, chunk_indices: &[usize]) -> Vec<usize> {
        let mut indices: Vec<usize> = chunk_indices
            .iter()
            .map(|&c| self.stripe_for_chunk(c))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    #[inline]
    fn lock_write_inline_1(&self, s0: usize) -> ClusterWriteGuard<'_> {
        let g0 = self.stripes[s0].write();
        ClusterWriteGuard::from_inline(1, [s0, 0, 0, 0], [Some(g0), None, None, None])
    }

    #[inline]
    fn lock_write_inline_2(&self, s0: usize, s1: usize) -> ClusterWriteGuard<'_> {
        debug_assert!(
            s0 < s1,
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let g0 = self.stripes[s0].write();
        let g1 = self.stripes[s1].write();
        ClusterWriteGuard::from_inline(2, [s0, s1, 0, 0], [Some(g0), Some(g1), None, None])
    }

    #[inline]
    fn lock_write_inline_slice(
        &self,
        indices: &[usize; INLINE_STRIPE_CAP],
        unique_len: usize,
    ) -> ClusterWriteGuard<'_> {
        debug_assert!(unique_len <= INLINE_STRIPE_CAP);
        debug_assert!(
            indices[..unique_len].windows(2).all(|w| w[0] < w[1]),
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let mut guards: [Option<RwLockWriteGuard<'_, ()>>; INLINE_STRIPE_CAP] =
            [None, None, None, None];
        for i in 0..unique_len {
            guards[i] = Some(self.stripes[indices[i]].write());
        }
        ClusterWriteGuard::from_inline(unique_len, *indices, guards)
    }

    #[inline]
    fn lock_read_inline_slice(
        &self,
        indices: &[usize; INLINE_STRIPE_CAP],
        unique_len: usize,
    ) -> ClusterReadGuard<'_> {
        debug_assert!(unique_len <= INLINE_STRIPE_CAP);
        debug_assert!(
            indices[..unique_len].windows(2).all(|w| w[0] < w[1]),
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let mut guards: [Option<RwLockReadGuard<'_, ()>>; INLINE_STRIPE_CAP] =
            [None, None, None, None];
        for i in 0..unique_len {
            guards[i] = Some(self.stripes[indices[i]].read());
        }
        ClusterReadGuard::from_inline(unique_len, *indices, guards)
    }

    fn lock_write_stripes_heap(&self, stripe_indices: Vec<usize>) -> ClusterWriteGuard<'_> {
        debug_assert!(
            stripe_indices.windows(2).all(|w| w[0] < w[1]),
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let mut guards = Vec::with_capacity(stripe_indices.len());
        for &idx in &stripe_indices {
            guards.push(self.stripes[idx].write());
        }
        ClusterWriteGuard::from_heap(stripe_indices, guards)
    }

    fn lock_read_stripes_heap(&self, stripe_indices: Vec<usize>) -> ClusterReadGuard<'_> {
        debug_assert!(
            stripe_indices.windows(2).all(|w| w[0] < w[1]),
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let mut guards = Vec::with_capacity(stripe_indices.len());
        for &idx in &stripe_indices {
            guards.push(self.stripes[idx].read());
        }
        ClusterReadGuard::from_heap(stripe_indices, guards)
    }
}

/// Sorts and deduplicates `buf[..len]` (`len <= INLINE_STRIPE_CAP`) in place on
/// the stack and returns the resulting strictly ascending unique length.
#[inline]
fn sort_dedup_inline(buf: &mut [usize; INLINE_STRIPE_CAP], len: usize) -> usize {
    if len <= 1 {
        return len;
    }
    // Insertion sort over at most 4 elements avoids heap allocation on hot paths.
    for i in 1..len {
        let key = buf[i];
        let mut j = i;
        while j > 0 && buf[j - 1] > key {
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[j] = key;
    }
    let mut write = 1usize;
    for read in 1..len {
        if buf[read] != buf[write - 1] {
            buf[write] = buf[read];
            write += 1;
        }
    }
    write
}

/// Returns the largest power of two `<= n` for `n >= 1`.
#[inline]
const fn floor_power_of_two(n: usize) -> usize {
    debug_assert!(n >= 1);
    1usize << (usize::BITS - 1 - n.leading_zeros())
}
