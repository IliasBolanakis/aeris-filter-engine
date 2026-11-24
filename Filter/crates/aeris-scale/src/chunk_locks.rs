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
        let _guard = self.stripes[stripe_idx].read();
        ChunkReadGuard { stripe_idx, _guard }
    }

    /// Acquires an exclusive write lock on the stripe owning `chunk_idx`.
    ///
    /// # Complexity
    /// `O(1)` uncontended time, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn write_chunk(&self, chunk_idx: usize) -> ChunkWriteGuard<'_> {
        let stripe_idx = self.stripe_for_chunk(chunk_idx);
        let _guard = self.stripes[stripe_idx].write();
        ChunkWriteGuard { stripe_idx, _guard }
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
            return ClusterReadGuard {
                storage: ClusterReadStorage::Inline {
                    len: 0,
                    indices: [0; INLINE_STRIPE_CAP],
                    _guards: [None, None, None, None],
                },
            };
        }
        if chunk_indices.len() <= INLINE_STRIPE_CAP {
            let mut buf = [0usize; INLINE_STRIPE_CAP];
            for (dst, &c) in buf.iter_mut().zip(chunk_indices.iter()) {
                *dst = self.stripe_for_chunk(c);
            }
            let unique_len = sort_dedup_inline(&mut buf, chunk_indices.len());
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
            return ClusterWriteGuard {
                storage: ClusterWriteStorage::Inline {
                    len: 0,
                    indices: [0; INLINE_STRIPE_CAP],
                    _guards: [None, None, None, None],
                },
            };
        }
        if chunk_indices.len() <= INLINE_STRIPE_CAP {
            let mut buf = [0usize; INLINE_STRIPE_CAP];
            for (dst, &c) in buf.iter_mut().zip(chunk_indices.iter()) {
                *dst = self.stripe_for_chunk(c);
            }
            let unique_len = sort_dedup_inline(&mut buf, chunk_indices.len());
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

    /// Acquires a single write stripe inline on the stack.
    #[inline]
    fn lock_write_inline_1(&self, s0: usize) -> ClusterWriteGuard<'_> {
        let g0 = self.stripes[s0].write();
        ClusterWriteGuard {
            storage: ClusterWriteStorage::Inline {
                len: 1,
                indices: [s0, 0, 0, 0],
                _guards: [Some(g0), None, None, None],
            },
        }
    }

    /// Acquires two strictly ascending write stripes (`s0 < s1`) inline on the stack.
    #[inline]
    fn lock_write_inline_2(&self, s0: usize, s1: usize) -> ClusterWriteGuard<'_> {
        debug_assert!(
            s0 < s1,
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let g0 = self.stripes[s0].write();
        let g1 = self.stripes[s1].write();
        ClusterWriteGuard {
            storage: ClusterWriteStorage::Inline {
                len: 2,
                indices: [s0, s1, 0, 0],
                _guards: [Some(g0), Some(g1), None, None],
            },
        }
    }

    /// Acquires `unique_len <= INLINE_STRIPE_CAP` strictly ascending write stripes inline.
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
        ClusterWriteGuard {
            storage: ClusterWriteStorage::Inline {
                len: unique_len,
                indices: *indices,
                _guards: guards,
            },
        }
    }

    /// Acquires `unique_len <= INLINE_STRIPE_CAP` strictly ascending read stripes inline.
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
        ClusterReadGuard {
            storage: ClusterReadStorage::Inline {
                len: unique_len,
                indices: *indices,
                _guards: guards,
            },
        }
    }

    /// Acquires heap-backed write guards for pre-sorted, deduplicated `stripe_indices`.
    fn lock_write_stripes_heap(&self, stripe_indices: Vec<usize>) -> ClusterWriteGuard<'_> {
        debug_assert!(
            stripe_indices.windows(2).all(|w| w[0] < w[1]),
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let mut guards = Vec::with_capacity(stripe_indices.len());
        for &idx in &stripe_indices {
            guards.push(self.stripes[idx].write());
        }
        ClusterWriteGuard {
            storage: ClusterWriteStorage::Heap {
                stripe_indices,
                _guards: guards,
            },
        }
    }

    /// Acquires heap-backed read guards for pre-sorted, deduplicated `stripe_indices`.
    fn lock_read_stripes_heap(&self, stripe_indices: Vec<usize>) -> ClusterReadGuard<'_> {
        debug_assert!(
            stripe_indices.windows(2).all(|w| w[0] < w[1]),
            "stripe indices must be strictly ascending to guarantee deadlock freedom"
        );
        let mut guards = Vec::with_capacity(stripe_indices.len());
        for &idx in &stripe_indices {
            guards.push(self.stripes[idx].read());
        }
        ClusterReadGuard {
            storage: ClusterReadStorage::Heap {
                stripe_indices,
                _guards: guards,
            },
        }
    }
}

/// Sorts and deduplicates `buf[..len]` (`len <= INLINE_STRIPE_CAP`) in place on
/// the stack and returns the resulting strictly ascending unique length.
#[inline]
fn sort_dedup_inline(buf: &mut [usize; INLINE_STRIPE_CAP], len: usize) -> usize {
    if len <= 1 {
        return len;
    }
    // Insertion sort over at most 4 elements (zero heap allocation).
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

/// RAII guard holding a shared read lock on a single [`ChunkLockTable`] stripe.
pub struct ChunkReadGuard<'a> {
    stripe_idx: usize,
    _guard: RwLockReadGuard<'a, ()>,
}

impl ChunkReadGuard<'_> {
    /// Returns the locked stripe index.
    #[inline]
    #[must_use]
    pub const fn stripe_index(&self) -> usize {
        self.stripe_idx
    }
}

/// RAII guard holding an exclusive write lock on a single [`ChunkLockTable`] stripe.
pub struct ChunkWriteGuard<'a> {
    stripe_idx: usize,
    _guard: RwLockWriteGuard<'a, ()>,
}

impl ChunkWriteGuard<'_> {
    /// Returns the locked stripe index.
    #[inline]
    #[must_use]
    pub const fn stripe_index(&self) -> usize {
        self.stripe_idx
    }
}

/// Internal backing storage for [`ClusterReadGuard`], keeping `<= 4` stripes inline.
enum ClusterReadStorage<'a> {
    Inline {
        len: usize,
        indices: [usize; INLINE_STRIPE_CAP],
        _guards: [Option<RwLockReadGuard<'a, ()>>; INLINE_STRIPE_CAP],
    },
    Heap {
        stripe_indices: Vec<usize>,
        _guards: Vec<RwLockReadGuard<'a, ()>>,
    },
}

/// RAII guard holding shared read locks on one or more [`ChunkLockTable`] stripes
/// acquired in strict ascending index order.
pub struct ClusterReadGuard<'a> {
    storage: ClusterReadStorage<'a>,
}

impl ClusterReadGuard<'_> {
    /// Returns the strictly ascending stripe indices held by this guard.
    #[inline]
    #[must_use]
    pub fn stripe_indices(&self) -> &[usize] {
        match &self.storage {
            ClusterReadStorage::Inline { len, indices, .. } => &indices[..*len],
            ClusterReadStorage::Heap { stripe_indices, .. } => stripe_indices.as_slice(),
        }
    }

    /// Returns the number of stripes held by this guard.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            ClusterReadStorage::Inline { len, .. } => *len,
            ClusterReadStorage::Heap { stripe_indices, .. } => stripe_indices.len(),
        }
    }

    /// Returns `true` if zero stripes are held by this guard.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Internal backing storage for [`ClusterWriteGuard`], keeping `<= 4` stripes inline.
enum ClusterWriteStorage<'a> {
    Inline {
        len: usize,
        indices: [usize; INLINE_STRIPE_CAP],
        _guards: [Option<RwLockWriteGuard<'a, ()>>; INLINE_STRIPE_CAP],
    },
    Heap {
        stripe_indices: Vec<usize>,
        _guards: Vec<RwLockWriteGuard<'a, ()>>,
    },
}

/// RAII guard holding exclusive write locks on one or more [`ChunkLockTable`]
/// stripes acquired in strict ascending index order.
pub struct ClusterWriteGuard<'a> {
    storage: ClusterWriteStorage<'a>,
}

impl ClusterWriteGuard<'_> {
    /// Returns the strictly ascending stripe indices held by this guard.
    #[inline]
    #[must_use]
    pub fn stripe_indices(&self) -> &[usize] {
        match &self.storage {
            ClusterWriteStorage::Inline { len, indices, .. } => &indices[..*len],
            ClusterWriteStorage::Heap { stripe_indices, .. } => stripe_indices.as_slice(),
        }
    }

    /// Returns the number of stripes held by this guard.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            ClusterWriteStorage::Inline { len, .. } => *len,
            ClusterWriteStorage::Heap { stripe_indices, .. } => stripe_indices.len(),
        }
    }

    /// Returns `true` if zero stripes are held by this guard.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Returns the largest power of two `<= n` for `n >= 1`.
#[inline]
const fn floor_power_of_two(n: usize) -> usize {
    debug_assert!(n >= 1);
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn stripe_sizing_and_power_of_two_invariants() {
        assert_eq!(ChunkLockTable::new(0).num_stripes(), 1);
        assert_eq!(ChunkLockTable::new(1).num_stripes(), 1);
        assert_eq!(ChunkLockTable::new(5).num_stripes(), 8);
        assert_eq!(ChunkLockTable::new(16).num_stripes(), 16);
        assert_eq!(
            ChunkLockTable::new(100_000).num_stripes(),
            DEFAULT_MAX_LOCK_STRIPES
        );
        assert_eq!(ChunkLockTable::with_max_stripes(64, 30).num_stripes(), 16);
        assert_eq!(ChunkLockTable::with_max_stripes(64, 0).num_stripes(), 1);
    }

    #[test]
    fn wraparound_cluster_range_sorts_strictly_ascending() {
        let table = ChunkLockTable::with_max_stripes(16, 16);
        assert_eq!(table.num_stripes(), 16);

        // Chunks 15..=17 wrap across stripe boundary -> stripes {15, 0, 1} -> sorted [0, 1, 15].
        let guard = table.write_cluster_range(15, 17);
        assert_eq!(guard.stripe_indices(), &[0, 1, 15]);
        drop(guard);

        // Reversed range endpoints normalize cleanly.
        let guard_rev = table.write_cluster_range(17, 15);
        assert_eq!(guard_rev.stripe_indices(), &[0, 1, 15]);
        drop(guard_rev);

        // Full-table coverage returns all stripes 0..16 in ascending order.
        let guard_full = table.write_cluster_range(3, 40);
        let expected: Vec<usize> = (0..16).collect();
        assert_eq!(guard_full.stripe_indices(), expected.as_slice());
        drop(guard_full);

        // Arbitrary unsorted duplicate chunk list deduplicates and sorts ascending.
        let rguard = table.read_chunks(&[20, 4, 15, 31, 0, 16]);
        assert_eq!(rguard.stripe_indices(), &[0, 4, 15]);
        drop(rguard);

        // Empty chunk slice succeeds with empty guard.
        let empty = table.read_chunks(&[]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        // Single-chunk & 2-chunk inline fast paths (including boundary wraparound and 1-stripe table).
        let g1 = table.write_cluster_range(5, 5);
        assert_eq!(g1.stripe_indices(), &[5]);
        drop(g1);

        let g2 = table.write_cluster_range(15, 16);
        assert_eq!(g2.stripe_indices(), &[0, 15]);
        drop(g2);

        let g4 = table.write_cluster_range(14, 17);
        assert_eq!(g4.stripe_indices(), &[0, 1, 14, 15]);
        drop(g4);

        let g5 = table.write_cluster_range(14, 18);
        assert_eq!(g5.stripe_indices(), &[0, 1, 2, 14, 15]);
        drop(g5);

        // Inline read_chunks / write_chunks with duplicates (<= 4 inputs).
        let rw_inline = table.write_chunks(&[19, 3, 18, 2]);
        assert_eq!(rw_inline.stripe_indices(), &[2, 3]);
        drop(rw_inline);

        let tiny = ChunkLockTable::with_max_stripes(1, 1);
        let tiny_g = tiny.write_cluster_range(7, 8);
        assert_eq!(tiny_g.stripe_indices(), &[0]);
        drop(tiny_g);
    }

    #[test]
    fn sixteen_thread_overlapping_cluster_writes_never_deadlock() {
        const THREADS: usize = 16;
        const OPS_PER_THREAD: usize = 400;
        const STRIPES: usize = 16;

        let table = Arc::new(ChunkLockTable::with_max_stripes(STRIPES, STRIPES));
        let counters: Arc<Vec<AtomicUsize>> =
            Arc::new((0..STRIPES).map(|_| AtomicUsize::new(0)).collect());
        let barrier = Arc::new(Barrier::new(THREADS));

        let mut handles = Vec::with_capacity(THREADS);
        for tid in 0..THREADS {
            let tbl = Arc::clone(&table);
            let cnts = Arc::clone(&counters);
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(0xA3E1_5000 + tid as u64);
                bar.wait();

                for step in 0..OPS_PER_THREAD {
                    match step % 8 {
                        // Wide overlapping wraparound cluster writes.
                        0..=3 => {
                            let start = rng.gen_range(0..64usize);
                            let span = rng.gen_range(1..=6usize);
                            let g = tbl.write_cluster_range(start, start + span);
                            assert!(g.stripe_indices().windows(2).all(|w| w[0] < w[1]));
                            for &s in g.stripe_indices() {
                                cnts[s].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        // Unsorted multi-chunk read locks.
                        4 | 5 => {
                            let chunks = [
                                rng.gen_range(0..64usize),
                                rng.gen_range(0..64usize),
                                rng.gen_range(0..64usize),
                            ];
                            let g = tbl.read_chunks(&chunks);
                            assert!(g.stripe_indices().windows(2).all(|w| w[0] < w[1]));
                            let _ = g
                                .stripe_indices()
                                .iter()
                                .map(|&s| cnts[s].load(Ordering::Relaxed))
                                .sum::<usize>();
                        }
                        // Single-chunk write lock.
                        6 => {
                            let c = rng.gen_range(0..64usize);
                            let g = tbl.write_chunk(c);
                            cnts[g.stripe_index()].fetch_add(1, Ordering::Relaxed);
                        }
                        // Periodic full-table write_all barrier competing with cluster locks.
                        _ => {
                            let g = tbl.write_all();
                            assert_eq!(g.len(), STRIPES);
                            for &s in g.stripe_indices() {
                                cnts[s].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("worker thread panicked during chunk lock stress");
        }

        let total_updates: usize = counters.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        assert!(total_updates > THREADS * OPS_PER_THREAD);
    }
}
