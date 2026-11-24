//! Safe Crossbeam epoch-based RCU cell ([`EpochCell`]) and cache-line-padded
//! reader-striped synchronization primitives.
//!
//! This module is the **only** module in `aeris-scale` permitted to contain
//! `unsafe` code. Every `unsafe` block isolates raw pointer dereference,
//! deferred reclamation from [`crossbeam_epoch::Atomic`], or `UnsafeCell`
//! interior access behind a safe, leak-free, race-free API consumed by the
//! rest of the Scale Plane.
//!
//! # Memory Ordering Protocol
//! - [`EpochCell::load`] reads the pointer with [`Ordering::Acquire`] under a
//!   pinned [`crossbeam_epoch::Guard`], synchronizing with the releasing swap.
//! - [`EpochCell::swap`] replaces the pointer with [`Ordering::AcqRel`],
//!   increments the monotonic [`Epoch`] counter with [`Ordering::AcqRel`],
//!   schedules deferred reclamation of the previous heap allocation via
//!   [`Guard::defer_destroy`], and calls [`Guard::flush`] so garbage is pushed
//!   promptly to the global collector.
//! - [`Drop`] executes under `&mut self` (exclusive ownership), swapping in
//!   [`Shared::null`] and immediately reclaiming the final [`Owned`] allocation.

use aeris_types::Epoch;
use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cell::{Cell, UnsafeCell};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};

/// Initial monotonic epoch assigned by [`EpochCell::new`].
pub const INITIAL_EPOCH: Epoch = 1;

/// Power-of-two number of cache-line-padded reader lock stripes inside [`StripedRwLock`].
///
/// Sized to `4` so 4 concurrent readers on a single filter (e.g., A/B 4.2) map
/// bijectively onto 4 distinct 128-byte-padded L1 cache lines, 8 threads across
/// 8 shards spread across `32` distinct lock lines (`0.25` threads/stripe), and
/// writers lock at most `4` stripes (halving cross-core invalidations and convoy
/// hold time at 8T compared to 8 stripes).
pub(crate) const READER_LOCK_STRIPES: usize = 4;

/// Power-of-two number of cache-line-padded per-shard query counter stripes.
///
/// Counter stripes are never locked by writers, so `8` stripes give zero
/// cross-core contention up to 8 threads with zero write-path cost.
pub(crate) const COUNTER_STRIPES: usize = 8;

const _: () =
    assert!(READER_LOCK_STRIPES.is_power_of_two() && READER_LOCK_STRIPES <= u8::BITS as usize);

/// Monotonic thread-slot allocator for zero-contention reader striping.
static NEXT_THREAD_STRIPE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Const-initialized native TLS slot avoiding lazy-init destructor overhead.
    static THREAD_STRIPE_SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Returns the calling thread's stable, monotonically assigned stripe slot.
///
/// # Complexity
/// `O(1)` single native TLS load after first call on the thread.
#[inline]
pub(crate) fn current_thread_stripe_slot() -> usize {
    THREAD_STRIPE_SLOT.with(|cell| {
        let slot = cell.get();
        if slot != usize::MAX {
            slot
        } else {
            init_thread_stripe_slot(cell)
        }
    })
}

/// Cold one-time thread stripe slot registration.
#[cold]
#[inline(never)]
fn init_thread_stripe_slot(cell: &Cell<usize>) -> usize {
    let slot = NEXT_THREAD_STRIPE.fetch_add(1, Ordering::Relaxed);
    cell.set(slot);
    slot
}

/// Opportunistically advances Crossbeam's epoch collector to reclaim retired
/// snapshots on the topology-mutating thread when no older guards are pinned,
/// preventing deferred shard/RocksDB drops from ambushing hot-path writes.
///
/// Completely non-blocking: if any concurrent reader still holds an older
/// [`Guard`], Crossbeam leaves deferred garbage queued safely.
#[inline]
pub(crate) fn try_reclaim_retired_epochs() {
    for _ in 0..4 {
        let guard = crossbeam_epoch::pin();
        guard.flush();
    }
}

/// Lock-free, Crossbeam epoch-protected RCU container for atomically swapping
/// immutable snapshots (`F_old -> F_new`, `RoutingSnapshot_old -> RoutingSnapshot_new`)
/// without stalling concurrent readers.
///
/// Both `ptr` and `epoch` are wrapped in [`CachePadded`] so hot-path reader
/// loads never share a cache line with adjacent writer locks or counters.
///
/// # Invariants
/// 1. `self.ptr` always points to a valid, heap-allocated `T` created via
///    [`Owned::new`] for the entire lifetime of `self` until [`Drop`] runs.
/// 2. Every [`EpochCell::swap`] advances `self.epoch` strictly monotonically
///    and retires the previous pointer via [`Guard::defer_destroy`], ensuring
///    no in-flight reader holding a pinned [`Guard`] can experience
///    use-after-free.
/// 3. [`Drop`] reclaims the final owned `T` exactly once, preventing leaks.
pub struct EpochCell<T> {
    ptr: CachePadded<Atomic<T>>,
    epoch: CachePadded<AtomicU64>,
}

impl<T> EpochCell<T> {
    /// Constructs a new [`EpochCell`] storing `val` at [`INITIAL_EPOCH`] (`1`).
    ///
    /// # Complexity
    /// `O(1)` time and one heap allocation for `Owned::new(val)`.
    #[must_use]
    pub fn new(val: T) -> Self {
        Self::with_epoch(val, INITIAL_EPOCH)
    }

    /// Constructs a new [`EpochCell`] storing `val` at an explicit `initial_epoch`.
    ///
    /// # Complexity
    /// `O(1)` time and one heap allocation for `Owned::new(val)`.
    #[must_use]
    pub fn with_epoch(val: T, initial_epoch: Epoch) -> Self {
        Self {
            ptr: CachePadded::new(Atomic::new(val)),
            epoch: CachePadded::new(AtomicU64::new(initial_epoch)),
        }
    }

    /// Loads a shared reference to the currently published `T` tied to the
    /// lifetime of the caller's pinned [`Guard`].
    ///
    /// # Preconditions
    /// The caller must hold a pinned `guard` obtained from [`crossbeam_epoch::pin`].
    ///
    /// # Complexity
    /// `O(1)` lock-free wait-free atomic load, zero heap allocations.
    #[inline]
    #[must_use]
    pub fn load<'g>(&self, guard: &'g Guard) -> &'g T {
        let shared = self.ptr.load(Ordering::Acquire, guard);
        debug_assert!(
            !shared.is_null(),
            "EpochCell pointer invariant violated: encountered null pointer during live load"
        );
        // SAFETY:
        // 1. `self.ptr` is always initialized in `new`/`with_epoch` via `Owned::new(val)`
        //    and only ever overwritten via `swap` with another non-null `Owned::new(new_val)`
        //    using `Ordering::AcqRel`. `Shared::null()` is only written inside `Drop`,
        //    which requires `&mut self` and therefore cannot race with `&self::load`.
        // 2. Loading with `Ordering::Acquire` establishes a happens-before relationship
        //    with the initializing/swapping `Ordering::AcqRel` store, guaranteeing the
        //    pointee `T` is fully initialized and properly aligned.
        // 3. `guard` pins the current thread into the active Crossbeam epoch; any
        //    concurrent `swap` retires the previous allocation via `guard.defer_destroy`,
        //    which guarantees `T` will not be dropped or deallocated until `guard` (and
        //    every other guard that could have observed `shared`) is dropped.
        unsafe { shared.deref() }
    }

    /// Convenience helper that pins a Crossbeam epoch guard, invokes `f` on the
    /// currently published `&T`, and unpins immediately upon return.
    ///
    /// # Complexity
    /// `O(1)` pin/load/unpin plus the cost of `f`.
    #[inline]
    pub fn with_ref<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let guard = crossbeam_epoch::pin();
        let val = self.load(&guard);
        f(val)
    }

    /// Atomically publishes `new_val`, increments the monotonic epoch counter,
    /// schedules deferred reclamation of the previous value, and returns the
    /// newly published [`Epoch`].
    ///
    /// Readers that already loaded the old pointer continue uninterrupted until
    /// their [`Guard`]s drop; new readers immediately observe `new_val`.
    ///
    /// # Complexity
    /// `O(1)` atomic swap plus deferred destruction enqueue.
    pub fn swap(&self, new_val: T) -> Epoch {
        let guard = crossbeam_epoch::pin();
        let owned = Owned::new(new_val);
        let old = self.ptr.swap(owned, Ordering::AcqRel, &guard);
        let new_epoch = self.epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1);

        if !old.is_null() {
            // SAFETY:
            // 1. `old` originated from `Owned::new(...)` during construction or a prior
            //    `swap` and is non-null, properly aligned, and valid for `T`.
            // 2. Because `self.ptr.swap(..., Ordering::AcqRel, &guard)` atomically
            //    removed `old` from `self.ptr`, this thread holds unique responsibility
            //    for retiring `old` exactly once; no future `load` can obtain `old`.
            // 3. `guard.defer_destroy(old)` defers running `T::drop` and freeing the
            //    box until Crossbeam proves no thread holds a `Guard` from an epoch
            //    that could still be dereferencing `old`.
            unsafe {
                guard.defer_destroy(old);
            }
            guard.flush();
        }

        new_epoch
    }

    /// Returns an exclusive mutable reference to the currently published `T`
    /// without pinning a Crossbeam epoch guard or issuing atomic fences.
    ///
    /// Because Rust's borrow checker enforces unique `&mut self` ownership, no
    /// concurrent `load`, `swap`, or `drop` can race with this access.
    ///
    /// # Complexity
    /// `O(1)` single relaxed pointer load, zero epoch pinning, zero allocations.
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY:
        // 1. `&mut self` statically guarantees exclusive access to `self.ptr`; no
        //    other thread can concurrently invoke `load`, `swap`, or `drop`.
        // 2. `crossbeam_epoch::unprotected()` is sound under exclusive ownership
        //    because no concurrent reclamation or pointer swap can occur.
        // 3. `self.ptr` always holds a non-null, properly aligned allocation created
        //    via `Owned::new` (and is only overwritten with `Shared::null()` inside
        //    `Drop::drop`), so `deref_mut()` yields a valid `&mut T` tied to `&mut self`.
        unsafe {
            let guard = crossbeam_epoch::unprotected();
            let mut shared = self.ptr.load(Ordering::Relaxed, guard);
            debug_assert!(
                !shared.is_null(),
                "EpochCell pointer invariant violated: encountered null pointer during get_mut"
            );
            shared.deref_mut()
        }
    }

    /// Returns the most recently published monotonic [`Epoch`].
    ///
    /// # Complexity
    /// `O(1)` atomic `Acquire` load.
    #[inline]
    #[must_use]
    pub fn epoch(&self) -> Epoch {
        self.epoch.load(Ordering::Acquire)
    }
}

impl<T> Drop for EpochCell<T> {
    fn drop(&mut self) {
        // SAFETY:
        // 1. `Drop::drop` holds `&mut self`, statically guaranteeing no concurrent
        //    `load` or `swap` can access `self.ptr`.
        // 2. `crossbeam_epoch::unprotected()` is sound under exclusive ownership
        //    because no other thread can concurrently load or retire `self.ptr`.
        // 3. The non-null pointer removed from `self.ptr` was allocated via
        //    `Owned::new` and has never been retired via `defer_destroy`, so
        //    reclaiming it via `into_owned()` runs `T::drop` and frees it once.
        unsafe {
            let guard = crossbeam_epoch::unprotected();
            let shared = self.ptr.swap(Shared::null(), Ordering::AcqRel, guard);
            if !shared.is_null() {
                drop(shared.into_owned());
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for EpochCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = crossbeam_epoch::pin();
        let current = self.load(&guard);
        f.debug_struct("EpochCell")
            .field("epoch", &self.epoch())
            .field("current", current)
            .finish()
    }
}

/// Cache-line-padded reader-striped reader-writer lock (`brlock`) eliminating
/// cross-core coherence bouncing among concurrent readers while keeping writer
/// overhead proportional only to the stripes actually used by readers.
///
/// # Concurrency & Lock-Ordering Protocol
/// - **One-time Stripe Registration:** A reader thread on stripe `idx` checks
///   `used_stripes_mask` (a read-mostly cache-line-padded `AtomicU8`). On the
///   very first read on stripe `idx` for this lock instance (at most
///   `READER_LOCK_STRIPES` times over the lock's lifetime), it acquires
///   `writer_lock`, sets bit `1 << idx` with `Ordering::Release`, and drops
///   `writer_lock` BEFORE locking `stripes[idx]`.
/// - **Readers (`read`, `read_with_slot`):** After registration, perform one
///   read-only `Acquire` check on `used_stripes_mask` and acquire a shared lock
///   on `self.stripes[idx]`, staying in the core's L1 cache with zero
///   reader-reader invalidations.
/// - **Writers (`write`):** Acquire `self.writer_lock` first (blocking any new
///   stripe registrations for the duration of the write), read
///   `used_stripes_mask`, and acquire exclusive write locks ONLY on stripes
///   whose bit is set in strict ascending bit order `0..READER_LOCK_STRIPES`.
///   - Write-only / single-threaded workloads (`mask == 0` or popcount `1`) lock
///     `0` or `1` stripes instead of all `8`!
/// - **Deadlock Freedom:** `register_reader_stripe` holds zero stripes when
///   locking `writer_lock`; readers never hold `writer_lock` or multiple
///   stripes; writers always lock `writer_lock` -> set stripes in ascending
///   order.
/// - **Zero Heap Allocations:** Both `read` and `write` store guards inline on
///   the stack.
pub(crate) struct StripedRwLock<T> {
    writer_lock: CachePadded<Mutex<()>>,
    used_stripes_mask: CachePadded<AtomicU8>,
    stripes: [CachePadded<RwLock<()>>; READER_LOCK_STRIPES],
    data: CachePadded<UnsafeCell<T>>,
}

// SAFETY:
// Transferring ownership of `StripedRwLock<T>` across threads transfers `T`,
// which is sound whenever `T: Send`.
unsafe impl<T: Send> Send for StripedRwLock<T> {}

// SAFETY:
// Sharing `&StripedRwLock<T>` across threads allows concurrent `&T` reads
// (requiring `T: Sync`) and mutually exclusive `&mut T` writes across threads
// (requiring `T: Send`). Mutual exclusion is enforced because no reader may
// lock `stripes[i]` until bit `i` is set in `used_stripes_mask` under
// `writer_lock`, and every writer holds `writer_lock` plus exclusive write
// guards on every stripe whose bit is set in `used_stripes_mask`.
unsafe impl<T: Send + Sync> Sync for StripedRwLock<T> {}

impl<T> StripedRwLock<T> {
    /// Creates a new [`StripedRwLock`] protecting `val`.
    #[must_use]
    pub(crate) fn new(val: T) -> Self {
        Self {
            writer_lock: CachePadded::new(Mutex::new(())),
            used_stripes_mask: CachePadded::new(AtomicU8::new(0)),
            stripes: std::array::from_fn(|_| CachePadded::new(RwLock::new(()))),
            data: CachePadded::new(UnsafeCell::new(val)),
        }
    }

    /// Returns an exclusive mutable reference to the protected `T` with zero locking.
    ///
    /// Rust's `&mut self` exclusivity statically guarantees no concurrent readers
    /// or writers exist.
    #[inline]
    #[must_use]
    pub(crate) fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Acquires a shared read guard using the calling thread's default stripe slot.
    #[inline]
    #[must_use]
    pub(crate) fn read(&self) -> StripedReadGuard<'_, T> {
        self.read_with_slot(current_thread_stripe_slot())
    }

    /// Acquires a shared read guard using a pre-fetched thread stripe `slot`.
    #[inline]
    #[must_use]
    pub(crate) fn read_with_slot(&self, slot: usize) -> StripedReadGuard<'_, T> {
        let idx = slot & (READER_LOCK_STRIPES - 1);
        let bit = 1u8 << idx;
        if (self.used_stripes_mask.load(Ordering::Acquire) & bit) == 0 {
            self.register_reader_stripe(bit);
        }
        let _stripe_guard = self.stripes[idx].read();
        StripedReadGuard {
            lock: self,
            _stripe_guard,
        }
    }

    /// Cold one-time registration marking `bit` active under `writer_lock`.
    #[cold]
    #[inline(never)]
    fn register_reader_stripe(&self, bit: u8) {
        let _w = self.writer_lock.lock();
        self.used_stripes_mask.fetch_or(bit, Ordering::Release);
    }

    /// Acquires an exclusive write guard by locking `writer_lock` followed by
    /// every registered reader stripe in strict ascending bit order.
    #[inline]
    #[must_use]
    pub(crate) fn write(&self) -> StripedWriteGuard<'_, T> {
        let _writer_guard = self.writer_lock.lock();
        let mask = self.used_stripes_mask.load(Ordering::Acquire);
        let mut stripe_guards: [Option<RwLockWriteGuard<'_, ()>>; READER_LOCK_STRIPES] =
            std::array::from_fn(|_| None);
        let mut rem = mask;
        while rem != 0 {
            let idx = rem.trailing_zeros() as usize;
            stripe_guards[idx] = Some(self.stripes[idx].write());
            rem &= rem - 1;
        }
        StripedWriteGuard {
            lock: self,
            _stripe_guards: stripe_guards,
            _writer_guard,
        }
    }
}

/// Shared read guard returned by [`StripedRwLock::read`] / [`StripedRwLock::read_with_slot`].
pub(crate) struct StripedReadGuard<'a, T> {
    lock: &'a StripedRwLock<T>,
    _stripe_guard: RwLockReadGuard<'a, ()>,
}

impl<T> Deref for StripedReadGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY:
        // 1. Before acquiring `self._stripe_guard` on stripe `i`, bit `i` was
        //    published in `used_stripes_mask` under `writer_lock`.
        // 2. Any writer (`StripedRwLock::write`) holds `writer_lock` and must
        //    acquire an exclusive write lock on every stripe with bit `i == 1`,
        //    so no `&mut T` can coexist with `self`.
        // 3. `parking_lot::RwLock::read` (or the initial `writer_lock` handoff)
        //    synchronizes-with all preceding writes, guaranteeing full visibility.
        unsafe { &*self.lock.data.get() }
    }
}

/// Exclusive write guard returned by [`StripedRwLock::write`].
///
/// Field declaration order guarantees `_stripe_guards` release BEFORE
/// `_writer_guard` (strict LIFO reverse of acquisition order).
pub(crate) struct StripedWriteGuard<'a, T> {
    lock: &'a StripedRwLock<T>,
    _stripe_guards: [Option<RwLockWriteGuard<'a, ()>>; READER_LOCK_STRIPES],
    _writer_guard: MutexGuard<'a, ()>,
}

impl<T> Deref for StripedWriteGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY:
        // `self` holds `_writer_guard` (blocking any new stripe registration)
        // and exclusive write locks on every registered stripe in
        // `used_stripes_mask`, proving zero readers and zero other writers
        // can access `self.lock.data`.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for StripedWriteGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY:
        // 1. `self` holds `_writer_guard` (preventing unregistered readers from
        //    entering any stripe) and exclusive write guards on every already-
        //    registered stripe, proving zero concurrent `StripedReadGuard`s or
        //    `StripedWriteGuard`s exist.
        // 2. `&mut self` statically prevents aliasing references through this
        //    guard, granting exclusive mutable access to `self.lock.data`.
        unsafe { &mut *self.lock.data.get() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Barrier};
    use std::thread;

    struct TrackedPayload {
        version: u64,
        words: Vec<u64>,
        checksum: u64,
        dropped_counter: Arc<AtomicUsize>,
    }

    impl TrackedPayload {
        fn new(version: u64, dropped_counter: Arc<AtomicUsize>) -> Self {
            let words: Vec<u64> = (0..16)
                .map(|i| version.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i))
                .collect();
            let checksum = words.iter().fold(0u64, |acc, &w| acc ^ w);
            Self {
                version,
                words,
                checksum,
                dropped_counter,
            }
        }

        fn verify(&self) -> bool {
            let computed = self.words.iter().fold(0u64, |acc, &w| acc ^ w);
            computed == self.checksum && self.words.len() == 16
        }
    }

    impl Drop for TrackedPayload {
        fn drop(&mut self) {
            self.dropped_counter.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn epoch_monotonicity_and_drop_reclamation() {
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = EpochCell::new(TrackedPayload::new(1, Arc::clone(&drops)));
        assert_eq!(cell.epoch(), INITIAL_EPOCH);

        cell.with_ref(|p| {
            assert_eq!(p.version, 1);
            assert!(p.verify());
        });

        let total_swaps = 128u64;
        for step in 2..=(total_swaps + 1) {
            let new_ep = cell.swap(TrackedPayload::new(step, Arc::clone(&drops)));
            assert_eq!(new_ep, step);
            assert_eq!(cell.epoch(), step);
        }

        cell.with_ref(|p| {
            assert_eq!(p.version, total_swaps + 1);
            assert!(p.verify());
        });

        drop(cell);

        // Advance Crossbeam's global epoch cleanly to flush deferred thread-local bags.
        for _ in 0..64 {
            let guard = crossbeam_epoch::pin();
            guard.flush();
        }

        let dropped = drops.load(Ordering::Acquire);
        assert!(
            dropped >= 1,
            "final owned payload must be reclaimed immediately on Drop"
        );
        assert!(
            dropped <= (total_swaps as usize) + 1,
            "no payload may ever be double-dropped: dropped={dropped}"
        );
    }

    #[test]
    fn concurrent_readers_during_rapid_swaps_zero_uaf_monotonic_epochs() {
        const READERS: usize = 8;
        const SWAPS_PER_WRITER: u64 = 500;
        const WRITERS: usize = 2;

        let drops = Arc::new(AtomicUsize::new(0));
        let cell = Arc::new(EpochCell::new(TrackedPayload::new(0, Arc::clone(&drops))));
        let barrier = Arc::new(Barrier::new(READERS + WRITERS));
        let done = Arc::new(AtomicBool::new(false));
        let total_reads = Arc::new(AtomicUsize::new(0));

        let mut reader_handles = Vec::with_capacity(READERS);
        for _ in 0..READERS {
            let cell_ref = Arc::clone(&cell);
            let bar = Arc::clone(&barrier);
            let done_ref = Arc::clone(&done);
            let reads_ref = Arc::clone(&total_reads);
            reader_handles.push(thread::spawn(move || {
                bar.wait();
                let mut local_reads = 0usize;
                let mut last_seen_epoch = 0u64;
                while !done_ref.load(Ordering::Acquire) || local_reads < 1000 {
                    let guard = crossbeam_epoch::pin();
                    let ep = cell_ref.epoch();
                    let payload = cell_ref.load(&guard);
                    assert!(
                        payload.verify(),
                        "observed corrupted payload during RCU swap"
                    );
                    assert!(
                        ep >= last_seen_epoch,
                        "epoch regressed: {ep} < {last_seen_epoch}"
                    );
                    last_seen_epoch = ep;
                    local_reads += 1;
                }
                reads_ref.fetch_add(local_reads, Ordering::Relaxed);
            }));
        }

        let mut writer_handles = Vec::with_capacity(WRITERS);
        for w in 0..WRITERS {
            let cell_ref = Arc::clone(&cell);
            let bar = Arc::clone(&barrier);
            let drops_ref = Arc::clone(&drops);
            writer_handles.push(thread::spawn(move || {
                bar.wait();
                let mut prev_epoch = 0u64;
                for i in 1..=SWAPS_PER_WRITER {
                    let version = (w as u64) * 10_000 + i;
                    let ep = cell_ref.swap(TrackedPayload::new(version, Arc::clone(&drops_ref)));
                    assert!(
                        ep > prev_epoch,
                        "swap epoch must strictly increase: {ep} <= {prev_epoch}"
                    );
                    prev_epoch = ep;
                }
            }));
        }

        for h in writer_handles {
            h.join().expect("writer thread panicked");
        }
        done.store(true, Ordering::Release);
        for h in reader_handles {
            h.join().expect("reader thread panicked");
        }

        let expected_final_epoch = INITIAL_EPOCH + (WRITERS as u64) * SWAPS_PER_WRITER;
        assert_eq!(cell.epoch(), expected_final_epoch);
        assert!(total_reads.load(Ordering::Relaxed) >= READERS * 1000);

        drop(cell);
        for _ in 0..64 {
            let guard = crossbeam_epoch::pin();
            guard.flush();
        }
        assert!(
            drops.load(Ordering::Acquire) <= (expected_final_epoch as usize),
            "double-drop detected"
        );
    }

    #[test]
    fn striped_rwlock_concurrent_readers_and_writers_preserve_invariants() {
        const READERS: usize = 12;
        const WRITERS: usize = 3;
        const UPDATES_PER_WRITER: u64 = 400;

        let lock = Arc::new(StripedRwLock::new((0u64, 7u64)));
        let barrier = Arc::new(Barrier::new(READERS + WRITERS));
        let done = Arc::new(AtomicBool::new(false));
        let total_reads = Arc::new(AtomicUsize::new(0));

        let mut reader_handles = Vec::with_capacity(READERS);
        for _ in 0..READERS {
            let lock_ref = Arc::clone(&lock);
            let bar = Arc::clone(&barrier);
            let done_ref = Arc::clone(&done);
            let reads_ref = Arc::clone(&total_reads);
            reader_handles.push(thread::spawn(move || {
                bar.wait();
                let mut local_reads = 0usize;
                let mut prev = 0u64;
                while !done_ref.load(Ordering::Acquire) || local_reads < 1_000 {
                    let g = lock_ref.read();
                    // Torn-write / mutual-exclusion invariant: pair is always (v, v * 3 + 7).
                    assert_eq!(
                        g.1,
                        g.0.wrapping_mul(3).wrapping_add(7),
                        "observed torn state in StripedRwLock: {:?}",
                        *g
                    );
                    assert!(g.0 >= prev, "observed regressed value: {} < {prev}", g.0);
                    prev = g.0;
                    local_reads += 1;
                }
                reads_ref.fetch_add(local_reads, Ordering::Relaxed);
            }));
        }

        let mut writer_handles = Vec::with_capacity(WRITERS);
        for _ in 0..WRITERS {
            let lock_ref = Arc::clone(&lock);
            let bar = Arc::clone(&barrier);
            writer_handles.push(thread::spawn(move || {
                bar.wait();
                for _ in 0..UPDATES_PER_WRITER {
                    let mut g = lock_ref.write();
                    let next = g.0 + 1;
                    g.0 = next;
                    g.1 = next.wrapping_mul(3).wrapping_add(7);
                }
            }));
        }

        for h in writer_handles {
            h.join().expect("striped_rwlock writer panicked");
        }
        done.store(true, Ordering::Release);
        for h in reader_handles {
            h.join().expect("striped_rwlock reader panicked");
        }

        let final_state = lock.read();
        let expected = (WRITERS as u64) * UPDATES_PER_WRITER;
        assert_eq!(final_state.0, expected);
        assert_eq!(final_state.1, expected.wrapping_mul(3).wrapping_add(7));
        assert!(total_reads.load(Ordering::Relaxed) >= READERS * 1_000);
    }

    #[test]
    fn striped_rwlock_late_reader_registration_races_safely_with_active_writers() {
        // Stress the `used_stripes_mask == 0 -> non-zero` transition: writers start
        // mutating with zero registered readers while fresh reader threads join
        // staggered mid-stream across all 8 stripe indices.
        const ROUNDS: usize = 16;
        for round in 0..ROUNDS {
            let lock = Arc::new(StripedRwLock::new((0u64, 7u64)));
            let stop = Arc::new(AtomicBool::new(false));

            let w_lock = Arc::clone(&lock);
            let w_stop = Arc::clone(&stop);
            let writer = thread::spawn(move || {
                for step in 1..=2_000u64 {
                    let mut g = w_lock.write();
                    g.0 = step;
                    g.1 = step.wrapping_mul(3).wrapping_add(7);
                }
                w_stop.store(true, Ordering::Release);
            });

            let mut readers = Vec::new();
            for slot in 0..READER_LOCK_STRIPES {
                let r_lock = Arc::clone(&lock);
                let r_stop = Arc::clone(&stop);
                let explicit_slot = (slot + round) & (READER_LOCK_STRIPES - 1);
                readers.push(thread::spawn(move || {
                    let mut last = 0u64;
                    let mut checks = 0usize;
                    while !r_stop.load(Ordering::Acquire) || checks < 64 {
                        let g = r_lock.read_with_slot(explicit_slot);
                        assert_eq!(g.1, g.0.wrapping_mul(3).wrapping_add(7));
                        assert!(g.0 >= last);
                        last = g.0;
                        checks += 1;
                    }
                }));
            }

            writer.join().expect("late-reg writer panicked");
            for r in readers {
                r.join().expect("late-reg reader panicked");
            }
            assert_eq!(lock.read().0, 2_000);
        }
    }

    #[test]
    fn epoch_cell_and_striped_rwlock_get_mut_provide_exclusive_direct_access() {
        let mut cell = EpochCell::new(10u64);
        assert_eq!(*cell.get_mut(), 10);
        *cell.get_mut() = 42;
        cell.with_ref(|v| assert_eq!(*v, 42));

        let new_ep = cell.swap(99);
        assert_eq!(new_ep, INITIAL_EPOCH + 1);
        assert_eq!(*cell.get_mut(), 99);
        *cell.get_mut() += 1;
        cell.with_ref(|v| assert_eq!(*v, 100));

        let mut lock = StripedRwLock::new(vec![1u64, 2, 3]);
        lock.get_mut().push(4);
        assert_eq!(&*lock.read(), &[1, 2, 3, 4]);
    }
}
