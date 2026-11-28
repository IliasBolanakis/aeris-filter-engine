//! Unit and concurrency stress tests for [`super::EpochCell`] and [`super::StripedRwLock`].

#![forbid(unsafe_code)]

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    // staggered mid-stream across all stripe indices.
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

#[test]
fn epoch_cell_with_epoch_and_debug_formatting() {
    let cell = EpochCell::with_epoch(42u64, 7);
    assert_eq!(cell.epoch(), 7);
    let dbg = format!("{cell:?}");
    assert!(dbg.contains("EpochCell"));
    assert!(dbg.contains("42"));
}
