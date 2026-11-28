//! Unit and concurrency tests for [`super::ChunkLockTable`] and stripe guards.

use super::*;
use aeris_types::SLOTS_PER_CHUNK;
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
fn single_chunk_slot_mapping_and_heap_chunk_paths() {
    let table = ChunkLockTable::with_max_stripes(16, 16);
    assert_eq!(table.stripe_for_slot(0), 0);
    assert_eq!(table.stripe_for_slot(SLOTS_PER_CHUNK * 3 + 7), 3);
    assert_eq!(table.stripe_for_slot(SLOTS_PER_CHUNK * 17), 1);

    let rg = table.read_chunk(19);
    assert_eq!(rg.stripe_index(), 3);
    drop(rg);

    let wg = table.write_chunk(31);
    assert_eq!(wg.stripe_index(), 15);
    drop(wg);

    // Empty write_chunks yields empty guard without locking.
    let empty_w = table.write_chunks(&[]);
    assert!(empty_w.is_empty());
    assert_eq!(empty_w.len(), 0);
    drop(empty_w);

    // Heap-backed read_chunks / write_chunks (> 4 inputs) deduplicate and sort.
    let heap_r = table.read_chunks(&[10, 2, 18, 5, 1, 26]);
    assert_eq!(heap_r.stripe_indices(), &[1, 2, 5, 10]);
    assert_eq!(heap_r.len(), 4);
    assert!(!heap_r.is_empty());
    drop(heap_r);

    let heap_w = table.write_chunks(&[9, 1, 8, 0, 4, 20]);
    assert_eq!(heap_w.stripe_indices(), &[0, 1, 4, 8, 9]);
    drop(heap_w);

    // Inline write_all for small stripe table (<= 4 stripes).
    let small = ChunkLockTable::with_max_stripes(4, 4);
    let all_small = small.write_all();
    assert_eq!(all_small.stripe_indices(), &[0, 1, 2, 3]);
    drop(all_small);

    let dbg = format!("{table:?}");
    assert!(dbg.contains("ChunkLockTable"));
    assert!(dbg.contains("16"));
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
