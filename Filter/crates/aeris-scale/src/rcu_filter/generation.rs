//! Filter generation snapshots published atomically through [`crate::epoch_cell::EpochCell`].

use crate::chunk_locks::ChunkLockTable;
use crate::epoch_cell::StripedRwLock;
use aeris_core::AerisFilter;
use aeris_types::{Epoch, FilterConfig, RangeFilter, ReverseMapStore};
use crossbeam_utils::CachePadded;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Immutable generation snapshot published atomically through [`crate::EpochCell`].
///
/// Couples a single [`AerisFilter`] instance with its matching [`FilterConfig`],
/// aligned [`ChunkLockTable`], generation-bound [`ReverseMapStore`], monotonic
/// [`Epoch`], and retirement flag so writers never mutate a swapped-out filter.
pub struct FilterGeneration {
    /// Monotonic RCU epoch assigned to this generation.
    pub epoch: Epoch,
    /// Validated geometric and probabilistic configuration active in this generation.
    pub config: FilterConfig,
    /// Fine-grained cache-line-padded chunk lock table aligned to `config.num_chunks()`.
    pub chunk_locks: ChunkLockTable,
    /// Active RAM [`AerisFilter`] protected by a reader-writer lock.
    pub filter: RwLock<AerisFilter>,
    /// Disk-resident [`ReverseMapStore`] aligned with this generation's fingerprints.
    reverse_map: Arc<dyn ReverseMapStore>,
    /// Set to `true` under `chunk_locks.write_all()` immediately when a successor
    /// generation is published via [`crate::EpochCell::swap`].
    retired: AtomicBool,
}

impl FilterGeneration {
    /// Constructs a live [`FilterGeneration`] for `filter` and `reverse_map` at `epoch`.
    ///
    /// # Complexity
    /// `O(num_stripes)` time and space.
    #[must_use]
    pub fn new(
        epoch: Epoch,
        filter: AerisFilter,
        reverse_map: Arc<dyn ReverseMapStore>,
        max_lock_stripes: usize,
    ) -> Self {
        let config = filter.config().clone();
        let chunk_locks = ChunkLockTable::with_max_stripes(config.num_chunks(), max_lock_stripes);
        Self {
            epoch,
            config,
            chunk_locks,
            filter: RwLock::new(filter),
            reverse_map,
            retired: AtomicBool::new(false),
        }
    }

    /// Returns `true` if a newer generation has already replaced this one.
    #[inline]
    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// Returns a cloned handle to the [`ReverseMapStore`] bound to this generation.
    #[inline]
    #[must_use]
    pub fn reverse_map(&self) -> Arc<dyn ReverseMapStore> {
        Arc::clone(&self.reverse_map)
    }
}

/// Internal cache-line-optimized generation published inside [`super::RcuAerisFilter`].
///
/// Uses [`StripedRwLock`] so concurrent readers on distinct threads never bounce
/// a single reader-count cache line, isolates `retired` and `live_keys` onto
/// dedicated cache lines, and enables lock-free `O(1)` live key count queries.
pub(super) struct RcuGeneration {
    pub(super) epoch: Epoch,
    pub(super) config: FilterConfig,
    pub(super) chunk_locks: ChunkLockTable,
    pub(super) filter: StripedRwLock<AerisFilter>,
    pub(super) reverse_map: Arc<dyn ReverseMapStore>,
    retired: CachePadded<AtomicBool>,
    pub(super) live_keys: CachePadded<AtomicUsize>,
}

impl RcuGeneration {
    #[must_use]
    pub(super) fn new(
        epoch: Epoch,
        filter: AerisFilter,
        reverse_map: Arc<dyn ReverseMapStore>,
        max_lock_stripes: usize,
    ) -> Self {
        let config = filter.config().clone();
        let initial_len = filter.len();
        let chunk_locks = ChunkLockTable::with_max_stripes(config.num_chunks(), max_lock_stripes);
        Self {
            epoch,
            config,
            chunk_locks,
            filter: StripedRwLock::new(filter),
            reverse_map,
            retired: CachePadded::new(AtomicBool::new(false)),
            live_keys: CachePadded::new(AtomicUsize::new(initial_len)),
        }
    }

    #[inline]
    #[must_use]
    pub(super) fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    #[inline]
    pub(super) fn mark_retired(&self) {
        self.retired.store(true, Ordering::Release);
    }

    #[inline]
    #[must_use]
    pub(super) fn reverse_map(&self) -> Arc<dyn ReverseMapStore> {
        Arc::clone(&self.reverse_map)
    }
}
