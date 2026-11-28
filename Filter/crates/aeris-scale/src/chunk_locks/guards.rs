//! RAII lock guards for single-chunk and multi-stripe cluster acquisitions on
//! [`super::ChunkLockTable`].

use super::INLINE_STRIPE_CAP;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

/// RAII guard holding a shared read lock on a single [`super::ChunkLockTable`] stripe.
pub struct ChunkReadGuard<'a> {
    stripe_idx: usize,
    _guard: RwLockReadGuard<'a, ()>,
}

impl<'a> ChunkReadGuard<'a> {
    /// Wraps a stripe read guard with its resolved stripe index.
    #[inline]
    pub(super) const fn new(stripe_idx: usize, guard: RwLockReadGuard<'a, ()>) -> Self {
        Self {
            stripe_idx,
            _guard: guard,
        }
    }

    /// Returns the locked stripe index.
    #[inline]
    #[must_use]
    pub const fn stripe_index(&self) -> usize {
        self.stripe_idx
    }
}

/// RAII guard holding an exclusive write lock on a single [`super::ChunkLockTable`] stripe.
pub struct ChunkWriteGuard<'a> {
    stripe_idx: usize,
    _guard: RwLockWriteGuard<'a, ()>,
}

impl<'a> ChunkWriteGuard<'a> {
    /// Wraps a stripe write guard with its resolved stripe index.
    #[inline]
    pub(super) const fn new(stripe_idx: usize, guard: RwLockWriteGuard<'a, ()>) -> Self {
        Self {
            stripe_idx,
            _guard: guard,
        }
    }

    /// Returns the locked stripe index.
    #[inline]
    #[must_use]
    pub const fn stripe_index(&self) -> usize {
        self.stripe_idx
    }
}

/// Shared inline-or-heap guard storage parameterized over lock guard type `G`,
/// keeping `<= INLINE_STRIPE_CAP` stripe acquisitions completely stack-allocated.
pub(super) enum ClusterStorage<G> {
    Inline {
        len: usize,
        indices: [usize; INLINE_STRIPE_CAP],
        _guards: [Option<G>; INLINE_STRIPE_CAP],
    },
    Heap {
        stripe_indices: Vec<usize>,
        _guards: Vec<G>,
    },
}

impl<G> ClusterStorage<G> {
    #[inline]
    const fn empty() -> Self {
        Self::Inline {
            len: 0,
            indices: [0; INLINE_STRIPE_CAP],
            _guards: [None, None, None, None],
        }
    }

    #[inline]
    const fn from_inline(
        len: usize,
        indices: [usize; INLINE_STRIPE_CAP],
        guards: [Option<G>; INLINE_STRIPE_CAP],
    ) -> Self {
        Self::Inline {
            len,
            indices,
            _guards: guards,
        }
    }

    #[inline]
    fn from_heap(stripe_indices: Vec<usize>, guards: Vec<G>) -> Self {
        Self::Heap {
            stripe_indices,
            _guards: guards,
        }
    }

    #[inline]
    fn stripe_indices(&self) -> &[usize] {
        match self {
            Self::Inline { len, indices, .. } => &indices[..*len],
            Self::Heap { stripe_indices, .. } => stripe_indices.as_slice(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Inline { len, .. } => *len,
            Self::Heap { stripe_indices, .. } => stripe_indices.len(),
        }
    }
}

/// RAII guard holding shared read locks on one or more [`super::ChunkLockTable`]
/// stripes acquired in strict ascending index order.
pub struct ClusterReadGuard<'a> {
    storage: ClusterStorage<RwLockReadGuard<'a, ()>>,
}

impl<'a> ClusterReadGuard<'a> {
    #[inline]
    pub(super) const fn empty() -> Self {
        Self {
            storage: ClusterStorage::empty(),
        }
    }

    #[inline]
    pub(super) const fn from_inline(
        len: usize,
        indices: [usize; INLINE_STRIPE_CAP],
        guards: [Option<RwLockReadGuard<'a, ()>>; INLINE_STRIPE_CAP],
    ) -> Self {
        Self {
            storage: ClusterStorage::from_inline(len, indices, guards),
        }
    }

    #[inline]
    pub(super) fn from_heap(
        stripe_indices: Vec<usize>,
        guards: Vec<RwLockReadGuard<'a, ()>>,
    ) -> Self {
        Self {
            storage: ClusterStorage::from_heap(stripe_indices, guards),
        }
    }

    /// Returns the strictly ascending stripe indices held by this guard.
    #[inline]
    #[must_use]
    pub fn stripe_indices(&self) -> &[usize] {
        self.storage.stripe_indices()
    }

    /// Returns the number of stripes held by this guard.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns `true` if zero stripes are held by this guard.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// RAII guard holding exclusive write locks on one or more [`super::ChunkLockTable`]
/// stripes acquired in strict ascending index order.
pub struct ClusterWriteGuard<'a> {
    storage: ClusterStorage<RwLockWriteGuard<'a, ()>>,
}

impl<'a> ClusterWriteGuard<'a> {
    #[inline]
    pub(super) const fn empty() -> Self {
        Self {
            storage: ClusterStorage::empty(),
        }
    }

    #[inline]
    pub(super) const fn from_inline(
        len: usize,
        indices: [usize; INLINE_STRIPE_CAP],
        guards: [Option<RwLockWriteGuard<'a, ()>>; INLINE_STRIPE_CAP],
    ) -> Self {
        Self {
            storage: ClusterStorage::from_inline(len, indices, guards),
        }
    }

    #[inline]
    pub(super) fn from_heap(
        stripe_indices: Vec<usize>,
        guards: Vec<RwLockWriteGuard<'a, ()>>,
    ) -> Self {
        Self {
            storage: ClusterStorage::from_heap(stripe_indices, guards),
        }
    }

    /// Returns the strictly ascending stripe indices held by this guard.
    #[inline]
    #[must_use]
    pub fn stripe_indices(&self) -> &[usize] {
        self.storage.stripe_indices()
    }

    /// Returns the number of stripes held by this guard.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns `true` if zero stripes are held by this guard.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
