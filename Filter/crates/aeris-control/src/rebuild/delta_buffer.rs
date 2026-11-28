//! Thread-safe FIFO [`DeltaBuffer`] capturing concurrent mutations during
//! asynchronous background rebuilds.

use aeris_types::Key;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::debug;

/// Default batch size (mutations per off-lock pass) for incremental [`DeltaBuffer`] pre-draining.
pub const DEFAULT_DELTA_PRE_DRAIN_BATCH_SIZE: usize = 1024;

/// Default residual [`DeltaBuffer`] tail threshold below which off-lock pre-draining stops
/// and hands off to the brief locked tail drain + atomic swap.
pub const DEFAULT_DELTA_PRE_DRAIN_TAIL_THRESHOLD: usize = 256;

/// Default maximum number of off-lock pre-drain batch passes before entering the locked tail swap.
pub const DEFAULT_DELTA_PRE_DRAIN_MAX_PASSES: usize = 64;

/// Single mutation captured in [`DeltaBuffer`] while an asynchronous rebuild is in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeltaMutation {
    /// Key inserted into `F_old` during background rebuild.
    Insert(Key),
    /// Key deleted from `F_old` during background rebuild.
    Delete(Key),
}

/// Internal mutex-protected state for [`DeltaBuffer`].
#[derive(Debug, Default)]
struct DeltaBufferInner {
    active: bool,
    ops: VecDeque<DeltaMutation>,
}

/// Thread-safe FIFO buffer recording concurrent mutations (`Insert` / `Delete`)
/// arriving while a background rebuild constructs `F_new`.
///
/// # Lock Ordering
/// `DeltaBuffer::inner` is a leaf mutex; it never acquires any other lock.
#[derive(Debug, Default)]
pub struct DeltaBuffer {
    inner: Mutex<DeltaBufferInner>,
}

impl DeltaBuffer {
    /// Creates an inactive, empty [`DeltaBuffer`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Activates mutation buffering and clears any leftover entries from prior runs.
    pub fn activate(&self) {
        let mut inner = self.inner.lock();
        inner.ops.clear();
        inner.active = true;
        debug!("activated DeltaBuffer for background rebuild");
    }

    /// Deactivates mutation buffering without discarding already-buffered operations.
    pub fn deactivate(&self) {
        let mut inner = self.inner.lock();
        inner.active = false;
        debug!(remaining_ops = inner.ops.len(), "deactivated DeltaBuffer");
    }

    /// Returns `true` if the buffer is currently capturing incoming mutations.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner.lock().active
    }

    /// Records an inserted `key` if the buffer is active, returning `true` if captured.
    pub fn record_insert(&self, key: Key) -> bool {
        let mut inner = self.inner.lock();
        if !inner.active {
            return false;
        }
        inner.ops.push_back(DeltaMutation::Insert(key));
        true
    }

    /// Records a deleted `key` if the buffer is active, returning `true` if captured.
    pub fn record_delete(&self, key: Key) -> bool {
        let mut inner = self.inner.lock();
        if !inner.active {
            return false;
        }
        inner.ops.push_back(DeltaMutation::Delete(key));
        true
    }

    /// Drains and returns up to `max_items` buffered mutations from the front of the
    /// queue in strict FIFO arrival order while keeping the buffer active for concurrent
    /// writers appending to the tail.
    ///
    /// # Complexity
    /// `O(min(max_items, len))` time and space; acquires `DeltaBuffer::inner` briefly.
    #[must_use]
    pub fn drain_batch(&self, max_items: usize) -> Vec<DeltaMutation> {
        if max_items == 0 {
            return Vec::new();
        }
        let mut inner = self.inner.lock();
        let take = max_items.min(inner.ops.len());
        inner.ops.drain(..take).collect()
    }

    /// Drains and returns all buffered mutations in strict FIFO arrival order.
    #[must_use]
    pub fn drain_all(&self) -> Vec<DeltaMutation> {
        let mut inner = self.inner.lock();
        inner.ops.drain(..).collect()
    }

    /// Atomically drains all buffered mutations in FIFO order AND deactivates the buffer
    /// under a single lock acquisition, preventing races at the swap boundary.
    #[must_use]
    pub fn drain_and_deactivate(&self) -> Vec<DeltaMutation> {
        let mut inner = self.inner.lock();
        inner.active = false;
        inner.ops.drain(..).collect()
    }

    /// Returns the number of mutations currently queued in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().ops.len()
    }

    /// Returns `true` if zero mutations are queued in the buffer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().ops.is_empty()
    }
}
