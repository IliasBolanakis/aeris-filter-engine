//! Reverse-map store implementations for `paper_parity` benchmarks.
//!
//! Provides both [`NoopReverseMap`] (zero-overhead scaling runs) and
//! [`InMemReverseMap`] (in-memory DRAM-parity store for adaptive benchmarks).

use aeris_types::{AerisResult, Fingerprint, Key, ReverseMapStore};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Zero-overhead no-op [`ReverseMapStore`] for pure filter `bulk_load` and
/// `dynamic_insert` scaling runs (parity with C++ pure `QF` benchmarks).
pub struct NoopReverseMap;

impl ReverseMapStore for NoopReverseMap {
    fn append_key(&self, _fp: &Fingerprint, _key: Key) -> AerisResult<()> {
        Ok(())
    }

    fn append_batch(&self, _entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
        Ok(())
    }

    fn remove_key(&self, _fp: &Fingerprint, _key: Key) -> AerisResult<()> {
        Ok(())
    }

    fn get_keys(&self, _fp: &Fingerprint) -> AerisResult<Vec<Key>> {
        Ok(Vec::new())
    }

    fn tombstone(&self, _fp: &Fingerprint) -> AerisResult<()> {
        Ok(())
    }

    fn replace_fingerprint(
        &self,
        _old_fp: &Fingerprint,
        _new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        Ok(())
    }

    fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        Ok(Vec::new())
    }

    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
        Ok(Box::new(std::iter::empty()))
    }

    fn entry_count_estimate(&self) -> u64 {
        0
    }

    fn flush(&self) -> AerisResult<()> {
        Ok(())
    }

    fn compact(&self) -> AerisResult<()> {
        Ok(())
    }

    #[inline]
    fn is_noop(&self) -> bool {
        true
    }
}

/// High-speed in-memory [`ReverseMapStore`] backed by `parking_lot::RwLock<HashMap>`
/// for DRAM-parity adaptive benchmarks (matching C++ `InMemArqf`'s `unordered_multimap`).
pub struct InMemReverseMap {
    map: RwLock<HashMap<Fingerprint, Vec<Key>>>,
    entry_count: AtomicU64,
}

impl InMemReverseMap {
    /// Creates an empty [`InMemReverseMap`] pre-sized for `capacity` fingerprint buckets.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(capacity)),
            entry_count: AtomicU64::new(0),
        }
    }
}

impl ReverseMapStore for InMemReverseMap {
    fn append_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let mut guard = self.map.write();
        guard.entry(fp.clone()).or_default().push(key);
        self.entry_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn append_batch(&self, entries: &[(Fingerprint, Key)]) -> AerisResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut guard = self.map.write();
        if guard.is_empty() {
            guard.reserve(entries.len());
        }
        for (fp, key) in entries {
            guard.entry(fp.clone()).or_default().push(*key);
        }
        self.entry_count
            .fetch_add(entries.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn remove_key(&self, fp: &Fingerprint, key: Key) -> AerisResult<()> {
        let mut guard = self.map.write();
        let mut should_remove = false;
        if let Some(vec) = guard.get_mut(fp) {
            if let Some(pos) = vec.iter().position(|&k| k == key) {
                vec.swap_remove(pos);
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
            }
            should_remove = vec.is_empty();
        }
        if should_remove {
            guard.remove(fp);
        }
        Ok(())
    }

    fn get_keys(&self, fp: &Fingerprint) -> AerisResult<Vec<Key>> {
        let guard = self.map.read();
        Ok(guard.get(fp).cloned().unwrap_or_default())
    }

    fn tombstone(&self, fp: &Fingerprint) -> AerisResult<()> {
        let mut guard = self.map.write();
        if let Some(removed) = guard.remove(fp) {
            self.entry_count
                .fetch_sub(removed.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn replace_fingerprint(
        &self,
        old_fp: &Fingerprint,
        new_entries: &[(Fingerprint, Key)],
    ) -> AerisResult<()> {
        let mut guard = self.map.write();
        let removed_count = guard.remove(old_fp).map(|v| v.len()).unwrap_or(0);
        for (fp, key) in new_entries {
            guard.entry(fp.clone()).or_default().push(*key);
        }
        let added = new_entries.len();
        if added >= removed_count {
            self.entry_count
                .fetch_add((added - removed_count) as u64, Ordering::Relaxed);
        } else {
            self.entry_count
                .fetch_sub((removed_count - added) as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn scan_all_keys(&self) -> AerisResult<Vec<Key>> {
        let guard = self.map.read();
        let mut all = Vec::with_capacity(self.entry_count.load(Ordering::Relaxed) as usize);
        for keys in guard.values() {
            all.extend_from_slice(keys);
        }
        all.sort_unstable();
        all.dedup();
        Ok(all)
    }

    fn iter_all_keys(&self) -> AerisResult<Box<dyn Iterator<Item = AerisResult<Key>> + '_>> {
        let keys = self.scan_all_keys()?;
        Ok(Box::new(keys.into_iter().map(Ok)))
    }

    fn entry_count_estimate(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }

    fn flush(&self) -> AerisResult<()> {
        Ok(())
    }

    fn compact(&self) -> AerisResult<()> {
        Ok(())
    }
}
