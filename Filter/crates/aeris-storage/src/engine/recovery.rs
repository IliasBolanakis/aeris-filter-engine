//! WAL replay, KV/Reverse Map reconciliation, and filter rebuild helpers for
//! [`super::CoupledStorageEngine`].

use crate::kv_store::ReferenceKvStore;
use crate::reverse_map::RocksDbReverseMap;
use crate::wal::CoupledWal;
use aeris_types::{
    AerisError, AerisResult, Fingerprint, Key, RangeFilter, ReverseMapStore, WalEngine, WalOp,
    WalRecord, MAX_QUOTIENT_BITS,
};
use std::collections::{BTreeMap, BTreeSet};
use tracing::warn;

/// Counters returned by [`replay_wal_records`].
pub(super) struct ReplayCounts {
    pub(super) puts: usize,
    pub(super) deletes: usize,
    pub(super) adapts: usize,
}

impl ReplayCounts {
    pub(super) const fn total(&self) -> usize {
        self.puts + self.deletes + self.adapts
    }
}

/// Idempotently replays `records` into `kv_store` and `reverse_map`.
pub(super) fn replay_wal_records(
    records: Vec<WalRecord>,
    kv_store: &ReferenceKvStore,
    reverse_map: &RocksDbReverseMap,
) -> AerisResult<ReplayCounts> {
    let mut counts = ReplayCounts {
        puts: 0,
        deletes: 0,
        adapts: 0,
    };
    for record in records {
        match record.op {
            WalOp::Put { key, value, fp } => {
                kv_store.put(key, &value)?;
                reverse_map.append_key(&fp, key)?;
                counts.puts += 1;
            }
            WalOp::Delete { key, fp } => {
                kv_store.delete(key)?;
                reverse_map.remove_key(&fp, key)?;
                counts.deletes += 1;
            }
            WalOp::Adapt {
                old_fps,
                new_entries,
                ..
            } => {
                reverse_map.replace_fingerprints_batch(&old_fps, &new_entries)?;
                counts.adapts += 1;
            }
            WalOp::Checkpoint { .. } => {}
        }
    }
    Ok(counts)
}

/// Incremental reconciliation state used by
/// [`super::CoupledStorageEngine::rebuild_filter_and_reconcile`].
pub(super) struct RebuildReconcileState<'a> {
    existing: &'a BTreeMap<Fingerprint, BTreeSet<Key>>,
    desired_before_expand: BTreeMap<Fingerprint, BTreeSet<Key>>,
    pending_after_expand: Vec<(Fingerprint, Key)>,
    pub(super) did_expand: bool,
    pub(super) tombstoned_count: usize,
    pub(super) merged_count: usize,
}

impl<'a> RebuildReconcileState<'a> {
    pub(super) fn new(existing: &'a BTreeMap<Fingerprint, BTreeSet<Key>>) -> Self {
        Self {
            existing,
            desired_before_expand: BTreeMap::new(),
            pending_after_expand: Vec::new(),
            did_expand: false,
            tombstoned_count: 0,
            merged_count: 0,
        }
    }

    pub(super) fn record_inserted(&mut self, fp: Fingerprint, key: Key) {
        if self.did_expand {
            self.pending_after_expand.push((fp, key));
        } else {
            self.desired_before_expand
                .entry(fp)
                .or_default()
                .insert(key);
        }
    }

    fn flush_before_expand(&mut self, reverse_map: &RocksDbReverseMap) -> AerisResult<()> {
        if !self.did_expand {
            // Synchronize Reverse Map with keys inserted so far before the first expand()
            // so any selective rejuvenation invoked by expand() sees exact entries.
            let (to_tombstone, to_merge) =
                compute_reverse_map_diff(self.existing, &self.desired_before_expand);
            self.tombstoned_count += to_tombstone.len();
            self.merged_count += to_merge.len();
            reverse_map.replace_fingerprints_batch(&to_tombstone, &to_merge)?;
            self.did_expand = true;
            return Ok(());
        }
        if !self.pending_after_expand.is_empty() {
            self.merged_count += self.pending_after_expand.len();
            reverse_map.append_batch(&self.pending_after_expand)?;
            self.pending_after_expand.clear();
        }
        Ok(())
    }

    pub(super) fn finalize(&mut self, reverse_map: &RocksDbReverseMap) -> AerisResult<()> {
        if self.did_expand {
            if !self.pending_after_expand.is_empty() {
                self.merged_count += self.pending_after_expand.len();
                reverse_map.append_batch(&self.pending_after_expand)?;
                self.pending_after_expand.clear();
            }
            reverse_map.flush()?;
            let _ = reverse_map.refresh_entry_count()?;
            return Ok(());
        }

        let (to_tombstone, to_merge) =
            compute_reverse_map_diff(self.existing, &self.desired_before_expand);
        self.tombstoned_count = to_tombstone.len();
        self.merged_count = to_merge.len();
        if !to_tombstone.is_empty() || !to_merge.is_empty() {
            reverse_map.replace_fingerprints_batch(&to_tombstone, &to_merge)?;
            reverse_map.flush()?;
            let _ = reverse_map.refresh_entry_count()?;
        }
        Ok(())
    }
}

/// Inserts `key` into `filter`, flushing pending Reverse Map state and invoking
/// `filter.expand` whenever `FilterFull` occurs (up to `MAX_QUOTIENT_BITS` times).
pub(super) fn insert_or_expand_during_rebuild<F: RangeFilter>(
    filter: &mut F,
    reverse_map: &RocksDbReverseMap,
    state: &mut RebuildReconcileState<'_>,
    key: Key,
) -> AerisResult<Fingerprint> {
    let mut expansions = 0u8;
    loop {
        match filter.insert(key) {
            Ok(fp) => return Ok(fp),
            Err(AerisError::FilterFull { .. }) if expansions < MAX_QUOTIENT_BITS => {
                state.flush_before_expand(reverse_map)?;
                filter.expand(reverse_map)?;
                expansions += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Computes the minimal `(old_fps_to_tombstone, new_entries_to_merge)` required to
/// bring `existing` Reverse Map entries into exact parity with `desired`.
pub(super) fn compute_reverse_map_diff(
    existing: &BTreeMap<Fingerprint, BTreeSet<Key>>,
    desired: &BTreeMap<Fingerprint, BTreeSet<Key>>,
) -> (Vec<Fingerprint>, Vec<(Fingerprint, Key)>) {
    let mut to_tombstone = Vec::new();
    let mut tombstoned_set = BTreeSet::new();

    for (fp, existing_keys) in existing {
        let needs_tombstone = match desired.get(fp) {
            None => true,
            Some(desired_keys) => !existing_keys.is_subset(desired_keys),
        };
        if needs_tombstone {
            to_tombstone.push(fp.clone());
            tombstoned_set.insert(fp.clone());
        }
    }

    let mut to_merge = Vec::new();
    for (fp, desired_keys) in desired {
        let existing_keys = if tombstoned_set.contains(fp) {
            None
        } else {
            existing.get(fp)
        };
        for &key in desired_keys {
            let already_present = existing_keys.is_some_and(|set| set.contains(&key));
            if !already_present {
                to_merge.push((fp.clone(), key));
            }
        }
    }

    (to_tombstone, to_merge)
}

/// Reconciles discrepancies between [`ReferenceKvStore`] and [`RocksDbReverseMap`]
/// after WAL replay. Returns `Ok(true)` if any corrective writes were performed.
pub(super) fn reconcile_kv_and_reverse_map(
    wal: &CoupledWal,
    kv_store: &ReferenceKvStore,
    reverse_map: &RocksDbReverseMap,
) -> AerisResult<bool> {
    let kv_keys: BTreeSet<Key> = kv_store.scan_all_keys()?.into_iter().collect();
    let rm_entries = reverse_map.scan_all_entries()?;
    let mut rm_keys = BTreeSet::new();
    for keys in rm_entries.values() {
        rm_keys.extend(keys.iter().copied());
    }

    let orphans: BTreeSet<Key> = rm_keys.difference(&kv_keys).copied().collect();
    let missing: BTreeSet<Key> = kv_keys.difference(&rm_keys).copied().collect();
    if orphans.is_empty() && missing.is_empty() {
        return Ok(false);
    }

    warn!(
        kv_count = kv_keys.len(),
        rm_count = rm_keys.len(),
        orphan_rm_keys = orphans.len(),
        missing_rm_keys = missing.len(),
        "reconciling discrepancy between ReferenceKvStore and RocksDbReverseMap"
    );

    let removed_orphans = remove_orphan_rm_keys(reverse_map, &rm_entries, &orphans)?;
    let restored_missing = restore_missing_rm_keys(wal, reverse_map, &missing)?;
    Ok(removed_orphans || restored_missing)
}

fn remove_orphan_rm_keys(
    reverse_map: &RocksDbReverseMap,
    rm_entries: &BTreeMap<Fingerprint, BTreeSet<Key>>,
    orphans: &BTreeSet<Key>,
) -> AerisResult<bool> {
    let mut mutated = false;
    if orphans.is_empty() {
        return Ok(false);
    }
    for (fp, keys) in rm_entries {
        for &key in keys {
            if orphans.contains(&key) {
                reverse_map.remove_key(fp, key)?;
                mutated = true;
            }
        }
    }
    Ok(mutated)
}

fn restore_missing_rm_keys(
    wal: &CoupledWal,
    reverse_map: &RocksDbReverseMap,
    missing: &BTreeSet<Key>,
) -> AerisResult<bool> {
    if missing.is_empty() {
        return Ok(false);
    }
    let recovered_fps = recover_latest_fps_from_wal(wal, missing)?;
    let mut batch = Vec::with_capacity(recovered_fps.len());
    let mut unresolved = 0usize;
    for &key in missing {
        if let Some(fp) = recovered_fps.get(&key) {
            batch.push((fp.clone(), key));
        } else {
            unresolved += 1;
        }
    }
    let mutated = if !batch.is_empty() {
        reverse_map.append_batch(&batch)?;
        true
    } else {
        false
    };
    if unresolved > 0 {
        warn!(
            unresolved_keys = unresolved,
            "some ReferenceKvStore keys lacked WAL fingerprint history; invoke rebuild_filter_and_reconcile to re-index"
        );
    }
    Ok(mutated)
}

/// Scans retained WAL history to determine the latest [`Fingerprint`] associated with
/// each key in `target_keys`.
pub(super) fn recover_latest_fps_from_wal(
    wal: &CoupledWal,
    target_keys: &BTreeSet<Key>,
) -> AerisResult<BTreeMap<Key, Fingerprint>> {
    let all_records = wal.replay_from(1)?;
    let mut latest: BTreeMap<Key, Fingerprint> = BTreeMap::new();
    for rec in all_records {
        match rec.op {
            WalOp::Put { key, fp, .. } if target_keys.contains(&key) => {
                latest.insert(key, fp);
            }
            WalOp::Delete { key, .. } if target_keys.contains(&key) => {
                latest.remove(&key);
            }
            WalOp::Adapt { new_entries, .. } => {
                for (fp, key) in new_entries {
                    if target_keys.contains(&key) {
                        latest.insert(key, fp);
                    }
                }
            }
            WalOp::Put { .. } | WalOp::Delete { .. } | WalOp::Checkpoint { .. } => {}
        }
    }
    Ok(latest)
}
