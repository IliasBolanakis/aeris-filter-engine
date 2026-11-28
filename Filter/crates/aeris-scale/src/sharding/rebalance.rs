//! Dynamic shard split, adjacent shard merge, and automated rebalance
//! operations for [`super::ShardedAerisSystem`].

use super::config::MAX_POST_MERGE_LOAD_FACTOR;
use super::routing::{compute_median_split_key, RoutingSnapshot};
use super::shard::{build_shard_from_state, ShardInstance};
use super::ShardedAerisSystem;
use crate::epoch_cell::try_reclaim_retired_epochs;
use aeris_types::{
    AerisError, AerisResult, Epoch, FilterConfig, FpLogEntry, Key, ShardBounds, ShardId,
};
use parking_lot::MutexGuard;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Diagnostic summary returned by [`ShardedAerisSystem::split_shard`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitReport {
    /// Identifier of the retired parent shard.
    pub old_shard_id: ShardId,
    /// Identifier of the newly created lower-half child shard `[min, split_key]`.
    pub left_shard_id: ShardId,
    /// Identifier of the newly created upper-half child shard `[split_key + 1, max]`.
    pub right_shard_id: ShardId,
    /// Data-driven median split boundary `split_key` (inclusive upper bound of left child).
    pub split_key: Key,
    /// Key bounds owned by `left_shard_id`.
    pub left_bounds: ShardBounds,
    /// Key bounds owned by `right_shard_id`.
    pub right_bounds: ShardBounds,
    /// Number of live keys migrated to `left_shard_id`.
    pub left_keys: usize,
    /// Number of live keys migrated to `right_shard_id`.
    pub right_keys: usize,
    /// Number of False-Positive Log entries replayed onto `left_shard_id`.
    pub fp_entries_replayed_left: usize,
    /// Number of False-Positive Log entries replayed onto `right_shard_id`.
    pub fp_entries_replayed_right: usize,
    /// Newly published routing topology [`Epoch`].
    pub new_routing_epoch: Epoch,
    /// Wall-clock duration of the split operation in nanoseconds.
    pub latency_ns: u64,
}

/// Diagnostic summary returned by [`ShardedAerisSystem::merge_adjacent_shards`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeReport {
    /// Identifier of the retired lower adjacent shard.
    pub left_shard_id: ShardId,
    /// Identifier of the retired upper adjacent shard.
    pub right_shard_id: ShardId,
    /// Identifier of the newly created merged shard `[left.min, right.max]`.
    pub merged_shard_id: ShardId,
    /// Key bounds owned by `merged_shard_id`.
    pub merged_bounds: ShardBounds,
    /// Total live keys consolidated into `merged_shard_id`.
    pub merged_keys: usize,
    /// Total combined False-Positive Log entries replayed onto `merged_shard_id`.
    pub fp_entries_replayed: usize,
    /// Newly published routing topology [`Epoch`].
    pub new_routing_epoch: Epoch,
    /// Wall-clock duration of the merge operation in nanoseconds.
    pub latency_ns: u64,
}

/// Diagnostic summary returned by [`ShardedAerisSystem::rebalance_if_needed`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RebalanceReport {
    /// Splits performed to relieve overloaded or hot-spot shards.
    pub splits: Vec<SplitReport>,
    /// Adjacent merges performed to reclaim RAM from underloaded shards.
    pub merges: Vec<MergeReport>,
    /// Active shard count after rebalancing completed.
    pub final_shard_count: usize,
    /// Active routing topology [`Epoch`] after rebalancing completed.
    pub final_routing_epoch: Epoch,
}

struct SplitChildren {
    left_id: ShardId,
    right_id: ShardId,
    split_key: Key,
    left_bounds: ShardBounds,
    right_bounds: ShardBounds,
    left_keys: usize,
    right_keys: usize,
    fp_left: usize,
    fp_right: usize,
    left_shard: ShardInstance,
    right_shard: ShardInstance,
}

struct MergePairContext {
    left_idx: usize,
    old_shards: Vec<Arc<ShardInstance>>,
    left: Arc<ShardInstance>,
    right: Arc<ShardInstance>,
}

impl ShardedAerisSystem {
    /// Splits `shard_id` at the exact **data-driven median key** of its live key
    /// set into two balanced contiguous child shards (`[min, split_key]` and
    /// `[split_key + 1, max]`), migrating and replaying relevant [`aeris_control::FalsePositiveLog`]
    /// entries for 100% monotonic adaptivity and publishing the updated
    /// [`RoutingSnapshot`] atomically via [`crate::EpochCell::swap`].
    ///
    /// # Errors
    /// - Returns [`AerisError::ShardNotFound`] if `shard_id` is not active.
    /// - Returns [`AerisError::InvalidConfig`] if splitting would exceed `max_shards`
    ///   or if `shard_id` spans only a single key (`min_key == max_key`).
    /// - Propagates RocksDB, filter, or FP log errors.
    pub fn split_shard(&self, shard_id: ShardId) -> AerisResult<SplitReport> {
        let topo_guard = self.topology_lock.lock();
        self.split_shard_locked(&topo_guard, shard_id)
    }

    /// Merges two strictly adjacent shards (`left_id`, `right_id` where
    /// `left.max_key + 1 == right.min_key`) into a single consolidated shard
    /// `[left.min_key, right.max_key]`, combining both shards' live keys and
    /// [`aeris_control::FalsePositiveLog`]s with full monotonic replay.
    ///
    /// Acquires `left.write_coordinator` followed by `right.write_coordinator` in
    /// **strict ascending `ShardBounds::min_key` order** to prevent deadlocks.
    ///
    /// # Errors
    /// - Returns [`AerisError::ShardNotFound`] if either shard is not active.
    /// - Returns [`AerisError::InvalidConfig`] if the shards are not adjacent in
    ///   `(left, right)` order or if merging would drop below `min_shards`.
    /// - Propagates RocksDB, filter, or FP log errors.
    pub fn merge_adjacent_shards(
        &self,
        left_id: ShardId,
        right_id: ShardId,
    ) -> AerisResult<MergeReport> {
        let topo_guard = self.topology_lock.lock();
        self.merge_adjacent_shards_locked(&topo_guard, left_id, right_id)
    }

    /// Evaluates all active shards under `topology_lock`, automatically splitting
    /// overloaded/hot-spot shards (exceeding `split_load_threshold`, `max_shard_q`,
    /// or `split_key_threshold`) and merging adjacent underloaded shards (both
    /// `< merge_load_threshold` with combined occupancy `<= 0.50`).
    ///
    /// # Errors
    /// Propagates any storage, filter, or FP log error encountered during split or merge.
    pub fn rebalance_if_needed(&self) -> AerisResult<RebalanceReport> {
        let topo_guard = self.topology_lock.lock();
        let mut report = RebalanceReport::default();

        self.run_split_pass(&topo_guard, &mut report)?;
        self.run_merge_pass(&topo_guard, &mut report)?;

        report.final_shard_count = self.shard_count();
        report.final_routing_epoch = self.routing_epoch();
        debug!(
            splits = report.splits.len(),
            merges = report.merges.len(),
            final_shard_count = report.final_shard_count,
            epoch = report.final_routing_epoch,
            "completed ShardedAerisSystem rebalance pass"
        );
        Ok(report)
    }

    fn run_split_pass(
        &self,
        topo_guard: &MutexGuard<'_, ()>,
        report: &mut RebalanceReport,
    ) -> AerisResult<()> {
        let mut unsplitable: HashSet<ShardId> = HashSet::new();
        for _ in 0..self.config.max_shards {
            let candidate = self.routing.with_ref(|snap| {
                if snap.shards.len() >= self.config.max_shards {
                    return None;
                }
                snap.shards
                    .iter()
                    .find(|s| !unsplitable.contains(&s.shard_id) && self.should_split_shard(s))
                    .map(|s| s.shard_id)
            });

            let Some(victim_id) = candidate else {
                break;
            };
            match self.split_shard_locked(topo_guard, victim_id) {
                Ok(split_rep) => report.splits.push(split_rep),
                Err(AerisError::InvalidConfig(reason)) => {
                    warn!(
                        shard_id = victim_id,
                        reason, "skipping unsplittable shard during rebalance"
                    );
                    unsplitable.insert(victim_id);
                }
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    fn run_merge_pass(
        &self,
        topo_guard: &MutexGuard<'_, ()>,
        report: &mut RebalanceReport,
    ) -> AerisResult<()> {
        for _ in 0..self.config.max_shards {
            let candidate_pair = self.routing.with_ref(|snap| {
                if snap.shards.len() <= self.config.min_shards {
                    return None;
                }
                snap.shards.windows(2).find_map(|pair| {
                    self.should_merge_pair(&pair[0], &pair[1])
                        .then_some((pair[0].shard_id, pair[1].shard_id))
                })
            });

            let Some((left_id, right_id)) = candidate_pair else {
                break;
            };
            let merge_rep = self.merge_adjacent_shards_locked(topo_guard, left_id, right_id)?;
            report.merges.push(merge_rep);
        }
        Ok(())
    }

    fn should_split_shard(&self, shard: &ShardInstance) -> bool {
        if shard.bounds.min_key >= shard.bounds.max_key {
            return false;
        }
        let (key_count, current_q, load) = shard.rcu_filter.split_check_metrics();
        if key_count < self.config.split_min_keys || key_count < 2 {
            return false;
        }
        let exceeds_load = load >= self.config.split_load_threshold;
        let exceeds_max_q = current_q > self.config.max_shard_q;
        let exceeds_key_cap = self
            .config
            .split_key_threshold
            .is_some_and(|limit| key_count >= limit);
        exceeds_load || exceeds_max_q || exceeds_key_cap
    }

    fn should_merge_pair(&self, left: &ShardInstance, right: &ShardInstance) -> bool {
        let left_load = left.rcu_filter.load_factor();
        let right_load = right.rcu_filter.load_factor();
        if left_load >= self.config.merge_load_threshold
            || right_load >= self.config.merge_load_threshold
        {
            return false;
        }
        let left_q = left.rcu_filter.config().q;
        let right_q = right.rcu_filter.config().q;
        let target_q = self.config.base_filter_config.q.max(left_q.min(right_q));
        let target_cap = (1usize << target_q) as f64;
        let combined_slots =
            (left.rcu_filter.occupied_slots() + right.rcu_filter.occupied_slots()) as f64;
        let combined_keys = (left.rcu_filter.len() + right.rcu_filter.len()) as f64;

        (combined_slots / target_cap) <= MAX_POST_MERGE_LOAD_FACTOR
            && (combined_keys / target_cap) <= MAX_POST_MERGE_LOAD_FACTOR
    }

    /// Derives the baseline [`FilterConfig`] for newly created split/merged children,
    /// resetting `q` to `self.config.base_filter_config.q` while preserving dynamic
    /// `(r, m, max_range_r, target_epsilon)` tuning from `parent_cfg`.
    fn derived_child_filter_config(&self, parent_cfg: &FilterConfig) -> FilterConfig {
        let mut child_cfg = parent_cfg.clone();
        child_cfg.q = self.config.base_filter_config.q;
        child_cfg
    }

    /// Executes `split_shard` while the caller holds `self.topology_lock`.
    fn split_shard_locked(
        &self,
        _topo_guard: &MutexGuard<'_, ()>,
        shard_id: ShardId,
    ) -> AerisResult<SplitReport> {
        let start = Instant::now();
        let (shard_idx, old_shards, target) = self.lookup_split_target(shard_id)?;

        // Level 2 lock (exclusive): quiesce writers on `target` only; readers on
        // `target` and all traffic on every other shard proceed uninterrupted.
        let write_guard = target.write_coordinator.write();
        if target.is_retired() {
            return Err(AerisError::ShardNotFound(shard_id));
        }

        let children = self.build_split_children(&target)?;
        let mut new_shards = Vec::with_capacity(old_shards.len() + 1);
        new_shards.extend_from_slice(&old_shards[..shard_idx]);
        new_shards.push(Arc::new(children.left_shard));
        new_shards.push(Arc::new(children.right_shard));
        new_shards.extend_from_slice(&old_shards[shard_idx + 1..]);
        let new_snapshot = RoutingSnapshot::new(new_shards)?;

        // Publish new RoutingSnapshot FIRST, then mark `target` retired BEFORE releasing
        // `write_guard`, guaranteeing waiting writers immediately observe the new route.
        let new_epoch = self.routing.swap(new_snapshot);
        target.mark_retired();
        drop(write_guard);
        drop(old_shards);
        drop(target);
        try_reclaim_retired_epochs();

        let latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            shard_id,
            left_id = children.left_id,
            right_id = children.right_id,
            split_key = children.split_key,
            left_keys = children.left_keys,
            right_keys = children.right_keys,
            fp_left = children.fp_left,
            fp_right = children.fp_right,
            epoch = new_epoch,
            latency_ns,
            "completed dynamic range shard split"
        );

        Ok(SplitReport {
            old_shard_id: shard_id,
            left_shard_id: children.left_id,
            right_shard_id: children.right_id,
            split_key: children.split_key,
            left_bounds: children.left_bounds,
            right_bounds: children.right_bounds,
            left_keys: children.left_keys,
            right_keys: children.right_keys,
            fp_entries_replayed_left: children.fp_left,
            fp_entries_replayed_right: children.fp_right,
            new_routing_epoch: new_epoch,
            latency_ns,
        })
    }

    fn lookup_split_target(
        &self,
        shard_id: ShardId,
    ) -> AerisResult<(usize, Vec<Arc<ShardInstance>>, Arc<ShardInstance>)> {
        let (idx, old_shards, target) = self.routing.with_ref(|snap| {
            let idx = snap
                .find_shard_index(shard_id)
                .ok_or(AerisError::ShardNotFound(shard_id))?;
            Ok::<_, AerisError>((idx, snap.shards.to_vec(), Arc::clone(&snap.shards[idx])))
        })?;

        if old_shards.len() >= self.config.max_shards {
            return Err(AerisError::InvalidConfig(format!(
                "cannot split shard {shard_id}: shard count {} already at max_shards {}",
                old_shards.len(),
                self.config.max_shards
            )));
        }
        if target.bounds.min_key >= target.bounds.max_key {
            return Err(AerisError::InvalidConfig(format!(
                "cannot split single-key shard {shard_id} [{}, {}]",
                target.bounds.min_key, target.bounds.max_key
            )));
        }
        Ok((idx, old_shards, target))
    }

    fn build_split_children(&self, target: &ShardInstance) -> AerisResult<SplitChildren> {
        let all_keys = target.active_reverse_map().scan_all_keys()?;
        let (_, fp_entries) = target.fp_log.snapshot_entries();

        let split_key = compute_median_split_key(target.bounds, &all_keys);
        let left_bounds = ShardBounds::new(target.bounds.min_key, split_key)?;
        let right_bounds = ShardBounds::new(split_key + 1, target.bounds.max_key)?;

        let split_pos = all_keys.partition_point(|&k| k <= split_key);
        let left_keys = &all_keys[..split_pos];
        let right_keys = &all_keys[split_pos..];

        let left_id = self.allocate_shard_id();
        let right_id = self.allocate_shard_id();
        let child_cfg = self.derived_child_filter_config(&target.rcu_filter.config());

        let (left_shard, fp_left) = build_shard_from_state(
            &self.root_dir,
            &self.config,
            left_id,
            left_bounds,
            &child_cfg,
            left_keys,
            &fp_entries,
        )?;
        let (right_shard, fp_right) = build_shard_from_state(
            &self.root_dir,
            &self.config,
            right_id,
            right_bounds,
            &child_cfg,
            right_keys,
            &fp_entries,
        )?;

        Ok(SplitChildren {
            left_id,
            right_id,
            split_key,
            left_bounds,
            right_bounds,
            left_keys: left_keys.len(),
            right_keys: right_keys.len(),
            fp_left,
            fp_right,
            left_shard,
            right_shard,
        })
    }

    /// Executes `merge_adjacent_shards` while the caller holds `self.topology_lock`.
    fn merge_adjacent_shards_locked(
        &self,
        _topo_guard: &MutexGuard<'_, ()>,
        left_id: ShardId,
        right_id: ShardId,
    ) -> AerisResult<MergeReport> {
        let start = Instant::now();
        let MergePairContext {
            left_idx,
            old_shards,
            left,
            right,
        } = self.lookup_merge_pair(left_id, right_id)?;

        // Level 2 lock (exclusive): strictly ascending `ShardBounds::min_key` order (`left` then `right`).
        let left_write = left.write_coordinator.write();
        let right_write = right.write_coordinator.write();
        if left.is_retired() {
            return Err(AerisError::ShardNotFound(left_id));
        }
        if right.is_retired() {
            return Err(AerisError::ShardNotFound(right_id));
        }

        let (merged_keys, combined_fps) = collect_merged_keys_and_fps(&left, &right)?;
        let merged_bounds = ShardBounds::new(left.bounds.min_key, right.bounds.max_key)?;
        let merged_id = self.allocate_shard_id();
        let child_cfg = self.derived_child_filter_config(&left.rcu_filter.config());

        let (merged_shard, fp_replayed) = build_shard_from_state(
            &self.root_dir,
            &self.config,
            merged_id,
            merged_bounds,
            &child_cfg,
            &merged_keys,
            &combined_fps,
        )?;

        let mut new_shards = Vec::with_capacity(old_shards.len() - 1);
        new_shards.extend_from_slice(&old_shards[..left_idx]);
        new_shards.push(Arc::new(merged_shard));
        new_shards.extend_from_slice(&old_shards[left_idx + 2..]);
        let new_snapshot = RoutingSnapshot::new(new_shards)?;

        // Publish new RoutingSnapshot FIRST, then mark both old shards retired BEFORE
        // releasing their `write_coordinator` locks.
        let new_epoch = self.routing.swap(new_snapshot);
        left.mark_retired();
        right.mark_retired();
        drop(right_write);
        drop(left_write);
        drop(old_shards);
        drop(right);
        drop(left);
        try_reclaim_retired_epochs();

        let latency_ns = start.elapsed().as_nanos() as u64;
        info!(
            left_id,
            right_id,
            merged_id,
            merged_keys = merged_keys.len(),
            fp_replayed,
            epoch = new_epoch,
            latency_ns,
            "completed adjacent range shard merge"
        );

        Ok(MergeReport {
            left_shard_id: left_id,
            right_shard_id: right_id,
            merged_shard_id: merged_id,
            merged_bounds,
            merged_keys: merged_keys.len(),
            fp_entries_replayed: fp_replayed,
            new_routing_epoch: new_epoch,
            latency_ns,
        })
    }

    fn lookup_merge_pair(
        &self,
        left_id: ShardId,
        right_id: ShardId,
    ) -> AerisResult<MergePairContext> {
        let (left_idx, old_shards, left, right) = self.routing.with_ref(|snap| {
            let l_idx = snap
                .find_shard_index(left_id)
                .ok_or(AerisError::ShardNotFound(left_id))?;
            let r_idx = snap
                .find_shard_index(right_id)
                .ok_or(AerisError::ShardNotFound(right_id))?;
            if l_idx + 1 != r_idx {
                return Err(AerisError::InvalidConfig(format!(
                    "shards ({left_id}, {right_id}) are not adjacent in ascending key order"
                )));
            }
            Ok::<_, AerisError>((
                l_idx,
                snap.shards.to_vec(),
                Arc::clone(&snap.shards[l_idx]),
                Arc::clone(&snap.shards[r_idx]),
            ))
        })?;

        if old_shards.len() <= self.config.min_shards {
            return Err(AerisError::InvalidConfig(format!(
                "cannot merge shards ({left_id}, {right_id}): shard count {} already at min_shards {}",
                old_shards.len(),
                self.config.min_shards
            )));
        }
        debug_assert!(
            left.bounds.min_key < right.bounds.min_key
                && left.bounds.max_key.checked_add(1) == Some(right.bounds.min_key),
            "adjacent shards must be strictly ascending and contiguous"
        );
        Ok(MergePairContext {
            left_idx,
            old_shards,
            left,
            right,
        })
    }
}

fn collect_merged_keys_and_fps(
    left: &ShardInstance,
    right: &ShardInstance,
) -> AerisResult<(Vec<Key>, Vec<FpLogEntry>)> {
    let left_keys = left.active_reverse_map().scan_all_keys()?;
    let right_keys = right.active_reverse_map().scan_all_keys()?;
    let mut merged_keys = Vec::with_capacity(left_keys.len() + right_keys.len());
    merged_keys.extend_from_slice(&left_keys);
    merged_keys.extend_from_slice(&right_keys);
    debug_assert!(
        merged_keys.windows(2).all(|w| w[0] < w[1]),
        "disjoint adjacent shard keys must concatenate in strict ascending order"
    );

    let (_, mut combined_fps) = left.fp_log.snapshot_entries();
    let (_, right_fps) = right.fp_log.snapshot_entries();
    combined_fps.extend(right_fps);
    Ok((merged_keys, combined_fps))
}
