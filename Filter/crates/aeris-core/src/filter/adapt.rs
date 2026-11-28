//! General overlay-planned range false-positive adaptation for [`AerisFilter`].

use super::AerisFilter;
use crate::keepsake::KeepsakeBox;
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, Fingerprint, KeepsakeSlotKind, Key, Memento,
    PartitionId, QueryRange, RangeFilter, ReverseMapStore,
};
use std::time::Instant;

type SubqueryHit = (PartitionId, Memento, Memento);

struct MatchedSubqueries {
    stack: [SubqueryHit; 8],
    stack_len: usize,
    overflow: Vec<SubqueryHit>,
}

impl MatchedSubqueries {
    fn as_slice(&self) -> &[SubqueryHit] {
        if self.overflow.is_empty() {
            &self.stack[..self.stack_len]
        } else {
            &self.overflow
        }
    }
}

struct AdaptOverlay {
    runs: Vec<(usize, usize, usize, Vec<KeepsakeBox>)>,
    keys: Vec<(Fingerprint, Vec<Key>)>,
    fresh_fps: Vec<Fingerprint>,
    tombstoned_disk_fps: Vec<Fingerprint>,
}

struct CollectedPartitionKeys {
    colliding_parts: Vec<(PartitionId, u16, Vec<Key>)>,
    query_part_min_bits: u16,
    query_part_keys: Vec<Key>,
    old_matched_slots: usize,
}

impl AerisFilter {
    pub(super) fn adapt_internal(
        &mut self,
        range: QueryRange,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<AdaptationReport> {
        let start = Instant::now();
        let mut report = AdaptationReport::default();
        if self.num_keys == 0 {
            return Ok(report);
        }

        let hits = self.collect_matching_subqueries(range, &mut report);
        if hits.stack_len == 0 {
            return Ok(report);
        }

        // Fast Path A (>95% of adaptations): single matching subquery hitting a single
        // fresh colliding box with one colliding partition and no peer remainder conflict.
        if hits.stack_len == 1 && hits.overflow.is_empty() {
            let (p_q, _, _) = hits.stack[0];
            let (home_u64, _) = self.quotient_and_remainder_hot(p_q);
            if self
                .try_adapt_single_box_fast(p_q, home_u64 as usize, reverse_map, &mut report)?
                .is_some()
            {
                self.log_adapt_done(range, &report, start, "in-place fast path");
                return Ok(report);
            }
        }

        let matching_slice = hits.as_slice();

        let mut overlay = AdaptOverlay {
            runs: Vec::with_capacity(matching_slice.len()),
            keys: Vec::with_capacity(4),
            fresh_fps: Vec::with_capacity(4),
            tombstoned_disk_fps: Vec::with_capacity(4),
        };

        for &(p_q, low_m, high_m) in matching_slice {
            self.plan_subquery_adaptation(
                p_q,
                low_m,
                high_m,
                reverse_map,
                &mut overlay,
                &mut report,
            )?;
        }

        if overlay.tombstoned_disk_fps.is_empty() && overlay.keys.is_empty() {
            return Ok(report);
        }

        self.commit_adapt_overlay(overlay, reverse_map)?;
        self.log_adapt_done(range, &report, start, "overlay planner");
        Ok(report)
    }

    fn collect_matching_subqueries(
        &self,
        range: QueryRange,
        report: &mut AdaptationReport,
    ) -> MatchedSubqueries {
        let mut stack = [(0u64, 0u64, 0u64); 8];
        let mut stack_len = 0usize;
        let mut overflow = Vec::new();

        for (p_q, low_m, high_m) in range.partition_subqueries(self.config.m) {
            report.subqueries_checked += 1;
            if self.query_partition_hot(p_q, low_m, high_m) {
                if stack_len < stack.len() {
                    stack[stack_len] = (p_q, low_m, high_m);
                    stack_len += 1;
                } else {
                    if overflow.is_empty() {
                        overflow.extend_from_slice(&stack);
                    }
                    overflow.push((p_q, low_m, high_m));
                }
            }
        }
        MatchedSubqueries {
            stack,
            stack_len,
            overflow,
        }
    }

    fn commit_adapt_overlay(
        &mut self,
        overlay: AdaptOverlay,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<()> {
        let mut encoded_updates: Vec<(usize, Vec<(u64, KeepsakeSlotKind)>)> =
            Vec::with_capacity(overlay.runs.len());
        let mut total_slot_delta: isize = 0;
        let mut total_key_delta: isize = 0;

        for (home, old_slots, old_keys, new_boxes) in &overlay.runs {
            let new_encoded = self.encode_run_boxes(new_boxes);
            let new_keys: usize = new_boxes.iter().map(|b| b.payload.key_count()).sum();
            total_slot_delta += (new_encoded.len() as isize) - (*old_slots as isize);
            total_key_delta += (new_keys as isize) - (*old_keys as isize);
            encoded_updates.push((*home, new_encoded));
        }

        self.check_can_add_slots(total_slot_delta)?;

        let total_rm: usize = overlay.keys.iter().map(|(_, ks)| ks.len()).sum();
        let mut new_rm_entries = Vec::with_capacity(total_rm);
        for (fp, keys) in overlay.keys {
            for k in keys {
                new_rm_entries.push((fp.clone(), k));
            }
        }

        if !overlay.tombstoned_disk_fps.is_empty() {
            for old_fp in &overlay.tombstoned_disk_fps[1..] {
                reverse_map.tombstone(old_fp)?;
            }
            reverse_map.replace_fingerprint(&overlay.tombstoned_disk_fps[0], &new_rm_entries)?;
        } else if !new_rm_entries.is_empty() {
            reverse_map.append_batch(&new_rm_entries)?;
        }

        for (home, new_encoded) in encoded_updates {
            let (cur_start, cur_len) = self
                .table
                .locate_run(home)
                .map(|(s, e)| (s, e - s + 1))
                .unwrap_or((0, 0));
            self.table
                .replace_region(home, cur_start, cur_len, &new_encoded);
        }
        self.occupied_slots = ((self.occupied_slots as isize) + total_slot_delta) as usize;
        self.num_keys = ((self.num_keys as isize) + total_key_delta) as usize;
        Ok(())
    }

    fn log_adapt_done(
        &self,
        range: QueryRange,
        report: &AdaptationReport,
        start: Instant,
        mode: &str,
    ) {
        tracing::debug!(
            range_start = range.start(),
            range_end = range.end(),
            subqueries_checked = report.subqueries_checked,
            boxes_split = report.boxes_split,
            keys_rehashed = report.keys_rehashed,
            extension_slots_added = report.extension_slots_added,
            rejuvenated_boxes = report.rejuvenated_boxes,
            load_factor = self.load_factor(),
            latency_ns = start.elapsed().as_nanos() as u64,
            mode,
            "completed AerisFilter adaptation"
        );
    }

    fn plan_subquery_adaptation(
        &self,
        p_q: PartitionId,
        low_m: Memento,
        high_m: Memento,
        reverse_map: &dyn ReverseMapStore,
        overlay: &mut AdaptOverlay,
        report: &mut AdaptationReport,
    ) -> AerisResult<()> {
        let (home_u64, _) = self.quotient_and_remainder_hot(p_q);
        let home = home_u64 as usize;
        let Some(run_idx) = self.load_or_find_overlay_run(home, &mut overlay.runs) else {
            return Ok(());
        };

        let Some((mut remaining_boxes, matched_boxes)) =
            self.partition_matched_boxes(&overlay.runs[run_idx].3, p_q, low_m, high_m)
        else {
            return Ok(());
        };

        let collected =
            self.collect_matched_box_keys(home, p_q, &matched_boxes, reverse_map, overlay, report)?;

        // Step 1: Resolve colliding partitions `p' != p_q` FIRST.
        let mut created_boxes = self.resolve_colliding_partitions(
            home,
            p_q,
            collected.colliding_parts,
            &mut remaining_boxes,
            &mut overlay.keys,
        );

        // Step 2: Extend `p_q` LAST once every colliding box's final length is known.
        if !collected.query_part_keys.is_empty() {
            let bx_q = self.extend_query_partition_last(
                home,
                p_q,
                collected.query_part_min_bits,
                &collected.query_part_keys,
                &remaining_boxes,
                &created_boxes,
            );
            let fp_q = bx_q.canonical_fingerprint(home, self.config.q, self.config.r);
            created_boxes.push((fp_q, bx_q, collected.query_part_keys));
        }

        let mut new_created_slots = 0usize;
        for (fp, bx, keys_vec) in created_boxes {
            new_created_slots += bx.encoded_slot_len(&self.geom);
            if !overlay.fresh_fps.iter().any(|f| f == &fp) {
                overlay.fresh_fps.push(fp.clone());
            }
            overlay_keys_extend(&mut overlay.keys, fp, keys_vec);
            remaining_boxes.push(bx);
        }

        remaining_boxes.sort_by_key(KeepsakeBox::sort_key);
        overlay.runs[run_idx].3 = remaining_boxes;
        report.boxes_split += matched_boxes.len();
        report.extension_slots_added +=
            new_created_slots.saturating_sub(collected.old_matched_slots);
        Ok(())
    }

    fn load_or_find_overlay_run(
        &self,
        home: usize,
        runs: &mut Vec<(usize, usize, usize, Vec<KeepsakeBox>)>,
    ) -> Option<usize> {
        if let Some(pos) = runs.iter().position(|e| e.0 == home) {
            return Some(pos);
        }
        if !self.table.is_occupied(home) {
            return None;
        }
        let decoded_full = self.decode_run(home);
        let old_slots: usize = decoded_full.iter().map(|&(_, len, _)| len).sum();
        let old_keys: usize = decoded_full
            .iter()
            .map(|(_, _, bx)| bx.payload.key_count())
            .sum();
        let boxes = decoded_full.into_iter().map(|(_, _, bx)| bx).collect();
        runs.push((home, old_slots, old_keys, boxes));
        Some(runs.len() - 1)
    }

    fn partition_matched_boxes(
        &self,
        current_boxes: &[KeepsakeBox],
        p_q: PartitionId,
        low_m: Memento,
        high_m: Memento,
    ) -> Option<(Vec<KeepsakeBox>, Vec<KeepsakeBox>)> {
        let q = self.config.q;
        let r = self.config.r;
        let mut matched_indices = Vec::with_capacity(2);
        let mut has_memento_hit = false;

        for (idx, bx) in current_boxes.iter().enumerate() {
            if bx.matches_partition(&self.hasher, p_q, q, r) {
                matched_indices.push(idx);
                if bx.payload.contains_in_range(low_m, high_m) {
                    has_memento_hit = true;
                }
            }
        }
        if !has_memento_hit {
            return None;
        }

        let mut remaining =
            Vec::with_capacity(current_boxes.len().saturating_sub(matched_indices.len()) + 2);
        let mut matched = Vec::with_capacity(matched_indices.len());
        let mut next_ptr = 0usize;
        for (idx, bx) in current_boxes.iter().enumerate() {
            if next_ptr < matched_indices.len() && matched_indices[next_ptr] == idx {
                matched.push(bx.clone());
                next_ptr += 1;
            } else {
                remaining.push(bx.clone());
            }
        }
        Some((remaining, matched))
    }

    fn collect_matched_box_keys(
        &self,
        home: usize,
        p_q: PartitionId,
        matched_boxes: &[KeepsakeBox],
        reverse_map: &dyn ReverseMapStore,
        overlay: &mut AdaptOverlay,
        report: &mut AdaptationReport,
    ) -> AerisResult<CollectedPartitionKeys> {
        let fresh_base = (self.config.q as u16) + (self.config.r as u16);
        let mut colliding_parts: Vec<(PartitionId, u16, Vec<Key>)> = Vec::with_capacity(2);
        let mut query_part_min_bits: u16 = 0;
        let mut query_part_keys: Vec<Key> = Vec::new();
        let mut old_matched_slots = 0usize;

        for bx in matched_boxes {
            old_matched_slots += bx.encoded_slot_len(&self.geom);
            if bx.age > 0 {
                report.rejuvenated_boxes += 1;
            }
            let old_fp = bx.canonical_fingerprint(home, self.config.q, self.config.r);
            let old_bits = old_fp.bit_len();
            let keys_for_box = self.fetch_overlay_box_keys(home, old_fp, reverse_map, overlay)?;

            report.keys_rehashed += keys_for_box.len();
            for key in keys_for_box {
                let p = self.config.partition_of(key);
                if p == p_q {
                    query_part_min_bits = query_part_min_bits.max(old_bits);
                    query_part_keys.push(key);
                } else if let Some(entry) = colliding_parts.iter_mut().find(|e| e.0 == p) {
                    entry.1 = entry.1.max(old_bits);
                    entry.2.push(key);
                } else {
                    colliding_parts.push((p, fresh_base.max(old_bits), vec![key]));
                }
            }
        }

        for (_, _, ks) in &mut colliding_parts {
            sort_and_dedup(ks);
        }
        sort_and_dedup(&mut query_part_keys);

        Ok(CollectedPartitionKeys {
            colliding_parts,
            query_part_min_bits,
            query_part_keys,
            old_matched_slots,
        })
    }

    fn fetch_overlay_box_keys(
        &self,
        home: usize,
        old_fp: Fingerprint,
        reverse_map: &dyn ReverseMapStore,
        overlay: &mut AdaptOverlay,
    ) -> AerisResult<Vec<Key>> {
        if let Some(fpos) = overlay.fresh_fps.iter().position(|f| f == &old_fp) {
            overlay.fresh_fps.swap_remove(fpos);
            return Ok(overlay
                .keys
                .iter()
                .position(|(f, _)| f == &old_fp)
                .map(|kpos| overlay.keys.swap_remove(kpos).1)
                .unwrap_or_default());
        }

        let old_bits = old_fp.bit_len();
        let mut combined = reverse_map.get_keys(&old_fp)?;
        if let Some(kpos) = overlay.keys.iter().position(|(f, _)| f == &old_fp) {
            combined.extend(overlay.keys.swap_remove(kpos).1);
        }
        sort_and_dedup(&mut combined);
        if combined.is_empty() {
            return Err(AerisError::StorageError(format!(
                "reverse map returned 0 keys for active keepsake box fingerprint ({old_bits} bits) at home {home}"
            )));
        }
        if !overlay.tombstoned_disk_fps.iter().any(|f| f == &old_fp) {
            overlay.tombstoned_disk_fps.push(old_fp);
        }
        Ok(combined)
    }
}

#[inline]
pub(super) fn sort_and_dedup(keys: &mut Vec<Key>) {
    if keys.len() > 1 {
        keys.sort_unstable();
        keys.dedup();
    }
}

/// Helper extending a flat `(Fingerprint, Vec<Key>)` overlay list while maintaining
/// deduplicated key sets without `BTreeMap`/`BTreeSet` heap overhead.
#[inline]
pub(super) fn overlay_keys_extend(
    overlay_keys: &mut Vec<(Fingerprint, Vec<Key>)>,
    fp: Fingerprint,
    keys: Vec<Key>,
) {
    if let Some((_, existing)) = overlay_keys.iter_mut().find(|(f, _)| f == &fp) {
        existing.extend(keys);
        sort_and_dedup(existing);
    } else {
        overlay_keys.push((fp, keys));
    }
}
