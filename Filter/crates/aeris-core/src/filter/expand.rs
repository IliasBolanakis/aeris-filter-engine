//! 2x InfiniFilter expansion with in-memory bit-splitting and selective rejuvenation.

use super::AerisFilter;
use crate::chunk::QuotientTable;
use crate::keepsake::KeepsakeBox;
use aeris_types::{AerisResult, ExpansionReport, Fingerprint, Key, RangeFilter, ReverseMapStore};
use std::collections::BTreeMap;
use std::time::Instant;

impl AerisFilter {
    pub(super) fn expand_internal(
        &mut self,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<ExpansionReport> {
        let start = Instant::now();
        let old_q = self.config.q;
        let new_config = self.config.expanded()?;
        let new_q = new_config.q;

        let mut new_table = QuotientTable::new(new_config.num_slots(), self.geom.slot_bits);
        let (boxes_transferred, new_occupied_slots, new_num_keys, rejuvenate_fps) =
            self.split_and_append_expanded_runs(&mut new_table);

        let boxes_rejuvenated = rejuvenate_fps.len();
        let new_max_occupied_slots =
            ((new_config.num_slots() as f64) * new_config.max_load_factor).floor() as usize;
        let mut new_filter = Self {
            config: new_config,
            hasher: self.hasher,
            geom: self.geom,
            table: new_table,
            num_keys: new_num_keys,
            occupied_slots: new_occupied_slots,
            max_occupied_slots: new_max_occupied_slots,
            expansion_cycles: self.expansion_cycles + 1,
        };

        let keys_rejuvenated = if rejuvenate_fps.is_empty() {
            0
        } else {
            new_filter.rejuvenate_aged_boxes(&rejuvenate_fps, reverse_map)?
        };

        *self = new_filter;
        let duration_ns = start.elapsed().as_nanos() as u64;

        tracing::info!(
            old_q,
            new_q,
            boxes_transferred,
            boxes_rejuvenated,
            keys_rejuvenated,
            load_factor = self.load_factor(),
            latency_ns = duration_ns,
            expansion_cycles = self.expansion_cycles,
            "completed 2x InfiniFilter expansion"
        );

        Ok(ExpansionReport {
            old_q,
            new_q,
            boxes_transferred,
            boxes_rejuvenated,
            keys_rejuvenated,
            duration_ns,
        })
    }

    fn split_and_append_expanded_runs(
        &self,
        new_table: &mut QuotientTable,
    ) -> (usize, usize, usize, Vec<Fingerprint>) {
        let old_q = self.config.q;
        let r = self.config.r;
        let mut write_cursor = 0usize;
        let mut boxes_transferred = 0usize;
        let mut new_occupied_slots = 0usize;
        let mut new_num_keys = 0usize;
        let mut rejuvenate_fps = Vec::new();

        for old_home in self.table.occupied_homes() {
            let boxes = self.decode_run(old_home);
            let mut split_0 = Vec::new();
            let mut split_1 = Vec::new();

            for (_, _, bx) in boxes {
                if let Some((new_home, split_box)) = bx.split_for_expansion(old_home, r) {
                    if (new_home & 1) == 0 {
                        split_0.push(split_box);
                    } else {
                        split_1.push(split_box);
                    }
                    boxes_transferred += 1;
                } else {
                    rejuvenate_fps.push(bx.canonical_fingerprint(old_home, old_q, r));
                }
            }

            self.append_split_half(
                old_home << 1,
                split_0,
                new_table,
                &mut write_cursor,
                &mut new_occupied_slots,
                &mut new_num_keys,
            );
            self.append_split_half(
                (old_home << 1) | 1,
                split_1,
                new_table,
                &mut write_cursor,
                &mut new_occupied_slots,
                &mut new_num_keys,
            );
        }
        (
            boxes_transferred,
            new_occupied_slots,
            new_num_keys,
            rejuvenate_fps,
        )
    }

    fn append_split_half(
        &self,
        target_home: usize,
        mut boxes: Vec<KeepsakeBox>,
        new_table: &mut QuotientTable,
        write_cursor: &mut usize,
        new_occupied_slots: &mut usize,
        new_num_keys: &mut usize,
    ) {
        if boxes.is_empty() {
            return;
        }
        boxes.sort_by_key(KeepsakeBox::sort_key);
        let run_keys: usize = boxes.iter().map(|b| b.payload.key_count()).sum();
        let run_slots = self.encode_run_boxes(&boxes);
        *new_occupied_slots += run_slots.len();
        *new_num_keys += run_keys;
        new_table.bulk_append_run(target_home, &run_slots, write_cursor);
    }

    fn rejuvenate_aged_boxes(
        &mut self,
        rejuvenate_fps: &[Fingerprint],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<usize> {
        // Read all keys from Reverse Map BEFORE modifying Reverse Map so storage errors
        // leave `self` completely intact.
        let mut keys_rejuvenated = 0usize;
        let mut key_min_bits: BTreeMap<Key, u16> = BTreeMap::new();
        for old_fp in rejuvenate_fps {
            let keys = reverse_map.get_keys(old_fp)?;
            keys_rejuvenated += keys.len();
            let floor_bits = old_fp.bit_len();
            for k in keys {
                key_min_bits
                    .entry(k)
                    .and_modify(|cur| *cur = (*cur).max(floor_bits))
                    .or_insert(floor_bits);
            }
        }

        // Insert descending by `min_fp_bits` so longer monotonic floors are established
        // before shorter unextended boxes at the same `(new_home, new_rem)`.
        let mut ordered_rejuv: Vec<(u16, Key)> = key_min_bits
            .into_iter()
            .map(|(k, bits)| (bits, k))
            .collect();
        ordered_rejuv.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let mut new_rm_entries = Vec::with_capacity(ordered_rejuv.len());
        for (min_bits, key) in ordered_rejuv {
            let new_fp = self.insert_key_internal(key, min_bits, false)?;
            new_rm_entries.push((new_fp, key));
        }

        // Tombstone all old `age == r` fingerprints first, then insert replacement entries
        // atomically so an unchanged `old_fp == new_fp` is never wiped.
        for old_fp in &rejuvenate_fps[1..] {
            reverse_map.tombstone(old_fp)?;
        }
        reverse_map.replace_fingerprint(&rejuvenate_fps[0], &new_rm_entries)?;
        Ok(keys_rejuvenated)
    }
}
