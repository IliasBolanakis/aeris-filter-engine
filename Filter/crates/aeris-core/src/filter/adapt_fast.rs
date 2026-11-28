//! In-place fast-path adaptation for single fresh colliding boxes.

use super::AerisFilter;
use crate::keepsake::{box_matches_partition_hot, KeepsakeBox};
use aeris_types::{
    AdaptationReport, AerisError, AerisResult, Fingerprint, KeepsakeSlotKind, Key, PartitionId,
    ReverseMapStore,
};

impl AerisFilter {
    /// Fast in-place adaptation path for the >95% common case: a single matching
    /// subquery `(p_q, low_m, high_m)` hitting a single fresh (`age == 0`) box
    /// whose keys all belong to one colliding partition `p_other != p_q`, with no
    /// peer remainder collisions or extended aged boxes in the same quotient run.
    ///
    /// Returns `Ok(Some(()))` on fast-path completion, `Ok(None)` when a rarer
    /// multi-partition / aged / peer-collision case requires the general overlay
    /// planner, or `Err(...)` on `FilterFull` / `StorageError` (leaving both RAM
    /// table and `reverse_map` untouched).
    pub(super) fn try_adapt_single_box_fast(
        &mut self,
        p_q: PartitionId,
        home: usize,
        reverse_map: &dyn ReverseMapStore,
        report: &mut AdaptationReport,
    ) -> AerisResult<Option<()>> {
        let Some((run_start, run_end)) = self.table.locate_run(home) else {
            return Ok(Some(()));
        };
        let w0_q = self.hasher.word(p_q, 0);
        let Some((box_start, box_len)) =
            self.scan_run_for_single_fresh_match(p_q, w0_q, run_start, run_end)
        else {
            return Ok(None);
        };
        if box_len == 0 {
            return Ok(Some(()));
        }

        let (bx, decoded_len) = KeepsakeBox::decode_from_table(&self.table, &self.geom, box_start);
        debug_assert_eq!(decoded_len, box_len);
        debug_assert_eq!(bx.age, 0);

        let old_fp = bx.canonical_fingerprint(home, self.config.q, self.config.r);
        let Some((p_other, keys)) =
            self.fetch_single_colliding_partition(home, &old_fp, p_q, reverse_map)?
        else {
            return Ok(None);
        };

        let ext_level = self.compute_diverging_ext_level(old_fp.bit_len(), p_other, p_q, w0_q);
        self.commit_single_box_adaptation(
            home,
            run_start,
            run_end,
            box_start,
            box_len,
            &bx,
            &old_fp,
            p_other,
            ext_level,
            &keys,
            reverse_map,
            report,
        )?;
        Ok(Some(()))
    }

    /// Scans `home`'s run with zero allocations.
    /// Returns `None` if fallback to the general planner is needed, `Some((0, 0))`
    /// if no box matched, or `Some((box_start, box_len))` for the sole match.
    fn scan_run_for_single_fresh_match(
        &self,
        p_q: PartitionId,
        w0_q: u64,
        run_start: usize,
        run_end: usize,
    ) -> Option<(usize, usize)> {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        let rem_q = if fresh_base <= 64 {
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (w0_q >> rem_shift) & rem_mask
        } else {
            self.hasher.quotient_and_remainder(p_q, q, r).1
        };
        let fresh_sentinel = 1u64 << r;
        let expected_fresh_high = fresh_sentinel | rem_q;

        let mut matched_box: Option<(usize, usize)> = None;
        let mut cursor = run_start;
        while cursor <= run_end {
            let (is_match, next_slot, high, kind0) = box_matches_partition_hot(
                &self.table,
                &self.geom,
                &self.hasher,
                p_q,
                w0_q,
                q,
                rem_q,
                cursor,
            );
            if is_match {
                if matched_box.is_some() || high < fresh_sentinel {
                    return None;
                }
                matched_box = Some((cursor, next_slot - cursor));
            } else if high == expected_fresh_high
                || (high < fresh_sentinel && kind0 == KeepsakeSlotKind::Extension)
            {
                return None;
            }
            cursor = next_slot;
        }
        Some(matched_box.unwrap_or((0, 0)))
    }

    fn fetch_single_colliding_partition(
        &self,
        home: usize,
        old_fp: &Fingerprint,
        p_q: PartitionId,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Option<(PartitionId, Vec<Key>)>> {
        let old_bits = old_fp.bit_len();
        let mut keys = reverse_map.get_keys(old_fp)?;
        if keys.is_empty() {
            return Err(AerisError::StorageError(format!(
                "reverse map returned 0 keys for active keepsake box fingerprint ({old_bits} bits) at home {home}"
            )));
        }
        if keys.len() > 1 {
            keys.sort_unstable();
            keys.dedup();
        }
        let p_other = self.config.partition_of(keys[0]);
        if p_other == p_q
            || keys[1..]
                .iter()
                .any(|&k| self.config.partition_of(k) != p_other)
        {
            return Ok(None);
        }
        Ok(Some((p_other, keys)))
    }

    fn compute_diverging_ext_level(
        &self,
        old_bits: u16,
        p_other: PartitionId,
        p_q: PartitionId,
        w0_q: u64,
    ) -> usize {
        let fresh_base = (self.config.q as u16) + (self.config.r as u16);
        let min_req = fresh_base.max(old_bits);
        let mut ext_level = self
            .geom
            .min_ext_level_for_bits(min_req.saturating_sub(fresh_base));
        let diff_w0 = self.hasher.word(p_other, 0) ^ w0_q;
        loop {
            let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
            let diverges = if bit_len <= 64 {
                (diff_w0 >> (64 - bit_len)) != 0
            } else {
                Fingerprint::from_hasher(&self.hasher, p_other, bit_len)
                    != Fingerprint::from_hasher(&self.hasher, p_q, bit_len)
            };
            if diverges {
                return ext_level;
            }
            ext_level += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_single_box_adaptation(
        &mut self,
        home: usize,
        run_start: usize,
        run_end: usize,
        box_start: usize,
        box_len: usize,
        old_box: &KeepsakeBox,
        old_fp: &Fingerprint,
        p_other: PartitionId,
        ext_level: usize,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
        report: &mut AdaptationReport,
    ) -> AerisResult<()> {
        let q = self.config.q;
        let r = self.config.r;
        let payload = self.payload_from_key_slice(keys);
        let new_box =
            KeepsakeBox::new_fresh(&self.hasher, p_other, q, &self.geom, ext_level, payload);
        let new_len = new_box.encoded_slot_len(&self.geom);
        let slot_delta = (new_len as isize) - (box_len as isize);
        let key_delta =
            (new_box.payload.key_count() as isize) - (old_box.payload.key_count() as isize);

        // Enforce `max_load_factor` atomically BEFORE any Reverse Map or RAM mutation.
        self.check_can_add_slots(slot_delta)?;

        let new_fp = new_box.canonical_fingerprint(home, q, r);
        if keys.len() == 1 {
            reverse_map.replace_fingerprint(old_fp, &[(new_fp, keys[0])])?;
        } else {
            let new_rm_entries: Vec<(Fingerprint, Key)> =
                keys.iter().map(|&k| (new_fp.clone(), k)).collect();
            reverse_map.replace_fingerprint(old_fp, &new_rm_entries)?;
        }

        let is_run_end = box_start + box_len - 1 == run_end;
        let encoded = new_box.encode(&self.geom, is_run_end);
        self.table
            .replace_region_known_run(home, run_start, run_end, box_start, box_len, &encoded);

        self.occupied_slots = ((self.occupied_slots as isize) + slot_delta) as usize;
        self.num_keys = ((self.num_keys as isize) + key_delta) as usize;
        report.boxes_split += 1;
        report.keys_rehashed += keys.len();
        report.extension_slots_added += new_len.saturating_sub(box_len);
        Ok(())
    }
}
