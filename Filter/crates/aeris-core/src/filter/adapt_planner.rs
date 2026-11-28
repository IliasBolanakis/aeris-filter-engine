//! Prefix-free extension planning helpers for colliding and query partitions.

use super::adapt::{overlay_keys_extend, sort_and_dedup};
use super::AerisFilter;
use crate::keepsake::KeepsakeBox;
use aeris_types::{Fingerprint, Key, PartitionId};

impl AerisFilter {
    pub(super) fn resolve_colliding_partitions(
        &self,
        home: usize,
        p_q: PartitionId,
        colliding_parts: Vec<(PartitionId, u16, Vec<Key>)>,
        remaining_boxes: &mut [KeepsakeBox],
        overlay_keys: &mut Vec<(Fingerprint, Vec<Key>)>,
    ) -> Vec<(Fingerprint, KeepsakeBox, Vec<Key>)> {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        let mut new_colliding_specs = Vec::with_capacity(colliding_parts.len());

        for (p_other, min_bits, keys_vec) in colliding_parts {
            let min_req = fresh_base.max(min_bits);
            if let Some(u_idx) = remaining_boxes.iter().position(|u| {
                u.total_fp_bits(q, r) >= min_req && u.matches_partition(&self.hasher, p_other, q, r)
            }) {
                let target_box = &mut remaining_boxes[u_idx];
                for &k in &keys_vec {
                    target_box.payload.insert_memento(self.config.memento_of(k));
                }
                let u_fp = target_box.canonical_fingerprint(home, q, r);
                overlay_keys_extend(overlay_keys, u_fp, keys_vec);
                continue;
            }

            let mut ext_level = self.geom.min_ext_level_for_bits(min_req - fresh_base);
            loop {
                let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
                let fp_other = Fingerprint::from_hasher(&self.hasher, p_other, bit_len);
                let fp_q = Fingerprint::from_hasher(&self.hasher, p_q, bit_len);
                if fp_other != fp_q
                    && !self.conflicts_with_remaining(home, &fp_other, remaining_boxes)
                {
                    break;
                }
                ext_level += 1;
            }
            new_colliding_specs.push((p_other, ext_level, keys_vec));
        }

        self.normalize_colliding_specs(home, remaining_boxes, &mut new_colliding_specs);
        self.build_grouped_colliding_boxes(home, new_colliding_specs)
    }

    fn build_grouped_colliding_boxes(
        &self,
        home: usize,
        specs: Vec<(PartitionId, usize, Vec<Key>)>,
    ) -> Vec<(Fingerprint, KeepsakeBox, Vec<Key>)> {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        let mut grouped: Vec<(Fingerprint, PartitionId, usize, Vec<Key>)> =
            Vec::with_capacity(specs.len());

        for (p_other, ext_level, keys_vec) in specs {
            let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
            let fp = Fingerprint::from_hasher(&self.hasher, p_other, bit_len);
            if let Some((_, _, _, existing_keys)) = grouped.iter_mut().find(|(f, _, _, _)| f == &fp)
            {
                existing_keys.extend(keys_vec);
                sort_and_dedup(existing_keys);
            } else {
                grouped.push((fp, p_other, ext_level, keys_vec));
            }
        }

        grouped
            .into_iter()
            .map(|(fp, p_rep, ext_level, keys_vec)| {
                let payload = self.payload_from_key_slice(&keys_vec);
                let bx =
                    KeepsakeBox::new_fresh(&self.hasher, p_rep, q, &self.geom, ext_level, payload);
                debug_assert_eq!(bx.canonical_fingerprint(home, q, r), fp);
                (fp, bx, keys_vec)
            })
            .collect()
    }

    pub(super) fn extend_query_partition_last(
        &self,
        home: usize,
        p_q: PartitionId,
        query_part_min_bits: u16,
        query_part_keys: &[Key],
        remaining_boxes: &[KeepsakeBox],
        created_boxes: &[(Fingerprint, KeepsakeBox, Vec<Key>)],
    ) -> KeepsakeBox {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        let min_req_q = fresh_base.max(query_part_min_bits);
        let mut ext_level_q = self
            .geom
            .min_ext_level_for_bits(min_req_q.saturating_sub(fresh_base));

        loop {
            let bit_len_q = fresh_base + self.geom.ext_bits_for_level(ext_level_q);
            let fp_q = Fingerprint::from_hasher(&self.hasher, p_q, bit_len_q);
            let conflicts_colliding = created_boxes
                .iter()
                .any(|(c_fp, _, _)| fp_q.is_prefix_of(c_fp) || c_fp.is_prefix_of(&fp_q));
            if !conflicts_colliding && !self.conflicts_with_remaining(home, &fp_q, remaining_boxes)
            {
                break;
            }
            ext_level_q += 1;
        }

        let payload_q = self.payload_from_key_slice(query_part_keys);
        KeepsakeBox::new_fresh(&self.hasher, p_q, q, &self.geom, ext_level_q, payload_q)
    }

    fn conflicts_with_remaining(
        &self,
        home: usize,
        cand_fp: &Fingerprint,
        remaining_boxes: &[KeepsakeBox],
    ) -> bool {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);
        remaining_boxes.iter().any(|u| {
            if u.total_fp_bits(q, r) < fresh_base {
                return false;
            }
            let u_fp = u.canonical_fingerprint(home, q, r);
            cand_fp.is_prefix_of(&u_fp) || u_fp.is_prefix_of(cand_fp)
        })
    }

    /// Ensures no two newly generated colliding boxes are strict prefixes of each other.
    fn normalize_colliding_specs(
        &self,
        home: usize,
        remaining_boxes: &[KeepsakeBox],
        specs: &mut [(PartitionId, usize, Vec<Key>)],
    ) {
        let fresh_base = (self.config.q as u16) + (self.config.r as u16);
        loop {
            let mut changed = false;
            for i in 0..specs.len() {
                let len_i = fresh_base + self.geom.ext_bits_for_level(specs[i].1);
                let fp_i = Fingerprint::from_hasher(&self.hasher, specs[i].0, len_i);
                for j in 0..specs.len() {
                    if i == j {
                        continue;
                    }
                    let len_j = fresh_base + self.geom.ext_bits_for_level(specs[j].1);
                    if len_i < len_j {
                        let fp_j = Fingerprint::from_hasher(&self.hasher, specs[j].0, len_j);
                        if fp_i.is_prefix_of(&fp_j) {
                            specs[i].1 = self.bump_ext_level_past_remaining(
                                home,
                                specs[i].0,
                                specs[j].1,
                                remaining_boxes,
                            );
                            changed = true;
                            break;
                        }
                    }
                }
                if changed {
                    break;
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn bump_ext_level_past_remaining(
        &self,
        home: usize,
        partition: PartitionId,
        mut ext_level: usize,
        remaining_boxes: &[KeepsakeBox],
    ) -> usize {
        let fresh_base = (self.config.q as u16) + (self.config.r as u16);
        loop {
            let new_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
            let new_fp = Fingerprint::from_hasher(&self.hasher, partition, new_len);
            if !self.conflicts_with_remaining(home, &new_fp, remaining_boxes) {
                return ext_level;
            }
            ext_level += 1;
        }
    }
}
