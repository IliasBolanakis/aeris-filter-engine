//! Zero-allocation point and range query paths for [`AerisFilter`].

use super::AerisFilter;
use crate::keepsake::box_matches_query_hot_with_age_hint;
use aeris_types::{Fingerprint, Key, Memento, PartitionId, QueryRange};
use std::collections::BTreeSet;

impl AerisFilter {
    /// Evaluates a single partition subquery `(partition, low_m, high_m)` with
    /// zero heap allocation.
    #[inline]
    pub(super) fn query_partition_hot(
        &self,
        partition: PartitionId,
        low_m: Memento,
        high_m: Memento,
    ) -> bool {
        let q = self.config.q;
        let r = self.config.r;
        let w0 = self.hasher.word(partition, 0);

        let (home_u64, rem_full) = if (q as u16) + (r as u16) <= 64 {
            let h = w0 >> (64 - q);
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (h, (w0 >> rem_shift) & rem_mask)
        } else {
            self.hasher.quotient_and_remainder(partition, q, r)
        };

        let home = home_u64 as usize;
        let Some((run_start, run_end)) = self.table.locate_run(home) else {
            return false;
        };

        let has_aged = self.expansion_cycles > 0;
        let mut cursor = run_start;
        while cursor <= run_end {
            let (matched, next_slot) = box_matches_query_hot_with_age_hint(
                &self.table,
                &self.geom,
                &self.hasher,
                partition,
                w0,
                q,
                rem_full,
                low_m,
                high_m,
                cursor,
                has_aged,
            );
            if matched {
                return true;
            }
            cursor = next_slot;
        }
        false
    }

    #[inline]
    pub(super) fn query_point_hot(&self, key: Key) -> bool {
        if self.num_keys == 0 {
            return false;
        }
        let p = self.config.partition_of(key);
        let mem = self.config.memento_of(key);
        self.query_partition_hot(p, mem, mem)
    }

    #[inline]
    pub(super) fn query_range_hot(&self, range: QueryRange) -> bool {
        if self.num_keys == 0 {
            return false;
        }
        if range.start() == 0 && range.end() == u64::MAX {
            return true;
        }
        let m = self.config.m.min(63);
        let mask = if m == 0 { 0u64 } else { (1u64 << m) - 1 };
        let start_part = range.start() >> m;
        let end_part = range.end() >> m;
        let start_m = range.start() & mask;
        let end_m = range.end() & mask;
        if start_part == end_part {
            return self.query_partition_hot(start_part, start_m, end_m);
        }
        if start_part + 1 == end_part {
            return self.query_partition_hot(start_part, start_m, mask)
                || self.query_partition_hot(end_part, 0, end_m);
        }
        for (partition, low_m, high_m) in range.partition_subqueries(self.config.m) {
            if self.query_partition_hot(partition, low_m, high_m) {
                return true;
            }
        }
        false
    }

    pub(super) fn locate_matching_fingerprints_internal(
        &self,
        range: QueryRange,
    ) -> Vec<Fingerprint> {
        if self.num_keys == 0 {
            return Vec::new();
        }
        let q = self.config.q;
        let r = self.config.r;
        let mut matched_fps = BTreeSet::new();

        for (partition, low_m, high_m) in range.partition_subqueries(self.config.m) {
            let (home_u64, _) = self.hasher.quotient_and_remainder(partition, q, r);
            let home = home_u64 as usize;
            if !self.table.is_occupied(home) {
                continue;
            }
            for (_, _, bx) in self.decode_run(home) {
                if bx.matches_partition(&self.hasher, partition, q, r)
                    && bx.payload.contains_in_range(low_m, high_m)
                {
                    matched_fps.insert(bx.canonical_fingerprint(home, q, r));
                }
            }
        }

        matched_fps.into_iter().collect()
    }
}
