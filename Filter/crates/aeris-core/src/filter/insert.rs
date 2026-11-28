//! Key insertion and deletion paths for [`AerisFilter`].

use super::AerisFilter;
use crate::keepsake::KeepsakeBox;
use aeris_types::{AerisResult, Fingerprint, KeepsakeSlotKind, Key, Memento, PartitionId};

struct InsertKeyCtx {
    p: PartitionId,
    mem: Memento,
    q: u8,
    r: u8,
    m: u8,
    single_word: bool,
    w0: u64,
    home_u64: u64,
    rem_full: u64,
}

impl AerisFilter {
    /// Inserts `key` with a minimum canonical fingerprint width `min_fp_bits`
    /// (`0` for normal inserts; `old_fp.bit_len()` during selective rejuvenation).
    pub(super) fn insert_key_internal(
        &mut self,
        key: Key,
        min_fp_bits: u16,
        check_load_limit: bool,
    ) -> AerisResult<Fingerprint> {
        let ctx = self.prepare_insert_ctx(key);
        if min_fp_bits == 0 {
            if let Some(fp) = self.try_insert_fresh_fast_path(&ctx, check_load_limit)? {
                return Ok(fp);
            }
        }
        self.insert_key_slow_path(&ctx, min_fp_bits, check_load_limit)
    }

    pub(super) fn delete_internal(&mut self, key: Key) -> AerisResult<Option<Fingerprint>> {
        if self.num_keys == 0 {
            return Ok(None);
        }
        let q = self.config.q;
        let r = self.config.r;
        let p = self.config.partition_of(key);
        let mem = self.config.memento_of(key);
        let (home_u64, _) = self.hasher.quotient_and_remainder(p, q, r);
        let home = home_u64 as usize;

        if !self.table.is_occupied(home) {
            return Ok(None);
        }

        let boxes = self.decode_run(home);
        let Some(idx) = self.find_best_delete_candidate(&boxes, p, mem) else {
            return Ok(None);
        };
        Ok(Some(self.apply_delete_to_run(home, &boxes, idx, mem)))
    }

    #[inline]
    fn prepare_insert_ctx(&self, key: Key) -> InsertKeyCtx {
        let q = self.config.q;
        let r = self.config.r;
        let m = self.config.m;
        let p = self.config.partition_of(key);
        let mem = self.config.memento_of(key);
        let single_word = (q as u16) + (r as u16) <= 64;
        let (w0, home_u64, rem_full) = if single_word {
            let w0 = self.hasher.word(p, 0);
            let h = w0 >> (64 - q);
            let rem_shift = 64 - (q + r);
            let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
            (w0, h, (w0 >> rem_shift) & rem_mask)
        } else {
            let (h, rem) = self.hasher.quotient_and_remainder(p, q, r);
            (0, h, rem)
        };
        InsertKeyCtx {
            p,
            mem,
            q,
            r,
            m,
            single_word,
            w0,
            home_u64,
            rem_full,
        }
    }

    #[inline]
    fn fresh_fingerprint(&self, ctx: &InsertKeyCtx) -> Fingerprint {
        if ctx.single_word {
            Fingerprint::from_u64_prefix(ctx.w0, ctx.q + ctx.r)
        } else {
            Fingerprint::from_parts(ctx.home_u64, ctx.q, ctx.rem_full, ctx.r, &[])
        }
    }

    #[inline]
    fn try_insert_fresh_fast_path(
        &mut self,
        ctx: &InsertKeyCtx,
        check_load_limit: bool,
    ) -> AerisResult<Option<Fingerprint>> {
        let home = ctx.home_u64 as usize;
        let fresh_sentinel = 1u64 << ctx.r;
        let high = fresh_sentinel | ctx.rem_full;
        let slot0 = if ctx.m > 0 {
            (high << ctx.m) | (ctx.mem & self.geom.m_mask)
        } else {
            high
        };

        // Fast Path A: Unoccupied home slot (zero heap allocations, zero backward scans).
        let Some((run_start, run_end)) = self.table.locate_run(home) else {
            if check_load_limit {
                self.check_can_add_slots(1)?;
            }
            self.table.insert_single_slot_new_run(home, slot0);
            self.occupied_slots += 1;
            self.num_keys += 1;
            return Ok(Some(self.fresh_fingerprint(ctx)));
        };

        // Fast Path B: Occupied home slot — zero-allocation linear run scan comparing
        // `box_high` directly against `high = (1 << r) | rem_full` (zero `LZCNT`).
        let mut cursor = run_start;
        let mut insert_pos: Option<usize> = None;
        let mut matched_box: Option<(usize, u64, KeepsakeSlotKind)> = None;

        while cursor <= run_end {
            let (box_slot0, kind0) = self.table.get_payload_and_kind(cursor);
            if kind0 == KeepsakeSlotKind::Extension {
                return Ok(None);
            }
            let box_high = box_slot0 >> ctx.m;
            if box_high == high {
                matched_box = Some((cursor, box_slot0, kind0));
            } else if insert_pos.is_none() && (box_high < fresh_sentinel || box_high > high) {
                insert_pos = Some(cursor);
            }
            cursor = if kind0.is_box_end() {
                cursor + 1
            } else {
                self.table.find_box_end(cursor + 1) + 1
            };
        }

        self.apply_fresh_occupied_fast_path(
            ctx,
            run_end,
            high,
            slot0,
            insert_pos,
            matched_box,
            check_load_limit,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn apply_fresh_occupied_fast_path(
        &mut self,
        ctx: &InsertKeyCtx,
        run_end: usize,
        high: u64,
        slot0: u64,
        insert_pos: Option<usize>,
        matched_box: Option<(usize, u64, KeepsakeSlotKind)>,
        check_load_limit: bool,
    ) -> AerisResult<Option<Fingerprint>> {
        let home = ctx.home_u64 as usize;
        match matched_box {
            // Subcase B1 (>99% of occupied-home inserts): brand-new unextended box.
            None => {
                if check_load_limit {
                    self.check_can_add_slots(1)?;
                }
                let pos = insert_pos.unwrap_or(run_end + 1);
                self.table
                    .insert_single_slot_in_existing_run(home, pos, run_end, slot0);
                self.occupied_slots += 1;
                self.num_keys += 1;
                Ok(Some(self.fresh_fingerprint(ctx)))
            }
            // Subcase B2: exact match with single-slot unextended age=0 box (`m > 0`).
            Some((box_start, existing_slot0, kind0)) if ctx.m > 0 && kind0.is_box_end() => {
                if check_load_limit {
                    self.check_can_add_slots(1)?;
                }
                let m0 = existing_slot0 & self.geom.m_mask;
                let mem_masked = ctx.mem & self.geom.m_mask;
                let (lo, hi) = if m0 <= mem_masked {
                    (m0, mem_masked)
                } else {
                    (mem_masked, m0)
                };
                let new_slot0 = (high << ctx.m) | lo;
                let new_slot1 = hi << self.geom.cont_header_bits;
                self.table
                    .expand_single_slot_to_two(home, box_start, new_slot0, new_slot1, kind0);
                self.occupied_slots += 1;
                self.num_keys += 1;
                Ok(Some(self.fresh_fingerprint(ctx)))
            }
            // Subcase B3: multi-slot continuation box or point mode (`m == 0`) count >= 2.
            Some(_) => Ok(None),
        }
    }

    fn insert_key_slow_path(
        &mut self,
        ctx: &InsertKeyCtx,
        min_fp_bits: u16,
        check_load_limit: bool,
    ) -> AerisResult<Fingerprint> {
        let home = ctx.home_u64 as usize;
        let fresh_base = (ctx.q as u16) + (ctx.r as u16);
        let min_req = fresh_base.max(min_fp_bits);

        if !self.table.is_occupied(home) {
            return self.insert_into_unoccupied_slow(ctx, min_req - fresh_base, check_load_limit);
        }

        let boxes = self.decode_run(home);
        let (exact_match_idx, peer_full_fps) = self.scan_decoded_run_for_insert(ctx, &boxes);

        if let Some(idx) = exact_match_idx {
            let (box_start, old_len, mut target) = boxes[idx].clone();
            let is_run_end = idx + 1 == boxes.len();
            target.payload.insert_memento(ctx.mem);
            let new_len = target.encoded_slot_len(&self.geom);
            let delta = (new_len as isize) - (old_len as isize);
            if check_load_limit {
                self.check_can_add_slots(delta)?;
            }
            let encoded = target.encode(&self.geom, is_run_end);
            self.table
                .replace_region(home, box_start, old_len, &encoded);
            self.occupied_slots = ((self.occupied_slots as isize) + delta) as usize;
            self.num_keys += 1;
            return Ok(target.canonical_fingerprint(home, ctx.q, ctx.r));
        }

        let ext_level =
            self.compute_prefix_free_ext_level(ctx.p, min_req - fresh_base, &peer_full_fps);
        self.insert_new_box_sorted(ctx, &boxes, ext_level, check_load_limit)
    }

    fn insert_into_unoccupied_slow(
        &mut self,
        ctx: &InsertKeyCtx,
        min_ext_bits: u16,
        check_load_limit: bool,
    ) -> AerisResult<Fingerprint> {
        let home = ctx.home_u64 as usize;
        let ext_level = self.geom.min_ext_level_for_bits(min_ext_bits);
        let bx = KeepsakeBox::new_fresh(
            &self.hasher,
            ctx.p,
            ctx.q,
            &self.geom,
            ext_level,
            self.singleton_payload(ctx.mem),
        );
        let needed = bx.encoded_slot_len(&self.geom);
        if check_load_limit {
            self.check_can_add_slots(needed as isize)?;
        }
        let encoded = bx.encode(&self.geom, true);
        self.table.replace_region(home, 0, 0, &encoded);
        self.occupied_slots += needed;
        self.num_keys += 1;
        Ok(bx.canonical_fingerprint(home, ctx.q, ctx.r))
    }

    fn scan_decoded_run_for_insert(
        &self,
        ctx: &InsertKeyCtx,
        boxes: &[(usize, usize, KeepsakeBox)],
    ) -> (Option<usize>, Vec<Fingerprint>) {
        let home = ctx.home_u64 as usize;
        let fresh_base = (ctx.q as u16) + (ctx.r as u16);
        let mut peer_full_fps = Vec::new();
        for (idx, (_, _, bx)) in boxes.iter().enumerate() {
            if bx.total_fp_bits(ctx.q, ctx.r) >= fresh_base {
                if bx.matches_partition(&self.hasher, ctx.p, ctx.q, ctx.r) {
                    return (Some(idx), peer_full_fps);
                }
                let peer_fp = bx.canonical_fingerprint(home, ctx.q, ctx.r);
                if peer_fp.extract_bits(ctx.q as u16, ctx.r) == ctx.rem_full {
                    peer_full_fps.push(peer_fp);
                }
            }
        }
        (None, peer_full_fps)
    }

    fn compute_prefix_free_ext_level(
        &self,
        p: PartitionId,
        min_ext_bits: u16,
        peer_full_fps: &[Fingerprint],
    ) -> usize {
        let fresh_base = (self.config.q as u16) + (self.config.r as u16);
        let mut ext_level = self.geom.min_ext_level_for_bits(min_ext_bits);
        if peer_full_fps.is_empty() {
            return ext_level;
        }
        loop {
            let bit_len = fresh_base + self.geom.ext_bits_for_level(ext_level);
            let cand_fp = Fingerprint::from_hasher(&self.hasher, p, bit_len);
            let conflicts = peer_full_fps
                .iter()
                .any(|peer| cand_fp.is_prefix_of(peer) || peer.is_prefix_of(&cand_fp));
            if !conflicts {
                return ext_level;
            }
            ext_level += 1;
        }
    }

    fn insert_new_box_sorted(
        &mut self,
        ctx: &InsertKeyCtx,
        boxes: &[(usize, usize, KeepsakeBox)],
        ext_level: usize,
        check_load_limit: bool,
    ) -> AerisResult<Fingerprint> {
        let home = ctx.home_u64 as usize;
        let new_box = KeepsakeBox::new_fresh(
            &self.hasher,
            ctx.p,
            ctx.q,
            &self.geom,
            ext_level,
            self.singleton_payload(ctx.mem),
        );
        let needed = new_box.encoded_slot_len(&self.geom);
        if check_load_limit {
            self.check_can_add_slots(needed as isize)?;
        }

        let new_sort_key = new_box.sort_key();
        let insert_idx = boxes.partition_point(|(_, _, b)| b.sort_key() <= new_sort_key);
        if insert_idx < boxes.len() {
            let insert_pos = boxes[insert_idx].0;
            let encoded = new_box.encode(&self.geom, false);
            self.table.replace_region(home, insert_pos, 0, &encoded);
        } else {
            let (_, run_end) = self
                .table
                .locate_run(home)
                .expect("home is occupied so run must exist");
            let encoded = new_box.encode(&self.geom, true);
            self.table.replace_region(home, run_end + 1, 0, &encoded);
        }

        self.occupied_slots += needed;
        self.num_keys += 1;
        Ok(new_box.canonical_fingerprint(home, ctx.q, ctx.r))
    }

    fn find_best_delete_candidate(
        &self,
        boxes: &[(usize, usize, KeepsakeBox)],
        p: PartitionId,
        mem: Memento,
    ) -> Option<usize> {
        let q = self.config.q;
        let r = self.config.r;
        // Prefer the longest matching fingerprint (and youngest age on tie) containing `mem`.
        let mut best: Option<(usize, u16, u8)> = None;
        for (idx, (_, _, bx)) in boxes.iter().enumerate() {
            if bx.matches_partition(&self.hasher, p, q, r) && bx.payload.contains_in_range(mem, mem)
            {
                let fp_bits = bx.total_fp_bits(q, r);
                let better = match best {
                    None => true,
                    Some((_, best_bits, best_age)) => {
                        fp_bits > best_bits || (fp_bits == best_bits && bx.age < best_age)
                    }
                };
                if better {
                    best = Some((idx, fp_bits, bx.age));
                }
            }
        }
        best.map(|(idx, _, _)| idx)
    }

    fn apply_delete_to_run(
        &mut self,
        home: usize,
        boxes: &[(usize, usize, KeepsakeBox)],
        idx: usize,
        mem: Memento,
    ) -> Fingerprint {
        let q = self.config.q;
        let r = self.config.r;
        let (box_start, old_len, mut target) = boxes[idx].clone();
        let is_run_end = idx + 1 == boxes.len();
        let removed_fp = target.canonical_fingerprint(home, q, r);
        let removed = target.payload.remove_one(mem);
        debug_assert!(removed, "contains_in_range(mem, mem) guaranteed presence");

        if target.payload.is_empty() {
            self.table.replace_region(home, box_start, old_len, &[]);
            self.occupied_slots -= old_len;
        } else {
            let new_len = target.encoded_slot_len(&self.geom);
            let encoded = target.encode(&self.geom, is_run_end);
            self.table
                .replace_region(home, box_start, old_len, &encoded);
            self.occupied_slots =
                ((self.occupied_slots as isize) + (new_len as isize) - (old_len as isize)) as usize;
        }
        self.num_keys -= 1;
        removed_fp
    }
}
