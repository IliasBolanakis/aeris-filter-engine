//! Structural, encoding, accounting, and prefix-freeness invariant checks for [`AerisFilter`].

use super::AerisFilter;
use crate::keepsake::{BoxPayload, KeepsakeBox};
use aeris_types::{AerisError, AerisResult, Fingerprint};
use std::collections::BTreeSet;

impl AerisFilter {
    /// Verifies all structural Quotient Filter metadata invariants, keepsake box
    /// encoding invariants, key/slot accounting, and global fingerprint uniqueness.
    ///
    /// Intended for automated testing, fuzzing, and diagnostic audits.
    ///
    /// # Errors
    /// Returns [`AerisError::InternalInvariant`] on the first invariant violation.
    pub fn verify_invariants(&self) -> AerisResult<()> {
        self.table.verify_metadata_invariants()?;

        let mut counted_slots = 0usize;
        let mut counted_keys = 0usize;
        let mut seen_fps: BTreeSet<Fingerprint> = BTreeSet::new();

        for home in self.table.occupied_homes() {
            let (slots, keys) = self.verify_run_invariants(home, &mut seen_fps)?;
            counted_slots += slots;
            counted_keys += keys;
        }

        if counted_slots != self.occupied_slots {
            return Err(AerisError::InternalInvariant(format!(
                "occupied_slots accounting mismatch: tracked={}, counted={counted_slots}",
                self.occupied_slots
            )));
        }
        if counted_keys != self.num_keys {
            return Err(AerisError::InternalInvariant(format!(
                "num_keys accounting mismatch: tracked={}, counted={counted_keys}",
                self.num_keys
            )));
        }
        Ok(())
    }

    fn verify_run_invariants(
        &self,
        home: usize,
        seen_fps: &mut BTreeSet<Fingerprint>,
    ) -> AerisResult<(usize, usize)> {
        let q = self.config.q;
        let r = self.config.r;
        let fresh_base = (q as u16) + (r as u16);

        let (rs, re) = self.table.locate_run(home).ok_or_else(|| {
            AerisError::InternalInvariant(format!("occupied home {home} missing run"))
        })?;

        let mut cursor = rs;
        let mut counted_slots = 0usize;
        let mut counted_keys = 0usize;
        let mut full_length_fps: Vec<Fingerprint> = Vec::new();

        while cursor <= re {
            let (bx, consumed) = KeepsakeBox::decode_from_table(&self.table, &self.geom, cursor);
            self.verify_box_invariants(home, cursor, re, consumed, &bx)?;

            let fp = bx.canonical_fingerprint(home, q, r);
            if !seen_fps.insert(fp.clone()) {
                return Err(AerisError::InternalInvariant(format!(
                    "duplicate canonical fingerprint at home {home}"
                )));
            }
            if fp.bit_len() >= fresh_base {
                full_length_fps.push(fp);
            }

            counted_slots += consumed;
            counted_keys += bx.payload.key_count();
            cursor += consumed;
        }

        verify_pairwise_prefix_freeness(home, &full_length_fps)?;
        Ok((counted_slots, counted_keys))
    }

    fn verify_box_invariants(
        &self,
        home: usize,
        cursor: usize,
        run_end: usize,
        consumed: usize,
        bx: &KeepsakeBox,
    ) -> AerisResult<()> {
        if consumed == 0 || cursor + consumed - 1 > run_end {
            return Err(AerisError::InternalInvariant(format!(
                "keepsake box at slot {cursor} overran run end {run_end}"
            )));
        }
        let end_slot = cursor + consumed - 1;
        let end_kind = self.table.slot_kind(end_slot);
        if end_kind.is_extension_slot() || !end_kind.is_box_end() {
            return Err(AerisError::InternalInvariant(format!(
                "terminal slot {end_slot} of keepsake box has invalid kind {end_kind:?}"
            )));
        }
        if end_slot == run_end && !end_kind.is_run_end() {
            return Err(AerisError::InternalInvariant(format!(
                "run end slot {run_end} is not BoxAndRunEnd"
            )));
        }
        if end_slot < run_end && end_kind.is_run_end() {
            return Err(AerisError::InternalInvariant(format!(
                "interior box end slot {end_slot} prematurely marked BoxAndRunEnd"
            )));
        }
        if bx.age > self.config.r {
            return Err(AerisError::InternalInvariant(format!(
                "keepsake box age {} exceeds r={}",
                bx.age, self.config.r
            )));
        }
        if bx.payload.is_empty() {
            return Err(AerisError::InternalInvariant(format!(
                "empty keepsake box at home {home}, slot {cursor}"
            )));
        }
        if let BoxPayload::Mementos(ref mems) = bx.payload {
            if !mems.windows(2).all(|w| w[0] <= w[1]) {
                return Err(AerisError::InternalInvariant(format!(
                    "unsorted mementos in keepsake box at home {home}, slot {cursor}"
                )));
            }
        }
        Ok(())
    }
}

fn verify_pairwise_prefix_freeness(
    home: usize,
    full_length_fps: &[Fingerprint],
) -> AerisResult<()> {
    for i in 0..full_length_fps.len() {
        for j in (i + 1)..full_length_fps.len() {
            if full_length_fps[i].is_prefix_of(&full_length_fps[j])
                || full_length_fps[j].is_prefix_of(&full_length_fps[i])
            {
                return Err(AerisError::InternalInvariant(format!(
                    "prefix conflict between full-length boxes in home {home}"
                )));
            }
        }
    }
    Ok(())
}
