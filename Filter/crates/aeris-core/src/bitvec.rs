//! Tightly bit-packed fixed-width slot array storing Quotient Filter payloads.
//!
//! Each physical slot holds `slot_bits = r + 1 + m` bits (`1 <= slot_bits <= 64`),
//! packed contiguously inside a `Vec<u64>` with one trailing zero-padded guard
//! word so unaligned cross-word reads and writes avoid boundary branches.

/// Tightly bit-packed array of `num_slots` unsigned integers of exact width
/// `slot_bits` (`1 <= slot_bits <= 64`).
///
/// # Invariants
/// - `1 <= slot_bits <= 64`.
/// - `words.len() == (num_slots * slot_bits).div_ceil(64) + 1` (includes 1 trailing
///   padding word so `words[word_idx + 1]` is always in bounds for any valid slot).
/// - All bits outside `[0, num_slots * slot_bits)` remain strictly zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPackedSlots {
    words: Vec<u64>,
    num_slots: usize,
    slot_bits: u8,
    mask: u64,
}

impl BitPackedSlots {
    /// Allocates a zero-initialized [`BitPackedSlots`] array of `num_slots` entries,
    /// each of exact width `slot_bits` bits.
    ///
    /// # Panics
    /// Panics if `slot_bits == 0` or `slot_bits > 64`.
    ///
    /// # Complexity
    /// `O(num_slots * slot_bits / 64)` time and space.
    pub fn new(num_slots: usize, slot_bits: u8) -> Self {
        assert!(
            (1..=64).contains(&slot_bits),
            "slot_bits must be in 1..=64, got {slot_bits}"
        );
        let mask = if slot_bits == 64 {
            u64::MAX
        } else {
            (1u64 << slot_bits) - 1
        };
        let word_count = Self::required_words(num_slots, slot_bits);
        Self {
            words: vec![0u64; word_count],
            num_slots,
            slot_bits,
            mask,
        }
    }

    /// Returns the number of addressable slots.
    #[inline]
    pub const fn len(&self) -> usize {
        self.num_slots
    }

    /// Returns `true` if `len() == 0`.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.num_slots == 0
    }

    /// Returns the fixed bit width per slot (`1..=64`).
    #[inline]
    pub const fn slot_bits(&self) -> u8 {
        self.slot_bits
    }

    /// Returns the bit mask `((1 << slot_bits) - 1)` (or `u64::MAX` for 64 bits).
    #[inline]
    pub const fn slot_mask(&self) -> u64 {
        self.mask
    }

    /// Returns the heap memory footprint of the backing word buffer in bytes.
    #[inline]
    pub fn memory_usage_bytes(&self) -> usize {
        self.words.len() * core::mem::size_of::<u64>()
    }

    /// Grows (or shrinks) the addressable slot count to `new_num_slots`,
    /// zero-initializing any newly added slots and preserving the trailing guard word.
    ///
    /// # Complexity
    /// Amortized `O(new_num_slots * slot_bits / 64)` when growing.
    pub fn resize(&mut self, new_num_slots: usize) {
        let new_words = Self::required_words(new_num_slots, self.slot_bits);
        self.words.resize(new_words, 0u64);
        if new_num_slots < self.num_slots {
            let total_bits = new_num_slots * (self.slot_bits as usize);
            let last_word = total_bits >> 6;
            let keep_bits = (total_bits & 63) as u32;
            if keep_bits == 0 {
                self.words[last_word] = 0;
            } else {
                self.words[last_word] &= (1u64 << keep_bits) - 1;
            }
            for w in &mut self.words[(last_word + 1)..] {
                *w = 0;
            }
        }
        self.num_slots = new_num_slots;
    }

    /// Reads the `slot_bits`-wide unsigned integer at `index`.
    ///
    /// # Panics
    /// Panics in debug builds (or via slice bounds check) if `index >= self.len()`.
    ///
    /// # Complexity
    /// `O(1)` time, zero heap allocations.
    #[inline]
    pub fn get(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.num_slots,
            "slot index {index} out of bounds {}",
            self.num_slots
        );
        let bit_pos = index * (self.slot_bits as usize);
        let word_idx = bit_pos >> 6;
        let bit_off = (bit_pos & 63) as u32;
        let w0 = self.words[word_idx];

        if bit_off + (self.slot_bits as u32) <= 64 {
            (w0 >> bit_off) & self.mask
        } else {
            // `bit_off` is strictly in 1..=63 here, so `64 - bit_off` is in 1..=63.
            let lo = w0 >> bit_off;
            let hi = self.words[word_idx + 1] << (64 - bit_off);
            (lo | hi) & self.mask
        }
    }

    /// Writes the low `slot_bits` bits of `value` into slot `index`.
    ///
    /// Any bits above `slot_bits` in `value` are masked off.
    ///
    /// # Panics
    /// Panics in debug builds (or via slice bounds check) if `index >= self.len()`.
    ///
    /// # Complexity
    /// `O(1)` time, zero heap allocations.
    #[inline]
    pub fn set(&mut self, index: usize, value: u64) {
        debug_assert!(
            index < self.num_slots,
            "slot index {index} out of bounds {}",
            self.num_slots
        );
        let masked = value & self.mask;
        let bit_pos = index * (self.slot_bits as usize);
        let word_idx = bit_pos >> 6;
        let bit_off = (bit_pos & 63) as u32;

        if bit_off + (self.slot_bits as u32) <= 64 {
            let clear = !(self.mask << bit_off);
            self.words[word_idx] = (self.words[word_idx] & clear) | (masked << bit_off);
        } else {
            // `bit_off` is strictly in 1..=63.
            let lo_bits = 64 - bit_off;
            let hi_bits = (self.slot_bits as u32) - lo_bits;
            let lo_keep_mask = (1u64 << bit_off) - 1;
            self.words[word_idx] = (self.words[word_idx] & lo_keep_mask) | (masked << bit_off);

            let hi_clear_mask = !((1u64 << hi_bits) - 1);
            self.words[word_idx + 1] =
                (self.words[word_idx + 1] & hi_clear_mask) | (masked >> lo_bits);
        }
    }

    /// Shifts slot payloads `[start .. end)` right by 1 position into
    /// `[start + 1 .. end + 1)` and zeroes slot `start`.
    ///
    /// # Preconditions
    /// `start <= end` and `end < self.len()`.
    ///
    /// # Complexity
    /// `O(end - start)` time, `O(1)` space.
    pub fn shift_right_one(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end && end < self.num_slots);
        let mut i = end;
        while i > start {
            let v = self.get(i - 1);
            self.set(i, v);
            i -= 1;
        }
        self.set(start, 0);
    }

    /// Shifts slot payloads `(start ..= end]` left by 1 position into
    /// `[start .. end)` and zeroes slot `end`.
    ///
    /// # Preconditions
    /// `start <= end` and `end < self.len()`.
    ///
    /// # Complexity
    /// `O(end - start)` time, `O(1)` space.
    pub fn shift_left_one(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end && end < self.num_slots);
        for i in start..end {
            let v = self.get(i + 1);
            self.set(i, v);
        }
        self.set(end, 0);
    }

    #[inline]
    const fn required_words(num_slots: usize, slot_bits: u8) -> usize {
        let total_bits = num_slots * (slot_bits as usize);
        total_bits.div_ceil(64) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn round_trip_across_all_representative_bit_widths() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xB17B_0001);
        for slot_bits in [1u8, 2, 7, 13, 17, 18, 23, 31, 32, 33, 47, 63, 64] {
            let n = 257;
            let mut slots = BitPackedSlots::new(n, slot_bits);
            let mask = if slot_bits == 64 {
                u64::MAX
            } else {
                (1u64 << slot_bits) - 1
            };
            let mut expected = vec![0u64; n];

            for (i, exp) in expected.iter_mut().enumerate() {
                let raw = rng.next_u64();
                slots.set(i, raw);
                *exp = raw & mask;
            }
            for (i, &exp) in expected.iter().enumerate() {
                assert_eq!(slots.get(i), exp, "width={slot_bits}, idx={i}");
            }

            // Overwrite pass to ensure no neighbour bit bleed.
            for (i, exp) in expected.iter_mut().enumerate().rev() {
                let raw = rng.next_u64();
                slots.set(i, raw);
                *exp = raw & mask;
            }
            for (i, &exp) in expected.iter().enumerate() {
                assert_eq!(slots.get(i), exp, "overwrite width={slot_bits}, idx={i}");
            }
        }
    }

    #[test]
    fn shift_right_and_left_preserve_exact_neighbors() {
        let mut slots = BitPackedSlots::new(128, 19);
        let mut model = vec![0u64; 128];
        let mask = (1u64 << 19) - 1;
        for (i, m) in model.iter_mut().enumerate() {
            let v = ((i as u64).wrapping_mul(2654435761) ^ 0x5A5A) & mask;
            slots.set(i, v);
            *m = v;
        }

        // Shift [10..70) right by 1 into [11..71), zeroing 10.
        slots.shift_right_one(10, 70);
        for i in (11..=70).rev() {
            model[i] = model[i - 1];
        }
        model[10] = 0;
        for (i, &exp) in model.iter().enumerate() {
            assert_eq!(slots.get(i), exp, "after shift_right at {i}");
        }

        // Shift (10..=70] left by 1 into [10..70), zeroing 70.
        slots.shift_left_one(10, 70);
        for i in 10..70 {
            model[i] = model[i + 1];
        }
        model[70] = 0;
        for (i, &exp) in model.iter().enumerate() {
            assert_eq!(slots.get(i), exp, "after shift_left at {i}");
        }
    }

    #[test]
    fn resize_grow_and_shrink_preserves_prefix_and_zeroes_tail() {
        let mut slots = BitPackedSlots::new(64, 17);
        for i in 0..64 {
            slots.set(i, (i as u64) + 100);
        }
        slots.resize(192);
        assert_eq!(slots.len(), 192);
        for i in 0..64 {
            assert_eq!(slots.get(i), (i as u64) + 100);
        }
        for i in 64..192 {
            assert_eq!(slots.get(i), 0);
        }
        slots.set(191, 0x1FFFF);
        assert_eq!(slots.get(191), 0x1FFFF);

        slots.resize(32);
        assert_eq!(slots.len(), 32);
        for i in 0..32 {
            assert_eq!(slots.get(i), (i as u64) + 100);
        }
        slots.resize(64);
        for i in 32..64 {
            assert_eq!(slots.get(i), 0);
        }
    }
}
