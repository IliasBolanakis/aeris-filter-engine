//! SIMD-accelerated 64-bit `POPCNT` and `PDEP` rank/select primitives with
//! portable scalar fallbacks.
//!
//! This module is the **only** module in `aeris-core` permitted to contain
//! `unsafe` code. Every `unsafe` block is guarded by runtime CPU feature
//! detection (`is_x86_feature_detected!`) and has a bit-exact portable scalar
//! fallback verified by unit tests.

/// Counts the number of set (`1`) bits in `x` using a portable scalar implementation.
///
/// # Complexity
/// `O(1)` time and `O(1)` space.
#[inline]
pub const fn popcnt64_scalar(x: u64) -> u32 {
    x.count_ones()
}

/// Counts the number of set (`1`) bits in `x`.
///
/// Uses the x86-64 `POPCNT` instruction when supported by the host CPU,
/// falling back transparently to [`popcnt64_scalar`] on all other targets.
///
/// # Complexity
/// `O(1)` time and `O(1)` space.
#[inline]
pub fn popcnt64(x: u64) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "popcnt"))]
    {
        // SAFETY: Compile-time `target_feature = "popcnt"` guarantees CPU support.
        unsafe { popcnt64_x86(x) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "popcnt")))]
    {
        if std::arch::is_x86_feature_detected!("popcnt") {
            // SAFETY: Runtime feature detection verified that this x86_64 CPU
            // supports the `popcnt` instruction set extension.
            return unsafe { popcnt64_x86(x) };
        }
        popcnt64_scalar(x)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        popcnt64_scalar(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
#[inline]
unsafe fn popcnt64_x86(x: u64) -> u32 {
    // SAFETY: Caller has verified `popcnt` support via `is_x86_feature_detected!`.
    // `_popcnt64` is a pure register-to-register instruction returning a count in `0..=64`.
    core::arch::x86_64::_popcnt64(x as i64) as u32
}

/// Deposits contiguous low bits of `src` into bit positions where `mask` has `1`s,
/// using a portable scalar loop (exact semantics of x86 BMI2 `PDEP`).
///
/// # Postconditions
/// - The `k`-th set bit (0-based from LSB) of `mask` receives bit `k` of `src`.
/// - All bit positions where `mask` is `0` are strictly `0` in the result.
///
/// # Complexity
/// `O(popcnt(mask))` worst-case iterations (`<= 64`), `O(1)` space.
#[inline]
pub const fn pdep_u64_scalar(mut src: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    while mask != 0 && src != 0 {
        let lowest_bit = mask.isolate_lowest_one();
        if (src & 1) != 0 {
            result |= lowest_bit;
        }
        mask &= mask - 1;
        src >>= 1;
    }
    result
}

/// Parallel Bits Deposit (`PDEP`): deposits contiguous low bits of `src` into
/// bit positions selected by set bits in `mask`.
///
/// Uses the x86-64 BMI2 `_pdep_u64` instruction when supported by the host CPU,
/// falling back transparently to [`pdep_u64_scalar`] on non-BMI2 CPUs or
/// non-x86 architectures.
///
/// # Complexity
/// `O(1)` time (single instruction on BMI2 CPUs), `O(1)` space.
#[inline]
pub fn pdep_u64(src: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // SAFETY: Compile-time `target_feature = "bmi2"` guarantees CPU support.
        unsafe { pdep_u64_bmi2(src, mask) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "bmi2")))]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            // SAFETY: Runtime feature detection verified that this x86_64 CPU
            // supports the `bmi2` instruction set extension.
            return unsafe { pdep_u64_bmi2(src, mask) };
        }
        pdep_u64_scalar(src, mask)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        pdep_u64_scalar(src, mask)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn pdep_u64_bmi2(src: u64, mask: u64) -> u64 {
    // SAFETY: Caller has verified `bmi2` support via `is_x86_feature_detected!`.
    // `_pdep_u64` is a pure register-to-register instruction with no memory accesses.
    core::arch::x86_64::_pdep_u64(src, mask)
}

/// Returns the 0-based bit index (`0..64`) of the `rank`-th (0-based) set bit
/// in `bitmap`, or `64` if `bitmap` contains at most `rank` set bits, using
/// only portable scalar operations.
///
/// # Complexity
/// `O(1)` space, at most 64 loop iterations.
#[inline]
pub const fn select64_scalar(bitmap: u64, rank: u32) -> u32 {
    if rank >= 64 {
        return 64;
    }
    pdep_u64_scalar(1u64 << rank, bitmap).trailing_zeros()
}

/// Returns the 0-based bit index (`0..64`) of the `rank`-th (0-based) set bit
/// in `bitmap`, or `64` if `bitmap` contains at most `rank` set bits.
///
/// Implemented via `pdep_u64(1u64 << rank, bitmap).trailing_zeros()` for `O(1)`
/// selection on BMI2 CPUs with automatic scalar fallback.
///
/// # Postconditions
/// - Returns `i < 64` iff `(bitmap >> i) & 1 == 1` and `rank64(bitmap, i) == rank + 1`.
/// - Returns `64` iff `popcnt64(bitmap) <= rank` (including when `rank >= 64`).
///
/// # Complexity
/// `O(1)` time on BMI2 CPUs, `O(1)` space.
#[inline]
pub fn select64(bitmap: u64, rank: u32) -> u32 {
    if rank >= 64 {
        return 64;
    }
    pdep_u64(1u64 << rank, bitmap).trailing_zeros()
}

/// Counts the number of set bits in `bitmap` within inclusive bit positions `0..=pos`.
///
/// If `pos >= 63`, all 64 bits of `bitmap` are counted.
///
/// # Complexity
/// `O(1)` time and `O(1)` space.
#[inline]
pub fn rank64(bitmap: u64, pos: u32) -> u32 {
    let clamped = if pos > 63 { 63 } else { pos };
    let mask = u64::MAX >> (63 - clamped);
    popcnt64(bitmap & mask)
}

/// Counts the number of set bits in `bitmap` within half-open bit positions `0..pos`
/// (strictly below `pos`).
///
/// Returns `0` when `pos == 0` and `popcnt64(bitmap)` when `pos >= 64`.
///
/// # Complexity
/// `O(1)` time and `O(1)` space.
#[inline]
pub fn rank64_exclusive(bitmap: u64, pos: u32) -> u32 {
    if pos == 0 {
        return 0;
    }
    if pos >= 64 {
        return popcnt64(bitmap);
    }
    let mask = (1u64 << pos) - 1;
    popcnt64(bitmap & mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn popcnt_matches_scalar_on_edges_and_random() {
        let edges = [
            0u64,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            1u64 << 63,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0x0F0F_0F0F_0F0F_0F0F,
            0x8000_0000_0000_0001,
        ];
        for x in edges {
            assert_eq!(popcnt64(x), popcnt64_scalar(x), "mismatch for {x:#x}");
        }
        for bit in 0..64 {
            let x = 1u64 << bit;
            assert_eq!(popcnt64(x), 1);
            assert_eq!(popcnt64(!x), 63);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(0x51ED_0001);
        for _ in 0..10_000 {
            let x = rng.next_u64();
            assert_eq!(popcnt64(x), popcnt64_scalar(x));
        }
    }

    #[test]
    fn pdep_matches_scalar_on_edges_and_random() {
        let edges = [
            0u64,
            1,
            0xFF,
            1u64 << 63,
            u64::MAX,
            u64::MAX - 1,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0x00FF_00FF_00FF_00FF,
        ];
        for &src in &edges {
            for &mask in &edges {
                assert_eq!(
                    pdep_u64(src, mask),
                    pdep_u64_scalar(src, mask),
                    "pdep mismatch for src={src:#x}, mask={mask:#x}"
                );
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(0x51ED_0002);
        for _ in 0..10_000 {
            let src = rng.next_u64();
            let mask = rng.next_u64();
            assert_eq!(pdep_u64(src, mask), pdep_u64_scalar(src, mask));
        }
    }

    #[test]
    fn select_and_rank_invariants_hold() {
        // Empty bitmap
        for r in [0, 1, 63, 64, 100] {
            assert_eq!(select64(0, r), 64);
            assert_eq!(select64_scalar(0, r), 64);
        }
        for p in [0, 1, 31, 63, 64, 100] {
            assert_eq!(rank64(0, p), 0);
            assert_eq!(rank64_exclusive(0, p), 0);
        }

        // Full bitmap
        for r in 0..64 {
            assert_eq!(select64(u64::MAX, r), r);
            assert_eq!(select64_scalar(u64::MAX, r), r);
            assert_eq!(rank64(u64::MAX, r), r + 1);
            assert_eq!(rank64_exclusive(u64::MAX, r), r);
        }
        assert_eq!(select64(u64::MAX, 64), 64);
        assert_eq!(rank64(u64::MAX, 64), 64);
        assert_eq!(rank64_exclusive(u64::MAX, 64), 64);

        // Single-bit bitmaps
        for bit in 0..64u32 {
            let bm = 1u64 << bit;
            assert_eq!(select64(bm, 0), bit);
            assert_eq!(select64(bm, 1), 64);
            for p in 0..64u32 {
                assert_eq!(rank64(bm, p), u32::from(p >= bit));
                assert_eq!(rank64_exclusive(bm, p), u32::from(p > bit));
            }
        }

        // Random bitmaps: verify rank/select round-trip and SIMD == scalar
        let mut rng = ChaCha8Rng::seed_from_u64(0x51ED_0003);
        for _ in 0..2_000 {
            let bm = rng.next_u64();
            let total = popcnt64(bm);
            for r in 0..=total.min(64) {
                let idx = select64(bm, r);
                assert_eq!(idx, select64_scalar(bm, r));
                if r < total {
                    assert!(idx < 64);
                    assert_ne!(bm & (1u64 << idx), 0);
                    assert_eq!(rank64(bm, idx), r + 1);
                    assert_eq!(rank64_exclusive(bm, idx), r);
                } else {
                    assert_eq!(idx, 64);
                }
            }
        }
    }
}
