//! In-place 8-bit LSD radix sort with L1 write-combining buffers for packed `u64` keys.

/// Bits per LSD radix sort pass (`8`-bit byte passes -> `256` histogram buckets).
const RADIX_BITS: usize = 8;

/// Number of histogram buckets per 8-bit LSD radix pass (`2^8 = 256`).
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// In-place 8-bit LSD radix sort for packed `u64` keys spanning the low
/// `total_bits` (`0..=64`) bits.
///
/// # Complexity
/// `O(n * ceil(total_bits / 8))` time, `8 * n` bytes transient scratch space.
pub(super) fn radix_sort_u64(data: &mut Vec<u64>, total_bits: usize) {
    const WC_THRESHOLD: usize = 65_536;

    let n = data.len();
    if n <= 1 || total_bits == 0 {
        return;
    }

    let num_passes = total_bits.min(64).div_ceil(RADIX_BITS);
    let counts = compute_radix_histograms(data, num_passes);
    if !counts
        .iter()
        .take(num_passes)
        .any(|hist| !hist.contains(&n))
    {
        return;
    }

    let mut tmp: Vec<u64> = vec![0u64; n];
    let mut in_tmp = false;

    for (pass, hist) in counts.iter().enumerate().take(num_passes) {
        if hist.contains(&n) {
            continue;
        }
        let mut offsets = [0usize; RADIX_BUCKETS];
        let mut sum = 0usize;
        for (off, &c) in offsets.iter_mut().zip(hist.iter()) {
            *off = sum;
            sum += c;
        }

        let shift = pass * RADIX_BITS;
        let (src, dst): (&[u64], &mut [u64]) = if in_tmp {
            (&tmp, data.as_mut_slice())
        } else {
            (data.as_slice(), &mut tmp)
        };
        if n >= WC_THRESHOLD {
            scatter_radix_pass_wc(src, dst, shift, &mut offsets);
        } else {
            scatter_radix_pass_direct(src, dst, shift, &mut offsets);
        }
        in_tmp = !in_tmp;
    }

    if in_tmp {
        std::mem::swap(data, &mut tmp);
    }
    drop(tmp);
}

fn compute_radix_histograms(data: &[u64], num_passes: usize) -> [[usize; RADIX_BUCKETS]; 8] {
    let mut counts = [[0usize; RADIX_BUCKETS]; 8];
    match num_passes {
        5 => {
            for &val in data {
                counts[0][(val & 0xFF) as usize] += 1;
                counts[1][((val >> 8) & 0xFF) as usize] += 1;
                counts[2][((val >> 16) & 0xFF) as usize] += 1;
                counts[3][((val >> 24) & 0xFF) as usize] += 1;
                counts[4][((val >> 32) & 0xFF) as usize] += 1;
            }
        }
        6 => {
            for &val in data {
                counts[0][(val & 0xFF) as usize] += 1;
                counts[1][((val >> 8) & 0xFF) as usize] += 1;
                counts[2][((val >> 16) & 0xFF) as usize] += 1;
                counts[3][((val >> 24) & 0xFF) as usize] += 1;
                counts[4][((val >> 32) & 0xFF) as usize] += 1;
                counts[5][((val >> 40) & 0xFF) as usize] += 1;
            }
        }
        _ => {
            for &val in data {
                for (pass, hist) in counts.iter_mut().enumerate().take(num_passes) {
                    let byte = ((val >> (pass * RADIX_BITS)) & 0xFF) as usize;
                    hist[byte] += 1;
                }
            }
        }
    }
    counts
}

fn scatter_radix_pass_wc(
    src: &[u64],
    dst: &mut [u64],
    shift: usize,
    offsets: &mut [usize; RADIX_BUCKETS],
) {
    const WC_BUF_LEN: usize = 8;
    // 16 KiB L1-resident write-combining buffer (256 buckets * 64 B cache line)
    // reduces DRAM TLB thrashing and bounds checks by 8x on large inputs.
    let mut wc_buf = [[0u64; WC_BUF_LEN]; RADIX_BUCKETS];
    let mut wc_len = [0u8; RADIX_BUCKETS];
    for &val in src {
        let bucket = ((val >> shift) & 0xFF) as usize;
        let len = wc_len[bucket] as usize;
        wc_buf[bucket][len] = val;
        let next_len = len + 1;
        if next_len == WC_BUF_LEN {
            let pos = offsets[bucket];
            dst[pos..(pos + WC_BUF_LEN)].copy_from_slice(&wc_buf[bucket]);
            offsets[bucket] = pos + WC_BUF_LEN;
            wc_len[bucket] = 0;
        } else {
            wc_len[bucket] = next_len as u8;
        }
    }
    for bucket in 0..RADIX_BUCKETS {
        let rem = wc_len[bucket] as usize;
        if rem > 0 {
            let pos = offsets[bucket];
            dst[pos..(pos + rem)].copy_from_slice(&wc_buf[bucket][..rem]);
        }
    }
}

fn scatter_radix_pass_direct(
    src: &[u64],
    dst: &mut [u64],
    shift: usize,
    offsets: &mut [usize; RADIX_BUCKETS],
) {
    for &val in src {
        let bucket = ((val >> shift) & 0xFF) as usize;
        let pos = offsets[bucket];
        dst[pos] = val;
        offsets[bucket] = pos + 1;
    }
}
