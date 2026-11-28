//! Deterministic 48-bit Feistel PRP key generation and streaming `*.keys.bin` output.

use crate::io::{ensure_parent_dir, IO_BUFFER_BYTES, KEYS_MAGIC, KEY_WRITE_CHUNK_LEN};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// 48-bit keyspace upper bound (`[0, 2^48)`).
pub const KEYSPACE_BITS: u32 = 48;
/// Total keys representable in the 48-bit keyspace (`2^48`).
pub const KEYSPACE_SIZE: u64 = 1u64 << KEYSPACE_BITS;
/// Bitmask for 48-bit keys (`2^48 - 1`).
pub const KEYSPACE_MASK: u64 = KEYSPACE_SIZE - 1;

/// SplitMix64 mixing function used for deterministic Feistel round keys and PRFs.
#[inline]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// 6-round balanced 24-bit Feistel pseudorandom permutation over `[0, 2^48)`.
///
/// # Mathematical Guarantees
/// - Strictly bijective on `[0, 2^48)`: mapping `i = 0..N` (`N <= 2^48`) produces
///   100% unique uniform 48-bit keys in shuffled order with `O(1)` extra RAM.
#[derive(Clone, Copy, Debug)]
pub struct Feistel48 {
    round_keys: [u64; 6],
}

impl Feistel48 {
    const HALF_MASK: u64 = (1u64 << 24) - 1;

    /// Derives 6 independent 64-bit round keys from `seed`.
    pub const fn from_seed(seed: u64) -> Self {
        let mut s = seed ^ 0xA3E1_5F01_9C4D_7B23;
        let mut rk = [0u64; 6];
        let mut i = 0;
        while i < 6 {
            s = splitmix64(s);
            rk[i] = s;
            i += 1;
        }
        Self { round_keys: rk }
    }

    /// Evaluates the bijective 48-bit permutation at index `x in [0, 2^48)`.
    #[inline]
    pub const fn permute(&self, x: u64) -> u64 {
        let mut l = (x >> 24) & Self::HALF_MASK;
        let mut r = x & Self::HALF_MASK;
        let mut i = 0;
        while i < 6 {
            let f = splitmix64(r ^ self.round_keys[i]) & Self::HALF_MASK;
            let next_l = r;
            let next_r = l ^ f;
            l = next_l;
            r = next_r;
            i += 1;
        }
        (l << 24) | r
    }
}

/// Streams `count` unique 48-bit keys to `out` using a 6-round Feistel PRP and
/// 16 MiB buffered sequential writes.
///
/// # Complexity
/// `O(count)` time and `O(1)` extra RAM (`~17 MiB` fixed I/O buffers).
///
/// # Errors
/// Returns an error if `count == 0`, `count > 2^48`, or file creation/writing fails.
pub fn generate_keys_file(count: u64, seed: u64, out: &Path) -> Result<u64, String> {
    if count == 0 {
        return Err("--count must be >= 1".to_string());
    }
    if count > KEYSPACE_SIZE {
        return Err(format!(
            "--count ({count}) exceeds 48-bit keyspace capacity ({KEYSPACE_SIZE})"
        ));
    }
    ensure_parent_dir(out).map_err(|e| format!("failed to create output directory: {e}"))?;

    let file = File::create(out)
        .map_err(|e| format!("failed to create key file '{}': {e}", out.display()))?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, file);

    writer
        .write_all(&KEYS_MAGIC)
        .and_then(|()| writer.write_all(&count.to_le_bytes()))
        .map_err(|e| format!("failed to write key file header: {e}"))?;

    let prp = Feistel48::from_seed(seed);
    let mut chunk_buf = vec![0u8; KEY_WRITE_CHUNK_LEN * 8];
    let mut emitted = 0u64;

    while emitted < count {
        let batch = ((count - emitted) as usize).min(KEY_WRITE_CHUNK_LEN);
        for i in 0..batch {
            let key = prp.permute(emitted + (i as u64));
            let off = i * 8;
            chunk_buf[off..off + 8].copy_from_slice(&key.to_le_bytes());
        }
        writer
            .write_all(&chunk_buf[..batch * 8])
            .map_err(|e| format!("failed to write key payload: {e}"))?;
        emitted += batch as u64;
    }

    writer
        .flush()
        .map_err(|e| format!("failed to flush key file: {e}"))?;
    Ok(16 + count * 8)
}
