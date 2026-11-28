//! Pure binary codecs and state-folding algebra for the Storage Plane.
//!
//! This module contains zero disk I/O, zero clocks, and zero `unsafe` code.
//! It defines:
//! 1. The deterministic binary WAL frame header and [`aeris_types::WalOp`] payload codec
//!    protected by IEEE CRC32 (`crc32fast`).
//! 2. The write-optimized Reverse Map merge operand codec (`AddKey`, `DelKey`,
//!    `CompactedSnapshot`) and associative `full_merge` / `partial_merge`
//!    state-folding functions used by the RocksDB `MergeOperator`.

mod reverse_map_codec;
mod wal_codec;

#[cfg(test)]
mod tests;

pub use reverse_map_codec::{
    count_keys_in_value, decode_key_list, encode_add_key_operand, encode_compacted_snapshot,
    encode_del_key_operand, fold_full_merge, fold_partial_merge,
};
pub use wal_codec::{
    compute_frame_crc32, decode_wal_op, encode_wal_frame, encode_wal_op, WalFrameHeader,
};

use aeris_types::{AerisError, AerisResult};

/// 4-byte magic prefix identifying every Aeris WAL frame (`b"AWAL"`).
pub const WAL_MAGIC: [u8; 4] = *b"AWAL";

/// Current on-disk WAL binary frame format version.
pub const WAL_FORMAT_VERSION: u8 = 1;

/// Fixed size in bytes of a WAL frame header:
/// `[magic: 4][version: 1][payload_len: 4][lsn: 8][crc32: 4] = 21 bytes`.
pub const WAL_HEADER_LEN: usize = 21;

/// Maximum permitted WAL payload size in bytes (64 MiB) to guard against
/// out-of-memory allocations when encountering corrupted length headers.
pub const MAX_WAL_PAYLOAD_LEN: u32 = 64 * 1024 * 1024;

/// Reverse Map merge operand tag: append a single 64-bit key (`0x01` + `u64` LE).
pub const RM_TAG_ADD_KEY: u8 = 0x01;
/// Reverse Map merge operand tag: remove a single 64-bit key (`0x02` + `u64` LE).
pub const RM_TAG_DEL_KEY: u8 = 0x02;
/// Reverse Map merge operand tag: canonical sorted key snapshot
/// (`0x03` + `u32` LE count + `count * u64` LE sorted keys).
pub const RM_TAG_COMPACTED_SNAPSHOT: u8 = 0x03;

/// Fixed encoded size in bytes of a single `AddKey` or `DelKey` operand.
pub const RM_SINGLE_KEY_OP_LEN: usize = 1 + 8;

/// Fixed header size in bytes of a `CompactedSnapshot` operand (`tag + u32 count`).
pub const RM_SNAPSHOT_HEADER_LEN: usize = 1 + 4;

/// Converts a slice/collection length into `u32`, returning an [`AerisError::EncodingError`]
/// labeled with `label` on overflow.
pub(super) fn checked_u32_len(len: usize, label: &str) -> AerisResult<u32> {
    u32::try_from(len)
        .map_err(|_| AerisError::EncodingError(format!("{label} length {len} overflows u32")))
}

/// Bounds-checked little-endian binary reader with zero heap allocations.
pub(super) struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    #[inline]
    pub(super) const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    pub(super) const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    pub(super) const fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    #[inline]
    pub(super) fn read_slice(&mut self, len: usize) -> AerisResult<&'a [u8]> {
        if self.remaining() < len {
            return Err(AerisError::EncodingError(format!(
                "unexpected EOF: need {len} bytes, only {} remaining",
                self.remaining()
            )));
        }
        let slice = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    #[inline]
    pub(super) fn read_u8(&mut self) -> AerisResult<u8> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    #[inline]
    pub(super) fn read_u32_le(&mut self) -> AerisResult<u32> {
        let s = self.read_slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    #[inline]
    pub(super) fn read_u64_le(&mut self) -> AerisResult<u64> {
        let s = self.read_slice(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
}
