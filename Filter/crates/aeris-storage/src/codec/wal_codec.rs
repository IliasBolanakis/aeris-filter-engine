//! Deterministic binary WAL frame header and [`WalOp`] payload codec.
//!
//! Protects every frame with a 21-byte header (`magic`, `version`, `payload_len`,
//! `lsn`, and IEEE CRC32 over `[version, payload_len_le, lsn_le, payload]`).

use super::{
    checked_u32_len, ByteCursor, MAX_WAL_PAYLOAD_LEN, WAL_FORMAT_VERSION, WAL_HEADER_LEN, WAL_MAGIC,
};
use aeris_types::{AerisError, AerisResult, Fingerprint, Key, Lsn, QueryRange, WalOp};

/// Binary tag for [`WalOp::Put`].
pub(super) const WAL_OP_TAG_PUT: u8 = 0x01;
/// Binary tag for [`WalOp::Delete`].
pub(super) const WAL_OP_TAG_DELETE: u8 = 0x02;
/// Binary tag for [`WalOp::Adapt`].
pub(super) const WAL_OP_TAG_ADAPT: u8 = 0x03;
/// Binary tag for [`WalOp::Checkpoint`].
pub(super) const WAL_OP_TAG_CHECKPOINT: u8 = 0x04;

/// Minimum encoded byte length of a length-prefixed [`Fingerprint`] (`u32 len + u16 bit_len`).
const MIN_LENGTH_PREFIXED_FP_BYTES: usize = 6;
/// Minimum encoded byte length of an `Adapt` new entry (`fp + u64 key`).
const MIN_ADAPT_NEW_ENTRY_BYTES: usize = MIN_LENGTH_PREFIXED_FP_BYTES + 8;

/// Parsed 21-byte header of an on-disk WAL frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalFrameHeader {
    /// Format version stored in the frame header.
    pub version: u8,
    /// Byte length of the serialized [`WalOp`] payload immediately following the header.
    pub payload_len: u32,
    /// Monotonic Log Sequence Number assigned to this frame.
    pub lsn: Lsn,
    /// IEEE CRC32 checksum covering `[version, payload_len_le, lsn_le, payload]`.
    pub crc32: u32,
}

impl WalFrameHeader {
    /// Parses a 21-byte WAL frame header slice, verifying magic bytes and format version.
    ///
    /// # Errors
    /// Returns [`AerisError::EncodingError`] if `bytes.len() < WAL_HEADER_LEN`,
    /// if the 4-byte magic does not match [`WAL_MAGIC`], or if the version does not
    /// match [`WAL_FORMAT_VERSION`].
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    pub fn decode(bytes: &[u8]) -> AerisResult<Self> {
        if bytes.len() < WAL_HEADER_LEN {
            return Err(AerisError::EncodingError(format!(
                "WAL header truncated: expected {WAL_HEADER_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0..4] != WAL_MAGIC {
            return Err(AerisError::EncodingError(format!(
                "invalid WAL frame magic: expected {:?}, got {:?}",
                WAL_MAGIC,
                &bytes[0..4]
            )));
        }
        let version = bytes[4];
        if version != WAL_FORMAT_VERSION {
            return Err(AerisError::EncodingError(format!(
                "unsupported WAL frame version {version} (expected {WAL_FORMAT_VERSION})"
            )));
        }
        let payload_len = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        let lsn = u64::from_le_bytes([
            bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15], bytes[16],
        ]);
        let crc32 = u32::from_le_bytes([bytes[17], bytes[18], bytes[19], bytes[20]]);
        Ok(Self {
            version,
            payload_len,
            lsn,
            crc32,
        })
    }

    /// Encodes this header into its canonical 21-byte representation.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space (stack-allocated).
    #[must_use]
    pub fn encode(&self) -> [u8; WAL_HEADER_LEN] {
        let mut out = [0u8; WAL_HEADER_LEN];
        out[0..4].copy_from_slice(&WAL_MAGIC);
        out[4] = self.version;
        out[5..9].copy_from_slice(&self.payload_len.to_le_bytes());
        out[9..17].copy_from_slice(&self.lsn.to_le_bytes());
        out[17..21].copy_from_slice(&self.crc32.to_le_bytes());
        out
    }
}

/// Computes the canonical IEEE CRC32 checksum for a WAL frame covering
/// `[version, payload_len_le, lsn_le, payload]`.
///
/// # Complexity
/// `O(payload.len())` time and `O(1)` space.
#[must_use]
pub fn compute_frame_crc32(version: u8, payload_len: u32, lsn: Lsn, payload: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&[version]);
    hasher.update(&payload_len.to_le_bytes());
    hasher.update(&lsn.to_le_bytes());
    hasher.update(payload);
    hasher.finalize()
}

/// Encodes `(lsn, op)` into a complete checksummed WAL binary frame.
///
/// Returns `(frame_bytes, crc32)`.
///
/// # Preconditions
/// - `lsn >= 1`.
/// - Encoded payload length must not exceed [`MAX_WAL_PAYLOAD_LEN`].
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if `lsn == 0` or if the encoded payload
/// exceeds [`MAX_WAL_PAYLOAD_LEN`].
///
/// # Complexity
/// `O(|op|)` time and `O(|op|)` space.
pub fn encode_wal_frame(lsn: Lsn, op: &WalOp) -> AerisResult<(Vec<u8>, u32)> {
    if lsn == 0 {
        return Err(AerisError::EncodingError(
            "WAL LSN must be strictly positive (>= 1)".to_string(),
        ));
    }
    let payload = encode_wal_op(op)?;
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        AerisError::EncodingError(format!(
            "WAL payload length {} overflows u32",
            payload.len()
        ))
    })?;
    if payload_len > MAX_WAL_PAYLOAD_LEN {
        return Err(AerisError::EncodingError(format!(
            "WAL payload length {payload_len} exceeds cap {MAX_WAL_PAYLOAD_LEN}"
        )));
    }
    let crc32 = compute_frame_crc32(WAL_FORMAT_VERSION, payload_len, lsn, &payload);
    let header = WalFrameHeader {
        version: WAL_FORMAT_VERSION,
        payload_len,
        lsn,
        crc32,
    };
    let mut frame = Vec::with_capacity(WAL_HEADER_LEN + payload.len());
    frame.extend_from_slice(&header.encode());
    frame.extend_from_slice(&payload);
    Ok((frame, crc32))
}

/// Deterministically serializes a [`WalOp`] into its compact binary payload encoding.
///
/// # Binary Wire Layout
/// - `Put`: `[0x01][key: u64 LE][val_len: u32 LE][val][fp_len: u32 LE][fp_bytes]`
/// - `Delete`: `[0x02][key: u64 LE][fp_len: u32 LE][fp_bytes]`
/// - `Adapt`: `[0x03][start: u64 LE][end: u64 LE][old_count: u32 LE]([fp_len][fp])*[new_count: u32 LE]([fp_len][fp][key: u64 LE])*`
/// - `Checkpoint`: `[0x04][up_to_lsn: u64 LE]`
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if any vector or slice length overflows `u32`.
///
/// # Complexity
/// `O(|op|)` time and `O(|op|)` space.
pub fn encode_wal_op(op: &WalOp) -> AerisResult<Vec<u8>> {
    let mut out = Vec::new();
    match op {
        WalOp::Put { key, value, fp } => {
            let fp_bytes = fp.to_bytes();
            let val_len = checked_u32_len(value.len(), "Put value")?;
            let fp_len = checked_u32_len(fp_bytes.len(), "Put fingerprint")?;
            out.reserve(1 + 8 + 4 + value.len() + 4 + fp_bytes.len());
            out.push(WAL_OP_TAG_PUT);
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&val_len.to_le_bytes());
            out.extend_from_slice(value);
            out.extend_from_slice(&fp_len.to_le_bytes());
            out.extend_from_slice(&fp_bytes);
        }
        WalOp::Delete { key, fp } => {
            let fp_bytes = fp.to_bytes();
            let fp_len = checked_u32_len(fp_bytes.len(), "Delete fingerprint")?;
            out.reserve(1 + 8 + 4 + fp_bytes.len());
            out.push(WAL_OP_TAG_DELETE);
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&fp_len.to_le_bytes());
            out.extend_from_slice(&fp_bytes);
        }
        WalOp::Adapt {
            old_fps,
            new_entries,
            range,
        } => encode_adapt_op(&mut out, old_fps, new_entries, *range)?,
        WalOp::Checkpoint { up_to_lsn } => {
            out.reserve(1 + 8);
            out.push(WAL_OP_TAG_CHECKPOINT);
            out.extend_from_slice(&up_to_lsn.to_le_bytes());
        }
    }
    Ok(out)
}

fn encode_adapt_op(
    out: &mut Vec<u8>,
    old_fps: &[Fingerprint],
    new_entries: &[(Fingerprint, Key)],
    range: QueryRange,
) -> AerisResult<()> {
    let old_count = checked_u32_len(old_fps.len(), "Adapt old_fps")?;
    let new_count = checked_u32_len(new_entries.len(), "Adapt new_entries")?;
    out.push(WAL_OP_TAG_ADAPT);
    out.extend_from_slice(&range.start().to_le_bytes());
    out.extend_from_slice(&range.end().to_le_bytes());
    out.extend_from_slice(&old_count.to_le_bytes());
    for fp in old_fps {
        write_length_prefixed_fp(out, fp)?;
    }
    out.extend_from_slice(&new_count.to_le_bytes());
    for (fp, key) in new_entries {
        write_length_prefixed_fp(out, fp)?;
        out.extend_from_slice(&key.to_le_bytes());
    }
    Ok(())
}

/// Decodes a [`WalOp`] previously encoded by [`encode_wal_op`].
///
/// Validates tag, field lengths, canonical [`Fingerprint`] padding invariants,
/// [`QueryRange`] ordering (`start <= end`), and absence of trailing garbage bytes.
///
/// # Errors
/// Returns [`AerisError::EncodingError`] on any malformed, truncated, or trailing input.
///
/// # Complexity
/// `O(bytes.len())` time and `O(bytes.len())` space.
pub fn decode_wal_op(bytes: &[u8]) -> AerisResult<WalOp> {
    let mut cursor = ByteCursor::new(bytes);
    let tag = cursor.read_u8()?;
    let op = match tag {
        WAL_OP_TAG_PUT => {
            let key = cursor.read_u64_le()?;
            let val_len = cursor.read_u32_le()? as usize;
            let value = cursor.read_slice(val_len)?.to_vec();
            let fp = read_length_prefixed_fp(&mut cursor)?;
            WalOp::Put { key, value, fp }
        }
        WAL_OP_TAG_DELETE => {
            let key = cursor.read_u64_le()?;
            let fp = read_length_prefixed_fp(&mut cursor)?;
            WalOp::Delete { key, fp }
        }
        WAL_OP_TAG_ADAPT => decode_adapt_op(&mut cursor)?,
        WAL_OP_TAG_CHECKPOINT => {
            let up_to_lsn = cursor.read_u64_le()?;
            WalOp::Checkpoint { up_to_lsn }
        }
        other => {
            return Err(AerisError::EncodingError(format!(
                "unknown WAL op tag 0x{other:02X}"
            )));
        }
    };
    if !cursor.is_empty() {
        return Err(AerisError::EncodingError(format!(
            "WAL op payload has {} trailing unconsumed bytes",
            cursor.remaining()
        )));
    }
    Ok(op)
}

fn decode_adapt_op(cursor: &mut ByteCursor<'_>) -> AerisResult<WalOp> {
    let start = cursor.read_u64_le()?;
    let end = cursor.read_u64_le()?;
    let range = QueryRange::new(start, end)
        .map_err(|err| AerisError::EncodingError(format!("invalid Adapt range: {err}")))?;

    let old_count = cursor.read_u32_le()? as usize;
    if old_count > cursor.remaining() / MIN_LENGTH_PREFIXED_FP_BYTES {
        return Err(AerisError::EncodingError(format!(
            "Adapt old_fps count {old_count} exceeds remaining {} bytes",
            cursor.remaining()
        )));
    }
    let mut old_fps = Vec::with_capacity(old_count);
    for _ in 0..old_count {
        old_fps.push(read_length_prefixed_fp(cursor)?);
    }

    let new_count = cursor.read_u32_le()? as usize;
    if new_count > cursor.remaining() / MIN_ADAPT_NEW_ENTRY_BYTES {
        return Err(AerisError::EncodingError(format!(
            "Adapt new_entries count {new_count} exceeds remaining {} bytes",
            cursor.remaining()
        )));
    }
    let mut new_entries = Vec::with_capacity(new_count);
    for _ in 0..new_count {
        let fp = read_length_prefixed_fp(cursor)?;
        let key = cursor.read_u64_le()?;
        new_entries.push((fp, key));
    }

    Ok(WalOp::Adapt {
        old_fps,
        new_entries,
        range,
    })
}

fn write_length_prefixed_fp(out: &mut Vec<u8>, fp: &Fingerprint) -> AerisResult<()> {
    let bytes = fp.to_bytes();
    let len = checked_u32_len(bytes.len(), "Fingerprint")?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&bytes);
    Ok(())
}

fn read_length_prefixed_fp(cursor: &mut ByteCursor<'_>) -> AerisResult<Fingerprint> {
    let len = cursor.read_u32_le()? as usize;
    let fp_bytes = cursor.read_slice(len)?;
    Fingerprint::from_bytes(fp_bytes)
}
