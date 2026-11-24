//! Pure binary codecs and state-folding algebra for the Storage Plane.
//!
//! This module contains zero disk I/O, zero clocks, and zero `unsafe` code.
//! It defines:
//! 1. The deterministic binary WAL frame header and [`WalOp`] payload codec
//!    protected by IEEE CRC32 (`crc32fast`).
//! 2. The write-optimized Reverse Map merge operand codec (`AddKey`, `DelKey`,
//!    `CompactedSnapshot`) and associative `full_merge` / `partial_merge`
//!    state-folding functions used by the RocksDB `MergeOperator`.

use aeris_types::{AerisError, AerisResult, Fingerprint, Key, Lsn, QueryRange, WalOp};
use std::collections::{BTreeMap, BTreeSet};

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

/// Binary tag for [`WalOp::Put`].
const WAL_OP_TAG_PUT: u8 = 0x01;
/// Binary tag for [`WalOp::Delete`].
const WAL_OP_TAG_DELETE: u8 = 0x02;
/// Binary tag for [`WalOp::Adapt`].
const WAL_OP_TAG_ADAPT: u8 = 0x03;
/// Binary tag for [`WalOp::Checkpoint`].
const WAL_OP_TAG_CHECKPOINT: u8 = 0x04;

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
        } => {
            let old_count = checked_u32_len(old_fps.len(), "Adapt old_fps")?;
            let new_count = checked_u32_len(new_entries.len(), "Adapt new_entries")?;
            out.push(WAL_OP_TAG_ADAPT);
            out.extend_from_slice(&range.start().to_le_bytes());
            out.extend_from_slice(&range.end().to_le_bytes());
            out.extend_from_slice(&old_count.to_le_bytes());
            for fp in old_fps {
                write_length_prefixed_fp(&mut out, fp)?;
            }
            out.extend_from_slice(&new_count.to_le_bytes());
            for (fp, key) in new_entries {
                write_length_prefixed_fp(&mut out, fp)?;
                out.extend_from_slice(&key.to_le_bytes());
            }
        }
        WalOp::Checkpoint { up_to_lsn } => {
            out.reserve(1 + 8);
            out.push(WAL_OP_TAG_CHECKPOINT);
            out.extend_from_slice(&up_to_lsn.to_le_bytes());
        }
    }
    Ok(out)
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

/// Encodes a Reverse Map `AddKey(key)` merge operand into a stack-allocated 9-byte array.
///
/// Wire layout: `[0x01][key: u64 LE]`.
///
/// # Complexity
/// `O(1)` time, zero heap allocations.
#[inline]
#[must_use]
pub fn encode_add_key_operand(key: Key) -> [u8; RM_SINGLE_KEY_OP_LEN] {
    let mut out = [0u8; RM_SINGLE_KEY_OP_LEN];
    out[0] = RM_TAG_ADD_KEY;
    out[1..9].copy_from_slice(&key.to_le_bytes());
    out
}

/// Encodes a Reverse Map `DelKey(key)` merge operand into a stack-allocated 9-byte array.
///
/// Wire layout: `[0x02][key: u64 LE]`.
///
/// # Complexity
/// `O(1)` time, zero heap allocations.
#[inline]
#[must_use]
pub fn encode_del_key_operand(key: Key) -> [u8; RM_SINGLE_KEY_OP_LEN] {
    let mut out = [0u8; RM_SINGLE_KEY_OP_LEN];
    out[0] = RM_TAG_DEL_KEY;
    out[1..9].copy_from_slice(&key.to_le_bytes());
    out
}

/// Encodes a sorted deduplicated set of keys into a canonical `CompactedSnapshot` value.
///
/// Wire layout: `[0x03][count: u32 LE][key_0: u64 LE]...[key_{n-1}: u64 LE]`.
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if `keys.len()` overflows `u32`.
///
/// # Complexity
/// `O(keys.len())` time and `O(keys.len())` space.
pub fn encode_compacted_snapshot(keys: &BTreeSet<Key>) -> AerisResult<Vec<u8>> {
    let count = checked_u32_len(keys.len(), "CompactedSnapshot")?;
    let mut out = Vec::with_capacity(RM_SNAPSHOT_HEADER_LEN + keys.len() * 8);
    out.push(RM_TAG_COMPACTED_SNAPSHOT);
    out.extend_from_slice(&count.to_le_bytes());
    for &key in keys {
        out.extend_from_slice(&key.to_le_bytes());
    }
    Ok(out)
}

/// Decodes a canonical `CompactedSnapshot` (or arbitrary valid Reverse Map operand stream)
/// into a sorted, deduplicated vector of keys.
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if the byte slice contains malformed tags
/// or truncated key fields.
///
/// # Complexity
/// `O(bytes.len())` time and `O(num_keys)` space.
pub fn decode_key_list(bytes: &[u8]) -> AerisResult<Vec<Key>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    // Fast path for the canonical single CompactedSnapshot written by full_merge.
    if bytes[0] == RM_TAG_COMPACTED_SNAPSHOT {
        if let Some(fast_keys) = try_decode_exact_snapshot(bytes)? {
            return Ok(fast_keys);
        }
    }
    let mut keys = BTreeSet::new();
    apply_operand_stream_to_set(bytes, &mut keys)?;
    Ok(keys.into_iter().collect())
}

/// Returns the number of live keys encoded in a canonical `CompactedSnapshot` (or
/// folded operand stream) without unnecessary overhead on the common snapshot path.
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if the value is malformed.
///
/// # Complexity
/// `O(1)` time for a canonical `CompactedSnapshot`; `O(bytes.len())` for raw operand streams.
pub fn count_keys_in_value(bytes: &[u8]) -> AerisResult<usize> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes[0] == RM_TAG_COMPACTED_SNAPSHOT && bytes.len() >= RM_SNAPSHOT_HEADER_LEN {
        let count = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        if bytes.len() == RM_SNAPSHOT_HEADER_LEN.saturating_add(count.saturating_mul(8)) {
            return Ok(count);
        }
    }
    Ok(decode_key_list(bytes)?.len())
}

/// Pure associative `full_merge` implementation for the Reverse Map `MergeOperator`.
///
/// Folds an optional existing base value (`CompactedSnapshot` or operand stream)
/// followed by all merge operands in chronological order into a [`BTreeSet<Key>`],
/// guaranteeing strict WAL replay idempotency, and emits a canonical
/// `CompactedSnapshot`.
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if `existing_value` or any operand in
/// `operands` is malformed.
///
/// # Complexity
/// `O(M log K)` where `M` is total encoded operations and `K` is the resulting key count.
pub fn fold_full_merge<'a, I>(existing_value: Option<&[u8]>, operands: I) -> AerisResult<Vec<u8>>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut keys = BTreeSet::new();
    if let Some(base) = existing_value {
        apply_operand_stream_to_set(base, &mut keys)?;
    }
    for operand in operands {
        apply_operand_stream_to_set(operand, &mut keys)?;
    }
    encode_compacted_snapshot(&keys)
}

/// Pure associative `partial_merge` implementation for the Reverse Map `MergeOperator`.
///
/// Combines a contiguous slice of operands without requiring the underlying base value:
/// - If any operand resets state via `CompactedSnapshot`, all prior deltas are superseded
///   and the folded result is emitted as a single canonical `CompactedSnapshot`.
/// - Otherwise (pure delta stream of `AddKey` / `DelKey`), redundant and cancelled
///   operations on the same key are collapsed to the latest net operation per key while
///   preserving exact associativity with any unknown underlying base state.
///
/// # Errors
/// Returns [`AerisError::EncodingError`] if any operand is malformed.
///
/// # Complexity
/// `O(M log D)` where `M` is total operations and `D` is distinct keys touched.
pub fn fold_partial_merge<'a, I>(operands: I) -> AerisResult<Vec<u8>>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    enum PartialState {
        Delta(BTreeMap<Key, bool>),
        Snapshot(BTreeSet<Key>),
    }

    let mut state = PartialState::Delta(BTreeMap::new());
    for operand in operands {
        let mut cursor = ByteCursor::new(operand);
        while !cursor.is_empty() {
            match cursor.read_u8()? {
                RM_TAG_ADD_KEY => {
                    let key = cursor.read_u64_le()?;
                    match &mut state {
                        PartialState::Delta(deltas) => {
                            deltas.insert(key, true);
                        }
                        PartialState::Snapshot(set) => {
                            set.insert(key);
                        }
                    }
                }
                RM_TAG_DEL_KEY => {
                    let key = cursor.read_u64_le()?;
                    match &mut state {
                        PartialState::Delta(deltas) => {
                            deltas.insert(key, false);
                        }
                        PartialState::Snapshot(set) => {
                            set.remove(&key);
                        }
                    }
                }
                RM_TAG_COMPACTED_SNAPSHOT => {
                    let count = cursor.read_u32_le()? as usize;
                    let keys = read_key_array(&mut cursor, count)?;
                    state = PartialState::Snapshot(keys.into_iter().collect());
                }
                other => {
                    return Err(AerisError::EncodingError(format!(
                        "unknown Reverse Map merge operand tag 0x{other:02X}"
                    )));
                }
            }
        }
    }

    match state {
        PartialState::Snapshot(set) => encode_compacted_snapshot(&set),
        PartialState::Delta(deltas) => {
            let mut out = Vec::with_capacity(deltas.len() * RM_SINGLE_KEY_OP_LEN);
            for (key, is_add) in deltas {
                let encoded = if is_add {
                    encode_add_key_operand(key)
                } else {
                    encode_del_key_operand(key)
                };
                out.extend_from_slice(&encoded);
            }
            Ok(out)
        }
    }
}

/// Applies all self-delimiting Reverse Map operands in `bytes` chronologically into `keys`.
fn apply_operand_stream_to_set(bytes: &[u8], keys: &mut BTreeSet<Key>) -> AerisResult<()> {
    let mut cursor = ByteCursor::new(bytes);
    while !cursor.is_empty() {
        match cursor.read_u8()? {
            RM_TAG_ADD_KEY => {
                let key = cursor.read_u64_le()?;
                keys.insert(key);
            }
            RM_TAG_DEL_KEY => {
                let key = cursor.read_u64_le()?;
                keys.remove(&key);
            }
            RM_TAG_COMPACTED_SNAPSHOT => {
                let count = cursor.read_u32_le()? as usize;
                let snapshot_keys = read_key_array(&mut cursor, count)?;
                keys.clear();
                keys.extend(snapshot_keys);
            }
            other => {
                return Err(AerisError::EncodingError(format!(
                    "unknown Reverse Map merge operand tag 0x{other:02X}"
                )));
            }
        }
    }
    Ok(())
}

/// Attempts fast-path decoding when `bytes` is a single canonical `CompactedSnapshot`.
/// Returns `Ok(None)` if trailing operands follow the snapshot (requiring full stream fold).
fn try_decode_exact_snapshot(bytes: &[u8]) -> AerisResult<Option<Vec<Key>>> {
    if bytes.len() < RM_SNAPSHOT_HEADER_LEN {
        return Err(AerisError::EncodingError(
            "truncated CompactedSnapshot header".to_string(),
        ));
    }
    let count = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
    let expected_len = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(RM_SNAPSHOT_HEADER_LEN))
        .ok_or_else(|| {
            AerisError::EncodingError("CompactedSnapshot length overflow".to_string())
        })?;
    if bytes.len() != expected_len {
        return Ok(None);
    }
    let mut keys = Vec::with_capacity(count);
    let mut offset = RM_SNAPSHOT_HEADER_LEN;
    for _ in 0..count {
        let k = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        keys.push(k);
        offset += 8;
    }
    Ok(Some(keys))
}

fn read_key_array(cursor: &mut ByteCursor<'_>, count: usize) -> AerisResult<Vec<Key>> {
    let needed = count
        .checked_mul(8)
        .ok_or_else(|| AerisError::EncodingError("key array byte count overflow".to_string()))?;
    if cursor.remaining() < needed {
        return Err(AerisError::EncodingError(format!(
            "truncated key array: need {needed} bytes for {count} keys, only {} remaining",
            cursor.remaining()
        )));
    }
    let mut keys = Vec::with_capacity(count);
    for _ in 0..count {
        keys.push(cursor.read_u64_le()?);
    }
    Ok(keys)
}

fn decode_adapt_op(cursor: &mut ByteCursor<'_>) -> AerisResult<WalOp> {
    let start = cursor.read_u64_le()?;
    let end = cursor.read_u64_le()?;
    let range = QueryRange::new(start, end)
        .map_err(|err| AerisError::EncodingError(format!("invalid Adapt range: {err}")))?;

    let old_count = cursor.read_u32_le()? as usize;
    // Each encoded fingerprint needs at least 4 (len) + 2 (bit_len) = 6 bytes.
    if old_count > cursor.remaining() / 6 {
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
    // Each new entry needs at least 6 (fp) + 8 (key) = 14 bytes.
    if new_count > cursor.remaining() / 14 {
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

fn checked_u32_len(len: usize, label: &str) -> AerisResult<u32> {
    u32::try_from(len)
        .map_err(|_| AerisError::EncodingError(format!("{label} length {len} overflows u32")))
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

/// Bounds-checked little-endian binary reader with zero allocations.
struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    #[inline]
    const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    const fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    #[inline]
    fn read_slice(&mut self, len: usize) -> AerisResult<&'a [u8]> {
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
    fn read_u8(&mut self) -> AerisResult<u8> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    #[inline]
    fn read_u32_le(&mut self) -> AerisResult<u32> {
        let s = self.read_slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    #[inline]
    fn read_u64_le(&mut self) -> AerisResult<u64> {
        let s = self.read_slice(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeris_types::PartitionHasher;

    #[test]
    fn wal_op_round_trip_all_variants() {
        let hasher = PartitionHasher::new(99);
        let fp_short = Fingerprint::from_hasher(&hasher, 17, 22);
        let fp_long = Fingerprint::from_hasher(&hasher, 42, 165);
        let fp_empty = Fingerprint::empty();

        let ops = [
            WalOp::Put {
                key: 0,
                value: Vec::new(),
                fp: fp_empty.clone(),
            },
            WalOp::Put {
                key: u64::MAX,
                value: b"aeris-storage-payload-v1".to_vec(),
                fp: fp_long.clone(),
            },
            WalOp::Delete {
                key: 123_456_789,
                fp: fp_short.clone(),
            },
            WalOp::Adapt {
                old_fps: vec![fp_short.clone(), fp_empty],
                new_entries: vec![(fp_long, 555), (fp_short, 777)],
                range: QueryRange::new(100, 999).unwrap(),
            },
            WalOp::Checkpoint { up_to_lsn: 42_000 },
        ];

        for (idx, op) in ops.iter().enumerate() {
            let encoded = encode_wal_op(op).unwrap();
            let decoded = decode_wal_op(&encoded).unwrap();
            assert_eq!(&decoded, op);

            let lsn = (idx as u64) + 1;
            let (frame, crc) = encode_wal_frame(lsn, op).unwrap();
            let header = WalFrameHeader::decode(&frame).unwrap();
            assert_eq!(header.version, WAL_FORMAT_VERSION);
            assert_eq!(header.lsn, lsn);
            assert_eq!(header.payload_len as usize, encoded.len());
            assert_eq!(header.crc32, crc);
            let recomputed = compute_frame_crc32(
                header.version,
                header.payload_len,
                header.lsn,
                &frame[WAL_HEADER_LEN..],
            );
            assert_eq!(recomputed, crc);
        }
    }

    #[test]
    fn wal_op_decode_rejects_malformed_inputs() {
        // Empty payload
        assert!(decode_wal_op(&[]).is_err());
        // Unknown tag
        assert!(decode_wal_op(&[0xFF]).is_err());
        // Trailing bytes on Checkpoint
        let mut cp = encode_wal_op(&WalOp::Checkpoint { up_to_lsn: 7 }).unwrap();
        cp.push(0x00);
        assert!(decode_wal_op(&cp).is_err());
        // Truncated Put
        let hasher = PartitionHasher::new(1);
        let fp = Fingerprint::from_hasher(&hasher, 1, 20);
        let put = encode_wal_op(&WalOp::Put {
            key: 1,
            value: vec![1, 2, 3],
            fp,
        })
        .unwrap();
        assert!(decode_wal_op(&put[..put.len() - 1]).is_err());
        // Invalid Adapt range (start > end)
        let mut bad_adapt = vec![WAL_OP_TAG_ADAPT];
        bad_adapt.extend_from_slice(&50u64.to_le_bytes());
        bad_adapt.extend_from_slice(&10u64.to_le_bytes());
        bad_adapt.extend_from_slice(&0u32.to_le_bytes());
        bad_adapt.extend_from_slice(&0u32.to_le_bytes());
        assert!(decode_wal_op(&bad_adapt).is_err());
        // LSN 0 rejected on frame encode
        assert!(encode_wal_frame(0, &WalOp::Checkpoint { up_to_lsn: 1 }).is_err());
    }

    #[test]
    fn wal_header_rejects_bad_magic_version_and_truncation() {
        assert!(WalFrameHeader::decode(&[0u8; WAL_HEADER_LEN - 1]).is_err());

        let (mut frame, _) = encode_wal_frame(1, &WalOp::Checkpoint { up_to_lsn: 0 }).unwrap();
        frame[0] = b'X';
        assert!(WalFrameHeader::decode(&frame).is_err());

        frame[0] = b'A';
        frame[4] = 99;
        assert!(WalFrameHeader::decode(&frame).is_err());
    }

    #[test]
    fn reverse_map_merge_algebra_is_idempotent_and_associative() {
        let a10 = encode_add_key_operand(10);
        let a20 = encode_add_key_operand(20);
        let a30 = encode_add_key_operand(30);
        let d10 = encode_del_key_operand(10);
        let d99 = encode_del_key_operand(99);

        // Full merge without base
        let merged = fold_full_merge(
            None,
            [
                a20.as_slice(),
                a10.as_slice(),
                a10.as_slice(), // duplicate add -> idempotent
                d99.as_slice(), // delete absent key -> no-op
                a30.as_slice(),
                d10.as_slice(),
            ],
        )
        .unwrap();
        assert_eq!(decode_key_list(&merged).unwrap(), vec![20, 30]);
        assert_eq!(count_keys_in_value(&merged).unwrap(), 2);

        // Associativity check: partial_merge([d10, a30, a10, d20]) applied to base {10, 20}
        let mut base_set = BTreeSet::new();
        base_set.insert(10);
        base_set.insert(20);
        let base_snap = encode_compacted_snapshot(&base_set).unwrap();

        let d20 = encode_del_key_operand(20);
        let ops = [
            d10.as_slice(),
            a30.as_slice(),
            a10.as_slice(),
            d20.as_slice(),
        ];

        let direct = fold_full_merge(Some(&base_snap), ops).unwrap();
        let partial = fold_partial_merge(ops).unwrap();
        let via_partial = fold_full_merge(Some(&base_snap), [partial.as_slice()]).unwrap();
        assert_eq!(direct, via_partial);
        assert_eq!(decode_key_list(&via_partial).unwrap(), vec![10, 30]);

        // Snapshot inside partial_merge supersedes earlier deltas
        let snap_partial =
            fold_partial_merge([a30.as_slice(), base_snap.as_slice(), d10.as_slice()]).unwrap();
        assert_eq!(snap_partial[0], RM_TAG_COMPACTED_SNAPSHOT);
        assert_eq!(decode_key_list(&snap_partial).unwrap(), vec![20]);
    }

    #[test]
    fn reverse_map_codec_rejects_corrupt_operands() {
        assert!(fold_full_merge(None, [&[0x99u8][..]]).is_err());
        assert!(fold_full_merge(None, [&[RM_TAG_ADD_KEY, 1, 2][..]]).is_err());
        assert!(fold_partial_merge([&[RM_TAG_DEL_KEY, 0][..]]).is_err());
        // Truncated CompactedSnapshot claiming 5 keys with 0 bytes
        let bad_snap = [RM_TAG_COMPACTED_SNAPSHOT, 5, 0, 0, 0];
        assert!(decode_key_list(&bad_snap).is_err());
        assert!(fold_partial_merge([bad_snap.as_slice()]).is_err());

        // Empty boundary checks: empty slice, empty CompactedSnapshot, empty partial_merge
        assert!(decode_key_list(&[]).unwrap().is_empty());
        assert_eq!(count_keys_in_value(&[]).unwrap(), 0);
        let empty_snap = encode_compacted_snapshot(&BTreeSet::new()).unwrap();
        assert_eq!(empty_snap, vec![RM_TAG_COMPACTED_SNAPSHOT, 0, 0, 0, 0]);
        assert!(decode_key_list(&empty_snap).unwrap().is_empty());
        assert_eq!(count_keys_in_value(&empty_snap).unwrap(), 0);
        let empty_partial = fold_partial_merge(std::iter::empty::<&[u8]>()).unwrap();
        assert!(empty_partial.is_empty());
    }
}
