//! Write-optimized Reverse Map merge operand codec and associative folding algebra.
//!
//! Defines `AddKey`, `DelKey`, and `CompactedSnapshot` binary encodings along with
//! the pure `fold_full_merge` and `fold_partial_merge` functions backing the
//! RocksDB `MergeOperator`.

use super::{
    checked_u32_len, ByteCursor, RM_SINGLE_KEY_OP_LEN, RM_SNAPSHOT_HEADER_LEN, RM_TAG_ADD_KEY,
    RM_TAG_COMPACTED_SNAPSHOT, RM_TAG_DEL_KEY,
};
use aeris_types::{AerisError, AerisResult, Key};
use std::collections::{BTreeMap, BTreeSet};

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

/// Intermediate state for associative [`fold_partial_merge`].
enum PartialMergeState {
    Delta(BTreeMap<Key, bool>),
    Snapshot(BTreeSet<Key>),
}

impl PartialMergeState {
    fn apply_operand_stream(&mut self, operand: &[u8]) -> AerisResult<()> {
        let mut cursor = ByteCursor::new(operand);
        while !cursor.is_empty() {
            match cursor.read_u8()? {
                RM_TAG_ADD_KEY => self.record_delta(cursor.read_u64_le()?, true),
                RM_TAG_DEL_KEY => self.record_delta(cursor.read_u64_le()?, false),
                RM_TAG_COMPACTED_SNAPSHOT => {
                    let count = cursor.read_u32_le()? as usize;
                    let keys = read_key_array(&mut cursor, count)?;
                    *self = Self::Snapshot(keys.into_iter().collect());
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

    fn record_delta(&mut self, key: Key, is_add: bool) {
        match self {
            Self::Delta(deltas) => {
                deltas.insert(key, is_add);
            }
            Self::Snapshot(set) => {
                if is_add {
                    set.insert(key);
                } else {
                    set.remove(&key);
                }
            }
        }
    }

    fn into_encoded(self) -> AerisResult<Vec<u8>> {
        match self {
            Self::Snapshot(set) => encode_compacted_snapshot(&set),
            Self::Delta(deltas) => {
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
    let mut state = PartialMergeState::Delta(BTreeMap::new());
    for operand in operands {
        state.apply_operand_stream(operand)?;
    }
    state.into_encoded()
}

/// Applies all self-delimiting Reverse Map operands in `bytes` chronologically into `keys`.
fn apply_operand_stream_to_set(bytes: &[u8], keys: &mut BTreeSet<Key>) -> AerisResult<()> {
    let mut cursor = ByteCursor::new(bytes);
    while !cursor.is_empty() {
        match cursor.read_u8()? {
            RM_TAG_ADD_KEY => {
                keys.insert(cursor.read_u64_le()?);
            }
            RM_TAG_DEL_KEY => {
                keys.remove(&cursor.read_u64_le()?);
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
pub(super) fn try_decode_exact_snapshot(bytes: &[u8]) -> AerisResult<Option<Vec<Key>>> {
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
