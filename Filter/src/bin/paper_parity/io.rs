//! Binary trace readers for `*.keys.bin` (`AERISKEY`) and `*.queries.bin` (`AERISQRY`).
//!
//! Reads the shared little-endian binary trace formats produced by `trace_gen`
//! using 16 MiB buffered sequential I/O.

use aeris_types::{Key, QueryRange};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// 8-byte magic prefix for binary key trace files.
pub const KEYS_MAGIC: [u8; 8] = *b"AERISKEY";

/// 8-byte magic prefix for binary query trace files.
pub const QUERIES_MAGIC: [u8; 8] = *b"AERISQRY";

/// Sequential I/O read buffer size (16 MiB).
const IO_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Chunk size (in 8-byte keys) for streaming key reads.
const KEY_READ_CHUNK_LEN: usize = 131_072; // 1 MiB

/// Chunk size (in 24-byte query records) for streaming query reads.
const QUERY_READ_CHUNK_LEN: usize = 65_536; // 1.5 MiB

/// Decoded binary query trace record (`start <= end`).
#[derive(Clone, Copy, Debug)]
pub struct QueryRecord {
    /// Closed query interval `[start, end]`.
    pub range: QueryRange,
    /// Ground-truth expected match count (`0` for verified-empty queries).
    pub expected: u64,
}

/// Reads all little-endian 64-bit keys from a `*.keys.bin` file.
///
/// # Format
/// - 16-byte header: `b"AERISKEY"` (8B) + `count: u64 LE` (8B).
/// - Payload: `count` little-endian `u64` keys.
///
/// # Errors
/// Returns an error on I/O failure, magic mismatch, `usize` overflow, or truncated payload.
pub fn read_keys_file(path: &Path) -> Result<Vec<Key>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open keys file '{}': {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);
    let count_usize = read_keys_header(&mut reader, path)?;

    let mut keys = Vec::with_capacity(count_usize);
    let mut buf = vec![0u8; KEY_READ_CHUNK_LEN * 8];
    let mut remaining = count_usize;

    while remaining > 0 {
        let batch = remaining.min(KEY_READ_CHUNK_LEN);
        let byte_len = batch * 8;
        reader
            .read_exact(&mut buf[..byte_len])
            .map_err(|e| format!("truncated keys payload: {e}"))?;
        for c in buf[..byte_len].as_chunks::<8>().0 {
            keys.push(u64::from_le_bytes(*c));
        }
        remaining -= batch;
    }

    Ok(keys)
}

fn read_keys_header<R: Read>(reader: &mut R, path: &Path) -> Result<usize, String> {
    let mut header = [0u8; 16];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("failed to read 16B keys header: {e}"))?;
    if header[0..8] != KEYS_MAGIC {
        return Err(format!(
            "invalid keys magic in '{}': expected {:?}, got {:?}",
            path.display(),
            KEYS_MAGIC,
            &header[0..8]
        ));
    }
    let count = u64::from_le_bytes(header[8..16].try_into().expect("8B count"));
    usize::try_from(count).map_err(|_| format!("key count {count} exceeds usize"))
}

/// Reads all 24-byte query records from a `*.queries.bin` file.
///
/// # Format
/// - 32-byte header: `b"AERISQRY"` (8B) + `count: u64 LE` (8B)
///   + `max_range_r: u64 LE` (8B) + `flags: u64 LE` (8B).
/// - Payload: `count` records of `{ start: u64 LE, end: u64 LE, expected: u64 LE }`.
///
/// # Errors
/// Returns an error on I/O failure, magic mismatch, truncated payload, or invalid `[start, end]`.
pub fn read_queries_file(path: &Path) -> Result<(Vec<QueryRecord>, u64), String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open queries file '{}': {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);
    let (count_usize, max_range_r) = read_queries_header(&mut reader, path)?;

    let mut queries = Vec::with_capacity(count_usize);
    let mut buf = vec![0u8; QUERY_READ_CHUNK_LEN * 24];
    let mut remaining = count_usize;

    while remaining > 0 {
        let batch = remaining.min(QUERY_READ_CHUNK_LEN);
        let byte_len = batch * 24;
        reader
            .read_exact(&mut buf[..byte_len])
            .map_err(|e| format!("truncated queries payload: {e}"))?;
        decode_query_chunk(&buf[..byte_len], &mut queries)?;
        remaining -= batch;
    }

    Ok((queries, max_range_r))
}

fn read_queries_header<R: Read>(reader: &mut R, path: &Path) -> Result<(usize, u64), String> {
    let mut header = [0u8; 32];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("failed to read 32B queries header: {e}"))?;
    if header[0..8] != QUERIES_MAGIC {
        return Err(format!(
            "invalid queries magic in '{}': expected {:?}, got {:?}",
            path.display(),
            QUERIES_MAGIC,
            &header[0..8]
        ));
    }
    let count = u64::from_le_bytes(header[8..16].try_into().expect("8B count"));
    let max_range_r = u64::from_le_bytes(header[16..24].try_into().expect("8B max_range_r"));
    let _flags = u64::from_le_bytes(header[24..32].try_into().expect("8B flags"));

    let count_usize =
        usize::try_from(count).map_err(|_| format!("query count {count} exceeds usize"))?;
    Ok((count_usize, max_range_r))
}

fn decode_query_chunk(chunk: &[u8], out: &mut Vec<QueryRecord>) -> Result<(), String> {
    for c in chunk.as_chunks::<24>().0 {
        let start = u64::from_le_bytes(c[0..8].try_into().expect("8B start"));
        let end = u64::from_le_bytes(c[8..16].try_into().expect("8B end"));
        let expected = u64::from_le_bytes(c[16..24].try_into().expect("8B expected"));
        let range = QueryRange::new(start, end)
            .map_err(|e| format!("invalid query range [{start}, {end}]: {e}"))?;
        out.push(QueryRecord { range, expected });
    }
    Ok(())
}
