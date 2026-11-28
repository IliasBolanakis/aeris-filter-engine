//! Buffered binary trace I/O for `AERISKEY` and `AERISQRY` files.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// 8-byte magic prefix for binary key trace files.
pub const KEYS_MAGIC: [u8; 8] = *b"AERISKEY";

/// 8-byte magic prefix for binary query trace files.
pub const QUERIES_MAGIC: [u8; 8] = *b"AERISQRY";

/// Binary query format flags (bit 0 = includes explicit `expected` field).
pub const QUERIES_FLAGS: u64 = 1;

/// Sequential I/O buffer size (16 MiB) for disk-speed streaming up to 5e8 keys.
pub const IO_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Chunk size (in u64 keys) assembled in memory before writing to the buffered writer.
pub const KEY_WRITE_CHUNK_LEN: usize = 131_072; // 1 MiB

/// Ensures the parent directory of `path` exists before creating an output file.
pub fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Reads all keys from a `*.keys.bin` file and returns them sorted ascending for
/// `O(log N)` emptiness verification.
///
/// # Errors
/// Returns an error if the file cannot be opened, has invalid `AERISKEY` magic,
/// or ends prematurely.
pub fn read_and_sort_keys(path: &Path) -> Result<Vec<u64>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open key file '{}': {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);
    let count_usize = read_keys_header(&mut reader, path)?;

    let mut keys = Vec::with_capacity(count_usize);
    let mut buf = vec![0u8; KEY_WRITE_CHUNK_LEN * 8];
    let mut remaining = count_usize;

    while remaining > 0 {
        let batch = remaining.min(KEY_WRITE_CHUNK_LEN);
        let byte_len = batch * 8;
        reader
            .read_exact(&mut buf[..byte_len])
            .map_err(|e| format!("truncated key payload in '{}': {e}", path.display()))?;
        for chunk in buf[..byte_len].as_chunks::<8>().0 {
            keys.push(u64::from_le_bytes(*chunk));
        }
        remaining -= batch;
    }

    keys.sort_unstable();
    Ok(keys)
}

fn read_keys_header<R: Read>(reader: &mut R, path: &Path) -> Result<usize, String> {
    let mut header = [0u8; 16];
    reader.read_exact(&mut header).map_err(|e| {
        format!(
            "failed to read 16B key header from '{}': {e}",
            path.display()
        )
    })?;
    if header[0..8] != KEYS_MAGIC {
        return Err(format!(
            "invalid key file magic in '{}': expected {:?}, got {:?}",
            path.display(),
            KEYS_MAGIC,
            &header[0..8]
        ));
    }
    let count = u64::from_le_bytes(header[8..16].try_into().expect("8B count"));
    usize::try_from(count).map_err(|_| format!("key count {count} exceeds usize"))
}

/// Buffered 24-byte query record writer for `*.queries.bin`.
pub struct QueryStreamWriter {
    writer: BufWriter<File>,
    buf: Vec<u8>,
    written_records: u64,
}

impl QueryStreamWriter {
    const CHUNK_RECORDS: usize = 65_536; // 1.5 MiB

    /// Creates `out` and writes the 32-byte `AERISQRY` header.
    ///
    /// # Errors
    /// Returns an error if parent directories or the output file cannot be created,
    /// or if writing the header fails.
    pub fn create(out: &Path, count: u64, max_range_r: u64) -> Result<Self, String> {
        ensure_parent_dir(out).map_err(|e| format!("failed to create output directory: {e}"))?;
        let file = File::create(out)
            .map_err(|e| format!("failed to create query file '{}': {e}", out.display()))?;
        let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, file);

        writer
            .write_all(&QUERIES_MAGIC)
            .and_then(|()| writer.write_all(&count.to_le_bytes()))
            .and_then(|()| writer.write_all(&max_range_r.to_le_bytes()))
            .and_then(|()| writer.write_all(&QUERIES_FLAGS.to_le_bytes()))
            .map_err(|e| format!("failed to write query header: {e}"))?;

        Ok(Self {
            writer,
            buf: Vec::with_capacity(Self::CHUNK_RECORDS * 24),
            written_records: 0,
        })
    }

    /// Appends one `{ start, end, expected }` record (`start <= end`), flushing when buffered chunk fills.
    #[inline]
    pub fn push(&mut self, start: u64, end: u64, expected: u64) -> Result<(), String> {
        debug_assert!(start <= end);
        self.buf.extend_from_slice(&start.to_le_bytes());
        self.buf.extend_from_slice(&end.to_le_bytes());
        self.buf.extend_from_slice(&expected.to_le_bytes());
        self.written_records += 1;
        if self.buf.len() >= Self::CHUNK_RECORDS * 24 {
            self.flush_chunk()?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> Result<(), String> {
        if !self.buf.is_empty() {
            self.writer
                .write_all(&self.buf)
                .map_err(|e| format!("failed to write query records: {e}"))?;
            self.buf.clear();
        }
        Ok(())
    }

    /// Flushes remaining buffered records and returns total bytes written (`32 + 24 * count`).
    pub fn finish(mut self) -> Result<u64, String> {
        self.flush_chunk()?;
        self.writer
            .flush()
            .map_err(|e| format!("failed to flush query file: {e}"))?;
        Ok(32 + self.written_records * 24)
    }
}
