//! Streaming, deduplicating iterator over live keys stored in [`super::RocksDbReverseMap`].

use super::HashSet;
use crate::codec::decode_key_list;
use aeris_types::{AerisError, AerisResult, Key};
use rocksdb::{DBIteratorWithThreadMode, DB};

/// Streaming, deduplicating iterator over live keys stored in [`super::RocksDbReverseMap`].
///
/// Decodes one RocksDB fingerprint entry at a time rather than materializing the
/// entire database upfront, making it suitable for I/O-throttled background rebuilds.
pub(super) struct ReverseMapKeyIter<'a> {
    db_iter: DBIteratorWithThreadMode<'a, DB>,
    current_batch: std::vec::IntoIter<Key>,
    seen: HashSet<Key>,
    errored: bool,
}

impl<'a> ReverseMapKeyIter<'a> {
    pub(super) fn new(db_iter: DBIteratorWithThreadMode<'a, DB>) -> Self {
        Self {
            db_iter,
            current_batch: Vec::new().into_iter(),
            seen: HashSet::new(),
            errored: false,
        }
    }
}

impl Iterator for ReverseMapKeyIter<'_> {
    type Item = AerisResult<Key>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.errored {
            return None;
        }
        loop {
            for key in self.current_batch.by_ref() {
                if self.seen.insert(key) {
                    return Some(Ok(key));
                }
            }
            let next_entry = self.db_iter.next()?;
            match next_entry {
                Ok((_raw_fp, raw_val)) => match decode_key_list(&raw_val) {
                    Ok(keys) => {
                        self.current_batch = keys.into_iter();
                    }
                    Err(err) => {
                        self.errored = true;
                        return Some(Err(err));
                    }
                },
                Err(err) => {
                    self.errored = true;
                    return Some(Err(AerisError::StorageError(format!(
                        "Reverse Map streaming iteration error: {err}"
                    ))));
                }
            }
        }
    }
}
