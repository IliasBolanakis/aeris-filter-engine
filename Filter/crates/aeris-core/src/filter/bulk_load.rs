//! Linear-time LSD radix bulk loading and key-iterator rebuilds for [`AerisFilter`].

use super::radix::radix_sort_u64;
use super::AerisFilter;
use crate::keepsake::{encode_unextended_fresh_box_into, KeepsakeGeometry};
use aeris_types::{
    AerisError, AerisResult, FilterConfig, Fingerprint, KeepsakeSlotKind, Key, Memento,
    PartitionHasher, RangeFilter, ReverseMapStore,
};
use std::time::Instant;

/// Maximum `(Fingerprint, Key)` entries buffered per streaming Reverse Map batch
/// during [`AerisFilter::bulk_load`] (`~3 MB` constant peak memory).
const REVERSE_MAP_STREAM_CHUNK: usize = 65_536;

impl AerisFilter {
    /// Constructs an [`AerisFilter`] via linear-time `O(K + N)` 8-bit LSD radix
    /// sorting and single-pass streaming AoS block emission, persisting all
    /// `(fingerprint, key)` associations to `reverse_map` in bounded-memory chunks.
    ///
    /// # Errors
    /// - Returns [`AerisError::InvalidConfig`] if `config` is invalid.
    /// - Returns [`AerisError::FilterFull`] if the encoded keepsake boxes exceed
    ///   `config.max_load_factor` (before allocating the table or mutating `reverse_map`).
    /// - Returns [`AerisError::StorageError`] if `reverse_map.append_batch` fails.
    ///
    /// # Complexity
    /// `O(K * ceil((q + r + m) / 8) + N)` time (`K = keys.len()`, `N = 2^q`),
    /// peak auxiliary memory `16 B/key` during radix sort (dropped before table allocation),
    /// and constant `~3 MB` Reverse Map streaming buffer (`65,536` entries/batch).
    pub fn bulk_load(
        config: FilterConfig,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let start = Instant::now();
        config.validate()?;
        if keys.is_empty() {
            return Self::new(config);
        }

        let total_bits = (config.q as u16) + (config.r as u16) + (config.m as u16);
        let filter = if total_bits <= 64 {
            Self::bulk_load_radix_streaming(config, keys, reverse_map)?
        } else {
            Self::bulk_load_wide_fallback(config, keys, reverse_map)?
        };

        tracing::debug!(
            keys = keys.len(),
            occupied_slots = filter.occupied_slots,
            load_factor = filter.load_factor(),
            latency_ns = start.elapsed().as_nanos() as u64,
            "completed bulk_load"
        );
        Ok(filter)
    }

    /// Rebuilds an [`AerisFilter`] from an arbitrary key iterator under `config`,
    /// populating `reverse_map` with the rebuilt `(fingerprint, key)` mappings.
    ///
    /// # Errors
    /// Propagates validation, capacity (`FilterFull`), or storage errors from [`Self::bulk_load`].
    pub fn rebuild_from_keys(
        config: FilterConfig,
        keys: impl IntoIterator<Item = Key>,
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let collected: Vec<Key> = keys.into_iter().collect();
        Self::bulk_load(config, &collected, reverse_map)
    }

    fn bulk_load_radix_streaming(
        config: FilterConfig,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let hasher = PartitionHasher::new(config.hash_seed);
        let geom = KeepsakeGeometry::from_config(&config);

        let mut packed = pack_keys_u64(&config, &hasher, keys);
        let total_bits = (config.q as usize) + (config.r as usize) + (config.m as usize);
        radix_sort_u64(&mut packed, total_bits);

        let total_slots = count_packed_run_slots(&packed, &geom);
        ensure_bulk_load_capacity(&config, keys.len(), total_slots)?;

        let mut filter = Self::new(config)?;
        emit_packed_runs_to_table(&mut filter, &packed);
        filter.occupied_slots = total_slots;
        filter.num_keys = keys.len();
        drop(packed);

        if !reverse_map.is_noop() {
            stream_reverse_map_fast(&filter, keys, reverse_map)?;
        }
        Ok(filter)
    }

    fn bulk_load_wide_fallback(
        config: FilterConfig,
        keys: &[Key],
        reverse_map: &dyn ReverseMapStore,
    ) -> AerisResult<Self> {
        let hasher = PartitionHasher::new(config.hash_seed);
        let geom = KeepsakeGeometry::from_config(&config);

        let mut projected = project_wide_keys(&config, &hasher, keys);
        projected.sort_unstable();

        let total_slots = count_projected_run_slots(&projected, &geom);
        ensure_bulk_load_capacity(&config, keys.len(), total_slots)?;

        let mut filter = Self::new(config)?;
        emit_projected_runs_to_table(&mut filter, &projected);
        filter.occupied_slots = total_slots;
        filter.num_keys = keys.len();
        drop(projected);

        if !reverse_map.is_noop() {
            stream_reverse_map_wide(&filter, keys, reverse_map)?;
        }
        Ok(filter)
    }
}

fn ensure_bulk_load_capacity(
    config: &FilterConfig,
    num_keys: usize,
    total_slots: usize,
) -> AerisResult<()> {
    let capacity = config.num_slots();
    let projected_alpha = (total_slots as f64) / (capacity as f64);
    if projected_alpha > config.max_load_factor {
        tracing::warn!(
            projected_alpha,
            max_load_factor = config.max_load_factor,
            capacity,
            keys = num_keys,
            "bulk_load rejected: projected load factor exceeds ceiling"
        );
        return Err(AerisError::FilterFull {
            load_factor: projected_alpha,
            capacity,
        });
    }
    Ok(())
}

fn pack_keys_u64(config: &FilterConfig, hasher: &PartitionHasher, keys: &[Key]) -> Vec<u64> {
    let rem_shift = 64 - ((config.q + config.r) as u32);
    let m_mask = config.memento_mask();
    let m_shift = config.m as u32;
    let mut packed = Vec::with_capacity(keys.len());
    if config.m > 0 {
        packed.extend(keys.iter().map(|&key| {
            let p = key >> m_shift;
            let mem = key & m_mask;
            let w0 = hasher.word(p, 0);
            ((w0 >> rem_shift) << m_shift) | mem
        }));
    } else {
        packed.extend(keys.iter().map(|&key| hasher.word(key, 0) >> rem_shift));
    }
    packed
}

fn count_packed_run_slots(packed: &[u64], geom: &KeepsakeGeometry) -> usize {
    let m_shift = geom.m as u32;
    let mut total_slots = 0usize;
    let mut i = 0usize;
    while i < packed.len() {
        let hr = packed[i] >> m_shift;
        let group_start = i;
        i += 1;
        while i < packed.len() && (packed[i] >> m_shift) == hr {
            i += 1;
        }
        let cnt = i - group_start;
        total_slots += if cnt == 1 {
            1
        } else {
            geom.unextended_box_slot_len(cnt)
        };
    }
    total_slots
}

fn emit_packed_runs_to_table(filter: &mut AerisFilter, packed: &[u64]) {
    let r = filter.config.r;
    let m = filter.config.m;
    let rem_mask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
    let m_mask = filter.config.memento_mask();
    let m_shift = m as u32;
    let rm_shift = (r + m) as u32;
    let fresh_sentinel = 1u64 << r;

    let mut run_buf: Vec<(u64, KeepsakeSlotKind)> = Vec::with_capacity(16);
    let mut mem_buf: Vec<Memento> = Vec::with_capacity(16);
    let mut writer = filter.table.bulk_writer();

    let mut i = 0usize;
    while i < packed.len() {
        let first = packed[i];
        let home = (first >> rm_shift) as usize;

        if i + 1 == packed.len() || ((packed[i + 1] >> rm_shift) as usize) != home {
            let rem = (first >> m_shift) & rem_mask;
            let high = fresh_sentinel | rem;
            let slot0 = if m > 0 {
                (high << m_shift) | (first & m_mask)
            } else {
                high
            };
            writer.append_single_slot_run(home, slot0);
            i += 1;
            continue;
        }

        run_buf.clear();
        while i < packed.len() && ((packed[i] >> rm_shift) as usize) == home {
            let hr = packed[i] >> m_shift;
            let rem = hr & rem_mask;
            let box_start = i;
            if m > 0 {
                mem_buf.clear();
                while i < packed.len() && (packed[i] >> m_shift) == hr {
                    mem_buf.push(packed[i] & m_mask);
                    i += 1;
                }
            } else {
                while i < packed.len() && (packed[i] >> m_shift) == hr {
                    i += 1;
                }
            }
            let count = i - box_start;
            let is_run_end = i == packed.len() || ((packed[i] >> rm_shift) as usize) != home;
            encode_unextended_fresh_box_into(
                &filter.geom,
                rem,
                if m > 0 { &mem_buf } else { &[] },
                count,
                is_run_end,
                &mut run_buf,
            );
        }
        writer.append_run(home, &run_buf);
    }
    writer.finish();
}

fn stream_reverse_map_fast(
    filter: &AerisFilter,
    keys: &[Key],
    reverse_map: &dyn ReverseMapStore,
) -> AerisResult<()> {
    let q_plus_r = filter.config.q + filter.config.r;
    let mut rm_chunk = Vec::with_capacity(REVERSE_MAP_STREAM_CHUNK.min(keys.len()));
    for chunk in keys.chunks(REVERSE_MAP_STREAM_CHUNK) {
        rm_chunk.clear();
        for &key in chunk {
            let p = filter.config.partition_of(key);
            let w0 = filter.hasher.word(p, 0);
            rm_chunk.push((Fingerprint::from_u64_prefix(w0, q_plus_r), key));
        }
        reverse_map.append_batch(&rm_chunk)?;
    }
    Ok(())
}

fn project_wide_keys(
    config: &FilterConfig,
    hasher: &PartitionHasher,
    keys: &[Key],
) -> Vec<(usize, u64, Memento)> {
    let mut projected = Vec::with_capacity(keys.len());
    for &key in keys {
        let p = config.partition_of(key);
        let mem = config.memento_of(key);
        let (home_u64, rem) = hasher.quotient_and_remainder(p, config.q, config.r);
        projected.push((home_u64 as usize, rem, mem));
    }
    projected
}

fn count_projected_run_slots(
    projected: &[(usize, u64, Memento)],
    geom: &KeepsakeGeometry,
) -> usize {
    let mut total_slots = 0usize;
    let mut i = 0usize;
    while i < projected.len() {
        let (home, rem, _) = projected[i];
        let start_i = i;
        i += 1;
        while i < projected.len() && projected[i].0 == home && projected[i].1 == rem {
            i += 1;
        }
        total_slots += geom.unextended_box_slot_len(i - start_i);
    }
    total_slots
}

fn emit_projected_runs_to_table(filter: &mut AerisFilter, projected: &[(usize, u64, Memento)]) {
    let m = filter.config.m;
    let mut run_buf: Vec<(u64, KeepsakeSlotKind)> = Vec::with_capacity(16);
    let mut mem_buf: Vec<Memento> = Vec::with_capacity(16);
    let mut write_cursor = 0usize;

    let mut i = 0usize;
    while i < projected.len() {
        let home = projected[i].0;
        run_buf.clear();

        while i < projected.len() && projected[i].0 == home {
            let rem = projected[i].1;
            let box_start = i;
            if m > 0 {
                mem_buf.clear();
                while i < projected.len() && projected[i].0 == home && projected[i].1 == rem {
                    mem_buf.push(projected[i].2);
                    i += 1;
                }
            } else {
                while i < projected.len() && projected[i].0 == home && projected[i].1 == rem {
                    i += 1;
                }
            }
            let count = i - box_start;
            let is_run_end = i == projected.len() || projected[i].0 != home;
            encode_unextended_fresh_box_into(
                &filter.geom,
                rem,
                if m > 0 { &mem_buf } else { &[] },
                count,
                is_run_end,
                &mut run_buf,
            );
        }

        filter
            .table
            .bulk_append_run(home, &run_buf, &mut write_cursor);
    }
}

fn stream_reverse_map_wide(
    filter: &AerisFilter,
    keys: &[Key],
    reverse_map: &dyn ReverseMapStore,
) -> AerisResult<()> {
    let q = filter.config.q;
    let r = filter.config.r;
    let mut rm_chunk = Vec::with_capacity(REVERSE_MAP_STREAM_CHUNK.min(keys.len()));
    for chunk in keys.chunks(REVERSE_MAP_STREAM_CHUNK) {
        rm_chunk.clear();
        for &key in chunk {
            let p = filter.config.partition_of(key);
            let (home_u64, rem) = filter.hasher.quotient_and_remainder(p, q, r);
            rm_chunk.push((Fingerprint::from_parts(home_u64, q, rem, r, &[]), key));
        }
        reverse_map.append_batch(&rm_chunk)?;
    }
    Ok(())
}
