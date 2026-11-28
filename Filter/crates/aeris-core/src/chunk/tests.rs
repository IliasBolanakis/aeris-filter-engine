use super::shift::{shift_metadata_left_one, shift_metadata_right_one};
use super::*;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn metadata_shift_matches_naive_bit_reference_across_boundaries() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC40C_0001);
    let num_chunks = 5;
    let total_bits = num_chunks * 64;
    let slot_bits = 13usize;
    let block_bytes = BLOCK_HEADER_BYTES + 8 * slot_bits;

    let boundary_pairs = [
        (0, 0),
        (0, 1),
        (0, 63),
        (63, 63),
        (63, 64),
        (0, 64),
        (0, 128),
        (62, 129),
        (63, 192),
        (1, 255),
        (64, 127),
        (64, 128),
        (127, 191),
        (0, total_bits - 1),
    ];

    for &(start, end) in &boundary_pairs {
        let mut blocks = vec![0u8; num_chunks * block_bytes + TRAILING_GUARD_BYTES];
        let mut init_offsets = vec![0u8; num_chunks];
        let mut init_occupied = vec![0u64; num_chunks];
        let mut init_payloads = vec![vec![0u8; 8 * slot_bits]; num_chunks];

        for c in 0..num_chunks {
            let base = c * block_bytes;
            init_offsets[c] = rng.gen();
            init_occupied[c] = rng.next_u64();
            blocks[base + BLOCK_OFFSET_POS] = init_offsets[c];
            write_u64_le(&mut blocks, base + BLOCK_OCCUPIED_POS, init_occupied[c]);
            write_u64_le(&mut blocks, base + BLOCK_RUNEND_POS, rng.next_u64());
            write_u64_le(&mut blocks, base + BLOCK_EXTENSION_POS, rng.next_u64());
            rng.fill_bytes(&mut init_payloads[c]);
            blocks[(base + BLOCK_HEADER_BYTES)..(base + block_bytes)]
                .copy_from_slice(&init_payloads[c]);
        }

        // Reference right shift.
        let mut ref_re: Vec<bool> = (0..total_bits)
            .map(|i| {
                let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_RUNEND_POS);
                ((w >> (i % 64)) & 1) != 0
            })
            .collect();
        let mut ref_ext: Vec<bool> = (0..total_bits)
            .map(|i| {
                let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_EXTENSION_POS);
                ((w >> (i % 64)) & 1) != 0
            })
            .collect();

        let mut right_blocks = blocks.clone();
        shift_metadata_right_one(&mut right_blocks, block_bytes, start, end);
        for i in ((start + 1)..=end).rev() {
            ref_re[i] = ref_re[i - 1];
            ref_ext[i] = ref_ext[i - 1];
        }
        ref_re[start] = false;
        ref_ext[start] = false;

        for i in 0..total_bits {
            let base = (i / 64) * block_bytes;
            let got_re =
                ((read_u64_le(&right_blocks, base + BLOCK_RUNEND_POS) >> (i % 64)) & 1) != 0;
            let got_ext =
                ((read_u64_le(&right_blocks, base + BLOCK_EXTENSION_POS) >> (i % 64)) & 1) != 0;
            assert_eq!(
                got_re, ref_re[i],
                "right shift runend at {i} for ({start},{end})"
            );
            assert_eq!(
                got_ext, ref_ext[i],
                "right shift ext at {i} for ({start},{end})"
            );
        }
        for c in 0..num_chunks {
            let base = c * block_bytes;
            assert_eq!(right_blocks[base + BLOCK_OFFSET_POS], init_offsets[c]);
            assert_eq!(
                read_u64_le(&right_blocks, base + BLOCK_OCCUPIED_POS),
                init_occupied[c]
            );
            assert_eq!(
                &right_blocks[(base + BLOCK_HEADER_BYTES)..(base + block_bytes)],
                init_payloads[c].as_slice()
            );
        }
        assert!(right_blocks[(num_chunks * block_bytes)..]
            .iter()
            .all(|&b| b == 0));

        // Reference left shift.
        let mut ref_re: Vec<bool> = (0..total_bits)
            .map(|i| {
                let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_RUNEND_POS);
                ((w >> (i % 64)) & 1) != 0
            })
            .collect();
        let mut ref_ext: Vec<bool> = (0..total_bits)
            .map(|i| {
                let w = read_u64_le(&blocks, (i / 64) * block_bytes + BLOCK_EXTENSION_POS);
                ((w >> (i % 64)) & 1) != 0
            })
            .collect();

        let mut left_blocks = blocks.clone();
        shift_metadata_left_one(&mut left_blocks, block_bytes, start, end);
        for i in start..end {
            ref_re[i] = ref_re[i + 1];
            ref_ext[i] = ref_ext[i + 1];
        }
        ref_re[end] = false;
        ref_ext[end] = false;

        for i in 0..total_bits {
            let base = (i / 64) * block_bytes;
            let got_re =
                ((read_u64_le(&left_blocks, base + BLOCK_RUNEND_POS) >> (i % 64)) & 1) != 0;
            let got_ext =
                ((read_u64_le(&left_blocks, base + BLOCK_EXTENSION_POS) >> (i % 64)) & 1) != 0;
            assert_eq!(
                got_re, ref_re[i],
                "left shift runend at {i} for ({start},{end})"
            );
            assert_eq!(
                got_ext, ref_ext[i],
                "left shift ext at {i} for ({start},{end})"
            );
        }
        for c in 0..num_chunks {
            let base = c * block_bytes;
            assert_eq!(left_blocks[base + BLOCK_OFFSET_POS], init_offsets[c]);
            assert_eq!(
                read_u64_le(&left_blocks, base + BLOCK_OCCUPIED_POS),
                init_occupied[c]
            );
            assert_eq!(
                &left_blocks[(base + BLOCK_HEADER_BYTES)..(base + block_bytes)],
                init_payloads[c].as_slice()
            );
        }
        assert!(left_blocks[(num_chunks * block_bytes)..]
            .iter()
            .all(|&b| b == 0));
    }
}

#[test]
fn payload_round_trip_across_edge_slot_bits_without_metadata_bleed() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xA05B_0001);
    let edge_widths = [1u8, 2, 7, 10, 13, 18, 32, 57, 58, 63, 64];

    for slot_bits in edge_widths {
        let mut table = QuotientTable::new(256, slot_bits);
        let total_slots = table.physical_slots();
        let total_chunks = table.total_chunks();
        let expected_block_bytes = BLOCK_HEADER_BYTES + 8 * (slot_bits as usize);
        assert_eq!(table.block_bytes, expected_block_bytes);
        assert_eq!(
            table.memory_usage_bytes(),
            total_chunks * expected_block_bytes + TRAILING_GUARD_BYTES
        );

        let mask = if slot_bits == 64 {
            u64::MAX
        } else {
            (1u64 << slot_bits) - 1
        };

        let mut exp_offsets = Vec::with_capacity(total_chunks);
        let mut exp_occupied = Vec::with_capacity(total_chunks);
        let mut exp_runend = Vec::with_capacity(total_chunks);
        let mut exp_extension = Vec::with_capacity(total_chunks);

        for c in 0..total_chunks {
            let off = rng.gen_range(1u8..250u8);
            let occ = rng.next_u64() | 0x8000_0000_0000_0001;
            let re = rng.next_u64() | 0xA5A5_5A5A_0000_0001;
            let ext = rng.next_u64() | 0x5A5A_A5A5_8000_0000;
            let base = c * table.block_bytes;
            table.blocks[base + BLOCK_OFFSET_POS] = off;
            write_u64_le(&mut table.blocks, base + BLOCK_OCCUPIED_POS, occ);
            write_u64_le(&mut table.blocks, base + BLOCK_RUNEND_POS, re);
            write_u64_le(&mut table.blocks, base + BLOCK_EXTENSION_POS, ext);
            exp_offsets.push(off);
            exp_occupied.push(occ);
            exp_runend.push(re);
            exp_extension.push(ext);
        }

        let mut expected = vec![0u64; total_slots];
        for (slot, exp) in expected.iter_mut().enumerate() {
            let raw = if slot % 17 == 0 {
                u64::MAX
            } else if slot % 19 == 0 {
                0
            } else {
                rng.next_u64()
            };
            table.set_payload(slot, raw);
            *exp = raw & mask;
        }

        for (slot, &exp) in expected.iter().enumerate() {
            let got = table.get_payload(slot);
            assert_eq!(got, exp, "pass1 width={slot_bits} slot={slot}");
            let (got2, kind) = table.get_payload_and_kind(slot);
            assert_eq!(got2, exp, "pass1 co-located width={slot_bits} slot={slot}");
            assert_eq!(kind, table.slot_kind(slot));
        }

        for (slot, exp) in expected.iter_mut().enumerate().rev() {
            let raw = rng.next_u64();
            table.set_payload(slot, raw);
            *exp = raw & mask;
        }

        for (slot, &exp) in expected.iter().enumerate() {
            let (got, kind) = table.get_payload_and_kind(slot);
            assert_eq!(got, exp, "pass2 width={slot_bits} slot={slot}");
            assert_eq!(kind, table.slot_kind(slot));
        }

        for c in 0..total_chunks {
            let base = c * table.block_bytes;
            assert_eq!(
                table.blocks[base + BLOCK_OFFSET_POS],
                exp_offsets[c],
                "offset bled width={slot_bits} chunk={c}"
            );
            assert_eq!(
                read_u64_le(&table.blocks, base + BLOCK_OCCUPIED_POS),
                exp_occupied[c],
                "occupied bled width={slot_bits} chunk={c}"
            );
            assert_eq!(
                read_u64_le(&table.blocks, base + BLOCK_RUNEND_POS),
                exp_runend[c],
                "runend bled width={slot_bits} chunk={c}"
            );
            assert_eq!(
                read_u64_le(&table.blocks, base + BLOCK_EXTENSION_POS),
                exp_extension[c],
                "extension bled width={slot_bits} chunk={c}"
            );
        }
        let guard_start = total_chunks * table.block_bytes;
        assert!(
            table.blocks[guard_start..].iter().all(|&b| b == 0),
            "guard bytes corrupted for width={slot_bits}"
        );
    }
}

#[test]
fn payload_shift_right_and_left_across_edge_slot_bits_and_blocks() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xA05B_0002);
    let edge_widths = [1u8, 2, 7, 10, 13, 18, 32, 57, 58, 63, 64];
    let pairs = [
        (0usize, 0usize),
        (0, 1),
        (1, 2),
        (10, 13),
        (5, 20),
        (60, 63),
        (62, 63),
        (0, 63),
        (63, 64),
        (10, 135),
        (0, 255),
        (62, 259),
    ];

    for slot_bits in edge_widths {
        let mask = if slot_bits == 64 {
            u64::MAX
        } else {
            (1u64 << slot_bits) - 1
        };

        for &(start, end) in &pairs {
            let mut table = QuotientTable::new(256, slot_bits);
            let total_slots = table.physical_slots();
            let mut model = vec![0u64; total_slots];
            for (i, m) in model.iter_mut().enumerate() {
                let v = rng.next_u64() & mask;
                table.set_payload(i, v);
                *m = v;
            }

            table.shift_payloads_right_one(start, end);
            for i in ((start + 1)..=end).rev() {
                model[i] = model[i - 1];
            }
            model[start] = 0;
            for (i, &exp) in model.iter().enumerate() {
                assert_eq!(
                    table.get_payload(i),
                    exp,
                    "shift_right width={slot_bits} pair=({start},{end}) slot={i}"
                );
            }

            table.shift_payloads_left_one(start, end);
            for i in start..end {
                model[i] = model[i + 1];
            }
            model[end] = 0;
            for (i, &exp) in model.iter().enumerate() {
                assert_eq!(
                    table.get_payload(i),
                    exp,
                    "shift_left width={slot_bits} pair=({start},{end}) slot={i}"
                );
            }

            let guard_start = table.total_chunks() * table.block_bytes;
            assert!(table.blocks[guard_start..].iter().all(|&b| b == 0));
        }
    }
}

#[test]
fn exact_memory_usage_and_bits_per_key_at_90_percent_load_factor() {
    let q = 20u8;
    let n = 1usize << q;
    let keys_90_pct = (n as f64) * 0.90;
    let r = 7u8;

    for (m, max_bpk) in [(2u8, 14.70f64), (5u8, 18.00f64), (10u8, 23.60f64)] {
        let slot_bits = r + 1 + m;
        let table = QuotientTable::new(n, slot_bits);
        let expected_block_bytes = 25 + 8 * (slot_bits as usize);
        let expected_bytes = table.total_chunks() * expected_block_bytes + TRAILING_GUARD_BYTES;
        assert_eq!(table.memory_usage_bytes(), expected_bytes);

        let bpk = (table.memory_usage_bytes() as f64) * 8.0 / keys_90_pct;
        assert!(
            bpk <= max_bpk,
            "r={r}, m={m} (slot_bits={slot_bits}): bpk={bpk:.4} exceeded {max_bpk:.2}"
        );
        if m == 5 {
            assert!(
                bpk - 16.85 <= 1.25,
                "m=5 delta over C++ 16.85 bpk was {:.4} > 1.25",
                bpk - 16.85
            );
        }
    }
}

#[test]
fn random_run_edits_preserve_table_invariants_and_handle_offset_overflow() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC40C_0002);
    let n = 128;
    let mut table = QuotientTable::new(n, 20);
    let mut model: BTreeMap<usize, Vec<(u64, KeepsakeSlotKind)>> = BTreeMap::new();

    // Force chunk-1 offset overflow (> 255) by building a 300-slot run at home 63.
    let mut big_run = Vec::with_capacity(300);
    for i in 0..299 {
        big_run.push((i as u64 + 1, KeepsakeSlotKind::PayloadInterior));
    }
    big_run.push((300, KeepsakeSlotKind::BoxAndRunEnd));
    table.replace_region(63, 0, 0, &big_run);
    model.insert(63, big_run);
    assert!(
        table.chunk_offset(1) >= 255,
        "expected offset overflow on chunk 1"
    );
    assert!(table.verify_metadata_invariants().is_ok());

    for _ in 0..400 {
        let home = rng.gen_range(0..n);
        let op = rng.gen_range(0..4);
        let existing = model.get(&home).cloned();
        match (existing, op) {
            (None, _) => {
                let len = rng.gen_range(1..=6);
                let mut slots: Vec<_> = (0..len)
                    .map(|i| {
                        let kind = if i + 1 == len {
                            KeepsakeSlotKind::BoxAndRunEnd
                        } else if i % 2 == 0 {
                            KeepsakeSlotKind::Extension
                        } else {
                            KeepsakeSlotKind::PayloadInterior
                        };
                        (rng.gen::<u64>() & ((1 << 20) - 1), kind)
                    })
                    .collect();
                slots.last_mut().unwrap().1 = KeepsakeSlotKind::BoxAndRunEnd;
                table.replace_region(home, 0, 0, &slots);
                model.insert(home, slots);
            }
            (Some(old), 0) => {
                let (rs, re) = table.locate_run(home).unwrap();
                assert_eq!(re - rs + 1, old.len());
                table.replace_region(home, rs, old.len(), &[]);
                model.remove(&home);
            }
            (Some(old), _) => {
                let (rs, re) = table.locate_run(home).unwrap();
                assert_eq!(re - rs + 1, old.len());
                let new_len = rng.gen_range(1..=10);
                let mut slots: Vec<_> = (0..new_len)
                    .map(|i| {
                        let kind = if i + 1 == new_len {
                            KeepsakeSlotKind::BoxAndRunEnd
                        } else {
                            KeepsakeSlotKind::BoxEnd
                        };
                        (rng.gen::<u64>() & ((1 << 20) - 1), kind)
                    })
                    .collect();
                slots.last_mut().unwrap().1 = KeepsakeSlotKind::BoxAndRunEnd;
                table.replace_region(home, rs, old.len(), &slots);
                model.insert(home, slots);
            }
        }

        table.verify_metadata_invariants().unwrap();
        let occupied_list: Vec<usize> = table.occupied_homes().collect();
        let expected_homes: Vec<usize> = model.keys().copied().collect();
        assert_eq!(occupied_list, expected_homes);

        for (&h, expected_slots) in &model {
            let (rs, re) = table.locate_run(h).unwrap();
            assert_eq!(re - rs + 1, expected_slots.len());
            for (i, &(exp_p, exp_k)) in expected_slots.iter().enumerate() {
                assert_eq!(table.get_payload(rs + i), exp_p);
                assert_eq!(table.slot_kind(rs + i), exp_k);
                assert_eq!(table.get_payload_and_kind(rs + i), (exp_p, exp_k));
            }
        }
    }
}

#[test]
fn chunk_metadata_bulk_writer_and_invariant_error_paths() {
    let mut meta = ChunkMetadata::default();
    assert!(!meta.is_occupied(7));
    meta.set_occupied(7);
    assert!(meta.is_occupied(7));
    meta.set_slot_kind(7, KeepsakeSlotKind::BoxAndRunEnd);
    assert_eq!(meta.slot_kind(7), KeepsakeSlotKind::BoxAndRunEnd);
    assert_eq!(meta.qrun_end_mask(), 1u64 << 7);
    meta.clear_occupied(7);
    assert!(!meta.is_occupied(7));

    // BulkTableWriter spanning chunk boundary and cluster jumps.
    let mut table = QuotientTable::new(128, 16);
    {
        let mut w = table.bulk_writer();
        w.append_single_slot_run(60, 0xAA);
        w.append_run(
            62,
            &[
                (1, KeepsakeSlotKind::PayloadInterior),
                (2, KeepsakeSlotKind::PayloadInterior),
                (3, KeepsakeSlotKind::PayloadInterior),
                (4, KeepsakeSlotKind::BoxAndRunEnd),
            ],
        );
        w.append_single_slot_run(64, 0xBB);
        w.finish();
    }
    table.verify_metadata_invariants().unwrap();
    assert_eq!(table.locate_run(60), Some((60, 60)));
    assert_eq!(table.locate_run(62), Some((62, 65)));
    assert_eq!(table.locate_run(64), Some((66, 66)));
    assert_eq!(table.chunk_offset(1), 2);

    // Corrupted offset triggers InternalInvariant.
    let mut bad_offset = table.clone();
    bad_offset.set_chunk_offset(1, 99);
    assert!(bad_offset.verify_metadata_invariants().is_err());

    // Corrupted guard byte triggers InternalInvariant.
    let mut bad_guard = table.clone();
    let last_idx = bad_guard.blocks.len() - 1;
    bad_guard.blocks[last_idx] = 1;
    assert!(bad_guard.verify_metadata_invariants().is_err());
}
