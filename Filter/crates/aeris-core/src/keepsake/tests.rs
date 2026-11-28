use super::*;
use crate::chunk::QuotientTable;
use aeris_types::{FilterConfig, Fingerprint, KeepsakeSlotKind, PartitionHasher};

#[test]
fn unary_age_counter_round_trips_for_all_ages() {
    for r in [4u8, 8, 12, 20, 40] {
        for age in 0..=r {
            let rem_bits = r - age;
            let max_rem = mask_for_bits(rem_bits);
            for rem in [0u64, 1 & max_rem, max_rem >> 1, max_rem] {
                let encoded = encode_age_remainder(r, age, rem);
                assert_ne!(encoded, 0);
                let (dec_age, dec_bits, dec_rem) = decode_age_remainder(r, encoded);
                assert_eq!(dec_age, age);
                assert_eq!(dec_bits, rem_bits);
                assert_eq!(dec_rem, rem);
            }
        }
    }
}

#[test]
fn keepsake_box_round_trip_and_hot_match_for_m_positive_and_zero() {
    let hasher = PartitionHasher::new(0x1234_5678_9ABC_DEF0);

    for &(r, m, max_range) in &[(12u8, 0u8, 1u64), (12, 1, 2), (12, 5, 32), (12, 10, 1024)] {
        let cfg = FilterConfig::new(8, r, max_range, r + m, 1.0 / 256.0).unwrap();
        let geom = KeepsakeGeometry::from_config(&cfg);
        let mut table = QuotientTable::new(cfg.num_slots(), geom.slot_bits);

        let partition = 0xDEAD_BEEF_u64;
        let (home_u64, rem_full) = hasher.quotient_and_remainder(partition, cfg.q, cfg.r);
        let home = home_u64 as usize;
        let w0 = hasher.word(partition, 0);

        for ext_level in 0..=3 {
            let payloads: Vec<BoxPayload> = if m == 0 {
                vec![
                    BoxPayload::PointCount(1),
                    BoxPayload::PointCount(2),
                    BoxPayload::PointCount(500),
                ]
            } else {
                let max_m = geom.m_mask;
                vec![
                    BoxPayload::Mementos(vec![0]),
                    BoxPayload::Mementos(vec![max_m]),
                    BoxPayload::Mementos(vec![0, 0, 1 & max_m, max_m, max_m]),
                    BoxPayload::Mementos((0..=max_m.min(11)).collect()),
                ]
            };

            for payload in payloads {
                let bx =
                    KeepsakeBox::new_fresh(&hasher, partition, cfg.q, &geom, ext_level, payload);
                let encoded = bx.encode(&geom, true);
                assert_eq!(encoded.len(), bx.encoded_slot_len(&geom));
                // Extension slot must NEVER be last slot of a keepsake box.
                assert!(!encoded.last().unwrap().1.is_extension_slot());
                assert!(encoded.last().unwrap().1.is_run_end());

                let old_len = table.locate_run(home).map(|(s, e)| e - s + 1).unwrap_or(0);
                let start = table.locate_run(home).map(|(s, _)| s).unwrap_or(0);
                table.replace_region(home, start, old_len, &encoded);
                let (rs, re) = table.locate_run(home).unwrap();
                assert_eq!(re - rs + 1, encoded.len());

                let (decoded, consumed) = KeepsakeBox::decode_from_table(&table, &geom, rs);
                assert_eq!(consumed, encoded.len());
                assert_eq!(decoded, bx);
                assert_eq!(
                    decoded.canonical_fingerprint(home, cfg.q, cfg.r),
                    Fingerprint::from_hasher(
                        &hasher,
                        partition,
                        (cfg.q as u16) + (cfg.r as u16) + geom.ext_bits_for_level(ext_level)
                    )
                );

                // Verify hot-path query matches exact decoded payload.
                let (hit_full, next_slot) = box_matches_query_hot(
                    &table,
                    &geom,
                    &hasher,
                    partition,
                    w0,
                    cfg.q,
                    rem_full,
                    0,
                    geom.m_mask,
                    rs,
                );
                assert!(hit_full);
                assert_eq!(next_slot, re + 1);

                // Non-matching partition must not match (except vanishingly unlikely collision).
                let other_part = partition ^ 0xFFFF_FFFF;
                let (_, other_rem) = hasher.quotient_and_remainder(other_part, cfg.q, cfg.r);
                let (other_hit, other_next) = box_matches_query_hot(
                    &table,
                    &geom,
                    &hasher,
                    other_part,
                    hasher.word(other_part, 0),
                    cfg.q,
                    other_rem,
                    0,
                    geom.m_mask,
                    rs,
                );
                assert_eq!(other_next, re + 1);
                assert_eq!(
                    other_hit,
                    bx.matches_partition(&hasher, other_part, cfg.q, cfg.r)
                );
            }
        }
    }
}

#[test]
fn expansion_bit_split_preserves_canonical_fingerprint_until_rejuvenation() {
    let hasher = PartitionHasher::new(999);
    let cfg = FilterConfig::new(8, 6, 32, 11, 1.0 / 8.0)
        .or_else(|_| FilterConfig::with_minimum_bits(8, 32, 0.05))
        .unwrap();
    let geom = KeepsakeGeometry::from_config(&cfg);
    let partition = 42_424_242u64;
    let (home0, _) = hasher.quotient_and_remainder(partition, cfg.q, cfg.r);

    let mut bx = KeepsakeBox::new_fresh(
        &hasher,
        partition,
        cfg.q,
        &geom,
        2,
        BoxPayload::Mementos(vec![3, 7, 19]),
    );
    let orig_fp = bx.canonical_fingerprint(home0 as usize, cfg.q, cfg.r);

    let mut cur_home = home0 as usize;
    let mut cur_q = cfg.q;
    for step in 0..cfg.r {
        let (next_home, next_box) = bx
            .split_for_expansion(cur_home, cfg.r)
            .expect("age < r must split in memory");
        cur_home = next_home;
        cur_q += 1;
        bx = next_box;
        assert_eq!(bx.age, step + 1);
        assert_eq!(bx.canonical_fingerprint(cur_home, cur_q, cfg.r), orig_fp);
        assert!(bx.matches_partition(&hasher, partition, cur_q, cfg.r));
    }

    // At age == r, next split returns None -> triggers selective rejuvenation!
    assert_eq!(bx.age, cfg.r);
    assert!(bx.split_for_expansion(cur_home, cfg.r).is_none());
}

#[test]
fn point_mode_multi_limb_count_round_trips_for_narrow_and_64bit_slots() {
    let hasher = PartitionHasher::new(77);
    for &(r, eps) in &[(7u8, 0.1), (63u8, 1.0 / 256.0)] {
        let cfg = FilterConfig::new(7, r, 1, r, eps).unwrap();
        let geom = KeepsakeGeometry::from_config(&cfg);
        let mut table = QuotientTable::new(cfg.num_slots(), geom.slot_bits);
        let part = 12345u64;
        let (home_u64, _) = hasher.quotient_and_remainder(part, cfg.q, cfg.r);
        let home = home_u64 as usize;

        for ext_level in [0usize, 1, 2] {
            for count in [1usize, 2, 255, 256, 70_000] {
                let bx = KeepsakeBox::new_fresh(
                    &hasher,
                    part,
                    cfg.q,
                    &geom,
                    ext_level,
                    BoxPayload::PointCount(count),
                );
                let enc = bx.encode(&geom, true);
                assert_eq!(enc.len(), bx.encoded_slot_len(&geom));
                assert!(!enc.last().unwrap().1.is_extension_slot());
                let (start, old_len) = table
                    .locate_run(home)
                    .map(|(s, e)| (s, e - s + 1))
                    .unwrap_or((0, 0));
                table.replace_region(home, start, old_len, &enc);
                let (rs, _) = table.locate_run(home).unwrap();
                let (dec, consumed) = KeepsakeBox::decode_from_table(&table, &geom, rs);
                assert_eq!(consumed, enc.len());
                assert_eq!(dec, bx);
            }
        }
    }
}

#[test]
fn hot_match_handles_memento_bounds_aged_boxes_and_partition_matching() {
    let hasher = PartitionHasher::new(0xABCD_1234);
    let cfg = FilterConfig::new(6, 7, 32, 12, 0.08).unwrap();
    let geom = KeepsakeGeometry::from_config(&cfg);
    let mut table = QuotientTable::new(cfg.num_slots(), geom.slot_bits);

    let part = 555u64;
    let (home_u64, rem_full) = hasher.quotient_and_remainder(part, cfg.q, cfg.r);
    let home = home_u64 as usize;
    let w0 = hasher.word(part, 0);

    // Multi-memento extended box exercising `cont_header_bits == 0` / continuation scan early exit.
    let bx = KeepsakeBox::new_fresh(
        &hasher,
        part,
        cfg.q,
        &geom,
        2,
        BoxPayload::Mementos(vec![4, 10, 10, 25]),
    );
    let enc = bx.encode(&geom, true);
    table.replace_region(home, 0, 0, &enc);
    let (rs, re) = table.locate_run(home).unwrap();

    // Below min, exact hit, between elements (early stop on mem > high_m), above max.
    assert_eq!(
        box_matches_query_hot(&table, &geom, &hasher, part, w0, cfg.q, rem_full, 0, 3, rs),
        (false, re + 1)
    );
    assert_eq!(
        box_matches_query_hot(&table, &geom, &hasher, part, w0, cfg.q, rem_full, 10, 10, rs),
        (true, re + 1)
    );
    assert_eq!(
        box_matches_query_hot(&table, &geom, &hasher, part, w0, cfg.q, rem_full, 11, 20, rs),
        (false, re + 1)
    );
    assert_eq!(
        box_matches_query_hot(&table, &geom, &hasher, part, w0, cfg.q, rem_full, 26, 31, rs),
        (false, re + 1)
    );

    // Partition-only hot match on extended box.
    let (pm, next, _, kind0) =
        box_matches_partition_hot(&table, &geom, &hasher, part, w0, cfg.q, rem_full, rs);
    assert!(pm);
    assert_eq!(next, re + 1);
    assert_eq!(kind0, KeepsakeSlotKind::Extension);

    // Aged box matching via `split_for_expansion`.
    let (aged_home, aged_bx) = bx.split_for_expansion(home, cfg.r).unwrap();
    let new_q = cfg.q + 1;
    let (_, aged_rem_full) = hasher.quotient_and_remainder(part, new_q, cfg.r);
    let mut aged_table = QuotientTable::new(1 << new_q, geom.slot_bits);
    let aged_enc = aged_bx.encode(&geom, true);
    aged_table.replace_region(aged_home, 0, 0, &aged_enc);
    let (ars, are) = aged_table.locate_run(aged_home).unwrap();
    assert_eq!(
        box_matches_query_hot(
            &aged_table,
            &geom,
            &hasher,
            part,
            w0,
            new_q,
            aged_rem_full,
            25,
            25,
            ars
        ),
        (true, are + 1)
    );
    let (apm, anext, _, _) = box_matches_partition_hot(
        &aged_table,
        &geom,
        &hasher,
        part,
        w0,
        new_q,
        aged_rem_full,
        ars,
    );
    assert!(apm);
    assert_eq!(anext, are + 1);
}

#[test]
fn geometry_and_payload_boundary_helpers() {
    assert_eq!(ext_bits_for_level(8, 5, 0), 0);
    assert_eq!(ext_bits_for_level(8, 5, 1), 5);
    assert_eq!(ext_bits_for_level(8, 5, 3), 5 + 2 * 13);
    assert_eq!(ext_bits_for_level(8, 0, 3), 24);

    assert_eq!(min_ext_level_for_bits(8, 5, 0), 0);
    assert_eq!(min_ext_level_for_bits(8, 5, 5), 1);
    assert_eq!(min_ext_level_for_bits(8, 5, 6), 2);
    assert_eq!(min_ext_level_for_bits(8, 0, 9), 2);

    let mut p = BoxPayload::Mementos(vec![1, 3, 7]);
    assert!(p.remove_one(3));
    assert!(!p.remove_one(3));
    assert_eq!(p.key_count(), 2);
    assert!(!p.contains_in_range(2, 6));
    assert!(p.contains_in_range(7, 10));

    let mut pt = BoxPayload::PointCount(1);
    assert!(pt.remove_one(0));
    assert!(pt.is_empty());
    assert!(!pt.remove_one(0));
    assert!(!pt.contains_in_range(0, 0));
}
