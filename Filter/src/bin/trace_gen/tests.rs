//! Unit tests for `trace_gen` PRNG bijection, CLI parsing, I/O contracts, and query generation.

use crate::cli::{parse_args, Command, QueriesCliConfig, QueryKind};
use crate::io::{read_and_sort_keys, QUERIES_FLAGS, QUERIES_MAGIC};
use crate::keys::{generate_keys_file, Feistel48, KEYSPACE_MASK, KEYSPACE_SIZE};
use crate::queries::{generate_queries_file, is_range_empty, ZipfianSampler, SHIFTING_R_PHASES};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;

#[test]
fn feistel48_is_unique_and_within_48bit_domain() {
    let prp = Feistel48::from_seed(1380);
    let n = 50_000u64;
    let mut seen = HashSet::with_capacity(n as usize);
    for i in 0..n {
        let k = prp.permute(i);
        assert!(k <= KEYSPACE_MASK, "key {k} exceeds 48-bit domain");
        assert!(seen.insert(k), "duplicate key generated at index {i}");
    }
    assert_eq!(prp.permute(42), Feistel48::from_seed(1380).permute(42));
    assert_ne!(prp.permute(42), Feistel48::from_seed(1381).permute(42));
}

#[test]
fn keys_and_all_query_kinds_round_trip_and_verify_empty() {
    let tmp_dir = env::temp_dir().join(format!("aeris_trace_gen_test_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).unwrap();
    let keys_path = tmp_dir.join("test.keys.bin");

    let bytes = generate_keys_file(2_000, 1380, &keys_path).unwrap();
    assert_eq!(bytes, 16 + 2_000 * 8);

    let sorted_keys = read_and_sort_keys(&keys_path).unwrap();
    assert_eq!(sorted_keys.len(), 2_000);
    assert!(sorted_keys.windows(2).all(|w| w[0] < w[1]));

    for (kind, range_len, expected_max_r) in [
        (QueryKind::ZipfCorr, 32u64, 32u64),
        (QueryKind::ZipfUncorr, 16u64, 16u64),
        (QueryKind::UniformEmpty, 1u64, 1u64),
        (QueryKind::ShiftingR, 16u64, 1024u64),
    ] {
        let q_path = tmp_dir.join(format!("test_{}.queries.bin", kind.as_str()));
        let cfg = QueriesCliConfig {
            keys_path: keys_path.clone(),
            kind,
            count: 400,
            range_len,
            alpha: 1.5,
            corr_d: 0.8,
            seed: 1380,
            out: q_path.clone(),
        };
        let (q_bytes, max_r) = generate_queries_file(&cfg).unwrap();
        assert_eq!(q_bytes, 32 + 400 * 24);
        assert_eq!(max_r, expected_max_r);

        let raw = fs::read(&q_path).unwrap();
        assert_eq!(&raw[0..8], &QUERIES_MAGIC);
        assert_eq!(u64::from_le_bytes(raw[8..16].try_into().unwrap()), 400);
        assert_eq!(
            u64::from_le_bytes(raw[16..24].try_into().unwrap()),
            expected_max_r
        );
        assert_eq!(
            u64::from_le_bytes(raw[24..32].try_into().unwrap()),
            QUERIES_FLAGS
        );

        for (idx, rec) in raw[32..].as_chunks::<24>().0.iter().enumerate() {
            let start = u64::from_le_bytes(rec[0..8].try_into().unwrap());
            let end = u64::from_le_bytes(rec[8..16].try_into().unwrap());
            let expected = u64::from_le_bytes(rec[16..24].try_into().unwrap());
            assert!(start <= end);
            assert_eq!(expected, 0);
            assert!(is_range_empty(&sorted_keys, start, end));
            if kind == QueryKind::ShiftingR {
                let phase_r = SHIFTING_R_PHASES[idx / 100];
                assert_eq!(end - start + 1, phase_r);
            } else {
                assert_eq!(end - start + 1, range_len);
            }
        }
    }

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn cli_parse_args_supports_keys_queries_and_rejects_invalid_inputs() {
    assert!(parse_args(&["trace_gen".to_string()]).is_err());

    let bad_sub: Vec<String> = ["trace_gen", "invalid"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert!(parse_args(&bad_sub)
        .unwrap_err()
        .contains("unknown subcommand"));

    let keys_args: Vec<String> = [
        "trace_gen",
        "keys",
        "--count",
        "100",
        "--seed",
        "77",
        "--out",
        "out.keys.bin",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(
        parse_args(&keys_args).unwrap(),
        Command::Keys {
            count: 100,
            seed: 77,
            out: PathBuf::from("out.keys.bin"),
        }
    );

    let q_args: Vec<String> = [
        "trace_gen",
        "queries",
        "--keys",
        "k.bin",
        "--kind",
        "zipf_corr",
        "--count",
        "50",
        "--range-len",
        "32",
        "--alpha",
        "1.25",
        "--corr-d",
        "0.6",
        "--out",
        "q.bin",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(
        parse_args(&q_args).unwrap(),
        Command::Queries(QueriesCliConfig {
            keys_path: PathBuf::from("k.bin"),
            kind: QueryKind::ZipfCorr,
            count: 50,
            range_len: 32,
            alpha: 1.25,
            corr_d: 0.6,
            seed: 1380,
            out: PathBuf::from("q.bin"),
        })
    );
}

#[test]
fn boundary_and_invalid_inputs_fail_cleanly() {
    let dummy_out = env::temp_dir().join("aeris_trace_gen_unused.bin");
    assert!(generate_keys_file(0, 1, &dummy_out)
        .unwrap_err()
        .contains(">= 1"));
    assert!(generate_keys_file(KEYSPACE_SIZE + 1, 1, &dummy_out)
        .unwrap_err()
        .contains("exceeds"));

    assert!(ZipfianSampler::new(0, 1.5).is_err());
    assert!(ZipfianSampler::new(10, 0.0).is_err());
    assert!(ZipfianSampler::new(10, f64::NAN).is_err());

    let mut cfg = QueriesCliConfig {
        keys_path: dummy_out.clone(),
        kind: QueryKind::UniformEmpty,
        count: 0,
        range_len: 1,
        alpha: 1.5,
        corr_d: 0.8,
        seed: 1,
        out: dummy_out.clone(),
    };
    assert!(generate_queries_file(&cfg).unwrap_err().contains("--count"));

    cfg.count = 10;
    cfg.range_len = 0;
    assert!(generate_queries_file(&cfg)
        .unwrap_err()
        .contains("--range-len"));

    cfg.range_len = 16;
    cfg.corr_d = 1.5;
    assert!(generate_queries_file(&cfg)
        .unwrap_err()
        .contains("--corr-d"));
}
