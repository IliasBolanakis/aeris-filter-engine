//! Unit tests for `paper_parity` submodules, invariants, and error contracts.

use crate::cli::{build_filter_config, parse_args, BenchMode, CliArgs, ImplVariant};
use crate::io::{read_keys_file, read_queries_file, QueryRecord, KEYS_MAGIC, QUERIES_MAGIC};
use crate::metrics::{compute_bits_per_key, compute_fpr, compute_p50_p99};
use crate::probe::is_range_empty;
use crate::reverse_map::{InMemReverseMap, NoopReverseMap};
use crate::shifting_r::{detect_query_phases, run_shifting_r};
use aeris_types::{Fingerprint, Key, PartitionHasher, QueryRange, ReverseMapStore};
use std::env;
use std::fs;
use std::path::PathBuf;

fn sample_cli_args() -> CliArgs {
    CliArgs {
        mode: BenchMode::ShiftingR,
        impl_variant: ImplVariant::RustOptimized,
        scenario: "unit_shifting_r".to_string(),
        keys_path: PathBuf::from("unit.keys.bin"),
        queries_path: Some(PathBuf::from("unit.queries.bin")),
        q: 13,
        r: 13,
        m: 4,
        budget_c: Some(17),
        target_epsilon: Some(0.0625),
        seed: 1380,
        static_bypass: false,
        out_path: None,
    }
}

#[test]
fn in_mem_reverse_map_satisfies_store_contract() {
    let rm = InMemReverseMap::with_capacity(16);
    assert!(!rm.is_noop());
    let hasher = PartitionHasher::new(42);
    let fp1 = Fingerprint::from_hasher(&hasher, 10, 20);
    let fp2 = Fingerprint::from_hasher(&hasher, 20, 24);

    rm.append_batch(&[]).unwrap();
    assert_eq!(rm.entry_count_estimate(), 0);

    rm.append_batch(&[(fp1.clone(), 100), (fp1.clone(), 101)])
        .unwrap();
    assert_eq!(rm.entry_count_estimate(), 2);
    assert_eq!(rm.get_keys(&fp1).unwrap(), vec![100, 101]);

    rm.remove_key(&fp1, 100).unwrap();
    assert_eq!(rm.get_keys(&fp1).unwrap(), vec![101]);
    assert_eq!(rm.entry_count_estimate(), 1);

    rm.replace_fingerprint(&fp1, &[(fp2.clone(), 101), (fp2.clone(), 202)])
        .unwrap();
    assert!(rm.get_keys(&fp1).unwrap().is_empty());
    assert_eq!(rm.get_keys(&fp2).unwrap(), vec![101, 202]);
    assert_eq!(rm.scan_all_keys().unwrap(), vec![101, 202]);

    rm.tombstone(&fp2).unwrap();
    assert!(rm.get_keys(&fp2).unwrap().is_empty());
    assert_eq!(rm.entry_count_estimate(), 0);
}

#[test]
fn noop_reverse_map_reports_noop_and_empty() {
    let rm = NoopReverseMap;
    assert!(rm.is_noop());
    assert_eq!(rm.entry_count_estimate(), 0);
    assert!(rm.scan_all_keys().unwrap().is_empty());
    assert_eq!(rm.iter_all_keys().unwrap().count(), 0);
}

#[test]
fn compute_p50_p99_handles_empty_and_sorted_percentiles() {
    assert_eq!(compute_p50_p99(vec![]), (0, 0));
    let samples: Vec<u64> = (1..=100).rev().collect();
    let (p50, p99) = compute_p50_p99(samples);
    assert_eq!(p50, 50);
    assert_eq!(p99, 99);
}

#[test]
fn compute_bits_per_key_and_fpr_handle_zero_denominators() {
    assert_eq!(compute_bits_per_key(1024, 0), 0.0);
    assert!((compute_bits_per_key(1024, 1024) - 8.0).abs() < 1e-12);
    assert_eq!(compute_fpr(5, 0), 0.0);
    assert!((compute_fpr(3, 12) - 0.25).abs() < 1e-12);
}

#[test]
fn cli_parse_args_handles_valid_and_invalid_inputs() {
    let err_empty = parse_args(&["paper_parity".to_string()]).unwrap_err();
    assert!(err_empty.contains("Usage: paper_parity"));

    let missing_q_args: Vec<String> = [
        "paper_parity",
        "--mode",
        "query_adapt",
        "--impl",
        "rust_naive",
        "--scenario",
        "s",
        "--keys",
        "k.bin",
        "--q",
        "10",
        "--r",
        "10",
        "--m",
        "4",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert!(parse_args(&missing_q_args)
        .unwrap_err()
        .contains("requires --queries"));

    let valid_args: Vec<String> = [
        "paper_parity",
        "--mode",
        "bulk_load",
        "--impl",
        "rust_optimized",
        "--scenario",
        "test",
        "--keys",
        "k.bin",
        "--q",
        "10",
        "--r",
        "10",
        "--m",
        "4",
        "--static-bypass",
        "--seed",
        "99",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let parsed = parse_args(&valid_args).unwrap();
    assert_eq!(parsed.mode, BenchMode::BulkLoad);
    assert_eq!(parsed.impl_variant, ImplVariant::RustOptimized);
    assert!(parsed.static_bypass);
    assert_eq!(parsed.seed, 99);
}

#[test]
fn build_filter_config_rejects_m_overflow_and_relaxes_low_r_epsilon() {
    let mut args = sample_cli_args();
    args.m = 64;
    assert!(build_filter_config(&args).unwrap_err().contains("<= 63"));

    args.m = 2;
    args.r = 4;
    args.budget_c = None;
    args.target_epsilon = None;
    let cfg = build_filter_config(&args).unwrap();
    assert!((cfg.target_epsilon - 0.25).abs() < 1e-12);
    assert_eq!(cfg.payload_budget_c, 6);
}

#[test]
fn binary_readers_reject_bad_magic_and_truncated_payloads() {
    let tmp_dir = env::temp_dir().join(format!("aeris_parity_io_test_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).unwrap();

    let bad_keys = tmp_dir.join("bad.keys.bin");
    fs::write(&bad_keys, b"BADMAGIC\x01\x00\x00\x00\x00\x00\x00\x00").unwrap();
    assert!(read_keys_file(&bad_keys)
        .unwrap_err()
        .contains("invalid keys magic"));

    let trunc_keys = tmp_dir.join("trunc.keys.bin");
    let mut raw_k = Vec::from(KEYS_MAGIC);
    raw_k.extend_from_slice(&2u64.to_le_bytes());
    raw_k.extend_from_slice(&42u64.to_le_bytes());
    fs::write(&trunc_keys, &raw_k).unwrap();
    assert!(read_keys_file(&trunc_keys)
        .unwrap_err()
        .contains("truncated keys payload"));

    let bad_queries = tmp_dir.join("bad.queries.bin");
    let mut raw_q = Vec::from(*b"NOTQUERY");
    raw_q.extend_from_slice(&[0u8; 24]);
    fs::write(&bad_queries, &raw_q).unwrap();
    assert!(read_queries_file(&bad_queries)
        .unwrap_err()
        .contains("invalid queries magic"));

    let valid_queries = tmp_dir.join("valid.queries.bin");
    let mut good_q = Vec::from(QUERIES_MAGIC);
    good_q.extend_from_slice(&1u64.to_le_bytes());
    good_q.extend_from_slice(&4u64.to_le_bytes());
    good_q.extend_from_slice(&1u64.to_le_bytes());
    good_q.extend_from_slice(&10u64.to_le_bytes());
    good_q.extend_from_slice(&13u64.to_le_bytes());
    good_q.extend_from_slice(&0u64.to_le_bytes());
    fs::write(&valid_queries, &good_q).unwrap();
    let (recs, max_r) = read_queries_file(&valid_queries).unwrap();
    assert_eq!(max_r, 4);
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].range.start(), 10);
    assert_eq!(recs[0].range.end(), 13);

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn detect_query_phases_groups_contiguous_lengths() {
    assert!(detect_query_phases(&[]).is_empty());
    let recs = vec![
        QueryRecord {
            range: QueryRange::new(0, 15).unwrap(),
            expected: 0,
        },
        QueryRecord {
            range: QueryRange::new(20, 35).unwrap(),
            expected: 0,
        },
        QueryRecord {
            range: QueryRange::new(100, 355).unwrap(),
            expected: 0,
        },
    ];
    assert_eq!(detect_query_phases(&recs), vec![(0, 2, 16), (2, 3, 256)]);
}

#[test]
fn shifting_r_optimized_reconfigures_and_preserves_zero_fn_and_zero_repeated_fp() {
    let keys: Vec<Key> = (0..4_000u64).map(|i| i * 25_000 + 7).collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();

    let mut queries = Vec::new();
    for &phase_r in &[16u64, 256, 32, 1024] {
        for j in 0..3_000u64 {
            let start = j * 25_000 + 2_000;
            let end = start + phase_r - 1;
            if is_range_empty(&sorted, start, end) {
                queries.push(QueryRecord {
                    range: QueryRange::new(start, end).unwrap(),
                    expected: 0,
                });
            }
        }
    }

    let args = sample_cli_args();
    let cfg = build_filter_config(&args).unwrap();
    let res = run_shifting_r(&args, cfg, &keys, &queries).unwrap();

    assert_eq!(res.fn_count, 0, "expected zero false negatives");
    assert_eq!(res.repeated_fp_count, 0, "expected zero repeated FPs");
    assert!(
        res.notes.contains("rebuilds=") && !res.notes.contains("rebuilds=0"),
        "expected dynamic-R rebuilds in notes: {}",
        res.notes
    );
}
