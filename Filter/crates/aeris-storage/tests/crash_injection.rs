//! Real subprocess `SIGKILL` (`TerminateProcess` / `kill -9`) fault-injection test
//! exercising 5 consecutive crash-recovery cycles against [`CoupledStorageEngine`].
//!
//! # Coordination Protocol (Zero `thread::sleep()`)
//! 1. The parent test spawns this same compiled test binary with
//!    `AERIS_CRASH_CHILD=1` and `--exact crash_injection_child_driver --nocapture`.
//! 2. The child opens `CoupledStorageEngine::open_and_recover` on the shared
//!    persistent directory and continuously writes deterministic `ChaCha8Rng` keys.
//! 3. Immediately AFTER each `engine.put(key, &payload, &fp)` completes (having
//!    executed WAL append + `fsync` + KV write + Reverse Map merge), the child writes
//!    `ACK:<lsn>:<key>\n` to `stdout` and flushes it.
//! 4. As soon as the parent reads `TARGET_ACKS_PER_CYCLE` `ACK:` lines from the pipe
//!    (while the child is already executing subsequent writes mid-flight), the parent
//!    violently kills the child via `child.kill()` (`TerminateProcess` on Windows,
//!    `SIGKILL` on Unix) and waits for OS handle reclamation via `child.wait()`.
//! 5. On cycles 1..=3, the parent also appends realistic torn-tail WAL bytes
//!    (mid-header, mid-payload, bad tail CRC32) before running `open_and_recover`.
//! 6. After EVERY crash, the parent verifies 0 lost KV keys, 0 lost Reverse Map
//!    keys, exact `kv_store == reverse_map` parity, and 0 false negatives on
//!    `range_non_empty(QueryRange::point(k))`.

use aeris_storage::{
    encode_wal_frame, CoupledStorageEngine, WAL_FILE_NAME, WAL_FORMAT_VERSION, WAL_HEADER_LEN,
    WAL_MAGIC,
};
use aeris_types::{Fingerprint, Key, Lsn, PartitionHasher, QueryRange, ReverseMapStore, WalOp};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::TempDir;

const CRASH_CHILD_ENV: &str = "AERIS_CRASH_CHILD";
const CRASH_DIR_ENV: &str = "AERIS_CRASH_DIR";
const CRASH_CYCLE_ENV: &str = "AERIS_CRASH_CYCLE";
const CRASH_CHECKPOINT_AT_ENV: &str = "AERIS_CRASH_CHECKPOINT_AT";

const HASH_SEED: u64 = 0xAE51_5709;
const FP_BITS: u16 = 22;
const CRASH_CYCLES: u64 = 5;
const TARGET_ACKS_PER_CYCLE: usize = 25;
const CHILD_SAFETY_ITERATION_CAP: usize = 50_000;

fn deterministic_fp(key: Key) -> Fingerprint {
    let hasher = PartitionHasher::new(HASH_SEED);
    Fingerprint::from_hasher(&hasher, key >> 5, FP_BITS)
}

fn deterministic_payload(key: Key) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(b"AERIS-KV:");
    out.extend_from_slice(&key.to_be_bytes());
    out.extend_from_slice(&(!key).to_le_bytes());
    out
}

/// Child-process driver executed only when `AERIS_CRASH_CHILD=1` is present in the
/// environment. Under normal `cargo test`, this test returns immediately.
#[test]
fn crash_injection_child_driver() {
    if env::var_os(CRASH_CHILD_ENV).is_none() {
        return;
    }

    let dir = env::var(CRASH_DIR_ENV).expect("AERIS_CRASH_DIR must be set for child");
    let cycle: u64 = env::var(CRASH_CYCLE_ENV)
        .expect("AERIS_CRASH_CYCLE must be set for child")
        .parse()
        .expect("AERIS_CRASH_CYCLE must parse as u64");
    let checkpoint_at: Option<usize> = env::var(CRASH_CHECKPOINT_AT_ENV)
        .ok()
        .and_then(|v| v.parse().ok());

    let (engine, _report) =
        CoupledStorageEngine::open_and_recover(&dir).expect("child open_and_recover failed");
    let mut rng = ChaCha8Rng::seed_from_u64(0xC8A5_0000_u64.wrapping_add(cycle.wrapping_mul(997)));
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for iter in 0..CHILD_SAFETY_ITERATION_CAP {
        // Keep keys distinct across cycles by partitioning the top 8 bits by cycle index.
        let key: Key = ((cycle + 1) << 56) | (rng.gen::<u64>() & 0x00FF_FFFF_FFFF_FFFF);
        let payload = deterministic_payload(key);
        let fp = deterministic_fp(key);

        let lsn = engine.put(key, &payload, &fp).expect("child put failed");

        // Emit ACK strictly AFTER engine.put (WAL append + fsync + RocksDB apply) returns.
        writeln!(out, "ACK:{lsn}:{key}").expect("child stdout write failed");
        out.flush().expect("child stdout flush failed");

        if checkpoint_at == Some(iter + 1) {
            engine.checkpoint().expect("child checkpoint failed");
        }
    }
}

/// Executes 5 consecutive violent `child.kill()` (`SIGKILL` / `TerminateProcess`)
/// crash-recovery cycles on a shared storage directory and verifies zero lost keys,
/// exact KV <-> Reverse Map parity, torn-tail truncation, and zero false negatives.
#[test]
fn five_cycle_sigkill_crash_recovery() {
    let dir = TempDir::new().expect("failed to create temp dir");
    let exe = env::current_exe().expect("failed to resolve current test executable");

    let mut all_acked: BTreeMap<Key, Lsn> = BTreeMap::new();
    let mut prev_max_lsn: Lsn = 0;

    for cycle in 0..CRASH_CYCLES {
        let mut cmd = Command::new(&exe);
        cmd.arg("--exact")
            .arg("crash_injection_child_driver")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(CRASH_CHILD_ENV, "1")
            .env(CRASH_DIR_ENV, dir.path())
            .env(CRASH_CYCLE_ENV, cycle.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if cycle == 4 {
            // Exercise mid-cycle checkpoint followed by post-checkpoint writes before kill.
            cmd.env(CRASH_CHECKPOINT_AT_ENV, "12");
        }

        let mut child = cmd.spawn().expect("failed to spawn crash child process");
        let stdout = child.stdout.take().expect("child stdout missing");
        let mut reader = BufReader::new(stdout);

        let mut cycle_acks = 0usize;
        let mut line = String::new();
        while cycle_acks < TARGET_ACKS_PER_CYCLE {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .expect("failed reading child stdout");
            assert!(
                bytes_read > 0,
                "crash child exited prematurely in cycle {cycle} after only {cycle_acks} acks"
            );
            if let Some((lsn, key)) = parse_ack_line(&line) {
                all_acked.insert(key, lsn);
                cycle_acks += 1;
            }
        }

        // Violently terminate child mid-flight (`SIGKILL` on Unix, `TerminateProcess` on Windows).
        child.kill().expect("failed to kill crash child process");
        let status = child.wait().expect("failed waiting for killed child");
        assert!(
            !status.success(),
            "cycle {cycle}: killed child unexpectedly exited with success status {status:?}"
        );

        // Drain any remaining complete ACK lines flushed into the pipe right before death.
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .expect("failed draining child stdout");
            if n == 0 {
                break;
            }
            if line.ends_with('\n') {
                if let Some((lsn, key)) = parse_ack_line(&line) {
                    all_acked.insert(key, lsn);
                }
            }
        }

        // Inject realistic torn-tail WAL corruption artifacts on cycles 1..=3.
        let injected_torn_bytes = inject_torn_tail_for_cycle(dir.path(), cycle, &all_acked);

        // Run full crash recovery and verify all durability invariants.
        let (engine, report) = CoupledStorageEngine::open_and_recover(dir.path())
            .unwrap_or_else(|err| panic!("cycle {cycle} open_and_recover failed: {err}"));

        if injected_torn_bytes > 0 {
            assert!(
                report.truncated_tail_bytes >= injected_torn_bytes,
                "cycle {cycle}: expected at least {injected_torn_bytes} truncated tail bytes, got {}",
                report.truncated_tail_bytes
            );
        }
        assert!(
            report.last_lsn > prev_max_lsn,
            "cycle {cycle}: last_lsn ({}) did not advance past previous ({prev_max_lsn})",
            report.last_lsn
        );
        prev_max_lsn = report.last_lsn;

        let kv_all_keys: Vec<Key> = engine.kv_store().scan_all_keys().unwrap();
        let rm_all_keys: Vec<Key> = engine.reverse_map().scan_all_keys().unwrap();
        let kv_set: BTreeSet<Key> = kv_all_keys.iter().copied().collect();
        let rm_set: BTreeSet<Key> = rm_all_keys.iter().copied().collect();

        // Invariant 1: Exact parity between primary KV store and Reverse Map.
        assert_eq!(
            kv_all_keys, rm_all_keys,
            "cycle {cycle}: KV store and Reverse Map diverged after crash recovery"
        );
        assert_eq!(report.live_key_count, kv_all_keys.len());

        // Invariant 2: 100% of acknowledged keys survived in BOTH KV store and Reverse Map.
        for (&acked_key, &acked_lsn) in &all_acked {
            let expected_val = deterministic_payload(acked_key);
            let actual_val = engine.kv_store().get(acked_key).unwrap();
            assert_eq!(
                actual_val.as_deref(),
                Some(expected_val.as_slice()),
                "cycle {cycle}: LOST acknowledged key {acked_key} (LSN {acked_lsn}) in ReferenceKvStore"
            );
            assert!(
                rm_set.contains(&acked_key),
                "cycle {cycle}: LOST acknowledged key {acked_key} (LSN {acked_lsn}) in RocksDbReverseMap"
            );
            let fp = deterministic_fp(acked_key);
            let fp_keys = engine.reverse_map().get_keys(&fp).unwrap();
            assert!(
                fp_keys.contains(&acked_key),
                "cycle {cycle}: Reverse Map fingerprint entry missing acknowledged key {acked_key}"
            );
        }

        // Invariant 3: Zero false negatives for every live key in ReferenceKvStore.
        for &live_key in &kv_set {
            assert!(
                engine
                    .kv_store()
                    .range_non_empty(QueryRange::point(live_key))
                    .unwrap(),
                "cycle {cycle}: false negative on range_non_empty for live key {live_key}"
            );
            assert!(
                rm_set.contains(&live_key),
                "cycle {cycle}: live KV key {live_key} missing from Reverse Map"
            );
        }
    }

    assert!(
        all_acked.len() >= (CRASH_CYCLES as usize) * TARGET_ACKS_PER_CYCLE,
        "expected at least {} total acknowledged keys across {CRASH_CYCLES} cycles, got {}",
        (CRASH_CYCLES as usize) * TARGET_ACKS_PER_CYCLE,
        all_acked.len()
    );
}

fn parse_ack_line(line: &str) -> Option<(Lsn, Key)> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("ACK:")?;
    let (lsn_str, key_str) = rest.split_once(':')?;
    let lsn: Lsn = lsn_str.parse().ok()?;
    let key: Key = key_str.parse().ok()?;
    Some((lsn, key))
}

/// Appends realistic torn-tail WAL bytes on cycles 1..=3 to simulate an OS write
/// interrupted mid-header, mid-payload, or with a corrupted trailing CRC32.
fn inject_torn_tail_for_cycle(dir: &Path, cycle: u64, acked: &BTreeMap<Key, Lsn>) -> u64 {
    let wal_path = dir.join(WAL_FILE_NAME);
    let next_lsn = acked.values().copied().max().unwrap_or(0) + 10_000;
    let fake_key = 0xDEAD_BEEF_0000_0000 | cycle;
    let fake_op = WalOp::Put {
        key: fake_key,
        value: vec![0xAB; 64],
        fp: deterministic_fp(fake_key),
    };
    let (mut frame, _) = encode_wal_frame(next_lsn, &fake_op).unwrap();

    let torn_bytes: Vec<u8> = match cycle {
        // Cycle 0 & 4: pure SIGKILL / TerminateProcess without synthetic tail bytes
        0 | 4 => return 0,
        // Cycle 1: 11 bytes (torn mid-header)
        1 => {
            let mut partial = Vec::with_capacity(11);
            partial.extend_from_slice(&WAL_MAGIC);
            partial.push(WAL_FORMAT_VERSION);
            partial.extend_from_slice(&[0x40, 0x00, 0x00, 0x00, 0x01, 0x02]);
            partial
        }
        // Cycle 2: valid 21-byte header + only 20 bytes of payload (torn mid-payload)
        2 => frame[..WAL_HEADER_LEN + 20].to_vec(),
        // Cycle 3: full-length frame with corrupted trailing byte (bad CRC32 at EOF)
        3 => {
            let last = frame.len() - 1;
            frame[last] ^= 0xFF;
            frame
        }
        _ => return 0,
    };

    let mut f = OpenOptions::new()
        .append(true)
        .open(&wal_path)
        .expect("failed opening WAL for torn-tail injection");
    f.write_all(&torn_bytes)
        .expect("failed writing torn-tail bytes");
    f.sync_all().expect("failed syncing torn-tail bytes");
    torn_bytes.len() as u64
}
