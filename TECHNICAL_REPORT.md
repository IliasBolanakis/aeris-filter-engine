# Aeris-Based Optimized Range Filtering System — Technical Engineering Report

**Author:** Aeris Filtering System Team  
**Target Platform:** Bare-Metal `x86_64` (Windows `x86_64-pc-windows-msvc`, 13th Gen Intel Core i7-13650HX, 14P/20L cores, `24.00 MiB` L3 cache, `63.67 GiB` DDR5)  
**Matched Toolchains:** `rustc 1.99.0` (`b940084d7`, Fat-LTO, `codegen-units=1`, `-C target-cpu=native`) vs. LLVM `clang`/`clang++ 23.1.6` (`-O3 -march=native -flto -msse4.2 -DNDEBUG`)  
**Workspace:** [Filter/Cargo.toml](Filter/Cargo.toml) (`aeris-types`, `aeris-core`, `aeris-storage`, `aeris-control`, `aeris-scale`, `aeris-sidecar`, `python-sidecar`)

---

## 1. Executive Summary & Correctness Scorecard

Range filters sit in front of LSM-trees and B-trees to reject empty range queries before incurring SSD I/O. The SIGMOD 2026 **Aeris Filter** (*Chesetti et al.*) established a range filter that simultaneously achieves spatial robustness, strong and monotonic adaptivity, and deamortized InfiniFilter expansion. Moving Aeris from a single-threaded research prototype into a self-tuning, crash-consistent, multi-core database subsystem exposed five operational failure modes: static range-width ($R$) degradation, loss of monotonicity across $R$ changes, synchronous expansion stalls without contraction, uncoupled KV/reverse-map crash desynchronization, and coarse-locked single-filter scaling.

This report documents the design, implementation, microarchitectural optimization, and empirical verification of the **Aeris-Based Optimized Range Filtering System**—a clean-room Rust and Python architecture spanning six Rust crates and an isolated out-of-process gRPC ML/RL sidecar.

### Master Correctness & Scale Scorecard

Every correctness invariant was verified across two independent bare-metal benchmark corpora: **48 sequential parity runs** against the untouched Official SIGMOD 2026 C++ implementation up to **`5x10^8` (500,000,000) keys** ([PAPER_VS_AERIS_COMPARISON.md](PAPER_VS_AERIS_COMPARISON.md), [parity_results/](parity_results/)), and **43 internal ablation metrics** across 6 subsystem suites ([Filter/ABLATION_REPORT.md](Filter/ABLATION_REPORT.md), generated `2026-11-24T13:50:37Z`).

| Verification Gate | Measured Result | Requirement | Status |
| :--- | ---: | :--- | :--- |
| **Total Verified Keys (`100%` FN Scan)** | **`>2.7B` keys** across 48 parity + 43 ablation runs | Every inserted key queried | **PASS** |
| **Total False Negatives (`fn_count`)** | **`0`** (parity: `0` / `48`; ablation: `0` / `43`) | `== 0` | **PASS** |
| **Total Repeated False Positives** | **`0`** across all adaptive, adversarial, and rebuild runs | `== 0` | **PASS** |
| **Max Scale Verified (`bulk_load` & `dynamic_insert`)** | **`5x10^8` keys** (`2.16 GiB` filter, `q=30`) | `<= 48.00 GiB` RSS | **PASS** |
| **Max Observed Peak RSS (All 48 Parity Runs)** | **`11.20 GiB`** (`11,471.4 MiB`, `+0.66 MiB` vs. C++) | Zero OS swap thrashing | **PASS** |
| **DRAM Spill Beyond Host L3 (`24.00 MiB`)** | **`27.53–44.31 MB`** (Sec. 5.2), **`42.25 MiB`** (ablation), **`2.16 GiB`** (`5e8`) | Strictly `> 24.00 MiB` | **CONFIRMED** |
| **`SIGKILL` Crash Recovery (`5` Cycles)** | **`0.00%`** lost/desync keys, **`0`** post-rebuild FNs | `== 0` (vs. `21.66%` uncoupled) | **PASS** |
| **Unsafe Surface Confinement** | Strictly [`simd.rs`](Filter/crates/aeris-core/src/simd.rs) & [`epoch_cell.rs`](Filter/crates/aeris-scale/src/epoch_cell.rs) | Zero `unsafe` added during optimization | **PASS** |

---

## 2. Theoretical Foundation & Gaps in the Paper Prototype

### 2.1 Theoretical Foundation

The Aeris Filter operates over a 64-bit integer key universe partitioned into contiguous intervals of width $P = 2^{\lceil \log_2 R \rceil}$, where $R$ is the configured maximum range query length. Each key $k$ decomposes into a partition prefix $p(k) = \lfloor k / P \rfloor$ and an $m$-bit offset suffix (*memento*) $m(k) = k \bmod P$, with $m = \lceil \log_2 R \rceil$.

An array of $N = 2^q$ Quotient Filter slots is organized into 64-slot blocks (each carrying an 8-bit cluster offset) using Robin Hood hashing and constant-time `RANK`/`SELECT` bitmask operations. Each slot stores $(r + m)$ payload bits plus 3 metadata bits (`occupied`, `runend`, `extension`), yielding a total space footprint of:

$$\alpha n \left(\log_2 \frac{R\alpha}{\epsilon} + 4.125\right) \text{ bits}$$

This matches Goswami's spatial-robustness lower bound of $\Omega(\log(R/\epsilon))$ bits per key. Nonempty partitions are represented as **keepsake boxes** keyed by an initial $l = (q + r)$-bit hash fingerprint $h(p(k))$. When a false positive occurs, the colliding keepsake box is split by extending its fingerprint in discrete increments (first $+m$ bits, then $+(r+m)$ bits per subsequent extension), using the dual-purposed `extension` bit (`extension=1, runend=0` marks an extension slot; `extension=1, runend=1` terminates a quotient run).

Under a fixed per-slot payload budget $C$, remainder bits and partition width obey the conservation invariant:

$$r + \log_2 P = C$$

To support deamortized $2\times$ expansion without full rebuilds, Aeris embeds an **InfiniFilter unary age counter** (`00..1`) in each keepsake box. Each $2\times$ expansion shifts one bit from the remainder into the canonical slot quotient and increments the unary age counter. Only the $\approx 2^{-r}$ fraction of fingerprints that exhaust their remainder bits are selectively rejuvenated from an on-disk write-optimized reverse map. Preserving strong and monotonic adaptivity across expansions requires enlarging the remainder to:

$$r = \log_2 \frac{1}{\epsilon} + \log_2 \log_2 \frac{1}{\epsilon} + 1$$

### 2.2 Five Production Gaps in the Paper Prototype

Integrating the standalone research prototype into a multi-core DBMS exposes five structural gaps (formalized in [Docs/Aeris_Based_Filtering_System.pdf](Docs/Aeris_Based_Filtering_System.pdf) and analyzed in [AERIS_MODIFICATIONS_AND_IMPACT.md](AERIS_MODIFICATIONS_AND_IMPACT.md)):

1. **Static Range Width ($R$):** Queries of length $L > R$ must be decomposed into $\lceil L / P \rceil + 1$ subqueries. Under a static small $R$, widening workloads suffer linear CPU fanout and linear FPR blowup (measured at `0.195788`–`0.212430` FPR at $R=1024$ against a target $\epsilon = 0.0625$).
2. **Loss of Monotonicity Across $R$ Changes:** Reconfiguring $P$ alters $p(k) = \lfloor k/P \rfloor$ and invalidates every existing partition hash $h(p(k))$. A naive rebuild from the reverse map discards prior adaptations, allowing previously resolved false positives to recur.
3. **Synchronous Expansion Stalls & Missing Contraction:** Approaching $\alpha = 95\%$ triggers exponential Robin Hood cluster growth and synchronous `expand()` pauses (`140.95 ms` measured), while bulk deletions never reclaim filter memory.
4. **Uncoupled Reverse Map Crash Desynchronization:** Writing to the primary KV store and the on-disk reverse map independently allows `SIGKILL` crashes to leave acknowledged keys absent from the reverse map. Measured across 5 crash cycles, uncoupled writes lost `21.66%` of reverse-map entries (`1,098` keys), producing `1,098` silent post-rebuild False Negatives.
5. **Coarse Concurrency, Monolithic Doubling, & Reactive-Only Resource Control:** Single-filter coarse locking collapses under multi-core contention (`615k ops/s` at 8T), `RwLock` blocks readers for `90.23 ms` during expansion, skewed writes force global $2\times$ RAM doubling, and reactive heuristics cannot pre-warm seasonal surges or allocate memory asymmetrically across tenants.

---

## 3. System Architecture & Engineering Decisions

The system is structured into five strictly separated planes across six Rust crates plus an out-of-process Python sidecar. Pure domain types and trait boundaries live in [`aeris-types`](Filter/crates/aeris-types/src/lib.rs).

### 3.1 Data Plane (`aeris-core`)

Implemented in [Filter/crates/aeris-core/src/](Filter/crates/aeris-core/src/):

- **Interleaved 64-Slot AoS Block Layout ([`chunk.rs`](Filter/crates/aeris-core/src/chunk.rs)):** Early Rust prototypes used split Structure-of-Arrays (SoA) metadata and payload arrays, causing a second TLB/cache miss per probe and `155 B/key` build RSS. Replacing SoA with contiguous interleaved 64-slot AoS blocks—each exactly `25 + 8 * (r + 1 + m)` bytes (`1 B` offset + `3 x 8 B` bitmasks + `8 * slot_bits B` bit-packed payloads, plus `+8` trailing guard bytes for unaligned `u64` reads)—co-locates metadata and payloads in the same cache-line neighborhood with zero new `unsafe` code (`unsafe` remains isolated to x86_64 `POPCNT`/`PDEP` intrinsics in [`simd.rs`](Filter/crates/aeris-core/src/simd.rs)). This lifted pre-expansion DRAM query throughput by `3.94x` (`3.45M` → `13.58M ops/s` at $\epsilon = 2^{-4}$) and cut `p99.9` latency from `4.40 us` to `400 ns`.
- **Word-Shifted Zero-Allocation `insert()` ([`chunk.rs`](Filter/crates/aeris-core/src/chunk.rs), [`filter.rs`](Filter/crates/aeris-core/src/filter.rs), [`fingerprint.rs`](Filter/crates/aeris-types/src/fingerprint.rs)):** Dynamic insertion eliminates heap allocation and backward chunk scans: `insert_single_slot_new_run` performs a direct single-block write when `occ_count == 0 && offset == 0`; `shift_partial_block_payload_right` shifts intra-block and boundary slots in 64-bit word steps; `Fingerprint::from_u64_prefix` is a panic-free `const fn` enabling LLVM dead-code elimination when callers discard return values; and integer `max_occupied_slots` replaces floating-point load checks on the hot path.
- **Block-Sequential `BulkTableWriter` & Streaming 8-Bit LSD Radix `bulk_load` ([`chunk.rs`](Filter/crates/aeris-core/src/chunk.rs), [`filter.rs`](Filter/crates/aeris-core/src/filter.rs), [`traits.rs`](Filter/crates/aeris-types/src/traits.rs)):** Bulk loading packs keys into `8 B/key` and sorts them with an 8-bit LSD radix sort equipped with a 16 KiB L1-resident write-combining scatter buffer (`256 x 8 u64`) above `65,536` keys. Instead of per-slot DRAM read-modify-writes, `BulkTableWriter` accumulates `occupied`, `runend`, `extension`, and payload `u64` words in stack/register buffers and flushes each 64-slot AoS block sequentially once. Adding a `dyn`-safe `ReverseMapStore::is_noop(&self) -> bool` method (default `false`, `true` for `NoopReverseMap`) bypasses Stage-5 `(Fingerprint, Key)` batch construction on standalone bulk loads. Together, these changes boosted Rust `bulk_load` by `1.41x–1.73x` over Pre-Boost (`5.7x–7.5x` over Pre-Opt), surpassing Official C++ from `10^6` through `5x10^8` keys at identical `11.20 GiB` RSS.
- **In-Place Stack-Buffered `adapt()`, Monotonic Cluster Scan, & Branch-Predictable Probing ([`filter.rs`](Filter/crates/aeris-core/src/filter.rs), [`chunk.rs`](Filter/crates/aeris-core/src/chunk.rs), [`keepsake.rs`](Filter/crates/aeris-core/src/keepsake.rs)):** Replacing the 4-phase `BTreeMap`/`BTreeSet` adaptation overlay with a zero-allocation stack-buffered single-box fast path (`try_adapt_single_box_fast`) plus a flat-`Vec` general overlay preserves colliding-first/query-partition-last ordering and atomic `FilterFull` rollback. `find_first_empty_slot` uses a strictly forward monotonic `(occupied, qrun_end)` jump scan instead of rescanning from block start; `replace_region_known_run` avoids redundant run-boundary discovery; `query_range` unrolls 1–2 partition queries; and `box_matches_query_hot` collapses age-tag checks into a single predictable `!has_aged` branch.

### 3.2 Storage Plane (`aeris-storage`)

Implemented in [Filter/crates/aeris-storage/src/](Filter/crates/aeris-storage/src/):

- **CRC32-Protected `CoupledWal` ([`wal.rs`](Filter/crates/aeris-storage/src/wal.rs), [`engine.rs`](Filter/crates/aeris-storage/src/engine.rs)):** Every key mutation is framed with a `crc32fast` checksum and written to `CoupledWal` before updating both the RocksDB primary KV store ([`kv_store.rs`](Filter/crates/aeris-storage/src/kv_store.rs)) and the write-optimized Reverse Map ([`reverse_map.rs`](Filter/crates/aeris-storage/src/reverse_map.rs)). On restart after `SIGKILL`, torn-tail recovery replays all valid CRC32 frames into both stores and truncates any partially written tail record.
- **Default Group-Commit Batching (`B = 1,000`):** `StorageConfig::default()` enforces `WalSyncPolicy::GroupCommit { batch_size: 1_000 }` paired with `CoupledStorageEngine::put_batch()` and `flush_wal()`. Per-operation `fsync` (`StorageConfig::strict_per_op_fsync()`) remains available as an explicit opt-in hook for single-op durability audits and crash-injection tests ([`crash_injection.rs`](Filter/crates/aeris-storage/tests/crash_injection.rs)).

### 3.3 Control Plane (`aeris-control`)

Implemented in [Filter/crates/aeris-control/src/](Filter/crates/aeris-control/src/):

- **Dynamic-$R$ Optimization & Static-$R$ Bypass ([`estimator.rs`](Filter/crates/aeris-control/src/estimator.rs), [`optimizer.rs`](Filter/crates/aeris-control/src/optimizer.rs)):** `SlidingWindowEstimator` tracks query widths with hysteresis and median-anchored outlier clipping; `DiscreteCostOptimizer` evaluates all candidate $m = \log_2 P$ values under $r + \log_2 P = C$ in `< 5 us`, balancing collision probability ($2^{-r}$) against subquery fanout ($\lceil L/P \rceil + 1$). Workloads with guaranteed fixed $R$ can enable `EstimatorConfig::with_enabled(false)` / `ControlledEngineConfig::static_r_only()` to eliminate estimator recording overhead entirely (`25.13M QPS` in Phase 0, `+8.0%` over C++).
- **Async Rebuild & Persistent `FalsePositiveLog` Replay ([`rebuild.rs`](Filter/crates/aeris-control/src/rebuild.rs), [`fp_log.rs`](Filter/crates/aeris-control/src/fp_log.rs)):** When $m$ changes, the coordinator scans the Reverse Map off the hot path to bulk-load $F_{\text{new}}$ while a `DeltaBuffer` records concurrent writes. Every confirmed FP interval $[a, b]$ is persisted in a CRC32-framed append-only `FalsePositiveLog` and replayed onto $F_{\text{new}}$ before atomic swap (`277.60 us` total for `1,044` entries in A/B 1; `648 us` for `1,038` entries in parity Shifting-$R$), guaranteeing `0` repeated FPs across $R$ transitions.
- **Proactive EWMA Trigger & Live Contraction ([`trigger.rs`](Filter/crates/aeris-control/src/trigger.rs)):** Compares estimated time-to-full $T_{\text{full}} = N \cdot (0.95 - \alpha) / \text{IngestionRate}_{\text{EWMA}}$ against estimated rebuild duration $T_{\text{rebuild}} = |\text{ReverseMap}| / \text{BulkLoadSpeed}$, initiating background expansion ahead of the $\alpha = 95\%$ cliff (firing at $\alpha = 0.8503$, swapping via RCU at $\alpha = 0.8925$, `0` reactive fallbacks). After sustained deletes, live contraction halves the table ($q \to q - 1$), reclaiming `49.98%` RAM with `0` FN and `0` repeated FPs.

### 3.4 Scale Plane (`aeris-scale`)

Implemented in [Filter/crates/aeris-scale/src/](Filter/crates/aeris-scale/src/):

- **Chunk Locks & Epoch-RCU Swap ([`chunk_locks.rs`](Filter/crates/aeris-scale/src/chunk_locks.rs), [`epoch_cell.rs`](Filter/crates/aeris-scale/src/epoch_cell.rs), [`rcu_filter.rs`](Filter/crates/aeris-scale/src/rcu_filter.rs)):** Striped 64-slot chunk locks serialize concurrent writers only on colliding clusters, while `crossbeam-epoch` RCU allows readers to traverse filter snapshots wait-free across live $2\times$ expansions.
- **Dynamic Range Sharding & 1T Direct Bypass ([`sharding.rs`](Filter/crates/aeris-scale/src/sharding.rs)):** Partitioning the keyspace into independent $2^{q_i}$ range shards allows hot-spot splits and cold-shard merges without doubling the entire filter. To protect single-threaded callers, `ShardingConfig::single_shard_bypass` (default `true`) skips binary-search routing when `shard_count == 1`, and `&mut self` direct paths (`query_point_direct`, `query_range_direct`, `insert_direct`) bypass epoch pinning and lock acquisition.

### 3.5 Isolated Advisory Plane (`aeris-sidecar` + `python-sidecar`)

Implemented in [Filter/crates/aeris-sidecar/src/](Filter/crates/aeris-sidecar/src/) and [Filter/python-sidecar/aeris_sidecar/](Filter/python-sidecar/aeris_sidecar/):

- **Strict Out-of-Process Isolation ([`server.py`](Filter/python-sidecar/aeris_sidecar/server.py), [`client.rs`](Filter/crates/aeris-sidecar/src/client.rs), [`transport.rs`](Filter/crates/aeris-sidecar/src/transport.rs)):** The Python gRPC server (`ProposeTuning`, `AllocateMemory`, `HealthCheck` over Unix Domain Sockets or loopback TCP with a bounded LRU `request_id` idempotency cache) runs completely off the Data Plane hot path. It hosts a seasonal workload forecaster ([`forecaster.py`](Filter/python-sidecar/aeris_sidecar/forecaster.py)) for pre-warming expansion before traffic spikes and an offline contextual bandit ([`bandit.py`](Filter/python-sidecar/aeris_sidecar/bandit.py)) for asymmetric multi-tenant RAM allocation.
- **Deterministic `SafetyGuardrail` & Instant Native Fallback ([`guardrail.rs`](Filter/crates/aeris-sidecar/src/guardrail.rs), [`advisor.rs`](Filter/crates/aeris-sidecar/src/advisor.rs)):** Every sidecar recommendation is validated deterministically in Rust against bounds ($r \ge r_{\min}(\epsilon)$, $r + m = C$, valid $\alpha \in [0.50, 0.90]$, finite non-overflowing memory budgets, unique shard IDs). On invalid proposals, RPC timeouts, or abrupt sidecar `SIGKILL`, the advisor falls back to the native Rust Control Plane in `6.50 us` p99 (`500 ns` p50) with zero Data Plane stalls (`78,469,103` concurrent queries at `p50=100 ns, p99=200 ns`).

---

## 4. Rejected Design Alternatives

Four candidate techniques were explicitly evaluated and rejected during architecture design ([Docs/Aeris_Based_Filtering_System.pdf](Docs/Aeris_Based_Filtering_System.pdf), Section 4):

| Rejected Alternative | Technical Reason for Rejection |
| :--- | :--- |
| **ML Inference in the Query Hot Path** | Imposes $>10\times$ latency overhead onto sub-100 ns bitwise RAM probes and exposes the query path to adversarial latency and model-evasion attacks. |
| **Online RL for $r$ vs. $\alpha$ in RAM** | Exploration noise introduces production FPR and latency instability for negligible gain ($\approx 0.2\%$ shift in load factor $\alpha$). |
| **Arbitrary Non-Power-of-2 Resizing (e.g., `1.4x`)** | Breaks InfiniFilter quotient bit-splitting ($q \to q+1$), destroying deamortized expansion and forcing full reverse-map rebuilds on every resize. |
| **Multi-$R$ Tiered Filters** | Doubles memory consumption and multiplies every `INSERT`/`DELETE` across redundant filter tiers; Dynamic-$R$ with background rebuild achieves multi-width adaptation within a single RAM budget. |

---

## 5. Experimental Methodology & Fair Parity Protocol

### 5.1 Host Hardware & Execution Discipline
- **CPU & Cache:** 13th Gen Intel Core i7-13650HX (`14` physical / `20` logical cores), **`24.00 MiB` (`24,576 KiB`) L3 cache**. Every DRAM-bound benchmark strictly exceeds L3 (`25.49–44.31 MB` in Section 5.2, `42.25 MiB` in internal ablations, up to `2.16 GiB` at `5x10^8` keys).
- **Memory & Safety Ceiling:** `63.67 GiB` DDR5 (`45.98–46.12 GiB` free at suite start), with a hard `48.00 GiB` RSS ceiling (peak observed RSS across all runs: `11.20 GiB`, zero OS swap thrashing).
- **Sequential Bare-Metal Execution:** All 48 parity runs executed natively on Windows `x86_64` (no WSL2/VM) and strictly sequentially so C++ and Rust never contended for L3 cache or DDR5 bandwidth. Identical pre-generated binary key (`*.keys.bin`) and query (`*.queries.bin`) traces from [`trace_gen.rs`](Filter/src/bin/trace_gen.rs) were loaded into RAM before timers started.
- **Untouched C++ Baseline:** Zero files in the Official SIGMOD 2026 C++ algorithm tree (`adaptiveqf/src/{gqf.c,hashutil.c,ll_table.c,partitioned_counter.c,arqf_inmem.cc}`) were modified; only minimal POSIX shims were supplied for native Windows compilation.

### 5.2 Matched LLVM Optimization Tiers
- **Official C++ (`cpp_official`):** LLVM `clang`/`clang++ 23.1.6` (`x86_64-pc-windows-msvc`, `lld`) with `-O3 -march=native -flto -msse4.2 -D__SSE4_2_ -DNDEBUG`.
- **Rust (`rust_naive`, `rust_optimized`):** `rustc 1.99.0` (`b940084d7`, `x86_64-pc-windows-msvc`) via [`paper_parity.rs`](Filter/src/bin/paper_parity.rs) with `RUSTFLAGS="-C target-cpu=native"`, `--release`, `profile.release.lto="fat"`, and `profile.release.codegen-units=1`.

### 5.3 Timer Granularity & Provenance Transparency
- **Throughput vs. Single-Op Percentiles:** Aggregate throughput (`QPS`, `keys/s`) is measured over wall-clock time across `500k–500M` operations on both sides and is directly comparable. Single-op latency percentiles (`p50`, `p99`) use calibrated invariant TSC (`RDTSC`, sub-ns precision) in C++ and `std::time::Instant` (Windows `QueryPerformanceCounter`, `100 ns` tick quantum) in Rust, which rounds sub-50 ns operations to `0 ns` or `100 ns`.
- **Benchmark Corpus Provenance:** The 43 internal ablation metrics in [Filter/ABLATION_REPORT.md](Filter/ABLATION_REPORT.md) were generated at **`2026-11-24T13:50:37Z`** (reflecting Stage-1 AoS/radix/control/scale/WAL optimizations). The 48 cross-language parity runs in [PAPER_VS_AERIS_COMPARISON.md](PAPER_VS_AERIS_COMPARISON.md) and [parity_results/](parity_results/) reflect the subsequent **Stage-2 (`/boost`)** pass (in-place `adapt()`, monotonic cluster scan, `BulkTableWriter` + `is_noop()`, and word-shifted `insert()`). Both corpora are reported with exact provenance so no numbers are conflated.

---

## 6. Empirical Evaluation

### 6.1 SIGMOD 2026 Section 5.2 — Zipfian ($\alpha=1.5$) Correlated ($D=0.8$) & Uncorrelated Queries (90% LF, 15.1M Keys)

Configuration: `q = 24` (`N = 16,777,216` slots), `15,099,494` unique keys (`90.00%` load factor), `1,000,000` queries, `r = 7` ($\epsilon = 2^{-4} = 0.0625$), adaptive in-memory reverse map. Filter sizes (`25.49–44.31 MB`) spill L3 by `1.06x–1.85x`.

| Workload ($\alpha=1.5$, 90% LF) | Official C++ QPS | Rust Pre-Boost → Post-Boost QPS | Post-Boost vs. C++ | C++ / Rust p99 (ns) | C++ / Rust FPR | C++ / Rust bpk | FN / Rep. |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | :---: |
| **Point Correlated** ($R=1, m=2$) | `24.46M` | `30.89M` → **`30.06M`** | **`+22.9%` (1.23x)** | `213.0 / 300` | `0.000337 / 0.000329` | `13.51 / 14.58` | `0 / 0` |
| **Point Uncorrelated** ($R=1, m=2$) | `42.54M` | `68.44M` → **`71.00M`** | **`+66.9%` (1.67x)** | `107.0 / 100` | `0.000004 / 0.000008` | `13.51 / 14.58` | `0 / 0` |
| **Short Range Correlated** ($R=32, m=5$) | `16.38M` | `10.97M` → **`15.50M`** (`+41.3%`) | `-5.3%` (`0.95x`) | `278.6 / 300` | `0.001478 / 0.001423` | `16.85 / 17.92` | `0 / 0` |
| **Short Range Uncorrelated** ($R=32, m=5$) | `31.26M` | `28.96M` → **`29.12M`** (`+0.5%`) | `-6.8%` (`0.93x`) | `137.0 / 100` | `0.000024 / 0.000027` | `16.85 / 17.92` | `0 / 0` |
| **Long Range Correlated** ($R=1024, m=10$) | `15.28M` | `11.67M` → **`13.92M`** (`+19.2%`) | `-8.9%` (`0.91x`) | `297.9 / 300` | `0.001425 / 0.001473` | `22.42 / 23.47` | `0 / 0` |
| **Long Range Uncorrelated** ($R=1024, m=10$) | `31.81M` | `27.42M` → **`30.16M`** (`+10.0%`) | `-5.2%` (`0.95x`) | `139.8 / 100` | `0.000033 / 0.000022` | `22.42 / 23.47` | `0 / 0` |

Pass A (`try_adapt_single_box_fast`, monotonic cluster scan, predictable `!has_aged`) closed the Pre-Boost correlated-range gap from `0.67x–0.76x` to `0.91x–0.95x` of C++, while Rust leads point queries by `+22.9%` to `+66.9%`. The constant `+1.06–1.08 bpk` space delta is strictly the 1-bit InfiniFilter unary age sentinel omitted by C++ `gqf.c`.

### 6.2 Pure Filter Scaling (`10^5` to `5x10^8` Keys, `r=7, m=5, R=32`)

Standalone `QF*` (C++) vs. `AerisFilter` with `NoopReverseMap` (Rust), followed by `500k–1M` read-only empty range queries and `100%` key FN verification (`0` FN at every tier):

| Scale (`N`, `q`, LF) | C++ Bulk (M/s) | Rust Bulk Pre→Post (M/s) | Rust/C++ Bulk | C++ / Rust Bulk QPS (M/s) | C++ Dyn. Ins. (M/s) | Rust Dyn. Ins. Pre→Post (M/s) | C++/Rust Ins. | C++ / Rust Bulk Peak RSS (MB) |
| :--- | ---: | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| **`10^5`** (`q=17`, `76.3%`) | `43.32` | `24.30` → `42.12` (`1.73x`) | `0.97x` | `25.96` / **`30.65`** (`1.18x`) | `13.91` | `10.40` → `12.21` (`+17.4%`) | `1.14x` | `17.6 / 34.0` |
| **`10^6`** (`q=21`, `47.7%`) | `35.37` | `25.83` → **`36.50`** (`1.41x`) | **`1.03x`** | `27.57` / **`31.48`** (`1.14x`) | `24.74` | `20.31` → `22.78` (`+12.2%`) | `1.09x` | `38.0 / 40.9` |
| **`10^7`** (`q=24`, `59.6%`) | `37.51` | `27.05` → **`41.01`** (`1.52x`) | **`1.09x`** | `13.56` / **`14.24`** (`1.05x`) | `13.11` | `9.44` → `10.46` (`+10.8%`) | `1.25x` | `255.4 / 256.2` |
| **`10^8`** (`q=27`, `74.5%`) | `37.34` | `26.82` → **`40.71`** (`1.52x`) | **`1.09x`** | `8.72` / **`9.22`** (`1.06x`) | `9.86` | `6.76` → `7.75` (`+14.6%`) | `1.27x` | `2,315.3 / 2,316.1` |
| **`2x10^8`** (`q=28`, `74.5%`) | `38.31` | `27.60` → **`40.79`** (`1.48x`) | **`1.06x`** | `8.83` / **`8.87`** (`1.00x`) | `9.71` | `6.59` → `7.57` (`+14.9%`) | `1.28x` | `4,604.2 / 4,604.9` |
| **`3x10^8`** (`q=29`, `55.9%`) | `33.39` | `26.98` → **`38.64`** (`1.43x`) | **`1.16x`** | `11.28` / **`11.43`** (`1.01x`) | `10.49` | `6.77` → `7.85` (`+16.0%`) | `1.34x` | `6,893.0 / 6,893.7` |
| **`5x10^8`** (`q=30`, `46.6%`) | `33.90` | `24.14` → **`35.17`** (`1.46x`) | **`1.04x`** | `12.69` / `11.59` (`0.91x`) | `10.79` | `6.70` → `6.41` (`-4.3%`*) | `1.68x` | `11,470.7 / 11,471.4` |

*\*At `5x10^8` keys (`2.16 GiB` filter + `4.0 GiB` key buffer), random `dynamic_insert` is TLB page-table-walk bound (`~156 ns/key`, within 4% run-to-run thermal variance vs. Pre-Boost's `6.70M`), while `10^5`–`3x10^8` gained `+10.8%` to `+17.4%` with `p99` dropping from `400` to `300 ns` at `10^8`–`3x10^8`.*

### 6.3 Adversarial 10% Repeat-FP Probing & Shifting-$R$ Parity

- **Adversarial 10% Repeat-FP (`15.1M` keys, 90% LF, `500k` queries, `49,998` targeted reprobes):** Post-Boost Rust jumped `+21.5%` (`6.19M` → **`7.52M QPS`** vs. C++ `7.62M`, **`98.7%` parity**), with `3,125` live adaptations, `0.006250` FPR (vs. C++ `0.006292`), **`0` repeated FPs**, and **`0` FN**.
- **Shifting-$R$ ($16 \to 256 \to 32 \to 1024$, `1M` keys, `2M` queries):**

| Implementation & Budget | Steady QPS (M/s) | Phase-0 ($R=16$) QPS | Phase-3 ($R=1024$) QPS | Overall p99 (ns) | Phase-3 p99 (ns) | Overall / Phase-3 FPR | FN / Rep. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Official C++ (`C=17`, Static `m=4`)** | `2.54` | `23.27M` | `0.873M` | `1,443.0` | `1,861.8` | `0.001182` / `0.003664` | `0 / 0` |
| **Official C++ (`C=11`, Paper-Min `m=4`)** | `2.02` | `22.44M` | `0.679M` | `3,038.3` | `3,610.2` | `0.066030` / **`0.195788`** | `0 / 0` |
| **Naive Rust (`C=11`, Static `m=4`)** | **`2.24`** (`+10.7%`) | — | — | `3,200` (was `15,800`) | — | `0.066171` / — | `0 / 0` |
| **Optimized Rust — Static Bypass (`C=17`)** | `3.16` (`+24.2%`) | **`25.13M`** (`+8.0%`) | `1.090M` | `1,200` | `1,400` | `0.001214` / `0.003784` | `0 / 0` |
| **Optimized Rust — Dynamic-$R$ (`C=17`, `m: 4→6→5→8`)** | **`5.47`** (**`2.15x–2.71x`**) | `13.95M` | **`4.062M`** (**`4.66x–5.98x`**) | **`300`** (**`4.81x–10.1x`**) | **`300.0`** (**`6.21x–12.03x`**) | `0.001200` / **`0.003736`** | `0 / 0` |

Total synchronous 3-rebuild time dropped to `750.0 ms` (`~250 ms`/rebuild), with `1,038` `FalsePositiveLog` entries replayed in `648 us` (`0` repeated FPs). Under heavy static `C=11` adaptation (`132,342` adaptations), Pass A lifted Naive Rust by `+132.2%` (`0.96M` → `2.24M QPS`), surpassing Official C++ (`2.02M QPS`) and cutting `p99` nearly `5x` (`15,800` → `3,200 ns`).

### 6.4 Evolution Summary (Pre-Opt → Pre-Boost → Post-Boost vs. Official C++)

| Dimension | Official C++ | Pre-Opt Rust | Pre-Boost Rust | Post-Boost Rust |
| :--- | :--- | :--- | :--- | :--- |
| **Bulk Load (`10^6–5x10^8`)** | `33.39M–38.31M/s` | `5.42M–6.37M/s` (`2e8` cap) | `24.14M–27.60M/s` | **`35.17M–41.01M/s`** (`1.03x–1.16x` C++) |
| **Dynamic Insert (`10^5–5x10^8`)** | **`9.71M–24.74M/s`** | `1.98M–7.41M/s` | `6.59M–20.31M/s` | `6.41M–22.78M/s` (`+11%–17%` up to `3e8`) |
| **Correlated Range (15.1M, 90% LF)** | `15.28M–16.38M QPS` | `13.24M–14.26M QPS` | `10.97M–11.67M QPS` | `13.92M–15.50M QPS` (`0.91x–0.95x` C++) |
| **Adversarial 10% Repeat-FP** | `7.62M QPS` | `6.06M QPS` | `6.19M QPS` | `7.52M QPS` (`0.99x` C++) |
| **High-Adapt Static `C=11`** | `2.02M QPS` | `0.94M QPS` | `0.96M QPS` | **`2.24M QPS`** (`1.11x` C++) |

### 6.5 Internal Ablation Summary (All 6 Suites, `Filter/ABLATION_REPORT.md`, `2026-11-24T13:50:37Z`)

| Suite | Verdict | Headline Measured Results (`0` FN, `0` Repeated FPs Across All 43 Metrics) |
| :--- | :--- | :--- |
| **Core DRAM Audit** (`10M` keys, `5M` queries, `32.25–42.25 MiB`) | **Measurable Gain** (`8/8`) | Pre-exp FPR `13.54x` ($\epsilon=2^{-4}$) / `27.59x` ($\epsilon=2^{-8}$) below target; post-8x-exp FPR `3.40x` / `5.94x` below target; adversarial 10% repeat-FP FPR **`84.70x` lower** (`0.100456` → `0.001186`); DRAM throughput `13.58M–15.05M ops/s` (`p99=300 ns`). |
| **A/B 1: Dynamic-$R$ + FP Log** (`1M` keys, `2M` queries) | **Measurable Gain** (`10` gain, `1` reg.) | **`+407.76%` (`5.08x`)** overall QPS (`3.24M` → `16.47M ops/s`); **`-83.33%` (`6.00x`)** p99 (`1.20 us` → `200 ns`); **`+939.93%` (`10.40x`)** at $R=1024$; **`57.08x`** lower FPR vs. `C=11`; mean RocksDB rebuild `1.442 s`; total FP replay **`277.60 us`** (`1,044` entries). Phase-0 active estimator tax `-26.64%` (eliminated by `static_r_only()`). |
| **A/B 2: Proactive EWMA + Contraction** (`1M` keys) | **Measurable Gain** (`2` gain, `2` neut., `0` reg.) | **`-94.86%` (`19.45x`)** max insert stall (`140.95 ms` → `7.25 ms`); **`-49.98%` (`2.00x`)** RAM after 70% deletes (`5.28` → `2.64 MiB`, `q=21→20`); transition insert p99 `-4.30%` (`9.30` → `8.90 us`) and p99.9 `+3.95%` (neutral). |
| **A/B 3: WAL-Coupled Storage** (`B=1,000` default) | **Expected Durability Overhead** | **`SIGKILL` loss/desync `-100.00%`** (`21.66%` / `1,098` FNs → **`0.00%`** / `0` FNs across 5 cycles). Default `B=1,000` group-commit hits **`149,059 ops/s`** (`-51.26%` / `2.05x` vs. uncoupled `305,850 ops/s`, **`109.7x` faster** than opt-in per-op `fsync` at `1,359 ops/s`). |
| **A/B 4: Scale Plane** (`5M–7M` keys, 90R/10W, 8-Shard + RCU) | **Measurable Gain** (`5` gain, `1` neut., `1` reg.) | **`+76.04%` (`1.76x`)** @ 2T, **`+489.07%` (`5.89x`)** @ 4T, **`+896.06%` (`9.96x`)** @ 8T (`6.12M` vs. `615k ops/s`); live-expand reader `p99.9` **`44.00x` lower** (`13.20 us` → `300 ns`, max stall **`1,234x` lower**: `90.23 ms` → `73.1 us`, `6.36M` vs. `10k` queries served); hot-spot RAM **`-13.83%`** (`1.32` → `1.14 MiB`) with neutral write p99 (`+2.94%`). 1T 8-shard overhead `-20.88%`. |
| **A/B 5: Isolated ML/RL Sidecar** (`1M` keys) | **Measurable Gain** (`6/6`) | **`-99.56%` (`228.14x`)** seasonal write stall (`113.89 ms` → `499.20 us`, `5.00x` earlier pre-warm at step `2` vs. `10`); **`+251.02%` (`3.51x`)** multi-tenant QPS (`6.70M` → `23.53M ops/s`, `1.50x` lower FPR); **`100%`** (`18/18`) guardrail accuracy; **`6.50 us`** post-`SIGKILL` fallback p99. |

---

## 7. Honest Limitations, Remaining Gaps & Production Deployment Guide

### 7.1 Remaining Gaps & Trade-Offs

1. **Single-Threaded Random `dynamic_insert` (`1.09x–1.68x` behind Official C++):** Despite `+10.8%–17.4%` gains up to `3x10^8` keys (`22.78M` vs. `24.74M` at `10^6`; `7.85M` vs. `10.49M` at `3x10^8`), Official C++ `qf_insert_memento` mutates a slightly smaller table (`2.03 GiB` vs. `2.16 GiB` at `5x10^8`) via unchecked raw pointers without unary-age encoding or safe slice bounds checks, retaining a `1.09x–1.34x` edge through `3x10^8` and `1.68x` at TLB-bound `5x10^8` random writes (`6.41M` vs. `10.79M`).
2. **Space Overhead (`+1.07` bits/key vs. Official C++):** Across Section 5.2 (90% LF, `r=7`), Rust uses `14.58 / 17.92 / 23.47 bpk` vs. C++ `13.51 / 16.85 / 22.42 bpk`. This `+1.06–1.08 bpk` delta is strictly the 1-bit InfiniFilter unary age sentinel required for deamortized selective rejuvenation (omitted by C++ `gqf.c`).
3. **WAL Durability Cost (`2.05x` throughput / `2.76x` disk I/O):** Under the default `B = 1,000` group-commit `CoupledWal`, sustained write throughput is `149,059 ops/s` (`25.03 MiB` disk) vs. `305,850 ops/s` (`9.05 MiB` disk) for uncoupled writes. This is the physical cost of crash consistency (`0.00%` vs. `21.66%` `SIGKILL` loss).
4. **1T Multi-Shard Routing Overhead (`-20.88%`):** Running an 8-shard `ShardedAerisSystem` (8 filters + 8 RocksDB instances) on a single thread costs `-20.88%` vs. a single `Mutex`-guarded filter (`1.51M` vs. `1.91M ops/s`), though scaling turns positive immediately at 2 threads (`+76.04%`).

### 7.2 Production Deployment Guide

| Deployment Dimension | Recommended Configuration | Rationale |
| :--- | :--- | :--- |
| **Range-Width Adaptation** | Default Dynamic-$R$ + `FalsePositiveLog`; switch to `ControlledEngineConfig::static_r_only()` only when $R$ is provably fixed. | Protects against `57x` FPR blowup and delivers `5.08x–10.40x` QPS on shifting workloads for `0.28–0.65 ms` FP replay; static bypass recovers `100%` of the Phase-0 estimator tax (`25.13M QPS`). |
| **Capacity Management** | Always enable Proactive EWMA Trigger + Live Contraction. | Cuts max foreground write stalls by `19.45x` (`7.25 ms`) and reclaims `49.98%` RAM after bulk deletes with `0` latency regressions. |
| **Storage Durability** | Keep `StorageConfig::default()` (`WalSyncPolicy::GroupCommit { batch_size: 1_000 }`) with `put_batch()`. | Eliminates `21.66%` `SIGKILL` reverse-map desync (`0` post-rebuild FNs) at `149k ops/s`, avoiding the `123.7x` penalty of per-op `fsync`. |
| **Concurrency & Sharding** | Use multi-shard `ShardedAerisSystem` + RCU for $\ge 2$ threads; configure `initial_shard_count = 1` + `query_*_direct()` for strictly 1T callers. | Delivers `1.76x` (2T) to `9.96x` (8T) throughput and `44x` lower live-expand reader `p99.9`, while avoiding `-20.88%` 8-shard overhead at 1T. |
| **Predictive Control** | Deploy `python-sidecar` (`aeris_sidecar.server`) for seasonal or multi-tenant deployments. | Delivers `228x` lower seasonal write stalls and `3.51x` multi-tenant effective QPS with `100%` deterministic `SafetyGuardrail` enforcement and `6.50 us` native fallback on crash. |
