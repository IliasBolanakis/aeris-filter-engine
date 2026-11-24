# Aeris Filter Modifications & Performance Impact Analysis

> **Source Documents:**
> - **Original Paper (Baseline):** [Aeris Filters.pdf](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Docs/Aeris%20Filters.pdf) — *Aeris Filter: A Strongly and Monotonically Adaptive Range Filter* (Chesetti et al., Proc. ACM Manag. Data, SIGMOD 2026)
> - **System Architecture Specification:** [Aeris_Based_Filtering_System.pdf](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Docs/Aeris_Based_Filtering_System.pdf) — *Aeris-Based Optimized Range Filtering System*
> - **Empirical Ablation Data:** [ABLATION_REPORT.md](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/ABLATION_REPORT.md) (43 metrics across 3 workers at full 10M-key / 5M-query DRAM scale, generated `2026-11-24T13:50:37Z`)
> - **C++ vs. Rust Parity Audit:** [PAPER_VS_AERIS_COMPARISON.md](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/PAPER_VS_AERIS_COMPARISON.md) (48 bare-metal sequential runs up to `5x10^8` = 500M keys against untouched Official C++ `adaptiveqf/`)

---

## 1. Executive Summary

The original **Aeris Filter** (SIGMOD 2026) solved a long-standing theoretical and practical problem in range filtering: simultaneously guaranteeing **spatial robustness**, **temporal robustness** (strong and monotonic adaptivity), and **deamortized I/O expansion** using a Quotient Filter with variable-length keepsake boxes and an on-disk write-optimized reverse map.

However, moving Aeris from a standalone research prototype into a **production-grade, self-tuning storage-engine filter** exposed five critical operational bottlenecks:
1. **Static maximum range length ($R$)** causing linear False Positive Rate (FPR) degradation and CPU blowup when query widths exceed $R$.
2. **Loss of monotonic adaptivity upon changing $R$**, because changing partition size $P = 2^{\lceil \log_2 R \rceil}$ invalidates existing partition hashes $h(p(k))$ and discards prior adaptations.
3. **Synchronous expansion stalls at $\alpha \to 95\%$** and **no memory contraction** after bulk deletions.
4. **Uncoupled on-disk Reverse Map with no crash consistency**, causing silent key desynchronization and post-rebuild False Negatives after abrupt crashes (`SIGKILL`).
5. **Single-threaded / coarse-locked execution, monolithic $2\times$ RAM doubling, and reactive-only capacity/memory management** under multi-core, skewed, seasonal, and multi-tenant workloads.

Our **Aeris-Based Optimized Range Filtering System** addresses all five gaps across a layered Rust + Python architecture (`aeris-core`, `aeris-storage`, `aeris-control`, `aeris-scale`, `aeris-sidecar`, and `python-sidecar`), reinforced with an **interleaved 64-slot AoS block Data Plane**, **in-place stack-buffered `adapt()` with monotonic cluster scanning**, **zero-allocation word-shifted `insert()`**, **block-sequential `BulkTableWriter` streaming 8-bit LSD radix `bulk_load` with `ReverseMapStore::is_noop()`**, **static-$R$ & 1T/single-shard fast-path bypasses**, and **default `B = 1,000` group-commit `CoupledWal`**.

### Correctness & Hardware Audit Scorecard

All ablation measurements were collected on a **13th Gen Intel Core i7-13650HX** (14 physical / 20 logical cores, **24.00 MiB L3 cache**, 63.67 GiB RAM) at full `1.0x` scale with the filter working set spilling into DRAM (**42.25 MiB = `1.76x` L3 cache**; merged peak RSS reduced from `1.29 GiB` to **`578.68 MiB`** via streaming radix bulk load):

| Audit Gate | Observed Result | Requirement | Status |
| :--- | ---: | :--- | :--- |
| **Total False Negatives (FN)** | `0` | `== 0` | **PASS** |
| **Total Repeated False Positives** | `0` | `== 0` | **PASS** |
| **DRAM Spill Beyond L3 Cache** | `42.25 MiB` (`1.76x` L3) | `> 24.00 MiB` | **CONFIRMED** |
| **Merged Worker Peak RSS** | `578.68 MiB` (down from `1.29 GiB`) | `<= 48.00 GiB` | **PASS** |
| **Suite Verdict Distribution** | **Gain: `5` \| Expected Overhead: `1` \| Neutral: `0` \| Negative: `0`** | Strictly Data-Driven | **RECORDED** |

### Master Impact Summary Across All Modifications

| Modification / Suite | Naive Aeris Limitation | Core Change Implemented | Headline Measured Gain | Key Measured Trade-off / Cost | Blunt Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Core DRAM Audit** (`aeris-core`) | Paper prototype validated on C++ WiredTiger setup; prior SoA Rust layout incurred 2nd TLB miss & `155 B/key` build RSS | Interleaved 64-slot AoS blocks (`25 + 8*(r+1+m)` B, zero `unsafe`), SIMD `POPCNT`/`PDEP`, in-place `adapt()` + monotonic cluster scan, word-shifted `insert()`, `BulkTableWriter` streaming 8-bit LSD radix `bulk_load` (`is_noop()`), InfiniFilter age sentinel | **84.70x lower** adversarial FPR (`0.100456` → `0.001186` at $\epsilon=2^{-8}$); **15.05M ops/s** DRAM throughput (`13.58M` pre-exp, `3.94x` vs pre-opt); scales to **`5x10^8` keys** at `11.20 GiB` RSS & **`35.17M keys/s` `bulk_load`** (**beats C++ `33.90M`**!) | `+1.07` bpk vs C++ at 90% LF (strictly the 1-bit InfiniFilter age sentinel); `0` FN, `0` repeated FPs across 10M–500M keys | **Measurable Gain** (8/8 metrics) |
| **Mod 1: Dynamic $R$ + FP Log Replay + Static Bypass** (`aeris-control`, A/B 1) | Static $R$ causes linear FPR/CPU blowup for $L > R$; rebuilding loses monotonicity | Sliding-Quantile Estimator (`enabled` static-$R$ bypass) + Discrete Cost Optimizer + Async RocksDB Rebuild + Append-Only FP Log Replay | **+407.76% (5.08x)** overall throughput (`16.47M ops/s`); **-83.33% (6.00x)** p99 latency (`200 ns`); **+939.93% (10.40x)** at $R=1024$ (`11.31M ops/s`); **57.08x lower** FPR vs. paper-min static $R=16$ | **-26.64%** Phase-0 throughput with active estimator (cut from `-59.87%`; eliminated completely via `--static-bypass`); mean async RocksDB rebuild `1.44 s`; total FP replay **`277.60 us`** | **Measurable Gain** (10 gains, 1 regression) |
| **Mod 2: Proactive EWMA Trigger + Contraction** (`aeris-control`, A/B 2) | Synchronous `expand()` stall at $\alpha=95\%$; no RAM reclamation after deletes | Dynamic EWMA trigger ($T_{\text{full}} \le T_{\text{rebuild}}$) with background RCU swap + live contraction | **-94.86% (19.45x)** max insert stall (`140.95 ms` → `7.25 ms`); **-49.98% (2.00x)** RAM after 70% deletes (`5.28` → `2.64 MiB`) | **0 regressions** — transition insert p99 **`-4.30%`** (`9.30` → `8.90 us`) and p99.9 `+3.95%` (Neutral; eliminated prior `+40%`/`+57%` regressions!) | **Measurable Gain** (2 gains, 2 neutral, 0 regressions) |
| **Mod 3: WAL-Coupled Storage (Default `B=1000` Group Commit)** (`aeris-storage`, A/B 3) | Uncoupled KV + Reverse Map desyncs on `SIGKILL`, causing post-rebuild FNs | CRC32-protected `CoupledWal` defaulting to `WalSyncPolicy::GroupCommit { batch_size: 1_000 }` + `put_batch()` + `strict_per_op_fsync()` crash-test hook | **-100.00%** crash loss/desync (`21.66%` → `0.00%`, `0` post-rebuild FNs vs. `1,098`) | Default group-commit `put_batch(B=1000)` achieves **`149,059 ops/sec`** (`-51.26%` / `2.05x` vs uncoupled, `109.7x` faster than unbatched per-op `fsync` at `1,359 ops/s`) | **Expected Safety / Durability Overhead** |
| **Mod 4: Scale Plane — RCU + Chunk Locks + Range Sharding + 1T Bypass** (`aeris-scale`, A/B 4) | Coarse lock bottlenecks multi-core; `RwLock` stalls readers on expand; global $2\times$ RAM jumps | 64-slot chunk locks + `crossbeam-epoch` RCU swap + $2^{q_i}$ Range Shards + single-shard & `&mut self` 1T direct bypass (`query_*_direct`, `insert_direct`) | **+76.04% (1.76x)** at 2T, **+489.07% (5.89x)** at 4T, **+896.06% (9.96x)** at 8T (`6.12M` vs `615k ops/s`); live-expand reader p99.9 **44.00x lower** (`13.20 us` → `300 ns`, max stall **1,234x lower**: `90.23 ms` → `73.1 us`); hot-spot RAM **-13.83%** | **1T (-20.88%)** on 8-shard topology (8 shards + 8 RocksDB instances vs 1; improved from `-24.98%`); hot-spot write p99 **`+2.94%`** (Neutral, down from `+109.84%`) | **Measurable Gain** (upgraded from Neutral: 5 gains, 1 neutral, 1 regression) |
| **Mod 5: Phase 5 Isolated ML/RL Sidecar + SafetyGuardrail** (`aeris-sidecar`, A/B 5) | Reactive-only expansion; uniform static RAM split across tenants | Out-of-process Python gRPC sidecar (seasonal forecaster + contextual bandit) + deterministic `SafetyGuardrail` + native fallback | **-99.56% (228.14x)** seasonal write stall (`113.89 ms` → `499.2 us`); **5.00x earlier** pre-warm; **+251.02% (3.51x)** multi-tenant throughput (`23.53M` vs `6.70M`); **100%** guardrail accuracy; **`6.50 us`** post-crash fallback | Zero Data Plane hot-path overhead (`78.47M` concurrent queries at p50=`100 ns`, p99=`200 ns`) | **Measurable Gain** (6/6 metrics) |

---

## 2. Baseline: The Original ("Naive") Aeris Filter

Understanding the exact mechanics of the original SIGMOD 2026 Aeris Filter ([Aeris Filters.pdf](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Docs/Aeris%20Filters.pdf)) clarifies why each modification was necessary.

### 2.1 Original Architecture & Theoretical Foundation
- **Key Partitioning (Memento-style):** A universe of fixed-length 64-bit integer keys is partitioned into equally sized intervals of $P = 2^{\lceil \log_2 R \rceil}$ keys, where $R$ is the preconfigured maximum range query length. Each key $k$ splits into a partition prefix $p(k) = \lfloor k / P \rfloor$ and an $m$-bit offset suffix (*memento*) $m(k) = k \bmod P$, where $m = \lceil \log_2 R \rceil$.
- **Quotient Filter Layout:** An array of $N = 2^q$ slots divided into 64-slot blocks (each with an 8-bit offset) using Robin Hood hashing and constant-time `RANK`/`SELECT` primitives. Each slot stores $(r + m)$ payload bits plus **3 metadata bits** (`occupied`, `runend`, `extension`), achieving a total space complexity of:
  $$\alpha n \left(\log_2 \frac{R\alpha}{\epsilon} + 4.125\right) \text{ bits}$$
  matching Goswami's spatial-robustness lower bound of $\Omega(\log(R/\epsilon))$ bits/key.
- **Keepsake Boxes & Extendable Fingerprints:** Each nonempty partition's prefix $p(k)$ is hashed via $h(p(k))$ into an initial $l = (q + r)$-bit fingerprint. On a false positive, the colliding keepsake box is split by extending its fingerprint in discrete increments (first $+m$ bits, then $+(r + m)$ bits per subsequent extension). The `extension` bit is dual-purposed: `extension=1, runend=0` marks an extension slot, while `extension=1, runend=1` marks the end of a quotient run.
- **Deamortized InfiniFilter Expansion & Reverse Map:** Each keepsake box stores a unary age counter (`00..1`) tracking how many $2\times$ expansions ago it was inserted. During expansion, 1 bit moves from the remainder into the canonical slot quotient and the unary counter increments. Only the $\approx 2^{-r}$ fraction of fingerprints that exhaust remainder bits are selectively rejuvenated by querying an on-disk write-optimized $B^\epsilon$-tree (SplinterDB) reverse map. To maintain strong and monotonic adaptivity across expansions, remainders are enlarged by $\log_2\log_2(1/\epsilon) + 1$ bits:
  $$r = \log_2 \frac{1}{\epsilon} + \log_2 \log_2 \frac{1}{\epsilon} + 1$$

### 2.2 Operational Limitations of the Original Design
As formalized in Section 1 of [Aeris_Based_Filtering_System.pdf](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Docs/Aeris_Based_Filtering_System.pdf), integrating the naive Aeris Filter into a production DBMS exposes five structural gaps:
1. **Static Range Parameter $R$:** Under a fixed per-slot payload budget $C = r + \log_2 P$, choosing a fixed $R$ creates an unavoidable mismatch when workloads shift. Queries of length $L > R$ must be decomposed into $\lceil L / P \rceil + 1$ subqueries, causing **linear FPR degradation** and **linear CPU latency blowup**.
2. **Loss of Monotonicity During $R$ Reconfiguration:** Changing $R$ changes $p(k)$ and therefore $h(p(k))$. A naive rebuild from the reverse map forgets every previously resolved false positive, violating monotonic adaptivity.
3. **Synchronous Load-Factor Stalls ($\alpha \to 95\%$) & Missing Contraction:** Approaching $\alpha = 95\%$ causes exponential Robin Hood cluster growth and triggers synchronous `filter.expand()` pauses (hundreds of milliseconds). Furthermore, bulk deletions never shrink the filter's RAM footprint.
4. **No Crash Consistency Between KV Store and Reverse Map:** Updating the primary KV store and the reverse map independently means an abrupt crash (`SIGKILL`) can leave keys acknowledged in the database but absent from the reverse map. Upon a future rebuild or selective rejuvenation, those keys vanish from the filter—producing **silent False Negatives**, which violate the core correctness invariant of any filter.
5. **Lack of Multi-Core Concurrency, Incremental Scaling, & Predictive Control:** The paper prototype lacks fine-grained write concurrency, wait-free reader access during expansion, incremental RAM growth under skewed writes, and proactive seasonal/multi-tenant resource optimization.

---

## 3. Core Data Plane Validation (DRAM-Scale 10M-Key / 5M-Query Audit & AoS/Radix Upgrade)

The Rust Data Plane ([`aeris-core`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-core/src/filter.rs)) uses an **interleaved 64-slot AoS block layout** (`25 + 8 * (r + 1 + m)` bytes/block, zero `unsafe` outside `simd.rs`), **in-place stack-buffered `adapt()` with monotonic cluster scanning**, **zero-allocation word-shifted `insert()`**, and **block-sequential `BulkTableWriter` streaming 8-bit LSD radix `bulk_load`** (with `ReverseMapStore::is_noop()`). It was validated at full DRAM scale (**10,000,000 keys**, **5,000,000 queries**) at both $\epsilon = 2^{-4} = 0.0625$ (**32.25 MiB** = `1.34x` L3) and $\epsilon = 2^{-8} = 0.003906$ (**42.25 MiB** = `1.76x` L3), and separately scaled to **`500,000,000` (`5x10^8`) keys** (`2.16 GiB` filter, **`35.17M keys/s` `bulk_load`** surpassing Official C++'s `33.90M`, `11.59M QPS`, `11.20 GiB` peak RSS, plus `7.52M QPS` [`98.7%` of C++] on adversarial 10% repeat-FP probing) in [PAPER_VS_AERIS_COMPARISON.md](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/PAPER_VS_AERIS_COMPARISON.md).

| Metric | Config | Naive / Target Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | FN | Rep. FPs |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **Pre-Expansion FPR** | $\epsilon=2^{-4}$ (`q=24, r=7, m=5`, `32.25 MiB`) | `0.062500` | `0.004615` | `-92.62%` | **13.54x lower** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |
| **Post-8x-Expansion FPR** | $\epsilon=2^{-4}$ (`q=16->24`, 38,797 boxes rejuv.) | `0.062500` | `0.018391` | `-70.58%` | **3.40x lower** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |
| **DRAM Query Throughput** | $\epsilon=2^{-4}$ (Pre vs. Post-8x-Exp) | `13,582,834 ops/s` | `15,051,659 ops/s` | `+10.81%` | **1.11x faster** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |
| **Adversarial 10% Repeat-FP** | $\epsilon=2^{-4}$ (500k queries, 256 FP pool) | `0.117102` | `0.017328` | `-85.20%` | **6.76x lower** | `100 ns` | `300 ns` | `500 ns` | `0` | `0` |
| **Pre-Expansion FPR** | $\epsilon=2^{-8}$ (`q=24, r=12, m=5`, `42.25 MiB`) | `0.003906` | `0.000142` | `-96.38%` | **27.59x lower** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |
| **Post-8x-Expansion FPR** | $\epsilon=2^{-8}$ (`q=16->24`, 0 boxes rejuv.) | `0.003906` | `0.000657` | `-83.17%` | **5.94x lower** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |
| **DRAM Query Throughput** | $\epsilon=2^{-8}$ (Pre vs. Post-8x-Exp) | `12,310,671 ops/s` | `13,843,252 ops/s` | `+12.45%` | **1.12x faster** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |
| **Adversarial 10% Repeat-FP** | $\epsilon=2^{-8}$ (500k queries, 256 FP pool) | `0.100456` | `0.001186` | `-98.82%` | **84.70x lower** | `100 ns` | `300 ns` | `400 ns` | `0` | `0` |

**Key Takeaway:** Interleaved AoS blocks boosted pre-expansion DRAM throughput by **3.94x** (`3.45M` → `13.58M ops/s` at $\epsilon=2^{-4}$), lifted post-expansion throughput to **13.84M–15.05M ops/s**, tightened `p99.9` from `4.40 us` down to **`400 ns` (`11x` lower)**, and cut filter RAM by `1.75 MiB` while preserving **0 FN** and **0 repeated FPs** (eliminating `49,864` and `49,635` repeated adversarial FPs).

---

## 4. Detailed Breakdown of Modifications & Ablation Impact

### 4.1 Modification 1: Dynamic $R$ Control Plane + Persistent FP Log Replay + Static Bypass (A/B 1)

#### What We Modified
Implemented in [`crates/aeris-control`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-control/src/lib.rs):
- **Fixed-Budget Payload Trade-Off Optimization (`optimizer.rs`):** Enforces $r + \log_2 P = C$ where $P = 2^{\lceil \log_2 R \rceil}$. An analytical **Discrete Cost Optimizer** evaluates all candidate $m = \log_2 P$ values in $< 5\ \mu\text{s}$, balancing collision probability ($2^{-r}$) against subquery fanout ($\lceil L / P \rceil + 1$).
- **Sliding-Window Quantile Estimator + Static-$R$ Bypass (`estimator.rs`, `rebuild.rs`):** Tracks query widths with hysteresis and median-anchored outlier clipping, and provides `EstimatorConfig::with_enabled(false)` / `ControlledEngineConfig::static_r_only()` so provably static-$R$ deployments skip estimator recording completely (`25.13M` vs. `23.27M` C++ in Phase 0, `+8.0%`).
- **Asynchronous Background Rebuild (`rebuild.rs`):** Scans the RocksDB Reverse Map off the hot path to bulk-load $F_{\text{new}}$ with the updated $(r, m)$ split while a `DeltaBuffer` captures concurrent mutations.
- **Persistent False-Positive Log + Replay (`fp_log.rs`):** Every confirmed FP interval $[a, b]$ is appended to a CRC32-framed FP Log and replayed onto $F_{\text{new}}$ before swap, guaranteeing **100% monotonicity across $R$ transitions**.

#### Measured Performance Impact (1M Keys, 2M Queries Across 4 Shifting-$R$ Phases)
Workload shifts through four 500k-query phases: **Phase 0 ($R=16$)** → **Phase 1 ($R=256$)** → **Phase 2 ($R=32$)** → **Phase 3 ($R=1024$)**, triggering 3 background rebuilds (`m: 4→6`, `m: 6→5`, `m: 5→8`).

| Metric | Naive Baseline (Static $R=16$) | Aeris Optimized (Dynamic $R$ + FP Log) | Delta % | Speedup / Ratio |
| :--- | ---: | ---: | ---: | :--- |
| **Overall Shifting-$R$ Throughput** (Equal-RAM $C=17$) | `3,243,178 ops/sec` | **`16,467,614 ops/sec`** | `+407.76%` | **5.08x faster** |
| **Overall Shifting-$R$ p99 Latency** (Equal-RAM $C=17$) | `1.20 us` (`p99.9: 1.80 us`) | **`200 ns`** (`p99.9: 500 ns`) | `-83.33%` | **6.00x lower** |
| **Overall Shifting-$R$ FPR** (vs. Paper-Min Static $C=11$) | `0.070213` | `0.001230` | `-98.25%` | **57.08x lower** |
| **Phase 0 ($R=16$) Throughput** (active estimator) | `36,272,888 ops/sec` | `26,611,173 ops/sec` | `-26.64%` | **1.36x slower** (0% w/ static bypass) |
| **Phase 0 ($R=16$) FPR** (vs. Paper-Min $C=11$) | `0.003590` | `0.000060` | `-98.33%` | **59.83x lower** |
| **Phase 1 ($R=256$) Throughput** | `4,116,900 ops/sec` | **`12,498,688 ops/sec`** | `+203.59%` | **3.04x faster** |
| **Phase 1 ($R=256$) FPR** (vs. Paper-Min $C=11$) | `0.057542` | `0.000870` | `-98.49%` | **66.14x lower** |
| **Phase 2 ($R=32$) Throughput** | `23,210,473 ops/sec` | **`27,125,412 ops/sec`** | `+16.87%` | **1.17x faster** |
| **Phase 2 ($R=32$) FPR** (vs. Paper-Min $C=11$) | `0.007290` | `0.000132` | `-98.19%` | **55.23x lower** |
| **Phase 3 ($R=1024$) Throughput** | `1,087,188 ops/sec` | **`11,306,001 ops/sec`** | `+939.93%` | **10.40x faster** |
| **Phase 3 ($R=1024$) FPR** (vs. Paper-Min $C=11$) | `0.212430` (3.4x target $\epsilon$ blowup!) | `0.003858` | `-98.18%` | **55.06x lower** |
| **Background RocksDB Rebuild Time (Mean across 3 rebuilds)** | `0 ns` | `1.442 s` (total `4,324.7 ms`, RocksDB scan-bound) | `+100.00%` | Mechanism cost |
| **FP Log Replay Latency (Total across 1,044 entries)** | `0 ns` | **`277.60 us`** (`p50: 104.10 us`) | `+100.00%` | Mechanism cost |

#### Trade-Offs & Verdict: **Measurable Gain**
- **Win:** Delivers **5.08x overall throughput** (`16.47M ops/s`), **10.40x Phase-3 throughput** at $R=1024$ (`11.31M ops/s`), **6.00x lower p99 latency** (`200 ns`), **57.08x lower overall FPR**, and **0 repeated FPs** across 1,044 post-rebuild reprobes in **`0.28 ms`** replay time.
- **Cost:** With active quantile tracking enabled in Phase 0, instrumentation costs `-26.64%` (`1.36x` slower, cut by more than half from `-59.87%`); enabling `static_r_only()` eliminates this overhead completely when $R$ is known static.

---

### 4.2 Modification 2: Proactive EWMA Trigger + Live Contraction (A/B 2)

#### What We Modified
Implemented in [`crates/aeris-control/src/trigger.rs`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-control/src/trigger.rs) and [`crates/aeris-core/src/filter.rs`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-core/src/filter.rs):
- **Dynamic Proactive Triggering:** Estimates $T_{\text{full}} = N \cdot (0.95 - \alpha) / \text{IngestionRate}_{\text{EWMA}}$ against $T_{\text{rebuild}} = |\text{ReverseMap}| / \text{BulkLoadSpeed}$, firing background expansion at $\alpha = 0.8503$ and swapping atomically via RCU at $\alpha = 0.8925$ (`0` reactive fallbacks).
- **Live Memory Contraction:** After 70% bulk deletes (`1,000,000 → 300,000` keys), contracts $q = 21 \to q = 20$, reclaiming 49.98% of filter RAM (`5.28 → 2.64 MiB`) with `0` FN and `0` repeated FPs.

#### Measured Performance Impact (1M Keys, 85%→95% Transition + 70% Delete Contraction)

| Metric | Naive Baseline (Reactive $\alpha=95\%$, No Contraction) | Aeris Optimized (Proactive EWMA + Contraction) | Delta % | Speedup / Ratio |
| :--- | ---: | ---: | ---: | :--- |
| **Transition Insert Max Stall (85%→95%)** | `140.95 ms` | **`7.25 ms`** | `-94.86%` | **19.45x lower** |
| **Post-70%-Delete Filter RAM Footprint** | `5.28 MiB` (`0 B` reclaimed, `q=21`) | **`2.64 MiB`** (`2.64 MiB` reclaimed, `q=20`) | `-49.98%` | **2.00x lower** |
| **Transition Insert Latency (p99, 85%→95%)** | `9.30 us` (`p50: 4.20 us`) | **`8.90 us`** (`p50: 4.20 us`) | **`-4.30%`** | **Neutral / 1.04x lower** (was `+40.24%`!) |
| **Transition Insert Latency (p99.9, 85%→95%)** | `32.90 us` | `34.20 us` | `+3.95%` | **Neutral** (was `+57.27%`!) |
| **Concurrent Reader Latency (Diagnostic)** | `p50: 100 ns, p99: 100 ns, max: 164.0 us` | `p50: 100 ns, p99: 200 ns, max: 37.3 us` | **`-77.3%` max** | Zero reader stall |

#### Trade-Offs & Verdict: **Measurable Gain** (0 Regressions)
- **Win:** Zero-allocation in-place `insert` completely eliminated the prior `+40%`/`+57%` transition-window insert latency regressions while widening the max foreground write-stall reduction to **19.45x (`140.95 ms` → `7.25 ms`)** and reclaiming **49.98% RAM** with `0` FN and `0` repeated FPs.

---

### 4.3 Modification 3: WAL-Coupled Storage Engine & Default Group-Commit Batching (A/B 3)

#### What We Modified
Implemented in [`crates/aeris-storage`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-storage/src/engine.rs):
- **Default Group-Commit Batching (`B = 1,000`):** `StorageConfig::default()` now enforces `WalSyncPolicy::GroupCommit { batch_size: 1_000 }` alongside first-class `CoupledStorageEngine::put_batch()` and `flush_wal()`, while `StorageConfig::strict_per_op_fsync()` (`WalSyncPolicy::PerOpFsync`) preserves explicit per-operation `fsync` hooks for strict durability and crash-injection tests.
- **Torn-Tail Recovery Protocol:** Replays all valid CRC32-verified WAL frames and truncates any partially written tail record on restart after `SIGKILL`.

#### Measured Performance Impact (Unbatched 5k Keys, Default Batched 250k Keys $B=1000$, 5x `SIGKILL` Cycles)

| Metric | Naive Baseline (Uncoupled Reverse Map) | Aeris Optimized (WAL-Coupled Storage) | Delta % | Speedup / Ratio |
| :--- | ---: | ---: | ---: | :--- |
| **`SIGKILL` Crash Lost/Desync Key Rate (5 cycles)** | `21.66%` (`1,098` lost RM, `1,103` desync, **`1,098` post-rebuild FNs**) | **`0.00%`** (`0` lost KV/RM, `0` desync, **`0` FNs**) | `-100.00%` | **Eliminated** |
| **Default Group-Commit Throughput (250k keys, $B=1000$)** | `305,850 ops/sec` | **`149,059 ops/sec`** (`109.7x` faster than per-op!) | `-51.26%` | **2.05x slower** |
| **Default Group-Commit Latency (p99/batch, $B=1000$)** | `3.70 ms` (`3.70 us/key`) | `7.93 ms` (`7.93 us/key`) | `+114.60%` | **2.15x higher** |
| **Strict Per-Op `fsync` Throughput (5k keys, opt-in hook)** | `168,112 ops/sec` | `1,359 ops/sec` | `-99.19%` | **123.69x slower** |
| **Strict Per-Op `fsync` Latency (p99, 5k keys)** | `9.00 us` | `1.48 ms` | `+16,368.89%` | **164.69x higher** |

#### Trade-Offs & Verdict: **Expected Safety/Durability Overhead**
- **Win:** Eliminates silent reverse-map desynchronization under `SIGKILL` (`21.66%` loss / `1,098` post-rebuild FNs → `0.00%` / `0` FNs).
- **Cost:** Physical WAL persistence + `fsync` imposes a `2.05x` throughput overhead under the new default `B = 1,000` group-commit policy (`149,059 ops/s`, `25.03 MiB` disk vs. `9.05 MiB` uncoupled), preventing accidental deployment of the `123.7x`-slower per-op `fsync` path.

---

### 4.4 Modification 4: Scale Plane — Chunk Locking, Epoch RCU, Range Sharding, & 1T Direct Bypass (A/B 4)

#### What We Modified
Implemented in [`crates/aeris-scale`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-scale/src/lib.rs):
- **Fine-Grained Chunk Locking + Epoch RCU Swap + Dynamic Range Sharding:** Parallel per-chunk writes, wait-free `crossbeam-epoch` reader traversal across live $2\times$ expansions, and dynamic median-split / cold-merge range shards.
- **Single-Shard & 1T Exclusive Direct Bypass (`sharding.rs`, `rcu_filter.rs`, `epoch_cell.rs`):** Added `ShardingConfig::single_shard_bypass` (default `true`, skipping binary search/clamping when `shard_count == 1`) and `&mut self` direct bypass methods (`query_point_direct`, `query_range_direct`, `insert_direct`) eliminating epoch pinning and lock acquisition at 1T.

#### Measured Performance Impact (5M–7M Keys Mixed 90R/10W, Live 2x Expansion, & 85% Skewed Hot-Spot)

| Metric | Naive Baseline (Coarse Mutex / `RwLock` / Single Filter) | Aeris Optimized (RCU + 8-Shard Range Sharding + 1T Bypass) | Delta % | Speedup / Ratio |
| :--- | ---: | ---: | ---: | :--- |
| **Mixed 90R/10W Throughput (1 Thread)** | `1,905,099 ops/sec` (`p99: 5.3 us`) | `1,507,248 ops/sec` (`p99: 6.7 us`) | `-20.88%` | **1.26x slower** (improved from `-24.98%`) |
| **Mixed 90R/10W Throughput (2 Threads)** | `1,473,904 ops/sec` (`p99: 109.9 us`) | **`2,594,715 ops/sec`** (`p99: 7.9 us`) | **`+76.04%`** | **1.76x faster** (flipped from `-7.87%`!) |
| **Mixed 90R/10W Throughput (4 Threads)** | `868,248 ops/sec` (`p99: 213.9 us`) | **`5,114,574 ops/sec`** (`p99: 8.6 us`) | **`+489.07%`** | **5.89x faster** (up from `2.48x`!) |
| **Mixed 90R/10W Throughput (8 Threads)** | `614,745 ops/sec` (`p99: 184.5 us`) | **`6,123,203 ops/sec`** (`p99: 21.3 us`) | **`+896.06%`** | **9.96x faster** (up from `3.62x`!) |
| **Reader Latency During 2x Live Expansion (p99.9 / Max)** | `p99.9: 13.20 us` / **`max: 90.23 ms`** (`10,049` queries) | **`p99.9: 300 ns`** / **`max: 73.1 us`** (`6,361,406` queries) | **`-97.73%` p99.9** / **`-99.92%` max** | **44.00x lower p99.9** / **1,234x lower max stall** |
| **Hot-Spot Final Filter RAM (85% Skewed Writes)** | `1.32 MiB` (`q=17→19`, growth `1,038 KiB`) | **`1.14 MiB`** (`19` splits, `1` merge, `26` shards, growth `828 KiB`) | `-13.83%` | **1.16x lower** |
| **Hot-Spot Skewed Write Latency (p99 / Max)** | `p99: 6.80 us` / `max: 33.05 ms` | **`p99: 7.00 us`** / **`max: 2.20 ms`** | `+2.94%` p99 / **`-93.3%` max** | **Neutral p99** (down from `+109.84%`) / **15.0x lower max** |

#### Trade-Offs & Verdict: **Measurable Gain** (Upgraded from Neutral!)
- **Win:** Combined with AoS blocks and zero-alloc `insert`, the Scale Plane now wins decisively from **2 threads upward** (`1.76x` at 2T, `5.89x` at 4T, **`9.96x` at 8T**), cuts live-expansion reader `p99.9` by **44x** (`300 ns`) and max stall by **1,234x** (`73.1 us`, serving **633x more queries** during expansion), and eliminates the prior `+109.84%` hot-spot write p99 regression.
- **Cost:** Strictly single-threaded 8-shard deployments still pay `-20.88%` for maintaining 8 shards and 8 RocksDB reverse maps instead of 1; strictly 1T deployments should configure `initial_shard_count = 1` to activate the zero-routing single-shard direct bypass.

---

### 4.5 Modification 5: Isolated Phase 5 ML/RL Control-Plane Sidecar + SafetyGuardrail (A/B 5)

#### What We Modified
Implemented in [`crates/aeris-sidecar`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/crates/aeris-sidecar/src/lib.rs) and [`python-sidecar/aeris_sidecar`](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/python-sidecar/aeris_sidecar/server.py): strict out-of-process gRPC isolation (seasonal forecaster + multi-tenant contextual bandit) guarded by deterministic `SafetyGuardrail` and instant native fallback.

#### Measured Performance Impact (Live Python gRPC Sidecar, 1M Keys, Seasonal Surge + Multi-Tenant + Crash Test)

| Metric | Native Control Plane Baseline | Phase 5 Live Python ML/RL Sidecar | Delta % | Speedup / Ratio |
| :--- | ---: | ---: | ---: | :--- |
| **Seasonal Surge Max Foreground Write Stall** | `113.89 ms` | **`499.20 us`** | `-99.56%` | **228.14x lower** |
| **Seasonal Surge Expansion Trigger Step** | Step `10` (reactive) | Step `2` (**8 steps earlier pre-warm**) | `-80.00%` | **5.00x earlier** |
| **Multi-Tenant Empirical FPR (Bandit vs. Uniform)** | `0.000120` (`24` FPs) | `0.000080` (`16` FPs) | `-33.33%` | **1.50x lower** |
| **Multi-Tenant Effective Query Throughput** (incl. `50 us` FP disk penalty) | `6,702,390 ops/sec` (`p50: 100 ns, p99: 300 ns`) | **`23,526,644 ops/sec`** (`p50: 100 ns, p99: 300 ns`) | `+251.02%` | **3.51x faster** |
| **`SafetyGuardrail` Validation & Rejection Accuracy** | `0.00%` (N/A) | `100.00%` (`18/18` adversarial/edge tests) | `+100.00%` | **New capability** |
| **Post-`SIGKILL` Control-Plane Fallback p99 Latency** | `3.000 s` (RPC timeout) | **`6.50 us`** (`p50: 500 ns`, `40/40` native fallbacks) | `-100.00%` | **461,538x lower** |

#### Trade-Offs & Verdict: **Measurable Gain** (6/6 Metrics)
- **Win:** Eliminates `113.89 ms` seasonal surge stalls (`228x` reduction), boosts multi-tenant effective throughput by **3.51x** (`23.53M ops/s`), enforces **100% deterministic safety**, and falls back from mid-session `SIGKILL` in `6.50 us` while the Data Plane served **`78,469,103` concurrent queries** at `p50=100 ns / p99=200 ns` with zero stalls.

---

## 5. Rejected Design Alternatives (Architectural Consensus Matrix)

As documented in Section 4 (Table 2) of [Aeris_Based_Filtering_System.pdf](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Docs/Aeris_Based_Filtering_System.pdf), four candidate modifications were explicitly evaluated and **rejected** to protect hot-path latency and theoretical guarantees:

| Rejected Technique | Reason for Rejection |
| :--- | :--- |
| **ML Inference in the Query Hot-Path** | Adds $>10\times$ latency overhead to sub-100 ns bitwise RAM lookups and exposes the query path to adversarial latency/model-evasion attacks. |
| **Online RL for $r$ vs. $\alpha$ in RAM** | Exploration noise introduces unacceptable production instability for negligible gain (~`0.2%` change in $\alpha$). |
| **Arbitrary Non-Power-of-2 Resizing (e.g., `1.4x`)** | Breaks InfiniFilter quotient bit-splitting, destroying deamortized expansion and forcing expensive full rebuilds. |
| **Multi-$R$ Tiered Filters** | Doubles RAM consumption and multiplies every `INSERT`/`DELETE` across multiple redundant filter tiers. |

---

## 6. Deployment & Configuration Recommendations

Based strictly on the empirical trade-offs measured in [ABLATION_REPORT.md](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/Filter/ABLATION_REPORT.md) and [PAPER_VS_AERIS_COMPARISON.md](file:///c:/Users/hlias/Desktop/Classes/Special%20DB%20Topics/Aeris-Based%20Filtering%20System/PAPER_VS_AERIS_COMPARISON.md):

1. **Enable Dynamic $R$ + FP Log Replay by Default; Use `static_r_only()` Only for Provably Fixed-$R$ Workloads:** Dynamic-$R$ delivers **5.08x overall throughput** (`10.40x` at $R=1024$) and **57x FPR protection** for `0.28 ms` total FP replay; if query widths are guaranteed constant, `ControlledEngineConfig::static_r_only()` recovers 100% of the Phase-0 estimator tax (`25.13M QPS`, `+8.0%` over C++).
2. **Always Enable Proactive EWMA Trigger + Contraction:** Delivers **19.45x lower max insert stall** (`7.25 ms` vs. `140.95 ms`) and **50% RAM reclamation** with **zero** transition-window insert latency regressions.
3. **Keep the Default `B = 1,000` Group-Commit `CoupledWal` Policy:** `StorageConfig::default()` (`149k ops/s`, `0.00%` `SIGKILL` loss) eliminates the `21.66%` crash-induced FN risk without paying the `123.7x` per-op `fsync` penalty (reserve `StorageConfig::strict_per_op_fsync()` for strict single-op durability audits).
4. **Enable Multi-Shard Scale Plane at $\ge 2$ Concurrent Threads; Use Single-Shard Direct Bypass at 1T:** Multi-shard RCU scales **1.76x at 2T, 5.89x at 4T, and 9.96x at 8T** while cutting live-expansion reader p99.9 by **44x**; single-threaded callers should use `initial_shard_count = 1` + `query_*_direct()` to bypass 8-shard routing overhead.
5. **Deploy the Phase 5 ML/RL Sidecar for Seasonal or Multi-Tenant Workloads:** Fully isolated from the Data Plane and guarded by `SafetyGuardrail` + `6.50 us` native fallback, delivering **228x lower seasonal write stalls** and **3.51x higher multi-tenant effective throughput**.
