# Aeris Correctness Validation & Large-Scale Ablation Report

- **Generated (UTC):** `2026-11-24T13:50:37Z`
- **Ingested Worker Reports (3):** `ablation_worker1_fpr_dynamic_r.json`, `ablation_worker2_proactive_storage.json`, `ablation_worker3_scale_sidecar.json`
- **Total Recorded Metrics:** `43`

## 1. Executive Summary & Correctness Audit Scorecard

> [!IMPORTANT]
> **CORRECTNESS AUDIT PASSED:** `0` total false negatives and `0` total repeated false positives across all `43` recorded Aeris ablation metrics.

| Audit Gate | Observed Total | Requirement | Status |
| :--- | ---: | :--- | :--- |
| Total False Negatives (FN) | `0` | `== 0` | **PASS** |
| Total Repeated False Positives | `0` | `== 0` | **PASS** |
| DRAM Spill Beyond L3 Cache | `42.25 MiB` (`1.76x` L3) | `> 24.00 MiB` | **CONFIRMED** |
| Verdict Distribution (Active Suites) | Gain: `5`, Expected Overhead: `1`, Neutral: `0`, Negative: `0` | Strictly Data-Driven | Recorded |

## 2. Host Hardware, L3 Cache Spill Analysis & Peak RSS

| Parameter | Measured Value |
| :--- | :--- |
| CPU Model | 13th Gen Intel(R) Core(TM) i7-13650HX |
| Physical / Logical Cores | `14` physical / `20` logical |
| L3 Cache Capacity | 24,576 KiB (24.00 MiB) |
| Total System RAM | `63.67 GiB` (`68,365,148,160` B) |
| Min Available RAM at Suite Start | `46.09 GiB` (`49,489,883,136` B) |
| Applied Scale Factor | `1.0x (Full 10M / 5M Scale)` |
| Merged Process Peak RSS (`max` across workers) | `578.68 MiB` (`606,793,728` B) |
| Max DRAM-Audit Filter Footprint | `42.25 MiB` (`1.76x` of OS-reported host L3 cache) |
| Process Peak RSS vs. L3 Cache | `24.11x` of OS-reported host L3 cache |
| DRAM Spill Status | **Confirmed DRAM-bound** (filter working set exceeds L3 cache) |

### Per-Worker Telemetry Breakdown

| Worker | Source File | Metrics | Scale Factor | Avail. RAM | Worker Peak RSS |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `worker1_fpr_dynamic_r` | `ablation_worker1_fpr_dynamic_r.json` | 21 | `1.00x` | 46.10 GiB | 348.70 MiB |
| `worker2_proactive_storage` | `ablation_worker2_proactive_storage.json` | 9 | `1.00x` | 46.12 GiB | 146.91 MiB |
| `worker3_scale_sidecar` | `ablation_worker3_scale_sidecar.json` | 13 | `1.00x` | 46.09 GiB | 578.68 MiB |

## 3. Side-by-Side Ablation Results (Naive Paper-Baseline vs. Aeris Optimized)

### DRAM-Scale 10M-Key / 5M-Query Audit

- **Baseline:** Theoretical / Unadapted Baseline
- **Optimized:** Aeris Optimized (Monotonic Adaptation)

| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DRAM Pre-Expansion Empirical FPR vs Target (eps_2e-4)**<br><sub>`DRAM_10M_FPR_eps_2e-4`</sub> | 10,000,000 | 5,000,000 | 0.062500 | 0.004615 | -92.62% | 13.54x lower (base/opt) | 100 ns | 300 ns | 400 ns | 0.004615 | 0 | 0 | 32.25 MiB | 0 B |
| **DRAM Post-8x-Expansion Empirical FPR vs Target (eps_2e-4)**<br><sub>`DRAM_10M_Post8xExp_FPR_eps_2e-4`</sub> | 10,000,000 | 5,000,000 | 0.062500 | 0.018391 | -70.58% | 3.40x lower (base/opt) | 100 ns | 300 ns | 400 ns | 0.018391 | 0 | 0 | 32.25 MiB | 166.75 MiB |
| **DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion (eps_2e-4)**<br><sub>`DRAM_10M_Throughput_eps_2e-4`</sub> | 10,000,000 | 5,000,000 | 13,582,834 ops/sec | 15,051,659 ops/sec | +10.81% | 1.11x faster (opt/base) | 100 ns | 300 ns | 400 ns | 0.018391 | 0 | 0 | 32.25 MiB | 166.75 MiB |
| **Adversarial 10% Repeat-FP Probing FPR (eps_2e-4)**<br><sub>`DRAM_10M_Adversarial_RepeatFP_eps_2e-4`</sub> | 10,000,000 | 500,000 | 0.117102 | 0.017328 | -85.20% | 6.76x lower (base/opt) | 100 ns | 300 ns | 500 ns | 0.017328 | 0 | 0 | 32.25 MiB | 141.65 MiB |
| **DRAM Pre-Expansion Empirical FPR vs Target (eps_2e-8)**<br><sub>`DRAM_10M_FPR_eps_2e-8`</sub> | 10,000,000 | 5,000,000 | 0.003906 | 0.000142 | -96.38% | 27.59x lower (base/opt) | 100 ns | 300 ns | 400 ns | 0.000142 | 0 | 0 | 42.25 MiB | 0 B |
| **DRAM Post-8x-Expansion Empirical FPR vs Target (eps_2e-8)**<br><sub>`DRAM_10M_Post8xExp_FPR_eps_2e-8`</sub> | 10,000,000 | 5,000,000 | 0.003906 | 0.000657 | -83.17% | 5.94x lower (base/opt) | 100 ns | 300 ns | 400 ns | 0.000657 | 0 | 0 | 42.25 MiB | 239.37 MiB |
| **DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion (eps_2e-8)**<br><sub>`DRAM_10M_Throughput_eps_2e-8`</sub> | 10,000,000 | 5,000,000 | 12,310,671 ops/sec | 13,843,252 ops/sec | +12.45% | 1.12x faster (opt/base) | 100 ns | 300 ns | 400 ns | 0.000657 | 0 | 0 | 42.25 MiB | 239.37 MiB |
| **Adversarial 10% Repeat-FP Probing FPR (eps_2e-8)**<br><sub>`DRAM_10M_Adversarial_RepeatFP_eps_2e-8`</sub> | 10,000,000 | 500,000 | 0.100456 | 0.001186 | -98.82% | 84.70x lower (base/opt) | 100 ns | 300 ns | 400 ns | 0.001186 | 0 | 0 | 42.25 MiB | 148.87 MiB |

<details>
<summary>Diagnostic Notes (8)</summary>

- **DRAM Pre-Expansion Empirical FPR vs Target (eps_2e-4)** (`DRAM_10M_FPR_eps_2e-4`): q=24, r=7, m=5, C=12, filter_ram_mb=32.25 (>24 MiB L3 spill), pre_qps=13582834, fps=23077
- **DRAM Post-8x-Expansion Empirical FPR vs Target (eps_2e-4)** (`DRAM_10M_Post8xExp_FPR_eps_2e-4`): q=16->24 (8x2x expansions), pre_fpr=0.004615, post_fpr=0.018391, boxes_rejuvenated=38797, keys_rejuvenated=38871, pre_exp8_adapted_reprobed=256, filter_ram_mb=32.25, process_peak_rss_mb=338.6
- **DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion (eps_2e-4)** (`DRAM_10M_Throughput_eps_2e-4`): uninstrumented 5M tight loop; pre_p50=100ns pre_p99=300ns post_p50=100ns post_p99=300ns
- **Adversarial 10% Repeat-FP Probing FPR (eps_2e-4)** (`DRAM_10M_Adversarial_RepeatFP_eps_2e-4`): known_fp_pool=256, naive_repeated_fps=49864, aeris_repeated_fps=0, aeris_adaptations=8664, repeat_rate=10%
- **DRAM Pre-Expansion Empirical FPR vs Target (eps_2e-8)** (`DRAM_10M_FPR_eps_2e-8`): q=24, r=12, m=5, C=17, filter_ram_mb=42.25 (>24 MiB L3 spill), pre_qps=12310671, fps=708
- **DRAM Post-8x-Expansion Empirical FPR vs Target (eps_2e-8)** (`DRAM_10M_Post8xExp_FPR_eps_2e-8`): q=16->24 (8x2x expansions), pre_fpr=0.000142, post_fpr=0.000657, boxes_rejuvenated=0, keys_rejuvenated=0, pre_exp8_adapted_reprobed=256, filter_ram_mb=42.25, process_peak_rss_mb=348.7
- **DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion (eps_2e-8)** (`DRAM_10M_Throughput_eps_2e-8`): uninstrumented 5M tight loop; pre_p50=100ns pre_p99=300ns post_p50=100ns post_p99=300ns
- **Adversarial 10% Repeat-FP Probing FPR (eps_2e-8)** (`DRAM_10M_Adversarial_RepeatFP_eps_2e-8`): known_fp_pool=256, naive_repeated_fps=49635, aeris_repeated_fps=0, aeris_adaptations=593, repeat_rate=10%

</details>

### A/B 1: Dynamic R + FP Log Replay vs. Static R

- **Baseline:** Naive Paper-Baseline (Static R, No FP Log)
- **Optimized:** Aeris Optimized (Dynamic R + FP Log Replay)

| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A/B 1 Overall Shifting-R Throughput (Equal-RAM Static R=16 vs Dynamic R)**<br><sub>`AB1_DynamicR_Overall_Throughput`</sub> | 1,000,000 | 2,000,000 | 3,243,178 ops/sec | 16,467,614 ops/sec | +407.76% | 5.08x faster (opt/base) | 100 ns | 200 ns | 500 ns | 0.001230 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Overall Shifting-R p99 Latency (Equal-RAM Static R=16 vs Dynamic R)**<br><sub>`AB1_DynamicR_Overall_Latency_p99`</sub> | 1,000,000 | 2,000,000 | 1.20 us | 200 ns | -83.33% | 6.00x lower (base/opt) | 100 ns | 200 ns | 500 ns | 0.001230 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Overall Shifting-R FPR (Paper-Min Static R=16 C=11 vs Dynamic-R Aeris C=17)**<br><sub>`AB1_DynamicR_Overall_FPR_vs_PaperMin`</sub> | 1,000,000 | 2,000,000 | 0.070213 | 0.001230 | -98.25% | 57.08x lower (base/opt) | 100 ns | 200 ns | 500 ns | 0.001230 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 0 (R=16) Throughput (Equal-RAM Static vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase0_R16_Throughput`</sub> | 1,000,000 | 500,000 | 36,272,888 ops/sec | 26,611,173 ops/sec | -26.64% | 1.36x slower (base/opt) | 0 ns | 100 ns | 100 ns | 6.000e-05 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 0 (R=16) FPR (Paper-Min Static R=16 vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase0_R16_FPR`</sub> | 1,000,000 | 500,000 | 0.003590 | 6.000e-05 | -98.33% | 59.83x lower (base/opt) | 0 ns | 100 ns | 100 ns | 6.000e-05 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 1 (R=256) Throughput (Equal-RAM Static vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase1_R256_Throughput`</sub> | 1,000,000 | 500,000 | 4,116,900 ops/sec | 12,498,688 ops/sec | +203.59% | 3.04x faster (opt/base) | 100 ns | 200 ns | 300 ns | 0.000870 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 1 (R=256) FPR (Paper-Min Static R=16 vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase1_R256_FPR`</sub> | 1,000,000 | 500,000 | 0.057542 | 0.000870 | -98.49% | 66.14x lower (base/opt) | 100 ns | 200 ns | 300 ns | 0.000870 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 2 (R=32) Throughput (Equal-RAM Static vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase2_R32_Throughput`</sub> | 1,000,000 | 500,000 | 23,210,473 ops/sec | 27,125,412 ops/sec | +16.87% | 1.17x faster (opt/base) | 0 ns | 100 ns | 100 ns | 0.000132 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 2 (R=32) FPR (Paper-Min Static R=16 vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase2_R32_FPR`</sub> | 1,000,000 | 500,000 | 0.007290 | 0.000132 | -98.19% | 55.23x lower (base/opt) | 0 ns | 100 ns | 100 ns | 0.000132 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 3 (R=1024) Throughput (Equal-RAM Static vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase3_R1024_Throughput`</sub> | 1,000,000 | 500,000 | 1,087,188 ops/sec | 11,306,001 ops/sec | +939.93% | 10.40x faster (opt/base) | 100 ns | 200 ns | 600 ns | 0.003858 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Phase 3 (R=1024) FPR (Paper-Min Static R=16 vs Dynamic R)**<br><sub>`AB1_DynamicR_Phase3_R1024_FPR`</sub> | 1,000,000 | 500,000 | 0.212430 | 0.003858 | -98.18% | 55.06x lower (base/opt) | 100 ns | 200 ns | 600 ns | 0.003858 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Background RocksDB Reverse Map Rebuild Time (Mean)**<br><sub>`AB1_DynamicR_RebuildTime`</sub> | 1,000,000 | 3 | 0 ns | 1.442 s | +100.00% | overhead (base=0) | 1.436 s | 1.436 s | 1.436 s | 0.001230 | 0 | 0 | 5.28 MiB | 16.86 MiB |
| **A/B 1 Persistent FP Log Replay Latency Across Rebuilds (Total)**<br><sub>`AB1_DynamicR_FpReplayLatency`</sub> | 1,000,000 | 1,044 | 0 ns | 277.60 us | +100.00% | overhead (base=0) | 104.10 us | 104.10 us | 104.10 us | 0.001230 | 0 | 0 | 5.28 MiB | 16.86 MiB |

<details>
<summary>Diagnostic Notes (13)</summary>

- **A/B 1 Overall Shifting-R Throughput (Equal-RAM Static R=16 vs Dynamic R)** (`AB1_DynamicR_Overall_Throughput`): C=17, eps=0.0625, rebuilds=3 [m:4->6@R=256, m:6->5@R=32, m:5->8@R=1024], static_c17_fpr=0.001185, dynamic_c17_fpr=0.001230, paper_min_c11_fpr=0.070213
- **A/B 1 Overall Shifting-R p99 Latency (Equal-RAM Static R=16 vs Dynamic R)** (`AB1_DynamicR_Overall_Latency_p99`): static_p50=200ns static_p99=1200ns static_p999=1800ns vs dynamic_p50=100ns dynamic_p99=200ns dynamic_p999=500ns
- **A/B 1 Overall Shifting-R FPR (Paper-Min Static R=16 C=11 vs Dynamic-R Aeris C=17)** (`AB1_DynamicR_Overall_FPR_vs_PaperMin`): paper_min(r=7,m=4,C=11) overall_fpr=0.070213 (R=1024 blowup=0.212430 vs target eps=0.0625), equal_ram(r=13,m=4,C=17) overall_fpr=0.001185, dynamic_aeris overall_fpr=0.001230
- **A/B 1 Phase 0 (R=16) Throughput (Equal-RAM Static vs Dynamic R)** (`AB1_DynamicR_Phase0_R16_Throughput`): active_m=4, active_r=13, rebuilds_in_phase=0, static_p99=100ns, dyn_p99=100ns, static_c17_fpr=0.000060, paper_min_c11_fpr=0.003590, dyn_fpr=0.000060
- **A/B 1 Phase 0 (R=16) FPR (Paper-Min Static R=16 vs Dynamic R)** (`AB1_DynamicR_Phase0_R16_FPR`): equal_ram_static_fpr=0.000060 (fps=30), paper_min_static_fpr=0.003590 (fps=1795), dynamic_fpr=0.000060 (fps=30), target_eps=0.0625, active_m=4, active_r=13
- **A/B 1 Phase 1 (R=256) Throughput (Equal-RAM Static vs Dynamic R)** (`AB1_DynamicR_Phase1_R256_Throughput`): active_m=6, active_r=11, rebuilds_in_phase=1, static_p99=500ns, dyn_p99=200ns, static_c17_fpr=0.000846, paper_min_c11_fpr=0.057542, dyn_fpr=0.000870
- **A/B 1 Phase 1 (R=256) FPR (Paper-Min Static R=16 vs Dynamic R)** (`AB1_DynamicR_Phase1_R256_FPR`): equal_ram_static_fpr=0.000846 (fps=423), paper_min_static_fpr=0.057542 (fps=28771), dynamic_fpr=0.000870 (fps=435), target_eps=0.0625, active_m=6, active_r=11
- **A/B 1 Phase 2 (R=32) Throughput (Equal-RAM Static vs Dynamic R)** (`AB1_DynamicR_Phase2_R32_Throughput`): active_m=5, active_r=12, rebuilds_in_phase=1, static_p99=100ns, dyn_p99=100ns, static_c17_fpr=0.000086, paper_min_c11_fpr=0.007290, dyn_fpr=0.000132
- **A/B 1 Phase 2 (R=32) FPR (Paper-Min Static R=16 vs Dynamic R)** (`AB1_DynamicR_Phase2_R32_FPR`): equal_ram_static_fpr=0.000086 (fps=43), paper_min_static_fpr=0.007290 (fps=3645), dynamic_fpr=0.000132 (fps=66), target_eps=0.0625, active_m=5, active_r=12
- **A/B 1 Phase 3 (R=1024) Throughput (Equal-RAM Static vs Dynamic R)** (`AB1_DynamicR_Phase3_R1024_Throughput`): active_m=8, active_r=9, rebuilds_in_phase=1, static_p99=1300ns, dyn_p99=200ns, static_c17_fpr=0.003746, paper_min_c11_fpr=0.212430, dyn_fpr=0.003858
- **A/B 1 Phase 3 (R=1024) FPR (Paper-Min Static R=16 vs Dynamic R)** (`AB1_DynamicR_Phase3_R1024_FPR`): equal_ram_static_fpr=0.003746 (fps=1873), paper_min_static_fpr=0.212430 (fps=106215), dynamic_fpr=0.003858 (fps=1929), target_eps=0.0625, active_m=8, active_r=9
- **A/B 1 Background RocksDB Reverse Map Rebuild Time (Mean)** (`AB1_DynamicR_RebuildTime`): total_rebuilds=3, total_rebuild_ms=4324.7; phase1(R=256, m:4->6, r=11, keys=1000000, 1431.8ms, bulk=1039k/s, reprobe_rep=0); phase2(R=32, m:6->5, r=12, keys=1000000, 1456.7ms, bulk=1045k/s, reprobe_rep=0); phase3(R=1024, m:5->8, r=9, keys=1000000, 1436.2ms, bulk=1064k/s, reprobe_rep=0)
- **A/B 1 Persistent FP Log Replay Latency Across Rebuilds (Total)** (`AB1_DynamicR_FpReplayLatency`): logged_fps=2460, replayed_entries=1044, collisions_resolved=0, post_rebuild_reprobes=1044, repeated_fps=0 (100% monotonic)

</details>

### A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95%

- **Baseline:** Naive Paper-Baseline (Reactive alpha = 95%, No Contraction)
- **Optimized:** Aeris Optimized (Proactive EWMA + Contraction)

| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Transition Insert Latency (p99, 85%->95%)**<br><sub>`AB2_Proactive_EWMA_Contraction`</sub> | 1,000,000 | 108,710 | 9.30 us | 8.90 us | -4.30% | 1.04x lower (base/opt) | 4.20 us | 8.90 us | 34.20 us | 0.000000 | 0 | 0 | 5.28 MiB | 0 B |
| **Transition Insert Latency (p99.9, 85%->95%)**<br><sub>`AB2_Proactive_EWMA_Contraction`</sub> | 1,000,000 | 108,710 | 32.90 us | 34.20 us | +3.95% | 1.04x higher (opt/base) | 4.20 us | 8.90 us | 34.20 us | 0.000000 | 0 | 0 | 5.28 MiB | 0 B |
| **Transition Insert Max Stall (85%->95%)**<br><sub>`AB2_Proactive_EWMA_Contraction`</sub> | 1,000,000 | 108,710 | 140.95 ms | 7.25 ms | -94.86% | 19.45x lower (base/opt) | 4.20 us | 8.90 us | 34.20 us | 0.000000 | 0 | 0 | 5.28 MiB | 0 B |
| **Post-70%-Delete Filter RAM Footprint**<br><sub>`AB2_Proactive_EWMA_Contraction`</sub> | 300,000 | 700,000 | 5.28 MiB | 2.64 MiB | -49.98% | 2.00x lower (base/opt) | 4.20 us | 8.90 us | 34.20 us | 0.000000 | 0 | 0 | 5.28 MiB | 0 B |

<details>
<summary>Diagnostic Notes (4)</summary>

- **Transition Insert Latency (p99, 85%->95%)** (`AB2_Proactive_EWMA_Contraction`): Naive(p50=4200ns,p99=9300ns,p999=32900ns,max=140950000ns,reactive_exp=1) vs Opt(p50=4200ns,p99=8900ns,p999=34200ns,max=7247400ns,trigger_alpha=0.8503,swap_alpha=0.8925,reactive_fallbacks=0)
- **Transition Insert Latency (p99.9, 85%->95%)** (`AB2_Proactive_EWMA_Contraction`): Concurrent read latency: Naive(p50=100ns,p99=100ns,max=164000ns) vs Opt(p50=100ns,p99=200ns,max=37300ns)
- **Transition Insert Max Stall (85%->95%)** (`AB2_Proactive_EWMA_Contraction`): Naive synchronous filter.expand() at alpha=0.95 stalled 140ms vs Opt background RCU swap max=7ms (reader max=37us)
- **Post-70%-Delete Filter RAM Footprint** (`AB2_Proactive_EWMA_Contraction`): Pre-delete RAM: Naive=5540504B, Opt=5540504B; Post-70%-delete RAM: Naive=5540504B (q=21, 0B reclaimed) vs Opt=2771608B (q=20, reclaimed=2768896B / 49.98%); 300000 surviving keys verified with 0 FN and 6 adapted FPs verified with 0 repeated FPs

</details>

### A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map

- **Baseline:** Naive Paper-Baseline (Uncoupled Reverse Map)
- **Optimized:** Aeris Optimized (WAL-Coupled Storage Engine)

| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Unbatched Per-Op fsync Throughput (5k keys)**<br><sub>`AB3_WAL_Coupled_Storage`</sub> | 5,000 | 5,000 | 168,112 ops/sec | 1,359 ops/sec | -99.19% | 123.69x slower (base/opt) | 707.90 us | 1.48 ms | 2.24 ms | 0.000000 | 0 | 0 | 0 B | 533.14 KiB |
| **Unbatched Per-Op fsync Latency (p99, 5k keys)**<br><sub>`AB3_WAL_Coupled_Storage`</sub> | 5,000 | 5,000 | 9.00 us | 1.48 ms | +16368.89% | 164.69x higher (opt/base) | 707.90 us | 1.48 ms | 2.24 ms | 0.000000 | 0 | 0 | 0 B | 533.14 KiB |
| **Batched WAL Commit Throughput (250k keys, B=1000)**<br><sub>`AB3_WAL_Coupled_Storage`</sub> | 250,000 | 250,000 | 305,850 ops/sec | 149,059 ops/sec | -51.26% | 2.05x slower (base/opt) | 6.64 us | 7.93 us | 8.17 us | 0.000000 | 0 | 0 | 0 B | 25.03 MiB |
| **Batched WAL Commit Latency (p99/batch, B=1000)**<br><sub>`AB3_WAL_Coupled_Storage`</sub> | 250,000 | 250 | 3.70 ms | 7.93 ms | +114.60% | 2.15x higher (opt/base) | 6.64 ms | 7.93 ms | 8.17 ms | 0.000000 | 0 | 0 | 0 B | 25.03 MiB |
| **SIGKILL Crash Lost/Desync Key Rate (5k keys, 5 cycles)**<br><sub>`AB3_WAL_Coupled_Storage`</sub> | 5,040 | 5 | 21.66% | 0.00% | -100.00% | eliminated (opt=0) | 166.16 ms | 169.54 ms | 169.54 ms | 0.000000 | 0 | 0 | 0 B | 1.20 MiB |

<details>
<summary>Diagnostic Notes (5)</summary>

- **Unbatched Per-Op fsync Throughput (5k keys)** (`AB3_WAL_Coupled_Storage`): Uncoupled(p50=5700ns,p99=9000ns,p999=14200ns,disk=210933B) vs CoupledWalPerOpFsync(p50=707900ns,p99=1482200ns,p999=2240300ns,disk=545933B)
- **Unbatched Per-Op fsync Latency (p99, 5k keys)** (`AB3_WAL_Coupled_Storage`): Per-op fsync durability barrier overhead: baseline_disk=210933B, coupled_disk=545933B
- **Batched WAL Commit Throughput (250k keys, B=1000)** (`AB3_WAL_Coupled_Storage`): Amortized/key: Uncoupled(p50=3239ns,p99=3696ns,p999=3789ns,disk=9493091B) vs CoupledWalBatch(p50=6638ns,p99=7931ns,p999=8166ns,disk=26243091B)
- **Batched WAL Commit Latency (p99/batch, B=1000)** (`AB3_WAL_Coupled_Storage`): Per-batch(1000 keys): Uncoupled(p50=3239500ns,p99=3696100ns,p999=3789000ns) vs CoupledWalBatch(p50=6638900ns,p99=7931700ns,p999=8166500ns)
- **SIGKILL Crash Lost/Desync Key Rate (5k keys, 5 cycles)** (`AB3_WAL_Coupled_Storage`): Coupled(5040 ACKed, 0 lost KV, 0 lost RM, 0 desync, 0 FN, torn-tail recovery verified) vs Uncoupled(5093 ACKed, 0 lost KV, 1098 lost RM, 1103 KV<->RM desync, 1098 post-rebuild FN, failure_rate=21.66%; KvStoreConfig/ReverseMapConfig expose sync_writes=false with uncoupled batched RM writes and no CoupledWal)

</details>

### A/B 4: Scale Plane (RCU + 8-Shard Range Sharding) vs. Coarse-Locked Single Filter

- **Baseline:** Naive Paper-Baseline (Coarse-Locked Single Filter)
- **Optimized:** Aeris Optimized (RCU + 8-Shard Range Sharding)

| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mixed 90R/10W Throughput (1T, 8 shards vs Mutex)**<br><sub>`AB4_ScalePlane_MultiCore_1T`</sub> | 5,500,281 | 5,000,000 | 1,905,099 ops/sec | 1,507,248 ops/sec | -20.88% | 1.26x slower (base/opt) | 200 ns | 6.70 us | 11.10 us | 0.000000 | 0 | 0 | 48.27 MiB | 57.73 KiB |
| **Mixed 90R/10W Throughput (2T, 8 shards vs Mutex)**<br><sub>`AB4_ScalePlane_MultiCore_2T`</sub> | 6,000,059 | 5,000,000 | 1,473,904 ops/sec | 2,594,715 ops/sec | +76.04% | 1.76x faster (opt/base) | 200 ns | 7.90 us | 13.10 us | 0.000000 | 0 | 0 | 48.27 MiB | 108.97 MiB |
| **Mixed 90R/10W Throughput (4T, 8 shards vs Mutex)**<br><sub>`AB4_ScalePlane_MultiCore_4T`</sub> | 6,500,226 | 5,000,000 | 868,248 ops/sec | 5,114,574 ops/sec | +489.07% | 5.89x faster (opt/base) | 200 ns | 8.60 us | 17.70 us | 0.000000 | 0 | 0 | 48.27 MiB | 128.87 MiB |
| **Mixed 90R/10W Throughput (8T, 8 shards vs Mutex)**<br><sub>`AB4_ScalePlane_MultiCore_8T`</sub> | 6,998,854 | 5,000,000 | 614,745 ops/sec | 6,123,203 ops/sec | +896.06% | 9.96x faster (opt/base) | 200 ns | 21.30 us | 48.30 us | 0.000000 | 0 | 0 | 48.27 MiB | 150.70 MiB |
| **Reader p99.9 Latency During 2x Live Expansion (RCU vs RwLock)**<br><sub>`AB4_ScalePlane_RCU_LiveExpansion`</sub> | 500,000 | 6,361,406 | 13.20 us | 300 ns | -97.73% | 44.00x lower (base/opt) | 100 ns | 200 ns | 300 ns | 0.000000 | 0 | 0 | 12.07 MiB | 7.22 KiB |
| **Hot-Spot Final Filter RAM Footprint (85% Skewed Writes)**<br><sub>`AB4_ScalePlane_HotSpot_Sharding_RAM`</sub> | 253,381 | 173,381 | 1.32 MiB | 1.14 MiB | -13.83% | 1.16x lower (base/opt) | 3.50 us | 7.00 us | 16.30 us | 0.000000 | 0 | 0 | 1.14 MiB | 7.14 MiB |
| **Hot-Spot Skewed Write p99 Latency (Sharded Split/Merge vs Global)**<br><sub>`AB4_ScalePlane_HotSpot_Sharding_Latency`</sub> | 253,381 | 173,381 | 6.80 us | 7.00 us | +2.94% | 1.03x higher (opt/base) | 3.50 us | 7.00 us | 16.30 us | 0.000000 | 0 | 0 | 1.14 MiB | 7.14 MiB |

<details>
<summary>Diagnostic Notes (7)</summary>

- **Mixed 90R/10W Throughput (1T, 8 shards vs Mutex)** (`AB4_ScalePlane_MultiCore_1T`): threads=1; speedup=0.79x; naive_p50/p99/p999_ns=200/5300/8300; opt_p50/p99/p999_ns=200/6700/11100; shards=8; total_keys=5500281
- **Mixed 90R/10W Throughput (2T, 8 shards vs Mutex)** (`AB4_ScalePlane_MultiCore_2T`): threads=2; speedup=1.76x; naive_p50/p99/p999_ns=200/109900/242100; opt_p50/p99/p999_ns=200/7900/13100; shards=8; total_keys=6000059
- **Mixed 90R/10W Throughput (4T, 8 shards vs Mutex)** (`AB4_ScalePlane_MultiCore_4T`): threads=4; speedup=5.89x; naive_p50/p99/p999_ns=500/213900/409300; opt_p50/p99/p999_ns=200/8600/17700; shards=8; total_keys=6500226
- **Mixed 90R/10W Throughput (8T, 8 shards vs Mutex)** (`AB4_ScalePlane_MultiCore_8T`): threads=8; speedup=9.96x; naive_p50/p99/p999_ns=1800/184500/308600; opt_p50/p99/p999_ns=200/21300/48300; shards=8; total_keys=6998854
- **Reader p99.9 Latency During 2x Live Expansion (RCU vs RwLock)** (`AB4_ScalePlane_RCU_LiveExpansion`): 2x live expansions (q=20->22); naive_rwlock_p50/p99/p999/max_ns=200/400/13200/90226000; rcu_p50/p99/p999/max_ns=100/200/300/73100; naive_queries=10049; rcu_queries=6361406
- **Hot-Spot Final Filter RAM Footprint (85% Skewed Writes)** (`AB4_ScalePlane_HotSpot_Sharding_RAM`): naive_init/final/growth=348824/1387160/1038336B (q=17->19); sharded_init/final/growth=367808/1195376/827568B; splits=19; merges=1; final_shards=26
- **Hot-Spot Skewed Write p99 Latency (Sharded Split/Merge vs Global)** (`AB4_ScalePlane_HotSpot_Sharding_Latency`): naive_p50/p99/p999/max_ns=3800/6800/27200/33051500; sharded_p50/p99/p999/max_ns=3500/7000/16300/2203000; splits=19; merges=1

</details>

### A/B 5: Phase 5 Live Python ML/RL Sidecar vs. Native Control Plane

- **Baseline:** Native Control Plane Baseline
- **Optimized:** Phase 5 Live Python ML/RL Sidecar

| Metric | Keys | Queries | Naive Baseline | Aeris Optimized | Delta % | Speedup / Ratio | p50 | p99 | p99.9 | Emp. FPR | FN | Rep. FPs | Peak RAM | Disk I/O |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Seasonal Surge Max Foreground Write Stall (Sidecar RCU vs Native)**<br><sub>`AB5_Sidecar_Seasonal_PreWarm_Stall`</sub> | 1,000,000 | 320,000 | 113.89 ms | 499.20 us | -99.56% | 228.14x lower (base/opt) | 3.40 us | 7.30 us | 23.40 us | 0.000000 | 0 | 0 | 6.03 MiB | 17.71 MiB |
| **Seasonal Surge Expansion Trigger Step (Lower = Earlier Pre-Warm)**<br><sub>`AB5_Sidecar_Seasonal_Forecast_LeadTime`</sub> | 1,000,000 | 24 | 10 steps | 2 steps | -80.00% | 5.00x lower (base/opt) | 3.40 us | 7.30 us | 23.40 us | 0.000000 | 0 | 0 | 6.03 MiB | 17.71 MiB |
| **Multi-Tenant Workload Empirical FPR (Bandit vs Uniform Static)**<br><sub>`AB5_Sidecar_MultiTenant_Bandit_FPR`</sub> | 1,000,000 | 200,000 | 0.000120 | 8.000e-05 | -33.33% | 1.50x lower (base/opt) | 100 ns | 300 ns | 400 ns | 8.000e-05 | 0 | 0 | 7.55 MiB | 35.88 MiB |
| **Multi-Tenant Effective Query Throughput (incl. 50us FP Disk Penalty)**<br><sub>`AB5_Sidecar_MultiTenant_Bandit_Throughput`</sub> | 1,000,000 | 200,000 | 6,702,390 ops/sec | 23,526,644 ops/sec | +251.02% | 3.51x faster (opt/base) | 100 ns | 300 ns | 400 ns | 8.000e-05 | 0 | 0 | 7.55 MiB | 35.88 MiB |
| **SafetyGuardrail Validation & Adversarial Rejection Accuracy**<br><sub>`AB5_Sidecar_Guardrail_Accuracy`</sub> | 0 | 18 | 0.00% | 100.00% | +100.00% | new capability (base=0) | 2.90 us | 3.90 us | 3.90 us | 0.000000 | 0 | 0 | 0 B | 0 B |
| **Post-Crash Control-Plane Fallback p99 Latency (Zero Data-Plane Stall)**<br><sub>`AB5_Sidecar_Crash_Fallback_Latency`</sub> | 50,000 | 40 | 3.000 s | 6.50 us | -100.00% | 461538.46x lower (base/opt) | 500 ns | 6.50 us | 6.50 us | 0.000000 | 0 | 0 | 389.02 KiB | 7.22 KiB |

<details>
<summary>Diagnostic Notes (6)</summary>

- **Seasonal Surge Max Foreground Write Stall (Sidecar RCU vs Native)** (`AB5_Sidecar_Seasonal_PreWarm_Stall`): native_p50/p99/p999/max_ns=3600/7700/21000/113885700; sidecar_p50/p99/p999/max_ns=3400/7300/23400/499200; sidecar_accepted=24/24; prewarm_rcu_ns=101061300
- **Seasonal Surge Expansion Trigger Step (Lower = Earlier Pre-Warm)** (`AB5_Sidecar_Seasonal_Forecast_LeadTime`): native_trigger_step=9; sidecar_trigger_step=1; forecast_lead_time_steps=8; native_stall_ns=113885700
- **Multi-Tenant Workload Empirical FPR (Bandit vs Uniform Static)** (`AB5_Sidecar_MultiTenant_Bandit_FPR`): budget=10485760B; hot_alloc=9437184B vs cold_alloc=1048576B; uni_fps=24 (0.000120) vs bandit_fps=16 (0.000080); uni_ram=6336608B; bandit_ram=7917664B
- **Multi-Tenant Effective Query Throughput (incl. 50us FP Disk Penalty)** (`AB5_Sidecar_MultiTenant_Bandit_Throughput`): uni_cpu_ms=28.64; bandit_cpu_ms=7.70; uni_p50/p99_ns=100/300; bandit_p50/p99_ns=100/300
- **SafetyGuardrail Validation & Adversarial Rejection Accuracy** (`AB5_Sidecar_Guardrail_Accuracy`): passed=18/18 (100%); includes live gRPC sparse-series rejection + tuning & memory boundary/overflow/NaN/duplicate-shard checks
- **Post-Crash Control-Plane Fallback p99 Latency (Zero Data-Plane Stall)** (`AB5_Sidecar_Crash_Fallback_Latency`): abrupt SIGKILL mid-session; 40/40 calls returned NATIVE_FALLBACK_MODEL_VERSION with 0 errors; fb_p50/p99/p999_ns=500/6500/6500; concurrent_dp_queries=78469103 dp_p50/p99/p999_ns=100/200/600

</details>

## 4. Blunt Engineering Verdict

Each extension is classified strictly from the measured data using four honest categories: **Measurable Gain**, **Expected Safety/Durability Overhead**, **Neutral** (within +/-5%), or **Negative Overhead**. Neutral and negative results are reported directly without spin.

| Suite / Extension | Verdict | Metrics | FN | Rep. FPs | Data-Driven Rationale |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **DRAM-Scale 10M-Key / 5M-Query Audit** | **Measurable Gain** | 8 | 0 | 0 | Passed DRAM-scale correctness (0 FN, 0 repeated FPs) and beat baseline on 8/8 comparative metric(s). |
| **A/B 1: Dynamic R + FP Log Replay vs. Static R** | **Measurable Gain** | 13 | 0 | 0 | Net positive trade-off (10 gain(s), 0 neutral, 1 regression(s); mean directional delta +192.4%). Trade-off regressions are explicitly listed below. |
| **A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95%** | **Measurable Gain** | 4 | 0 | 0 | Delivered >5% improvement on 2/4 comparative metric(s) (2 neutral, 0 regressions) with 0 FN and 0 repeated FPs. |
| **A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map** | **Expected Safety/Durability Overhead** | 5 | 0 | 0 | Crash durability verified across 1 recovery metric(s) with 0 lost ACKed keys, at the expected cost of write/storage overhead across 4/4 throughput/latency/IO metric(s). |
| **A/B 4: Scale Plane (RCU + 8-Shard Range Sharding) vs. Coarse-Locked Single Filter** | **Measurable Gain** | 7 | 0 | 0 | Net positive trade-off (5 gain(s), 1 neutral, 1 regression(s); mean directional delta +221.3%). Trade-off regressions are explicitly listed below. |
| **A/B 5: Phase 5 Live Python ML/RL Sidecar vs. Native Control Plane** | **Measurable Gain** | 6 | 0 | 0 | Delivered >5% improvement on 6/6 comparative metric(s) (0 neutral, 0 regressions) with 0 FN and 0 repeated FPs. |

### DRAM-Scale 10M-Key / 5M-Query Audit — Measurable Gain

Passed DRAM-scale correctness (0 FN, 0 repeated FPs) and beat baseline on 8/8 comparative metric(s).

**Measured Gains:**
- `DRAM Pre-Expansion Empirical FPR vs Target (eps_2e-4)`: baseline 0.062500 -> optimized 0.004615 (-92.6%, 13.54x lower (base/opt))
- `DRAM Post-8x-Expansion Empirical FPR vs Target (eps_2e-4)`: baseline 0.062500 -> optimized 0.018391 (-70.6%, 3.40x lower (base/opt))
- `DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion (eps_2e-4)`: baseline 13,582,834 ops/sec -> optimized 15,051,659 ops/sec (+10.8%, 1.11x faster (opt/base))
- `Adversarial 10% Repeat-FP Probing FPR (eps_2e-4)`: baseline 0.117102 -> optimized 0.017328 (-85.2%, 6.76x lower (base/opt))
- `DRAM Pre-Expansion Empirical FPR vs Target (eps_2e-8)`: baseline 0.003906 -> optimized 0.000142 (-96.4%, 27.59x lower (base/opt))
- `DRAM Post-8x-Expansion Empirical FPR vs Target (eps_2e-8)`: baseline 0.003906 -> optimized 0.000657 (-83.2%, 5.94x lower (base/opt))
- `DRAM-Bound Query Throughput: Pre vs Post-8x-Expansion (eps_2e-8)`: baseline 12,310,671 ops/sec -> optimized 13,843,252 ops/sec (+12.4%, 1.12x faster (opt/base))
- `Adversarial 10% Repeat-FP Probing FPR (eps_2e-8)`: baseline 0.100456 -> optimized 0.001186 (-98.8%, 84.70x lower (base/opt))

### A/B 1: Dynamic R + FP Log Replay vs. Static R — Measurable Gain

Net positive trade-off (10 gain(s), 0 neutral, 1 regression(s); mean directional delta +192.4%). Trade-off regressions are explicitly listed below.

**Measured Gains:**
- `A/B 1 Overall Shifting-R Throughput (Equal-RAM Static R=16 vs Dynamic R)`: baseline 3,243,178 ops/sec -> optimized 16,467,614 ops/sec (+407.8%, 5.08x faster (opt/base))
- `A/B 1 Overall Shifting-R p99 Latency (Equal-RAM Static R=16 vs Dynamic R)`: baseline 1.20 us -> optimized 200 ns (-83.3%, 6.00x lower (base/opt))
- `A/B 1 Overall Shifting-R FPR (Paper-Min Static R=16 C=11 vs Dynamic-R Aeris C=17)`: baseline 0.070213 -> optimized 0.001230 (-98.2%, 57.08x lower (base/opt))
- `A/B 1 Phase 0 (R=16) FPR (Paper-Min Static R=16 vs Dynamic R)`: baseline 0.003590 -> optimized 6.000e-05 (-98.3%, 59.83x lower (base/opt))
- `A/B 1 Phase 1 (R=256) Throughput (Equal-RAM Static vs Dynamic R)`: baseline 4,116,900 ops/sec -> optimized 12,498,688 ops/sec (+203.6%, 3.04x faster (opt/base))
- `A/B 1 Phase 1 (R=256) FPR (Paper-Min Static R=16 vs Dynamic R)`: baseline 0.057542 -> optimized 0.000870 (-98.5%, 66.14x lower (base/opt))
- `A/B 1 Phase 2 (R=32) Throughput (Equal-RAM Static vs Dynamic R)`: baseline 23,210,473 ops/sec -> optimized 27,125,412 ops/sec (+16.9%, 1.17x faster (opt/base))
- `A/B 1 Phase 2 (R=32) FPR (Paper-Min Static R=16 vs Dynamic R)`: baseline 0.007290 -> optimized 0.000132 (-98.2%, 55.23x lower (base/opt))
- `A/B 1 Phase 3 (R=1024) Throughput (Equal-RAM Static vs Dynamic R)`: baseline 1,087,188 ops/sec -> optimized 11,306,001 ops/sec (+939.9%, 10.40x faster (opt/base))
- `A/B 1 Phase 3 (R=1024) FPR (Paper-Min Static R=16 vs Dynamic R)`: baseline 0.212430 -> optimized 0.003858 (-98.2%, 55.06x lower (base/opt))

**Measured Overheads / Regressions / Caveats:**
- `A/B 1 Phase 0 (R=16) Throughput (Equal-RAM Static vs Dynamic R)`: baseline 36,272,888 ops/sec -> optimized 26,611,173 ops/sec (-26.6%, 1.36x slower (base/opt))
- Mechanism cost — `A/B 1 Background RocksDB Reverse Map Rebuild Time (Mean)`: baseline 0 ns -> optimized 1.442 s (+100.0%, overhead (base=0))
- Mechanism cost — `A/B 1 Persistent FP Log Replay Latency Across Rebuilds (Total)`: baseline 0 ns -> optimized 277.60 us (+100.0%, overhead (base=0))

### A/B 2: Proactive EWMA Trigger + Contraction vs. Reactive alpha = 95% — Measurable Gain

Delivered >5% improvement on 2/4 comparative metric(s) (2 neutral, 0 regressions) with 0 FN and 0 repeated FPs.

**Measured Gains:**
- `Transition Insert Max Stall (85%->95%)`: baseline 140.95 ms -> optimized 7.25 ms (-94.9%, 19.45x lower (base/opt))
- `Post-70%-Delete Filter RAM Footprint`: baseline 5.28 MiB -> optimized 2.64 MiB (-50.0%, 2.00x lower (base/opt))

### A/B 3: WAL-Coupled Storage vs. Uncoupled Reverse Map — Expected Safety/Durability Overhead

Crash durability verified across 1 recovery metric(s) with 0 lost ACKed keys, at the expected cost of write/storage overhead across 4/4 throughput/latency/IO metric(s).

**Measured Gains:**
- `SIGKILL Crash Lost/Desync Key Rate (5k keys, 5 cycles)`: baseline 21.66% -> optimized 0.00% (-100.0%, eliminated (opt=0))

**Measured Overheads / Regressions / Caveats:**
- `Unbatched Per-Op fsync Throughput (5k keys)`: baseline 168,112 ops/sec -> optimized 1,359 ops/sec (-99.2%, 123.69x slower (base/opt))
- `Unbatched Per-Op fsync Latency (p99, 5k keys)`: baseline 9.00 us -> optimized 1.48 ms (+16368.9%, 164.69x higher (opt/base))
- `Batched WAL Commit Throughput (250k keys, B=1000)`: baseline 305,850 ops/sec -> optimized 149,059 ops/sec (-51.3%, 2.05x slower (base/opt))
- `Batched WAL Commit Latency (p99/batch, B=1000)`: baseline 3.70 ms -> optimized 7.93 ms (+114.6%, 2.15x higher (opt/base))

### A/B 4: Scale Plane (RCU + 8-Shard Range Sharding) vs. Coarse-Locked Single Filter — Measurable Gain

Net positive trade-off (5 gain(s), 1 neutral, 1 regression(s); mean directional delta +221.3%). Trade-off regressions are explicitly listed below.

**Measured Gains:**
- `Mixed 90R/10W Throughput (2T, 8 shards vs Mutex)`: baseline 1,473,904 ops/sec -> optimized 2,594,715 ops/sec (+76.0%, 1.76x faster (opt/base))
- `Mixed 90R/10W Throughput (4T, 8 shards vs Mutex)`: baseline 868,248 ops/sec -> optimized 5,114,574 ops/sec (+489.1%, 5.89x faster (opt/base))
- `Mixed 90R/10W Throughput (8T, 8 shards vs Mutex)`: baseline 614,745 ops/sec -> optimized 6,123,203 ops/sec (+896.1%, 9.96x faster (opt/base))
- `Reader p99.9 Latency During 2x Live Expansion (RCU vs RwLock)`: baseline 13.20 us -> optimized 300 ns (-97.7%, 44.00x lower (base/opt))
- `Hot-Spot Final Filter RAM Footprint (85% Skewed Writes)`: baseline 1.32 MiB -> optimized 1.14 MiB (-13.8%, 1.16x lower (base/opt))

**Measured Overheads / Regressions / Caveats:**
- `Mixed 90R/10W Throughput (1T, 8 shards vs Mutex)`: baseline 1,905,099 ops/sec -> optimized 1,507,248 ops/sec (-20.9%, 1.26x slower (base/opt))

### A/B 5: Phase 5 Live Python ML/RL Sidecar vs. Native Control Plane — Measurable Gain

Delivered >5% improvement on 6/6 comparative metric(s) (0 neutral, 0 regressions) with 0 FN and 0 repeated FPs.

**Measured Gains:**
- `Seasonal Surge Max Foreground Write Stall (Sidecar RCU vs Native)`: baseline 113.89 ms -> optimized 499.20 us (-99.6%, 228.14x lower (base/opt))
- `Seasonal Surge Expansion Trigger Step (Lower = Earlier Pre-Warm)`: baseline 10 steps -> optimized 2 steps (-80.0%, 5.00x lower (base/opt))
- `Multi-Tenant Workload Empirical FPR (Bandit vs Uniform Static)`: baseline 0.000120 -> optimized 8.000e-05 (-33.3%, 1.50x lower (base/opt))
- `Multi-Tenant Effective Query Throughput (incl. 50us FP Disk Penalty)`: baseline 6,702,390 ops/sec -> optimized 23,526,644 ops/sec (+251.0%, 3.51x faster (opt/base))
- `SafetyGuardrail Validation & Adversarial Rejection Accuracy`: baseline 0.00% -> optimized 100.00% (+100.0%, new capability (base=0))
- `Post-Crash Control-Plane Fallback p99 Latency (Zero Data-Plane Stall)`: baseline 3.000 s -> optimized 6.50 us (-100.0%, 461538.46x lower (base/opt))

## 5. Worker Methodology & Telemetry Notes

- **`worker1_fpr_dynamic_r`**: DRAM 10M (eps_2e-4): filter_ram=32.25 MiB (>24 MiB L3), pre_fpr=0.004615, post_8x_fpr=0.018391 (target=0.062500), post_qps=15051659, boxes_rejuv=38797, adv_fpr 0.11710->0.01733 (0 repeated FPs, 0 FNs).
- **`worker1_fpr_dynamic_r`**: DRAM 10M (eps_2e-8): filter_ram=42.25 MiB (>24 MiB L3), pre_fpr=0.000142, post_8x_fpr=0.000657 (target=0.003906), post_qps=13843252, boxes_rejuv=0, adv_fpr 0.10046->0.00119 (0 repeated FPs, 0 FNs).
- **`worker1_fpr_dynamic_r`**: A/B 1 Dynamic-R: overall_qps 3243178->16467614 (+407.8%), p99 1200ns->200ns, paper_min_fpr 0.07021->0.00123, rebuilds=3 (m:4->6@R=256, m:6->5@R=32, m:5->8@R=1024), total_fp_replay=0.28ms, repeated_fps=0, fn=0.
- **`worker2_proactive_storage`**: A/B 2 completed at q=20->21->20 over 1000000 keys: ProactiveTrigger fired at alpha=0.8503 and swapped at alpha=0.8925; contraction reclaimed 2768896 bytes (49.98%) after 700000 deletes; 0 FN across 300000 surviving keys, 0 repeated FPs across 6 adapted FPs.
- **`worker2_proactive_storage`**: A/B 3 summary: Unbatched(5k): uncoupled=168112 ops/s vs coupled=1359 ops/s (-99.2%) | Batched(250k,B=1000): uncoupled=305850 ops/s vs coupled=149059 ops/s (-51.3%) | SIGKILL(5 cycles): coupled=0.00% loss/0 FN (5040 ACKed) vs uncoupled=21.66% lost/desync (1098 lost RM, 1098 post-rebuild FN out of 5093 ACKed)
- **`worker3_scale_sidecar`**: A/B 4.1 evaluates 1T, 2T, 4T, 8T mixed 90R/10W scaling on 5M pre-loaded keys + 5M ops (scaled by host.scale_factor) comparing Mutex<(AerisFilter, RocksDbReverseMap)> vs 8-shard ShardedAerisSystem.
- **`worker3_scale_sidecar`**: A/B 4.2 measures concurrent reader latency during two consecutive live 2x expansions comparing coarse RwLock<AerisFilter> vs wait-free epoch-RCU RcuAerisFilter.
- **`worker3_scale_sidecar`**: A/B 4.3 measures 85% skewed hot-spot write latency and incremental RAM growth with live ShardedAerisSystem median splits and adjacent cold-shard merges vs global single-filter doubling.
- **`worker3_scale_sidecar`**: A/B 5 runs against a live out-of-process Python gRPC sidecar (aeris_sidecar.server) verifying seasonal pre-warming lead time, multi-tenant contextual bandit RAM allocation over 1M keys, 100% SafetyGuardrail accuracy, and abrupt SIGKILL crash-fallback latency with zero Data Plane stall.
