# Aeris-Based Optimized Range Filtering System

A clean-room Rust and Python implementation and systems extension of the **Aeris Filter** (*Chesetti et al., SIGMOD 2026*), developed for **Special Topics in Database Systems (COMP613)** at the **School of Electrical and Computer Engineering, Technical University of Crete (2026–2027)**.

For the complete architectural rationale, mathematical formulations (including the offline contextual bandit and seasonal forecaster models), and bare-metal benchmark evaluation up to 500M keys against the official C++ reference, see the compiled technical report:

> **[Aeris_Technical_Report.pdf](Aeris_Technical_Report.pdf)**

---

## Overview

The original Aeris Filter provides spatial robustness, strong and monotonic adaptivity, and deamortized expansion. This project extends it into a modular, production-oriented range filtering engine organized across five planes:

- **Data Plane (`aeris-core`):** Interleaved 64-slot AoS Quotient Filter with SIMD `POPCNT`/`PDEP` navigation, in-place adaptation, zero-allocation inserts, and streaming 8-bit LSD radix bulk loading.
- **Storage Plane (`aeris-storage`):** Crash-consistent CRC32 `CoupledWal` (group-commit batching by default) coupling a RocksDB primary KV store with a write-optimized Reverse Map.
- **Control Plane (`aeris-control`):** Dynamic-R sliding-window quantile estimator (with static-R bypass), discrete cost optimizer, async rebuilds with CRC32 `FalsePositiveLog` replay, and proactive EWMA expansion/contraction.
- **Scale Plane (`aeris-scale`):** 64-slot striped chunk locks, wait-free `crossbeam-epoch` RCU swaps, dynamic range sharding, and single-shard/1T direct bypasses.
- **Advisory Plane (`aeris-sidecar` + `python-sidecar`):** Out-of-process Python gRPC sidecar hosting a Ridge-regularized LinUCB / Doubly-Robust offline contextual bandit and a damped-trend Fourier seasonal forecaster, guarded by a deterministic Rust `SafetyGuardrail` with automatic native fallback.

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| [Aeris_Technical_Report.pdf](Aeris_Technical_Report.pdf) | Full technical report covering design decisions, ML/RL models, and empirical evaluation. |
| [Filter/crates/aeris-types/](Filter/crates/aeris-types/) | Core domain types, cross-plane traits, configs, and telemetry schemas. |
| [Filter/crates/aeris-core/](Filter/crates/aeris-core/) | Data Plane Quotient Filter, keepsake boxes, and SIMD primitives. |
| [Filter/crates/aeris-storage/](Filter/crates/aeris-storage/) | CRC32 `CoupledWal`, RocksDB KV store, and Reverse Map. |
| [Filter/crates/aeris-control/](Filter/crates/aeris-control/) | Dynamic-R estimator/optimizer, `FalsePositiveLog`, and EWMA trigger. |
| [Filter/crates/aeris-scale/](Filter/crates/aeris-scale/) | Chunk locking, epoch-RCU filter swaps, and dynamic range sharding. |
| [Filter/crates/aeris-sidecar/](Filter/crates/aeris-sidecar/) | Async gRPC client, deterministic `SafetyGuardrail`, and native fallback. |
| [Filter/python-sidecar/](Filter/python-sidecar/) | Python gRPC server with offline contextual bandit and seasonal forecaster. |
| [Filter/src/bin/](Filter/src/bin/) | Deterministic binary trace generator (`trace_gen`) and parity driver (`paper_parity`). |
| [parity_results/](parity_results/) | Raw JSON benchmark outputs (`10^5` to `5x10^8` keys). |

---

## Quick Start

### Prerequisites
- Rust toolchain (`rustc` >= 1.85) + C/C++ toolchain (`clang` or MSVC for `rocksdb`).
- Python >= 3.11 (optional, for the gRPC advisory sidecar).

### Build, Lint, and Test (Rust Workspace)

```bash
cd Filter
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib --bins
```

### Run Benchmarks (`trace_gen` + `paper_parity`)

```bash
cd Filter
RUSTFLAGS="-C target-cpu=native" cargo build --release --bins \
  --config 'profile.release.lto="fat"' --config 'profile.release.codegen-units=1'

./target/release/trace_gen keys --count 1000000 --seed 1380 --out keys_1e6.keys.bin
./target/release/paper_parity --mode bulk_load --impl rust_naive --scenario bulk_1e6 \
  --keys keys_1e6.keys.bin --q 21 --r 7 --m 5
```

### Optional: Python gRPC Advisory Sidecar

```bash
cd Filter/python-sidecar
python -m pip install -e ".[dev]"
pytest
python -m aeris_sidecar.server --bind 127.0.0.1:50051
```

---

## Attribution

Clean-room Rust and Python implementation inspired by:

> P. Chesetti, N. Dayan, M. Bender, M. Farach-Colton, and R. Johnson, *"Aeris Filter: A Strongly and Monotonically Adaptive Range Filter"*, Proc. ACM Manag. Data (SIGMOD 2026).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
