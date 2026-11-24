//! # `aeris-core`
//!
//! Data Plane implementation of the Aeris Filter in RAM:
//! 64-slot Quotient Filter chunks, SIMD `POPCNT`/`PDEP` rank/select with portable
//! scalar fallbacks, 3-bit keepsake box encoding, Robin Hood displacement,
//! extendable fingerprints, InfiniFilter unary age counters, and selective /
//! opportunistic rejuvenation.
//!
//! # Safety Policy
//! `unsafe` code is strictly isolated inside [`simd`] behind runtime CPU feature
//! detection (`is_x86_feature_detected!`) with explicit `// SAFETY:` proofs and
//! bit-exact portable scalar fallbacks. All other modules contain zero `unsafe`.

#![deny(missing_docs)]

pub mod bitvec;
pub mod chunk;
pub mod filter;
pub mod keepsake;
pub mod simd;

pub use aeris_types;
pub use bitvec::BitPackedSlots;
pub use chunk::{ChunkMetadata, QuotientTable};
pub use filter::AerisFilter;
pub use keepsake::{BoxPayload, KeepsakeBox, KeepsakeGeometry};
