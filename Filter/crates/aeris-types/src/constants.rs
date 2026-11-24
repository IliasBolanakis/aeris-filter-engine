//! Architectural and bit-layout constants for the Aeris Range Filtering System.
//!
//! All constants follow the source-of-truth specification in
//! `Docs/Aeris_Based_Filtering_System.pdf` and `Docs/Aeris Filters.pdf`.

/// Number of slots in a single Quotient Filter metadata chunk.
///
/// Each chunk carries three 64-bit bitmaps (`occupied`, `runend`, `extension`)
/// and an 8-bit right-shift offset, enabling `O(1)` rank/select navigation
/// using `POPCNT` and `PDEP` instructions.
pub const SLOTS_PER_CHUNK: usize = 64;

/// Log base 2 of [`SLOTS_PER_CHUNK`].
pub const SLOTS_PER_CHUNK_LOG2: u8 = 6;

/// Minimum supported quotient bit width (`N = 2^6 = 64`, i.e., one full chunk).
pub const MIN_QUOTIENT_BITS: u8 = 6;

/// Maximum supported quotient bit width (`N = 2^40` slots).
pub const MAX_QUOTIENT_BITS: u8 = 40;

/// Number of metadata bits maintained per Quotient Filter slot
/// (`occupied`, `runend`, `extension`).
pub const METADATA_BITS_PER_SLOT: usize = 3;

/// Maximum unsigned value representable in a chunk's 8-bit inline offset.
pub const MAX_CHUNK_OFFSET: u8 = u8::MAX;

/// Hard load factor saturation ceiling of the Quotient Filter (`alpha <= 0.95`).
///
/// Beyond this threshold, Robin Hood cluster lengths grow exponentially and
/// cause severe insertion and lookup latency stalls.
pub const MAX_LOAD_FACTOR: f64 = 0.95;

/// Default target load factor used during normal operation (`alpha = 0.90`).
pub const DEFAULT_TARGET_LOAD_FACTOR: f64 = 0.90;

/// Minimum load factor accepted by the deterministic Phase 5 `SafetyGuardrail`.
pub const MIN_GUARDRAIL_ALPHA: f64 = 0.50;

/// Maximum load factor accepted by the deterministic Phase 5 `SafetyGuardrail`.
pub const MAX_GUARDRAIL_ALPHA: f64 = 0.90;

/// Minimum proactive rebuild safety margin `k` accepted by the `SafetyGuardrail`.
///
/// A margin `k >= 1.0` ensures background rebuilds begin at or before the
/// projected time to saturation (`T_full <= T_rebuild * k`).
pub const MIN_GUARDRAIL_PROACTIVE_MARGIN: f64 = 1.0;

/// Maximum proactive rebuild safety margin `k` accepted by the `SafetyGuardrail`.
pub const MAX_GUARDRAIL_PROACTIVE_MARGIN: f64 = 3.0;

/// Default minimum ML/RL model confidence required by the `SafetyGuardrail`.
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.80;

/// Maximum discrete memento bit width (`m = log2(P)`) evaluated by the
/// deterministic Discrete Cost Optimizer (`m in 0..=20`).
pub const MAX_MEMENTO_BITS: u8 = 20;

/// Bit width of the fixed-length unsigned integer key domain (`u64`).
pub const KEY_BITS: u8 = 64;

/// Default deterministic hash seed used when callers do not supply one.
pub const DEFAULT_HASH_SEED: u64 = 0xA3E1_5F01_9C4D_7B23;

/// Classification of a Quotient Filter slot according to the Aeris 3-bit
/// keepsake box metadata encoding (`runend`, `extension`).
///
/// # Encoding Invariants
/// - `extension = 1, runend = 0`: Extension bits slot. An extension slot is
///   never the last slot of a keepsake box.
/// - `extension = 0, runend = 0`: Non-terminal payload/memento slot inside a
///   keepsake box.
/// - `extension = 0, runend = 1`: Terminal slot of a keepsake box that is not
///   the final keepsake box in its quotient run.
/// - `extension = 1, runend = 1`: Terminal slot of a keepsake box that also
///   terminates its enclosing quotient run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeepsakeSlotKind {
    /// Non-terminal payload or memento continuation slot (`runend = 0, extension = 0`).
    PayloadInterior,
    /// Fingerprint extension slot (`runend = 0, extension = 1`).
    Extension,
    /// Final slot of a keepsake box, followed by another keepsake box in the same run
    /// (`runend = 1, extension = 0`).
    BoxEnd,
    /// Final slot of both a keepsake box and its enclosing quotient run
    /// (`runend = 1, extension = 1`).
    BoxAndRunEnd,
}

impl KeepsakeSlotKind {
    /// Decodes the slot classification from the `(runend, extension)` metadata bits.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    #[inline]
    pub const fn from_bits(runend: bool, extension: bool) -> Self {
        match (runend, extension) {
            (false, false) => Self::PayloadInterior,
            (false, true) => Self::Extension,
            (true, false) => Self::BoxEnd,
            (true, true) => Self::BoxAndRunEnd,
        }
    }

    /// Encodes the slot classification back into `(runend, extension)` metadata bits.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    #[inline]
    pub const fn to_bits(self) -> (bool, bool) {
        match self {
            Self::PayloadInterior => (false, false),
            Self::Extension => (false, true),
            Self::BoxEnd => (true, false),
            Self::BoxAndRunEnd => (true, true),
        }
    }

    /// Returns `true` if this slot holds fingerprint extension bits.
    #[inline]
    pub const fn is_extension_slot(self) -> bool {
        matches!(self, Self::Extension)
    }

    /// Returns `true` if this slot terminates its keepsake box.
    #[inline]
    pub const fn is_box_end(self) -> bool {
        matches!(self, Self::BoxEnd | Self::BoxAndRunEnd)
    }

    /// Returns `true` if this slot terminates its quotient run.
    #[inline]
    pub const fn is_run_end(self) -> bool {
        matches!(self, Self::BoxAndRunEnd)
    }
}
