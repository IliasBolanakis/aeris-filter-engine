//! Configuration structures and formal mathematical invariants for Aeris filters.
//!
//! Enforces:
//! - `N = 2^q` (power-of-two slot table aligned to 64-slot chunks),
//! - `P = 2^ceil(log2(R))` (power-of-two partition size),
//! - `r + log2(P) = C` (fixed per-slot RAM payload budget),
//! - `r >= ceil(log2(1/epsilon) + log2(log2(1/epsilon)) + 1)` (strong monotonic
//!   adaptivity across unbounded InfiniFilter expansions).

use crate::constants::{
    DEFAULT_HASH_SEED, DEFAULT_MIN_CONFIDENCE, DEFAULT_TARGET_LOAD_FACTOR, MAX_GUARDRAIL_ALPHA,
    MAX_GUARDRAIL_PROACTIVE_MARGIN, MAX_LOAD_FACTOR, MAX_MEMENTO_BITS, MAX_QUOTIENT_BITS,
    MIN_GUARDRAIL_ALPHA, MIN_GUARDRAIL_PROACTIVE_MARGIN, MIN_QUOTIENT_BITS, SLOTS_PER_CHUNK,
};
use crate::error::{AerisError, AerisResult};
use crate::range::{Key, Memento, PartitionId};
use serde::{Deserialize, Serialize};

/// Validated geometric and probabilistic configuration for an Aeris Filter instance.
///
/// # Mathematical Invariants
/// Every instance produced by [`FilterConfig::new`] or verified by [`FilterConfig::validate`]
/// satisfies:
/// 1. `MIN_QUOTIENT_BITS <= q <= MAX_QUOTIENT_BITS` and `N = 2^q` slots.
/// 2. `0.0 < target_epsilon < 0.5`.
/// 3. `max_range_r >= 1`, `m = ceil(log2(max_range_r))`, and `P = 2^m`.
/// 4. `r + m == payload_budget_c` and `payload_budget_c <= 64`.
/// 5. `r >= FilterConfig::min_remainder_bits(target_epsilon)`.
/// 6. `0.0 < max_load_factor <= 0.95`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Quotient bit width `q` (`N = 2^q` slots).
    pub q: u8,
    /// Remainder bit width `r` (effective hash remainder bits on fresh insertion).
    pub r: u8,
    /// Memento bit width `m = log2(P) = ceil(log2(R))`.
    pub m: u8,
    /// Fixed per-slot payload bit budget `C = r + m`.
    pub payload_budget_c: u8,
    /// Maximum supported query range length `R` for which `FPR <= epsilon` is guaranteed.
    pub max_range_r: u64,
    /// Target worst-case false positive rate `epsilon`.
    pub target_epsilon: f64,
    /// Maximum permitted load factor `alpha` before triggering expansion (`<= 0.95`).
    pub max_load_factor: f64,
    /// Deterministic 64-bit hash seed for `h(p(k))`.
    pub hash_seed: u64,
}

impl FilterConfig {
    /// Computes the minimum remainder bit width `r_min(epsilon)` required for
    /// strong and monotonic adaptivity across unbounded expansions:
    ///
    /// `r_min(epsilon) = ceil(log2(1 / epsilon) + log2(log2(1 / epsilon)) + 1)`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `epsilon` is not finite or not in `(0.0, 0.5)`.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    pub fn min_remainder_bits(epsilon: f64) -> AerisResult<u8> {
        if !epsilon.is_finite() || epsilon <= 0.0 || epsilon >= 0.5 {
            return Err(AerisError::InvalidConfig(format!(
                "target_epsilon must be finite and in (0.0, 0.5), got {epsilon}"
            )));
        }
        let inv = 1.0 / epsilon;
        let log_inv = inv.log2();
        // Because epsilon < 0.5, log2(1/epsilon) > 1.0, so log2(log2(1/epsilon)) > 0.0.
        let raw = log_inv + log_inv.log2() + 1.0;
        let ceil_val = raw.ceil();
        if ceil_val > 60.0 {
            return Err(AerisError::InvalidConfig(format!(
                "target_epsilon {epsilon} requires {ceil_val} remainder bits, exceeding 60-bit cap"
            )));
        }
        Ok(ceil_val as u8)
    }

    /// Computes `ceil(log2(r))` for `r >= 1` using exact integer arithmetic.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `r == 0`.
    ///
    /// # Complexity
    /// `O(1)` time and `O(1)` space.
    #[inline]
    pub fn ceil_log2_u64(r: u64) -> AerisResult<u8> {
        if r == 0 {
            return Err(AerisError::InvalidConfig(
                "max_range_r must be at least 1".to_string(),
            ));
        }
        if r == 1 {
            return Ok(0);
        }
        Ok((64 - (r - 1).leading_zeros()) as u8)
    }

    /// Constructs and validates a new [`FilterConfig`] from explicit parameters.
    ///
    /// Automatically computes `m = ceil(log2(max_range_r))` and validates that
    /// `r + m == payload_budget_c` and `r >= r_min(target_epsilon)`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if any mathematical invariant is violated.
    pub fn new(
        q: u8,
        r: u8,
        max_range_r: u64,
        payload_budget_c: u8,
        target_epsilon: f64,
    ) -> AerisResult<Self> {
        let m = Self::ceil_log2_u64(max_range_r)?;
        let cfg = Self {
            q,
            r,
            m,
            payload_budget_c,
            max_range_r,
            target_epsilon,
            max_load_factor: DEFAULT_TARGET_LOAD_FACTOR,
            hash_seed: DEFAULT_HASH_SEED,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Convenience constructor that chooses the minimum valid `r = r_min(target_epsilon)`
    /// and sets `C = r + ceil(log2(max_range_r))`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if any parameter is out of bounds.
    pub fn with_minimum_bits(q: u8, max_range_r: u64, target_epsilon: f64) -> AerisResult<Self> {
        let r = Self::min_remainder_bits(target_epsilon)?;
        let m = Self::ceil_log2_u64(max_range_r)?;
        let c = r.checked_add(m).ok_or_else(|| {
            AerisError::InvalidConfig("r + m overflows u8 payload budget".to_string())
        })?;
        Self::new(q, r, max_range_r, c, target_epsilon)
    }

    /// Overrides the hash seed, returning the updated configuration.
    #[inline]
    pub const fn with_hash_seed(mut self, hash_seed: u64) -> Self {
        self.hash_seed = hash_seed;
        self
    }

    /// Overrides `max_load_factor` after validating `(0.0, 0.95]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `alpha` is outside `(0.0, 0.95]`.
    pub fn with_max_load_factor(mut self, alpha: f64) -> AerisResult<Self> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha > MAX_LOAD_FACTOR {
            return Err(AerisError::InvalidConfig(format!(
                "max_load_factor must be in (0.0, {MAX_LOAD_FACTOR}], got {alpha}"
            )));
        }
        self.max_load_factor = alpha;
        Ok(self)
    }

    /// Verifies all formal mathematical invariants of the configuration.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on the first violated invariant.
    pub fn validate(&self) -> AerisResult<()> {
        if self.q < MIN_QUOTIENT_BITS || self.q > MAX_QUOTIENT_BITS {
            return Err(AerisError::InvalidConfig(format!(
                "quotient bits q={} must be in [{}, {}]",
                self.q, MIN_QUOTIENT_BITS, MAX_QUOTIENT_BITS
            )));
        }

        let expected_m = Self::ceil_log2_u64(self.max_range_r)?;
        if self.m != expected_m {
            return Err(AerisError::InvalidConfig(format!(
                "m={} does not match ceil(log2(R={}))={}",
                self.m, self.max_range_r, expected_m
            )));
        }
        if self.m > 63 {
            return Err(AerisError::InvalidConfig(format!(
                "m={} must be <= 63 so P = 2^m fits in u64",
                self.m
            )));
        }

        let sum = self.r.checked_add(self.m);
        if sum != Some(self.payload_budget_c) {
            return Err(AerisError::InvalidConfig(format!(
                "payload budget invariant violated: r({}) + m({}) != C({})",
                self.r, self.m, self.payload_budget_c
            )));
        }
        if self.payload_budget_c == 0 || self.payload_budget_c > 63 {
            return Err(AerisError::InvalidConfig(format!(
                "payload_budget_c={} must be in [1, 63] to fit slot+age encoding in u64",
                self.payload_budget_c
            )));
        }

        let r_min = Self::min_remainder_bits(self.target_epsilon)?;
        if self.r < r_min {
            return Err(AerisError::InvalidConfig(format!(
                "remainder bits r={} below monotonic adaptivity floor r_min({})={}",
                self.r, self.target_epsilon, r_min
            )));
        }

        if !self.max_load_factor.is_finite()
            || self.max_load_factor <= 0.0
            || self.max_load_factor > MAX_LOAD_FACTOR
        {
            return Err(AerisError::InvalidConfig(format!(
                "max_load_factor={} must be in (0.0, {}]",
                self.max_load_factor, MAX_LOAD_FACTOR
            )));
        }

        Ok(())
    }

    /// Returns total Quotient Filter slot count `N = 2^q`.
    #[inline]
    pub const fn num_slots(&self) -> usize {
        1usize << self.q
    }

    /// Returns total 64-slot chunk count `N / 64`.
    #[inline]
    pub const fn num_chunks(&self) -> usize {
        self.num_slots() / SLOTS_PER_CHUNK
    }

    /// Returns partition size `P = 2^m = 2^ceil(log2(R))`.
    #[inline]
    pub const fn partition_size(&self) -> u64 {
        1u64 << self.m
    }

    /// Returns the bit mask for extracting mementos `(1 << m) - 1`.
    #[inline]
    pub const fn memento_mask(&self) -> u64 {
        if self.m == 0 {
            0
        } else {
            (1u64 << self.m) - 1
        }
    }

    /// Extracts the partition prefix `p(k) = k >> m` of a 64-bit key.
    #[inline]
    pub const fn partition_of(&self, key: Key) -> PartitionId {
        key >> self.m
    }

    /// Extracts the memento suffix `m(k) = k & ((1 << m) - 1)` of a 64-bit key.
    #[inline]
    pub const fn memento_of(&self, key: Key) -> Memento {
        key & self.memento_mask()
    }

    /// Reconstructs the 64-bit key `(partition << m) | memento`.
    #[inline]
    pub const fn compose_key(&self, partition: PartitionId, memento: Memento) -> Key {
        (partition << self.m) | (memento & self.memento_mask())
    }

    /// Returns the bit width of the first fingerprint extension (`m = ceil(log2(R))`
    /// when `m > 0`, or a full slot `r` when `m == 0`).
    #[inline]
    pub const fn first_extension_bits(&self) -> u8 {
        if self.m > 0 {
            self.m
        } else {
            self.r
        }
    }

    /// Returns the bit width of each subsequent fingerprint extension (`r + m`,
    /// one full payload slot).
    #[inline]
    pub const fn subsequent_extension_bits(&self) -> u8 {
        self.r + self.m
    }

    /// Returns the bit width of a single storage slot including the 1-bit
    /// InfiniFilter unary age sentinel (`r + 1 + m = C + 1` bits).
    #[inline]
    pub const fn physical_slot_bits(&self) -> u8 {
        self.payload_budget_c + 1
    }

    /// Produces a derived configuration for a new memento width `new_m` under
    /// the same fixed RAM payload budget `C = r + m`, with `new_r = C - new_m`
    /// and `new_R = 2^new_m`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `new_m > payload_budget_c` or if
    /// the resulting `new_r` falls below `r_min(target_epsilon)`.
    pub fn reconfigure_for_m(&self, new_m: u8) -> AerisResult<Self> {
        let new_r = self.payload_budget_c.checked_sub(new_m).ok_or_else(|| {
            AerisError::InvalidConfig(format!(
                "new_m ({new_m}) exceeds payload_budget_c ({})",
                self.payload_budget_c
            ))
        })?;
        let new_max_range = 1u64.checked_shl(new_m as u32).ok_or_else(|| {
            AerisError::InvalidConfig(format!("new_m ({new_m}) overflows u64 partition size"))
        })?;
        let mut next = self.clone();
        next.m = new_m;
        next.r = new_r;
        next.max_range_r = new_max_range;
        next.validate()?;
        Ok(next)
    }

    /// Produces the expanded configuration with `q + 1` (`2x` capacity).
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `q + 1 > MAX_QUOTIENT_BITS`.
    pub fn expanded(&self) -> AerisResult<Self> {
        let mut next = self.clone();
        next.q = self.q.checked_add(1).ok_or_else(|| {
            AerisError::InvalidConfig("quotient bit width overflow on expansion".to_string())
        })?;
        next.validate()?;
        Ok(next)
    }

    /// Produces the contracted configuration with `q - 1` (`0.5x` capacity).
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] if `q - 1 < MIN_QUOTIENT_BITS`.
    pub fn contracted(&self) -> AerisResult<Self> {
        let mut next = self.clone();
        next.q = self.q.checked_sub(1).ok_or_else(|| {
            AerisError::InvalidConfig("quotient bit width underflow on contraction".to_string())
        })?;
        next.validate()?;
        Ok(next)
    }
}

/// Deterministic cost-model parameters consumed by the Control Plane [`crate::traits::CostOptimizer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostModelParams {
    /// CPU cost (nanoseconds) per single Quotient Filter partition probe.
    pub cpu_probe_ns: f64,
    /// Storage I/O penalty (nanoseconds) incurred on a false positive disk lookup.
    pub disk_io_penalty_ns: f64,
    /// Direct storage scan cost (nanoseconds) when bypassing the range filter.
    pub direct_scan_cost_ns: f64,
    /// Minimum relative cost reduction required before triggering an `R` rebuild
    /// (hysteresis threshold, e.g., `0.15` = 15% improvement required).
    pub rebuild_benefit_threshold: f64,
    /// Maximum discrete `m = log2(P)` value evaluated by the optimizer.
    pub max_m: u8,
}

impl Default for CostModelParams {
    fn default() -> Self {
        Self {
            cpu_probe_ns: 35.0,
            disk_io_penalty_ns: 50_000.0,
            direct_scan_cost_ns: 60_000.0,
            rebuild_benefit_threshold: 0.15,
            max_m: MAX_MEMENTO_BITS,
        }
    }
}

/// Deterministic bounds enforced by the Phase 5 `SafetyGuardrail` on every
/// out-of-process ML/RL sidecar proposal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyBounds {
    /// Target worst-case false positive rate `epsilon` (determines `r_min(epsilon)`).
    pub target_epsilon: f64,
    /// Fixed per-slot payload budget `C` (`r + m <= C`).
    pub payload_budget_c: u8,
    /// Upper bound on memento width `m <= m_max`.
    pub m_max: u8,
    /// Lower bound on target load factor (`>= 0.50`).
    pub min_alpha: f64,
    /// Upper bound on target load factor (`<= 0.90`).
    pub max_alpha: f64,
    /// Lower bound on proactive trigger safety margin `k` (`>= 1.0`).
    pub min_proactive_margin_k: f64,
    /// Upper bound on proactive trigger safety margin `k` (`<= 3.0`).
    pub max_proactive_margin_k: f64,
    /// Minimum required model confidence in `[0.0, 1.0]`.
    pub min_confidence: f64,
}

impl SafetyBounds {
    /// Constructs default safety bounds derived from a validated [`FilterConfig`].
    pub fn from_filter_config(cfg: &FilterConfig) -> Self {
        Self {
            target_epsilon: cfg.target_epsilon,
            payload_budget_c: cfg.payload_budget_c,
            m_max: MAX_MEMENTO_BITS.min(cfg.payload_budget_c),
            min_alpha: MIN_GUARDRAIL_ALPHA,
            max_alpha: MAX_GUARDRAIL_ALPHA,
            min_proactive_margin_k: MIN_GUARDRAIL_PROACTIVE_MARGIN,
            max_proactive_margin_k: MAX_GUARDRAIL_PROACTIVE_MARGIN,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }
}

/// Inclusive key-space boundaries `[min_key, max_key]` owned by a Range Shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShardBounds {
    /// Inclusive lower key bound owned by the shard.
    pub min_key: Key,
    /// Inclusive upper key bound owned by the shard.
    pub max_key: Key,
}

impl ShardBounds {
    /// Constructs validated shard bounds `[min_key, max_key]`.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidRange`] if `min_key > max_key`.
    #[inline]
    pub const fn new(min_key: Key, max_key: Key) -> AerisResult<Self> {
        if min_key > max_key {
            return Err(AerisError::InvalidRange {
                start: min_key,
                end: max_key,
            });
        }
        Ok(Self { min_key, max_key })
    }

    /// Returns bounds covering the entire 64-bit unsigned key universe `[0, u64::MAX]`.
    #[inline]
    pub const fn full_universe() -> Self {
        Self {
            min_key: 0,
            max_key: u64::MAX,
        }
    }

    /// Returns `true` if `key` falls within this shard's inclusive key range.
    #[inline]
    pub const fn contains(&self, key: Key) -> bool {
        key >= self.min_key && key <= self.max_key
    }

    /// Returns `true` if `[min_key, max_key]` intersects `[start, end]`.
    #[inline]
    pub const fn overlaps_range(&self, start: Key, end: Key) -> bool {
        self.min_key <= end && start <= self.max_key
    }
}
