//! Configuration and hysteresis thresholds for [`super::ShardedAerisSystem`].

use aeris_control::{EstimatorConfig, FpLogConfig};
use aeris_storage::ReverseMapConfig;
use aeris_types::{AerisError, AerisResult, FilterConfig, MAX_LOAD_FACTOR, MAX_QUOTIENT_BITS};

use crate::chunk_locks::DEFAULT_MAX_LOCK_STRIPES;

/// Default initial number of contiguous range shards (`+12.5%` incremental RAM growth per shard).
pub const DEFAULT_INITIAL_SHARD_COUNT: usize = 8;

/// Default load factor ceiling at which a shard qualifies for dynamic median splitting.
pub const DEFAULT_SPLIT_LOAD_THRESHOLD: f64 = 0.80;

/// Default minimum live key count required before splitting a shard instead of expanding.
pub const DEFAULT_SPLIT_MIN_KEYS: usize = 128;

/// Default load factor below which two adjacent shards are eligible for merging.
pub const DEFAULT_MERGE_LOAD_THRESHOLD: f64 = 0.20;

/// Maximum permitted post-merge load ratio (`<= 0.50`) ensuring merged shards remain well below split threshold.
pub const MAX_POST_MERGE_LOAD_FACTOR: f64 = 0.50;

/// Default minimum number of active shards retained across merges.
pub const DEFAULT_MIN_SHARDS: usize = 1;

/// Default maximum number of active shards permitted across splits.
pub const DEFAULT_MAX_SHARDS: usize = 256;

/// Configuration parameters governing [`super::ShardedAerisSystem`] topology,
/// split/merge hysteresis thresholds, and per-shard storage/control engines.
#[derive(Clone, Debug)]
pub struct ShardingConfig {
    /// Number of uniform contiguous shards initialized across `[0, u64::MAX]` (default `8`).
    pub initial_shard_count: usize,
    /// Baseline validated [`FilterConfig`] applied to newly created shards.
    pub base_filter_config: FilterConfig,
    /// Load factor ceiling above which a shard with `>= split_min_keys` triggers a split (`0.80`).
    pub split_load_threshold: f64,
    /// Minimum live keys required before splitting a shard (`128`).
    pub split_min_keys: usize,
    /// Upper cap on a single shard's quotient width `q`; shards exceeding `max_shard_q`
    /// are split rather than growing monolithically.
    pub max_shard_q: u8,
    /// Optional explicit key-count ceiling per shard triggering a split during rebalance.
    pub split_key_threshold: Option<usize>,
    /// Load factor below which two adjacent shards may merge if combined occupancy is `<= 0.50` (`0.20`).
    pub merge_load_threshold: f64,
    /// Lower floor on total active shard count (`1`).
    pub min_shards: usize,
    /// Upper ceiling on total active shard count (`256`).
    pub max_shards: usize,
    /// Maximum cache-line-padded chunk lock stripes allocated per shard generation.
    pub max_lock_stripes_per_shard: usize,
    /// RocksDB options for each shard's [`aeris_storage::RocksDbReverseMap`].
    pub reverse_map_config: ReverseMapConfig,
    /// Persistence options for each shard's [`aeris_control::FalsePositiveLog`].
    pub fp_log_config: FpLogConfig,
    /// Sliding-window workload estimator options for each shard.
    pub estimator_config: EstimatorConfig,
    /// When `true` (default) and the active topology contains a single shard,
    /// `query_point` / `query_range` (and their `_direct` variants) bypass
    /// `route_key_index`, `max_keys` comparisons, and range clamping completely.
    pub single_shard_bypass: bool,
}

impl ShardingConfig {
    /// Creates a [`ShardingConfig`] with production defaults centered on `base_filter_config`.
    #[must_use]
    pub fn new(base_filter_config: FilterConfig) -> Self {
        let max_shard_q = base_filter_config
            .q
            .saturating_add(2)
            .min(MAX_QUOTIENT_BITS);
        Self {
            initial_shard_count: DEFAULT_INITIAL_SHARD_COUNT,
            base_filter_config,
            split_load_threshold: DEFAULT_SPLIT_LOAD_THRESHOLD,
            split_min_keys: DEFAULT_SPLIT_MIN_KEYS,
            max_shard_q,
            split_key_threshold: None,
            merge_load_threshold: DEFAULT_MERGE_LOAD_THRESHOLD,
            min_shards: DEFAULT_MIN_SHARDS,
            max_shards: DEFAULT_MAX_SHARDS,
            max_lock_stripes_per_shard: DEFAULT_MAX_LOCK_STRIPES,
            reverse_map_config: ReverseMapConfig::default(),
            fp_log_config: FpLogConfig::default(),
            estimator_config: EstimatorConfig::default(),
            single_shard_bypass: true,
        }
    }

    /// Overrides `initial_shard_count`.
    #[must_use]
    pub const fn with_initial_shard_count(mut self, count: usize) -> Self {
        self.initial_shard_count = count;
        self
    }

    /// Overrides `split_load_threshold`.
    #[must_use]
    pub const fn with_split_load_threshold(mut self, threshold: f64) -> Self {
        self.split_load_threshold = threshold;
        self
    }

    /// Overrides `split_min_keys`.
    #[must_use]
    pub const fn with_split_min_keys(mut self, min_keys: usize) -> Self {
        self.split_min_keys = min_keys;
        self
    }

    /// Overrides `max_shard_q`.
    #[must_use]
    pub const fn with_max_shard_q(mut self, max_q: u8) -> Self {
        self.max_shard_q = max_q;
        self
    }

    /// Overrides `split_key_threshold`.
    #[must_use]
    pub const fn with_split_key_threshold(mut self, threshold: Option<usize>) -> Self {
        self.split_key_threshold = threshold;
        self
    }

    /// Overrides `merge_load_threshold`.
    #[must_use]
    pub const fn with_merge_load_threshold(mut self, threshold: f64) -> Self {
        self.merge_load_threshold = threshold;
        self
    }

    /// Overrides `min_shards`.
    #[must_use]
    pub const fn with_min_shards(mut self, min_shards: usize) -> Self {
        self.min_shards = min_shards;
        self
    }

    /// Overrides `max_shards`.
    #[must_use]
    pub const fn with_max_shards(mut self, max_shards: usize) -> Self {
        self.max_shards = max_shards;
        self
    }

    /// Overrides `reverse_map_config`.
    #[must_use]
    pub fn with_reverse_map_config(mut self, cfg: ReverseMapConfig) -> Self {
        self.reverse_map_config = cfg;
        self
    }

    /// Overrides `fp_log_config`.
    #[must_use]
    pub const fn with_fp_log_config(mut self, cfg: FpLogConfig) -> Self {
        self.fp_log_config = cfg;
        self
    }

    /// Overrides `estimator_config`.
    #[must_use]
    pub fn with_estimator_config(mut self, cfg: EstimatorConfig) -> Self {
        self.estimator_config = cfg;
        self
    }

    /// Overrides `single_shard_bypass` (default `true`).
    #[must_use]
    pub const fn with_single_shard_bypass(mut self, enabled: bool) -> Self {
        self.single_shard_bypass = enabled;
        self
    }

    /// Validates all structural, numerical, and sub-component invariants.
    ///
    /// # Errors
    /// Returns [`AerisError::InvalidConfig`] on the first violated invariant.
    pub fn validate(&self) -> AerisResult<()> {
        self.base_filter_config.validate()?;
        self.fp_log_config.validate()?;
        self.estimator_config.validate()?;
        self.validate_shard_counts()?;
        self.validate_thresholds()
    }

    fn validate_shard_counts(&self) -> AerisResult<()> {
        if self.min_shards == 0 {
            return Err(AerisError::InvalidConfig(
                "min_shards must be at least 1".to_string(),
            ));
        }
        if self.max_shards < self.min_shards {
            return Err(AerisError::InvalidConfig(format!(
                "max_shards ({}) must be >= min_shards ({})",
                self.max_shards, self.min_shards
            )));
        }
        if self.initial_shard_count < self.min_shards || self.initial_shard_count > self.max_shards
        {
            return Err(AerisError::InvalidConfig(format!(
                "initial_shard_count ({}) must be in [min_shards={}, max_shards={}]",
                self.initial_shard_count, self.min_shards, self.max_shards
            )));
        }
        if self.max_shard_q < self.base_filter_config.q || self.max_shard_q > MAX_QUOTIENT_BITS {
            return Err(AerisError::InvalidConfig(format!(
                "max_shard_q ({}) must be in [base_q={}, {}]",
                self.max_shard_q, self.base_filter_config.q, MAX_QUOTIENT_BITS
            )));
        }
        if self.max_lock_stripes_per_shard == 0 {
            return Err(AerisError::InvalidConfig(
                "max_lock_stripes_per_shard must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_thresholds(&self) -> AerisResult<()> {
        if !self.split_load_threshold.is_finite()
            || self.split_load_threshold <= 0.0
            || self.split_load_threshold > MAX_LOAD_FACTOR
        {
            return Err(AerisError::InvalidConfig(format!(
                "split_load_threshold ({}) must be finite and in (0.0, {MAX_LOAD_FACTOR}]",
                self.split_load_threshold
            )));
        }
        if !self.merge_load_threshold.is_finite()
            || self.merge_load_threshold < 0.0
            || self.merge_load_threshold >= self.split_load_threshold
        {
            return Err(AerisError::InvalidConfig(format!(
                "merge_load_threshold ({}) must be in [0.0, split_load_threshold={})",
                self.merge_load_threshold, self.split_load_threshold
            )));
        }
        if let Some(key_cap) = self.split_key_threshold {
            if key_cap < 2 {
                return Err(AerisError::InvalidConfig(format!(
                    "split_key_threshold ({key_cap}) must be at least 2 when specified"
                )));
            }
        }
        Ok(())
    }
}
