//! Configuration and decision telemetry types for [`super::ControlledAerisEngine`].

use super::throttle::IoThrottleConfig;
use crate::estimator::EstimatorConfig;
use crate::fp_log::FpLogConfig;
use crate::trigger::{ProactiveTriggerConfig, TriggerEvaluation};
use aeris_storage::{KvStoreConfig, ReverseMapConfig};
use aeris_types::{
    AerisResult, CostModelParams, FilterConfig, OptimizationDecision, WorkloadStats,
};
use serde::{Deserialize, Serialize};

/// Configuration bundle for [`super::ControlledAerisEngine`].
#[derive(Clone, Debug)]
pub struct ControlledEngineConfig {
    /// Initial [`FilterConfig`] for the active [`aeris_core::AerisFilter`].
    pub filter_config: FilterConfig,
    /// Configuration for [`crate::estimator::SlidingWindowEstimator`].
    pub estimator_config: EstimatorConfig,
    /// Analytical cost model for [`crate::optimizer::DiscreteCostOptimizer`].
    pub cost_model: CostModelParams,
    /// Configuration for [`crate::trigger::ProactiveTrigger`].
    pub trigger_config: ProactiveTriggerConfig,
    /// Configuration for [`crate::fp_log::FalsePositiveLog`].
    pub fp_log_config: FpLogConfig,
    /// I/O rate-limiting policy for background rebuild scans.
    pub throttle_config: IoThrottleConfig,
    /// RocksDB options for [`aeris_storage::ReferenceKvStore`].
    pub kv_store_config: KvStoreConfig,
    /// RocksDB options for generational [`aeris_storage::RocksDbReverseMap`] instances.
    pub reverse_map_config: ReverseMapConfig,
    /// Enables dynamic-`R` workload estimation and dynamic-`R` rebuild recommendations
    /// (default `true`). Setting `false` activates the zero-overhead static-`R` fast-path
    /// bypass while keeping monotonic FP adaptation and proactive capacity triggers active.
    pub dynamic_r_enabled: bool,
}

impl ControlledEngineConfig {
    /// Creates a [`ControlledEngineConfig`] with production defaults around `filter_config`.
    #[must_use]
    pub fn new(filter_config: FilterConfig) -> Self {
        Self {
            filter_config,
            estimator_config: EstimatorConfig::default(),
            cost_model: CostModelParams::default(),
            trigger_config: ProactiveTriggerConfig::default(),
            fp_log_config: FpLogConfig::default(),
            throttle_config: IoThrottleConfig::default(),
            kv_store_config: KvStoreConfig::default(),
            reverse_map_config: ReverseMapConfig::default(),
            dynamic_r_enabled: true,
        }
    }

    /// Overrides [`Self::dynamic_r_enabled`] and synchronizes `estimator_config.enabled`.
    #[must_use]
    pub const fn with_dynamic_r_enabled(mut self, enabled: bool) -> Self {
        self.dynamic_r_enabled = enabled;
        self.estimator_config.enabled = enabled;
        self
    }

    /// Creates a [`ControlledEngineConfig`] configured for static-`R` execution only:
    /// sets `dynamic_r_enabled = false` and `estimator_config.enabled = false` to bypass
    /// per-query workload estimation and suppress dynamic-`R` rebuilds.
    #[must_use]
    pub fn static_r_only(filter_config: FilterConfig) -> Self {
        Self::new(filter_config).with_dynamic_r_enabled(false)
    }

    /// Validates every sub-component configuration.
    ///
    /// # Errors
    /// Returns [`aeris_types::AerisError::InvalidConfig`] on the first violated invariant.
    pub fn validate(&self) -> AerisResult<()> {
        self.filter_config.validate()?;
        self.estimator_config.validate()?;
        self.trigger_config.validate()?;
        self.fp_log_config.validate()?;
        self.throttle_config.validate()?;
        Ok(())
    }
}

/// Result of answering a closed range query through [`super::ControlledAerisEngine::query_range`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryOutcome {
    /// Ground-truth emptiness answer (`true` = at least one live key exists in range).
    pub non_empty: bool,
    /// Whether the RAM [`aeris_core::AerisFilter`] returned `true` (always `false` when bypassed).
    pub filter_returned_positive: bool,
    /// `true` iff the filter returned `true` while [`aeris_storage::ReferenceKvStore`] confirmed empty.
    pub was_false_positive: bool,
    /// `true` iff the filter was bypassed directly to [`aeris_storage::ReferenceKvStore`].
    pub bypassed_filter: bool,
    /// `true` iff `adapt()` and [`crate::fp_log::FalsePositiveLog::record`] ran to eliminate the FP.
    pub adapted: bool,
}

/// Combined Control Plane evaluation produced by [`super::ControlledAerisEngine::evaluate_control_plane`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPlaneDecision {
    /// Sliding-window workload statistics snapshot.
    pub workload: WorkloadStats,
    /// Discrete Cost Optimizer recommendation (`m*`, `r*`, bypass, rebuild gate).
    pub optimization: OptimizationDecision,
    /// Proactive EWMA expansion / contraction state-machine evaluation.
    pub trigger: TriggerEvaluation,
}
