"""Phase 5 Out-of-Process ML/RL Control Plane Sidecar for Aeris.

Provides:
- :class:`SeasonalWorkloadForecaster`: Pure-NumPy damped-trend + Fourier
  harmonic seasonal workload forecaster with calibrated confidence estimation.
- :class:`OfflineContextualBandit`: Pure-NumPy Ridge-regularized LinUCB /
  Doubly-Robust offline contextual bandit for multi-tenant shard RAM allocation.
- Offline training pipeline (:mod:`aeris_sidecar.trainer`) and gRPC server
  (:mod:`aeris_sidecar.server`).
"""

from .bandit import OfflineContextualBandit
from .forecaster import ForecastResult, SeasonalWorkloadForecaster

__all__ = [
    "ForecastResult",
    "OfflineContextualBandit",
    "SeasonalWorkloadForecaster",
]
