"""Conformal calibration: distribution-free risk control for the FinEKG system.

The verifier's third duty. Beyond gating (inference) and rewarding (training),
the same evidence/rank nonconformity drives a *risk controller* with finite-
sample guarantees:

- stateless kernels (`functional`): the split-conformal quantile / prediction
  set used everywhere a single exchangeable threshold suffices;
- streaming calibrators (registry `conformal_calibrators`): ``split`` (static,
  exchangeability-only baseline), ``aci`` (adaptive, drift-robust),
  ``weighted`` (recency-weighted, non-exchangeable), ``crc`` (Conformal Risk
  Control, bounds a monotone loss such as the coverage-set miss rate);
- diagnostics (`metrics`): rolling coverage / drift gap over time, plus the
  selective-prediction kernels re-exported from `core.eval.faithfulness`.

Importing the package registers all calibrators, so config-driven selection
(`build_calibrator("aci", alpha=0.1)`) works without importing submodules.
"""

from __future__ import annotations

from finekg.core.calibration.aci import AdaptiveConformal
from finekg.core.calibration.base import (
    Calibrator,
    build_calibrator,
    conformal_calibrators,
)
from finekg.core.calibration.crc import RiskControlCalibrator, conformal_risk_threshold
from finekg.core.calibration.functional import (
    average_set_size,
    conformal_quantile,
    empirical_coverage,
    prediction_set,
)
from finekg.core.calibration.metrics import (
    accuracy_at_coverage,
    crc_empirical_risk,
    drift_coverage_gap,
    rolling_coverage,
    set_size_efficiency,
)
from finekg.core.calibration.propagation import (
    BudgetSplit,
    CrossStageResult,
    allocate_budget,
    allocate_budget_conditional,
    binomial_upper_confidence,
    compare_cross_stage_methods,
    run_cross_stage,
)
from finekg.core.calibration.split import SplitConformal
from finekg.core.calibration.weighted import WeightedConformal

__all__ = [
    # stateless kernels (back-compat surface)
    "conformal_quantile",
    "prediction_set",
    "empirical_coverage",
    "average_set_size",
    # streaming calibrators + registry
    "Calibrator",
    "conformal_calibrators",
    "build_calibrator",
    "SplitConformal",
    "AdaptiveConformal",
    "WeightedConformal",
    "RiskControlCalibrator",
    "conformal_risk_threshold",
    # cross-stage risk propagation (CS-CRP)
    "BudgetSplit",
    "allocate_budget",
    "allocate_budget_conditional",
    "binomial_upper_confidence",
    "CrossStageResult",
    "run_cross_stage",
    "compare_cross_stage_methods",
    # diagnostics
    "rolling_coverage",
    "drift_coverage_gap",
    "accuracy_at_coverage",
    "set_size_efficiency",
    "crc_empirical_risk",
]
