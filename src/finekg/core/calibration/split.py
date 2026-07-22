"""Static split-conformal calibrator: one fixed threshold, exchangeability-only.

The drift-robust baseline to beat. Valid 1 - alpha coverage when calibration and
test scores are exchangeable; under temporal/regime drift its realised coverage
wanders away from the target (which is exactly what `aci` / `weighted` fix).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from finekg.core.calibration.base import Calibrator, conformal_calibrators
from finekg.core.calibration.functional import conformal_quantile

__all__ = ["SplitConformal"]


@conformal_calibrators.register("split")
class SplitConformal(Calibrator):
    """Textbook split conformal: threshold = (1 - alpha) quantile of cal scores."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._q = math.inf

    def fit(self, cal_scores: Sequence[float]) -> SplitConformal:
        self._q = conformal_quantile(cal_scores, self.alpha)
        return self

    def threshold(self) -> float:
        return self._q
