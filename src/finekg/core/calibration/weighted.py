"""Recency-weighted (non-exchangeable) conformal calibration.

When exchangeability fails because of drift, weight calibration scores by
recency so post-shift observations dominate the quantile (Tibshirani et al.,
2019; Barber et al., 2023, nonexchangeable conformal). The threshold is the
weighted ``(1 - alpha)`` quantile of the realised nonconformity stream, which is
seeded from the calibration pool and extended online via ``observe``.

Weights decay geometrically with age (half-life ``halflife`` steps), so the
threshold tracks the recent regime without the parameter-update dynamics of ACI
— a complementary drift-robust baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from finekg.core.calibration.base import Calibrator, conformal_calibrators

__all__ = ["WeightedConformal"]


@conformal_calibrators.register("weighted")
class WeightedConformal(Calibrator):
    """Threshold = recency-weighted (1 - alpha) quantile of realised scores."""

    def __init__(self, alpha: float = 0.1, halflife: float = 50.0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if halflife <= 0.0:
            raise ValueError("halflife must be positive")
        self.alpha = alpha
        self.halflife = halflife
        self._scores: list[float] = []  # realised nonconformity, oldest first

    def fit(self, cal_scores: Sequence[float]) -> WeightedConformal:
        self._scores = [float(s) for s in cal_scores]
        return self

    def threshold(self) -> float:
        n = len(self._scores)
        if n == 0:
            return math.inf
        # Most-recent score (index n-1) has weight 1; older ones decay by age.
        weights = [0.5 ** ((n - 1 - i) / self.halflife) for i in range(n)]
        order = sorted(range(n), key=lambda i: self._scores[i])
        target = (1.0 - self.alpha) * sum(weights)
        cum = 0.0
        for i in order:
            cum += weights[i]
            if cum >= target:
                return self._scores[i]
        return self._scores[order[-1]]

    def observe(self, miscovered: bool, score: float | None = None) -> None:
        if score is not None:
            self._scores.append(float(score))
