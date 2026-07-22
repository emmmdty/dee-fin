"""Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021).

Online calibration for the streaming-extrapolation setting (TKG forecasting):
the gold answer of a query is revealed once its timestamp passes, so we can feed
realised (mis)coverage back and re-estimate the effective miscoverage level.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where ``err_t = 1`` iff query t was miscovered. Miscoverage shrinks ``alpha_t``
(-> a larger set next step), coverage grows it (-> a tighter set). The realised
miscoverage rate then satisfies ``|(1/T) sum err_t - alpha| <= |alpha_1 -
alpha_{T+1}| / (gamma * T)`` *irrespective of the data-generating process* — so
long-run coverage holds under arbitrary distribution shift, unlike static split
conformal.

Implementation note: the state ``alpha_t`` is kept **unclipped** (it may stray
slightly outside [0, 1]); only the *emitted* threshold saturates — ``alpha_t <=
0`` emits ``+inf`` (admit all) and ``alpha_t >= 1`` emits ``-inf`` (admit none).
Clipping the state instead would break the telescoping identity above and void
the guarantee under a support shift the calibration pool cannot represent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from finekg.core.calibration.base import Calibrator, conformal_calibrators
from finekg.core.calibration.functional import conformal_quantile

__all__ = ["AdaptiveConformal"]


@conformal_calibrators.register("aci")
class AdaptiveConformal(Calibrator):
    """ACI: re-estimate the effective miscoverage level from realised coverage.

    ``gamma`` is the adaptation step size — larger reacts faster to shift but
    tracks more noisily; 0.01–0.1 is the usual range. The threshold is the
    ``(1 - alpha_t)`` quantile of the fixed calibration pool, with the two
    saturated regimes handled explicitly (``alpha_t <= 0`` admits all,
    ``alpha_t >= 1`` admits none).
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self._cal: list[float] = []

    def fit(self, cal_scores: Sequence[float]) -> AdaptiveConformal:
        self._cal = sorted(cal_scores)
        self.alpha_t = self.alpha
        return self

    def threshold(self) -> float:
        if self.alpha_t <= 0.0:
            return math.inf  # admit every candidate
        if self.alpha_t >= 1.0:
            return -math.inf  # admit none
        return conformal_quantile(self._cal, self.alpha_t)

    def observe(self, miscovered: bool, score: float | None = None) -> None:
        # The state is intentionally NOT clipped to [0, 1]; clipping it would
        # break the telescoping coverage identity. Only `threshold` saturates.
        err = 1.0 if miscovered else 0.0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
