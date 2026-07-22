"""Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster, 2022).

Generalises split-conformal coverage to bounding the expectation of any
*monotone* loss. Given calibration losses ``L_i(lambda)`` that are
non-increasing as the control parameter ``lambda`` loosens the decision (admit
more / predict a larger set), pick the tightest lambda whose CRC-bounded
empirical risk meets the target:

    lambda_hat = inf { lambda : (n / (n + 1)) * R_hat_n(lambda)
                                 + B / (n + 1)  <=  alpha }

then ``E[L_test(lambda_hat)] <= alpha`` (finite-sample, distribution-free under
exchangeability). ``B`` is the loss upper bound (1 for a 0/1 miss loss).

Two uses in this project: bounding the **miss rate** of a forecasting coverage
set (the ``crc`` calibrator below) and bounding the **false-edge / miss rate**
of risk-controlled graph construction (see `relations.admission`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from finekg.core.calibration.base import Calibrator, conformal_calibrators

__all__ = ["conformal_risk_threshold", "RiskControlCalibrator"]


def conformal_risk_threshold(
    lambdas: Sequence[float],
    mean_losses: Sequence[float],
    n: int,
    alpha: float,
    loss_bound: float = 1.0,
) -> float:
    """Tightest ``lambda`` meeting the CRC bound, scanning tight -> loose.

    ``lambdas`` must be ordered so the decision loosens (mean loss is
    non-increasing) as the index grows; ``mean_losses[i]`` is ``R_hat_n`` at
    ``lambdas[i]`` over ``n`` calibration points. Returns the first lambda whose
    bound ``(n/(n+1)) R + B/(n+1) <= alpha`` holds, else the loosest lambda
    (and ``+inf`` if there is nothing to choose from).
    """
    if len(lambdas) != len(mean_losses):
        raise ValueError("lambdas and mean_losses must be the same length")
    if not lambdas:
        return math.inf
    if n <= 0:
        return float(lambdas[-1])
    for lam, risk in zip(lambdas, mean_losses, strict=True):
        if (n / (n + 1)) * risk + loss_bound / (n + 1) <= alpha:
            return float(lam)
    return float(lambdas[-1])


@conformal_calibrators.register("crc")
class RiskControlCalibrator(Calibrator):
    """Bound the forecasting coverage set's miss rate ``P(gold not in set)``.

    The control parameter is the set size ``k``; the loss is ``1[gold_rank > k]``
    (monotone non-increasing in ``k``). Picks the smallest ``k`` whose CRC bound
    on the calibration miss rate is ``<= alpha`` — a tighter, guarantee-carrying
    alternative to choosing ``k`` by the bare empirical quantile.
    """

    def __init__(self, alpha: float = 0.1, max_k: int | None = None) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if max_k is not None and max_k <= 0:
            raise ValueError("max_k must be positive when set")
        self.alpha = alpha
        self.max_k = max_k
        self._k: float = math.inf

    def fit(self, cal_scores: Sequence[float]) -> RiskControlCalibrator:
        scores = [float(s) for s in cal_scores]
        n = len(scores)
        if n == 0:
            self._k = math.inf
            return self
        kmax = self.max_k or int(math.ceil(max(scores)))
        ks = list(range(1, kmax + 1))
        mean_losses = [sum(1.0 for s in scores if s > k) / n for k in ks]
        self._k = conformal_risk_threshold(ks, mean_losses, n, self.alpha, 1.0)
        return self

    def threshold(self) -> float:
        return self._k
