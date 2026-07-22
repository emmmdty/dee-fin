"""Split-conformal primitives: distribution-free prediction sets / abstention.

Given nonconformity scores on a held-out calibration set, `conformal_quantile`
returns a threshold q such that test prediction sets ``{y : nonconformity(x, y)
<= q}`` cover the truth with probability >= 1 - alpha (finite-sample, under
exchangeability). The forecasting stage feeds it a nonconformity derived from
*evidence faithfulness* / gold rank rather than raw confidence, so abstention is
driven by whether a forecast has a faithful evidence path — the system's
organizing principle.

These are the stateless kernels; the streaming/adaptive calibrators in
`split.py` / `aci.py` / `weighted.py` / `crc.py` build on them. Pure-Python and
dependency-free, so they run and are unit-tested on a CPU box.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "conformal_quantile",
    "prediction_set",
    "empirical_coverage",
    "average_set_size",
]


def conformal_quantile(nonconformity: Sequence[float], alpha: float) -> float:
    """Split-conformal threshold for target miscoverage ``alpha`` in (0, 1).

    Returns the ``ceil((n + 1)(1 - alpha))``-th smallest calibration score, the
    standard finite-sample quantile. Returns ``+inf`` when the calibration set
    is too small to guarantee 1 - alpha coverage, so the caller admits every
    candidate instead of abstaining without justification.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = sorted(nonconformity)
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return scores[k - 1]


def prediction_set(
    candidate_nonconformity: Mapping[str, float], threshold: float
) -> list[str]:
    """Candidates whose nonconformity is within the conformal ``threshold``."""
    return [c for c, s in candidate_nonconformity.items() if s <= threshold]


def empirical_coverage(sets: Sequence[Sequence[str]], gold: Sequence[str]) -> float:
    """Fraction of queries whose gold answer falls inside the prediction set."""
    if len(sets) != len(gold):
        raise ValueError("sets and gold must be the same length")
    if not sets:
        return 0.0
    covered = sum(1 for s, g in zip(sets, gold, strict=True) if g in s)
    return covered / len(sets)


def average_set_size(sets: Sequence[Sequence[str]]) -> float:
    """Mean prediction-set size — the efficiency half of the coverage trade-off."""
    return sum(len(s) for s in sets) / len(sets) if sets else 0.0
