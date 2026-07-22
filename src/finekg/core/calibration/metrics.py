"""Drift-aware calibration diagnostics.

Complements the selective-prediction metrics in `core.eval.faithfulness`
(risk-coverage curve, AURC, ECE) with the *temporal* checks that motivate the
adaptive calibrators: does realised coverage track 1 - alpha over time, or does
it drift? Plus a selective-accuracy summary reused by the downstream trading
evaluation.

Pure-Python; runs on CPU. The headline experiment compares ``drift_coverage_gap``
for ``split`` (large gap under shift) vs ``aci`` / ``weighted`` (small gap).
"""

from __future__ import annotations

from collections.abc import Sequence

# Re-export the existing selective-prediction kernels so callers have one import
# surface for calibration diagnostics (no duplication of risk-coverage / AURC).
from finekg.core.eval.faithfulness import (
    RiskCoveragePoint,
    aurc,
    expected_calibration_error,
    risk_coverage_curve,
    selective_risk_at_coverage,
)

__all__ = [
    "rolling_coverage",
    "drift_coverage_gap",
    "accuracy_at_coverage",
    "set_size_efficiency",
    "crc_empirical_risk",
    # re-exported from core.eval.faithfulness
    "RiskCoveragePoint",
    "risk_coverage_curve",
    "selective_risk_at_coverage",
    "aurc",
    "expected_calibration_error",
]


def rolling_coverage(covered: Sequence[bool], window: int = 50) -> list[float]:
    """Coverage rate within each trailing window over a time-ordered stream.

    ``covered[t]`` is True iff query t's gold answer fell inside its admitted
    set. Returns one coverage value per step (the mean over the last ``window``
    outcomes), so a plot reveals whether coverage holds at 1 - alpha throughout
    or sags/spikes when the distribution shifts.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float] = []
    for t in range(len(covered)):
        lo = max(0, t - window + 1)
        chunk = covered[lo : t + 1]
        out.append(sum(1 for c in chunk if c) / len(chunk))
    return out


def drift_coverage_gap(
    covered: Sequence[bool], target: float, window: int = 50
) -> float:
    """Worst-case deviation of rolling coverage from the ``target`` (= 1 - alpha).

    The single number that separates a drift-robust calibrator (small gap) from
    static split conformal (large gap under regime shift).
    """
    roll = rolling_coverage(covered, window)
    return max((abs(c - target) for c in roll), default=0.0)


def accuracy_at_coverage(
    confidences: Sequence[float], correct: Sequence[bool], coverage: float
) -> float:
    """Accuracy among the most-confident ``coverage`` fraction of predictions.

    The selective-prediction summary used by the downstream trading evaluation:
    keep only the queries the system is most confident / most faithful about and
    report how often it is right there.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    n = len(confidences)
    if n == 0:
        return 0.0
    keep = max(1, int(coverage * n))
    order = sorted(zip(confidences, correct, strict=True), key=lambda p: p[0], reverse=True)
    kept = order[:keep]
    return sum(1 for _, ok in kept if ok) / len(kept)


def set_size_efficiency(sets: Sequence[Sequence[str]]) -> float:
    """Mean coverage-set size (lower is more efficient at equal coverage)."""
    return sum(len(s) for s in sets) / len(sets) if sets else 0.0


def crc_empirical_risk(losses: Sequence[float]) -> float:
    """Mean realised loss — the quantity Conformal Risk Control bounds."""
    return sum(losses) / len(losses) if losses else 0.0
