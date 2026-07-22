"""Intervention-based evidence faithfulness — the system's verification currency.

An edge or a forecast is *faithful* only if its cited evidence actually drives
the prediction: ablate the evidence and the model's score should drop.
`intervention_faithfulness` turns a (base, ablated) score pair into a [0, 1]
faithfulness value; the curve / summary functions turn per-item faithfulness into
selective-prediction diagnostics (risk-coverage, AURC) and a calibration check
(ECE).

This module is deliberately model-agnostic: it consumes scores, not models, so
the actual ablation (mask an evidence span, drop supporting facts) lives in the
relation / forecasting stages while the metrics here stay pure and are exercised
on CPU with the heuristic / frequency baselines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "intervention_faithfulness",
    "RiskCoveragePoint",
    "risk_coverage_curve",
    "selective_risk_at_coverage",
    "aurc",
    "expected_calibration_error",
]


def intervention_faithfulness(base_score: float, ablated_score: float) -> float:
    """Relative drop in a prediction's score when its supporting evidence is
    removed.

    0 = ablation changes nothing (the evidence was decorative); 1 = ablation
    destroys the prediction (it was fully evidence-driven). Clipped to [0, 1],
    so an ablation that *raises* the score reads as unfaithful (0), not negative.
    """
    if base_score <= 0.0:
        return 0.0
    drop = (base_score - ablated_score) / base_score
    return max(0.0, min(1.0, drop))


@dataclass
class RiskCoveragePoint:
    coverage: float
    risk: float
    threshold: float


def risk_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool]
) -> list[RiskCoveragePoint]:
    """Selective-prediction curve driven by ``scores`` (e.g. faithfulness).

    Keep the highest-scoring predictions first; at each prefix report coverage
    (fraction kept) and risk (error rate among kept). A score that ranks correct
    predictions above wrong ones yields low risk at low coverage — exactly what
    a useful abstention signal should do.
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must be the same length")
    pairs = sorted(zip(scores, correct, strict=True), key=lambda p: p[0], reverse=True)
    points: list[RiskCoveragePoint] = []
    errors = 0
    for i, (score, ok) in enumerate(pairs, start=1):
        errors += 0 if ok else 1
        points.append(
            RiskCoveragePoint(coverage=i / len(pairs), risk=errors / i, threshold=score)
        )
    return points


def selective_risk_at_coverage(
    scores: Sequence[float], correct: Sequence[bool], coverage: float
) -> float:
    """Risk at the largest achievable coverage <= ``coverage``."""
    eligible = [p for p in risk_coverage_curve(scores, correct) if p.coverage <= coverage]
    return eligible[-1].risk if eligible else 0.0


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better): a single summary of
    how well ``scores`` supports selective prediction."""
    curve = risk_coverage_curve(scores, correct)
    return sum(p.risk for p in curve) / len(curve) if curve else 0.0


def expected_calibration_error(
    probs: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """ECE: weighted average |confidence - accuracy| over equal-width bins.

    Checks that a faithfulness / confidence score means what it claims — a
    prerequisite for trusting it as an abstention signal.
    """
    if len(probs) != len(correct):
        raise ValueError("probs and correct must be the same length")
    if not probs:
        return 0.0
    n = len(probs)
    conf_sum = [0.0] * n_bins
    acc_sum = [0.0] * n_bins
    count = [0] * n_bins
    for p, ok in zip(probs, correct, strict=True):
        b = min(n_bins - 1, max(0, int(p * n_bins)))
        conf_sum[b] += p
        acc_sum[b] += 1.0 if ok else 0.0
        count[b] += 1
    ece = 0.0
    for b in range(n_bins):
        if count[b] == 0:
            continue
        conf = conf_sum[b] / count[b]
        acc = acc_sum[b] / count[b]
        ece += (count[b] / n) * abs(conf - acc)
    return ece
