"""Intervention faithfulness and its selective-prediction diagnostics, pinned
down with hand-built scores so the semantics are unambiguous on CPU.
"""

from __future__ import annotations

import pytest

from finekg.core.eval.faithfulness import (
    aurc,
    expected_calibration_error,
    intervention_faithfulness,
    risk_coverage_curve,
    selective_risk_at_coverage,
)


def test_intervention_faithfulness_bounds() -> None:
    assert intervention_faithfulness(1.0, 0.0) == 1.0  # evidence drives the prediction
    assert intervention_faithfulness(1.0, 1.0) == 0.0  # decorative evidence
    assert intervention_faithfulness(1.0, 0.5) == pytest.approx(0.5)
    assert intervention_faithfulness(0.0, 0.0) == 0.0  # no base signal
    assert intervention_faithfulness(1.0, 2.0) == 0.0  # ablation raised score -> clipped


def test_risk_coverage_curve_rewards_good_ranking() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    curve = risk_coverage_curve(scores, correct)
    assert curve[0].coverage == pytest.approx(0.25)
    assert curve[0].risk == 0.0  # most faithful kept -> no error
    assert curve[-1].coverage == 1.0
    assert curve[-1].risk == pytest.approx(0.5)


def test_selective_risk_at_coverage() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    assert selective_risk_at_coverage(scores, correct, 0.5) == 0.0
    assert selective_risk_at_coverage(scores, correct, 1.0) == pytest.approx(0.5)


def test_aurc_lower_for_separating_scores() -> None:
    correct = [True, True, False, False]
    good = aurc([0.9, 0.8, 0.2, 0.1], correct)  # faithfulness separates right from wrong
    bad = aurc([0.1, 0.2, 0.8, 0.9], correct)  # anti-correlated
    assert good < bad


def test_ece_zero_when_perfectly_calibrated() -> None:
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([1.0, 1.0], [True, False]) == pytest.approx(0.5)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve([0.1], [True, False])
