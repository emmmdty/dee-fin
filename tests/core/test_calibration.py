"""Conformal calibration must give the textbook finite-sample threshold and the
distribution-free coverage guarantee — checked deterministically on CPU.
"""

from __future__ import annotations

import math

import pytest

from finekg.core.calibration import (
    average_set_size,
    conformal_quantile,
    empirical_coverage,
    prediction_set,
)


def test_conformal_quantile_is_the_finite_sample_order_statistic() -> None:
    scores = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0, n = 10
    # k = ceil(11 * 0.8) = 9 -> 9th smallest = 0.9
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(0.9)


def test_conformal_quantile_infinite_when_sample_too_small() -> None:
    assert conformal_quantile([0.1, 0.2, 0.3], alpha=0.1) == math.inf  # k = 4 > n = 3
    assert conformal_quantile([], alpha=0.1) == math.inf


def test_conformal_quantile_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        conformal_quantile([0.1], alpha=0.0)


def test_prediction_set_filters_by_threshold() -> None:
    assert prediction_set({"a": 0.1, "b": 0.5, "c": 0.9}, 0.5) == ["a", "b"]


def test_empirical_coverage_and_set_size() -> None:
    sets = [["a"], ["b", "c"]]
    assert empirical_coverage(sets, ["a", "c"]) == 1.0
    assert empirical_coverage(sets, ["a", "z"]) == 0.5
    assert average_set_size(sets) == pytest.approx(1.5)


def test_coverage_guarantee_holds_on_exchangeable_scores() -> None:
    # Calibration and test nonconformity from the same fixed set: empirical
    # coverage must meet the 1 - alpha target.
    alpha = 0.2
    cal = [i / 100 for i in range(100)]
    q = conformal_quantile(cal, alpha)
    test_scores = [i / 100 for i in range(100)]
    sets = [["y"] if s <= q else [] for s in test_scores]
    gold = ["y"] * len(test_scores)
    assert empirical_coverage(sets, gold) >= 1 - alpha
