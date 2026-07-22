"""The streaming calibrators are the verifier's third duty (risk control), and
their reason to exist is distribution shift: static split conformal loses its
coverage under drift, while ACI (level adaptation) and weighted conformal
(recency re-quantiling) keep it. These checks build a deterministic drifting
nonconformity stream on CPU and assert exactly that separation, plus the
Conformal-Risk-Control selection math.
"""

from __future__ import annotations

import math

import pytest

from finekg.core.calibration import (
    AdaptiveConformal,
    RiskControlCalibrator,
    SplitConformal,
    WeightedConformal,
    accuracy_at_coverage,
    build_calibrator,
    conformal_calibrators,
    conformal_quantile,
    conformal_risk_threshold,
    drift_coverage_gap,
    rolling_coverage,
)

ALPHA = 0.1
TARGET = 1.0 - ALPHA


def _drift_stream() -> tuple[list[float], list[float]]:
    """Calibration scores from an early regime, then a time-ordered test stream
    that shifts up by +0.8 halfway through (the model's nonconformity grows when
    the regime changes — e.g. a market regime shift)."""
    cal = [i / 100 for i in range(100)]  # early regime: U[0, 1), q@0.1 -> 0.90
    regime_a = [(i * 7 % 100) / 100 for i in range(100)]  # same support as cal
    regime_b = [0.8 + (i * 7 % 100) / 100 for i in range(100)]  # shifted up by 0.8
    return cal, regime_a + regime_b


def _stream_coverage(calibrator, cal: list[float], test: list[float]) -> list[bool]:
    """Run the calibrator over the stream, returning per-step coverage."""
    calibrator.fit(cal)
    covered: list[bool] = []
    for score in test:
        q = calibrator.threshold()
        is_covered = score <= q
        covered.append(is_covered)
        calibrator.observe(miscovered=not is_covered, score=score)
    return covered


def test_registry_exposes_the_four_calibrators() -> None:
    assert set(conformal_calibrators.available()) == {"split", "aci", "weighted", "crc"}
    assert isinstance(build_calibrator("aci", alpha=0.1), AdaptiveConformal)


def test_split_reproduces_the_static_quantile() -> None:
    cal = [i / 10 for i in range(1, 11)]
    cal_obj = SplitConformal(alpha=0.2).fit(cal)
    assert cal_obj.threshold() == pytest.approx(conformal_quantile(cal, 0.2))


def test_split_coverage_collapses_under_drift() -> None:
    cal, test = _drift_stream()
    covered = _stream_coverage(SplitConformal(alpha=ALPHA), cal, test)
    first_half = sum(covered[:100]) / 100
    second_half = sum(covered[100:]) / 100
    # Exchangeable first half meets the target; the shifted half does not.
    assert first_half >= TARGET - 0.02
    assert second_half < 0.2


def test_aci_maintains_coverage_and_beats_split_drift_gap() -> None:
    cal, test = _drift_stream()
    aci = AdaptiveConformal(alpha=ALPHA, gamma=0.1)
    aci_cov = _stream_coverage(aci, cal, test)
    split_cov = _stream_coverage(SplitConformal(alpha=ALPHA), cal, test)

    # Long-run coverage holds (ACI errs toward covering when the pool cannot
    # represent the shifted scale) ...
    assert sum(aci_cov) / len(aci_cov) >= TARGET - 0.05
    # ... and the rolling-coverage drift is far smaller than static split's.
    aci_gap = drift_coverage_gap(aci_cov, TARGET, window=50)
    split_gap = drift_coverage_gap(split_cov, TARGET, window=50)
    assert aci_gap < split_gap
    assert aci_gap < 0.3
    # The level adapted downward under the sustained second-half miscoverage.
    assert aci.alpha_t < ALPHA


def test_weighted_tracks_the_target_in_the_shifted_regime() -> None:
    cal, test = _drift_stream()
    weighted = WeightedConformal(alpha=ALPHA, halflife=20.0)
    cov = _stream_coverage(weighted, cal, test)
    split_cov = _stream_coverage(SplitConformal(alpha=ALPHA), cal, test)
    # Re-quantiling on recent realised scores recovers coverage in regime B,
    # where static split has all but collapsed.
    weighted_b = sum(cov[150:]) / 50
    split_b = sum(split_cov[150:]) / 50
    assert weighted_b > split_b
    assert abs(weighted_b - TARGET) <= 0.2


def test_conformal_risk_threshold_picks_the_tightest_lambda_meeting_the_bound() -> None:
    # bound(lambda) = (n/(n+1)) * risk + B/(n+1); n=100, B=1, alpha=0.15
    # lambda=2 risk 0.30 -> 0.307 (fail); lambda=3 risk 0.10 -> 0.109 (pass)
    chosen = conformal_risk_threshold(
        lambdas=[1, 2, 3, 4], mean_losses=[0.5, 0.3, 0.1, 0.0], n=100, alpha=0.15
    )
    assert chosen == 3


def test_risk_control_calibrator_bounds_the_miss_rate() -> None:
    cal_ranks = [1, 2, 3, 4, 5] * 19 + [6, 7, 8, 9, 10]  # n=100, 5 tail ranks > 5
    crc = RiskControlCalibrator(alpha=0.1).fit(cal_ranks)
    k = crc.threshold()
    assert k == 5
    empirical_miss = sum(1 for r in cal_ranks if r > k) / len(cal_ranks)
    assert empirical_miss <= 0.1


def test_rolling_coverage_and_accuracy_at_coverage() -> None:
    assert rolling_coverage([True, True, True], window=2) == [1.0, 1.0, 1.0]
    assert rolling_coverage([True, False], window=2)[-1] == pytest.approx(0.5)
    # Confidence that ranks correct predictions first -> high selective accuracy.
    conf = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    assert accuracy_at_coverage(conf, correct, coverage=0.5) == pytest.approx(1.0)
    assert accuracy_at_coverage(conf, correct, coverage=1.0) == pytest.approx(0.5)


def test_empty_and_degenerate_inputs_are_safe() -> None:
    assert SplitConformal(alpha=0.1).fit([]).threshold() == math.inf
    assert WeightedConformal(alpha=0.1).fit([]).threshold() == math.inf
    assert RiskControlCalibrator(alpha=0.1).fit([]).threshold() == math.inf
