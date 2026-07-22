"""Cross-stage conformal risk propagation (CS-CRP) tests.

Synthetic drift: a tight reasoning regime (small gold ranks) shifts mid-stream to
a drifted regime (large ranks), with the calibration pool drawn only from the
tight regime. This is the setting where static (exchangeability-only) conformal
under-covers and the drift-adaptive composition must hold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from finekg.core.calibration.propagation import (
    BudgetSplit,
    allocate_budget,
    allocate_budget_conditional,
    binomial_upper_confidence,
    compare_cross_stage_methods,
    run_cross_stage,
)


def _drifting_stream(n: int = 3000, alpha_edge: float = 0.05, seed: int = 13):
    """Reachable flags (CRC admission FNR = alpha_edge) + drifting gold ranks."""
    rng = np.random.default_rng(seed)
    reachable = (rng.random(n) > alpha_edge).tolist()
    ranks = np.empty(n)
    cut = n // 3
    ranks[:cut] = rng.integers(1, 5, size=cut)  # regime A: tight
    ranks[cut:] = rng.integers(6, 25, size=n - cut)  # regime B: drifted (larger ranks)
    gold_ranks = [float(r) if reachable[t] else math.inf for t, r in enumerate(ranks)]
    cal_ranks = rng.integers(1, 5, size=400).astype(float).tolist()  # regime A only
    return reachable, gold_ranks, cal_ranks


def test_allocate_budget_splits_and_validates():
    s = allocate_budget(0.1, edge_share=0.5)
    assert isinstance(s, BudgetSplit)
    assert s.alpha_edge == pytest.approx(0.05)
    assert s.alpha_pred == pytest.approx(0.05)
    assert s.alpha_edge + s.alpha_pred == pytest.approx(s.alpha_total)
    # naive allocation = nothing reserved for reachability loss
    assert allocate_budget(0.1, edge_share=0.0).alpha_pred == pytest.approx(0.1)
    with pytest.raises(ValueError):
        allocate_budget(1.5)
    with pytest.raises(ValueError):
        allocate_budget(0.1, edge_share=2.0)


def test_cs_crp_holds_coverage_under_drift_when_others_fail():
    reachable, gold_ranks, cal_ranks = _drifting_stream()
    res = compare_cross_stage_methods(
        reachable, gold_ranks, cal_ranks, alpha_total=0.1, edge_share=0.5
    )
    naive, bonf, cs = res["naive"], res["bonferroni"], res["cs_crp"]
    target = 0.9

    # naive: static + no reachability budget -> under-covers under drift
    assert naive.composed_coverage < target
    # CS-CRP: budget split + drift-adaptive reasoning -> composed coverage holds
    assert cs.composed_coverage >= target - 0.05
    # CS-CRP is the most drift-robust (smallest composed drift gap)
    assert cs.composed_drift_gap < bonf.composed_drift_gap
    assert cs.composed_drift_gap < naive.composed_drift_gap
    # ... and covers strictly better than the naive strawman
    assert cs.composed_coverage > naive.composed_coverage
    # reachability is bounded by the construction FNR budget (~1 - alpha_edge)
    assert 0.9 <= cs.reachable_rate <= 1.0


def test_run_cross_stage_length_mismatch_raises():
    with pytest.raises(ValueError):
        run_cross_stage([True, False], [1.0], [1.0], alpha_total=0.1, alpha_pred=0.05)


def _low_loss_stream(n: int = 3000, unreach: float = 0.01, seed: int = 17):
    """Admission barely drops gold (reach ~= 0.99) while the fixed split still
    reserves half the budget — the regime where the union bound is wasteful and
    conditional reallocation should tighten sets. Continuous heavy-tailed ranks
    so the calibration quantile actually responds to the effective level."""
    rng = np.random.default_rng(seed)
    reachable = (rng.random(n) > unreach).tolist()
    cut = n // 3
    ranks = np.empty(n)
    ranks[:cut] = rng.exponential(3.0, size=cut) + 1.0  # regime A: tight
    ranks[cut:] = rng.exponential(9.0, size=n - cut) + 1.0  # regime B: drifted
    gold_ranks = [float(r) if reachable[t] else math.inf for t, r in enumerate(ranks)]
    cal_ranks = (rng.exponential(3.0, size=400) + 1.0).tolist()  # regime A only
    cal_reachable = (rng.random(400) > unreach).tolist()
    return reachable, gold_ranks, cal_ranks, cal_reachable


def test_binomial_upper_confidence_closed_form_and_monotone():
    n, delta = 100, 0.05
    # zero failures: P(Bin(n, p) = 0) = (1 - p)^n = delta has a closed form
    assert binomial_upper_confidence(0, n, delta) == pytest.approx(
        1.0 - delta ** (1.0 / n), abs=1e-9
    )
    # all failures: nothing rules out p = 1
    assert binomial_upper_confidence(n, n, delta) == 1.0
    # monotone in the failure count, always a valid rate
    bounds = [binomial_upper_confidence(k, n, delta) for k in (0, 1, 5, 20, 50)]
    assert all(0.0 < b <= 1.0 for b in bounds)
    assert bounds == sorted(bounds)
    # the bound is above the empirical rate (it is an upper bound)
    assert binomial_upper_confidence(5, n, delta) > 0.05


def test_allocate_budget_conditional_recycles_unused_edge_budget():
    rng = np.random.default_rng(7)
    cal_reachable = (rng.random(2000) > 0.01).tolist()  # admission loses ~1%
    fixed = allocate_budget(0.1, edge_share=0.5)
    cond = allocate_budget_conditional(cal_reachable, 0.1, alpha_edge=fixed.alpha_edge)
    # certified unreachability is far below the reserved 0.05 -> budget recycled
    assert cond.alpha_edge <= fixed.alpha_edge
    assert cond.alpha_pred > fixed.alpha_pred
    assert cond.alpha_pred < cond.alpha_total
    # lossless admission (all reachable), large n: alpha_pred approaches the
    # naive allocation because nothing needs reserving
    lossless = allocate_budget_conditional([True] * 2000, 0.1)
    assert lossless.alpha_pred > 0.9 * 0.1


def test_allocate_budget_conditional_validates_and_clamps_infeasible():
    with pytest.raises(ValueError):
        allocate_budget_conditional([], 0.1)
    with pytest.raises(ValueError):
        allocate_budget_conditional([True], 1.5)
    with pytest.raises(ValueError):
        allocate_budget_conditional([True], 0.1, delta=0.0)
    # admission so lossy that the certified loss exceeds the whole budget:
    # the reasoning side degrades to (near) admit-everything rather than crash
    infeasible = allocate_budget_conditional([False] * 200 + [True] * 200, 0.1)
    assert infeasible.alpha_pred == pytest.approx(1e-4)


def test_cs_crp_cond_tightens_sets_and_keeps_composed_coverage():
    reachable, gold_ranks, cal_ranks, cal_reachable = _low_loss_stream()
    res = compare_cross_stage_methods(
        reachable,
        gold_ranks,
        cal_ranks,
        alpha_total=0.1,
        edge_share=0.5,
        cal_reachable=cal_reachable,
    )
    assert set(res) == {"naive", "bonferroni", "cs_crp", "cs_crp_cond"}
    cs, cond = res["cs_crp"], res["cs_crp_cond"]
    target = 0.9
    # the conditional correction keeps the composed guarantee ...
    assert cond.composed_coverage >= target - 0.05
    # ... while spending the recycled budget on strictly tighter sets
    assert cond.mean_set_size < cs.mean_set_size
    # without calibration reachability the comparison is unchanged (back-compat)
    legacy = compare_cross_stage_methods(
        reachable, gold_ranks, cal_ranks, alpha_total=0.1, edge_share=0.5
    )
    assert set(legacy) == {"naive", "bonferroni", "cs_crp"}
