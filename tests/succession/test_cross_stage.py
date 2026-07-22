"""M3b controlled cross-stage sweep: the CS-CRP composition over reachability loss.

The coverage math is locked in `tests/core/test_propagation.py`; these pin the
*harness* -- reachability is induced at a target loss, threaded into
`compare_cross_stage_methods`, and the sweep shows naive conformal falling below
the drift-adaptive conditional composition once reachability loss appears. The
reasoning ranks are a stand-in here; on the real run they are SeDGPL's.
"""

from __future__ import annotations

import random

from finekg.succession.cross_stage import cross_stage_sweep, induce_reachability


def test_induce_reachability_random_hits_the_target_loss():
    reach = induce_reachability(100, 0.3, random.Random(0))
    assert sum(not r for r in reach) == 30  # 30% unreachable


def test_induce_reachability_hardest_drops_the_worst_ranked_queries():
    ranks = [float(i) for i in range(10)]  # 9 is the hardest to rank
    reach = induce_reachability(10, 0.3, random.Random(0), ranks=ranks, mode="hardest")
    assert [i for i, r in enumerate(reach) if not r] == [7, 8, 9]  # the 3 largest ranks


def _ranks(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [float(rng.randint(0, 5)) for _ in range(n)]


def test_sweep_row_structure_and_reachable_rate():
    out = cross_stage_sweep(_ranks(400, 1), _ranks(400, 2), losses=(0.2,), alpha_total=0.2)
    row = out["curve"][0]
    assert row["reachable_rate"] == 0.8 and row["target"] == 0.8
    for method in ("naive", "bonferroni", "cs_crp", "cs_crp_cond"):
        assert f"{method}_coverage" in row and f"{method}_set_size" in row


def test_naive_falls_below_conditional_cs_crp_once_loss_appears():
    cal, test = _ranks(800, 1), _ranks(800, 2)
    out = cross_stage_sweep(cal, test, losses=(0.0, 0.15), alpha_total=0.2, seed=0)
    by_loss = {row["loss"]: row for row in out["curve"]}
    # No reachability loss: naive holds near target.
    assert by_loss[0.0]["naive_coverage"] >= by_loss[0.0]["target"] - 0.05
    # With loss: the conditional CS-CRP reserves reachability budget and covers
    # better than naive, which spent the whole budget on reasoning.
    hi = by_loss[0.15]
    assert hi["naive_coverage"] < hi["cs_crp_cond_coverage"]
    assert hi["cs_crp_cond_coverage"] >= hi["target"] - 0.08
