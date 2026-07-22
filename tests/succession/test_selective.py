"""M3a: the selective conformal head over CGEP.

`selective.py` is a *bridge* -- it turns a predictor's candidate scores into the
gold ranks the `core.calibration` conformal machinery consumes, then reports a
risk-coverage curve. The conformal coverage math itself is locked in
`tests/core/test_propagation.py`; these tests pin the bridge: ranks map correctly,
an unscorable instance becomes an infinite (guaranteed-miss) rank, the strict
tie-break threads through, and the curve tightens as alpha rises. Every predictor
here is torch-free, so the head is exercised end-to-end on CPU exactly as SeDGPL
will be on GPU.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from finekg.succession.data.cgep import CgepInstance, CgepNode
from finekg.succession.metrics import sedgpl_rank
from finekg.succession.predictor import (
    RandomSuccessorPredictor,
    SuccessorPredictor,
    UnscorableInstance,
)
from finekg.succession.selective import cgep_gold_ranks, selective_report


def _instance(i: int, k: int, rng: random.Random) -> CgepInstance:
    """A scoreable instance with `k` distinct-trigger candidates and a random gold.

    Only candidates / label matter to a score()-based head, so nodes and edges are
    the minimum a `CgepInstance` needs to exist.
    """
    cands = tuple(
        CgepNode(node_id=f"c{i}_{j}", event_type="T", trigger=f"e{i}_{j}", sentence="")
        for j in range(k)
    )
    label = rng.randrange(k)
    anchor = CgepNode(node_id=f"n{i}", event_type="T", trigger=f"a{i}", sentence="")
    return CgepInstance(
        instance_id=f"x{i}", doc_id="d", nodes=(anchor, cands[label]),
        edges=((0, "cause", 1),), candidates=cands, label=label,
    )


def _corpus(n: int, k: int, seed: int = 0) -> list[CgepInstance]:
    rng = random.Random(seed)
    return [_instance(i, k, rng) for i in range(n)]


class _Unscorable(SuccessorPredictor):
    """Cannot score anything -- exercises the guaranteed-miss branch."""

    def fit(self, instances: Sequence[CgepInstance]) -> None:
        _ = instances

    def score(self, instance: CgepInstance) -> list[float]:
        raise UnscorableInstance(instance.instance_id)


def test_cgep_gold_ranks_match_the_direct_rank_and_are_all_reachable():
    insts = _corpus(6, k=8)
    predictor = RandomSuccessorPredictor(seed=1)
    ranks, reachable = cgep_gold_ranks(predictor, insts)
    assert reachable == [True] * 6  # gold ECG: every query is reachable
    for instance, rank in zip(insts, ranks, strict=True):
        assert rank == float(sedgpl_rank(predictor.score(instance), instance.label))


def test_unscorable_instances_become_infinite_rank_but_stay_reachable():
    insts = _corpus(3, k=8)
    ranks, reachable = cgep_gold_ranks(_Unscorable(), insts)
    assert ranks == [math.inf] * 3  # a guaranteed miss, never a flat optimistic win
    assert reachable == [True] * 3  # reachability is a construction-stage notion, not scoring


def test_strict_tie_break_charges_the_duplicate_trigger():
    gold = CgepNode(node_id="g", event_type="T", trigger="A", sentence="")
    dup = CgepNode(node_id="d", event_type="T", trigger="A", sentence="")  # ties gold
    other = CgepNode(node_id="b", event_type="T", trigger="B", sentence="")
    instance = CgepInstance(
        instance_id="x", doc_id="d", nodes=(gold, other),
        edges=((0, "cause", 1),), candidates=(gold, dup, other), label=0,
    )
    predictor = RandomSuccessorPredictor(seed=1)
    (optimistic,), _ = cgep_gold_ranks(predictor, [instance], strict=False)
    (strict,), _ = cgep_gold_ranks(predictor, [instance], strict=True)
    assert strict == optimistic + 1  # the tied duplicate is charged against gold


def test_selective_head_covers_at_the_target_level():
    # A random scorer gives ~uniform gold ranks; split conformal must still cover
    # at 1 - alpha (the formal guarantee that does not depend on accuracy).
    insts = _corpus(400, k=20)
    predictor = RandomSuccessorPredictor(seed=1)
    cal, test = insts[:200], insts[200:]
    report = selective_report(predictor, cal, test, alphas=(0.1,), reasoning="split")
    row = report["curve"][0]
    assert report["n_cal"] == 200 and report["n_test"] == 200
    assert row["target"] == 0.9
    assert row["coverage"] >= 0.85  # 1 - alpha minus finite-sample slack


def test_risk_coverage_curve_tightens_as_alpha_rises():
    insts = _corpus(400, k=20)
    predictor = RandomSuccessorPredictor(seed=1)
    cal, test = insts[:200], insts[200:]
    report = selective_report(predictor, cal, test, alphas=(0.05, 0.2, 0.4), reasoning="split")
    targets = [row["target"] for row in report["curve"]]
    sizes = [row["mean_set_size"] for row in report["curve"]]
    assert targets == [0.95, 0.8, 0.6]  # 1 - alpha, decreasing
    assert sizes[0] >= sizes[1] >= sizes[2]  # a higher risk budget buys tighter sets
