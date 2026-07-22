"""SeDGPL's optimistic tie-break is load-bearing, not cosmetic."""

from __future__ import annotations

import pytest

from finekg.succession.metrics import cgep_metrics, sedgpl_rank, strict_rank


def test_rank_counts_strictly_better_candidates():
    scores = [0.9, 0.5, 0.7]
    assert sedgpl_rank(scores, label=2) == 1  # only 0.9 beats gold's 0.7
    assert sedgpl_rank(scores, label=0) == 0
    assert sedgpl_rank(scores, label=1) == 2


def test_ties_are_resolved_in_golds_favour_the_way_sedgpl_does():
    # `predtCandi.sort(reverse=True); predtCandi.index(labelScore)` returns the
    # first slot with that score, so every tied candidate is ranked behind gold.
    scores = [0.7, 0.7, 0.7, 0.9]
    assert sedgpl_rank(scores, label=0) == 1
    assert sedgpl_rank(scores, label=2) == 1  # position among the ties is irrelevant


def test_strict_rank_states_what_the_optimistic_count_hides():
    scores = [0.7, 0.7, 0.7, 0.9]
    assert strict_rank(scores, label=0) == 3  # the two other 0.7s now outrank gold
    assert sedgpl_rank(scores, label=0) == 1


def test_the_two_ranks_agree_when_nothing_ties():
    scores = [0.1, 0.4, 0.9, 0.2]
    for label in range(len(scores)):
        assert sedgpl_rank(scores, label) == strict_rank(scores, label)


def test_duplicate_triggers_inflate_the_metric_exactly_as_in_the_release():
    # Two candidates share a mention, so they share a token id and a score. Gold
    # is one of them. Optimistic ranking hands gold the better slot for free.
    scores = [0.8, 0.8, 0.9, 0.3]
    assert cgep_metrics([sedgpl_rank(scores, 0)])["mrr"] == pytest.approx(0.5)
    assert cgep_metrics([strict_rank(scores, 0)])["mrr"] == pytest.approx(1 / 3)


def test_metrics_average_reciprocal_rank_and_hits():
    ranks = [0, 2, 9]
    metrics = cgep_metrics(ranks, hits_at=(1, 3, 10))
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 10) / 3)
    assert metrics["hits@1"] == pytest.approx(1 / 3)
    assert metrics["hits@3"] == pytest.approx(2 / 3)
    assert metrics["hits@10"] == pytest.approx(1.0)
    assert metrics["n"] == 3.0


def test_metrics_on_no_predictions_are_zero_not_nan():
    metrics = cgep_metrics([])
    assert metrics["mrr"] == 0.0
    assert metrics["n"] == 0.0
    assert all(v == 0.0 for k, v in metrics.items() if k.startswith("hits@"))
