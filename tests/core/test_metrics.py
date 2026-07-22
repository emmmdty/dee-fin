"""Exact-value tests for the shared metrics (relation / coref / mrr)."""

from __future__ import annotations

import pytest

from finekg.core.eval import conll_coref_f1, mrr_hits, relation_prf
from finekg.core.schema import RelationEdge, RelationType


def _temporal(h: str, t: str) -> RelationEdge:
    return RelationEdge(head_id=h, tail_id=t, relation_type=RelationType.TEMPORAL, subtype="BEFORE")


def _overlap(h: str, t: str) -> RelationEdge:
    return RelationEdge(
        head_id=h,
        tail_id=t,
        relation_type=RelationType.TEMPORAL,
        subtype="OVERLAP",
    )


def _causal(h: str, t: str) -> RelationEdge:
    return RelationEdge(head_id=h, tail_id=t, relation_type=RelationType.CAUSAL, subtype="CAUSE")


def test_relation_prf_perfect_match() -> None:
    gold = [_temporal("a", "b"), _temporal("b", "c")]
    pred = [_temporal("a", "b"), _temporal("b", "c")]
    scores = relation_prf(pred, gold)
    assert scores["micro"]["f1"] == pytest.approx(1.0)
    assert scores["temporal"]["precision"] == pytest.approx(1.0)


def test_relation_prf_partial() -> None:
    gold = [_temporal("a", "b"), _temporal("b", "c")]
    pred = [_temporal("a", "b"), _temporal("a", "c")]  # one correct, one wrong
    micro = relation_prf(pred, gold)["micro"]
    assert micro["precision"] == pytest.approx(0.5)
    assert micro["recall"] == pytest.approx(0.5)
    assert micro["f1"] == pytest.approx(0.5)


def test_relation_prf_symmetric_coref_is_order_invariant() -> None:
    coref = RelationType.COREFERENCE
    gold = [RelationEdge(head_id="a", tail_id="b", relation_type=coref, directed=False)]
    pred = [RelationEdge(head_id="b", tail_id="a", relation_type=coref, directed=False)]
    assert relation_prf(pred, gold)["micro"]["f1"] == pytest.approx(1.0)


def test_conll_coref_known_values() -> None:
    gold = [{"a", "b", "c"}]
    pred = [{"a", "b"}, {"c"}]
    scores = conll_coref_f1(pred, gold)
    assert scores["muc_f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert scores["b_cubed_f1"] == pytest.approx(0.7143, abs=1e-3)
    assert scores["ceafe_f1"] == pytest.approx(0.5333, abs=1e-3)
    assert scores["conll_f1"] == pytest.approx((2 / 3 + 0.7143 + 0.5333) / 3, abs=1e-3)


def test_mrr_hits_known_values() -> None:
    gold = ["b", "c"]
    rankings = [["b", "x"], ["y", "c", "z"]]
    scores = mrr_hits(gold, rankings, ks=(1, 3))
    assert scores["mrr"] == pytest.approx(0.75)
    assert scores["hits@1"] == pytest.approx(0.5)
    assert scores["hits@3"] == pytest.approx(1.0)


def test_mrr_hits_missing_gold_counts_as_zero() -> None:
    scores = mrr_hits(["z"], [["a", "b"]], ks=(1,))
    assert scores["mrr"] == pytest.approx(0.0)
    assert scores["hits@1"] == pytest.approx(0.0)


def test_mrr_hits_time_aware_filtered_raises_rank() -> None:
    # gold "b"; competitor "c" ranks above it -> raw rank 2 (MRR 0.5). The
    # time-aware filtered setting drops the other-true "c" -> rank 1 (MRR 1.0).
    gold, rankings = ["b"], [["c", "b", "d"]]
    assert mrr_hits(gold, rankings)["mrr"] == pytest.approx(0.5)
    assert mrr_hits(gold, rankings, filter_sets=[{"c"}])["mrr"] == pytest.approx(1.0)
    # the gold itself is never filtered, even if present in its filter set
    assert mrr_hits(gold, rankings, filter_sets=[{"c", "b"}])["mrr"] == pytest.approx(1.0)


def test_relation_prf_temporal_closure_recovers_recall() -> None:
    # MAVEN-ERE ships temporal gold as a transitive closure; a model that emits
    # only the sparse generating chain is unfairly penalised under exact match.
    chain = ["a", "b", "c", "d", "e"]
    pred = [_temporal(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]  # 4 edges
    gold = [
        _temporal(chain[i], chain[j])
        for i in range(len(chain))
        for j in range(i + 1, len(chain))
    ]  # 10 edges = closure of the chain
    raw = relation_prf(pred, gold)["temporal"]
    assert raw["precision"] == pytest.approx(1.0)
    assert raw["recall"] == pytest.approx(0.4)  # 4 / 10 — the distortion
    closed = relation_prf(pred, gold, temporal_closure=True)["temporal"]
    assert closed["precision"] == pytest.approx(1.0)
    assert closed["recall"] == pytest.approx(1.0)
    assert closed["f1"] == pytest.approx(1.0)


def test_relation_prf_closure_only_strict_temporal() -> None:
    # OVERLAP is not a strict order: closure must NOT manufacture a->c.
    pred = [_overlap("a", "b"), _overlap("b", "c")]
    gold = [_overlap("a", "b"), _overlap("b", "c")]
    closed = relation_prf(pred, gold, temporal_closure=True)["temporal"]
    assert closed["n_pred"] == 2
    assert closed["n_gold"] == 2
    assert closed["f1"] == pytest.approx(1.0)


def test_relation_prf_closure_leaves_causal_untouched() -> None:
    # Causality is not transitively closed in MAVEN-ERE; the toggle is temporal-only.
    pred = [_causal("a", "b"), _causal("b", "c")]
    gold = [_causal("a", "b"), _causal("b", "c"), _causal("a", "c")]
    closed = relation_prf(pred, gold, temporal_closure=True)["causal"]
    assert closed["recall"] == pytest.approx(2 / 3)


def test_relation_prf_closure_does_not_change_default() -> None:
    # Backward compatibility: the default path is untouched exact match.
    gold = [_temporal("a", "b"), _temporal("b", "c")]
    pred = [_temporal("a", "b"), _temporal("a", "c")]
    assert relation_prf(pred, gold)["micro"]["f1"] == pytest.approx(0.5)


def test_relation_prf_temporal_closure_omits_reflexive_pairs_from_cycles() -> None:
    pred = [_temporal("a", "b"), _temporal("b", "a")]
    gold = [_temporal("a", "b"), _temporal("b", "a")]

    temporal = relation_prf(pred, gold, temporal_closure=True)["temporal"]

    assert temporal["n_pred"] == 2
    assert temporal["n_gold"] == 2
    assert temporal["f1"] == pytest.approx(1.0)
