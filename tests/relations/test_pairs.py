"""Pair-classification harness: the same-setting bridge between generative
edge lists and supervised pair classifiers (MAVEN-ERE convention: gold mentions
given, score per-pair labels). Load-bearing bits: mention order comes from text
position (the loader appends per event, not in reading order), hallucinated
mention ids are penalised *and* counted, and the window diagnostic separates
the structural recall ceiling from model misses.
"""

from __future__ import annotations

import pytest

from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.pairs import (
    candidate_pairs,
    edges_to_pair_labels,
    mention_order,
    pair_examples,
    pair_prf,
    window_recall_ceiling,
)


def _node(mid: str, sent: int, pos: int) -> EventNode:
    return EventNode(
        event_id=mid,
        event_type="E",
        doc_id="d1",
        trigger=mid,
        trigger_evidence=[
            EvidenceSpan(doc_id="d1", char_start=pos, char_end=pos + 1, sent_id=sent, text=mid)
        ],
    )


def _doc() -> RelationDocument:
    # textual order: m1 (s0,0) < m2 (s0,10) < m3 (s1,0) < m4 (s2,0);
    # nodes deliberately shuffled to prove order comes from spans, not the list
    nodes = [_node("m3", 1, 0), _node("m1", 0, 0), _node("m4", 2, 0), _node("m2", 0, 10)]
    gold = [
        RelationEdge(
            head_id="m1", tail_id="m2", relation_type=RelationType.COREFERENCE, directed=False
        ),
        RelationEdge(
            head_id="m1", tail_id="m3", relation_type=RelationType.TEMPORAL, subtype="BEFORE"
        ),
        RelationEdge(
            head_id="m1", tail_id="m4", relation_type=RelationType.CAUSAL, subtype="CAUSE"
        ),
    ]
    return RelationDocument(doc_id="d1", nodes=nodes, gold_edges=gold)


def test_mention_order_uses_text_position() -> None:
    order = mention_order(_doc())
    assert [m for m, _ in sorted(order.items(), key=lambda kv: kv[1])] == ["m1", "m2", "m3", "m4"]


def test_candidate_pairs_full_and_windowed() -> None:
    doc = _doc()
    assert len(candidate_pairs(doc)) == 12  # 4 * 3 ordered pairs
    near = candidate_pairs(doc, max_distance=1)
    assert len(near) == 6  # only textual neighbours, both directions
    assert ("m1", "m2") in near and ("m2", "m1") in near
    assert ("m1", "m3") not in near


def test_edges_to_pair_labels_dedups_by_confidence() -> None:
    edges = [
        RelationEdge(
            head_id="m1", tail_id="m3", relation_type=RelationType.TEMPORAL,
            subtype="OVERLAP", confidence=0.4,
        ),
        RelationEdge(
            head_id="m1", tail_id="m3", relation_type=RelationType.TEMPORAL,
            subtype="BEFORE", confidence=0.9,
        ),
    ]
    labels = edges_to_pair_labels(edges, family=RelationType.TEMPORAL)
    assert labels == {("m1", "m3"): "BEFORE"}


def test_pair_prf_perfect_and_hallucinated() -> None:
    doc = _doc()
    perfect = pair_prf(doc.gold_edges, doc)
    assert perfect["micro"]["f1"] == pytest.approx(1.0)
    assert perfect["coreference"]["f1"] == pytest.approx(1.0)
    assert perfect["diagnostics"]["hallucinated_pred_pairs"] == 0

    hallucinated = doc.gold_edges + [
        RelationEdge(
            head_id="m1", tail_id="ghost", relation_type=RelationType.CAUSAL, subtype="CAUSE"
        )
    ]
    res = pair_prf(hallucinated, doc)
    # the ghost pair is a false positive AND separately accounted
    assert res["diagnostics"]["hallucinated_pred_pairs"] == 1
    assert res["causal"]["precision"] == pytest.approx(0.5)
    assert res["causal"]["recall"] == pytest.approx(1.0)


def test_pair_prf_windowed_reports_structural_ceiling() -> None:
    doc = _doc()
    res = pair_prf(doc.gold_edges, doc, max_distance=2)
    # causal m1->m4 (distance 3) leaves the universe on both sides
    assert res["diagnostics"]["out_of_window_gold"] == 1
    assert res["causal"]["n_gold"] == 0
    assert res["temporal"]["f1"] == pytest.approx(1.0)


def test_window_recall_ceiling_counts_reachable_gold() -> None:
    doc = _doc()
    # window of 2 consecutive mentions: only the adjacent coref pair fits
    ceiling = window_recall_ceiling([doc], window_events=2)
    assert ceiling["reachable_gold"] == 1
    assert ceiling["total_gold"] == 3
    assert ceiling["ceiling"] == pytest.approx(1 / 3)
    # a window spanning the whole document reaches everything
    assert window_recall_ceiling([doc], window_events=4)["ceiling"] == pytest.approx(1.0)


def test_pair_examples_cover_universe_with_labels() -> None:
    doc = _doc()
    examples = pair_examples(doc)
    by_pair = {(e.head_id, e.tail_id): e for e in examples}
    assert len(examples) == 12
    assert by_pair[("m1", "m3")].labels == {"temporal": "BEFORE"}
    assert by_pair[("m1", "m2")].labels == {"coreference": "COREF"}
    # symmetric coref labels both directions; unrelated pairs carry no labels
    assert by_pair[("m2", "m1")].labels == {"coreference": "COREF"}
    assert by_pair[("m3", "m4")].labels == {}
    assert by_pair[("m1", "m4")].distance == 3
