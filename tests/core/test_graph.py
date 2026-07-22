"""Tests for graph primitives and consistency diagnostics."""

from __future__ import annotations

from finekg.core.eval import consistency_report
from finekg.core.graph import (
    close_pairs,
    coreference_clusters,
    find_cycles,
    is_acyclic,
    transitive_closure_pairs,
)
from finekg.core.schema import EventGraph, EventNode, RelationEdge, RelationType


def _node(eid: str) -> EventNode:
    return EventNode(event_id=eid, event_type="E", doc_id="d")


def _causal(h: str, t: str) -> RelationEdge:
    return RelationEdge(head_id=h, tail_id=t, relation_type=RelationType.CAUSAL, subtype="CAUSE")


def _coref(h: str, t: str) -> RelationEdge:
    return RelationEdge(
        head_id=h, tail_id=t, relation_type=RelationType.COREFERENCE, directed=False
    )


def _temporal(h: str, t: str) -> RelationEdge:
    return RelationEdge(
        head_id=h,
        tail_id=t,
        relation_type=RelationType.TEMPORAL,
        subtype="BEFORE",
    )


def _graph(edges: list[RelationEdge]) -> EventGraph:
    ids = {e.head_id for e in edges} | {e.tail_id for e in edges}
    return EventGraph(nodes={i: _node(i) for i in ids}, edges=edges)


def test_coreference_clusters_connected_components() -> None:
    g = _graph([_coref("a", "b"), _coref("b", "c"), _coref("x", "y")])
    clusters = {frozenset(c) for c in coreference_clusters(g)}
    assert frozenset({"a", "b", "c"}) in clusters
    assert frozenset({"x", "y"}) in clusters


def test_causal_cycle_detection() -> None:
    cyclic = _graph([_causal("a", "b"), _causal("b", "a")])
    assert not is_acyclic(cyclic, RelationType.CAUSAL)
    assert len(find_cycles(cyclic, RelationType.CAUSAL)) >= 1

    acyclic = _graph([_causal("a", "b"), _causal("b", "c")])
    assert is_acyclic(acyclic, RelationType.CAUSAL)


def test_consistency_report_flags_causal_cycle() -> None:
    report = consistency_report(_graph([_causal("a", "b"), _causal("b", "a")]))
    assert report["causal_cycle_count"] >= 1.0


def test_close_pairs_chain() -> None:
    assert close_pairs([("a", "b"), ("b", "c")]) == {("a", "b"), ("b", "c"), ("a", "c")}


def test_close_pairs_empty() -> None:
    assert close_pairs([]) == set()


def test_close_pairs_diamond_adds_only_implied() -> None:
    pairs = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    closed = close_pairs(pairs)
    assert ("a", "d") in closed  # implied through both b and c
    assert closed == set(pairs) | {("a", "d")}


def test_close_pairs_cycle_omits_reflexive_pairs() -> None:
    assert close_pairs([("a", "b"), ("b", "a")]) == {("a", "b"), ("b", "a")}


def test_transitive_closure_pairs_cycle_omits_reflexive_pairs() -> None:
    graph = _graph([_temporal("a", "b"), _temporal("b", "a")])
    assert transitive_closure_pairs(graph, RelationType.TEMPORAL, {"BEFORE"}) == {
        ("a", "b"),
        ("b", "a"),
    }
