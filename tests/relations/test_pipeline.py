"""Tests for relation extraction + consistency repair."""

from __future__ import annotations

import pytest

from finekg.core.eval import consistency_report
from finekg.core.io import load_event_nodes
from finekg.core.schema import EventGraph, EventNode, RelationEdge, RelationType
from finekg.relations import RelationPipeline
from finekg.relations.consistency import GreedyConsistencySolver, consistency_solvers
from finekg.relations.extractor.heuristic import HeuristicRelationExtractor


def _dated(event_id: str, subject: str, day: str) -> EventNode:
    return EventNode(event_id=event_id, event_type="E", doc_id=event_id,
                     subject=subject, time_anchor=day)


def test_temporal_scope_corpus_chains_across_companies() -> None:
    # Opt-in scope links two unrelated companies just because their dates are
    # adjacent. Pinned so the (surprising) behaviour stays visible.
    nodes = [_dated("a", "600959", "2021-01-01"), _dated("b", "000001", "2021-01-02")]
    edges = HeuristicRelationExtractor(temporal_scope="corpus")._temporal(nodes)
    assert [(e.head_id, e.tail_id) for e in edges] == [("a", "b")]


def test_temporal_scope_defaults_to_subject() -> None:
    # The default must not invent a BEFORE edge between two unrelated companies;
    # `corpus` chaining is a per-corpus opt-in, not the safe default.
    nodes = [_dated("a", "600959", "2021-01-01"), _dated("b", "000001", "2021-01-02")]
    extractor = HeuristicRelationExtractor()
    assert extractor.temporal_scope == "subject"
    assert extractor._temporal(nodes) == []


def test_temporal_scope_subject_chains_within_one_company() -> None:
    nodes = [
        _dated("a1", "600959", "2021-01-01"),
        _dated("b1", "000001", "2021-01-02"),
        _dated("a2", "600959", "2021-01-03"),
    ]
    edges = HeuristicRelationExtractor(temporal_scope="subject")._temporal(nodes)
    # a1 -> a2 only: no cross-company date adjacency, and b1 has no successor.
    assert [(e.head_id, e.tail_id) for e in edges] == [("a1", "a2")]


def test_temporal_scope_subject_avoids_quadratic_blowup() -> None:
    # 3 companies x 40 dates. Corpus scope chains 120 globally-sorted nodes;
    # subject scope emits one chain per company.
    nodes = [
        _dated(f"{s}-{d}", s, f"2021-01-{d:02d}")
        for s in ("A", "B", "C")
        for d in range(1, 41)
    ]
    corpus = HeuristicRelationExtractor(temporal_scope="corpus")._temporal(nodes)
    subject = HeuristicRelationExtractor(temporal_scope="subject")._temporal(nodes)
    assert len(subject) == 3 * 39
    assert len(corpus) > len(subject)
    assert all(e.head_id.split("-")[0] == e.tail_id.split("-")[0] for e in subject)


def test_unknown_temporal_scope_raises() -> None:
    with pytest.raises(ValueError, match="unknown temporal_scope"):
        HeuristicRelationExtractor(temporal_scope="bogus")


def test_heuristic_pipeline_builds_consistent_graph(fixtures_dir) -> None:
    nodes = load_event_nodes(fixtures_dir / "event_graph_zh" / "event_nodes.jsonl")
    graph = RelationPipeline().build_graph(nodes)
    assert len(graph.nodes) == len(nodes)
    assert len(graph.edges) > 0
    report = consistency_report(graph)
    # The greedy solver must leave no causal/temporal cycles.
    assert report["causal_cycle_count"] == 0.0
    assert report["temporal_cycle_count"] == 0.0


def test_greedy_solver_breaks_injected_causal_cycle() -> None:
    nodes = {i: EventNode(event_id=i, event_type="E", doc_id="d") for i in ("a", "b", "c")}
    edges = [
        RelationEdge(head_id="a", tail_id="b", relation_type=RelationType.CAUSAL, confidence=0.9),
        RelationEdge(head_id="b", tail_id="c", relation_type=RelationType.CAUSAL, confidence=0.8),
        RelationEdge(head_id="c", tail_id="a", relation_type=RelationType.CAUSAL, confidence=0.3),
    ]
    cyclic = EventGraph(nodes=nodes, edges=edges)
    assert consistency_report(cyclic)["causal_cycle_count"] >= 1.0

    solved = consistency_solvers.create("greedy").solve(cyclic)
    assert consistency_report(solved)["causal_cycle_count"] == 0.0
    # The weakest edge (c->a, conf 0.3) is the one dropped.
    causal_pairs = {(e.head_id, e.tail_id) for e in solved.edges_of_type(RelationType.CAUSAL)}
    assert ("c", "a") not in causal_pairs


def test_greedy_temporal_closure_omits_reflexive_pairs() -> None:
    edges = [
        RelationEdge(
            head_id="a",
            tail_id="b",
            relation_type=RelationType.TEMPORAL,
            subtype="BEFORE",
        ),
        RelationEdge(
            head_id="b",
            tail_id="a",
            relation_type=RelationType.TEMPORAL,
            subtype="BEFORE",
        ),
    ]

    implied_pairs = {
        (e.head_id, e.tail_id) for e in GreedyConsistencySolver._temporal_closure(edges)
    }

    assert ("a", "a") not in implied_pairs
    assert ("b", "b") not in implied_pairs


def test_identity_solver_is_noop_ablation() -> None:
    nodes = {i: EventNode(event_id=i, event_type="E", doc_id="d") for i in ("a", "b")}
    edges = [RelationEdge(head_id="a", tail_id="b", relation_type=RelationType.CAUSAL)]
    graph = EventGraph(nodes=nodes, edges=edges)
    out = consistency_solvers.create("identity").solve(graph)
    assert len(out.edges) == 1


def test_heuristic_causal_edges_are_input_order_invariant() -> None:
    """Tied/missing time anchors carry no order: the cue table must decide the
    direction, so reversing the node list cannot change the emitted edge set."""
    from finekg.relations.extractor.heuristic import HeuristicRelationExtractor

    pledge = EventNode(event_id="x1", event_type="EquityPledge", doc_id="d", subject="ACME")
    freeze = EventNode(event_id="x2", event_type="EquityFreeze", doc_id="d", subject="ACME")
    extractor = HeuristicRelationExtractor()

    forward = {e.key() for e in extractor.extract([pledge, freeze])}
    reversed_ = {e.key() for e in extractor.extract([freeze, pledge])}
    assert forward == reversed_
    # The cue (EquityPledge -> EquityFreeze) fires regardless of input order.
    assert ("x1", "x2", RelationType.CAUSAL.value, "CAUSE") in forward
