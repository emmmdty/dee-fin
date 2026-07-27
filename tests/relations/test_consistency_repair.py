"""RepairTrace: the consistency solver must emit a structured, auditable record
of every edit (dropped/added edge, the violation it resolves, before/after
consistency counts) *without* changing the graph `solve()` already produces.
Checked deterministically on CPU.
"""

from __future__ import annotations

from ekg.core.eval.consistency import consistency_report
from ekg.core.schema import EventGraph, EventNode, RelationEdge, RelationType
from ekg.relations.consistency import (
    GreedyConsistencySolver,
    RepairTrace,
    consistency_solvers,
)


def _nodes(*ids: str) -> dict[str, EventNode]:
    return {i: EventNode(event_id=i, event_type="E", doc_id="d") for i in ids}


def _causal(head: str, tail: str, conf: float, subtype: str = "CAUSE") -> RelationEdge:
    return RelationEdge(
        head_id=head,
        tail_id=tail,
        relation_type=RelationType.CAUSAL,
        subtype=subtype,
        confidence=conf,
    )


def _before(head: str, tail: str, conf: float) -> RelationEdge:
    return RelationEdge(
        head_id=head,
        tail_id=tail,
        relation_type=RelationType.TEMPORAL,
        subtype="BEFORE",
        confidence=conf,
    )


def test_trace_records_causal_cycle_drop_with_before_after() -> None:
    edges = [_causal("a", "b", 0.9), _causal("b", "c", 0.8), _causal("c", "a", 0.3)]
    graph = EventGraph(nodes=_nodes("a", "b", "c"), edges=edges)

    solved, trace = GreedyConsistencySolver().solve_with_trace(graph)

    assert isinstance(trace, RepairTrace)
    assert consistency_report(solved)["causal_cycle_count"] == 0.0
    drops = [e for e in trace.edits if e.action == "drop" and e.violation == "causal_cycle"]
    # The weakest pair in the cycle (c->a, conf 0.3) is the one dropped.
    assert [e.edge_key for e in drops] == [("c", "a", "causal", "CAUSE")]
    assert trace.before["causal_cycle_count"] >= 1.0
    assert trace.after["causal_cycle_count"] == 0.0
    assert trace.counts["edges_before"] == 3
    assert trace.counts["edges_after"] == 2


def test_trace_records_temporal_closure_additions() -> None:
    graph = EventGraph(
        nodes=_nodes("a", "b", "c"), edges=[_before("a", "b", 0.9), _before("b", "c", 0.9)]
    )

    _, trace = GreedyConsistencySolver(close_temporal=True).solve_with_trace(graph)

    adds = [e for e in trace.edits if e.action == "add" and e.violation == "temporal_closure"]
    assert [e.edge_key for e in adds] == [("a", "c", "temporal", "BEFORE")]


def test_trace_records_coreference_dedup_drop() -> None:
    edges = [
        RelationEdge(
            head_id="a", tail_id="b", relation_type=RelationType.COREFERENCE,
            directed=False, confidence=0.9,
        ),
        RelationEdge(
            head_id="b", tail_id="a", relation_type=RelationType.COREFERENCE,
            directed=False, confidence=0.4,  # loser on the same unordered pair
        ),
    ]
    graph = EventGraph(nodes=_nodes("a", "b"), edges=edges)

    solved, trace = GreedyConsistencySolver().solve_with_trace(graph)

    assert len(solved.edges_of_type(RelationType.COREFERENCE)) == 1
    drops = [e for e in trace.edits if e.violation == "coref_dedup"]
    assert len(drops) == 1
    assert drops[0].action == "drop"
    assert drops[0].confidence == 0.4


def test_identity_solver_emits_empty_trace_and_unchanged_graph() -> None:
    graph = EventGraph(nodes=_nodes("a", "b"), edges=[_causal("a", "b", 0.5)])

    out, trace = consistency_solvers.create("identity").solve_with_trace(graph)

    assert isinstance(trace, RepairTrace)
    assert trace.edits == []
    assert len(out.edges) == 1
    assert trace.before == trace.after


def test_solve_with_trace_leaves_solve_output_unchanged() -> None:
    # The trace path must not alter the graph solve() already produces (default lock).
    edges = [
        _causal("a", "b", 0.9),
        _causal("b", "c", 0.8),
        _causal("c", "a", 0.3),
        _before("a", "b", 0.7),
        _before("b", "c", 0.7),
    ]
    graph = EventGraph(nodes=_nodes("a", "b", "c"), edges=edges)
    solver = GreedyConsistencySolver()

    plain = solver.solve(graph)
    traced, _ = solver.solve_with_trace(graph)

    assert plain.edges == traced.edges
