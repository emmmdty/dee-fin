"""Round-trip and adapter tests for the data contracts."""

from __future__ import annotations

from ekg.core.io import dump_event_graph, load_event_graph
from ekg.core.schema import EventGraph, EventNode, EvidenceSpan, RelationEdge, RelationType


def test_event_graph_round_trip(tmp_path) -> None:
    node = EventNode(
        event_id="e1",
        event_type="EquityPledge",
        doc_id="d1",
        subject="甲公司",
        time_anchor="2021-01-05",
        trigger_evidence=[EvidenceSpan(doc_id="d1", char_start=0, char_end=3, text="甲公司")],
    )
    edge = RelationEdge(
        head_id="e1", tail_id="e2", relation_type=RelationType.TEMPORAL, subtype="BEFORE"
    )
    graph = EventGraph(nodes={"e1": node}, edges=[edge])

    path = tmp_path / "graph.json"
    dump_event_graph(path, graph)
    restored = load_event_graph(path)
    assert restored.nodes["e1"].subject == "甲公司"
    assert restored.edges[0].relation_type is RelationType.TEMPORAL
    assert restored.edges[0].subtype == "BEFORE"
