"""Round-trip and adapter tests for the data contracts."""

from __future__ import annotations

from finekg.core.io import dump_event_graph, event_nodes_from_sarge, load_event_graph
from finekg.core.schema import EventGraph, EventNode, EvidenceSpan, RelationEdge, RelationType


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


def test_event_nodes_from_sarge_adapter() -> None:
    records = [
        {
            "event_id": "x1",
            "event_type": "EquityFreeze",
            "doc_id": "doc",
            "arguments": {"被冻结方": "甲公司"},
            "evidence": {"被冻结方": [{"char_start": 0, "char_end": 3, "text": "甲公司"}]},
            "time_anchor": "2021-02-10",
        }
    ]
    nodes = event_nodes_from_sarge(records)
    assert len(nodes) == 1
    assert nodes[0].event_type == "EquityFreeze"
    assert nodes[0].argument_evidence["被冻结方"][0].text == "甲公司"


def test_event_nodes_from_sarge_degrades_gracefully() -> None:
    nodes = event_nodes_from_sarge([{"doc_id": "d"}])
    assert len(nodes) == 1
    assert nodes[0].event_id  # synthesized id


def test_event_nodes_from_sarge_reads_the_key_the_converter_writes() -> None:
    # `scripts/sarge_to_event_nodes.py` writes `argument_evidence`. Reading only
    # the legacy `evidence` key silently emptied every node's evidence.
    records = [
        {
            "event_id": "x1",
            "doc_id": "doc",
            "arguments": {"被冻结方": "甲公司"},
            "argument_evidence": {
                "被冻结方": [{"char_start": 0, "char_end": 3, "text": "甲公司"}]
            },
        }
    ]
    nodes = event_nodes_from_sarge(records)
    assert nodes[0].argument_evidence["被冻结方"][0].text == "甲公司"


def test_event_nodes_from_sarge_passes_metadata_through() -> None:
    # The `object` annotation rides in `metadata` because EventNode's schema is
    # frozen; dropping it here severed entity-level quads from their converter.
    nodes = event_nodes_from_sarge([{"doc_id": "d", "metadata": {"object": "乙公司"}}])
    assert nodes[0].metadata == {"object": "乙公司"}
