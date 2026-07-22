"""The multi-agent relation subsystem must build the same kind of consistent,
evidence-grounded graph as the single-extractor pipeline — and additionally
abstain on edges whose evidence cannot be grounded (the verifier's job).
"""

from __future__ import annotations

import pytest

from finekg.agents.protocol import Blackboard, Message
from finekg.core.eval import consistency_report
from finekg.core.io import load_event_nodes
from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.agents import GroundingVerifierAgent
from finekg.relations.pipeline import MultiAgentRelationConfig, MultiAgentRelationPipeline


def test_multiagent_pipeline_builds_consistent_grounded_graph(fixtures_dir) -> None:
    nodes = load_event_nodes(fixtures_dir / "event_graph_zh" / "event_nodes.jsonl")
    graph = MultiAgentRelationPipeline().build_graph(nodes)
    assert len(graph.nodes) == len(nodes)
    assert len(graph.edges) > 0
    report = consistency_report(graph)
    assert report["causal_cycle_count"] == 0.0
    assert report["temporal_cycle_count"] == 0.0
    # the verifier annotated faithfulness on the admitted (non-closure) edges
    assert any(e.faithfulness is not None for e in graph.edges)


def test_no_verifier_ablation_still_builds_consistent_graph(fixtures_dir) -> None:
    nodes = load_event_nodes(fixtures_dir / "event_graph_zh" / "event_nodes.jsonl")
    graph = MultiAgentRelationPipeline(MultiAgentRelationConfig(use_verifier=False)).build_graph(
        nodes
    )
    assert len(graph.edges) > 0
    assert consistency_report(graph)["causal_cycle_count"] == 0.0


def test_grounding_verifier_admits_grounded_and_abstains_ungrounded() -> None:
    nodes = [
        EventNode(event_id="a", event_type="E", doc_id="d"),
        EventNode(event_id="b", event_type="E", doc_id="d"),
    ]
    grounded = RelationEdge(
        head_id="a",
        tail_id="b",
        relation_type=RelationType.CAUSAL,
        confidence=0.8,
        evidence=[EvidenceSpan(doc_id="d", char_start=0, char_end=5, text="abcde")],
    )
    ungrounded = RelationEdge(
        head_id="a",
        tail_id="b",
        relation_type=RelationType.TEMPORAL,
        subtype="BEFORE",
        confidence=0.7,
        evidence=[],
    )
    board = Blackboard(context={"nodes": nodes, "ext_context": None})
    board.post(Message(role="p", kind="propose", payload=[grounded, ungrounded]))

    admitted = GroundingVerifierAgent(threshold=0.5).act(board).payload
    by_key = {edge.key(): edge for edge in admitted}
    assert grounded.key() in by_key
    assert ungrounded.key() not in by_key
    assert by_key[grounded.key()].faithfulness == pytest.approx(0.8)
    assert by_key[grounded.key()].admitted is True
    # The verifier annotates copies: the proposers' posted edges stay pristine
    # (the blackboard transcript is provenance).
    assert grounded.faithfulness is None
    assert ungrounded.faithfulness is None
    assert ungrounded.admitted is True  # the schema default, untouched
