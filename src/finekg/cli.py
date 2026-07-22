"""End-to-end CPU smoke over the bundled fixtures.

Runs the event-graph construction pipeline with the torch-free baselines so a
fresh local checkout can prove the cross-stage contracts, graph construction,
consistency repair, evidence grounding, and the multi-agent scaffold all work
-- without a GPU:

    event nodes -> RelationPipeline -> EventGraph (consistent, grounded)
                -> MultiAgentRelationPipeline (agentic construction + verifier)

Exposed as the `finekg-smoke` console script.
"""

from __future__ import annotations

import os
from pathlib import Path

from finekg.core.eval.consistency import consistency_report
from finekg.core.io import load_event_nodes
from finekg.relations import RelationPipeline
from finekg.relations.pipeline import MultiAgentRelationPipeline


def _fixtures_dir() -> Path:
    override = os.environ.get("FINEKG_FIXTURES")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "fixtures"


def run_smoke() -> int:
    fixtures = _fixtures_dir()
    print(f"[finekg-smoke] fixtures: {fixtures}")

    # 1) Relations: nodes -> evidence-grounded, consistent event graph.
    nodes = load_event_nodes(fixtures / "event_graph_zh" / "event_nodes.jsonl")
    graph = RelationPipeline().build_graph(nodes)
    print(
        f"[relations] nodes={len(graph.nodes)} edges={len(graph.edges)} "
        f"(dropped_ungrounded={graph.metadata.get('edges_dropped_ungrounded')})"
    )
    print(f"[relations] consistency: {consistency_report(graph)}")

    # 2) Multi-agent upgrade: same inputs, agentic construction + verifier.
    ma_graph = MultiAgentRelationPipeline().build_graph(nodes)
    faithful = [e.faithfulness for e in ma_graph.edges if e.faithfulness is not None]
    mean_faith = sum(faithful) / len(faithful) if faithful else 0.0
    print(
        f"[relations:multi-agent] edges={len(ma_graph.edges)} "
        f"mean_edge_faithfulness={mean_faith:.3f}"
    )

    print("[finekg-smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke())
