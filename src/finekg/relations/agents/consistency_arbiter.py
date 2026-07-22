"""Consistency-arbiter agent: resolves the admitted edges into a coherent graph.

It reads the verifier's admitted edges (or, in the −verifier ablation, the raw
proposals), assembles an `EventGraph`, and runs the registered consistency
solver — the classic global-consistency repair (break causal/temporal cycles,
close temporal transitively, dedup coreference) re-cast as the arbiter's tool.

Why keep a symbolic solver in an LLM-era, agentic system: graph-level guarantees
like acyclicity and transitive closure are exactly what free-form agent debate
cannot promise, so the arbiter delegates them to a deterministic solver
(neuro-symbolic division of labour).
"""

from __future__ import annotations

from finekg.agents.protocol import Agent, Blackboard, Message, agent_roles
from finekg.core.schema import EventGraph, RelationEdge
from finekg.relations.consistency import consistency_solvers

__all__ = ["ConsistencyArbiterAgent"]


@agent_roles.register("consistency_arbiter")
class ConsistencyArbiterAgent(Agent):
    """Turns the admitted edges into a globally-consistent `EventGraph`."""

    role = "consistency_arbiter"

    def __init__(self, solver: str = "greedy", solver_kwargs: dict | None = None) -> None:
        self._solver = consistency_solvers.create(solver, **(solver_kwargs or {}))
        self._solver_name = solver

    def act(self, board: Blackboard) -> Message:
        nodes = board.context["nodes"]
        verified = board.latest("verify")
        if verified is not None:
            edges: list[RelationEdge] = list(verified.payload)
        else:  # −verifier ablation: arbitrate the raw proposals (deduped)
            unique: dict[tuple[str, str, str, str], RelationEdge] = {}
            for message in board.of_kind("propose"):
                for edge in message.payload:
                    unique.setdefault(edge.key(), edge)
            edges = list(unique.values())
        graph = EventGraph(
            nodes={node.event_id: node for node in nodes},
            edges=edges,
            metadata={
                "builder": "multi_agent",
                "consistency": self._solver_name,
                "edges_admitted": str(len(edges)),
            },
        )
        return Message(role=self.role, kind="aggregate", payload=self._solver.solve(graph))
