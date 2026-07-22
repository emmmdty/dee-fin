"""Role-specialised relation proposer agents.

A proposer wraps a registered relation extractor and emits only the relation
types it is responsible for, so a committee (coreference / temporal / causal)
covers the relation families with role-specific judgement instead of one
monolithic pass. The CPU stub backs onto the deterministic `heuristic`
extractor; the server swaps in the `llm` extractor prompted for the role — same
agent, a different `extractor` name.
"""

from __future__ import annotations

from collections.abc import Iterable

from finekg.agents.protocol import Agent, Blackboard, Message, agent_roles
from finekg.core.schema import RelationType
from finekg.relations.extractor import relation_extractors
from finekg.relations.extractor.base import ExtractionContext

__all__ = ["ProposerAgent"]


@agent_roles.register("relation_proposer")
class ProposerAgent(Agent):
    """Proposes the relations of its assigned types among the context's nodes."""

    def __init__(
        self,
        role: str = "relation_proposer",
        relation_types: Iterable[str] | None = None,
        extractor: str = "heuristic",
        extractor_kwargs: dict | None = None,
    ) -> None:
        self.role = role
        self.relation_types = (
            {RelationType(t) for t in relation_types} if relation_types else set(RelationType)
        )
        self._extractor = relation_extractors.create(extractor, **(extractor_kwargs or {}))

    def act(self, board: Blackboard) -> Message:
        nodes = board.context["nodes"]
        context: ExtractionContext | None = board.context.get("ext_context")
        edges = [
            edge
            for edge in self._extractor.extract(nodes, context)
            if edge.relation_type in self.relation_types
        ]
        return Message(
            role=self.role,
            kind="propose",
            payload=edges,
            rationale=f"{len(edges)} {sorted(t.value for t in self.relation_types)} edge(s)",
        )
