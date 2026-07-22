"""Grounding / faithfulness verifier agent — the relation-stage face of the
system's verification currency.

It reads the proposed edges off the blackboard, checks each one's evidence
against the source text, and assigns a per-edge faithfulness score; edges below
the admission threshold are abstained (kept out of the graph). The CPU proxy
uses the existing grounding check — is the cited span locatable? — scaled by the
edge's own confidence. The server verifier replaces the proxy with a real
intervention (mask the evidence span, re-query the LLM, measure the prediction
drop) behind the same `Verifier.score` contract, so the pipeline and metrics are
unchanged.
"""

from __future__ import annotations

from finekg.agents.protocol import Agent, Blackboard, Message, Verifier, agent_roles
from finekg.core.schema import EventNode, RelationEdge
from finekg.relations.extractor.base import ExtractionContext
from finekg.relations.grounding import ground_relations

__all__ = [
    "edge_grounding_faithfulness",
    "GroundingFaithfulnessVerifier",
    "GroundingVerifierAgent",
]


def edge_grounding_faithfulness(
    edge: RelationEdge,
    nodes: list[EventNode],
    context: ExtractionContext | None = None,
    *,
    require_evidence: bool = True,
) -> float:
    """CPU proxy for an edge's evidence faithfulness.

    The edge's confidence when its cited evidence can be located in the source
    text (the grounding check), else 0.0 — so an ungrounded edge scores zero
    faithfulness regardless of how confident the proposer was.
    """
    grounded = ground_relations([edge], nodes, context, require_evidence=require_evidence)
    return float(edge.confidence) if grounded.kept else 0.0


class GroundingFaithfulnessVerifier(Verifier):
    """The faithfulness-scoring contract used by the verifier agent (and reusable
    on its own for evaluation)."""

    def score(self, candidate: RelationEdge, board: Blackboard) -> float:
        nodes = board.context["nodes"]
        context = board.context.get("ext_context")
        return edge_grounding_faithfulness(candidate, nodes, context)


@agent_roles.register("grounding_verifier")
class GroundingVerifierAgent(Agent):
    """Scores every proposed edge and admits only the sufficiently faithful ones."""

    role = "grounding_verifier"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def act(self, board: Blackboard) -> Message:
        nodes = board.context["nodes"]
        context: ExtractionContext | None = board.context.get("ext_context")
        # Debate rounds repeat proposers, so collapse to unique edges by identity.
        unique: dict[tuple[str, str, str, str], RelationEdge] = {}
        for message in board.of_kind("propose"):
            for edge in message.payload:
                unique.setdefault(edge.key(), edge)
        # Annotate copies, not the proposers' objects: the blackboard transcript
        # is provenance, so the "propose" payloads must survive verification
        # unchanged (grounding also rewrites evidence spans in place).
        edges = [edge.model_copy(deep=True) for edge in unique.values()]

        grounded = ground_relations(edges, nodes, context, require_evidence=True)
        grounded_keys = {edge.key() for edge in grounded.kept}
        admitted: list[RelationEdge] = []
        for edge in edges:
            edge.faithfulness = float(edge.confidence) if edge.key() in grounded_keys else 0.0
            edge.admitted = edge.faithfulness >= self.threshold
            if edge.admitted:
                admitted.append(edge)
        return Message(
            role=self.role,
            kind="verify",
            payload=admitted,
            rationale=f"admitted {len(admitted)}/{len(edges)} (thr={self.threshold})",
        )
