"""Evidence grounding for predicted relations (anti-fabrication).

Every relation must point at real source text. For neural predictions that
carry an `evidence_quote`, we verify the quote actually occurs in the document
and rewrite it into a precise character span; relations whose evidence cannot
be located are dropped. This is what turns "the model said so" into "the text
says so", and directly supports the faithfulness metric of the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from finekg.core.schema import EventNode, RelationEdge
from finekg.relations.extractor.base import ExtractionContext

__all__ = ["GroundingResult", "ground_relations"]


@dataclass
class GroundingResult:
    kept: list[RelationEdge]
    dropped: list[RelationEdge]

    @property
    def drop_rate(self) -> float:
        total = len(self.kept) + len(self.dropped)
        return len(self.dropped) / total if total else 0.0


def _locate(quote: str, text: str) -> tuple[int, int] | None:
    if not quote:
        return None
    idx = text.find(quote)
    return (idx, idx + len(quote)) if idx >= 0 else None


def ground_relations(
    edges: list[RelationEdge],
    nodes: list[EventNode],
    context: ExtractionContext | None = None,
    require_evidence: bool = True,
) -> GroundingResult:
    """Keep only relations whose evidence can be verified.

    A relation is grounded if it already carries evidence spans, or if its
    quote can be located in `context.doc_text`. When `require_evidence` is
    False, ungrounded relations are kept (used to measure grounding's effect).

    Mutates in place: a located quote has its span rewritten to the precise
    character offsets (and `doc_id` filled in). Callers that must keep the
    input edges pristine — e.g. agents annotating blackboard payloads — should
    pass copies.
    """
    doc_text = context.doc_text if context else {}
    node_by_id = {n.event_id: n for n in nodes}
    kept: list[RelationEdge] = []
    dropped: list[RelationEdge] = []

    for edge in edges:
        doc_id = node_by_id[edge.head_id].doc_id if edge.head_id in node_by_id else ""
        text = doc_text.get(doc_id, "")
        grounded = False
        for span in edge.evidence:
            located = _locate(span.text, text) if text and span.text else None
            if located is not None:
                span.doc_id = span.doc_id or doc_id
                span.char_start, span.char_end = located
                grounded = True
            elif span.char_end > span.char_start:
                grounded = True  # already-positioned evidence (e.g. heuristic)
        if grounded or not require_evidence:
            kept.append(edge)
        else:
            dropped.append(edge)

    return GroundingResult(kept=kept, dropped=dropped)
