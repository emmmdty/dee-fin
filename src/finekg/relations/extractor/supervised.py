"""Discriminative supervised relation extractor (RoBERTa pair-classification).

Reproduces the official MAVEN-ERE strong baseline: gold event mentions are
given and the model labels every candidate mention pair. Encode-once RoBERTa +
per-family heads. torch is imported lazily inside methods so the package
imports on a CPU-only machine without the `llm` extra.
"""

from __future__ import annotations

from finekg.core.schema import EventNode, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)
from finekg.relations.pairs import candidate_pairs

__all__ = ["SupervisedRelationExtractor"]

# family value (pairs.py / RelationType.value) -> contract RelationType
_FAMILY_TYPE = {
    "temporal": RelationType.TEMPORAL,
    "causal": RelationType.CAUSAL,
    "subevent": RelationType.SUBEVENT,
}


@relation_extractors.register("supervised")
class SupervisedRelationExtractor(RelationExtractor):
    """Labels every document-level candidate mention pair via per-family heads.

    The scoring model is torch-backed and loaded lazily on the first `extract`;
    `__init__` stays torch-free so the pipeline instantiates on CPU.
    """

    def __init__(
        self, checkpoint_path: str | None = None, max_distance: int | None = None
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.max_distance = max_distance
        self._model = None  # lazy-loaded on first extract (needs torch)

    def _candidate_pairs(self, nodes: list[EventNode]) -> list[tuple[str, str]]:
        if not nodes:
            return []
        doc = RelationDocument(doc_id=nodes[0].doc_id, nodes=nodes, gold_edges=[])
        return candidate_pairs(doc, self.max_distance)

    def extract(
        self, nodes: list[EventNode], context: ExtractionContext | None = None
    ) -> list[RelationEdge]:
        pairs = self._candidate_pairs(nodes)
        if not pairs:
            return []
        by_id = {n.event_id: n for n in nodes}
        scored = self._score_pairs(nodes, pairs, context)
        edges: list[RelationEdge] = []
        for (head, tail), families in scored.items():
            for family, (subtype, prob) in families.items():
                edges.append(
                    RelationEdge(
                        head_id=head,
                        tail_id=tail,
                        relation_type=_FAMILY_TYPE[family],
                        subtype=subtype,
                        directed=True,
                        confidence=prob,
                        evidence=list(by_id[head].trigger_evidence)
                        + list(by_id[tail].trigger_evidence),
                    )
                )
        return edges

    def _score_pairs(
        self,
        nodes: list[EventNode],
        pairs: list[tuple[str, str]],
        context: ExtractionContext | None,
    ) -> dict[tuple[str, str], dict[str, tuple[str, float]]]:
        """Predict a (subtype, prob) per non-NONE family for each candidate pair.

        Model-backed; implemented in Task 3 (torch). Returns only the pairs the
        heads assign a non-NONE label, keyed by (head_id, tail_id) then family.
        """
        raise NotImplementedError  # Task 3: model-backed scoring
