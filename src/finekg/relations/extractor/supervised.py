"""Discriminative supervised relation extractor (RoBERTa pair-classification).

Reproduces the official MAVEN-ERE strong baseline: gold event mentions are
given and the model labels every candidate mention pair. A RoBERTa encoder pools
each trigger's representation `h_i`; the pair feature `[h_i; h_j; h_i⊙h_j;
|h_i−h_j|]` feeds one linear head per relation family (index 0 = NONE).

torch/transformers are imported behind an availability guard (as in
`succession/model.py`), so the module imports and the extractor **registers** on a
CPU box; only running `extract` (encoding + scoring) needs the `llm` extra + GPU.

Mention location is fail-fast (as in `succession/encode.py`): a trigger that
cannot be found in its sentence raises rather than reading a wrong token.
"""

from __future__ import annotations

from pathlib import Path

from finekg.core.schema import EventNode, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)
from finekg.relations.pairs import candidate_pairs

__all__ = ["TORCH_AVAILABLE", "SupervisedRelationExtractor"]

# family value (RelationType.value) -> contract RelationType
_FAMILY_TYPE = {
    "temporal": RelationType.TEMPORAL,
    "causal": RelationType.CAUSAL,
    "subevent": RelationType.SUBEVENT,
}

# Ordered labels per family; index 0 is the negative (NONE) class. Subtypes match
# what `data/maven_ere.py` emits (upper-cased temporal/causal keys; SUBEVENT_OF).
_FAMILY_SUBTYPES: dict[str, tuple[str, ...]] = {
    "temporal": ("NONE", "BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"),
    "causal": ("NONE", "CAUSE", "PRECONDITION"),
    "subevent": ("NONE", "SUBEVENT_OF"),
}

try:  # pragma: no cover - exercised on the GPU server
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - the local CPU path
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class PairClassifier(nn.Module):
        """One linear head per family over the 4-way pair feature."""

        def __init__(self, hidden_size: int, subtype_counts: dict[str, int]) -> None:
            super().__init__()
            self.heads = nn.ModuleDict(
                {fam: nn.Linear(hidden_size * 4, n) for fam, n in subtype_counts.items()}
            )

        def forward(self, pair_feats: torch.Tensor) -> dict[str, torch.Tensor]:
            return {fam: head(pair_feats) for fam, head in self.heads.items()}

    def _pair_features(head_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        """`[h_i; h_j; h_i⊙h_j; |h_i−h_j|]` — the standard pair-classification feature."""
        return torch.cat(
            [head_emb, tail_emb, head_emb * tail_emb, (head_emb - tail_emb).abs()], dim=-1
        )


@relation_extractors.register("supervised")
class SupervisedRelationExtractor(RelationExtractor):
    """Labels every document-level candidate mention pair via per-family heads.

    The encoder + heads are torch-backed and loaded lazily on the first `extract`;
    `__init__` stays torch-free so the pipeline instantiates on CPU.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        max_distance: int | None = None,
        max_length: int = 256,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.max_distance = max_distance
        self.max_length = max_length
        self._model = None  # heads; lazy-loaded on first extract (needs torch)
        self._encoder = None
        self._tokenizer = None
        self._device = "cpu"

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

    # ---- torch-backed scoring (lazy) ------------------------------------- #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "supervised extract needs torch + transformers: install the `llm` extra "
                "(uv sync --extra llm). This is a GPU-only path."
            )
        if not self.checkpoint_path:
            raise ValueError("supervised: checkpoint_path is required to run extract")
        ckpt = Path(self.checkpoint_path)
        heads_file = ckpt / "heads.pt"
        if not heads_file.exists():
            raise FileNotFoundError(f"supervised: trained heads not found at {heads_file}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        self._encoder = AutoModel.from_pretrained(str(ckpt))
        counts = {fam: len(subs) for fam, subs in _FAMILY_SUBTYPES.items()}
        self._model = PairClassifier(self._encoder.config.hidden_size, counts)
        self._model.load_state_dict(torch.load(heads_file, map_location="cpu"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder.to(self._device).eval()
        self._model.to(self._device).eval()

    def _encode_mentions(self, nodes: list[EventNode], doc_text: str) -> dict[str, torch.Tensor]:
        """Trigger representation per node: encode each sentence once, pool the
        token covering the trigger. Unlocatable triggers raise (no silent pos-0)."""
        lines = doc_text.split("\n")
        by_sent: dict[int, list[EventNode]] = {}
        for node in nodes:
            span = node.trigger_evidence[0] if node.trigger_evidence else None
            if span is None or span.sent_id is None:
                raise ValueError(
                    f"supervised: node {node.event_id} lacks a sentence-anchored trigger"
                )
            by_sent.setdefault(span.sent_id, []).append(node)

        embs: dict[str, torch.Tensor] = {}
        for sent_id, group in by_sent.items():
            if not 0 <= sent_id < len(lines):
                raise ValueError(f"supervised: sent_id {sent_id} out of range in {group[0].doc_id}")
            sentence = lines[sent_id]
            enc = self._tokenizer(
                sentence,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            inputs = {k: v.to(self._device) for k, v in enc.items() if k != "offset_mapping"}
            with torch.no_grad():
                hidden = self._encoder(**inputs).last_hidden_state[0]
            for node in group:
                char = sentence.find(node.trigger)
                if char < 0:
                    raise ValueError(
                        f"supervised: trigger {node.trigger!r} of {node.event_id} not in its "
                        "sentence -- mention unlocatable, refusing to read a wrong token"
                    )
                tok = next((i for i, (s, e) in enumerate(offsets) if s <= char < e), None)
                if tok is None:
                    raise ValueError(
                        f"supervised: trigger of {node.event_id} fell outside the tokenised "
                        "span (truncated) -- raise max_length"
                    )
                embs[node.event_id] = hidden[tok]
        return embs

    def _score_pairs(
        self,
        nodes: list[EventNode],
        pairs: list[tuple[str, str]],
        context: ExtractionContext | None,
    ) -> dict[tuple[str, str], dict[str, tuple[str, float]]]:
        """Per non-NONE family, the (subtype, prob) each candidate pair is assigned."""
        self._ensure_model()
        doc_text = context.doc_text.get(nodes[0].doc_id, "") if context and nodes else ""
        if not doc_text:
            raise ValueError("supervised: extract needs context.doc_text for the document")
        embs = self._encode_mentions(nodes, doc_text)
        feats = torch.stack([_pair_features(embs[h], embs[t]) for h, t in pairs])
        with torch.no_grad():
            logits = self._model(feats.to(self._device))
        result: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}
        for family, subtypes in _FAMILY_SUBTYPES.items():
            probs = torch.softmax(logits[family], dim=-1)
            conf, idx = probs.max(dim=-1)
            for pair, i, c in zip(pairs, idx.tolist(), conf.tolist(), strict=True):
                if i == 0:  # NONE
                    continue
                result.setdefault(pair, {})[family] = (subtypes[i], float(c))
        return result
