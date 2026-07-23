"""Discriminative supervised relation extractor (RoBERTa pair-classification).

Reproduces the official MAVEN-ERE strong baseline: gold event mentions are
given and the model labels every candidate mention pair. A RoBERTa encoder pools
each trigger's representation `h_i`; the pair feature `[h_i; h_j; h_i⊙h_j;
|h_i−h_j|]` feeds one linear head per relation family (index 0 = NONE).

torch/transformers are imported behind an availability guard (as in
`succession/model.py`), so the module imports and the extractor **registers** on a
CPU box; only running `extract` (encoding + scoring) needs the `llm` extra + GPU.

`locate_trigger_token` (pure-Python, fail-fast) and `encode_trigger_reps` (the
shared encoder pooling) are reused by both inference and training so a mention is
pooled identically either way; an unlocatable trigger raises rather than reading a
wrong token (as in `succession/encode.py`).
"""

from __future__ import annotations

import re
from pathlib import Path

from finekg.core.schema import EventNode, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)
from finekg.relations.pairs import candidate_pairs

__all__ = [
    "TORCH_AVAILABLE",
    "FAMILY_SUBTYPES",
    "SupervisedRelationExtractor",
    "encode_trigger_reps",
    "locate_trigger_token",
]

# family value (RelationType.value) -> contract RelationType
_FAMILY_TYPE = {
    "temporal": RelationType.TEMPORAL,
    "causal": RelationType.CAUSAL,
    "subevent": RelationType.SUBEVENT,
}

# Ordered labels per family; index 0 is the negative (NONE) class. Subtypes match
# what `data/maven_ere.py` emits (upper-cased temporal/causal keys; SUBEVENT_OF).
FAMILY_SUBTYPES: dict[str, tuple[str, ...]] = {
    "temporal": ("NONE", "BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"),
    "causal": ("NONE", "CAUSE", "PRECONDITION"),
    "subevent": ("NONE", "SUBEVENT_OF"),
}


def locate_trigger_token(sentence: str, trigger: str, offsets: list[tuple[int, int]]) -> int:
    """Token index whose char span covers the trigger; fail-fast if unlocatable.

    `offsets` is a tokenizer's `offset_mapping` for `sentence`. Matching is
    **case-insensitive on a word boundary**: MAVEN-ERE's `trigger_word` is
    lower-cased while the sentence keeps its original casing, so a sentence-initial
    or proper-noun trigger ("armed" in "Armed police officers ...") only matches
    case-insensitively -- 0.65% of mentions, which an exact `find` loses. The word
    boundary stops a substring ("arm" inside "armed") from pooling a wrong token.
    Anything still unlocatable raises rather than silently reading position 0;
    shared by inference and training so both pool identically.
    """
    match = re.search(rf"\b{re.escape(trigger)}\b", sentence, re.IGNORECASE)
    if match is None:
        raise ValueError(f"trigger {trigger!r} not in sentence -- unlocatable mention")
    tok = next((i for i, (s, e) in enumerate(offsets) if s <= match.start() < e), None)
    if tok is None:
        raise ValueError(f"trigger {trigger!r} fell outside the tokenised span (truncated)")
    return tok


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

    def encode_trigger_reps(encoder, tokenizer, nodes, doc_text, max_length, device="cpu"):
        """Per-node trigger representation: encode each sentence once, pool the token
        covering the trigger. Gradient flows when the encoder is in train mode --
        the caller wraps inference in `no_grad`. Fail-fast on unlocatable triggers.
        """
        lines = doc_text.split("\n")
        by_sent: dict[int, list] = {}
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
            enc = tokenizer(
                sentence,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            inputs = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}
            hidden = encoder(**inputs).last_hidden_state[0]
            for node in group:
                embs[node.event_id] = hidden[locate_trigger_token(sentence, node.trigger, offsets)]
        return embs


@relation_extractors.register("supervised")
class SupervisedRelationExtractor(RelationExtractor):
    """Labels every document-level candidate mention pair via per-family heads.

    The encoder + heads are torch-backed and loaded lazily on the first `extract`;
    `__init__` stays torch-free so the pipeline instantiates on CPU.

    `max_length` is 512 because MAVEN-ERE's longest sentence is 1691 chars = 322 BPE
    tokens: at 256 such a sentence is truncated before its trigger, and
    `locate_trigger_token` (correctly) refuses to pool a wrong token instead.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        max_distance: int | None = None,
        max_length: int = 512,
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
        counts = {fam: len(subs) for fam, subs in FAMILY_SUBTYPES.items()}
        self._model = PairClassifier(self._encoder.config.hidden_size, counts)
        self._model.load_state_dict(torch.load(heads_file, map_location="cpu"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder.to(self._device).eval()
        self._model.to(self._device).eval()

    def _encode_mentions(self, nodes: list[EventNode], doc_text: str) -> dict[str, torch.Tensor]:
        """Trigger reps for inference: the shared encoder pooling under `no_grad`."""
        with torch.no_grad():
            return encode_trigger_reps(
                self._encoder, self._tokenizer, nodes, doc_text, self.max_length, self._device
            )

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
        for family, subtypes in FAMILY_SUBTYPES.items():
            probs = torch.softmax(logits[family], dim=-1)
            conf, idx = probs.max(dim=-1)
            for pair, i, c in zip(pairs, idx.tolist(), conf.tolist(), strict=True):
                if i == 0:  # NONE
                    continue
                result.setdefault(pair, {})[family] = (subtypes[i], float(c))
        return result
