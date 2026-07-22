"""Loader for the CCKS financial event-causality data (Chinese grounding set).

The CCKS-2021 financial causality task gives, per text, cause/effect event pairs
where each event has a type and elements (region / product / industry). We
normalize each pair into two `EventNode`s and one grounded CAUSAL edge, so the
same relation pipeline and metrics apply to the Chinese financial setting.

The loader expects a normalized JSON-Lines form (produced by the download/prep
script from the official release):

    {"text_id": "...", "text": "...",
     "causal_pairs": [{"cause": {"type": ..., "region": ..., "product": ..., "industry": ...},
                       "effect": {...},
                       "cause_span": [s, e], "effect_span": [s, e]}]}
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from finekg.core.io import read_jsonl
from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument

__all__ = ["load_ccks_causal"]


def _event_node(doc_id: str, uid: str, tag: str, spec: dict, span: list[int]) -> EventNode:
    s, e = (span + [0, 0])[:2]
    args = {k: str(v) for k, v in spec.items() if k != "type" and v}
    return EventNode(
        event_id=f"{uid}::{tag}",
        event_type=str(spec.get("type", "Unknown")),
        doc_id=doc_id,
        arguments=args,
        trigger_evidence=[EvidenceSpan(doc_id=doc_id, char_start=int(s), char_end=int(e))],
    )


def _parse(record: dict) -> RelationDocument:
    text_id = str(record.get("text_id", record.get("id", "")))
    nodes: list[EventNode] = []
    gold: list[RelationEdge] = []
    for i, pair in enumerate(record.get("causal_pairs", [])):
        uid = f"{text_id}#{i}"
        cause = _event_node(
            text_id, uid, "cause", pair.get("cause", {}), pair.get("cause_span", [])
        )
        effect = _event_node(
            text_id, uid, "effect", pair.get("effect", {}), pair.get("effect_span", [])
        )
        nodes.extend([cause, effect])
        gold.append(
            RelationEdge(
                head_id=cause.event_id,
                tail_id=effect.event_id,
                relation_type=RelationType.CAUSAL,
                subtype="CAUSE",
                evidence=cause.trigger_evidence + effect.trigger_evidence,
            )
        )
    return RelationDocument(
        doc_id=text_id, nodes=nodes, gold_edges=gold, doc_text=str(record.get("text", ""))
    )


def load_ccks_causal(path: str | Path) -> Iterator[RelationDocument]:
    """Yield one `RelationDocument` per line of the normalized CCKS causal file."""
    for record in read_jsonl(path):
        yield _parse(record)
