"""Loader for MAVEN-ERE (event coreference / temporal / causal / subevent).

Normalizes the official MAVEN-ERE JSON-Lines format into our contracts:

- nodes are event *mentions* (so coreference is a real clustering task);
- coreference gold = all within-event mention pairs (each event is a cluster);
- temporal / causal / subevent gold = relations between events, attached to each
  event's representative (first) mention.

The loader is tolerant of missing keys so partial fixtures still parse. The
official dataset is fetched by `scripts/download_datasets.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from finekg.core.io import read_jsonl
from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType

__all__ = ["RelationDocument", "load_maven_ere"]


@dataclass
class RelationDocument:
    doc_id: str
    nodes: list[EventNode]
    gold_edges: list[RelationEdge]
    doc_text: str = ""
    representative: dict[str, str] = field(default_factory=dict)  # event_id -> mention_id


def _mention_span(
    doc_id: str, mention: dict, lines: list[str], line_starts: list[int]
) -> list[EvidenceSpan]:
    """Character span of the trigger inside the canonical doc text.

    The official `offset` field is *token*-based, so it must not be stored as
    character positions; instead the trigger word is located in its `sent_id`
    line of the doc text. An unlocatable trigger gets an unpositioned span
    (start == end == 0) whose `text` still supports quote-based grounding.
    """
    sent_id = mention.get("sent_id")
    sid = sent_id if isinstance(sent_id, int) and sent_id >= 0 else None
    trigger = str(mention.get("trigger_word", ""))
    char_start = char_end = 0
    if sid is not None and sid < len(lines) and trigger:
        pos = lines[sid].find(trigger)
        if pos >= 0:
            char_start = line_starts[sid] + pos
            char_end = char_start + len(trigger)
    return [
        EvidenceSpan(
            doc_id=doc_id,
            char_start=char_start,
            char_end=char_end,
            sent_id=sid,
            text=trigger,
        )
    ]


def _doc_text(record: dict) -> str:
    """Best-effort raw text: `text`, joined `sentences`, or joined `tokens`.

    Needed by evidence grounding (quote-in-text checks); absent fields degrade
    to "" so partial fixtures still parse.
    """
    if record.get("text"):
        return str(record["text"])
    sentences = record.get("sentences")
    if sentences:
        return "\n".join(str(s) for s in sentences)
    tokens = record.get("tokens")
    if tokens:
        return "\n".join(" ".join(str(t) for t in sent) for sent in tokens)
    return ""


def _parse_document(record: dict) -> RelationDocument:
    doc_id = str(record.get("id", record.get("doc_id", "")))
    nodes: list[EventNode] = []
    gold: list[RelationEdge] = []
    representative: dict[str, str] = {}

    # Canonical text first: mention spans are located inside it, so the char
    # offsets are valid for the same `doc_text` the document carries.
    doc_text = _doc_text(record)
    lines = doc_text.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)

    for event in record.get("events", []):
        event_id = str(event.get("id"))
        mentions = event.get("mention") or event.get("mentions") or []
        mention_ids: list[str] = []
        for m in mentions:
            # Namespace mention ids by document for corpus-level uniqueness.
            mid = f"{doc_id}::{m.get('id')}"
            mention_ids.append(mid)
            nodes.append(
                EventNode(
                    event_id=mid,
                    event_type=str(event.get("type", "Unknown")),
                    doc_id=doc_id,
                    trigger=m.get("trigger_word", ""),
                    trigger_evidence=_mention_span(doc_id, m, lines, line_starts),
                    metadata={"event": event_id},
                )
            )
        if mention_ids:
            representative[event_id] = mention_ids[0]
        # Coreference gold: every within-event mention pair.
        for a, b in combinations(mention_ids, 2):
            gold.append(
                RelationEdge(
                    head_id=a, tail_id=b, relation_type=RelationType.COREFERENCE, directed=False
                )
            )

    def rep(eid: str) -> str | None:
        return representative.get(str(eid))

    for subtype, pairs in (record.get("temporal_relations") or {}).items():
        for h, t in pairs:
            rh, rt = rep(h), rep(t)
            if rh and rt:
                gold.append(
                    RelationEdge(
                        head_id=rh,
                        tail_id=rt,
                        relation_type=RelationType.TEMPORAL,
                        subtype=str(subtype).upper(),
                    )
                )
    for subtype, pairs in (record.get("causal_relations") or {}).items():
        for h, t in pairs:
            rh, rt = rep(h), rep(t)
            if rh and rt:
                gold.append(
                    RelationEdge(
                        head_id=rh,
                        tail_id=rt,
                        relation_type=RelationType.CAUSAL,
                        subtype=str(subtype).upper(),
                    )
                )
    for pair in record.get("subevent_relations") or []:
        rh, rt = rep(pair[0]), rep(pair[1])
        if rh and rt:
            gold.append(
                RelationEdge(
                    head_id=rh,
                    tail_id=rt,
                    relation_type=RelationType.SUBEVENT,
                    subtype="SUBEVENT_OF",
                )
            )

    return RelationDocument(
        doc_id=doc_id,
        nodes=nodes,
        gold_edges=gold,
        doc_text=doc_text,
        representative=representative,
    )


def load_maven_ere(path: str | Path) -> Iterator[RelationDocument]:
    """Yield one `RelationDocument` per line of a MAVEN-ERE jsonl file."""
    for record in read_jsonl(path):
        yield _parse_document(record)
