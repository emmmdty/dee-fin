"""Serialization helpers and the financial-application SARGE adapter.

All on-disk exchange uses JSON Lines. `load_event_nodes` / `dump_event_nodes`
and the `EventGraph` (de)serializers are the canonical readers/writers; every
data loader normalizes into these so the rest of the pipeline is format-blind.

`event_nodes_from_sarge` adapts SARGE's canonical event records into `EventNode`s
for Phase G. SARGE is not the v4 Ch1 implementation; this remains a field-mapping
shim for the financial application and historical artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from finekg.core.schema import EventGraph, EventNode, EvidenceSpan

__all__ = [
    "read_jsonl",
    "write_jsonl",
    "load_event_nodes",
    "dump_event_nodes",
    "load_event_graph",
    "dump_event_graph",
    "event_nodes_from_sarge",
]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed object per non-empty line."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write objects as JSON Lines; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_event_nodes(path: str | Path) -> list[EventNode]:
    return [EventNode.model_validate(row) for row in read_jsonl(path)]


def dump_event_nodes(path: str | Path, nodes: Iterable[EventNode]) -> int:
    return write_jsonl(path, (n.model_dump(mode="json") for n in nodes))


def load_event_graph(path: str | Path) -> EventGraph:
    return EventGraph.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def dump_event_graph(path: str | Path, graph: EventGraph) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _spans_from_sarge(raw: Any, doc_id: str) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        spans.append(
            EvidenceSpan(
                doc_id=item.get("doc_id", doc_id),
                char_start=int(item.get("char_start", 0)),
                char_end=int(item.get("char_end", item.get("char_start", 0))),
                sent_id=item.get("sent_id"),
                text=item.get("text", ""),
            )
        )
    return spans


def event_nodes_from_sarge(records: Iterable[dict[str, Any]]) -> list[EventNode]:
    """Adapt SARGE canonical event records into `EventNode`s.

    Expected per-record keys (missing keys degrade gracefully): `event_id`,
    `event_type`, `doc_id`, `arguments` (role->value), `argument_evidence`
    (role->list-of-spans), `trigger`, `time_anchor`, `subject`, `metadata`.

    `scripts/sarge_to_event_nodes.py` writes `argument_evidence`; the legacy
    `evidence` spelling is still read so older dumps keep loading. Reading only
    the legacy key silently emptied the evidence of every node produced by that
    script, which `--from-sarge` graph builds then treated as ungrounded.
    """
    nodes: list[EventNode] = []
    for i, rec in enumerate(records):
        doc_id = str(rec.get("doc_id", ""))
        event_id = str(rec.get("event_id") or f"{doc_id}::evt{i}")
        evidence_map = rec.get("argument_evidence") or rec.get("evidence") or {}
        argument_evidence = {
            role: _spans_from_sarge(spans, doc_id) for role, spans in evidence_map.items()
        }
        nodes.append(
            EventNode(
                event_id=event_id,
                event_type=str(rec.get("event_type", "Unknown")),
                doc_id=doc_id,
                trigger=str(rec.get("trigger", "")),
                trigger_evidence=_spans_from_sarge(rec.get("trigger_evidence"), doc_id),
                arguments={k: str(v) for k, v in (rec.get("arguments", {}) or {}).items()},
                argument_evidence=argument_evidence,
                time_anchor=rec.get("time_anchor"),
                subject=rec.get("subject"),
                confidence=float(rec.get("confidence", 1.0)),
                metadata={str(k): str(v) for k, v in (rec.get("metadata") or {}).items()},
            )
        )
    return nodes
