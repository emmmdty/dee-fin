"""Serialization helpers for the frozen cross-stage contract.

All on-disk exchange uses JSON Lines. `load_event_nodes` / `dump_event_nodes`
and the `EventGraph` (de)serializers are the canonical readers/writers; every
data loader normalizes into these so the rest of the pipeline is format-blind.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ekg.core.schema import EventGraph, EventNode

__all__ = [
    "read_jsonl",
    "write_jsonl",
    "load_event_nodes",
    "dump_event_nodes",
    "load_event_graph",
    "dump_event_graph",
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
