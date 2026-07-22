"""GRPO prompt dataset construction from relation documents (pure CPU).

Documents are split into event windows of at most `window_events` nodes — one
prompt per window — which bounds prompt length and doubles as the curriculum
difficulty axis (difficulty = number of events in the window). Gold edges are
restricted to pairs inside the same window so the task reward stays exact.

The reward functions need the source document back at scoring time, but TRL
dataset columns must stay JSON-simple; `DocStore` resolves the `doc_key`
column to the windowed `RelationDocument`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.llm import build_relation_prompt

__all__ = ["GrpoSample", "DocStore", "window_document", "build_grpo_dataset", "to_rows"]


@dataclass(frozen=True)
class GrpoSample:
    prompt: str
    doc_key: str
    difficulty: float


class DocStore:
    """In-memory doc_key -> windowed document resolver for reward functions."""

    def __init__(self) -> None:
        self._docs: dict[str, RelationDocument] = {}

    def add(self, doc: RelationDocument) -> str:
        if doc.doc_id in self._docs:
            raise ValueError(f"duplicate doc_key {doc.doc_id!r}")
        self._docs[doc.doc_id] = doc
        return doc.doc_id

    def get(self, key: str) -> RelationDocument:
        if key not in self._docs:
            raise KeyError(f"unknown doc_key {key!r}; was the store built from this dataset?")
        return self._docs[key]

    def __len__(self) -> int:
        return len(self._docs)


def window_document(doc: RelationDocument, window_events: int) -> list[RelationDocument]:
    """Split a document into event windows with window-local gold edges.

    Windows with fewer than two events are skipped — no relations to extract.
    """
    if window_events < 2:
        raise ValueError("window_events must be at least 2")
    windows: list[RelationDocument] = []
    for j, start in enumerate(range(0, len(doc.nodes), window_events)):
        nodes = doc.nodes[start : start + window_events]
        if len(nodes) < 2:
            continue
        ids = {n.event_id for n in nodes}
        gold = [e for e in doc.gold_edges if e.head_id in ids and e.tail_id in ids]
        windows.append(
            RelationDocument(
                doc_id=f"{doc.doc_id}#w{j}",
                nodes=nodes,
                gold_edges=gold,
                doc_text=doc.doc_text,
            )
        )
    return windows


def build_grpo_dataset(
    docs: Iterable[RelationDocument], window_events: int = 12
) -> tuple[list[GrpoSample], DocStore]:
    """One GRPO sample per event window, plus the store resolving its document."""
    store = DocStore()
    samples: list[GrpoSample] = []
    for doc in docs:
        for window in window_document(doc, window_events):
            key = store.add(window)
            samples.append(
                GrpoSample(
                    # The document text rides along: without it the model has
                    # nothing to quote and the grounding reward is unsatisfiable.
                    prompt=build_relation_prompt(window.nodes, doc_text=window.doc_text),
                    doc_key=key,
                    difficulty=float(len(window.nodes)),
                )
            )
    return samples, store


def to_rows(samples: Iterable[GrpoSample]) -> list[dict]:
    """Rows for `datasets.Dataset.from_list` — extra columns reach the reward fn."""
    return [
        {"prompt": s.prompt, "doc_key": s.doc_key, "difficulty": s.difficulty} for s in samples
    ]
