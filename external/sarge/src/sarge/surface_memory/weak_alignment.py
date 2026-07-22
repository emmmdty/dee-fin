from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from sarge.surface_memory.types import SurfaceMemory
from sarge.data.loader import V2DatasetDocument

AlignmentStatus = Literal["located", "unlocated"]


@dataclass(frozen=True)
class GoldArgument:
    doc_id: str
    event_index: int
    argument_index: int
    event_type: str
    role: str
    text: str


@dataclass(frozen=True)
class WeakAlignmentRecord:
    doc_id: str
    event_index: int
    argument_index: int
    event_type: str
    role: str
    argument_text: str
    status: AlignmentStatus
    candidate_ids: list[str] = field(default_factory=list)
    match_count: int = 0
    ambiguous: bool = False
    content_match_count: int = 0
    candidate_match_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "event_index": self.event_index,
            "argument_index": self.argument_index,
            "event_type": self.event_type,
            "role": self.role,
            "argument_text": self.argument_text,
            "status": self.status,
            "candidate_ids": list(self.candidate_ids),
            "match_count": self.match_count,
            "ambiguous": self.ambiguous,
            "content_match_count": self.content_match_count,
            "candidate_match_count": self.candidate_match_count,
        }


def align_gold_arguments(document: V2DatasetDocument, memory: SurfaceMemory) -> list[WeakAlignmentRecord]:
    if document.gold is None:
        raise ValueError("weak alignment requires gold-visible train/dev documents")
    if document.doc_id != memory.doc_id:
        raise ValueError(f"document and surface memory doc_id mismatch: {document.doc_id!r} != {memory.doc_id!r}")

    records: list[WeakAlignmentRecord] = []
    for argument in iter_gold_arguments(document):
        candidate_ids = [
            candidate.candidate_id for candidate in memory.candidates if candidate.surface.strip() == argument.text
        ]
        content_match_count = _content_match_count(document, argument.text)
        candidate_match_count = len(candidate_ids)
        located = bool(candidate_match_count or content_match_count)
        match_count = candidate_match_count if candidate_match_count else content_match_count
        records.append(
            WeakAlignmentRecord(
                doc_id=argument.doc_id,
                event_index=argument.event_index,
                argument_index=argument.argument_index,
                event_type=argument.event_type,
                role=argument.role,
                argument_text=argument.text,
                status="located" if located else "unlocated",
                candidate_ids=candidate_ids,
                match_count=match_count,
                ambiguous=candidate_match_count > 1 or content_match_count > 1,
                content_match_count=content_match_count,
                candidate_match_count=candidate_match_count,
            )
        )
    return records


def iter_gold_arguments(document: V2DatasetDocument) -> Iterable[GoldArgument]:
    if document.gold is None:
        return
    argument_index = 0
    for event_index, event in enumerate(document.gold.events):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", "")).strip()
        arguments = event.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for role, values in arguments.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                text = str(value.get("text") or "").strip()
                if not text:
                    continue
                yield GoldArgument(
                    doc_id=document.doc_id,
                    event_index=event_index,
                    argument_index=argument_index,
                    event_type=event_type,
                    role=str(role),
                    text=text,
                )
                argument_index += 1


def _content_match_count(document: V2DatasetDocument, text: str) -> int:
    seen_sources: set[str] = set()
    count = 0
    for source in (document.input.content, document.input.content_raw):
        if source is None or source in seen_sources:
            continue
        seen_sources.add(source)
        count += len(_strict_find_all(source, text))
    return count


def _strict_find_all(content: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    position = content.find(needle)
    while position >= 0:
        starts.append(position)
        position = content.find(needle, position + 1)
    return starts
