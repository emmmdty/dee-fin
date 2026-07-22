from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class SurfaceCandidateDict(TypedDict, total=False):
    candidate_id: str
    doc_id: str
    surface: str
    context: str
    chunk_id: str
    source: str
    char_start: int
    char_end: int
    event_type: str
    role: str
    role_score: float
    metadata: dict[str, object]


class SurfaceMemoryDict(TypedDict):
    doc_id: str
    candidates: list[SurfaceCandidateDict]
    source: str


@dataclass(frozen=True)
class SurfaceCandidate:
    candidate_id: str
    doc_id: str
    surface: str
    context: str
    chunk_id: str
    source: str = "rule"
    char_start: int | None = None
    char_end: int | None = None
    event_type: str | None = None
    role: str | None = None
    role_score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceMemory:
    doc_id: str
    candidates: list[SurfaceCandidate] = field(default_factory=list)
    source: str = "document_surface"
