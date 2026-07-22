from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from sarge.data.canonical import CanonicalEventRecord, CanonicalEventRecordDict


class GeneratedCandidateSetDict(TypedDict, total=False):
    candidate_id: str
    doc_id: str
    events: list[CanonicalEventRecordDict]
    parse_status: str
    generation_score: float
    slot_plan_ids: list[str]
    diagnostics: dict[str, object]


class MRSFeatureVector(TypedDict, total=False):
    candidate_id: str
    doc_id: str
    features: dict[str, float]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class GeneratedCandidateSet:
    candidate_id: str
    doc_id: str
    events: list[CanonicalEventRecord] = field(default_factory=list)
    parse_status: str = "unknown"
    generation_score: float | None = None
    slot_plan_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

