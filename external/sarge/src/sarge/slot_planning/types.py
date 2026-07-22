from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class EventSlotPlanDict(TypedDict, total=False):
    doc_id: str
    event_type: str
    slot_id: int
    count_confidence: float
    role_prior: dict[str, float]
    supporting_candidate_ids: list[str]
    metadata: dict[str, object]


@dataclass(frozen=True)
class EventSlotPlan:
    doc_id: str
    event_type: str
    slot_id: int
    count_confidence: float
    role_prior: dict[str, float] = field(default_factory=dict)
    supporting_candidate_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

