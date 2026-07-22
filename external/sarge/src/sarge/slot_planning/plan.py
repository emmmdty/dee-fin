from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sarge.data.schema import DatasetSchema

FORBIDDEN_SLOT_PLAN_KEYS = frozenset(
    {
        "gold",
        "events",
        "arguments",
        "argument_text",
        "text",
        "norm_text",
        "content",
        "content_raw",
        "surface",
        "context",
    }
)


@dataclass(frozen=True)
class EventSlot:
    event_type: str
    slot_id: int
    count_confidence: float
    role_prior: dict[str, float] = field(default_factory=dict)
    supporting_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "slot_id": self.slot_id,
            "count_confidence": float(self.count_confidence),
            "role_prior": dict(self.role_prior),
            "supporting_candidates": list(self.supporting_candidates),
        }


@dataclass(frozen=True)
class SlotPlanDocument:
    doc_id: str
    dataset: str
    slots: list[EventSlot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return slot_plan_to_dict(self)


def slot_plan_to_dict(plan: SlotPlanDocument) -> dict[str, Any]:
    return {
        "doc_id": plan.doc_id,
        "dataset": plan.dataset,
        "slots": [slot.to_dict() for slot in plan.slots],
    }


def slot_plan_from_dict(payload: dict[str, Any]) -> SlotPlanDocument:
    slots = []
    for raw_slot in payload.get("slots") or []:
        if not isinstance(raw_slot, dict):
            raise ValueError("slot entries must be mappings")
        slots.append(
            EventSlot(
                event_type=str(raw_slot.get("event_type", "")),
                slot_id=int(raw_slot.get("slot_id", 0)),
                count_confidence=float(raw_slot.get("count_confidence", 0.0)),
                role_prior={str(role): float(score) for role, score in (raw_slot.get("role_prior") or {}).items()},
                supporting_candidates=[str(candidate) for candidate in (raw_slot.get("supporting_candidates") or [])],
            )
        )
    return SlotPlanDocument(
        doc_id=str(payload.get("doc_id", "")),
        dataset=str(payload.get("dataset", "")),
        slots=slots,
    )


def validate_slot_plan(plan: SlotPlanDocument, schema: DatasetSchema) -> None:
    if not plan.doc_id:
        raise ValueError("slot plan missing doc_id")
    if plan.dataset != schema.dataset_id:
        raise ValueError(f"slot plan dataset must be {schema.dataset_id!r}; got {plan.dataset!r}")
    _validate_no_forbidden_keys(slot_plan_to_dict(plan))
    seen_slots: set[tuple[str, int]] = set()
    for slot in plan.slots:
        event_type = schema.validate_event_type(slot.event_type)
        if slot.slot_id < 0:
            raise ValueError(f"slot_id must be non-negative: {event_type}/{slot.slot_id}")
        slot_key = (event_type, slot.slot_id)
        if slot_key in seen_slots:
            raise ValueError(f"duplicate slot: {event_type}/{slot.slot_id}")
        seen_slots.add(slot_key)
        if not 0.0 <= slot.count_confidence <= 1.0:
            raise ValueError(f"count_confidence must be between 0 and 1: {event_type}/{slot.slot_id}")
        for role, score in slot.role_prior.items():
            schema.validate_role(event_type, role)
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"role_prior score must be between 0 and 1: {event_type}/{role}")


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_SLOT_PLAN_KEYS:
                raise ValueError(f"slot plan contains forbidden key: {key}")
            _validate_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _validate_no_forbidden_keys(child)
