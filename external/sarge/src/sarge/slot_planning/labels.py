from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sarge.data.loader import V2DatasetDocument
from sarge.data.schema import DatasetSchema

COUNT_BUCKETS = ("0", "1", "2", "3+")


@dataclass(frozen=True)
class EventTypeSlotLabel:
    event_type: str
    presence: bool
    event_count: int
    count_bucket: str
    role_occupancy: dict[str, bool] = field(default_factory=dict)
    same_type_multi_event: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "presence": self.presence,
            "event_count": self.event_count,
            "count_bucket": self.count_bucket,
            "role_occupancy": dict(self.role_occupancy),
            "same_type_multi_event": self.same_type_multi_event,
        }


@dataclass(frozen=True)
class RecordSlotLabel:
    event_type: str
    slot_id: int
    role_occupancy: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "slot_id": self.slot_id,
            "role_occupancy": dict(self.role_occupancy),
        }


@dataclass(frozen=True)
class SlotLabelDocument:
    doc_id: str
    dataset: str
    split: str
    event_type_labels: list[EventTypeSlotLabel] = field(default_factory=list)
    record_slot_labels: list[RecordSlotLabel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "dataset": self.dataset,
            "split": self.split,
            "event_type_labels": [label.to_dict() for label in self.event_type_labels],
            "record_slot_labels": [label.to_dict() for label in self.record_slot_labels],
        }


def count_bucket(event_count: int) -> str:
    count = int(event_count)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def derive_slot_labels(document: V2DatasetDocument, schema: DatasetSchema) -> SlotLabelDocument:
    if document.gold is None:
        raise ValueError("slot labels require a gold-visible document")

    events = document.gold.events
    event_counts = Counter(str(event.get("event_type", "")).strip() for event in events)
    role_occupancy_by_event_type: dict[str, dict[str, bool]] = {
        event_type: {role: False for role in roles} for event_type, roles in schema.event_roles.items()
    }
    record_slot_labels: list[RecordSlotLabel] = []
    next_slot_id: defaultdict[str, int] = defaultdict(int)

    for event in events:
        event_type = schema.validate_event_type(str(event.get("event_type", "")))
        slot_id = next_slot_id[event_type]
        next_slot_id[event_type] += 1
        record_role_occupancy = _role_occupancy(event, schema, event_type)
        for role, occupied in record_role_occupancy.items():
            role_occupancy_by_event_type[event_type][role] = role_occupancy_by_event_type[event_type][role] or occupied
        record_slot_labels.append(
            RecordSlotLabel(event_type=event_type, slot_id=slot_id, role_occupancy=record_role_occupancy)
        )

    event_type_labels = []
    for event_type, roles in schema.event_roles.items():
        event_count = int(event_counts.get(event_type, 0))
        event_type_labels.append(
            EventTypeSlotLabel(
                event_type=event_type,
                presence=event_count > 0,
                event_count=event_count,
                count_bucket=count_bucket(event_count),
                role_occupancy={role: role_occupancy_by_event_type[event_type][role] for role in roles},
                same_type_multi_event=event_count > 1,
            )
        )
    return SlotLabelDocument(
        doc_id=document.doc_id,
        dataset=document.input.dataset_id,
        split=document.input.split,
        event_type_labels=event_type_labels,
        record_slot_labels=record_slot_labels,
    )


def _role_occupancy(event: dict[str, Any], schema: DatasetSchema, event_type: str) -> dict[str, bool]:
    arguments = event.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    occupancy: dict[str, bool] = {}
    for role in schema.event_roles[event_type]:
        values = arguments.get(role) or []
        occupancy[role] = isinstance(values, list) and len(values) > 0
    return occupancy
