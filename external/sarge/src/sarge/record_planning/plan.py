from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sarge.data.schema import DatasetSchema
from sarge.postprocess.rule_planner import ANCHOR_ROLES_BY_EVENT_TYPE


@dataclass(frozen=True)
class RecordPlanInstance:
    record_id: str
    event_type: str
    anchors: dict[str, list[str]]

    @property
    def anchor_signature(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (role, tuple(_normalize_text(value) for value in values if _normalize_text(value)))
            for role, values in self.anchors.items()
            if values
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "event_type": self.event_type,
            "anchors": {role: list(values) for role, values in self.anchors.items() if values},
        }


def build_record_plan(events: list[dict[str, Any]], schema: DatasetSchema) -> list[RecordPlanInstance]:
    """Build deterministic record-instance plans from gold event records.

    The plan is a supervision target, not an evaluator artifact. For each
    same-document event group, it selects anchor roles that make same-type
    records distinguishable when possible, then assigns stable record ids.
    """

    validated_events = [_validated_event(event, schema) for event in events]
    positions_by_event_type: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(validated_events):
        positions_by_event_type[str(event["event_type"])].append(index)

    anchor_roles_by_position: dict[int, tuple[str, ...]] = {}
    for event_type, positions in positions_by_event_type.items():
        group = [validated_events[index] for index in positions]
        selected_roles = _select_anchor_roles(event_type, group, schema)
        for index in positions:
            anchor_roles_by_position[index] = selected_roles

    plan: list[RecordPlanInstance] = []
    for index, event in enumerate(validated_events, 1):
        event_type = str(event["event_type"])
        anchors = _anchors_for_event(event, anchor_roles_by_position.get(index - 1, ()))
        plan.append(RecordPlanInstance(record_id=f"R{index}", event_type=event_type, anchors=anchors))
    return plan


def events_to_record_plan_output(events: list[dict[str, Any]], schema: DatasetSchema) -> dict[str, Any]:
    plan = build_record_plan(events, schema)
    planned_events = []
    for instance, event in zip(plan, events, strict=True):
        validated = _validated_event(event, schema)
        planned_events.append(
            {
                "record_id": instance.record_id,
                "event_type": instance.event_type,
                "arguments": _minimal_text_arguments(validated, schema),
            }
        )
    return {
        "record_plan": [instance.to_dict() for instance in plan],
        "events": planned_events,
    }


def _validated_event(event: dict[str, Any], schema: DatasetSchema) -> dict[str, Any]:
    schema.validate_event_record(event)
    event_type = schema.validate_event_type(str(event.get("event_type") or ""))
    arguments = event.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"event_type": event_type, "arguments": arguments}


def _select_anchor_roles(
    event_type: str,
    events: list[dict[str, Any]],
    schema: DatasetSchema,
) -> tuple[str, ...]:
    role_priority = _role_priority(event_type, schema)
    if len(events) <= 1:
        for role in role_priority:
            if any(_role_values(event, role) for event in events):
                return (role,)
        return ()

    selected: list[str] = []
    best_distinct_count = 0
    for role in role_priority:
        if not any(_role_values(event, role) for event in events):
            continue
        candidate = [*selected, role]
        signatures = [_signature_for_roles(event, candidate) for event in events]
        distinct_count = len(set(signatures))
        if distinct_count > best_distinct_count:
            selected.append(role)
            best_distinct_count = distinct_count
        if _all_non_empty_and_unique(signatures):
            break

    if selected:
        return tuple(selected)
    return tuple(role for role in role_priority if any(_role_values(event, role) for event in events))


def _role_priority(event_type: str, schema: DatasetSchema) -> tuple[str, ...]:
    configured = tuple(role for role in ANCHOR_ROLES_BY_EVENT_TYPE.get(event_type, ()) if role in schema.event_roles[event_type])
    remainder = tuple(role for role in schema.event_roles[event_type] if role not in configured)
    return (*configured, *remainder)


def _anchors_for_event(event: dict[str, Any], selected_roles: tuple[str, ...]) -> dict[str, list[str]]:
    anchors: dict[str, list[str]] = {}
    for role in selected_roles:
        values = _role_values(event, role)
        if values:
            anchors[role] = values
    return anchors


def _minimal_text_arguments(event: dict[str, Any], schema: DatasetSchema) -> dict[str, list[str]]:
    event_type = str(event["event_type"])
    arguments = event.get("arguments") or {}
    output: dict[str, list[str]] = {}
    for role in schema.event_roles[event_type]:
        values = _role_values(event, role)
        if values:
            output[role] = values
    return output


def _signature_for_roles(event: dict[str, Any], roles: list[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((role, tuple(_role_values(event, role))) for role in roles if _role_values(event, role))


def _all_non_empty_and_unique(signatures: list[tuple[tuple[str, tuple[str, ...]], ...]]) -> bool:
    return all(signatures) and len(set(signatures)) == len(signatures)


def _role_values(event: dict[str, Any], role: str) -> list[str]:
    arguments = event.get("arguments") or {}
    values = arguments.get(role) or []
    normalized = []
    seen = set()
    for value in values:
        if isinstance(value, dict):
            text = _normalize_text(value.get("text") or value.get("argument") or "")
        else:
            text = _normalize_text(value)
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())
