from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sarge.surface_memory.types import SurfaceCandidate
from sarge.data.schema import DatasetSchema
from sarge.slot_planning.plan import SlotPlanDocument

MOCK_GETM_MODES = frozenset({"empty", "schema_only", "echo_candidates"})


@dataclass(frozen=True)
class MockGetmBackend:
    mode: str = "empty"

    def __post_init__(self) -> None:
        if self.mode not in MOCK_GETM_MODES:
            raise ValueError(f"mock GETM mode must be one of {sorted(MOCK_GETM_MODES)}; got {self.mode!r}")

    def generate_one(
        self,
        *,
        prompt: str,
        document: Any,
        schema: DatasetSchema,
        surface_candidates: list[SurfaceCandidate],
        slot_plan: SlotPlanDocument,
        candidate_index: int,
    ) -> str:
        del prompt, document, candidate_index
        if self.mode == "empty":
            payload = {"events": []}
        elif self.mode == "schema_only":
            payload = {"events": _slot_events(schema, slot_plan, surface_candidates, echo=False)}
        else:
            payload = {"events": _slot_events(schema, slot_plan, surface_candidates, echo=True)}
        return json.dumps(payload, ensure_ascii=False)


def _slot_events(
    schema: DatasetSchema,
    slot_plan: SlotPlanDocument,
    surface_candidates: list[SurfaceCandidate],
    *,
    echo: bool,
) -> list[dict[str, Any]]:
    if slot_plan.slots:
        return [
            _event_for_slot(
                schema,
                slot.event_type,
                slot.slot_id,
                slot.role_prior,
                slot.supporting_candidates,
                surface_candidates,
                echo=echo,
            )
            for slot in slot_plan.slots
        ]
    if not echo or not surface_candidates or not schema.event_roles:
        return []
    event_type = next(iter(schema.event_roles))
    role = schema.event_roles[event_type][0]
    candidate = surface_candidates[0]
    return [
        {
            "event_type": event_type,
            "slot_id": 0,
            "arguments": {role: [{"text": candidate.surface, "source_candidate_id": candidate.candidate_id}]},
        }
    ]


def _event_for_slot(
    schema: DatasetSchema,
    event_type: str,
    slot_id: int,
    role_prior: dict[str, float],
    supporting_candidate_ids: list[str],
    surface_candidates: list[SurfaceCandidate],
    *,
    echo: bool,
) -> dict[str, Any]:
    schema.validate_event_type(event_type)
    event: dict[str, Any] = {"event_type": event_type, "slot_id": slot_id, "arguments": {}}
    if not echo:
        return event

    candidates_by_id = {candidate.candidate_id: candidate for candidate in surface_candidates}
    ordered_candidates = [
        candidates_by_id[candidate_id]
        for candidate_id in supporting_candidate_ids
        if candidate_id in candidates_by_id
    ]
    ordered_candidates.extend(
        candidate
        for candidate in surface_candidates
        if candidate.candidate_id not in {item.candidate_id for item in ordered_candidates}
    )
    if not ordered_candidates:
        return event

    role = _best_role(schema, event_type, role_prior)
    candidate = ordered_candidates[0]
    event["arguments"] = {
        role: [{"text": candidate.surface, "source_candidate_id": candidate.candidate_id}]
    }
    return event


def _best_role(schema: DatasetSchema, event_type: str, role_prior: dict[str, float]) -> str:
    roles = schema.event_roles[event_type]
    if role_prior:
        role, _ = max(role_prior.items(), key=lambda item: (float(item[1]), item[0]))
        if role in roles:
            return role
    return roles[0]
