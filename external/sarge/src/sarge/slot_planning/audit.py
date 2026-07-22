from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sarge.data.schema import DatasetSchema
from sarge.slot_planning.plan import FORBIDDEN_SLOT_PLAN_KEYS, SlotPlanDocument, slot_plan_to_dict, validate_slot_plan


def audit_slot_plans(plans: Iterable[SlotPlanDocument], schema: DatasetSchema) -> dict[str, Any]:
    plan_list = list(plans)
    invalid_plans: list[dict[str, str]] = []
    forbidden_key_violations: list[dict[str, str]] = []
    total_slots = 0
    for plan in plan_list:
        total_slots += len(plan.slots)
        try:
            validate_slot_plan(plan, schema)
        except ValueError as exc:
            invalid_plans.append({"doc_id": plan.doc_id, "error": str(exc)})
        for key in sorted(_find_forbidden_keys(slot_plan_to_dict(plan))):
            forbidden_key_violations.append({"doc_id": plan.doc_id, "key": key})

    return {
        "document_count": len(plan_list),
        "slot_count_total": total_slots,
        "slot_count_per_doc": {plan.doc_id: len(plan.slots) for plan in plan_list},
        "slots_per_doc": (total_slots / len(plan_list)) if plan_list else None,
        "invalid_plan_count": len(invalid_plans),
        "invalid_plans": invalid_plans,
        "forbidden_key_violation_count": len(forbidden_key_violations),
        "forbidden_key_violations": forbidden_key_violations,
    }


def _find_forbidden_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_SLOT_PLAN_KEYS:
                keys.add(str(key))
            keys.update(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_find_forbidden_keys(child))
    return keys
