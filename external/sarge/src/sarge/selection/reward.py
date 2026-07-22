from __future__ import annotations

from collections import Counter
from typing import Any

from sarge.data.loader import V2DatasetDocument
from sarge.data.schema import DatasetSchema

METRIC_SOURCE = "sarge.lightweight_surface_record_reward.v0"


def compute_reward_rows(
    candidates: list[dict[str, Any]],
    *,
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    lambda_record: float = 0.5,
) -> list[dict[str, Any]]:
    documents_by_doc = {document.doc_id: document for document in documents}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        doc_id = str(candidate.get("doc_id", ""))
        document = documents_by_doc.get(doc_id)
        if document is None:
            raise ValueError(f"candidate doc_id not found in gold-visible documents: {doc_id}")
        rows.append(
            compute_candidate_reward(
                candidate,
                document=document,
                schema=schema,
                lambda_record=lambda_record,
            )
        )
    return rows


def compute_candidate_reward(
    candidate: dict[str, Any],
    *,
    schema: DatasetSchema,
    document: V2DatasetDocument | None = None,
    gold_events: list[dict[str, Any]] | None = None,
    lambda_record: float = 0.5,
) -> dict[str, Any]:
    if gold_events is None:
        if document is None or document.gold is None:
            raise ValueError("MRS reward construction requires gold-visible train/dev documents")
        gold_events = document.gold.events

    candidate_events = _events(candidate)
    surface_f1 = _multiset_f1(_surface_units(candidate_events), _surface_units(gold_events))
    record_f1 = _multiset_f1(_record_units(candidate_events), _record_units(gold_events))
    penalties = _penalties(candidate, candidate_events=candidate_events, gold_events=gold_events, schema=schema)
    reward = surface_f1 + (lambda_record * record_f1) - sum(penalties.values())

    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "doc_id": str(candidate.get("doc_id") or (document.doc_id if document else "")),
        "candidate_index": _candidate_index(candidate),
        "reward": float(reward),
        "metric_source": METRIC_SOURCE,
        "uses_gold": True,
        "components": {
            "surface_f1": float(surface_f1),
            "record_f1": float(record_f1),
            "lambda_record": float(lambda_record),
            **{name: float(value) for name, value in penalties.items()},
        },
    }


def _events(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = candidate.get("events") or []
    return [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else []


def _surface_units(events: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    units: list[tuple[str, str, str]] = []
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        arguments = event.get("arguments") or {}
        if not event_type or not isinstance(arguments, dict):
            continue
        for role, values in arguments.items():
            role_name = str(role).strip()
            if not role_name:
                continue
            raw_values = values if isinstance(values, list) else [values]
            for value in raw_values:
                text = _argument_text(value)
                if text:
                    units.append((event_type, role_name, text))
    return units


def _record_units(events: list[dict[str, Any]]) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    records: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        if not event_type:
            continue
        role_values: list[tuple[str, str]] = []
        arguments = event.get("arguments") or {}
        if isinstance(arguments, dict):
            for role, values in arguments.items():
                role_name = str(role).strip()
                raw_values = values if isinstance(values, list) else [values]
                for value in raw_values:
                    text = _argument_text(value)
                    if role_name and text:
                        role_values.append((role_name, text))
        records.append((event_type, tuple(sorted(role_values))))
    return records


def _multiset_f1(predicted: list[Any], gold: list[Any]) -> float:
    pred_counter = Counter(predicted)
    gold_counter = Counter(gold)
    if not pred_counter and not gold_counter:
        return 1.0
    if not pred_counter or not gold_counter:
        return 0.0
    true_positive = sum((pred_counter & gold_counter).values())
    precision = true_positive / sum(pred_counter.values())
    recall = true_positive / sum(gold_counter.values())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _penalties(
    candidate: dict[str, Any],
    *,
    candidate_events: list[dict[str, Any]],
    gold_events: list[dict[str, Any]],
    schema: DatasetSchema,
) -> dict[str, float]:
    diagnostics = candidate.get("diagnostics") or {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    schema_count = (
        _number(diagnostics.get("schema_violation"))
        + _number(diagnostics.get("unknown_event_type"))
        + _number(diagnostics.get("unknown_role"))
    )
    duplicate_count = _number(diagnostics.get("duplicate_argument"))
    candidate_argument_count = len(_surface_units(candidate_events))
    gold_argument_count = len(_surface_units(gold_events))
    count_error = abs(len(candidate_events) - len(gold_events))
    invalid_schema_records = _invalid_schema_record_count(candidate_events, schema)
    return {
        "penalty_schema": 0.10 * (schema_count + invalid_schema_records),
        "penalty_duplicate": 0.05 * duplicate_count,
        "penalty_empty": 0.25 if candidate_argument_count == 0 and gold_argument_count > 0 else 0.0,
        "penalty_count": 0.02 * count_error,
    }


def _invalid_schema_record_count(events: list[dict[str, Any]], schema: DatasetSchema) -> int:
    count = 0
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type not in schema.event_roles:
            count += 1
            continue
        arguments = event.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for role in arguments:
            if str(role).strip() not in schema.event_roles[event_type]:
                count += 1
    return count


def _argument_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return str(value).strip()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _candidate_index(candidate: dict[str, Any]) -> int:
    try:
        return int(candidate.get("candidate_index", 0))
    except (TypeError, ValueError):
        return 0
