from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from sarge.data.schema import DatasetSchema

FEATURE_NAMES = (
    "schema_valid_rate",
    "role_coverage",
    "duplicate_argument_rate",
    "unknown_event_type_count",
    "unknown_role_count",
    "empty_prediction",
    "candidate_length",
    "avg_logprob",
    "grounding_confidence",
    "lesp_event_count_agreement",
    "self_consistency_argument_jaccard",
    "self_consistency_event_type_jaccard",
)


def compute_feature_rows(
    candidates: Iterable[dict[str, Any]],
    *,
    schema: DatasetSchema,
    surface_memories: Mapping[str, dict[str, Any]] | None = None,
    slot_plans: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidate_rows = list(candidates)
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        by_doc[str(candidate.get("doc_id", ""))].append(candidate)

    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        doc_id = str(candidate.get("doc_id", ""))
        rows.append(
            compute_candidate_features(
                candidate,
                schema=schema,
                surface_memory=(surface_memories or {}).get(doc_id),
                slot_plan=(slot_plans or {}).get(doc_id),
                peer_candidates=by_doc.get(doc_id, [candidate]),
            )
        )
    return rows


def compute_candidate_features(
    candidate: dict[str, Any],
    *,
    schema: DatasetSchema,
    surface_memory: dict[str, Any] | None = None,
    slot_plan: dict[str, Any] | None = None,
    peer_candidates: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = _events(candidate)
    diagnostics = _diagnostics(candidate)
    argument_triples = _argument_triples(events)
    candidate_length = len(argument_triples)
    duplicate_count = _number(candidate.get("duplicate_argument"), diagnostics.get("duplicate_argument"))
    unknown_event_type_count = _number(candidate.get("unknown_event_type"), diagnostics.get("unknown_event_type"))
    unknown_role_count = _number(candidate.get("unknown_role"), diagnostics.get("unknown_role"))
    schema_violation_count = _number(candidate.get("schema_violation"), diagnostics.get("schema_violation"))

    features = {
        "schema_valid_rate": _schema_valid_rate(events, diagnostics),
        "role_coverage": _role_coverage(events, schema),
        "duplicate_argument_rate": duplicate_count / max(candidate_length + duplicate_count, 1.0),
        "unknown_event_type_count": unknown_event_type_count,
        "unknown_role_count": unknown_role_count,
        "empty_prediction": 1.0 if candidate_length == 0 else 0.0,
        "candidate_length": float(candidate_length),
        "avg_logprob": _avg_logprob(candidate),
        "grounding_confidence": _grounding_confidence(events, surface_memory),
        "lesp_event_count_agreement": _lesp_event_count_agreement(events, slot_plan),
    }
    consistency = _self_consistency_features(candidate, peer_candidates or [candidate])
    features.update(consistency)

    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "doc_id": str(candidate.get("doc_id", "")),
        "candidate_index": candidate_index(candidate),
        "parse_status": str(candidate.get("parse_status", "")),
        "features": {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES},
        "diagnostics": {
            **diagnostics,
            "schema_violation_count": schema_violation_count,
            "argument_count": candidate_length,
            "lesp_available": slot_plan is not None,
            "surface_memory_available": surface_memory is not None,
        },
    }


def candidate_index(candidate: dict[str, Any]) -> int:
    raw_index = candidate.get("candidate_index")
    if raw_index is not None:
        try:
            return int(raw_index)
        except (TypeError, ValueError):
            pass
    candidate_id = str(candidate.get("candidate_id", ""))
    match = re.search(r":getm:(\d+)$", candidate_id)
    if match:
        return int(match.group(1))
    return 0


def argument_signature_set(candidate: dict[str, Any]) -> set[tuple[str, str, str]]:
    return set(_argument_triples(_events(candidate)))


def event_type_set(candidate: dict[str, Any]) -> set[str]:
    return {str(event.get("event_type", "")).strip() for event in _events(candidate) if event.get("event_type")}


def _events(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = candidate.get("events") or []
    return [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else []


def _diagnostics(candidate: dict[str, Any]) -> dict[str, Any]:
    diagnostics = candidate.get("diagnostics") or {}
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _number(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _schema_valid_rate(events: list[dict[str, Any]], diagnostics: dict[str, Any]) -> float:
    accepted_units = _number(diagnostics.get("accepted_event_count"))
    if accepted_units == 0.0 and events:
        accepted_units = float(len(events))
    invalid_units = (
        _number(diagnostics.get("schema_violation"))
        + _number(diagnostics.get("unknown_event_type"))
        + _number(diagnostics.get("unknown_role"))
        + _number(diagnostics.get("duplicate_argument"))
    )
    if accepted_units == 0.0 and invalid_units == 0.0:
        return 1.0
    return accepted_units / max(accepted_units + invalid_units, 1.0)


def _role_coverage(events: list[dict[str, Any]], schema: DatasetSchema) -> float:
    total_roles = 0
    filled_roles = 0
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        roles = schema.event_roles.get(event_type)
        if not roles:
            continue
        total_roles += len(roles)
        arguments = event.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for role in roles:
            values = arguments.get(role) or []
            if isinstance(values, list) and any(_argument_text(value) for value in values):
                filled_roles += 1
    return filled_roles / total_roles if total_roles else 0.0


def _avg_logprob(candidate: dict[str, Any]) -> float:
    for key in ("avg_logprob", "logprob", "generation_score"):
        value = candidate.get(key)
        if value is not None:
            return _number(value)
    diagnostics = _diagnostics(candidate)
    for key in ("avg_logprob", "logprob"):
        value = diagnostics.get(key)
        if value is not None:
            return _number(value)

    logprobs: list[float] = []
    _collect_logprobs(candidate.get("events"), logprobs)
    return sum(logprobs) / len(logprobs) if logprobs else 0.0


def _collect_logprobs(value: Any, logprobs: list[float]) -> None:
    if isinstance(value, dict):
        if "logprob" in value:
            logprobs.append(_number(value.get("logprob")))
        for child in value.values():
            _collect_logprobs(child, logprobs)
    elif isinstance(value, list):
        for child in value:
            _collect_logprobs(child, logprobs)


def _grounding_confidence(events: list[dict[str, Any]], surface_memory: dict[str, Any] | None) -> float:
    argument_texts = [text for _, _, text in _argument_triples(events)]
    if not argument_texts:
        return 0.0
    if not surface_memory:
        return 0.0
    surface_scores: dict[str, float] = {}
    for candidate in surface_memory.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        surface = str(candidate.get("surface", "")).strip()
        if not surface:
            continue
        score = _number(candidate.get("role_score"))
        surface_scores[surface] = max(surface_scores.get(surface, 0.0), score if score > 0.0 else 1.0)
    if not surface_scores:
        return 0.0
    scores = [surface_scores.get(text, 0.0) for text in argument_texts]
    return sum(scores) / len(scores)


def _lesp_event_count_agreement(events: list[dict[str, Any]], slot_plan: dict[str, Any] | None) -> float:
    if slot_plan is None:
        return 0.0
    predicted_counts = Counter(str(event.get("event_type", "")).strip() for event in events if event.get("event_type"))
    planned_counts = Counter()
    for slot in slot_plan.get("slots") or []:
        if isinstance(slot, dict):
            event_type = str(slot.get("event_type", "")).strip()
            if event_type:
                planned_counts[event_type] += 1
    all_types = set(predicted_counts) | set(planned_counts)
    if not all_types:
        return 1.0
    total_error = sum(abs(predicted_counts[event_type] - planned_counts[event_type]) for event_type in all_types)
    denominator = max(sum(predicted_counts.values()), sum(planned_counts.values()), 1)
    return max(0.0, 1.0 - (total_error / denominator))


def _self_consistency_features(
    candidate: dict[str, Any],
    peer_candidates: Iterable[dict[str, Any]],
) -> dict[str, float]:
    peers = list(peer_candidates)
    if len(peers) <= 1:
        return {
            "self_consistency_argument_jaccard": 1.0,
            "self_consistency_event_type_jaccard": 1.0,
        }
    target_args = argument_signature_set(candidate)
    target_types = event_type_set(candidate)
    argument_scores: list[float] = []
    event_type_scores: list[float] = []
    target_id = str(candidate.get("candidate_id", ""))
    for peer in peers:
        if str(peer.get("candidate_id", "")) == target_id:
            continue
        argument_scores.append(_jaccard(target_args, argument_signature_set(peer)))
        event_type_scores.append(_jaccard(target_types, event_type_set(peer)))
    return {
        "self_consistency_argument_jaccard": sum(argument_scores) / len(argument_scores) if argument_scores else 1.0,
        "self_consistency_event_type_jaccard": (
            sum(event_type_scores) / len(event_type_scores) if event_type_scores else 1.0
        ),
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _argument_triples(events: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
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
                    triples.append((event_type, role_name, text))
    return triples


def _argument_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return str(value).strip()
