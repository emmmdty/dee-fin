from __future__ import annotations

from typing import Any, Literal

from sarge.generation.candidate_types import GeneratedCandidateSet
from sarge.data.canonical import CanonicalArgument, CanonicalEventRecord
from sarge.data.schema import DatasetSchema
from sarge.generation.parser import (
    DEFAULT_GETM_OUTPUT_FORMAT,
    _empty_diagnostics,
    _extract_json_payload,
    _generation_diagnostics,
    _has_schema_diagnostics,
    _increment,
    _raw_events,
    candidate_set_to_canonical_prediction,
    candidate_set_to_dict,
)
from sarge.generation.prompt import normalize_output_format

ParserAblationMode = Literal[
    "frozen_strict",
    "drop_invalid_role_only",
    "keep_event_schema_valid_args",
]

PARSER_ABLATION_MODES: tuple[ParserAblationMode, ...] = (
    "frozen_strict",
    "drop_invalid_role_only",
    "keep_event_schema_valid_args",
)

ABLATION_DIAGNOSTIC_KEYS = (
    "dropped_event_count",
    "dropped_role_count",
    "dropped_event_unknown_event_type_count",
    "dropped_event_unknown_role_count",
    "dropped_event_invalid_shape_count",
    "dropped_role_unknown_role_count",
    "dropped_role_invalid_value_shape_count",
)


def parse_getm_output_ablation(
    raw_output: str,
    *,
    doc_id: str,
    candidate_id: str,
    schema: DatasetSchema,
    mode: ParserAblationMode,
    generation_score: float | None = None,
    response_prefix: str | None = None,
    response_prefix_used: bool = False,
    prompt: str | None = None,
    surface_candidate_count: int | None = None,
    generation_metadata: dict[str, Any] | None = None,
    token_metadata: dict[str, Any] | None = None,
    output_format: str = DEFAULT_GETM_OUTPUT_FORMAT,
) -> GeneratedCandidateSet:
    if mode not in PARSER_ABLATION_MODES:
        raise ValueError(f"unknown parser ablation mode: {mode}")

    normalized_output_format = normalize_output_format(output_format)
    diagnostics = _empty_ablation_diagnostics(mode)
    diagnostics.update(
        _generation_diagnostics(
            raw_output=raw_output,
            prompt=prompt,
            surface_candidate_count=surface_candidate_count,
            generation_metadata=generation_metadata,
            token_metadata=token_metadata,
            include_parse_error_subtypes=False,
        )
    )
    try:
        payload = _extract_json_payload(
            raw_output,
            diagnostics=diagnostics,
            response_prefix=response_prefix,
            response_prefix_used=response_prefix_used,
        )
    except Exception as exc:
        diagnostics.update(
            _generation_diagnostics(
                raw_output=raw_output,
                prompt=prompt,
                surface_candidate_count=surface_candidate_count,
                generation_metadata=generation_metadata,
                token_metadata=token_metadata,
                include_parse_error_subtypes=True,
            )
        )
        diagnostics["parse_error"] = 1
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return GeneratedCandidateSet(
            candidate_id=candidate_id,
            doc_id=doc_id,
            events=[],
            parse_status="parse_error",
            generation_score=generation_score,
            diagnostics=diagnostics,
        )

    if isinstance(payload, list):
        from sarge.generation.parser import _record_repair

        _record_repair(diagnostics, "top_level_array_wrapped")
        payload = {"events": payload}

    raw_events = _raw_events(payload)
    events: list[CanonicalEventRecord] = []
    for raw_event in raw_events:
        event = _parse_event_ablation(
            raw_event,
            schema=schema,
            diagnostics=diagnostics,
            output_format=normalized_output_format,
            mode=mode,
        )
        if event is not None:
            events.append(event)
    diagnostics["raw_event_count"] = len(raw_events)
    diagnostics["accepted_event_count"] = len(events)
    parse_status = "schema_violation" if _has_schema_diagnostics(diagnostics) else "ok"
    return GeneratedCandidateSet(
        candidate_id=candidate_id,
        doc_id=doc_id,
        events=events,
        parse_status=parse_status,
        generation_score=generation_score,
        diagnostics=diagnostics,
    )


def _empty_ablation_diagnostics(mode: ParserAblationMode) -> dict[str, Any]:
    diagnostics = _empty_diagnostics()
    for key in ABLATION_DIAGNOSTIC_KEYS:
        diagnostics[key] = 0
    diagnostics["parser_ablation_mode"] = mode
    return diagnostics


def _parse_event_ablation(
    raw_event: Any,
    *,
    schema: DatasetSchema,
    diagnostics: dict[str, Any],
    output_format: str,
    mode: ParserAblationMode,
) -> CanonicalEventRecord | None:
    if not isinstance(raw_event, dict):
        diagnostics["invalid_event_object_count"] += 1
        diagnostics["schema_violation"] += 1
        _drop_event(diagnostics, "invalid_shape")
        return None

    raw_event_type = raw_event.get("event_type", "")
    if not isinstance(raw_event_type, str):
        diagnostics["event_type_not_string_count"] += 1
        diagnostics["schema_violation"] += 1
        _drop_event(diagnostics, "invalid_shape")
        return None
    event_type = raw_event_type.strip()
    if event_type not in schema.event_roles:
        _increment(diagnostics, "unknown_event_type", "unknown_event_type_count")
        diagnostics["schema_violation"] += 1
        _drop_event(diagnostics, "unknown_event_type")
        return None

    if "arguments" in raw_event:
        raw_arguments = raw_event.get("arguments")
    else:
        raw_arguments = raw_event.get("arguments_by_role")
    if not isinstance(raw_arguments, dict):
        diagnostics["invalid_arguments_shape_count"] += 1
        diagnostics["schema_violation"] += 1
        _drop_event(diagnostics, "invalid_shape")
        return None

    invalid_role_seen = False
    for role in raw_arguments:
        role_name = str(role).strip()
        if role_name not in schema.event_roles[event_type]:
            _increment(diagnostics, "unknown_role", "unknown_role_count")
            diagnostics["schema_violation"] += 1
            _drop_role(diagnostics, "unknown_role")
            invalid_role_seen = True
    if invalid_role_seen and mode == "frozen_strict":
        _drop_event(diagnostics, "unknown_role")
        return None

    arguments: dict[str, list[CanonicalArgument]] = {}
    for role, raw_values in raw_arguments.items():
        role_name = str(role).strip()
        if role_name not in schema.event_roles[event_type]:
            continue
        if not isinstance(raw_values, list):
            diagnostics["role_value_not_list_count"] += 1
            diagnostics["schema_violation"] += 1
            if mode == "keep_event_schema_valid_args":
                _drop_role(diagnostics, "invalid_value_shape")
            continue
        seen_texts: set[str] = set()
        values = []
        for text in _argument_texts_ablation(
            raw_values,
            output_format=output_format,
            diagnostics=diagnostics,
        ):
            if text in seen_texts:
                _increment(diagnostics, "duplicate_argument", "duplicate_argument_count")
                continue
            seen_texts.add(text)
            values.append(CanonicalArgument(text=text))
        if values:
            arguments[role_name] = values
    if not arguments:
        diagnostics["empty_arguments_count"] += 1
    return CanonicalEventRecord(event_type=event_type, arguments=arguments)


def _argument_texts_ablation(
    values: list[Any],
    *,
    output_format: str,
    diagnostics: dict[str, Any],
) -> list[str]:
    texts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            if output_format == "minimal_text" and "source_candidate_id" in value:
                diagnostics["unexpected_source_candidate_id_count"] += 1
            text = str(value.get("text", "")).strip()
        else:
            text = str(value).strip()
        if text:
            texts.append(text)
    return texts


def _drop_event(diagnostics: dict[str, Any], reason: str) -> None:
    diagnostics["dropped_event_count"] = int(diagnostics.get("dropped_event_count", 0) or 0) + 1
    if reason == "unknown_event_type":
        diagnostics["dropped_event_unknown_event_type_count"] += 1
    elif reason == "unknown_role":
        diagnostics["dropped_event_unknown_role_count"] += 1
    elif reason == "invalid_shape":
        diagnostics["dropped_event_invalid_shape_count"] += 1


def _drop_role(diagnostics: dict[str, Any], reason: str) -> None:
    diagnostics["dropped_role_count"] = int(diagnostics.get("dropped_role_count", 0) or 0) + 1
    if reason == "unknown_role":
        diagnostics["dropped_role_unknown_role_count"] += 1
    elif reason == "invalid_value_shape":
        diagnostics["dropped_role_invalid_value_shape_count"] += 1


__all__ = [
    "ABLATION_DIAGNOSTIC_KEYS",
    "PARSER_ABLATION_MODES",
    "ParserAblationMode",
    "candidate_set_to_canonical_prediction",
    "candidate_set_to_dict",
    "parse_getm_output_ablation",
]
