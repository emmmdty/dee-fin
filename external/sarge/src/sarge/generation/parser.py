from __future__ import annotations

import json
from typing import Any

from sarge.generation.candidate_types import GeneratedCandidateSet
from sarge.data.canonical import CanonicalArgument, CanonicalEventRecord
from sarge.data.schema import DatasetSchema
from sarge.generation.diagnostics import build_generation_diagnostics
from sarge.generation.prompt import DEFAULT_GETM_OUTPUT_FORMAT, normalize_output_format
from sarge.evaluation.export import strip_auxiliary_fields, validate_minimal_canonical_prediction

PROMPT_COPY_MARKERS = (
    "[Schema]",
    "[Document]",
    "[Surface Candidates]",
    "[Instruction]",
    "[Event Slot Plan]",
    "[Dataset]",
)

DIAGNOSTIC_KEYS = (
    "parse_error",
    "schema_violation",
    "unknown_event_type",
    "unknown_event_type_count",
    "unknown_role",
    "unknown_role_count",
    "duplicate_argument",
    "duplicate_argument_count",
    "accepted_event_count",
    "raw_event_count",
    "repaired_count",
    "extracted_json_object_count",
    "markdown_fence_stripped_count",
    "leading_text_removed_count",
    "trailing_text_removed_count",
    "copied_prompt_marker_count",
    "no_complete_json_object_count",
    "top_level_array_wrapped_count",
    "response_prefix_used",
    "response_prefix_reconstructed_count",
    "response_prefix_array_reconstructed_count",
    "invalid_event_object_count",
    "invalid_arguments_shape_count",
    "empty_arguments_count",
    "unexpected_source_candidate_id_count",
    "event_type_not_string_count",
    "role_value_not_list_count",
)


def parse_getm_output(
    raw_output: str,
    *,
    doc_id: str,
    candidate_id: str,
    schema: DatasetSchema,
    generation_score: float | None = None,
    response_prefix: str | None = None,
    response_prefix_used: bool = False,
    prompt: str | None = None,
    surface_candidate_count: int | None = None,
    generation_metadata: dict[str, Any] | None = None,
    token_metadata: dict[str, Any] | None = None,
    output_format: str = DEFAULT_GETM_OUTPUT_FORMAT,
) -> GeneratedCandidateSet:
    normalized_output_format = normalize_output_format(output_format)
    diagnostics = _empty_diagnostics()
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
        _record_repair(diagnostics, "top_level_array_wrapped")
        payload = {"events": payload}

    raw_events = _raw_events(payload)
    events: list[CanonicalEventRecord] = []
    for raw_event in raw_events:
        event = _parse_event(
            raw_event,
            schema=schema,
            diagnostics=diagnostics,
            output_format=normalized_output_format,
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


def candidate_set_to_canonical_prediction(
    candidate: GeneratedCandidateSet,
    *,
    schema: DatasetSchema | None = None,
) -> dict[str, Any]:
    pred_doc = {
        "doc_id": candidate.doc_id,
        "events": [
            {
                "event_type": event.event_type,
                "arguments": {
                    role: [{"text": argument.text} for argument in values]
                    for role, values in event.arguments.items()
                },
            }
            for event in candidate.events
        ],
    }
    canonical = strip_auxiliary_fields(pred_doc, schema=schema)
    validate_minimal_canonical_prediction(canonical, schema=schema)
    return canonical


def candidate_set_to_dict(candidate: GeneratedCandidateSet) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "doc_id": candidate.doc_id,
        "parse_status": candidate.parse_status,
        "generation_score": candidate.generation_score,
        "slot_plan_ids": list(candidate.slot_plan_ids),
        "diagnostics": dict(candidate.diagnostics),
        "events": candidate_set_to_canonical_prediction(candidate)["events"],
    }


def _empty_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {key: 0 for key in DIAGNOSTIC_KEYS}
    diagnostics["repair_type_counts"] = {}
    return diagnostics


def _generation_diagnostics(
    *,
    raw_output: str,
    prompt: str | None,
    surface_candidate_count: int | None,
    generation_metadata: dict[str, Any] | None,
    token_metadata: dict[str, Any] | None,
    include_parse_error_subtypes: bool,
) -> dict[str, Any]:
    generation = dict(generation_metadata or {})
    token = dict(token_metadata or {})
    return build_generation_diagnostics(
        raw_output=raw_output,
        prompt=prompt,
        surface_candidate_count=surface_candidate_count,
        max_new_tokens=_optional_int(token.get("max_new_tokens"), generation.get("max_new_tokens")),
        prompt_token_count=_optional_int(token.get("prompt_token_count")),
        prompt_token_count_source=_optional_str(token.get("prompt_token_count_source")),
        prompt_token_budget=_optional_int(token.get("prompt_token_budget"), generation.get("prompt_token_budget")),
        full_prompt_token_count=_optional_int(token.get("full_prompt_token_count")),
        prompt_packing_strategy=_optional_str(token.get("prompt_packing_strategy")),
        prompt_prefix_token_keep_count=_optional_int(token.get("prompt_prefix_token_keep_count")),
        prompt_suffix_token_keep_count=_optional_int(token.get("prompt_suffix_token_keep_count")),
        prompt_middle_token_drop_count=_optional_int(token.get("prompt_middle_token_drop_count")),
        prompt_delimiter_present_after_packing=_optional_bool(token.get("prompt_delimiter_present_after_packing")),
        response_prefix_present_after_packing=_optional_bool(token.get("response_prefix_present_after_packing")),
        prompt_section_char_counts=_optional_int_mapping(
            token.get("prompt_section_char_counts"),
            generation.get("prompt_section_char_counts"),
        ),
        prompt_section_token_counts=_optional_int_mapping(
            token.get("prompt_section_token_counts"),
            generation.get("prompt_section_token_counts"),
        ),
        generated_token_count=_optional_int(token.get("generated_token_count")),
        generated_token_count_source=_optional_str(token.get("generated_token_count_source")),
        hit_max_new_tokens=_optional_bool(token.get("hit_max_new_tokens")),
        hit_max_new_tokens_source=_optional_str(token.get("hit_max_new_tokens_source")),
        ended_with_eos=_optional_bool(token.get("ended_with_eos")),
        ended_with_eos_source=_optional_str(token.get("ended_with_eos_source")),
        ended_with_eos_reason=_optional_str(token.get("ended_with_eos_reason")),
        stop_reason=_optional_str(token.get("stop_reason"), generation.get("stop_reason")),
        balanced_stop_applied=_optional_bool(token.get("balanced_stop_applied")),
        stopped_output_char_count=_optional_int(token.get("stopped_output_char_count")),
        include_parse_error_subtypes=include_parse_error_subtypes,
    )


def _optional_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "1", "yes"}:
            return True
        if stripped in {"false", "0", "no"}:
            return False
    return None


def _optional_str(*values: Any) -> str | None:
    for candidate in values:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def _optional_int_mapping(*values: Any) -> dict[str, int] | None:
    for value in values:
        if not isinstance(value, dict):
            continue
        mapping: dict[str, int] = {}
        for key, raw_value in value.items():
            parsed = _optional_int(raw_value)
            if parsed is not None:
                mapping[str(key)] = parsed
        if mapping:
            return mapping
    return None


def _extract_json_payload(
    raw_output: str,
    *,
    diagnostics: dict[str, Any],
    response_prefix: str | None = None,
    response_prefix_used: bool = False,
) -> Any:
    stripped = str(raw_output).strip()
    diagnostics["copied_prompt_marker_count"] = _count_prompt_markers(stripped)
    if response_prefix_used:
        diagnostics["response_prefix_used"] = 1
    if not stripped:
        diagnostics["no_complete_json_object_count"] = 1
        raise ValueError("empty model output")
    if _is_candidate_list_continuation(diagnostics):
        diagnostics["no_complete_json_object_count"] = 1
        raise ValueError("candidate-list continuation cannot be safely JSON-repaired")

    payload = _extract_json_payload_once(
        stripped,
        diagnostics=diagnostics,
        allow_extract=not response_prefix_used,
    )
    if payload is not None:
        return payload

    reconstructed = _reconstruct_response_prefix(
        stripped,
        response_prefix=response_prefix,
        response_prefix_used=response_prefix_used,
    )
    if reconstructed is not None:
        reconstructed_text, repair_type = reconstructed
        payload = _extract_json_payload_once(
            reconstructed_text,
            diagnostics=diagnostics,
            base_repair={repair_type: 1},
        )
        if payload is not None:
            return payload

    if response_prefix_used:
        payload = _extract_json_payload_once(stripped, diagnostics=diagnostics)
        if payload is not None:
            return payload

    if not _contains_complete_json_object(stripped):
        diagnostics["no_complete_json_object_count"] = 1
    raise ValueError("no valid JSON payload found")


def _extract_json_payload_once(
    text: str,
    *,
    diagnostics: dict[str, Any],
    base_repair: dict[str, int] | None = None,
    allow_extract: bool = True,
) -> Any | None:
    stripped = str(text).strip()
    if not stripped:
        return None

    fenced, fence_repair = _strip_markdown_fence(stripped)
    repairs = dict(base_repair or {})
    repairs.update(fence_repair)
    payload = _decode_full_json(fenced)
    if payload is not None:
        _commit_repairs(diagnostics, repairs)
        return payload

    if not allow_extract:
        return None
    payload = _extract_first_complete_json_object(fenced, diagnostics=diagnostics, base_repair=repairs)
    if payload is not None:
        return payload
    return None


def _is_candidate_list_continuation(diagnostics: dict[str, Any]) -> bool:
    return int(diagnostics.get("starts_with_candidate_fragment", 0) or 0) > 0


def _reconstruct_response_prefix(
    text: str,
    *,
    response_prefix: str | None,
    response_prefix_used: bool,
) -> tuple[str, str] | None:
    prefix = str(response_prefix or "").strip()
    if not response_prefix_used or not prefix:
        return None
    stripped = text.strip()
    if stripped.startswith(prefix):
        return None
    repair_type = (
        "response_prefix_array_reconstructed"
        if _is_events_array_prefix(prefix, stripped)
        else "response_prefix_reconstructed"
    )
    return f"{prefix}{stripped}", repair_type


def _is_events_array_prefix(prefix: str, continuation: str) -> bool:
    compact_prefix = "".join(prefix.split())
    stripped = continuation.lstrip()
    if compact_prefix in {'{"events":', '{"events":['}:
        return stripped.startswith(("[", "{", "]"))
    return False


def _count_prompt_markers(text: str) -> int:
    return sum(text.count(marker) for marker in PROMPT_COPY_MARKERS)


def _contains_complete_json_object(text: str) -> bool:
    start = text.find("{")
    return start >= 0 and _balanced_json_object_end(text, start) is not None


def _strip_markdown_fence(text: str) -> tuple[str, dict[str, int]]:
    lines = text.splitlines()
    if len(lines) < 2:
        return text, {}
    opener = lines[0].strip().lower()
    if opener not in {"```", "```json"}:
        return text, {}
    if lines[-1].strip() != "```":
        return text, {}
    return "\n".join(lines[1:-1]).strip(), {"markdown_fence_stripped": 1}


def _decode_full_json(candidate: str) -> Any | None:
    try:
        payload, end_index = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    if candidate[end_index:].strip():
        return None
    return payload


def _extract_first_complete_json_object(
    text: str,
    *,
    diagnostics: dict[str, Any],
    base_repair: dict[str, int],
) -> Any | None:
    start = text.find("{")
    if start < 0:
        return None
    end = _balanced_json_object_end(text, start)
    if end is None:
        return None
    candidate = text[start:end]
    payload = _decode_full_json(candidate)
    if payload is None:
        return None

    repairs = dict(base_repair)
    repairs["extracted_json_object"] = 1
    if text[:start].strip():
        repairs["leading_text_removed"] = 1
    if text[end:].strip():
        repairs["trailing_text_removed"] = 1
    _commit_repairs(diagnostics, repairs)
    return payload


def _balanced_json_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _commit_repairs(diagnostics: dict[str, Any], repairs: dict[str, int]) -> None:
    for repair_type, count in repairs.items():
        if count > 0:
            _record_repair(diagnostics, repair_type)


def _record_repair(diagnostics: dict[str, Any], repair_type: str) -> None:
    repair_type_counts = diagnostics.setdefault("repair_type_counts", {})
    repair_type_counts[repair_type] = int(repair_type_counts.get(repair_type, 0)) + 1
    diagnostics["repaired_count"] = 1
    if repair_type == "extracted_json_object":
        diagnostics["extracted_json_object_count"] += 1
    elif repair_type == "markdown_fence_stripped":
        diagnostics["markdown_fence_stripped_count"] += 1
    elif repair_type == "leading_text_removed":
        diagnostics["leading_text_removed_count"] += 1
    elif repair_type == "trailing_text_removed":
        diagnostics["trailing_text_removed_count"] += 1
    elif repair_type == "top_level_array_wrapped":
        diagnostics["top_level_array_wrapped_count"] += 1
    elif repair_type == "response_prefix_reconstructed":
        diagnostics["response_prefix_reconstructed_count"] += 1
    elif repair_type == "response_prefix_array_reconstructed":
        diagnostics["response_prefix_reconstructed_count"] += 1
        diagnostics["response_prefix_array_reconstructed_count"] += 1


def _raw_events(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        events = payload.get("events") or []
    elif isinstance(payload, list):
        events = payload
    else:
        return []
    return events if isinstance(events, list) else []


def _parse_event(
    raw_event: Any,
    *,
    schema: DatasetSchema,
    diagnostics: dict[str, Any],
    output_format: str,
) -> CanonicalEventRecord | None:
    if not isinstance(raw_event, dict):
        diagnostics["invalid_event_object_count"] += 1
        diagnostics["schema_violation"] += 1
        return None

    raw_event_type = raw_event.get("event_type", "")
    if not isinstance(raw_event_type, str):
        diagnostics["event_type_not_string_count"] += 1
        diagnostics["schema_violation"] += 1
        return None
    event_type = raw_event_type.strip()
    if event_type not in schema.event_roles:
        _increment(diagnostics, "unknown_event_type", "unknown_event_type_count")
        diagnostics["schema_violation"] += 1
        return None

    if "arguments" in raw_event:
        raw_arguments = raw_event.get("arguments")
    else:
        raw_arguments = raw_event.get("arguments_by_role")
    if not isinstance(raw_arguments, dict):
        diagnostics["invalid_arguments_shape_count"] += 1
        diagnostics["schema_violation"] += 1
        return None

    invalid_role_seen = False
    for role in raw_arguments:
        role_name = str(role).strip()
        if role_name not in schema.event_roles[event_type]:
            _increment(diagnostics, "unknown_role", "unknown_role_count")
            diagnostics["schema_violation"] += 1
            invalid_role_seen = True
    if invalid_role_seen:
        return None

    arguments: dict[str, list[CanonicalArgument]] = {}
    for role, raw_values in raw_arguments.items():
        role_name = str(role).strip()
        if not isinstance(raw_values, list):
            diagnostics["role_value_not_list_count"] += 1
            diagnostics["schema_violation"] += 1
            continue
        seen_texts: set[str] = set()
        values = []
        for text in _argument_texts(
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


def _increment(diagnostics: dict[str, Any], key: str, *alias_keys: str) -> None:
    diagnostics[key] = int(diagnostics.get(key, 0) or 0) + 1
    for alias_key in alias_keys:
        diagnostics[alias_key] = int(diagnostics.get(alias_key, 0) or 0) + 1


def _argument_texts(
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


def _has_schema_diagnostics(diagnostics: dict[str, Any]) -> bool:
    return any(
        int(diagnostics.get(key, 0)) > 0
        for key in (
            "schema_violation",
            "unknown_event_type",
            "unknown_role",
            "duplicate_argument",
            "invalid_event_object_count",
            "invalid_arguments_shape_count",
            "event_type_not_string_count",
            "role_value_not_list_count",
        )
    )
