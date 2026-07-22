from __future__ import annotations

import json
import re
from typing import Any

DIAGNOSTIC_VERSION = "sarge_generation_diagnostics_v1"

INSTRUCTION_SENTENCES = (
    "Return ONLY one valid JSON object.",
    "Do not wrap it in markdown fences.",
    "Do not explain.",
    "Do not repeat the document, schema, candidates, or instruction.",
    "Do not output YAML.",
    "The first character of your answer must be `{` and the last character must be `}`.",
    "The top-level JSON object must contain only the `events` key.",
    "Do not output fields outside the event type and role schema.",
    "Use the original dataset schema labels exactly.",
    "Do not invent event types or roles.",
    "The event type must match the schema.",
    "The arguments must be a list of objects with keys matching the schema.",
    'Each argument must have a "text" field and optionally a "source_candidate_id" field.',
    "The source_candidate_id is the ID from the Surface Candidates list.",
    "The text field must be a string that matches the original text in the document.",
    "The text field must not contain any extra characters or whitespace.",
)

CANDIDATE_LINE_RE = re.compile(
    r"(?m)^\s*(?:-\s*)?(?:id=)?=?[^\s|:]+:(?:csg|ec):[^\s|]+"
    r"\s*\|\s*(?:event_type=[^|]+\|\s*)?text=[^|]*\|\s*(?:chunk=[^|]*\|\s*)?context=",
)
CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+:(?:csg|ec):[A-Za-z0-9_.-]+")

PARSE_ERROR_SUBTYPE_ORDER = (
    "no_json_started",
    "candidate_list_continuation",
    "instruction_loop",
    "array_continuation_not_reconstructed",
    "truncated_or_hit_max_new_tokens",
    "malformed_json",
)
PRIMARY_SUBTYPE_PRIORITY = (
    "instruction_loop",
    "candidate_list_continuation",
    "array_continuation_not_reconstructed",
    "truncated_or_hit_max_new_tokens",
    "no_json_started",
    "malformed_json",
)


def build_generation_diagnostics(
    *,
    raw_output: str,
    prompt: str | None = None,
    surface_candidate_count: int | None = None,
    max_new_tokens: int | None = None,
    prompt_token_count: int | None = None,
    prompt_token_count_source: str | None = None,
    prompt_token_budget: int | None = None,
    full_prompt_token_count: int | None = None,
    prompt_packing_strategy: str | None = None,
    prompt_prefix_token_keep_count: int | None = None,
    prompt_suffix_token_keep_count: int | None = None,
    prompt_middle_token_drop_count: int | None = None,
    prompt_delimiter_present_after_packing: bool | None = None,
    response_prefix_present_after_packing: bool | None = None,
    prompt_section_char_counts: dict[str, int] | None = None,
    prompt_section_token_counts: dict[str, int] | None = None,
    generated_token_count: int | None = None,
    generated_token_count_source: str | None = None,
    hit_max_new_tokens: bool | None = None,
    hit_max_new_tokens_source: str | None = None,
    ended_with_eos: bool | None = None,
    ended_with_eos_source: str | None = None,
    ended_with_eos_reason: str | None = None,
    stop_reason: str | None = None,
    balanced_stop_applied: bool | None = None,
    stopped_output_char_count: int | None = None,
    include_parse_error_subtypes: bool = True,
) -> dict[str, Any]:
    raw_text = str(raw_output or "")
    prompt_text = str(prompt or "")
    stripped = raw_text.lstrip()
    starts_with_json_object = stripped.startswith("{")
    starts_with_json_array = stripped.startswith("[")
    candidate_line_copy_count = len(CANDIDATE_LINE_RE.findall(raw_text))
    candidate_id_copy_count = len(CANDIDATE_ID_RE.findall(raw_text))
    instruction_sentence_copy_count, instruction_loop_count = _instruction_copy_counts(raw_text)
    inferred_hit_max_new_tokens, inferred_hit_source = _hit_max_new_tokens(
        hit_max_new_tokens=hit_max_new_tokens,
        hit_max_new_tokens_source=hit_max_new_tokens_source,
        generated_token_count=generated_token_count,
        generated_token_count_source=generated_token_count_source,
        max_new_tokens=max_new_tokens,
        ended_with_eos=ended_with_eos,
    )
    brace_balance_state = _brace_balance_state(raw_text)

    diagnostics: dict[str, Any] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "prompt_token_count": prompt_token_count,
        "prompt_token_count_source": prompt_token_count_source,
        "prompt_token_budget": prompt_token_budget,
        "full_prompt_token_count": full_prompt_token_count,
        "prompt_packing_strategy": prompt_packing_strategy,
        "prompt_prefix_token_keep_count": prompt_prefix_token_keep_count,
        "prompt_suffix_token_keep_count": prompt_suffix_token_keep_count,
        "prompt_middle_token_drop_count": prompt_middle_token_drop_count,
        "prompt_delimiter_present_after_packing": prompt_delimiter_present_after_packing,
        "response_prefix_present_after_packing": response_prefix_present_after_packing,
        "prompt_token_limit_hit": _prompt_token_limit_hit(
            prompt_token_count=prompt_token_count,
            prompt_token_budget=prompt_token_budget,
        ),
        "prompt_section_char_counts": dict(prompt_section_char_counts or {}),
        "prompt_section_token_counts": dict(prompt_section_token_counts or {}),
        "generated_token_count": generated_token_count,
        "generated_token_count_source": generated_token_count_source,
        "hit_max_new_tokens": inferred_hit_max_new_tokens,
        "hit_max_new_tokens_source": inferred_hit_source,
        "ended_with_eos": ended_with_eos,
        "ended_with_eos_source": ended_with_eos_source,
        "ended_with_eos_reason": ended_with_eos_reason,
        "stop_reason": stop_reason,
        "balanced_stop_applied": balanced_stop_applied,
        "stopped_output_char_count": stopped_output_char_count,
        "raw_output_prefix_120": raw_text[:120],
        "raw_output_suffix_120": raw_text[-120:] if raw_text else "",
        "candidate_line_copy_count": candidate_line_copy_count,
        "candidate_id_copy_count": candidate_id_copy_count,
        "starts_with_candidate_fragment": int(_starts_with_candidate_fragment(stripped)),
        "starts_with_instruction_text": int(_starts_with_instruction_text(stripped)),
        "instruction_sentence_copy_count": instruction_sentence_copy_count,
        "instruction_loop_count": instruction_loop_count,
        "starts_with_json_object": int(starts_with_json_object),
        "starts_with_json_array": int(starts_with_json_array),
        "brace_balance_state": brace_balance_state,
        "surface_candidate_count": surface_candidate_count,
        "prompt_char_count": len(prompt_text),
        "raw_output_char_count": len(raw_text),
    }
    subtypes = (
        _parse_error_subtypes(
            stripped=stripped,
            starts_with_json_object=starts_with_json_object,
            starts_with_json_array=starts_with_json_array,
            brace_balance_state=brace_balance_state,
            candidate_line_copy_count=candidate_line_copy_count,
            instruction_loop_count=instruction_loop_count,
            hit_max_new_tokens=inferred_hit_max_new_tokens,
        )
        if include_parse_error_subtypes
        else []
    )
    diagnostics["parse_error_subtypes"] = subtypes
    diagnostics["parse_error_primary_subtype"] = _primary_subtype(subtypes)
    return diagnostics


def generation_diagnostic_fields(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in diagnostics.items()
        if key
        in {
            "diagnostic_version",
            "prompt_token_count",
            "prompt_token_count_source",
            "prompt_token_budget",
            "full_prompt_token_count",
            "prompt_packing_strategy",
            "prompt_prefix_token_keep_count",
            "prompt_suffix_token_keep_count",
            "prompt_middle_token_drop_count",
            "prompt_delimiter_present_after_packing",
            "response_prefix_present_after_packing",
            "prompt_token_limit_hit",
            "prompt_section_char_counts",
            "prompt_section_token_counts",
            "generated_token_count",
            "generated_token_count_source",
            "hit_max_new_tokens",
            "hit_max_new_tokens_source",
            "ended_with_eos",
            "ended_with_eos_source",
            "ended_with_eos_reason",
            "stop_reason",
            "balanced_stop_applied",
            "stopped_output_char_count",
            "raw_output_prefix_120",
            "raw_output_suffix_120",
            "candidate_line_copy_count",
            "candidate_id_copy_count",
            "starts_with_candidate_fragment",
            "starts_with_instruction_text",
            "instruction_sentence_copy_count",
            "instruction_loop_count",
            "starts_with_json_object",
            "starts_with_json_array",
            "brace_balance_state",
            "surface_candidate_count",
            "prompt_char_count",
            "raw_output_char_count",
            "parse_error_subtypes",
            "parse_error_primary_subtype",
        }
    }


def aggregate_parse_diagnostics(
    parsed_rows: list[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    k: int | None,
    generation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, Any] = {}
    subtype_counts: dict[str, int] = {}
    primary_subtype_counts: dict[str, int] = {}
    prompt_token_counts: list[int] = []
    prompt_token_limit_hit_rows = 0
    prompt_token_budget: int | None = None
    stop_reason_counts: dict[str, int] = {}
    balanced_stop_applied_count = 0
    section_char_counts: dict[str, list[int]] = {}
    section_token_counts: dict[str, list[int]] = {}
    for row in parsed_rows:
        status = str(row.get("parse_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        diagnostics = row.get("diagnostics") or {}
        if not isinstance(diagnostics, dict):
            continue
        _add_diagnostic_counts(diagnostic_counts, diagnostics)
        prompt_count = diagnostics.get("prompt_token_count")
        if isinstance(prompt_count, int):
            prompt_token_counts.append(prompt_count)
        budget = diagnostics.get("prompt_token_budget")
        if isinstance(budget, int) and prompt_token_budget is None:
            prompt_token_budget = budget
        if diagnostics.get("prompt_token_limit_hit") is True:
            prompt_token_limit_hit_rows += 1
        stop_reason = diagnostics.get("stop_reason")
        if isinstance(stop_reason, str) and stop_reason:
            stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        if diagnostics.get("balanced_stop_applied") is True:
            balanced_stop_applied_count += 1
        _collect_section_counts(section_char_counts, diagnostics.get("prompt_section_char_counts"))
        _collect_section_counts(section_token_counts, diagnostics.get("prompt_section_token_counts"))
        if status == "parse_error":
            for subtype in _as_str_list(diagnostics.get("parse_error_subtypes")):
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
            primary = diagnostics.get("parse_error_primary_subtype")
            if isinstance(primary, str) and primary:
                primary_subtype_counts[primary] = primary_subtype_counts.get(primary, 0) + 1

    result: dict[str, Any] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "dataset": dataset,
        "split": split,
        "k": k,
        "candidate_count": len(parsed_rows),
        "parse_status_counts": dict(sorted(status_counts.items())),
        "stopped_output_parse_status_counts": dict(sorted(status_counts.items())),
        "diagnostic_counts": _sort_diagnostic_counts(diagnostic_counts),
        "parse_error_subtype_counts": dict(sorted(subtype_counts.items())),
        "parse_error_primary_subtype_counts": dict(sorted(primary_subtype_counts.items())),
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "balanced_stop_applied_count": balanced_stop_applied_count,
        "prompt_token_summary": _prompt_token_summary(
            prompt_token_counts,
            prompt_token_budget=prompt_token_budget,
            prompt_token_limit_hit_rows=prompt_token_limit_hit_rows,
        ),
        "prompt_section_char_summary": _section_summary(section_char_counts),
        "prompt_section_token_summary": _section_summary(section_token_counts),
    }
    if generation_metadata:
        result["generation"] = dict(generation_metadata)
        for key in (
            "response_prefix_used",
            "response_prefix",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "repetition_penalty",
            "output_format",
            "enable_balanced_json_stopping",
            "stop_after_balanced_events_json",
        ):
            if key in generation_metadata:
                result[key] = generation_metadata[key]
    return result


def _instruction_copy_counts(text: str) -> tuple[int, int]:
    total = 0
    loop_count = 0
    for sentence in INSTRUCTION_SENTENCES:
        count = text.count(sentence)
        total += count
        if count > 1:
            loop_count += count - 1
    return total, loop_count


def _hit_max_new_tokens(
    *,
    hit_max_new_tokens: bool | None,
    hit_max_new_tokens_source: str | None,
    generated_token_count: int | None,
    generated_token_count_source: str | None,
    max_new_tokens: int | None,
    ended_with_eos: bool | None,
) -> tuple[bool | None, str | None]:
    if hit_max_new_tokens is not None:
        return bool(hit_max_new_tokens), hit_max_new_tokens_source
    if generated_token_count is None or max_new_tokens is None:
        return None, None
    hit = int(generated_token_count) >= int(max_new_tokens) and ended_with_eos is not True
    return hit, generated_token_count_source


def _prompt_token_limit_hit(
    *,
    prompt_token_count: int | None,
    prompt_token_budget: int | None,
) -> bool:
    if prompt_token_count is None or prompt_token_budget is None:
        return False
    return int(prompt_token_count) >= int(prompt_token_budget)


def _starts_with_candidate_fragment(stripped: str) -> bool:
    return bool(stripped and CANDIDATE_LINE_RE.match(stripped))


def _starts_with_instruction_text(stripped: str) -> bool:
    return any(stripped.startswith(sentence) for sentence in INSTRUCTION_SENTENCES)


def _parse_error_subtypes(
    *,
    stripped: str,
    starts_with_json_object: bool,
    starts_with_json_array: bool,
    brace_balance_state: str,
    candidate_line_copy_count: int,
    instruction_loop_count: int,
    hit_max_new_tokens: bool | None,
) -> list[str]:
    found: set[str] = set()
    if stripped and not starts_with_json_object and not starts_with_json_array:
        found.add("no_json_started")
    if candidate_line_copy_count > 0:
        found.add("candidate_list_continuation")
    if instruction_loop_count > 0:
        found.add("instruction_loop")
    if starts_with_json_array and brace_balance_state != "balanced":
        found.add("array_continuation_not_reconstructed")
    if hit_max_new_tokens is True:
        found.add("truncated_or_hit_max_new_tokens")
    if (starts_with_json_object or starts_with_json_array) and brace_balance_state != "balanced":
        found.add("malformed_json")
    return [subtype for subtype in PARSE_ERROR_SUBTYPE_ORDER if subtype in found]


def _primary_subtype(subtypes: list[str]) -> str | None:
    subtype_set = set(subtypes)
    for subtype in PRIMARY_SUBTYPE_PRIORITY:
        if subtype in subtype_set:
            return subtype
    return subtypes[0] if subtypes else None


def _brace_balance_state(text: str) -> str:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return "no_json_started"
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in stripped:
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
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return "malformed_or_extra_closer"
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return "malformed_or_extra_closer"
    if in_string:
        return "incomplete_string"
    if stack:
        return "incomplete_open"
    return "balanced" if _full_json_decodes(stripped) else "malformed_json"


def _full_json_decodes(text: str) -> bool:
    try:
        _, end_index = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return not text[end_index:].strip()


def _add_diagnostic_counts(total: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    for key, value in diagnostics.items():
        if isinstance(value, bool):
            total[key] = int(total.get(key, 0)) + int(value)
        elif isinstance(value, int):
            total[key] = int(total.get(key, 0)) + value
        elif isinstance(value, dict):
            nested = total.setdefault(key, {})
            if not isinstance(nested, dict):
                continue
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, bool):
                    nested[str(nested_key)] = int(nested.get(str(nested_key), 0)) + int(nested_value)
                elif isinstance(nested_value, int):
                    nested[str(nested_key)] = int(nested.get(str(nested_key), 0)) + nested_value


def _sort_diagnostic_counts(diagnostic_counts: dict[str, Any]) -> dict[str, Any]:
    sorted_counts: dict[str, Any] = {}
    for key in sorted(diagnostic_counts):
        value = diagnostic_counts[key]
        if isinstance(value, dict):
            sorted_counts[key] = dict(sorted(value.items()))
        else:
            sorted_counts[key] = value
    return sorted_counts


def _prompt_token_summary(
    values: list[int],
    *,
    prompt_token_budget: int | None,
    prompt_token_limit_hit_rows: int,
) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "max": None,
            "mean": None,
            "prompt_token_budget": prompt_token_budget,
            "rows_at_budget": 0,
            "prompt_token_limit_hit_rows": prompt_token_limit_hit_rows,
        }
    return {
        "count": len(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "prompt_token_budget": prompt_token_budget,
        "rows_at_budget": (
            sum(1 for value in values if value >= prompt_token_budget)
            if prompt_token_budget is not None
            else 0
        ),
        "prompt_token_limit_hit_rows": prompt_token_limit_hit_rows,
    }


def _collect_section_counts(target: dict[str, list[int]], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, raw_count in value.items():
        if isinstance(raw_count, int):
            target.setdefault(str(key), []).append(raw_count)


def _section_summary(counts: dict[str, list[int]]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for section, values in sorted(counts.items()):
        summary[section] = {
            "max": max(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
        }
    return summary


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]
