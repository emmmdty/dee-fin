from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from sarge.surface_memory.types import SurfaceCandidate
from sarge.data.loader import V2DatasetDocument
from sarge.data.schema import DatasetSchema
from sarge.generation.prompt import DEFAULT_GETM_OUTPUT_FORMAT, build_getm_prompt, normalize_output_format
from sarge.record_planning import events_to_record_plan_output
from sarge.slot_planning.plan import SlotPlanDocument

IGNORE_LABEL = -100
DEFAULT_RESPONSE_PREFIX = '{"events":'

PROMPT_RENDER_OPTION_KEYS = frozenset(
    {
        "max_surface_candidates",
        "candidate_context_chars",
        "candidate_render_mode",
        "enable_candidate_filtering",
        "max_candidates_per_type",
        "dedupe_surface_candidates",
        "drop_low_value_company_fragments",
    }
)


def build_getm_sft_sample(
    document: V2DatasetDocument,
    schema: DatasetSchema,
    *,
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
    slot_plan: SlotPlanDocument | dict[str, Any] | None,
    output_format: str = DEFAULT_GETM_OUTPUT_FORMAT,
    prompt_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_output_format = normalize_output_format(output_format)
    candidate_list = list(surface_candidates)
    prompt = build_getm_prompt(
        dataset=schema.dataset_id,
        schema=schema,
        document=document.input,
        surface_candidates=candidate_list,
        slot_plan=slot_plan,
        output_format=normalized_output_format,
        **_prompt_render_options(prompt_options),
    )
    row: dict[str, Any] = {
        "doc_id": document.doc_id,
        "dataset": schema.dataset_id,
        "split": document.input.split,
        "prompt": prompt,
    }
    if document.input.split in {"dev", "test"}:
        return row
    if document.gold is None:
        raise ValueError("train GETM SFT samples require gold-visible documents")
    row["output"] = gold_events_to_getm_output(
        document.gold.events,
        schema,
        candidate_list,
        output_format=normalized_output_format,
    )
    return row


def build_sft_training_examples(
    *,
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    examples: list[dict[str, list[int]]] = []
    audit = {
        "row_count": 0,
        "example_count": 0,
        "dropped_example_count": 0,
        "truncated_prompt_count": 0,
        "truncated_answer_count": 0,
        "prompt_label_count": 0,
        "answer_label_count": 0,
        "all_prompt_labels_masked": True,
        "all_examples_have_answer_labels": True,
        "use_chat_template": _generation_bool(config, "use_chat_template", True),
        "use_response_prefix": _generation_bool(config, "use_response_prefix", True),
        "response_prefix": _generation_value(config, "response_prefix", DEFAULT_RESPONSE_PREFIX),
    }
    for raw_row in rows:
        row = dict(raw_row)
        audit["row_count"] += 1
        prompt = str(row.get("prompt") or "")
        output = row.get("output") if "output" in row else {"events": []}
        prefix_text = render_sft_user_assistant_prefix(tokenizer, prompt, config)
        target_text = render_sft_answer_suffix(tokenizer, output, config)
        prefix_ids = _encode_text(tokenizer, prefix_text)
        target_ids = _encode_text(tokenizer, target_text)
        if len(target_ids) >= max_seq_len:
            target_ids = target_ids[: max_seq_len - 1]
            audit["truncated_answer_count"] += 1
        max_prefix_tokens = max_seq_len - len(target_ids)
        if max_prefix_tokens < 1 or not target_ids:
            audit["dropped_example_count"] += 1
            audit["all_examples_have_answer_labels"] = False
            continue
        if len(prefix_ids) > max_prefix_tokens:
            prefix_ids = prefix_ids[:max_prefix_tokens]
            audit["truncated_prompt_count"] += 1
        input_ids = prefix_ids + target_ids
        labels = [IGNORE_LABEL] * len(prefix_ids) + target_ids
        if labels[: len(prefix_ids)] != [IGNORE_LABEL] * len(prefix_ids):
            audit["all_prompt_labels_masked"] = False
        if not any(label != IGNORE_LABEL for label in labels):
            audit["all_examples_have_answer_labels"] = False
        audit["prompt_label_count"] += len(prefix_ids)
        audit["answer_label_count"] += len(target_ids)
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    audit["example_count"] = len(examples)
    return examples, audit


def render_sft_user_assistant_prefix(tokenizer: Any, prompt: str, config: dict[str, Any] | None = None) -> str:
    generation = _generation_config(config)
    response_prefix = str(generation["response_prefix"])
    if generation["use_chat_template"] and hasattr(tokenizer, "apply_chat_template"):
        if generation["use_response_prefix"]:
            return str(
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_prefix},
                    ],
                    tokenize=False,
                    continue_final_message=True,
                )
            )
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    if generation["use_response_prefix"]:
        return f"{prompt}\n{response_prefix}"
    return prompt


def render_sft_training_text(tokenizer: Any, prompt: str, output: Any, config: dict[str, Any] | None = None) -> str:
    return render_sft_user_assistant_prefix(tokenizer, prompt, config) + render_sft_answer_suffix(
        tokenizer, output, config
    )


def render_sft_answer_suffix(tokenizer: Any, output: Any, config: dict[str, Any] | None = None) -> str:
    target = json.dumps(output or {"events": []}, ensure_ascii=False, separators=(",", ":"))
    generation = _generation_config(config)
    if generation["use_response_prefix"]:
        prefix = str(generation["response_prefix"])
        if target.startswith(prefix):
            target = target[len(prefix) :]
    return f"{target}{getattr(tokenizer, 'eos_token', None) or ''}"


def audit_sft_targets(rows: Iterable[dict[str, Any]], schema: DatasetSchema) -> dict[str, Any]:
    audit = {
        "row_count": 0,
        "event_count": 0,
        "target_schema_valid": True,
        "invalid_target_event_type_count": 0,
        "invalid_target_role_count": 0,
        "literal_role_key_count": 0,
    }
    for row in rows:
        audit["row_count"] += 1
        output = row.get("output") or {}
        events = output.get("events") if isinstance(output, dict) else []
        if not isinstance(events, list):
            audit["target_schema_valid"] = False
            continue
        for event in events:
            if not isinstance(event, dict):
                audit["target_schema_valid"] = False
                continue
            audit["event_count"] += 1
            event_type = str(event.get("event_type") or "")
            if event_type not in schema.event_roles:
                audit["invalid_target_event_type_count"] += 1
                audit["target_schema_valid"] = False
                continue
            arguments = event.get("arguments") or {}
            if not isinstance(arguments, dict):
                audit["target_schema_valid"] = False
                continue
            for role in arguments:
                if str(role) == "role":
                    audit["literal_role_key_count"] += 1
                if str(role) not in schema.event_roles[event_type]:
                    audit["invalid_target_role_count"] += 1
                    audit["target_schema_valid"] = False
    return audit


def gold_events_to_getm_output(
    events: list[dict[str, Any]],
    schema: DatasetSchema,
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
    *,
    output_format: str = DEFAULT_GETM_OUTPUT_FORMAT,
) -> dict[str, Any]:
    normalized_output_format = normalize_output_format(output_format)
    if normalized_output_format == "record_plan":
        return events_to_record_plan_output(events, schema)

    candidate_by_surface = _candidate_by_surface(surface_candidates)
    output_events: list[dict[str, Any]] = []
    for event in events:
        schema.validate_event_record(event)
        event_type = schema.validate_event_type(str(event.get("event_type", "")))
        arguments = event.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        output_arguments: dict[str, list[Any]] = {}
        for role in schema.event_roles[event_type]:
            values = arguments.get(role) or []
            if not isinstance(values, list):
                continue
            output_values = []
            for value in values:
                if not isinstance(value, dict):
                    continue
                text = str(value.get("text", "")).strip()
                if not text:
                    continue
                if normalized_output_format == "minimal_text":
                    output_values.append(text)
                else:
                    output_values.append(
                        {
                            "text": text,
                            "source_candidate_id": candidate_by_surface.get(text),
                        }
                    )
            if output_values:
                output_arguments[role] = output_values
        output_events.append({"event_type": event_type, "arguments": output_arguments})
    return {"events": output_events}


def _generation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    getm = (config or {}).get("getm") or {}
    raw = getm.get("generation") or {}
    output_format = normalize_output_format(getm.get("output_format", DEFAULT_GETM_OUTPUT_FORMAT))
    default_use_response_prefix = output_format != "record_plan"
    default_response_prefix = DEFAULT_RESPONSE_PREFIX if default_use_response_prefix else ""
    return {
        "use_chat_template": _as_bool(raw.get("use_chat_template", True)),
        "use_response_prefix": _as_bool(raw.get("use_response_prefix", default_use_response_prefix)),
        "response_prefix": str(raw.get("response_prefix", default_response_prefix)),
    }


def _generation_bool(config: dict[str, Any] | None, key: str, default: bool) -> bool:
    return bool(_generation_config(config).get(key, default))


def _generation_value(config: dict[str, Any] | None, key: str, default: Any) -> Any:
    return _generation_config(config).get(key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _candidate_by_surface(
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
) -> dict[str, str]:
    by_surface: dict[str, str] = {}
    for candidate in surface_candidates:
        if isinstance(candidate, SurfaceCandidate):
            surface = candidate.surface
            candidate_id = candidate.candidate_id
        elif isinstance(candidate, dict):
            surface = str(candidate.get("surface", ""))
            candidate_id = str(candidate.get("candidate_id", ""))
        else:
            continue
        surface = surface.strip()
        candidate_id = candidate_id.strip()
        if surface and candidate_id and surface not in by_surface:
            by_surface[surface] = candidate_id
    return by_surface


def _prompt_render_options(prompt_options: dict[str, Any] | None) -> dict[str, Any]:
    if not prompt_options:
        return {}
    return {
        key: value
        for key, value in prompt_options.items()
        if key in PROMPT_RENDER_OPTION_KEYS
    }
