from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sarge.surface_memory.types import SurfaceCandidate
from sarge.data.loader import V2DocumentInput
from sarge.data.schema import DatasetSchema
from sarge.slot_planning.plan import SlotPlanDocument, slot_plan_to_dict

FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "gold",
        "events",
        "events_gold",
        "raw_annotations",
        "arguments",
        "norm_text",
        "empty_roles",
        "event_id",
    }
)

GETM_OUTPUT_FORMATS = frozenset({"minimal_text", "argument_object", "record_plan"})
DEFAULT_GETM_OUTPUT_FORMAT = "minimal_text"
GETM_CANDIDATE_RENDER_MODES = frozenset({"full", "compact"})
DEFAULT_GETM_CANDIDATE_RENDER_MODE = "full"
GETM_PROMPT_BASELINE_MODES = frozenset(
    {
        "direct_json",
        "schema_only",
        "role_safe",
        "role_safe_surface_only",
        "role_safe_slot_plan_only",
        "role_safe_surface_memory",
    }
)
DEFAULT_GETM_PROMPT_BASELINE_MODE = "role_safe_surface_memory"
PROMPT_SECTION_NAMES = ("schema", "document", "candidates", "slot_plan", "instruction")

LOW_VALUE_COMPANY_FRAGMENTS = frozenset(
    {
        "公司",
        "本公司",
        "有限公司",
        "上市公司",
        "关于公司",
        "现将公司",
        "代表公司",
    }
)
GENERIC_INSTITUTION_SURFACES = (
    "深圳证券交易所",
    "深圳证券交易所上市公司回购股份实施细则",
    "中国证券登记结算有限责任公司",
)
COMPANY_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "集团有限公司",
    "控股有限公司",
    "证券股份有限公司",
    "银行股份有限公司",
)
HIGH_VALUE_RULE_PRIORITY = {
    "money": 0,
    "share_quantity": 1,
    "percentage": 2,
    "arabic_date": 3,
    "chinese_date": 3,
    "full_company_name": 4,
    "person_name": 5,
    "quoted_entity": 6,
    "stock_code": 7,
    "company_fragment": 8,
    "other": 9,
}


@dataclass(frozen=True)
class GetmPromptBuildResult:
    prompt: str
    selected_surface_candidates: list[dict[str, Any]]
    prompt_metadata: dict[str, Any]


@dataclass(frozen=True)
class _CandidateRender:
    text: str
    selected_candidates: list[dict[str, Any]]


def build_getm_prompt(
    *,
    dataset: str,
    schema: DatasetSchema,
    document: V2DocumentInput | dict[str, Any],
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
    slot_plan: SlotPlanDocument | dict[str, Any] | None,
    max_surface_candidates: int = 40,
    candidate_context_chars: int | None = None,
    candidate_render_mode: str = DEFAULT_GETM_CANDIDATE_RENDER_MODE,
    enable_candidate_filtering: bool = False,
    max_candidates_per_type: int | None = None,
    dedupe_surface_candidates: bool = False,
    drop_low_value_company_fragments: bool = False,
    use_response_prefix: bool = False,
    response_prefix: str | None = None,
    prompt_delimiter: str = "### RESPONSE_JSON",
    output_format: str = DEFAULT_GETM_OUTPUT_FORMAT,
    baseline_mode: str = DEFAULT_GETM_PROMPT_BASELINE_MODE,
) -> str:
    return build_getm_prompt_result(
        dataset=dataset,
        schema=schema,
        document=document,
        surface_candidates=surface_candidates,
        slot_plan=slot_plan,
        max_surface_candidates=max_surface_candidates,
        candidate_context_chars=candidate_context_chars,
        candidate_render_mode=candidate_render_mode,
        enable_candidate_filtering=enable_candidate_filtering,
        max_candidates_per_type=max_candidates_per_type,
        dedupe_surface_candidates=dedupe_surface_candidates,
        drop_low_value_company_fragments=drop_low_value_company_fragments,
        use_response_prefix=use_response_prefix,
        response_prefix=response_prefix,
        prompt_delimiter=prompt_delimiter,
        output_format=output_format,
        baseline_mode=baseline_mode,
    ).prompt


def build_getm_prompt_result(
    *,
    dataset: str,
    schema: DatasetSchema,
    document: V2DocumentInput | dict[str, Any],
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
    slot_plan: SlotPlanDocument | dict[str, Any] | None,
    max_surface_candidates: int = 40,
    candidate_context_chars: int | None = None,
    candidate_render_mode: str = DEFAULT_GETM_CANDIDATE_RENDER_MODE,
    enable_candidate_filtering: bool = False,
    max_candidates_per_type: int | None = None,
    dedupe_surface_candidates: bool = False,
    drop_low_value_company_fragments: bool = False,
    use_response_prefix: bool = False,
    response_prefix: str | None = None,
    prompt_delimiter: str = "### RESPONSE_JSON",
    output_format: str = DEFAULT_GETM_OUTPUT_FORMAT,
    baseline_mode: str = DEFAULT_GETM_PROMPT_BASELINE_MODE,
) -> GetmPromptBuildResult:
    doc = _document_payload(document)
    _reject_forbidden_keys(doc)
    slot_payload = _slot_plan_payload(slot_plan)
    _reject_forbidden_keys(slot_payload)
    normalized_output_format = normalize_output_format(output_format)
    normalized_render_mode = normalize_candidate_render_mode(candidate_render_mode)
    normalized_baseline_mode = normalize_prompt_baseline_mode(baseline_mode)
    include_schema = normalized_baseline_mode != "direct_json"
    include_surface_memory = normalized_baseline_mode in {
        "role_safe_surface_only",
        "role_safe_surface_memory",
    }
    include_slot_plan = normalized_baseline_mode in {
        "role_safe_slot_plan_only",
        "role_safe_surface_memory",
    }
    schema_text = _render_schema(schema) if include_schema else ""
    document_text = "\n".join((f"doc_id: {doc['doc_id']}", "content:", str(doc["content"])))
    if include_surface_memory:
        candidate_render = _render_surface_candidates(
            surface_candidates,
            max_items=max_surface_candidates,
            context_chars=candidate_context_chars,
            render_mode=normalized_render_mode,
            enable_filtering=enable_candidate_filtering,
            max_candidates_per_type=max_candidates_per_type,
            dedupe_surface_candidates=dedupe_surface_candidates,
            drop_low_value_company_fragments=drop_low_value_company_fragments,
        )
    else:
        candidate_render = _CandidateRender(text="", selected_candidates=[])
    slot_text = _render_slot_plan(slot_payload) if include_slot_plan else ""
    instruction_text = "\n".join(
        _render_instructions(
            use_response_prefix=use_response_prefix,
            response_prefix=response_prefix,
            prompt_delimiter=prompt_delimiter,
            output_format=normalized_output_format,
            schema=schema,
            baseline_mode=normalized_baseline_mode,
        )
    )
    section_texts = {
        "schema": schema_text,
        "document": document_text,
        "candidates": candidate_render.text,
        "slot_plan": slot_text,
        "instruction": instruction_text,
    }

    prompt_parts = [
        "[Dataset]",
        f"dataset_id: {dataset}",
        f"schema_dataset: {schema.schema_dataset}",
        "",
    ]
    if include_schema:
        prompt_parts.extend(("[Schema]", schema_text, ""))
    prompt_parts.extend(("[Document]", document_text, ""))
    if include_surface_memory:
        prompt_parts.extend(("[Surface Candidates]", candidate_render.text, ""))
    if include_slot_plan:
        prompt_parts.extend(("[Event Slot Plan]", slot_text, ""))
    prompt_parts.extend(("[Instruction]", instruction_text))
    prompt = "\n".join(prompt_parts)
    return GetmPromptBuildResult(
        prompt=prompt,
        selected_surface_candidates=candidate_render.selected_candidates,
        prompt_metadata={
            "baseline_mode": normalized_baseline_mode,
            "candidate_render_mode": normalized_render_mode,
            "candidate_context_chars": _normalize_context_chars(
                candidate_context_chars,
                render_mode=normalized_render_mode,
            ),
            "max_surface_candidates": int(max_surface_candidates),
            "enable_candidate_filtering": bool(enable_candidate_filtering),
            "max_candidates_per_type": (
                int(max_candidates_per_type) if max_candidates_per_type is not None else None
            ),
            "dedupe_surface_candidates": bool(dedupe_surface_candidates),
            "drop_low_value_company_fragments": bool(drop_low_value_company_fragments),
            "selected_surface_candidate_count": len(candidate_render.selected_candidates),
            "prompt_section_char_counts": {
                section: len(section_texts.get(section, "")) for section in PROMPT_SECTION_NAMES
            },
        },
    )


def normalize_output_format(output_format: str | None) -> str:
    normalized = str(output_format or DEFAULT_GETM_OUTPUT_FORMAT).strip()
    if normalized not in GETM_OUTPUT_FORMATS:
        raise ValueError(
            f"getm output_format must be one of {sorted(GETM_OUTPUT_FORMATS)}; got {output_format!r}"
        )
    return normalized


def normalize_candidate_render_mode(candidate_render_mode: str | None) -> str:
    normalized = str(candidate_render_mode or DEFAULT_GETM_CANDIDATE_RENDER_MODE).strip()
    if normalized not in GETM_CANDIDATE_RENDER_MODES:
        raise ValueError(
            "getm prompt candidate_render_mode must be one of "
            f"{sorted(GETM_CANDIDATE_RENDER_MODES)}; got {candidate_render_mode!r}"
        )
    return normalized


def normalize_prompt_baseline_mode(baseline_mode: str | None) -> str:
    normalized = str(baseline_mode or DEFAULT_GETM_PROMPT_BASELINE_MODE).strip()
    if normalized not in GETM_PROMPT_BASELINE_MODES:
        raise ValueError(
            "getm prompt baseline_mode must be one of "
            f"{sorted(GETM_PROMPT_BASELINE_MODES)}; got {baseline_mode!r}"
        )
    return normalized


def _document_payload(document: V2DocumentInput | dict[str, Any]) -> dict[str, Any]:
    if isinstance(document, V2DocumentInput):
        return {
            "doc_id": document.doc_id,
            "dataset_id": document.dataset_id,
            "dataset": document.dataset,
            "split": document.split,
            "content": document.content,
        }
    if not isinstance(document, dict):
        raise TypeError("document must be V2DocumentInput or mapping")
    return dict(document)


def _slot_plan_payload(slot_plan: SlotPlanDocument | dict[str, Any] | None) -> dict[str, Any]:
    if slot_plan is None:
        return {"slots": []}
    if isinstance(slot_plan, SlotPlanDocument):
        return slot_plan_to_dict(slot_plan)
    if not isinstance(slot_plan, dict):
        raise TypeError("slot_plan must be SlotPlanDocument, mapping, or None")
    return dict(slot_plan)


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_PROMPT_KEYS:
                raise ValueError(f"forbidden prompt key: {key}")
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _render_schema(schema: DatasetSchema) -> str:
    lines = []
    for event_type, roles in schema.event_roles.items():
        lines.append(f"- {event_type}: {', '.join(roles)}")
    return "\n".join(lines) if lines else "(none)"


def _render_surface_candidates(
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
    *,
    max_items: int,
    context_chars: int | None,
    render_mode: str,
    enable_filtering: bool,
    max_candidates_per_type: int | None,
    dedupe_surface_candidates: bool,
    drop_low_value_company_fragments: bool,
) -> _CandidateRender:
    selected = _select_surface_candidates(
        surface_candidates,
        max_items=max_items,
        enable_filtering=enable_filtering,
        max_candidates_per_type=max_candidates_per_type,
        dedupe_surface_candidates=dedupe_surface_candidates,
        drop_low_value_company_fragments=drop_low_value_company_fragments,
    )
    if render_mode == "compact":
        return _CandidateRender(
            text=_render_compact_surface_candidates(selected, context_chars=context_chars),
            selected_candidates=selected,
        )
    return _CandidateRender(
        text=_render_full_surface_candidates(selected, context_chars=context_chars),
        selected_candidates=selected,
    )


def _select_surface_candidates(
    surface_candidates: Iterable[SurfaceCandidate | dict[str, Any]],
    *,
    max_items: int,
    enable_filtering: bool,
    max_candidates_per_type: int | None,
    dedupe_surface_candidates: bool,
    drop_low_value_company_fragments: bool,
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    payloads: list[dict[str, Any]] = []
    for candidate in surface_candidates:
        payload = _candidate_payload(candidate)
        candidate_id = str(payload.get("candidate_id", "")).strip()
        surface = str(payload.get("surface", "")).strip()
        if not candidate_id or not surface:
            continue
        payload["candidate_id"] = candidate_id
        payload["surface"] = surface
        payload["_original_index"] = len(payloads)
        payload["_candidate_type"] = _candidate_type(payload)
        payload["_candidate_priority"] = _candidate_priority(payload)
        if enable_filtering and drop_low_value_company_fragments and _should_drop_low_value_candidate(payload):
            continue
        payloads.append(payload)

    if enable_filtering:
        payloads = sorted(
            payloads,
            key=lambda item: (
                int(item.get("_candidate_priority", HIGH_VALUE_RULE_PRIORITY["other"])),
                int(item.get("_original_index", 0)),
            ),
        )
    if dedupe_surface_candidates:
        payloads = _dedupe_by_surface(payloads)
    if enable_filtering and max_candidates_per_type is not None:
        payloads = _cap_by_type(payloads, max_candidates_per_type=max_candidates_per_type)
    return [_strip_private_candidate_keys(payload) for payload in payloads[:max_items]]


def _render_full_surface_candidates(
    selected: list[dict[str, Any]],
    *,
    context_chars: int | None,
) -> str:
    lines = []
    for payload in selected:
        candidate_id = str(payload.get("candidate_id", "")).strip()
        surface = str(payload.get("surface", "")).strip()
        context = _bounded_context(payload.get("context"), context_chars=context_chars, render_mode="full")
        chunk_id = str(payload.get("chunk_id", "")).strip()
        parts = [f"id={candidate_id}", f"text={surface}"]
        if chunk_id:
            parts.append(f"chunk={chunk_id}")
        if context:
            parts.append(f"context={context}")
        if payload.get("role_score") is not None:
            parts.append(f"role_score={float(payload['role_score']):.4f}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines) if lines else "(none)"


def _render_compact_surface_candidates(
    selected: list[dict[str, Any]],
    *,
    context_chars: int | None,
) -> str:
    lines = []
    for index, payload in enumerate(selected):
        surface = str(payload.get("surface", "")).strip()
        if not surface:
            continue
        line = f"[c{index}] {surface}"
        context = _bounded_context(payload.get("context"), context_chars=context_chars, render_mode="compact")
        if context:
            line += f" | ctx={context}"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def _candidate_payload(candidate: SurfaceCandidate | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, SurfaceCandidate):
        return {
            "candidate_id": candidate.candidate_id,
            "surface": candidate.surface,
            "context": candidate.context,
            "chunk_id": candidate.chunk_id,
            "role_score": candidate.role_score,
            "metadata": dict(candidate.metadata),
        }
    if not isinstance(candidate, dict):
        raise TypeError("surface candidates must be SurfaceCandidate or mapping")
    return dict(candidate)


def _candidate_type(payload: dict[str, Any]) -> str:
    rule_names = _candidate_rule_names(payload)
    surface = str(payload.get("surface", "")).strip()
    if "money" in rule_names or _looks_like_money(surface):
        return "money"
    if "share_quantity" in rule_names or _looks_like_share_quantity(surface):
        return "share_quantity"
    if "percentage" in rule_names or _looks_like_percentage(surface):
        return "percentage"
    if "arabic_date" in rule_names or _looks_like_arabic_date(surface):
        return "arabic_date"
    if "chinese_date" in rule_names:
        return "chinese_date"
    if _looks_like_full_company_name(surface):
        return "full_company_name"
    if "person_after_title" in rule_names or "person_with_honorific" in rule_names:
        return "person_name"
    if "quoted_entity" in rule_names:
        return "quoted_entity"
    if "stock_code" in rule_names:
        return "stock_code"
    if "company_fragment" in rule_names:
        return "company_fragment"
    return "other"


def _candidate_rule_names(payload: dict[str, Any]) -> set[str]:
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw_rule_names = metadata.get("rule_names") or []
    if isinstance(raw_rule_names, str):
        return {raw_rule_names}
    if isinstance(raw_rule_names, (list, tuple, set)):
        return {str(name) for name in raw_rule_names if str(name).strip()}
    return set()


def _candidate_priority(payload: dict[str, Any]) -> int:
    candidate_type = str(payload.get("_candidate_type") or _candidate_type(payload))
    priority = HIGH_VALUE_RULE_PRIORITY.get(candidate_type, HIGH_VALUE_RULE_PRIORITY["other"])
    if _is_generic_institution(str(payload.get("surface", ""))):
        priority += 20
    return priority


def _should_drop_low_value_candidate(payload: dict[str, Any]) -> bool:
    surface = str(payload.get("surface", "")).strip()
    candidate_type = str(payload.get("_candidate_type") or _candidate_type(payload))
    if candidate_type == "company_fragment" and surface in LOW_VALUE_COMPANY_FRAGMENTS:
        return True
    return candidate_type == "company_fragment" and _is_overlong_company_fragment(surface)


def _dedupe_by_surface(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for payload in payloads:
        surface_key = _normalize_surface_key(str(payload.get("surface", "")))
        if not surface_key or surface_key in seen:
            continue
        seen.add(surface_key)
        deduped.append(payload)
    return deduped


def _cap_by_type(
    payloads: list[dict[str, Any]],
    *,
    max_candidates_per_type: int,
) -> list[dict[str, Any]]:
    if max_candidates_per_type < 0:
        return payloads
    counts: dict[str, int] = {}
    capped = []
    for payload in payloads:
        candidate_type = str(payload.get("_candidate_type") or _candidate_type(payload))
        current_count = counts.get(candidate_type, 0)
        if current_count >= max_candidates_per_type:
            continue
        counts[candidate_type] = current_count + 1
        capped.append(payload)
    return capped


def _strip_private_candidate_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def _bounded_context(value: Any, *, context_chars: int | None, render_mode: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized_context_chars = _normalize_context_chars(context_chars, render_mode=render_mode)
    if normalized_context_chars is None:
        return text
    if normalized_context_chars <= 0:
        return ""
    return text[:normalized_context_chars]


def _normalize_context_chars(context_chars: int | None, *, render_mode: str) -> int | None:
    if context_chars is None:
        return None if render_mode == "full" else 0
    return max(0, int(context_chars))


def _normalize_surface_key(surface: str) -> str:
    return re.sub(r"\s+", "", surface).strip()


def _is_generic_institution(surface: str) -> bool:
    return any(marker in surface for marker in GENERIC_INSTITUTION_SURFACES)


def _is_overlong_company_fragment(surface: str) -> bool:
    if len(_chinese_chars(surface)) <= 30:
        return False
    if _looks_like_money(surface) or _looks_like_share_quantity(surface) or _looks_like_percentage(surface):
        return False
    if _looks_like_arabic_date(surface) or _looks_like_full_company_name(surface):
        return False
    return True


def _looks_like_full_company_name(surface: str) -> bool:
    stripped = surface.strip()
    if stripped in LOW_VALUE_COMPANY_FRAGMENTS:
        return False
    return any(stripped.endswith(suffix) and len(stripped) > len(suffix) for suffix in COMPANY_SUFFIXES)


def _looks_like_money(surface: str) -> bool:
    return bool(
        re.search(
            r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:万|亿)?\s*(?:人民币|亿元|万元|元|美元|万美元)",
            surface,
        )
    )


def _looks_like_share_quantity(surface: str) -> bool:
    return bool(re.search(r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:万|亿)?\s*(?:股|股份|股权)", surface))


def _looks_like_percentage(surface: str) -> bool:
    return bool(re.search(r"[0-9]+(?:\.[0-9]+)?\s*[%％]", surface))


def _looks_like_arabic_date(surface: str) -> bool:
    return bool(
        re.search(
            r"[0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}",
            surface,
        )
    )


def _chinese_chars(surface: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", surface))


def _render_slot_plan(slot_payload: dict[str, Any]) -> str:
    raw_slots = slot_payload.get("slots") or []
    if not isinstance(raw_slots, list) or not raw_slots:
        return "(none)"

    lines = []
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, dict):
            continue
        event_type = str(raw_slot.get("event_type", "")).strip()
        if not event_type:
            continue
        slot_id = int(raw_slot.get("slot_id", 0))
        count_confidence = float(raw_slot.get("count_confidence", 0.0))
        role_prior = raw_slot.get("role_prior") or {}
        role_text = ", ".join(
            f"{role}:{float(score):.3f}" for role, score in role_prior.items()
        )
        supporting = ", ".join(str(item) for item in (raw_slot.get("supporting_candidates") or []))
        line = f"- {event_type} slot_id={slot_id} count_confidence={count_confidence:.3f}"
        if role_text:
            line += f" role_prior={role_text}"
        if supporting:
            line += f" supporting_candidates={supporting}"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def _render_instructions(
    *,
    use_response_prefix: bool,
    response_prefix: str | None,
    prompt_delimiter: str,
    output_format: str,
    schema: DatasetSchema,
    baseline_mode: str,
) -> tuple[str, ...]:
    prefix = str(response_prefix or "").strip()
    delimiter = str(prompt_delimiter or "### RESPONSE_JSON").strip() or "### RESPONSE_JSON"
    normalized_baseline_mode = normalize_prompt_baseline_mode(baseline_mode)
    shape = _event_shape(output_format, schema, baseline_mode=normalized_baseline_mode)
    if use_response_prefix and _compact(prefix) == '{"events":[':
        return (
            "The assistant response has already started with the configured response prefix.",
            f"Configured response prefix: {prefix}.",
            "Continue only the JSON continuation.",
            "If there are no valid events, output exactly ]}.",
            "Do not repeat schema, document, candidates, or instructions.",
            *_shared_event_instructions(
                shape=shape,
                array_continuation=True,
                output_format=output_format,
                schema=schema,
                baseline_mode=normalized_baseline_mode,
            ),
            f"Begin your answer after the delimiter below; output only the continuation after {prefix}.",
            delimiter,
        )
    if use_response_prefix and _compact(prefix) == '{"events":':
        return (
            "The assistant response has already started with the configured response prefix.",
            f"Configured response prefix: {prefix}.",
            "Continue only the JSON continuation.",
            "Continue with the JSON array value only, then close the top-level object.",
            "If there are no valid events, output exactly []}.",
            "Do not repeat schema, document, candidates, or instructions.",
            *_shared_event_instructions(
                shape=shape,
                array_continuation=True,
                output_format=output_format,
                schema=schema,
                baseline_mode=normalized_baseline_mode,
            ),
            f"Begin your answer after the delimiter below; output only the continuation after {prefix}.",
            delimiter,
        )
    if use_response_prefix and prefix:
        return (
            "The assistant response has already started with the configured response prefix.",
            f"Configured response prefix: {prefix}.",
            "Continue only the JSON continuation.",
            "Do not repeat schema, document, candidates, or instructions.",
            *_shared_event_instructions(
                shape=shape,
                array_continuation=True,
                output_format=output_format,
                schema=schema,
                baseline_mode=normalized_baseline_mode,
            ),
            f"Begin your answer after the delimiter below; output only the continuation after {prefix}.",
            delimiter,
        )
    return (
        "Return ONLY one valid JSON object. Do not wrap it in markdown fences. Do not explain.",
        "Do not repeat the document, schema, candidates, or instruction. Do not output YAML.",
        _top_level_shape_instruction(output_format),
        "Generate strict JSON only in this shape:",
        _full_shape(output_format, schema, baseline_mode=normalized_baseline_mode),
        *_shared_event_instructions(
            shape=shape,
            array_continuation=False,
            output_format=output_format,
            schema=schema,
            baseline_mode=normalized_baseline_mode,
        ),
        _no_event_instruction(normalized_baseline_mode, output_format=output_format),
        _non_gold_instruction(normalized_baseline_mode),
        "Begin your answer after the delimiter below; output JSON only and do not repeat any prompt section.",
        delimiter,
    )


def _shared_event_instructions(
    *,
    shape: str,
    array_continuation: bool,
    output_format: str,
    schema: DatasetSchema,
    baseline_mode: str,
) -> tuple[str, ...]:
    normalized_baseline_mode = normalize_prompt_baseline_mode(baseline_mode)
    shape_instruction = f"Generate event objects only in this shape: {shape}."
    instructions = [
        "Do not wrap it in markdown fences. Do not explain.",
        shape_instruction if array_continuation else "Use only the requested JSON shape.",
    ]
    if normalized_baseline_mode == "direct_json":
        instructions.extend(
            (
                "Infer event records directly from the document text.",
                "Use concise event type and argument role names in the JSON output.",
            )
        )
    elif normalized_baseline_mode == "schema_only":
        instructions.extend(
            (
                "Use event_type and argument role labels from the schema text when possible.",
                "Do not translate, fold, alias, or normalize schema labels.",
            )
        )
    else:
        instructions.extend(
            (
                "Do not output fields outside the event type and role schema.",
                "Use the original dataset schema labels exactly.",
                "arguments keys must be valid roles for the generated event_type.",
                *_schema_role_instruction_lines(schema),
                "event_type and role names must be copied exactly from the schema; do not translate, fold, alias,"
                " or normalize labels.",
            )
        )
    if normalized_baseline_mode in {"role_safe_surface_only", "role_safe_surface_memory"}:
        instructions.append(
            "Prefer copying text from Surface Candidates when the candidate text matches the document evidence."
        )
    if output_format == "record_plan":
        instructions.extend(
            (
                "First populate `record_plan` with one item per event instance.",
                "`record_plan` items must use stable `record_id` values such as R1, R2, and R3.",
                "Each generated event must include the `record_id` of its planned event instance.",
                "Use private anchors such as counterparties, dates, quantities, or amounts to separate same-type records.",
            )
        )
    if normalized_baseline_mode != "direct_json":
        instructions.append("Do not invent event types or roles.")
    instructions.append(_non_gold_instruction(normalized_baseline_mode))
    if output_format == "argument_object":
        instructions.insert(
            -2,
            "source_candidate_id is an internal candidate-stage field; use it only inside argument objects when known.",
        )
    return tuple(instructions)


def _schema_role_instruction_lines(schema: DatasetSchema) -> tuple[str, ...]:
    lines = []
    for event_type, roles in schema.event_roles.items():
        lines.append(f"For {event_type}, valid argument keys are: {', '.join(roles)}.")
    return tuple(lines)


def _full_shape(output_format: str, schema: DatasetSchema, *, baseline_mode: str) -> str:
    if output_format == "record_plan":
        event_type, role = _example_event_type_and_role(schema)
        return (
            f'{{"record_plan":[{{"record_id":"R1","event_type":"{event_type}",'
            f'"anchors":{{"{role}":["..."]}}}}],"events":['
            + _event_shape(output_format, schema, baseline_mode=baseline_mode)
            + "]}"
        )
    return '{"events":[' + _event_shape(output_format, schema, baseline_mode=baseline_mode) + "]}"


def _event_shape(output_format: str, schema: DatasetSchema, *, baseline_mode: str) -> str:
    if normalize_prompt_baseline_mode(baseline_mode) == "direct_json":
        event_type, role = "EventType", "RoleName"
    else:
        event_type, role = _example_event_type_and_role(schema)
    if output_format == "minimal_text":
        return f'{{"event_type":"{event_type}","arguments":{{"{role}":["..."]}}}}'
    if output_format == "record_plan":
        return f'{{"record_id":"R1","event_type":"{event_type}","arguments":{{"{role}":["..."]}}}}'
    return (
        f'{{"event_type":"{event_type}","arguments":{{"{role}":[{{"text":"..."}},'
        '{"text":"...","source_candidate_id":"candidate-id-or-null"}]}}'
    )


def _top_level_shape_instruction(output_format: str) -> str:
    if output_format == "record_plan":
        return "The top-level JSON object must contain only the `record_plan` and `events` keys."
    return "The top-level JSON object must contain only the `events` key."


def _example_event_type_and_role(schema: DatasetSchema) -> tuple[str, str]:
    for event_type, roles in schema.event_roles.items():
        if roles:
            return event_type, roles[0]
    return "EventType", "RoleName"


def _no_event_instruction(baseline_mode: str, *, output_format: str = DEFAULT_GETM_OUTPUT_FORMAT) -> str:
    if output_format == "record_plan":
        return (
            "Do not invent event types or roles. If no valid event is present, "
            'return exactly {"record_plan": [], "events": []}.'
        )
    if normalize_prompt_baseline_mode(baseline_mode) == "direct_json":
        return 'If no event is present, return exactly {"events": []}.'
    return 'Do not invent event types or roles. If no valid event is present, return exactly {"events": []}.'


def _non_gold_instruction(baseline_mode: str) -> str:
    if normalize_prompt_baseline_mode(baseline_mode) in {
        "role_safe_slot_plan_only",
        "role_safe_surface_memory",
    }:
        return "The prompt contains no gold event records; treat the Event Slot Plan as a non-gold prior only."
    return "The prompt contains no gold event records."


def _compact(text: str) -> str:
    return "".join(text.split())
