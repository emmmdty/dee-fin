from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_CANDIDATE_LINE_RE = re.compile(
    r"(?m)^\s*(?:-\s*)?(?:id=)?=?[^\s|:]+:(?:csg|ec):[^\s|]+"
    r"\s*\|\s*(?:event_type=[^|]+\|\s*)?text=[^|]*\|\s*(?:chunk=[^|]*\|\s*)?context=",
)


@dataclass(frozen=True)
class BalancedJsonStoppingResult:
    raw_output: str
    stopped_output: str
    stop_reason: str
    balanced_stop_applied: bool


def apply_balanced_json_stopping(
    raw_output: str,
    *,
    enabled: bool,
    stop_after_balanced_events_json: bool,
    response_prefix: str | None = None,
    response_prefix_used: bool = False,
    hit_max_new_tokens: bool | None = None,
    ended_with_eos: bool | None = None,
) -> BalancedJsonStoppingResult:
    raw_text = str(raw_output or "")
    if not enabled or not stop_after_balanced_events_json:
        return _result(raw_text, raw_text, "disabled")
    if _starts_with_candidate_line(raw_text):
        return _result(raw_text, raw_text, _fallback_stop_reason(hit_max_new_tokens, ended_with_eos))

    for reconstructed, raw_start_index in _reconstructed_candidates(
        raw_text,
        response_prefix=response_prefix,
        response_prefix_used=response_prefix_used,
    ):
        closed_end = _balanced_events_json_end(reconstructed)
        if closed_end is None or closed_end < raw_start_index:
            continue
        stopped_end = closed_end - raw_start_index
        stopped_text = raw_text[:stopped_end].rstrip()
        return _result(raw_text, stopped_text, "balanced_json_closed")

    return _result(raw_text, raw_text, _fallback_stop_reason(hit_max_new_tokens, ended_with_eos))


def _result(raw_output: str, stopped_output: str, stop_reason: str) -> BalancedJsonStoppingResult:
    return BalancedJsonStoppingResult(
        raw_output=raw_output,
        stopped_output=stopped_output,
        stop_reason=stop_reason,
        balanced_stop_applied=stop_reason == "balanced_json_closed" and stopped_output != raw_output,
    )


def _fallback_stop_reason(hit_max_new_tokens: bool | None, ended_with_eos: bool | None) -> str:
    if hit_max_new_tokens is True:
        return "max_new_tokens"
    if ended_with_eos is True:
        return "eos"
    return "no_stop"


def _starts_with_candidate_line(raw_text: str) -> bool:
    return bool(_CANDIDATE_LINE_RE.match(raw_text.lstrip()))


def _reconstructed_candidates(
    raw_text: str,
    *,
    response_prefix: str | None,
    response_prefix_used: bool,
) -> list[tuple[str, int]]:
    stripped = raw_text.lstrip()
    candidates: list[tuple[str, int]] = [(raw_text, 0)]
    prefix = str(response_prefix or "")
    if not response_prefix_used or not prefix:
        return candidates
    if stripped.startswith(prefix.strip()):
        return candidates
    if _looks_like_full_events_json(stripped):
        return candidates

    compact_prefix = _compact(prefix)
    if compact_prefix == '{"events":' and stripped.startswith("["):
        candidates.append((f"{prefix}{raw_text}", len(prefix)))
    elif compact_prefix == '{"events":[' and stripped.startswith(("{", "]")):
        candidates.append((f"{prefix}{raw_text}", len(prefix)))
    return candidates


def _looks_like_full_events_json(text: str) -> bool:
    end = _balanced_events_json_end(text)
    return end is not None and not text[end:].strip()


def _balanced_events_json_end(text: str) -> int | None:
    start = _first_non_space(text)
    if start is None or text[start] != "{":
        return None
    end = _balanced_json_value_end(text, start)
    if end is None:
        return None
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    return end if isinstance(events, list) else None


def _balanced_json_value_end(text: str, start: int) -> int | None:
    stack: list[str] = []
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
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return None
            if not stack:
                return index + 1
    return None


def _first_non_space(text: str) -> int | None:
    for index, char in enumerate(text):
        if not char.isspace():
            return index
    return None


def _compact(value: Any) -> str:
    return "".join(str(value or "").split())
