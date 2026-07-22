from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sarge.data.canonical import (
    CANONICAL_ARGUMENT_KEYS,
    CANONICAL_DOCUMENT_KEYS,
    CANONICAL_EVENT_RECORD_KEYS,
)
from sarge.data.jsonl import write_jsonl
from sarge.data.schema import DatasetSchema


def strip_auxiliary_fields(candidate_doc: dict[str, Any], *, schema: DatasetSchema | None = None) -> dict[str, Any]:
    if not isinstance(candidate_doc, dict):
        raise ValueError("prediction document must be a mapping")
    stripped_events = []
    raw_events = candidate_doc.get("events") or []
    if not isinstance(raw_events, list):
        raise ValueError("prediction document events must be a list")
    for event in raw_events:
        if not isinstance(event, dict):
            raise ValueError("prediction event must be a mapping")
        event_type = str(event.get("event_type", "")).strip()
        if schema is not None:
            schema.validate_event_type(event_type)
        stripped_events.append(
            {
                "event_type": event_type,
                "arguments": _strip_arguments(
                    event.get("arguments") or {},
                    event_type=event_type,
                    schema=schema,
                ),
            }
        )
    return {
        "doc_id": str(candidate_doc.get("doc_id", "")).strip(),
        "events": stripped_events,
    }


def validate_minimal_canonical_prediction(
    pred_doc: dict[str, Any],
    *,
    schema: DatasetSchema | None = None,
) -> None:
    if not isinstance(pred_doc, dict):
        raise ValueError("canonical prediction must be a mapping")
    doc_id = pred_doc.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("canonical prediction doc_id must be a non-empty string")
    _validate_keys(pred_doc, CANONICAL_DOCUMENT_KEYS, label="document")
    events = pred_doc.get("events")
    if not isinstance(events, list):
        raise ValueError("canonical prediction events must be a list")
    for event_index, event in enumerate(events, 1):
        _validate_event(event, event_index=event_index, schema=schema)


def export_predictions(
    pred_docs: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    schema: DatasetSchema | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for pred_doc in pred_docs:
        stripped = strip_auxiliary_fields(pred_doc, schema=schema)
        validate_minimal_canonical_prediction(stripped, schema=schema)
        rows.append(stripped)
    return write_jsonl(output_path, rows)


def _strip_arguments(
    arguments: Any,
    *,
    event_type: str,
    schema: DatasetSchema | None,
) -> dict[str, list[dict[str, str]]]:
    if not isinstance(arguments, dict):
        raise ValueError("prediction event arguments must be a mapping")
    stripped: dict[str, list[dict[str, str]]] = {}
    for role, values in arguments.items():
        role_name = str(role).strip()
        if not role_name:
            continue
        if schema is not None:
            schema.validate_role(event_type, role_name)
        if not isinstance(values, list):
            raise ValueError(f"prediction argument values must be a list: {role_name}")
        role_values: list[dict[str, str]] = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"prediction argument value must be a mapping: {role_name}")
            text = str(value.get("text", "")).strip()
            if text:
                role_values.append({"text": text})
        if role_values:
            stripped[role_name] = role_values
    return stripped


def _validate_event(event: Any, *, event_index: int, schema: DatasetSchema | None) -> None:
    if not isinstance(event, dict):
        raise ValueError(f"canonical prediction events[{event_index}] must be a mapping")
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError(f"canonical prediction events[{event_index}].event_type must be a non-empty string")
    if schema is not None:
        schema.validate_event_type(event_type)
    _validate_keys(event, CANONICAL_EVENT_RECORD_KEYS, label=f"event {event_index}")
    arguments = event.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"canonical prediction events[{event_index}].arguments must be a mapping")
    for role, values in arguments.items():
        role_name = str(role).strip()
        if not role_name:
            raise ValueError(f"canonical prediction events[{event_index}] contains an empty role")
        if schema is not None:
            schema.validate_role(event_type, role_name)
        if not isinstance(values, list):
            raise ValueError(f"canonical prediction events[{event_index}].arguments[{role_name!r}] must be a list")
        for value_index, value in enumerate(values, 1):
            _validate_argument(value, event_index=event_index, role=role_name, value_index=value_index)


def _validate_argument(value: Any, *, event_index: int, role: str, value_index: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"canonical prediction events[{event_index}].arguments[{role!r}][{value_index}] must be a mapping"
        )
    _validate_keys(value, CANONICAL_ARGUMENT_KEYS, label="argument")
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"canonical prediction events[{event_index}].arguments[{role!r}][{value_index}].text "
            "must be a non-empty string"
        )


def _validate_keys(payload: dict[str, Any], expected: frozenset[str], *, label: str) -> None:
    actual = set(payload)
    unexpected = actual - set(expected)
    missing = set(expected) - actual
    if unexpected:
        raise ValueError(f"Unexpected {label} keys: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"Missing {label} keys: {sorted(missing)}")
