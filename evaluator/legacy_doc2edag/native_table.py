from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.canonical.normalize import normalize_optional_text, normalize_text

FORMAT_NAME = "procnet_native_event_table_v1"


@dataclass(frozen=True)
class NativeEventTableDocument:
    document_id: str
    gold: tuple[tuple[tuple[str | None, ...], ...], ...]
    pred: tuple[tuple[tuple[str | None, ...], ...], ...]


@dataclass(frozen=True)
class NativeEventTable:
    dataset: str
    split: str | None
    seed: int | str | None
    event_types: tuple[str, ...]
    event_type_fields: dict[str, tuple[str, ...]]
    documents: tuple[NativeEventTableDocument, ...]


def load_native_event_table(path: str | Path) -> NativeEventTable:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_native_event_table(data)


def parse_native_event_table(data: Any) -> NativeEventTable:
    if not isinstance(data, dict):
        raise ValueError("native event table must be a JSON object")
    if data.get("format") != FORMAT_NAME:
        raise ValueError(f'native event table format must be "{FORMAT_NAME}"')

    dataset = _required_text(data, "dataset")
    split = normalize_optional_text(data.get("split"))
    seed = data.get("seed")
    if seed is not None and not isinstance(seed, (int, str)):
        raise ValueError("native event table field 'seed' must be int, string, or null")

    event_types = _parse_event_types(data.get("event_types"))
    event_type_fields = _parse_event_type_fields(data.get("event_type_fields"), event_types)

    raw_documents = data.get("documents")
    if not isinstance(raw_documents, list):
        raise ValueError("native event table field 'documents' must be a list")
    documents = tuple(
        _parse_document(row, index, event_types, event_type_fields) for index, row in enumerate(raw_documents)
    )
    return NativeEventTable(
        dataset=dataset,
        split=split,
        seed=seed,
        event_types=event_types,
        event_type_fields=event_type_fields,
        documents=documents,
    )


def _parse_event_types(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("native event table field 'event_types' must be a non-empty list")
    event_types = []
    seen = set()
    for index, raw_event_type in enumerate(value):
        event_type = normalize_optional_text(raw_event_type)
        if event_type is None:
            raise ValueError(f"event_types[{index}] must be a non-empty string")
        if event_type in seen:
            raise ValueError(f"duplicate event type in native event table: {event_type}")
        seen.add(event_type)
        event_types.append(event_type)
    return tuple(event_types)


def _parse_event_type_fields(value: Any, event_types: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError("native event table field 'event_type_fields' must be an object")
    parsed: dict[str, tuple[str, ...]] = {}
    for event_type in event_types:
        raw_roles = value.get(event_type)
        if not isinstance(raw_roles, list):
            raise ValueError(f"event_type_fields[{event_type!r}] must be a list")
        roles = []
        for role_index, raw_role in enumerate(raw_roles):
            role = normalize_optional_text(raw_role)
            if role is None:
                raise ValueError(f"event_type_fields[{event_type!r}][{role_index}] must be a non-empty string")
            roles.append(role)
        parsed[event_type] = tuple(roles)
    return parsed


def _parse_document(
    row: Any,
    document_index: int,
    event_types: tuple[str, ...],
    event_type_fields: dict[str, tuple[str, ...]],
) -> NativeEventTableDocument:
    if not isinstance(row, dict):
        raise ValueError(f"documents[{document_index}] must be an object")
    document_id = normalize_optional_text(row.get("document_id"))
    if document_id is None:
        raise ValueError(f"documents[{document_index}].document_id must be a non-empty string")

    gold = _parse_side(row.get("gold"), document_index, "gold", event_types, event_type_fields)
    pred = _parse_side(row.get("pred"), document_index, "pred", event_types, event_type_fields)
    return NativeEventTableDocument(document_id=document_id, gold=gold, pred=pred)


def _parse_side(
    value: Any,
    document_index: int,
    side_name: str,
    event_types: tuple[str, ...],
    event_type_fields: dict[str, tuple[str, ...]],
) -> tuple[tuple[tuple[str | None, ...], ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"documents[{document_index}].{side_name} must be a list")
    if len(value) != len(event_types):
        raise ValueError(
            f"documents[{document_index}].{side_name} must have {len(event_types)} event-type entries"
        )

    parsed_event_groups = []
    for event_index, raw_records in enumerate(value):
        event_type = event_types[event_index]
        role_count = len(event_type_fields[event_type])
        if not isinstance(raw_records, list):
            raise ValueError(f"documents[{document_index}].{side_name}[{event_index}] must be a list")
        parsed_records = []
        for record_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, list):
                raise ValueError(
                    f"documents[{document_index}].{side_name}[{event_index}][{record_index}] must be a list"
                )
            if len(raw_record) != role_count:
                raise ValueError(
                    f"documents[{document_index}].{side_name}[{event_index}][{record_index}] "
                    f"must have {role_count} role slots"
                )
            parsed_records.append(tuple(_parse_slot_value(slot) for slot in raw_record))
        parsed_event_groups.append(tuple(parsed_records))
    return tuple(parsed_event_groups)


def _parse_slot_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    return normalized


def _required_text(data: dict[str, Any], key: str) -> str:
    value = normalize_optional_text(data.get(key))
    if value is None:
        raise ValueError(f"native event table field {key!r} must be a non-empty string")
    return normalize_text(value)
