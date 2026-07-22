from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord


@dataclass
class LoadResult:
    documents: list[CanonicalDocument]
    diagnostics: dict[str, int] = field(default_factory=dict)


def load_documents(path: str | Path, dataset: str | None = None) -> LoadResult:
    source = Path(path)
    diagnostics = {"parse_failure_count": 0, "missing_document_id_count": 0, "empty_prediction_count": 0}
    rows: list[Any] = []

    try:
        if source.suffix.lower() == ".jsonl":
            with source.open("r", encoding="utf-8", newline="") as handle:
                for line in handle:
                    stripped = line.rstrip("\n").rstrip("\r")
                    if not stripped:
                        continue
                    try:
                        rows.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        diagnostics["parse_failure_count"] += 1
        else:
            data = json.loads(source.read_text(encoding="utf-8"))
            if _looks_like_document_row(data):
                rows = [data]
            elif isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                for key in ("documents", "data", "records"):
                    if isinstance(data.get(key), list):
                        rows = data[key]
                        break
                else:
                    rows = [data]
    except json.JSONDecodeError:
        diagnostics["parse_failure_count"] += 1

    documents = []
    for index, row in enumerate(rows):
        try:
            document = adapt_document(row, dataset=dataset, index=index)
            if document.document_id.startswith("line_index_text_hash:"):
                diagnostics["missing_document_id_count"] += 1
            if not document.records:
                diagnostics["empty_prediction_count"] += 1
            documents.append(document)
        except (TypeError, ValueError, KeyError):
            diagnostics["parse_failure_count"] += 1
    return LoadResult(documents=documents, diagnostics=diagnostics)


def adapt_document(row: Any, dataset: str | None = None, index: int = 0) -> CanonicalDocument:
    if isinstance(row, list) and len(row) == 2 and isinstance(row[1], dict):
        return _adapt_chfinann(row)
    if not isinstance(row, dict):
        raise TypeError("document row must be a dict or ChFinAnn pair")

    document_id = _derive_document_id(row, dataset=dataset, index=index)
    if isinstance(row.get("event_list"), list):
        records = [_adapt_duee_event(document_id, event) for event in row["event_list"] if isinstance(event, dict)]
    elif isinstance(row.get("events"), list):
        records = [_adapt_generic_event(document_id, event) for event in row["events"] if isinstance(event, dict)]
    elif isinstance(row.get("predictions"), list):
        records = [_adapt_generic_event(document_id, event) for event in row["predictions"] if isinstance(event, dict)]
    else:
        records = []
    return CanonicalDocument(document_id=document_id, records=[record for record in records if record is not None])


def _looks_like_document_row(data: Any) -> bool:
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
        return "recguid_eventname_eventdict_list" in data[1]
    if isinstance(data, dict):
        return any(key in data for key in ("events", "event_list", "predictions", "document_id", "doc_id", "id"))
    return False


def _adapt_chfinann(row: list[Any]) -> CanonicalDocument:
    document_id = normalize_text(str(row[0]))
    payload = row[1]
    records = []
    for item in payload.get("recguid_eventname_eventdict_list", []):
        if not isinstance(item, list) or len(item) < 3 or not isinstance(item[2], dict):
            continue
        recguid, event_name, arguments = item[:3]
        records.append(
            CanonicalEventRecord(
                document_id=document_id,
                event_type=normalize_text(str(event_name)),
                record_id=str(recguid),
                arguments=_flat_arguments(arguments),
            )
        )
    return CanonicalDocument(document_id=document_id, records=records)


def _adapt_duee_event(document_id: str, event: dict[str, Any]) -> CanonicalEventRecord | None:
    event_type = normalize_optional_text(event.get("event_type"))
    if event_type is None:
        return None
    arguments: dict[str, list[str]] = {}
    for argument in event.get("arguments", []):
        if not isinstance(argument, dict):
            continue
        role = normalize_optional_text(argument.get("role"))
        value = normalize_optional_text(argument.get("argument"))
        if role and value:
            arguments.setdefault(role, []).append(value)
    return CanonicalEventRecord(
        document_id=document_id,
        event_type=event_type,
        record_id=_optional_record_id(event),
        arguments=arguments,
    )


def _adapt_generic_event(document_id: str, event: dict[str, Any]) -> CanonicalEventRecord | None:
    event_type = normalize_optional_text(event.get("event_type"))
    if event_type is None:
        return None

    if isinstance(event.get("arguments"), dict):
        arguments = _dict_arguments(event["arguments"])
    elif isinstance(event.get("arguments"), list):
        arguments = {}
        for argument in event["arguments"]:
            if isinstance(argument, dict):
                role = normalize_optional_text(argument.get("role"))
                value = normalize_optional_text(argument.get("argument") or argument.get("text") or argument.get("value"))
                if role and value:
                    arguments.setdefault(role, []).append(value)
    else:
        arguments = _flat_arguments(event)

    return CanonicalEventRecord(
        document_id=document_id,
        event_type=event_type,
        record_id=_optional_record_id(event),
        arguments=arguments,
    )


def _flat_arguments(event: dict[str, Any]) -> dict[str, list[str]]:
    ignored = {"event_type", "event_id", "id", "record_id", "trigger"}
    return _dict_arguments({key: value for key, value in event.items() if key not in ignored})


def _dict_arguments(arguments: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_role, raw_value in arguments.items():
        role = normalize_optional_text(raw_role)
        if role is None:
            continue
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            normalized = normalize_optional_text(value)
            if normalized:
                result.setdefault(role, []).append(normalized)
    return result


def _derive_document_id(row: dict[str, Any], dataset: str | None, index: int) -> str:
    for key in ("document_id", "doc_id", "id"):
        value = normalize_optional_text(row.get(key))
        if value:
            return value
    for key in ("content", "text"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    label = normalize_text(dataset or "dataset")
    return f"line_index_text_hash:{index}:{label}"


def _optional_record_id(event: dict[str, Any]) -> str | None:
    for key in ("record_id", "event_id", "id"):
        value = event.get(key)
        if value is not None:
            return str(value)
    return None
