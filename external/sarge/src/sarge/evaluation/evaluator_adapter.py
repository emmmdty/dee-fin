from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionConversionReport:
    source_path: Path
    output_path: Path
    document_count: int
    event_count: int
    value_count: int
    skipped_value_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "document_count": self.document_count,
            "event_count": self.event_count,
            "value_count": self.value_count,
            "skipped_value_count": self.skipped_value_count,
        }


def convert_sarge_predictions_to_evaluator(
    source_path: str | Path,
    output_path: str | Path,
) -> PredictionConversionReport:
    source = Path(source_path)
    output = Path(output_path)
    rows = []
    event_count = 0
    value_count = 0
    skipped_value_count = 0

    for row in _read_jsonl(source):
        document_id = _document_id(row)
        predictions = []
        raw_events = row.get("events") if isinstance(row.get("events"), list) else row.get("predictions")
        for raw_event in raw_events or []:
            if not isinstance(raw_event, dict):
                continue
            event_type = _clean_text(raw_event.get("event_type"))
            if not event_type:
                continue
            arguments, stats = _convert_arguments(raw_event.get("arguments") or {})
            skipped_value_count += stats["skipped"]
            value_count += stats["kept"]
            event: dict[str, Any] = {"event_type": event_type, "arguments": arguments}
            if raw_event.get("record_id") is not None:
                event["record_id"] = str(raw_event["record_id"])
            predictions.append(event)
        event_count += len(predictions)
        rows.append({"document_id": document_id, "predictions": predictions})

    _write_jsonl(output, rows)
    return PredictionConversionReport(
        source_path=source,
        output_path=output,
        document_count=len(rows),
        event_count=event_count,
        value_count=value_count,
        skipped_value_count=skipped_value_count,
    )


def _convert_arguments(arguments: Any) -> tuple[dict[str, list[str]], dict[str, int]]:
    if not isinstance(arguments, dict):
        return {}, {"kept": 0, "skipped": 0}
    output: dict[str, list[str]] = {}
    kept = 0
    skipped = 0
    for role, raw_values in arguments.items():
        role_name = _clean_text(role)
        if not role_name:
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for raw_value in values:
            text = _value_text(raw_value)
            if text:
                output.setdefault(role_name, []).append(text)
                kept += 1
            else:
                skipped += 1
    return output, {"kept": kept, "skipped": skipped}


def _value_text(value: Any) -> str | None:
    if isinstance(value, dict):
        return _clean_text(value.get("text"))
    return _clean_text(value)


def _document_id(row: dict[str, Any]) -> str:
    for key in ("document_id", "doc_id", "id"):
        value = _clean_text(row.get(key))
        if value:
            return value
    raise ValueError("prediction row missing document_id/doc_id/id")


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
