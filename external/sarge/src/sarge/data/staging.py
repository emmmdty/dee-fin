"""Convert copied processed jsonl + schema assets into SARGE-staged form.

The copied processed directory uses one schema shape (list of
``{event_type, role_list}``) and dataset-specific document layouts (DuEE-Fin
keeps text+event_list, ChFinAnn keeps a ``[doc_id, payload]`` pair, DocFEE
keeps content+events). SARGE pipelines expect a single canonical jsonl shape
(``{doc_id, dataset, split, content, events}``) and a single schema shape
(``{dataset, canonical_version, event_types: [{event_type, roles}]}``).

This module strips sampling stratification, train-focus duplication, manifest
writing, and evaluator-gold writing (not required for inference). Use
``stage_dataset()`` to materialize a staging directory that
``sarge.pipeline.infer.run_inference`` can read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGED_CANONICAL_VERSION = "sarge.staged.v1"


@dataclass(frozen=True)
class ConvertedDocument:
    canonical_row: dict[str, Any]


def stage_dataset(
    *,
    dataset: str,
    processed_root: str | Path,
    output_root: str | Path,
    splits: Sequence[str] = ("train", "dev"),
    limit: int | None = None,
) -> Path:
    """Materialize a SARGE-readable staging dir from a copied processed dataset.

    Writes ``<output_root>/<dataset>/{schema.json,<split>.jsonl}`` files in the
    shape expected by ``sarge.data.schema.load_schema`` and
    ``sarge.data.loader.load_documents``. Returns the staged dataset root.
    """
    processed_dir = Path(processed_root) / dataset
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"missing processed dataset directory: {processed_dir}")
    staged_dir = Path(output_root) / dataset
    staged_dir.mkdir(parents=True, exist_ok=True)

    event_roles = load_event_roles(processed_dir / "schema.json", dataset=dataset)
    schema_payload = to_sarge_schema(dataset, event_roles)
    _write_json(staged_dir / "schema.json", schema_payload)

    for split in splits:
        source_path = _source_split_path(processed_dir, dataset=dataset, split=split)
        rows = list(_iter_source_rows(source_path, limit=limit))
        converted = [convert_processed_document(row, dataset=dataset, split=split, index=idx) for idx, row in enumerate(rows)]
        out_path = staged_dir / f"{split}.jsonl"
        _write_jsonl(out_path, (c.canonical_row for c in converted))
    return staged_dir


def convert_processed_document(row: Any, *, dataset: str, split: str, index: int) -> ConvertedDocument:
    if dataset == "DuEE-Fin-dev500":
        return _convert_duee_document(row, dataset=dataset, split=split, index=index)
    if dataset == "ChFinAnn-Doc2EDAG":
        return _convert_chfinann_document(row, dataset=dataset, split=split, index=index)
    if dataset == "DocFEE-dev1000":
        return _convert_docfee_document(row, dataset=dataset, split=split, index=index)
    raise ValueError(f"unsupported dataset for SARGE staging: {dataset}")


def load_event_roles(schema_path: str | Path, *, dataset: str) -> dict[str, list[str]]:
    """Read schema.json (multiple shapes accepted) and return a flat
    ``{event_type: [role, ...]}`` dict in declaration order."""
    payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _event_roles_from_list(payload)
    if isinstance(payload, dict) and isinstance(payload.get("properties"), dict):
        return {
            str(event_type): list((spec.get("properties") or {}).keys())
            for event_type, spec in payload["properties"].items()
            if isinstance(spec, dict)
        }
    if isinstance(payload, dict):
        return {
            str(event_type): [str(role) for role in roles]
            for event_type, roles in payload.items()
            if isinstance(roles, list)
        }
    raise ValueError(f"unsupported schema shape: {schema_path}")


def to_sarge_schema(dataset: str, event_roles: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "canonical_version": STAGED_CANONICAL_VERSION,
        "event_types": [
            {"event_type": event_type, "roles": list(roles)}
            for event_type, roles in event_roles.items()
        ],
    }


# ---------------------------------------------------------------------------
# Document converters
# ---------------------------------------------------------------------------

def _convert_duee_document(row: Any, *, dataset: str, split: str, index: int) -> ConvertedDocument:
    if not isinstance(row, dict):
        raise ValueError(f"{dataset}/{split}/{index}: expected JSON object")
    doc_id = _document_id(row, dataset=dataset, index=index)
    events = [_event_from_duee(item, record_id=str(i)) for i, item in enumerate(row.get("event_list") or [])]
    events = [event for event in events if event is not None]
    content = str(row.get("text") or "")
    title = row.get("title") or ""
    meta = {"title": title} if title else None
    return ConvertedDocument(canonical_row=_canonical_row(doc_id, dataset, split, content, events, meta=meta))


def _convert_chfinann_document(row: Any, *, dataset: str, split: str, index: int) -> ConvertedDocument:
    if not (isinstance(row, list) and len(row) == 2 and isinstance(row[1], dict)):
        raise ValueError(f"{dataset}/{split}/{index}: expected ChFinAnn [doc_id, payload] pair")
    doc_id = str(row[0]).strip()
    payload = row[1]
    content = "\n".join(str(sentence) for sentence in payload.get("sentences") or [])
    events: list[dict[str, Any]] = []
    for event_index, item in enumerate(payload.get("recguid_eventname_eventdict_list") or []):
        if not (isinstance(item, list) and len(item) >= 3 and isinstance(item[2], dict)):
            continue
        record_id, event_type, arguments = item[:3]
        event = _event_from_argument_mapping(str(event_type), arguments, record_id=str(record_id or event_index))
        if event is not None:
            events.append(event)
    return ConvertedDocument(canonical_row=_canonical_row(doc_id, dataset, split, content, events))


def _convert_docfee_document(row: Any, *, dataset: str, split: str, index: int) -> ConvertedDocument:
    if not isinstance(row, dict):
        raise ValueError(f"{dataset}/{split}/{index}: expected JSON object")
    content = str(row.get("content") or "")
    doc_id = _document_id(row, dataset=dataset, index=index, text_for_hash=content)
    events = [
        _event_from_generic(item, record_id=str(i))
        for i, item in enumerate(row.get("events") or [])
    ]
    events = [event for event in events if event is not None]
    return ConvertedDocument(canonical_row=_canonical_row(doc_id, dataset, split, content, events))


# ---------------------------------------------------------------------------
# Event converters
# ---------------------------------------------------------------------------

def _event_from_duee(event: Any, *, record_id: str) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("event_type") or "").strip()
    if not event_type:
        return None
    arguments: dict[str, list[dict[str, str]]] = {}
    for argument in event.get("arguments") or []:
        if not isinstance(argument, dict):
            continue
        role = str(argument.get("role") or "").strip()
        value = _clean_text(argument.get("argument"))
        if role and value:
            arguments.setdefault(role, []).append({"text": value})
    return {"event_type": event_type, "record_id": record_id, "arguments": arguments}


def _event_from_generic(event: Any, *, record_id: str) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("event_type") or "").strip()
    if not event_type:
        return None
    if isinstance(event.get("arguments"), dict):
        return _event_from_argument_mapping(event_type, event["arguments"], record_id=record_id)
    arguments = {k: v for k, v in event.items() if k not in {"event_type", "event_id", "id", "record_id", "trigger"}}
    return _event_from_argument_mapping(event_type, arguments, record_id=record_id)


def _event_from_argument_mapping(event_type: str, arguments: dict[str, Any], *, record_id: str) -> dict[str, Any] | None:
    event_type = str(event_type).strip()
    if not event_type:
        return None
    normalized: dict[str, list[dict[str, str]]] = {}
    for role, raw_values in arguments.items():
        role_name = str(role).strip()
        if not role_name:
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for value in values:
            text = _clean_text(value)
            if text:
                normalized.setdefault(role_name, []).append({"text": text})
    return {"event_type": event_type, "record_id": record_id, "arguments": normalized}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_roles_from_list(payload: list[Any]) -> dict[str, list[str]]:
    event_roles: dict[str, list[str]] = {}
    for item in payload:
        if not isinstance(item, dict) or "event_type" not in item:
            continue
        event_type = str(item["event_type"]).strip()
        if not event_type:
            continue
        roles: list[str] = []
        if isinstance(item.get("arguments"), list):
            roles = [str(role).strip() for role in item["arguments"] if str(role).strip()]
        elif isinstance(item.get("role_list"), list):
            roles = [
                str(role["role"]).strip()
                for role in item["role_list"]
                if isinstance(role, dict) and str(role.get("role") or "").strip()
            ]
        event_roles[event_type] = roles
    return event_roles


def _canonical_row(
    doc_id: str,
    dataset: str,
    split: str,
    content: str,
    events: list[dict[str, Any]],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "doc_id": doc_id,
        "dataset": dataset,
        "split": split,
        "content": content,
        "events": events,
    }
    clean_meta = {k: v for k, v in (meta or {}).items() if v not in (None, "")}
    if clean_meta:
        row["meta"] = clean_meta
    return row


def _document_id(row: dict[str, Any], *, dataset: str, index: int, text_for_hash: str | None = None) -> str:
    for key in ("document_id", "doc_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = text_for_hash if text_for_hash is not None else str(row.get("content") or row.get("text") or "")
    if text:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"line_index_text_hash:{index}:{dataset}"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _source_split_path(processed_dir: Path, *, dataset: str, split: str) -> Path:
    candidates = [
        processed_dir / f"{split}.jsonl",
        processed_dir / f"{split}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{dataset}/{split}: no source file in {processed_dir}")


def _iter_source_rows(path: Path, *, limit: int | None) -> Iterable[Any]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            count = 0
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    for index, row in enumerate(rows):
        yield row
        if limit is not None and index + 1 >= limit:
            return


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
