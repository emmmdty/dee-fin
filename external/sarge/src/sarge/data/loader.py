from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sarge.data.jsonl import iter_jsonl
from sarge.data.schema import DatasetSchema, load_schema

DocumentMode = Literal["predict", "train", "eval_internal"]
ALLOWED_MODES = frozenset({"predict", "train", "eval_internal"})


@dataclass(frozen=True)
class V2DocumentInput:
    doc_id: str
    dataset_id: str
    dataset: str
    split: str
    content: str
    content_raw: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "dataset_id": self.dataset_id,
            "dataset": self.dataset,
            "split": self.split,
            "content": self.content,
        }
        if self.content_raw is not None:
            payload["content_raw"] = self.content_raw
        if self.meta:
            payload["meta"] = deepcopy(self.meta)
        return payload


@dataclass(frozen=True)
class V2GoldDocument:
    doc_id: str
    dataset_id: str
    dataset: str
    split: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "dataset_id": self.dataset_id,
            "dataset": self.dataset,
            "split": self.split,
            "events": deepcopy(self.events),
        }


@dataclass(frozen=True)
class V2DatasetDocument:
    input: V2DocumentInput
    gold: V2GoldDocument | None = None

    @property
    def doc_id(self) -> str:
        return self.input.doc_id


def load_documents(
    dataset: str,
    split: str,
    data_root: str | Path = "data",
    *,
    mode: DocumentMode = "predict",
    limit: int | None = None,
) -> list[V2DatasetDocument]:
    return list(iter_documents(dataset, split, data_root=data_root, mode=mode, limit=limit))


def iter_documents(
    dataset: str,
    split: str,
    data_root: str | Path = "data",
    *,
    mode: DocumentMode = "predict",
    limit: int | None = None,
):
    _validate_mode(mode)
    dataset_id = str(dataset).strip()
    split_name = str(split).strip()
    if not dataset_id:
        raise ValueError("dataset is required")
    if not split_name:
        raise ValueError("split is required")

    data_root_path = Path(data_root)
    schema = load_schema(dataset_id, data_root=data_root_path)
    documents_path = data_root_path / dataset_id / f"{split_name}.jsonl"
    for row in iter_jsonl(documents_path, limit=limit):
        yield _document_from_row(row, dataset_id=dataset_id, split=split_name, mode=mode, schema=schema)


def _document_from_row(
    row: dict[str, Any],
    *,
    dataset_id: str,
    split: str,
    mode: DocumentMode,
    schema: DatasetSchema,
) -> V2DatasetDocument:
    doc_id = str(row.get("doc_id", "")).strip()
    if not doc_id:
        raise ValueError(f"{dataset_id}/{split}: document missing doc_id")
    dataset_name = str(row.get("dataset") or schema.schema_dataset or dataset_id)
    source_split = str(row.get("split") or split)
    content = str(row.get("content") or row.get("content_raw") or "")
    content_raw = row.get("content_raw")
    input_doc = V2DocumentInput(
        doc_id=doc_id,
        dataset_id=dataset_id,
        dataset=dataset_name,
        split=source_split,
        content=content,
        content_raw=str(content_raw) if content_raw is not None else None,
        meta=deepcopy(row.get("meta") or {}),
    )
    if mode == "predict":
        return V2DatasetDocument(input=input_doc, gold=None)

    raw_events = deepcopy(row.get("events") or [])
    if not isinstance(raw_events, list):
        raise ValueError(f"{dataset_id}/{split}/{doc_id}: events must be a list")
    for event in raw_events:
        schema.validate_event_record(event)
    return V2DatasetDocument(
        input=input_doc,
        gold=V2GoldDocument(
            doc_id=doc_id,
            dataset_id=dataset_id,
            dataset=dataset_name,
            split=source_split,
            events=raw_events,
        ),
    )


def _validate_mode(mode: str) -> None:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)}; got {mode!r}")
