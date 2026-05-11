from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evaluator.canonical.loaders import load_documents

from scripts.baseline.procnet.procnet_wrapper import write_jsonl


def validate_duee_split_paths(*, train: Path, dev: Path, test: Path) -> None:
    resolved = {"train": train.resolve(), "dev": dev.resolve(), "test": test.resolve()}
    if len(set(resolved.values())) != 3:
        raise ValueError(f"DuEE-Fin train/dev/test must be distinct files: {resolved}")
    missing = [name for name, path in resolved.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing DuEE-Fin split file(s): " + ", ".join(missing))


def processed_paths(project_root: Path, dataset: str) -> dict[str, Path]:
    root = project_root / "data" / "processed" / dataset
    if dataset == "ChFinAnn-Doc2EDAG":
        names = {"train": "train.json", "dev": "dev.json", "test": "test.json", "schema": "schema.json"}
    elif dataset == "DuEE-Fin-dev500":
        names = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl", "schema": "schema.json"}
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    paths = {key: root / name for key, name in names.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing processed file(s): " + ", ".join(missing))
    return paths


def stage_dataset(project_root: Path, run_dir: Path, dataset: str) -> dict[str, Path]:
    paths = processed_paths(project_root, dataset)
    staged_root = run_dir / "staged_data" / dataset
    staged_root.mkdir(parents=True, exist_ok=True)
    if dataset == "DuEE-Fin-dev500":
        validate_duee_split_paths(train=paths["train"], dev=paths["dev"], test=paths["test"])
        staged_paths = _stage_duee_as_docee(paths, staged_root)
    else:
        staged_paths = _stage_chfinann(paths, staged_root)
    runtime_data = run_dir / "procnet_runtime" / "Data"
    runtime_data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        shutil.copyfile(staged_paths[split], runtime_data / f"{split}.json")
    shutil.copyfile(staged_paths["schema"], runtime_data / "schema.json")
    return {
        **staged_paths,
        "runtime_data": runtime_data,
        "source_train": paths["train"],
        "source_dev": paths["dev"],
        "source_test": paths["test"],
        "source_schema": paths["schema"],
    }


def _stage_chfinann(paths: dict[str, Path], staged_root: Path) -> dict[str, Path]:
    staged = {}
    for key in ("train", "dev", "test", "schema"):
        target = staged_root / paths[key].name
        shutil.copyfile(paths[key], target)
        staged[key] = target
    return staged


def _stage_duee_as_docee(paths: dict[str, Path], staged_root: Path) -> dict[str, Path]:
    schema = load_procnet_schema(paths["schema"])
    staged_schema = staged_root / "schema.json"
    staged_schema.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged = {"schema": staged_schema}
    for split in ("train", "dev", "test"):
        rows = _read_jsonl(paths[split])
        converted = [_duee_row_to_docee_pair(row, schema, index=index) for index, row in enumerate(rows)]
        target = staged_root / f"{split}.json"
        target.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged[split] = target
    return staged


def _duee_row_to_docee_pair(row: dict[str, Any], schema: dict[str, list[str]], *, index: int) -> list[Any]:
    doc_id = str(row.get("id") or f"doc_index_{index:06d}")
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    combined_text = text if text.startswith("原标题：") else f"原标题：{title}。原文：{text}"
    sentences = _split_sentences(combined_text)
    ann_mspan2guess_field: dict[str, str] = {}
    recguid_eventname_eventdict_list = []
    for event_index, event in enumerate(row.get("event_list") or []):
        event_type = event.get("event_type")
        if event_type not in schema:
            continue
        arguments: dict[str, str | None] = {role: None for role in schema[event_type]}
        for argument in event.get("arguments") or []:
            role = argument.get("role")
            value = argument.get("argument")
            if role in arguments and isinstance(value, str) and value:
                arguments[role] = value
                ann_mspan2guess_field.setdefault(value, role)
        recguid_eventname_eventdict_list.append([event_index, event_type, arguments])
    ann_mspan2dranges = {
        span: _find_span_positions(sentences, span) for span in sorted(ann_mspan2guess_field)
    }
    payload = {
        "sentences": sentences,
        "ann_valid_mspans": sorted(ann_mspan2guess_field),
        "ann_valid_dranges": sorted(
            [position for positions in ann_mspan2dranges.values() for position in positions],
            key=lambda item: (item[0], item[1], item[2]),
        ),
        "ann_mspan2dranges": ann_mspan2dranges,
        "ann_mspan2guess_field": ann_mspan2guess_field,
        "recguid_eventname_eventdict_list": recguid_eventname_eventdict_list,
    }
    return [doc_id, payload]


def export_canonical_split(
    output_root: Path,
    *,
    split: str,
    gold_documents: list[dict[str, Any]],
    pred_documents: list[dict[str, Any]],
) -> tuple[Path, Path]:
    gold_path = output_root / f"{split}.canonical.gold.jsonl"
    pred_path = output_root / f"{split}.canonical.pred.jsonl"
    write_jsonl(gold_path, gold_documents)
    write_jsonl(pred_path, pred_documents)
    return gold_path, pred_path


def export_canonical_gold_from_source(
    *,
    source_path: Path,
    output_root: Path,
    dataset: str,
    split: str,
) -> Path:
    loaded = load_documents(source_path, dataset=dataset)
    rows = [
        {
            "document_id": document.document_id,
            "events": [_canonical_event_to_json(record) for record in document.records],
        }
        for document in loaded.documents
    ]
    gold_path, _ = export_canonical_split(output_root, split=split, gold_documents=rows, pred_documents=[])
    return gold_path


def canonical_predictions_from_native_table(native_table: dict[str, Any]) -> list[dict[str, Any]]:
    event_types = native_table["event_types"]
    event_type_fields = native_table["event_type_fields"]
    rows = []
    for document in native_table["documents"]:
        predictions = []
        for event_index, records in enumerate(document["pred"]):
            event_type = event_types[event_index]
            roles = event_type_fields[event_type]
            for record_index, record in enumerate(records):
                arguments = {
                    role: value
                    for role, value in zip(roles, record)
                    if value is not None
                }
                if arguments:
                    predictions.append(
                        {
                            "event_type": event_type,
                            "record_id": str(record_index),
                            "arguments": arguments,
                        }
                    )
        rows.append({"document_id": document["document_id"], "predictions": predictions})
    return rows


def _canonical_event_to_json(record: Any) -> dict[str, Any]:
    return {
        "event_type": record.event_type,
        "record_id": record.record_id,
        "arguments": record.arguments,
    }


def load_procnet_schema(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        schema = {}
        for item in data:
            if not isinstance(item, dict) or "event_type" not in item:
                continue
            if isinstance(item.get("arguments"), list):
                roles = [str(role) for role in item["arguments"]]
            elif isinstance(item.get("role_list"), list):
                roles = [str(role["role"]) for role in item["role_list"] if isinstance(role, dict) and "role" in role]
            else:
                roles = []
            schema[str(item["event_type"])] = roles
        return schema
    if isinstance(data, dict):
        return {str(event_type): [str(role) for role in roles] for event_type, roles in data.items() if isinstance(roles, list)}
    raise ValueError(f"unsupported schema shape: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _split_sentences(text: str) -> list[str]:
    sentences = []
    start = 0
    for index, char in enumerate(text):
        if char in "。！？\n":
            sentence = text[start : index + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def _find_span_positions(sentences: list[str], span: str) -> list[list[int]]:
    positions = []
    if not span:
        return positions
    for sentence_index, sentence in enumerate(sentences):
        start = 0
        while True:
            offset = sentence.find(span, start)
            if offset < 0:
                break
            positions.append([sentence_index, offset, offset + len(span)])
            start = offset + len(span)
    return positions
