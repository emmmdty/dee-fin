from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORMAT_NAME = "procnet_native_event_table_v1"


def write_native_event_table(
    path: Path,
    *,
    dataset: str,
    split: str,
    seed: int,
    epoch: int | str | None,
    event_types: list[str],
    event_type_fields: dict[str, list[str]],
    documents: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = build_native_event_table(
        dataset=dataset,
        split=split,
        seed=seed,
        epoch=epoch,
        event_types=event_types,
        event_type_fields=event_type_fields,
        documents=documents,
        metadata=metadata,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_native_event_table(
    *,
    dataset: str,
    split: str,
    seed: int,
    epoch: int | str | None,
    event_types: list[str],
    event_type_fields: dict[str, list[str]],
    documents: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not event_types:
        raise ValueError("event_types must be non-empty")
    for event_type in event_types:
        roles = event_type_fields.get(event_type)
        if not isinstance(roles, list):
            raise ValueError(f"missing event_type_fields for {event_type}")
    normalized_documents = [
        _normalize_document(doc, index=index, event_types=event_types, event_type_fields=event_type_fields)
        for index, doc in enumerate(documents)
    ]
    return {
        "format": FORMAT_NAME,
        "dataset": dataset,
        "seed": seed,
        "split": split,
        "epoch": epoch,
        "event_types": event_types,
        "event_type_fields": {event_type: event_type_fields[event_type] for event_type in event_types},
        "documents": normalized_documents,
        "metadata": metadata,
    }


def matrices_from_raw_results(metric: Any, raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    event_types, event_type_fields = event_schema_from_metric(metric)
    gold = _gold_record_mat_list(metric, [result.get("event_ans", []) for result in raw_results])
    pred = _pred_record_mat_list(metric, [result.get("event_pred", []) for result in raw_results])
    documents = []
    for index, result in enumerate(raw_results):
        documents.append(
            {
                "document_id": str(result.get("doc_id") or f"doc_index_{index:06d}"),
                "gold": _jsonable_matrix(gold[index]),
                "pred": _jsonable_matrix(pred[index]),
            }
        )
    return {
        "event_types": event_types,
        "event_type_fields": event_type_fields,
        "documents": documents,
    }


def event_schema_from_metric(metric: Any) -> tuple[list[str], dict[str, list[str]]]:
    type_to_index = dict(metric.event_type_type_to_index)
    type_to_index.pop("Null", None)
    index_to_type = list(metric.event_type_index_to_type)[1:]
    event_types = [index_to_type[i] for i in range(len(index_to_type))]
    event_type_fields = {event_type: list(metric.event_schema[event_type]) for event_type in event_types}
    return event_types, event_type_fields


def _gold_record_mat_list(metric: Any, events_ans: list[list[dict[str, Any]]]) -> list[list[list[tuple[Any, ...]]]]:
    type_to_index = dict(metric.event_type_type_to_index)
    type_to_index.pop("Null", None)
    type_to_index = {key: value - 1 for key, value in type_to_index.items()}
    role_to_index = metric.event_role_relation_to_index
    event_num = len(type_to_index)
    event_schema = metric.event_schema

    gold_record_mat_list = []
    for event_ans in events_ans:
        gold_record_mat = [[] for _ in range(event_num)]
        for e_ans in event_ans:
            roles_dict = {value: key for key, value in e_ans.items() if key != "EventType"}
            event_type = metric.event_type_index_to_type[e_ans["EventType"]]
            event_type_id = type_to_index[event_type]
            roles_tuple = []
            for role_name in event_schema[event_type]:
                role_index = role_to_index[role_name]
                roles_tuple.append(roles_dict.get(role_index))
            gold_record_mat[event_type_id].append(tuple(roles_tuple))
        gold_record_mat_list.append(gold_record_mat)
    return gold_record_mat_list


def _pred_record_mat_list(metric: Any, events_pred: list[list[dict[str, Any]]]) -> list[list[list[tuple[Any, ...]]]]:
    from procnet.utils.util_structure import UtilStructure

    type_to_index = dict(metric.event_type_type_to_index)
    type_to_index.pop("Null", None)
    type_to_index = {key: value - 1 for key, value in type_to_index.items()}
    role_to_index = metric.event_role_relation_to_index
    event_num = len(type_to_index)
    event_schema = metric.event_schema

    pred_record_mat_list = []
    for event_pred in events_pred:
        pred_record_mat = [[] for _ in range(event_num)]
        for e_pred in event_pred:
            event_type_index = UtilStructure.find_max_number_index(e_pred["EventType"])
            event_type = metric.event_type_index_to_type[event_type_index]
            if event_type == "Null":
                continue
            event_type_id = type_to_index[event_type]
            roles_dict: dict[int, list[Any]] = {}
            for key, value in e_pred.items():
                if key == "EventType":
                    continue
                max_p, index = UtilStructure.find_max_and_number_index(value)
                roles_dict.setdefault(index, []).append([key, max_p])
            for key in list(roles_dict):
                best_key = max(roles_dict[key], key=lambda item: item[1])[0]
                roles_dict[key] = best_key
            roles_tuple = []
            for role_name in event_schema[event_type]:
                role_index = role_to_index[role_name]
                roles_tuple.append(roles_dict.get(role_index))
            pred_record_mat[event_type_id].append(tuple(roles_tuple))
        pred_record_mat_list.append(pred_record_mat)
    return pred_record_mat_list


def _normalize_document(
    doc: dict[str, Any],
    *,
    index: int,
    event_types: list[str],
    event_type_fields: dict[str, list[str]],
) -> dict[str, Any]:
    document_id = str(doc.get("document_id") or f"doc_index_{index:06d}")
    gold = _normalize_side(doc.get("gold"), event_types=event_types, event_type_fields=event_type_fields)
    pred = _normalize_side(doc.get("pred"), event_types=event_types, event_type_fields=event_type_fields)
    return {"document_id": document_id, "gold": gold, "pred": pred}


def _normalize_side(
    side: Any,
    *,
    event_types: list[str],
    event_type_fields: dict[str, list[str]],
) -> list[list[list[Any]]]:
    if not isinstance(side, list) or len(side) != len(event_types):
        raise ValueError("native side must be a list with one entry per event type")
    normalized = []
    for event_index, event_type in enumerate(event_types):
        role_count = len(event_type_fields[event_type])
        records = side[event_index]
        if not isinstance(records, list):
            raise ValueError("native event records must be lists")
        normalized_records = []
        for record in records:
            if not isinstance(record, (list, tuple)) or len(record) != role_count:
                raise ValueError(f"record for {event_type} must have {role_count} role slots")
            normalized_records.append([_jsonable_slot(value) for value in record])
        normalized.append(normalized_records)
    return normalized


def _jsonable_matrix(matrix: Any) -> list[Any]:
    if isinstance(matrix, tuple):
        return [_jsonable_matrix(item) for item in matrix]
    if isinstance(matrix, list):
        return [_jsonable_matrix(item) for item in matrix]
    return _jsonable_slot(matrix)


def _jsonable_slot(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return " ".join(str(item) for item in value)
    return str(value)
