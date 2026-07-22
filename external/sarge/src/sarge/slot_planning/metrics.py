from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from sarge.data.loader import V2DatasetDocument
from sarge.data.schema import DatasetSchema
from sarge.slot_planning.labels import count_bucket
from sarge.slot_planning.plan import SlotPlanDocument

ROLE_PRIOR_THRESHOLD = 0.5


def compute_slot_plan_metrics(
    predictions: Iterable[SlotPlanDocument],
    gold_documents: Iterable[V2DatasetDocument],
    schema: DatasetSchema,
    *,
    role_threshold: float = ROLE_PRIOR_THRESHOLD,
) -> dict[str, float | int | None]:
    prediction_by_doc_id = {prediction.doc_id: prediction for prediction in predictions}
    gold_document_list = list(gold_documents)
    for document in gold_document_list:
        if document.gold is None:
            raise ValueError("slot plan metrics require gold-visible eval documents")

    presence_tp = presence_fp = presence_fn = 0
    count_correct = 0
    bucket_correct = 0
    count_total = 0
    same_type_multi_total = 0
    same_type_multi_hit = 0

    gold_role_keys: set[tuple[str, str, str]] = set()
    pred_role_keys: set[tuple[str, str, str]] = set()

    for document in gold_document_list:
        assert document.gold is not None
        prediction = prediction_by_doc_id.get(document.doc_id)
        pred_counts = _predicted_event_counts(prediction)
        gold_counts = Counter(str(event.get("event_type", "")).strip() for event in document.gold.events)
        gold_roles = _gold_role_occupancy(document, schema)
        pred_roles = _predicted_role_occupancy(document.doc_id, prediction, schema, role_threshold=role_threshold)
        gold_role_keys.update(gold_roles)
        pred_role_keys.update(pred_roles)

        for event_type in schema.event_roles:
            gold_count = int(gold_counts.get(event_type, 0))
            pred_count = int(pred_counts.get(event_type, 0))
            gold_present = gold_count > 0
            pred_present = pred_count > 0
            if gold_present and pred_present:
                presence_tp += 1
            elif pred_present and not gold_present:
                presence_fp += 1
            elif gold_present and not pred_present:
                presence_fn += 1
            if pred_count == gold_count:
                count_correct += 1
            if count_bucket(pred_count) == count_bucket(gold_count):
                bucket_correct += 1
            count_total += 1
            if gold_count > 1:
                same_type_multi_total += 1
                if pred_count > 1:
                    same_type_multi_hit += 1

    presence_precision, presence_recall, presence_f1 = _prf(presence_tp, presence_fp, presence_fn)
    role_tp = len(gold_role_keys & pred_role_keys)
    role_fp = len(pred_role_keys - gold_role_keys)
    role_fn = len(gold_role_keys - pred_role_keys)
    role_precision, role_recall, role_f1 = _prf(role_tp, role_fp, role_fn)

    return {
        "document_count": len(gold_document_list),
        "event_presence_precision": presence_precision,
        "event_presence_recall": presence_recall,
        "event_presence_f1": presence_f1,
        "event_count_accuracy": _rate(count_correct, count_total),
        "event_count_correct": count_correct,
        "event_count_total": count_total,
        "event_count_bucket_accuracy": _rate(bucket_correct, count_total),
        "same_type_multi_event_recall": _rate(same_type_multi_hit, same_type_multi_total),
        "same_type_multi_event_gold_total": same_type_multi_total,
        "role_occupancy_precision": role_precision,
        "role_occupancy_recall": role_recall,
        "role_occupancy_f1": role_f1,
    }


def _predicted_event_counts(prediction: SlotPlanDocument | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if prediction is None:
        return counts
    for slot in prediction.slots:
        counts[slot.event_type] += 1
    return counts


def _gold_role_occupancy(document: V2DatasetDocument, schema: DatasetSchema) -> set[tuple[str, str, str]]:
    assert document.gold is not None
    keys: set[tuple[str, str, str]] = set()
    for event in document.gold.events:
        event_type = schema.validate_event_type(str(event.get("event_type", "")))
        arguments = event.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        for role in schema.event_roles[event_type]:
            values = arguments.get(role) or []
            if isinstance(values, list) and values:
                keys.add((document.doc_id, event_type, role))
    return keys


def _predicted_role_occupancy(
    doc_id: str,
    prediction: SlotPlanDocument | None,
    schema: DatasetSchema,
    *,
    role_threshold: float,
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if prediction is None:
        return keys
    for slot in prediction.slots:
        event_type = schema.validate_event_type(slot.event_type)
        for role, score in slot.role_prior.items():
            schema.validate_role(event_type, role)
            if score >= role_threshold:
                keys.add((doc_id, event_type, role))
    return keys


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        return precision, recall, 0.0 if precision is not None and recall is not None else None
    return precision, recall, 2 * precision * recall / (precision + recall)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
