from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluator.canonical.grouping import (
    empty_subset_counts,
    gold_subset_for_document,
    group_by_event_type,
    merge_documents,
)
from evaluator.canonical.report import add_diagnostics, build_report
from evaluator.canonical.schema import EventSchema
from evaluator.canonical.stats import Counts, role_value_sets
from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord
from evaluator.canonical.validate import validate_documents
from evaluator.docfee_official.schema import DOCFEE_CHINESE_SCHEMA


def evaluate_docfee_official(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
    *,
    dataset: str,
    schema: EventSchema | None = None,
    input_paths: dict[str, str | None] | None = None,
    schema_path: str | None = None,
    loader_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_schema = schema or DOCFEE_CHINESE_SCHEMA
    diagnostics = add_diagnostics(
        {"parse_failure_count": 0},
        loader_diagnostics,
        validate_documents(gold_documents, effective_schema).to_report(),
        validate_documents(pred_documents, effective_schema).to_report(),
    )
    overall = Counts()
    per_event: dict[str, Counts] = defaultdict(Counts)
    subset_counts = empty_subset_counts()

    gold_by_id = merge_documents(gold_documents)
    pred_by_id = merge_documents(pred_documents)
    for document_id in sorted(gold_by_id):
        gold_records = gold_by_id.get(document_id, [])
        pred_records = pred_by_id.get(document_id, [])
        document_counts = _score_document(gold_records, pred_records, per_event)
        overall.add(document_counts)
        if gold_records:
            subset_counts[gold_subset_for_document(gold_records)].add(document_counts)

    return build_report(
        dataset=dataset,
        metric_family="docfee_official_style",
        overall=overall,
        per_event=per_event,
        subset_metrics=subset_counts,
        diagnostics=diagnostics,
        input_paths=input_paths,
        schema_path=schema_path,
        matching_policy={
            "name": "docfee_official_style_greedy_role_value_overlap",
            "event_type_constrained": True,
            "unit": "role:value",
            "prediction_record_removed_after_match": True,
        },
    )


def _score_document(
    gold_records: list[CanonicalEventRecord],
    pred_records: list[CanonicalEventRecord],
    per_event: dict[str, Counts],
) -> Counts:
    document_counts = Counts()
    gold_by_event = group_by_event_type(gold_records)
    pred_by_event = group_by_event_type(pred_records)
    for event_name in sorted(set(gold_by_event) | set(pred_by_event)):
        event_counts = _score_event_group(gold_by_event.get(event_name, []), pred_by_event.get(event_name, []))
        per_event[event_name].add(event_counts)
        document_counts.add(event_counts)
    return document_counts


def _score_event_group(gold_group: list[CanonicalEventRecord], pred_group: list[CanonicalEventRecord]) -> Counts:
    gold_role_values = [_role_value_strings(record) for record in gold_group]
    pred_role_values = [_role_value_strings(record) for record in pred_group]
    gold_total = sum(len(values) for values in gold_role_values)
    pred_total = sum(len(values) for values in pred_role_values)
    hit_total = 0
    remaining_pred = list(pred_role_values)

    for gold_values in gold_role_values:
        best_index = None
        best_hit = 0
        for pred_index, pred_values in enumerate(remaining_pred):
            hit = len(gold_values & pred_values)
            if hit > best_hit:
                best_hit = hit
                best_index = pred_index
        if best_index is not None:
            hit_total += best_hit
            del remaining_pred[best_index]

    return Counts(tp=hit_total, fp=pred_total - hit_total, fn=gold_total - hit_total)


def _role_value_strings(record: CanonicalEventRecord) -> set[str]:
    values = set()
    for role, role_values in role_value_sets(record).items():
        for value in role_values:
            values.add(f"{role}:{value}")
    return values
