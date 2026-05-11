from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluator.canonical.grouping import (
    all_document_ids,
    empty_subset_counts,
    gold_subset_for_document,
    group_by_event_type,
    merge_documents,
)
from evaluator.canonical.report import add_diagnostics, build_report
from evaluator.canonical.schema import EventSchema
from evaluator.canonical.stats import (
    Counts,
    count_record_pair,
    count_unmatched_gold,
    count_unmatched_pred,
    exact_record_match,
)
from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord
from evaluator.canonical.validate import validate_documents
from evaluator.unified_strict.matcher import match_records


def evaluate_unified_strict(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
    *,
    dataset: str,
    schema: EventSchema | None = None,
    input_paths: dict[str, str | None] | None = None,
    schema_path: str | None = None,
    loader_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics = add_diagnostics(
        loader_diagnostics,
        validate_documents(gold_documents, schema).to_report(),
        validate_documents(pred_documents, schema).to_report(),
    )
    overall = Counts()
    per_event: dict[str, Counts] = defaultdict(Counts)
    subset_counts = empty_subset_counts()
    matcher_counts: dict[str, int] = defaultdict(int)
    exact_matches = 0
    matched_pairs = 0

    gold_by_id = merge_documents(gold_documents)
    pred_by_id = merge_documents(pred_documents)

    for document_id in all_document_ids(gold_documents, pred_documents):
        gold_records = gold_by_id.get(document_id, [])
        pred_records = pred_by_id.get(document_id, [])
        document_counts = _score_document(
            gold_records,
            pred_records,
            per_event,
            matcher_counts,
            lambda pred_group, gold_group: match_records(pred_group, gold_group),
            exact_match_sink=lambda pred_record, gold_record: exact_record_match(pred_record, gold_record),
        )
        overall.add(document_counts["counts"])
        exact_matches += document_counts["exact_matches"]
        matched_pairs += document_counts["matched_pairs"]
        if gold_records:
            subset_counts[gold_subset_for_document(gold_records)].add(document_counts["counts"])

    diagnostics.update(
        {
            "record_exact_match_count": exact_matches,
            "matched_record_pair_count": matched_pairs,
            "matcher_algorithm_counts": dict(sorted(matcher_counts.items())),
        }
    )
    return build_report(
        dataset=dataset,
        metric_family="unified_strict",
        overall=overall,
        per_event=per_event,
        subset_metrics=subset_counts,
        diagnostics=diagnostics,
        input_paths=input_paths,
        schema_path=schema_path,
        matching_policy={
            "name": "event_type_constrained_global_bipartite",
            "event_type_constrained": True,
            "objective": "maximize_strict_role_value_tp",
        },
    )


def _score_document(
    gold_records: list[CanonicalEventRecord],
    pred_records: list[CanonicalEventRecord],
    per_event: dict[str, Counts],
    matcher_counts: dict[str, int],
    matcher,
    exact_match_sink,
) -> dict[str, Any]:
    from evaluator.canonical.stats import event_type

    document_counts = Counts()
    exact_matches = 0
    matched_pairs = 0
    gold_by_event = group_by_event_type(gold_records)
    pred_by_event = group_by_event_type(pred_records)
    for event_name in sorted(set(gold_by_event) | set(pred_by_event)):
        gold_group = gold_by_event.get(event_name, [])
        pred_group = pred_by_event.get(event_name, [])
        pairs, matcher_info = matcher(pred_group, gold_group)
        matcher_counts[str(matcher_info.get("algorithm", "unknown"))] += 1
        used_pred = set()
        used_gold = set()
        event_counts = Counts()

        for pred_index, gold_index in pairs:
            used_pred.add(pred_index)
            used_gold.add(gold_index)
            pair_counts = count_record_pair(pred_group[pred_index], gold_group[gold_index])
            event_counts.add(pair_counts)
            if exact_match_sink(pred_group[pred_index], gold_group[gold_index]):
                exact_matches += 1
            matched_pairs += 1

        for pred_index, pred_record in enumerate(pred_group):
            if pred_index not in used_pred:
                event_counts.add(count_unmatched_pred(pred_record))
        for gold_index, gold_record in enumerate(gold_group):
            if gold_index not in used_gold:
                event_counts.add(count_unmatched_gold(gold_record))

        per_event[event_name].add(event_counts)
        document_counts.add(event_counts)

    return {"counts": document_counts, "exact_matches": exact_matches, "matched_pairs": matched_pairs}
