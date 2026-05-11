from __future__ import annotations

from collections import defaultdict

from evaluator.canonical.stats import Counts, document_id, event_type, record_unit_count
from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord


def merge_documents(documents: list[CanonicalDocument]) -> dict[str, list[CanonicalEventRecord]]:
    merged: dict[str, list[CanonicalEventRecord]] = defaultdict(list)
    for document in documents:
        merged[str(document.document_id)].extend(document.records)
    return dict(merged)


def group_by_event_type(records: list[CanonicalEventRecord]) -> dict[str, list[CanonicalEventRecord]]:
    grouped: dict[str, list[CanonicalEventRecord]] = defaultdict(list)
    for record in records:
        grouped[event_type(record)].append(record)
    return dict(grouped)


def gold_subset_for_document(records: list[CanonicalEventRecord]) -> str:
    non_empty_records = sum(1 for record in records if record_unit_count(record) > 0)
    return "multi_event" if non_empty_records > 1 else "single_event"


def empty_subset_counts() -> dict[str, Counts]:
    return {"single_event": Counts(), "multi_event": Counts()}


def all_document_ids(gold_documents: list[CanonicalDocument], pred_documents: list[CanonicalDocument]) -> list[str]:
    ids = {str(document.document_id) for document in gold_documents}
    ids.update(str(document.document_id) for document in pred_documents)
    return sorted(ids)
