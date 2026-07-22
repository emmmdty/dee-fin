from __future__ import annotations

from dataclasses import dataclass

from evaluator.canonical.grouping import (
    all_document_ids,
    group_by_event_type,
    merge_documents,
)
from evaluator.canonical.normalize import normalize_text
from evaluator.canonical.stats import role_value_sets
from evaluator.canonical.types import CanonicalDocument
from evaluator.canonical.validate import validate_documents
from evaluator.unified_strict.matcher import match_records

__all__ = [
    "MisallocationDiagnostic",
    "compute_misallocation_diagnostic",
    "validate_documents",
]


@dataclass
class _Counts:
    total_pred_args: int = 0
    hallucinated: int = 0
    ungrounded: int = 0
    ambiguous_repeated: int = 0
    misallocated: int = 0
    correctly_allocated: int = 0
    eligible_non_misallocated_fp: int = 0

    def add(self, other: "_Counts") -> None:
        self.total_pred_args += other.total_pred_args
        self.hallucinated += other.hallucinated
        self.ungrounded += other.ungrounded
        self.ambiguous_repeated += other.ambiguous_repeated
        self.misallocated += other.misallocated
        self.correctly_allocated += other.correctly_allocated
        self.eligible_non_misallocated_fp += other.eligible_non_misallocated_fp

    @property
    def eligible_denominator(self) -> int:
        return self.total_pred_args - self.hallucinated - self.ungrounded - self.ambiguous_repeated

    def to_dict(self) -> dict[str, float | int]:
        total = self.total_pred_args
        eligible = self.eligible_denominator
        return {
            "total_pred_args": total,
            "hallucinated": self.hallucinated,
            "ungrounded": self.ungrounded,
            "ambiguous_repeated": self.ambiguous_repeated,
            "misallocated": self.misallocated,
            "correctly_allocated": self.correctly_allocated,
            "eligible_non_misallocated_fp": self.eligible_non_misallocated_fp,
            "eligible_denominator": eligible,
            "misallocated_rate_total": (self.misallocated / total) if total else 0.0,
            "misallocated_rate_eligible": (self.misallocated / eligible) if eligible else 0.0,
            "ambiguous_excluded_rate": (self.ambiguous_repeated / total) if total else 0.0,
            "hallucinated_argument_rate": (self.hallucinated / total) if total else 0.0,
            "ungrounded_argument_rate": (self.ungrounded / total) if total else 0.0,
        }


@dataclass
class MisallocationDiagnostic:
    """Implements the misallocation diagnostic from
    docs/method/carve_method_design_v1_3.md §4.2.

    The primary main-paper rate is ``misallocated_rate_eligible``:

        misallocated_rate_eligible = #misallocated / #eligible

    where eligible = total predicted arguments minus hallucinated minus
    ungrounded minus ambiguous-repeated. Without document texts the
    hallucinated count is zero (no detection possible); without predictor
    evidence pointers the ungrounded count is zero. In those cases the
    reported numbers are a lower bound on hallucinated/ungrounded and an
    upper bound on the eligible denominator. Callers that need the strict
    §4.2 contract must supply both inputs.
    """

    overall: dict[str, float | int]
    per_event: dict[str, dict[str, float | int]]
    inputs: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall,
            "per_event": self.per_event,
            "inputs": self.inputs,
            "definition_reference": "docs/method/carve_method_design_v1_3.md §4.2",
        }


def compute_misallocation_diagnostic(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
    *,
    document_texts: dict[str, str] | None = None,
    grounded_predictions: dict[str, set[tuple[int, str, str]]] | None = None,
) -> MisallocationDiagnostic:
    """Compute misallocation diagnostic rates for unified-strict predictions.

    Args:
        gold_documents: canonical gold documents.
        pred_documents: canonical predicted documents.
        document_texts: optional mapping document_id -> raw text. When
            provided, predicted arguments whose normalized value is not a
            substring of the normalized text are counted as ``hallucinated``.
        grounded_predictions: optional mapping document_id -> set of tuples
            ``(pred_record_doc_index, normalized_role, normalized_value)`` of
            predictions the predictor reports as grounded on a high-evidence
            sentence. When provided, predicted arguments not in the set (and
            not already hallucinated) are counted as ``ungrounded``. The
            ``pred_record_doc_index`` indexes into the document-flat list of
            predicted records after ``merge_documents``.

    Returns:
        A :class:`MisallocationDiagnostic` with overall and per-event counts
        and rates. Rates are in the [0, 1] range. The definition is
        ``docs/method/carve_method_design_v1_3.md`` §4.2.
    """
    overall = _Counts()
    per_event: dict[str, _Counts] = {}

    normalized_text_by_doc: dict[str, str] = {}
    if document_texts is not None:
        for raw_doc_id, raw_text in document_texts.items():
            normalized_text_by_doc[str(raw_doc_id)] = normalize_text(raw_text)

    gold_by_id = merge_documents(gold_documents)
    pred_by_id = merge_documents(pred_documents)

    for doc_id in all_document_ids(gold_documents, pred_documents):
        doc_text = normalized_text_by_doc.get(doc_id)
        doc_grounded = grounded_predictions.get(doc_id) if grounded_predictions else None

        gold_records = gold_by_id.get(doc_id, [])
        pred_records = pred_by_id.get(doc_id, [])
        if not pred_records:
            continue

        pred_doc_index_by_obj = {id(pred_records[i]): i for i in range(len(pred_records))}
        gold_by_event = group_by_event_type(gold_records)
        pred_by_event = group_by_event_type(pred_records)

        for event_name in sorted(set(gold_by_event) | set(pred_by_event)):
            pred_group = pred_by_event.get(event_name, [])
            if not pred_group:
                continue
            gold_group = gold_by_event.get(event_name, [])

            gold_role_value_counts: dict[tuple[str, str], int] = {}
            for gold_record in gold_group:
                gold_sets = role_value_sets(gold_record)
                for role, values in gold_sets.items():
                    for value in values:
                        key = (role, value)
                        gold_role_value_counts[key] = gold_role_value_counts.get(key, 0) + 1
            ambiguous_set = {key for key, count in gold_role_value_counts.items() if count >= 2}
            present_in_any_gold = set(gold_role_value_counts.keys())

            pairs, _ = match_records(pred_group, gold_group)
            aligned_gold_index_by_pred = dict(pairs)

            event_counts = per_event.setdefault(event_name, _Counts())

            for local_pred_index, pred_record in enumerate(pred_group):
                pred_sets = role_value_sets(pred_record)
                doc_pred_index = pred_doc_index_by_obj[id(pred_record)]
                aligned_gold_index = aligned_gold_index_by_pred.get(local_pred_index)
                aligned_gold_sets = (
                    role_value_sets(gold_group[aligned_gold_index])
                    if aligned_gold_index is not None
                    else {}
                )

                for role, values in pred_sets.items():
                    for value in values:
                        overall.total_pred_args += 1
                        event_counts.total_pred_args += 1

                        if doc_text is not None and value not in doc_text:
                            overall.hallucinated += 1
                            event_counts.hallucinated += 1
                            continue

                        if doc_grounded is not None and (doc_pred_index, role, value) not in doc_grounded:
                            overall.ungrounded += 1
                            event_counts.ungrounded += 1
                            continue

                        if (role, value) in ambiguous_set:
                            overall.ambiguous_repeated += 1
                            event_counts.ambiguous_repeated += 1
                            continue

                        if value in aligned_gold_sets.get(role, set()):
                            overall.correctly_allocated += 1
                            event_counts.correctly_allocated += 1
                        elif (role, value) in present_in_any_gold:
                            overall.misallocated += 1
                            event_counts.misallocated += 1
                        else:
                            overall.eligible_non_misallocated_fp += 1
                            event_counts.eligible_non_misallocated_fp += 1

    return MisallocationDiagnostic(
        overall=overall.to_dict(),
        per_event={name: counts.to_dict() for name, counts in sorted(per_event.items())},
        inputs={
            "document_texts_provided": document_texts is not None,
            "grounded_predictions_provided": grounded_predictions is not None,
        },
    )
