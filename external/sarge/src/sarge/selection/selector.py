from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sarge.data.loader import V2DatasetDocument
from sarge.data.schema import DatasetSchema
from sarge.selection.features import compute_feature_rows
from sarge.selection.ranker import default_rule_based_model, score_with_model
from sarge.evaluation.export import strip_auxiliary_fields, validate_minimal_canonical_prediction


@dataclass(frozen=True)
class MRSSelectionResult:
    selected_rows: list[dict[str, Any]]
    score_rows: list[dict[str, Any]]
    canonical_predictions: list[dict[str, Any]]


def select_candidate_rows(
    *,
    candidates: list[dict[str, Any]],
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    model: dict[str, Any] | None = None,
    surface_memories: dict[str, dict[str, Any]] | None = None,
    slot_plans: dict[str, dict[str, Any]] | None = None,
) -> MRSSelectionResult:
    _validate_predict_documents(documents)
    model_payload = model or default_rule_based_model()
    feature_rows = compute_feature_rows(
        candidates,
        schema=schema,
        surface_memories=surface_memories,
        slot_plans=slot_plans,
    )
    candidates_by_id = {str(candidate.get("candidate_id", "")): candidate for candidate in candidates}
    score_rows: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        features = feature_row.get("features") or {}
        score_rows.append(
            {
                "doc_id": feature_row["doc_id"],
                "candidate_id": feature_row["candidate_id"],
                "candidate_index": int(feature_row.get("candidate_index", 0)),
                "score": score_with_model(model_payload, features),
                "selected": False,
                "features": dict(features),
                "diagnostics": dict(feature_row.get("diagnostics") or {}),
                "selector_mode": str(model_payload.get("mode", "")),
            }
        )

    score_rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        score_rows_by_doc[str(row["doc_id"])].append(row)

    selected_rows: list[dict[str, Any]] = []
    canonical_predictions: list[dict[str, Any]] = []
    for document in documents:
        rows = score_rows_by_doc.get(document.doc_id, [])
        if not rows:
            raise ValueError(f"no MRS candidates available for document: {document.doc_id}")
        selected_row = max(rows, key=_selection_key)
        selected_row["selected"] = True
        selected_candidate = dict(candidates_by_id[str(selected_row["candidate_id"])])
        selected_rows.append({**selected_candidate, "mrs_score": float(selected_row["score"])})
        canonical = strip_auxiliary_fields(
            {"doc_id": document.doc_id, "events": selected_candidate.get("events") or []}
        )
        validate_minimal_canonical_prediction(canonical)
        canonical_predictions.append(canonical)

    return MRSSelectionResult(
        selected_rows=selected_rows,
        score_rows=score_rows,
        canonical_predictions=canonical_predictions,
    )


def _validate_predict_documents(documents: list[V2DatasetDocument]) -> None:
    for document in documents:
        if document.gold is not None:
            raise ValueError("MRS selector inference documents must not expose gold")


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    features = row.get("features") or {}
    return (
        float(row.get("score", 0.0)),
        float(features.get("self_consistency_argument_jaccard", 0.0)),
        float(features.get("schema_valid_rate", 0.0)),
        -float(features.get("empty_prediction", 0.0)),
        -int(row.get("candidate_index", 0)),
    )
