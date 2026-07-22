from __future__ import annotations

from typing import Any

from sarge.postprocess.rule_planner import EventRecord
from sarge.record_binding.assembler import BindingAssembler, BindingDiagnostics, BindingScoreProvider


def bind_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    schema,
    threshold: float = 0.85,
    score_provider: BindingScoreProvider | None = None,
) -> tuple[list[dict[str, Any]], list[BindingDiagnostics]]:
    """Apply record binding to canonical prediction rows.

    The canonical JSONL contract is preserved: each output row contains the
    original ``doc_id`` and an ``events`` list of canonical event records.
    """

    assembler = BindingAssembler(schema=schema, threshold=threshold)
    bound_rows: list[dict[str, Any]] = []
    diagnostics: list[BindingDiagnostics] = []

    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        events = row.get("events") or []
        if not isinstance(events, list):
            events = []
        records = [EventRecord.from_canonical(event) for event in events if isinstance(event, dict)]
        planned, doc_diagnostics = assembler.bind_document(
            records,
            score_provider=score_provider,
            doc_id=doc_id,
        )
        bound_rows.append(
            {
                "doc_id": doc_id,
                "events": [record.to_canonical() for record in planned],
            }
        )
        diagnostics.append(doc_diagnostics)

    return bound_rows, diagnostics
