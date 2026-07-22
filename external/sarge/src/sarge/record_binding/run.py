from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sarge.data.jsonl import read_jsonl, write_jsonl
from sarge.data.schema import load_schema
from sarge.evaluation.export import validate_minimal_canonical_prediction
from sarge.record_binding.assembler import BindingScoreProvider
from sarge.record_binding.prediction import bind_prediction_rows


def run_record_binding(
    *,
    input_run_root: str | Path,
    output_run_root: str | Path,
    dataset: str,
    split: str,
    data_root: str | Path,
    threshold: float = 0.85,
    score_provider: BindingScoreProvider | None = None,
) -> dict[str, Any]:
    input_prediction_path = _prediction_path(input_run_root, dataset=dataset, split=split)
    output_prediction_path = _prediction_path(output_run_root, dataset=dataset, split=split)
    output_root = Path(output_run_root)

    schema = load_schema(dataset, data_root=data_root)
    rows = read_jsonl(input_prediction_path)
    bound_rows, diagnostics = bind_prediction_rows(
        rows,
        schema=schema,
        threshold=threshold,
        score_provider=score_provider,
    )
    for row in bound_rows:
        validate_minimal_canonical_prediction(row, schema=schema)

    write_jsonl(output_prediction_path, bound_rows)

    summary = _summarize_run(
        dataset=dataset,
        split=split,
        threshold=threshold,
        input_prediction_path=input_prediction_path,
        output_prediction_path=output_prediction_path,
        diagnostics=diagnostics,
    )
    diagnostics_payload = {
        "summary": summary,
        "documents": [item.to_dict() for item in diagnostics],
    }
    _write_json(output_root / "diagnostics" / "record_binding.json", diagnostics_payload)
    _write_json(output_root / "summary.json", summary)
    return summary


def _prediction_path(run_root: str | Path, *, dataset: str, split: str) -> Path:
    return Path(run_root) / "predictions" / dataset / f"{split}.canonical.pred.jsonl"


def _summarize_run(
    *,
    dataset: str,
    split: str,
    threshold: float,
    input_prediction_path: Path,
    output_prediction_path: Path,
    diagnostics,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "threshold": float(threshold),
        "docs": len(diagnostics),
        "events_before": sum(item.events_before for item in diagnostics),
        "events_after": sum(item.events_after for item in diagnostics),
        "merge_count": sum(item.merge_count for item in diagnostics),
        "blocked_count": sum(item.blocked_count for item in diagnostics),
        "input_prediction_path": str(input_prediction_path),
        "output_prediction_path": str(output_prediction_path),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
