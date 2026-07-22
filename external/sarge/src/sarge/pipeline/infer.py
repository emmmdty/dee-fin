from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarge.surface_memory.candidate_builder import build_surface_memories
from sarge.surface_memory.builder import surface_memory_to_dict
from sarge.data.loader import load_documents
from sarge.data.jsonl import read_jsonl, write_jsonl
from sarge.data.schema import load_schema
from sarge.generation.candidate_generator import generate_getm_candidate_files
from sarge.models.mock_backend import MockGetmBackend
from sarge.slot_planning.audit import audit_slot_plans
from sarge.slot_planning.baseline import TrainPriorPlanner
from sarge.slot_planning.plan import SlotPlanDocument, slot_plan_from_dict, slot_plan_to_dict, validate_slot_plan
from sarge.selection.selector import select_candidate_rows
from sarge.selection.ranker import default_rule_based_model
from sarge.evaluation.handoff import DEFAULT_DATA_REPO_ROOT, EvaluatorHandoff, build_evaluator_handoff
from sarge.evaluation.export import export_predictions
from sarge.pipeline.manifest import write_run_manifest


def _default_mock_backend_factory() -> Any:
    return MockGetmBackend(mode="echo_candidates")


@dataclass(frozen=True)
class InferenceResult:
    run_id: str
    run_root: Path
    prediction_path: Path
    run_manifest_path: Path
    handoff_command: str
    handoff_script_exists: bool
    handoff: EvaluatorHandoff


def run_inference(
    *,
    dataset: str = "DuEE-Fin-dev500",
    split: str = "dev",
    data_root: str | Path = "data",
    out_root: str | Path = "runs",
    run_id: str | None = None,
    seed: int = 13,
    k: int = 4,
    slot_plan_path: str | Path | None = None,
    data_repo_root: str | Path = DEFAULT_DATA_REPO_ROOT,
    evaluator_out_dir: str | Path | None = None,
    limit: int | None = None,
    command_infer: str | None = None,
    source_commit: str | None = None,
    backend: Any | None = None,
) -> InferenceResult:
    if k < 1:
        raise ValueError("k must be >= 1")
    resolved_run_id = run_id or _default_run_id(dataset, split)
    run_root = Path(out_root) / resolved_run_id
    intermediate_dir = run_root / "intermediate"
    diagnostics_dir = run_root / "diagnostics"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema(dataset, data_root=data_root)
    documents = load_documents(dataset, split, data_root=data_root, mode="predict", limit=limit)
    if not documents:
        raise ValueError(f"no documents loaded for {dataset}/{split}")

    surface_memories = build_surface_memories(documents)
    surface_memory_by_doc = {memory.doc_id: memory for memory in surface_memories}
    surface_memory_rows = [surface_memory_to_dict(memory) for memory in surface_memories]
    surface_memory_path = write_jsonl(intermediate_dir / "surface_memory.jsonl", surface_memory_rows)
    _write_json(
        diagnostics_dir / "surface_memory_summary.json",
        {
            "dataset": dataset,
            "split": split,
            "document_count": len(surface_memories),
            "candidate_count_total": sum(len(memory.candidates) for memory in surface_memories),
            "candidate_count_per_doc": {memory.doc_id: len(memory.candidates) for memory in surface_memories},
            "gold_visible": False,
        },
    )

    slot_plans = _load_or_build_slot_plans(
        dataset=dataset,
        split=split,
        data_root=data_root,
        schema=schema,
        documents=documents,
        slot_plan_path=slot_plan_path,
    )
    slot_plan_rows = [slot_plan_to_dict(plan) for plan in slot_plans]
    slot_plan_path_out = write_jsonl(intermediate_dir / "slot_plan.jsonl", slot_plan_rows)
    slot_plan_audit = audit_slot_plans(slot_plans, schema)
    _write_json(diagnostics_dir / "slot_plan_audit.json", slot_plan_audit)
    slot_plan_by_doc = {plan.doc_id: plan for plan in slot_plans}
    slot_plan_rows_by_doc = {row["doc_id"]: row for row in slot_plan_rows}

    getm_dir = intermediate_dir / "getm"
    active_backend = backend if backend is not None else _default_mock_backend_factory()
    getm_output = generate_getm_candidate_files(
        documents=documents,
        dataset=dataset,
        split=split,
        schema=schema,
        backend=active_backend,
        k=k,
        out_dir=getm_dir,
        surface_memories=surface_memory_by_doc,
        slot_plans=slot_plan_by_doc,
    )

    candidate_rows = read_jsonl(getm_output.parsed_candidates_path)
    mrs_result = select_candidate_rows(
        candidates=candidate_rows,
        documents=documents,
        schema=schema,
        model=default_rule_based_model(),
        surface_memories={row["doc_id"]: row for row in surface_memory_rows},
        slot_plans=slot_plan_rows_by_doc,
    )
    mrs_dir = intermediate_dir / "mrs"
    selector_scores_path = write_jsonl(mrs_dir / f"selector_scores.{split}.jsonl", mrs_result.score_rows)
    selected_candidates_path = write_jsonl(mrs_dir / f"selected_candidates.{split}.jsonl", mrs_result.selected_rows)
    prediction_path = run_root / "predictions" / dataset / f"{split}.canonical.pred.jsonl"
    export_predictions(mrs_result.canonical_predictions, prediction_path)
    _write_json(
        diagnostics_dir / "selection_summary.json",
        {
            "dataset": dataset,
            "split": split,
            "document_count": len(documents),
            "candidate_count": len(candidate_rows),
            "selected_count": len(mrs_result.selected_rows),
            "selector_gold_visible": False,
            "model_mode": "rule_based",
            "selector_scores": str(selector_scores_path),
            "selected_candidates": str(selected_candidates_path),
        },
    )

    handoff = build_evaluator_handoff(
        run_root=run_root,
        dataset=dataset,
        split=split,
        data_repo_root=data_repo_root,
        out_dir=evaluator_out_dir,
    )
    _write_json(diagnostics_dir / "evaluator_handoff.json", handoff.to_dict())
    backend_name = type(active_backend).__name__
    is_mock_backend = isinstance(active_backend, MockGetmBackend)
    backend_metadata = _json_ready_dict(_backend_generation_metadata(active_backend))
    manifest_path = write_run_manifest(
        run_root,
        run_id=resolved_run_id,
        dataset=dataset,
        split=split,
        seed=seed,
        command_infer=command_infer,
        # __file__ → src/sarge/pipeline/infer.py
        # parents: 0=pipeline 1=sarge 2=src 3=<project root> 4=<one above project>
        # We want the project root (which is the git repo).
        repo_root=Path(__file__).resolve().parents[3],
        backend=backend_name,
        backend_metadata=backend_metadata,
        generation_metadata=backend_metadata,
        limit=limit,
        document_count=len(documents),
        source_commit=source_commit,
        model_performance_evidence=not is_mock_backend,
        notes=f"SARGE inference via {backend_name}",
    )
    pipeline_summary = {
        "run_id": resolved_run_id,
        "run_root": str(run_root),
        "dataset": dataset,
        "split": split,
        "document_count": len(documents),
        "limit": limit,
        "backend": backend_name,
        "backend_metadata": backend_metadata,
        "model_performance_evidence": not is_mock_backend,
        "surface_memory": str(surface_memory_path),
        "slot_plan": str(slot_plan_path_out),
        "parsed_candidates": str(getm_output.parsed_candidates_path),
        "final_prediction": str(prediction_path),
        "run_manifest": str(manifest_path),
        "handoff_command": handoff.command,
        "artifact_layer_available": handoff.script_exists,
    }
    if is_mock_backend:
        pipeline_summary["mock_backend_notice"] = (
            "Mock GETM output is for pipeline smoke only, not model performance evidence."
        )
    _write_json(
        diagnostics_dir / "pipeline_summary.json",
        pipeline_summary,
    )

    return InferenceResult(
        run_id=resolved_run_id,
        run_root=run_root,
        prediction_path=prediction_path,
        run_manifest_path=manifest_path,
        handoff_command=handoff.command,
        handoff_script_exists=handoff.script_exists,
        handoff=handoff,
    )


def _load_or_build_slot_plans(
    *,
    dataset: str,
    split: str,
    data_root: str | Path,
    schema: Any,
    documents: list[Any],
    slot_plan_path: str | Path | None,
) -> list[SlotPlanDocument]:
    if slot_plan_path is not None:
        plans = [slot_plan_from_dict(row) for row in read_jsonl(slot_plan_path)]
        for plan in plans:
            validate_slot_plan(plan, schema)
        return plans
    train_documents = load_documents(dataset, "train", data_root=data_root, mode="train")
    planner = TrainPriorPlanner.fit(schema, train_documents)
    del split
    return planner.predict(documents)


def _default_run_id(dataset: str, split: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("_") or "dataset"
    split_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", split).strip("_") or "split"
    return f"sarge_infer_{dataset_slug}_{split_slug}_{timestamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _backend_generation_metadata(backend: Any) -> dict[str, Any]:
    metadata = getattr(backend, "generation_metadata", None)
    if callable(metadata):
        metadata = metadata()
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _json_ready_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in payload.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
