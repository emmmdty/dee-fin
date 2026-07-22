from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarge.data.jsonl import read_jsonl, write_jsonl
from sarge.data.loader import V2DatasetDocument, load_documents
from sarge.data.schema import DatasetSchema, load_schema
from sarge.evaluation.export import validate_minimal_canonical_prediction
from sarge.generation.parser import candidate_set_to_canonical_prediction, parse_getm_output
from sarge.models.qwen_backend import QwenGetmBackend
from sarge.record_planning import RecordPlanInstance, build_record_plan
from sarge.experiments.predicted_record_plan_pilot import _plan_from_payload
from sarge.experiments.record_grounded_pilot import (
    GenerationTrace,
    _build_record_conditioned_prompt,
    _qwen_config,
    _record_event_from_candidate,
    _select_same_type_documents,
    _write_run_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run record-grounding ablations from an existing predicted-record-plan run."
    )
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--doc-limit", type=int, default=109)
    parser.add_argument("--scan-limit", type=int, default=500)
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--carry-oracle-plan-anchors", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    schema = load_schema(args.dataset, data_root=args.data_root)
    documents = _load_eval_documents(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        source_run_root=args.source_run_root,
        doc_limit=args.doc_limit,
        scan_limit=args.scan_limit,
    )
    if not documents:
        raise ValueError("no evaluation documents selected for record-grounding ablation")

    predicted_plans = _load_predicted_plans(args.source_run_root, schema=schema)
    type_count_plans = _strip_plan_anchors(predicted_plans)
    oracle_plans = _oracle_plans(documents, schema=schema, max_records=args.max_records)

    run_root = args.out_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    backend = QwenGetmBackend(
        config=_qwen_config(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    )

    type_count_rows, type_count_traces = _run_record_conditioned_extraction(
        backend=backend,
        documents=documents,
        schema=schema,
        dataset=args.dataset,
        plans_by_doc=type_count_plans,
        mode="type_count_only_record_grounded",
        max_records=args.max_records,
        carry_plan_anchors=False,
    )
    oracle_rows, oracle_traces = _run_record_conditioned_extraction(
        backend=backend,
        documents=documents,
        schema=schema,
        dataset=args.dataset,
        plans_by_doc=oracle_plans,
        mode="oracle_record_grounded",
        max_records=args.max_records,
        carry_plan_anchors=args.carry_oracle_plan_anchors,
    )

    _write_run_outputs(
        run_root=run_root / "type_count_only_record_grounded",
        dataset=args.dataset,
        split=args.split,
        predictions=type_count_rows,
        traces=type_count_traces,
        summary={
            "mode": "type_count_only_record_grounded",
            "dataset": args.dataset,
            "split": args.split,
            "doc_count": len(type_count_rows),
            "record_count": sum(len(type_count_plans.get(doc.doc_id, [])) for doc in documents),
            "source_run_root": str(args.source_run_root),
        },
    )
    _write_run_outputs(
        run_root=run_root / "oracle_record_grounded",
        dataset=args.dataset,
        split=args.split,
        predictions=oracle_rows,
        traces=oracle_traces,
        summary={
            "mode": "oracle_record_grounded",
            "dataset": args.dataset,
            "split": args.split,
            "doc_count": len(oracle_rows),
            "record_count": sum(len(oracle_plans.get(doc.doc_id, [])) for doc in documents),
            "carry_plan_anchors": bool(args.carry_oracle_plan_anchors),
        },
    )
    write_jsonl(
        run_root / "type_count_only_record_plans.jsonl",
        [
            {"doc_id": doc_id, **plan.to_dict()}
            for doc_id, plans in type_count_plans.items()
            for plan in plans
        ],
    )
    write_jsonl(
        run_root / "oracle_record_plans.jsonl",
        [
            {"doc_id": doc_id, **plan.to_dict()}
            for doc_id, plans in oracle_plans.items()
            for plan in plans
        ],
    )
    _write_json(
        run_root / "summary.json",
        {
            "dataset": args.dataset,
            "split": args.split,
            "selected_doc_ids": [doc.doc_id for doc in documents],
            "source_run_root": str(args.source_run_root),
            "type_count_only_run_root": str(run_root / "type_count_only_record_grounded"),
            "oracle_record_grounded_run_root": str(run_root / "oracle_record_grounded"),
            "type_count_record_count": sum(len(type_count_plans.get(doc.doc_id, [])) for doc in documents),
            "oracle_record_count": sum(len(oracle_plans.get(doc.doc_id, [])) for doc in documents),
            "model_path": str(args.model_path),
            "adapter_path": str(args.adapter_path),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "carry_oracle_plan_anchors": bool(args.carry_oracle_plan_anchors),
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "docs": len(documents),
                "type_count_records": sum(len(type_count_plans.get(doc.doc_id, [])) for doc in documents),
                "oracle_records": sum(len(oracle_plans.get(doc.doc_id, [])) for doc in documents),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_eval_documents(
    *,
    dataset: str,
    split: str,
    data_root: Path,
    source_run_root: Path,
    doc_limit: int,
    scan_limit: int,
) -> list[V2DatasetDocument]:
    summary_path = source_run_root / "summary.json"
    if not summary_path.exists():
        return _select_same_type_documents(
            dataset=dataset,
            split=split,
            data_root=data_root,
            limit=doc_limit,
            scan_limit=scan_limit,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_ids = [str(doc_id) for doc_id in summary.get("selected_doc_ids", []) if doc_id]
    if not selected_ids:
        return _select_same_type_documents(
            dataset=dataset,
            split=split,
            data_root=data_root,
            limit=doc_limit,
            scan_limit=scan_limit,
        )
    by_id = {
        document.doc_id: document
        for document in load_documents(dataset, split, data_root=data_root, mode="eval_internal", limit=scan_limit)
    }
    return [by_id[doc_id] for doc_id in selected_ids[:doc_limit] if doc_id in by_id]


def _load_predicted_plans(source_run_root: Path, *, schema: DatasetSchema) -> dict[str, list[RecordPlanInstance]]:
    rows = read_jsonl(source_run_root / "predicted_record_plans.jsonl")
    by_doc: dict[str, list[RecordPlanInstance]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        plan = _plan_from_payload(row, index=len(by_doc.get(doc_id, [])) + 1, schema=schema)
        if plan is None:
            continue
        by_doc.setdefault(doc_id, []).append(plan)
    return by_doc


def _strip_plan_anchors(
    plans_by_doc: dict[str, list[RecordPlanInstance]],
) -> dict[str, list[RecordPlanInstance]]:
    return {
        doc_id: [
            RecordPlanInstance(record_id=plan.record_id, event_type=plan.event_type, anchors={})
            for plan in plans
        ]
        for doc_id, plans in plans_by_doc.items()
    }


def _oracle_plans(
    documents: list[V2DatasetDocument],
    *,
    schema: DatasetSchema,
    max_records: int,
) -> dict[str, list[RecordPlanInstance]]:
    by_doc: dict[str, list[RecordPlanInstance]] = {}
    emitted = 0
    for document in documents:
        if emitted >= max_records:
            by_doc[document.doc_id] = []
            continue
        plans = build_record_plan(document.gold.events if document.gold else [], schema)
        kept = plans[: max(0, max_records - emitted)]
        by_doc[document.doc_id] = kept
        emitted += len(kept)
    return by_doc


def _run_record_conditioned_extraction(
    *,
    backend: QwenGetmBackend,
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    dataset: str,
    plans_by_doc: dict[str, list[RecordPlanInstance]],
    mode: str,
    max_records: int,
    carry_plan_anchors: bool,
) -> tuple[list[dict[str, Any]], list[GenerationTrace]]:
    predictions: list[dict[str, Any]] = []
    traces: list[GenerationTrace] = []
    emitted = 0
    for document in documents:
        doc_events: list[dict[str, Any]] = []
        for index, record_plan in enumerate(plans_by_doc.get(document.doc_id, [])):
            if emitted >= max_records:
                break
            prompt = _build_record_conditioned_prompt(
                dataset=dataset,
                schema=schema,
                document=document,
                record_plan=record_plan,
            )
            t0 = time.monotonic()
            raw_output = backend.generate_one(
                prompt=prompt,
                document=document.input,
                schema=schema,
                surface_candidates=[],
                slot_plan=None,
                candidate_index=index,
            )
            elapsed = time.monotonic() - t0
            candidate = parse_getm_output(
                raw_output,
                doc_id=document.doc_id,
                candidate_id=f"{document.doc_id}:{mode}:{record_plan.record_id}",
                schema=schema,
                response_prefix='{"events":',
                response_prefix_used=True,
                prompt=prompt,
                surface_candidate_count=0,
                generation_metadata=backend.last_generation_metadata,
                output_format="minimal_text",
            )
            record_event = _record_event_from_candidate(
                candidate_set_to_canonical_prediction(candidate, schema=schema),
                record_plan=record_plan,
                carry_plan_anchors=carry_plan_anchors,
            )
            if record_event is not None:
                doc_events.append(record_event)
            traces.append(
                GenerationTrace(
                    doc_id=document.doc_id,
                    unit_id=record_plan.record_id,
                    mode=mode,
                    event_type=record_plan.event_type,
                    raw_output=raw_output,
                    parse_status=candidate.parse_status,
                    elapsed_sec=elapsed,
                    diagnostics=dict(candidate.diagnostics),
                )
            )
            emitted += 1
        prediction = {"doc_id": document.doc_id, "events": doc_events}
        validate_minimal_canonical_prediction(prediction, schema=schema)
        predictions.append(prediction)
    return predictions, traces


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
