from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarge.data.jsonl import write_jsonl
from sarge.data.loader import V2DatasetDocument, load_documents
from sarge.data.schema import DatasetSchema, load_schema
from sarge.evaluation.export import export_predictions, validate_minimal_canonical_prediction
from sarge.generation.parser import candidate_set_to_canonical_prediction, parse_getm_output
from sarge.generation.prompt import build_getm_prompt
from sarge.models.qwen_backend import QwenGetmBackend
from sarge.record_planning import RecordPlanInstance, build_record_plan
from sarge.surface_memory.builder import build_surface_memory


@dataclass(frozen=True)
class GenerationTrace:
    doc_id: str
    unit_id: str
    mode: str
    event_type: str | None
    raw_output: str
    parse_status: str
    elapsed_sec: float
    diagnostics: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pilot regular extraction against oracle record-grounded extraction on same-type documents."
    )
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--doc-limit", type=int, default=3)
    parser.add_argument("--scan-limit", type=int, default=200)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--baseline-mode", default="role_safe_surface_memory")
    parser.add_argument("--carry-plan-anchors", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    schema = load_schema(args.dataset, data_root=args.data_root)
    documents = _select_same_type_documents(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        limit=args.doc_limit,
        scan_limit=args.scan_limit,
    )
    if not documents:
        raise ValueError(f"no same-type multi-event documents found in first {args.scan_limit} docs")

    run_name = args.run_name or _default_run_name(args.dataset, args.split)
    run_root = args.out_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    config = _qwen_config(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    backend = QwenGetmBackend(config=config)

    regular_rows, regular_traces = _run_regular_extraction(
        backend=backend,
        documents=documents,
        schema=schema,
        dataset=args.dataset,
        baseline_mode=args.baseline_mode,
    )
    oracle_rows, oracle_traces, plan_rows = _run_oracle_record_grounded_extraction(
        backend=backend,
        documents=documents,
        schema=schema,
        dataset=args.dataset,
        max_records=args.max_records,
        carry_plan_anchors=args.carry_plan_anchors,
    )

    regular_root = run_root / "regular"
    oracle_root = run_root / "oracle_record_grounded"
    _write_run_outputs(
        run_root=regular_root,
        dataset=args.dataset,
        split=args.split,
        predictions=regular_rows,
        traces=regular_traces,
        summary={
            "mode": "regular",
            "dataset": args.dataset,
            "split": args.split,
            "doc_count": len(regular_rows),
            "prediction_count": len(regular_rows),
        },
    )
    _write_run_outputs(
        run_root=oracle_root,
        dataset=args.dataset,
        split=args.split,
        predictions=oracle_rows,
        traces=oracle_traces,
        summary={
            "mode": "oracle_record_grounded",
            "dataset": args.dataset,
            "split": args.split,
            "doc_count": len(oracle_rows),
            "record_count": len(plan_rows),
            "carry_plan_anchors": bool(args.carry_plan_anchors),
        },
    )
    write_jsonl(run_root / "oracle_record_plans.jsonl", plan_rows)
    _write_json(
        run_root / "summary.json",
        {
            "dataset": args.dataset,
            "split": args.split,
            "selected_doc_ids": [doc.doc_id for doc in documents],
            "regular_run_root": str(regular_root),
            "oracle_record_grounded_run_root": str(oracle_root),
            "oracle_record_count": len(plan_rows),
            "model_path": str(args.model_path),
            "adapter_path": str(args.adapter_path),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    )
    print(json.dumps({"run_root": str(run_root), "docs": len(documents), "records": len(plan_rows)}, ensure_ascii=False))
    return 0


def _select_same_type_documents(
    *,
    dataset: str,
    split: str,
    data_root: Path,
    limit: int,
    scan_limit: int,
) -> list[V2DatasetDocument]:
    documents = load_documents(dataset, split, data_root=data_root, mode="eval_internal", limit=scan_limit)
    selected: list[V2DatasetDocument] = []
    for document in documents:
        events = (document.gold.events if document.gold is not None else [])
        counts = Counter(str(event.get("event_type") or "") for event in events)
        if any(count > 1 for count in counts.values()):
            selected.append(document)
        if len(selected) >= limit:
            break
    return selected


def _run_regular_extraction(
    *,
    backend: QwenGetmBackend,
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    dataset: str,
    baseline_mode: str,
) -> tuple[list[dict[str, Any]], list[GenerationTrace]]:
    predictions: list[dict[str, Any]] = []
    traces: list[GenerationTrace] = []
    for document in documents:
        memory = build_surface_memory(document.input)
        prompt = build_getm_prompt(
            dataset=dataset,
            schema=schema,
            document=document.input,
            surface_candidates=memory.candidates,
            slot_plan=None,
            max_surface_candidates=20,
            candidate_context_chars=0,
            candidate_render_mode="compact",
            enable_candidate_filtering=True,
            max_candidates_per_type=6,
            dedupe_surface_candidates=True,
            drop_low_value_company_fragments=True,
            output_format="minimal_text",
            baseline_mode=baseline_mode,
        )
        t0 = time.monotonic()
        raw_output = backend.generate_one(
            prompt=prompt,
            document=document.input,
            schema=schema,
            surface_candidates=memory.candidates,
            slot_plan=None,
            candidate_index=0,
        )
        elapsed = time.monotonic() - t0
        candidate = parse_getm_output(
            raw_output,
            doc_id=document.doc_id,
            candidate_id=f"{document.doc_id}:regular",
            schema=schema,
            response_prefix='{"events":',
            response_prefix_used=True,
            prompt=prompt,
            surface_candidate_count=len(memory.candidates),
            generation_metadata=backend.last_generation_metadata,
            output_format="minimal_text",
        )
        prediction = candidate_set_to_canonical_prediction(candidate, schema=schema)
        validate_minimal_canonical_prediction(prediction, schema=schema)
        predictions.append(prediction)
        traces.append(
            GenerationTrace(
                doc_id=document.doc_id,
                unit_id="regular",
                mode="regular",
                event_type=None,
                raw_output=raw_output,
                parse_status=candidate.parse_status,
                elapsed_sec=elapsed,
                diagnostics=dict(candidate.diagnostics),
            )
        )
    return predictions, traces


def _run_oracle_record_grounded_extraction(
    *,
    backend: QwenGetmBackend,
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    dataset: str,
    max_records: int,
    carry_plan_anchors: bool,
) -> tuple[list[dict[str, Any]], list[GenerationTrace], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    traces: list[GenerationTrace] = []
    plan_rows: list[dict[str, Any]] = []
    emitted_records = 0
    for document in documents:
        gold_events = document.gold.events if document.gold is not None else []
        plan = build_record_plan(gold_events, schema)
        doc_events: list[dict[str, Any]] = []
        for index, record_plan in enumerate(plan):
            if emitted_records >= max_records:
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
                candidate_id=f"{document.doc_id}:{record_plan.record_id}",
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
                    mode="oracle_record_grounded",
                    event_type=record_plan.event_type,
                    raw_output=raw_output,
                    parse_status=candidate.parse_status,
                    elapsed_sec=elapsed,
                    diagnostics=dict(candidate.diagnostics),
                )
            )
            plan_rows.append({"doc_id": document.doc_id, **record_plan.to_dict()})
            emitted_records += 1
        predictions.append({"doc_id": document.doc_id, "events": doc_events})
    return predictions, traces, plan_rows


def _build_record_conditioned_prompt(
    *,
    dataset: str,
    schema: DatasetSchema,
    document: V2DatasetDocument,
    record_plan: RecordPlanInstance,
) -> str:
    roles = ", ".join(schema.event_roles[record_plan.event_type])
    anchor_lines = []
    for role, values in record_plan.anchors.items():
        anchor_lines.append(f"- {role}: {'; '.join(values)}")
    anchors = "\n".join(anchor_lines) if anchor_lines else "(none)"
    return "\n".join(
        [
            "[Dataset]",
            f"dataset_id: {dataset}",
            "",
            "[Target Schema]",
            f"- {record_plan.event_type}: {roles}",
            "",
            "[Document]",
            f"doc_id: {document.doc_id}",
            "content:",
            document.input.content,
            "",
            "[Target Record]",
            f"record_id: {record_plan.record_id}",
            f"event_type: {record_plan.event_type}",
            "anchors:",
            anchors,
            "",
            "[Instruction]",
            "Return exactly one JSON object and no other text.",
            'The JSON object must have this shape: {"events":[{"event_type":"...","arguments":{"role":["value"]}}]}',
            "Extract only the target record described by [Target Record].",
            "Do not include values from another record, even if they have the same event type.",
            "Use only role names listed in [Target Schema].",
            "Keep original text spans from the document; do not normalize or translate values.",
            "If the target record is not supported by the document, return {\"events\":[]}.",
            "",
            "### RESPONSE_JSON",
        ]
    )


def _record_event_from_candidate(
    prediction: dict[str, Any],
    *,
    record_plan: RecordPlanInstance,
    carry_plan_anchors: bool,
) -> dict[str, Any] | None:
    events = prediction.get("events") or []
    if not isinstance(events, list):
        return None
    selected = None
    for event in events:
        if isinstance(event, dict) and event.get("event_type") == record_plan.event_type:
            selected = event
            break
    if selected is None:
        return None
    arguments = selected.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    if carry_plan_anchors:
        arguments = dict(arguments)
        for role, values in record_plan.anchors.items():
            arguments[role] = [{"text": value} for value in values]
    return {"event_type": record_plan.event_type, "arguments": arguments}


def _qwen_config(*, model_path: Path, adapter_path: Path, seed: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "version": "record-grounded-pilot",
        "run": {
            "profile": "record_grounded_pilot",
            "dry_run": False,
            "real_run": True,
            "real_run_resource_monitor": {"enabled": False},
        },
        "getm": {
            "backend": "qwen",
            "output_format": "minimal_text",
            "prompt": {
                "max_surface_candidates": 20,
                "candidate_context_chars": 0,
                "candidate_render_mode": "compact",
                "enable_candidate_filtering": True,
                "max_candidates_per_type": 6,
                "dedupe_surface_candidates": True,
                "drop_low_value_company_fragments": True,
                "prompt_token_budget": 4096,
                "fail_on_prompt_token_limit": False,
                "baseline_mode": "role_safe_surface_memory",
            },
            "qwen": {
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "model_path": str(model_path),
                "adapter_path": str(adapter_path),
                "quantization": "4-bit NF4",
                "double_quantization": True,
                "compute_dtype": "bf16",
            },
            "generation": {
                "k_candidates": 1,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "temperature": None,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "use_chat_template": True,
                "use_response_prefix": True,
                "response_prefix": '{"events":',
                "seed": int(seed),
                "deterministic": True,
                "deterministic_warn_only": True,
                "enable_balanced_json_stopping": True,
                "stop_after_balanced_events_json": True,
            },
        },
    }


def _write_run_outputs(
    *,
    run_root: Path,
    dataset: str,
    split: str,
    predictions: list[dict[str, Any]],
    traces: list[GenerationTrace],
    summary: dict[str, Any],
) -> None:
    prediction_path = run_root / "predictions" / dataset / f"{split}.canonical.pred.jsonl"
    export_predictions(predictions, prediction_path)
    write_jsonl(run_root / "intermediate" / "generation_traces.jsonl", [asdict(trace) for trace in traces])
    event_count = sum(len(row.get("events", [])) for row in predictions)
    value_count = sum(
        len(values)
        for row in predictions
        for event in row.get("events", [])
        for values in (event.get("arguments") or {}).values()
    )
    _write_json(
        run_root / "summary.json",
        {
            **summary,
            "prediction_path": str(prediction_path),
            "event_count": event_count,
            "value_count": value_count,
            "generation_trace_count": len(traces),
            "elapsed_sec": round(sum(trace.elapsed_sec for trace in traces), 3),
        },
    )


def _default_run_name(dataset: str, split: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_dataset = "".join(ch if ch.isalnum() else "_" for ch in dataset).strip("_")
    return f"record_grounded_pilot_{safe_dataset}_{split}_{timestamp}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
