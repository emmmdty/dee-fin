from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarge.data.jsonl import write_jsonl
from sarge.data.loader import V2DatasetDocument, load_documents
from sarge.data.schema import DatasetSchema, load_schema
from sarge.evaluation.export import export_predictions, validate_minimal_canonical_prediction
from sarge.generation.parser import candidate_set_to_canonical_prediction, parse_getm_output
from sarge.models.qwen_backend import QwenGetmBackend, train_sft
from sarge.record_planning import RecordPlanInstance, build_record_plan
from sarge.experiments.record_grounded_pilot import (
    GenerationTrace,
    _build_record_conditioned_prompt,
    _qwen_config as _extractor_qwen_config,
    _record_event_from_candidate,
    _run_regular_extraction,
    _select_same_type_documents,
    _write_run_outputs,
)

PLAN_RESPONSE_PREFIX = '{"record_plan":'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a plan-only adapter, then test predicted record plans in two-stage extraction."
    )
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--extractor-adapter-path", type=Path, required=True)
    parser.add_argument("--plan-adapter-path", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--plan-train-docs", type=int, default=64)
    parser.add_argument("--plan-train-steps", type=int, default=24)
    parser.add_argument("--doc-limit", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=200)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--plan-max-new-tokens", type=int, default=256)
    parser.add_argument("--extract-max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)

    schema = load_schema(args.dataset, data_root=args.data_root)
    run_name = args.run_name or _default_run_name(args.dataset, args.split)
    run_root = args.out_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    if args.plan_adapter_path is None:
        train_docs = load_documents(
            args.dataset,
            "train",
            data_root=args.data_root,
            mode="train",
            limit=args.plan_train_docs,
        )
        plan_rows = [_build_plan_sft_row(document, schema=schema) for document in train_docs]
        plan_config = _plan_qwen_config(
            model_path=args.model_path,
            seed=args.seed,
            max_new_tokens=args.plan_max_new_tokens,
            max_train_steps=args.plan_train_steps,
        )
        train_t0 = time.monotonic()
        train_manifest = train_sft(plan_config, plan_rows, run_root / "plan_training")
        train_secs = time.monotonic() - train_t0
        plan_adapter = Path(str(train_manifest["adapter_dir"]))
        train_doc_count = len(train_docs)
        reused_plan_adapter = False
    else:
        plan_adapter = args.plan_adapter_path
        if not plan_adapter.exists():
            raise FileNotFoundError(f"plan adapter not found: {plan_adapter}")
        train_secs = 0.0
        train_doc_count = int(args.plan_train_docs)
        reused_plan_adapter = True

    eval_docs = _select_same_type_documents(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        limit=args.doc_limit,
        scan_limit=args.scan_limit,
    )
    if not eval_docs:
        raise ValueError(f"no same-type multi-event documents found in first {args.scan_limit} docs")

    plan_gen_config = _plan_qwen_config(
        model_path=args.model_path,
        seed=args.seed,
        max_new_tokens=args.plan_max_new_tokens,
        max_train_steps=args.plan_train_steps,
        adapter_path=plan_adapter,
    )
    plan_backend = QwenGetmBackend(config=plan_gen_config)
    predicted_plans, plan_traces, plan_eval = _predict_record_plans(
        backend=plan_backend,
        documents=eval_docs,
        schema=schema,
        dataset=args.dataset,
        max_records=args.max_records,
    )
    del plan_backend
    _release_cuda_cache()

    extractor_config = _extractor_qwen_config(
        model_path=args.model_path,
        adapter_path=args.extractor_adapter_path,
        seed=args.seed,
        max_new_tokens=args.extract_max_new_tokens,
    )
    extractor_backend = QwenGetmBackend(config=extractor_config)
    regular_rows, regular_traces = _run_regular_extraction(
        backend=extractor_backend,
        documents=eval_docs,
        schema=schema,
        dataset=args.dataset,
        baseline_mode="role_safe_surface_memory",
    )
    predicted_rows, extraction_traces = _run_predicted_record_grounded_extraction(
        backend=extractor_backend,
        documents=eval_docs,
        schema=schema,
        dataset=args.dataset,
        predicted_plans=predicted_plans,
        max_records=args.max_records,
    )

    _write_run_outputs(
        run_root=run_root / "regular",
        dataset=args.dataset,
        split=args.split,
        predictions=regular_rows,
        traces=regular_traces,
        summary={
            "mode": "regular",
            "dataset": args.dataset,
            "split": args.split,
            "doc_count": len(regular_rows),
        },
    )
    _write_run_outputs(
        run_root=run_root / "predicted_record_grounded",
        dataset=args.dataset,
        split=args.split,
        predictions=predicted_rows,
        traces=extraction_traces,
        summary={
            "mode": "predicted_record_grounded",
            "dataset": args.dataset,
            "split": args.split,
            "doc_count": len(predicted_rows),
            "predicted_record_count": sum(len(plans) for plans in predicted_plans.values()),
        },
    )
    write_jsonl(run_root / "plan_generation_traces.jsonl", [asdict(trace) for trace in plan_traces])
    write_jsonl(
        run_root / "predicted_record_plans.jsonl",
        [
            {"doc_id": doc_id, **plan.to_dict()}
            for doc_id, plans in predicted_plans.items()
            for plan in plans
        ],
    )
    _write_json(
        run_root / "summary.json",
        {
            "dataset": args.dataset,
            "split": args.split,
            "selected_doc_ids": [doc.doc_id for doc in eval_docs],
            "train_docs": train_doc_count,
            "plan_train_steps": args.plan_train_steps,
            "plan_train_secs": round(train_secs, 1),
            "reused_plan_adapter": reused_plan_adapter,
            "plan_adapter_path": str(plan_adapter),
            "extractor_adapter_path": str(args.extractor_adapter_path),
            "regular_run_root": str(run_root / "regular"),
            "predicted_record_grounded_run_root": str(run_root / "predicted_record_grounded"),
            "plan_eval": plan_eval,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "plan_adapter": str(plan_adapter),
                "docs": len(eval_docs),
                "plan_eval": plan_eval,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _build_plan_sft_row(document: V2DatasetDocument, *, schema: DatasetSchema) -> dict[str, Any]:
    if document.gold is None:
        raise ValueError("plan SFT requires gold-visible train documents")
    return {
        "doc_id": document.doc_id,
        "dataset": schema.dataset_id,
        "split": document.input.split,
        "prompt": _build_plan_prompt(dataset=schema.dataset_id, schema=schema, document=document),
        "output": {"record_plan": [plan.to_dict() for plan in build_record_plan(document.gold.events, schema)]},
    }


def _predict_record_plans(
    *,
    backend: QwenGetmBackend,
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    dataset: str,
    max_records: int,
) -> tuple[dict[str, list[RecordPlanInstance]], list[GenerationTrace], dict[str, Any]]:
    predicted_by_doc: dict[str, list[RecordPlanInstance]] = {}
    traces: list[GenerationTrace] = []
    oracle_by_doc = {
        document.doc_id: build_record_plan(document.gold.events if document.gold else [], schema)
        for document in documents
    }
    emitted = 0
    for document in documents:
        prompt = _build_plan_prompt(dataset=dataset, schema=schema, document=document)
        t0 = time.monotonic()
        raw_output = backend.generate_one(
            prompt=prompt,
            document=document.input,
            schema=schema,
            surface_candidates=[],
            slot_plan=None,
            candidate_index=0,
        )
        elapsed = time.monotonic() - t0
        plans, parse_status, diagnostics = _parse_plan_output(
            raw_output,
            schema=schema,
            max_remaining=max(0, max_records - emitted),
        )
        emitted += len(plans)
        predicted_by_doc[document.doc_id] = plans
        traces.append(
            GenerationTrace(
                doc_id=document.doc_id,
                unit_id="record_plan",
                mode="predicted_record_plan",
                event_type=None,
                raw_output=raw_output,
                parse_status=parse_status,
                elapsed_sec=elapsed,
                diagnostics=diagnostics,
            )
        )
    return predicted_by_doc, traces, _evaluate_plans(predicted_by_doc, oracle_by_doc)


def _run_predicted_record_grounded_extraction(
    *,
    backend: QwenGetmBackend,
    documents: list[V2DatasetDocument],
    schema: DatasetSchema,
    dataset: str,
    predicted_plans: dict[str, list[RecordPlanInstance]],
    max_records: int,
) -> tuple[list[dict[str, Any]], list[GenerationTrace]]:
    predictions: list[dict[str, Any]] = []
    traces: list[GenerationTrace] = []
    emitted = 0
    for document in documents:
        doc_events: list[dict[str, Any]] = []
        for index, record_plan in enumerate(predicted_plans.get(document.doc_id, [])):
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
                carry_plan_anchors=False,
            )
            if record_event is not None:
                doc_events.append(record_event)
            traces.append(
                GenerationTrace(
                    doc_id=document.doc_id,
                    unit_id=record_plan.record_id,
                    mode="predicted_record_grounded",
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


def _build_plan_prompt(*, dataset: str, schema: DatasetSchema, document: V2DatasetDocument) -> str:
    schema_lines = [f"- {event_type}: {', '.join(roles)}" for event_type, roles in schema.event_roles.items()]
    return "\n".join(
        [
            "[Dataset]",
            f"dataset_id: {dataset}",
            "",
            "[Schema]",
            "\n".join(schema_lines),
            "",
            "[Document]",
            f"doc_id: {document.doc_id}",
            "content:",
            document.input.content,
            "",
            "[Instruction]",
            "Return exactly one JSON object and no other text.",
            'The JSON object must have this shape: {"record_plan":[{"record_id":"R1","event_type":"...","anchors":{"role":["value"]}}]}',
            "Create one record_plan item for each event record in the document.",
            "For repeated same-type records, choose private anchors that distinguish records.",
            "Use only event types and role names listed in [Schema].",
            "Do not output event arguments other than anchors.",
            "Keep anchor values as original text spans from the document.",
            "",
            "### RESPONSE_JSON",
        ]
    )


def _parse_plan_output(
    raw_output: str,
    *,
    schema: DatasetSchema,
    max_remaining: int,
) -> tuple[list[RecordPlanInstance], str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"raw_output_chars": len(raw_output), "invalid_item_count": 0}
    if max_remaining <= 0:
        diagnostics["max_records_exhausted"] = 1
        return [], "skipped", diagnostics
    try:
        payload = _extract_plan_json(raw_output)
    except Exception as exc:
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return [], "parse_error", diagnostics
    raw_plan = payload.get("record_plan") if isinstance(payload, dict) else None
    if not isinstance(raw_plan, list):
        diagnostics["error"] = "missing record_plan list"
        return [], "schema_violation", diagnostics
    plans: list[RecordPlanInstance] = []
    for index, item in enumerate(raw_plan, 1):
        if len(plans) >= max_remaining:
            break
        plan = _plan_from_payload(item, index=index, schema=schema)
        if plan is None:
            diagnostics["invalid_item_count"] += 1
            continue
        plans.append(plan)
    diagnostics["accepted_plan_count"] = len(plans)
    return plans, "ok" if plans else "empty", diagnostics


def _extract_plan_json(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    candidates = []
    if text.startswith("["):
        candidates.append(f"{PLAN_RESPONSE_PREFIX}{text}")
        if not text.endswith("}"):
            candidates.append(f"{PLAN_RESPONSE_PREFIX}{text}" + "}")
    candidates.append(text)
    if not text.startswith("{") and not text.startswith("["):
        candidates.append(f"{PLAN_RESPONSE_PREFIX}{text}")
    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        end = _balanced_json_end(candidate, start)
        if end is None:
            continue
        payload = json.loads(candidate[start:end])
        if isinstance(payload, list):
            return {"record_plan": payload}
        return payload
    if text.startswith("["):
        prefix_items = _extract_object_items_from_array_prefix(text)
        if prefix_items:
            return {"record_plan": prefix_items}
    raise ValueError("no complete JSON object")


def _extract_object_items_from_array_prefix(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        end = _balanced_json_end(text, start)
        if end is None:
            break
        try:
            payload = json.loads(text[start:end])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            items.append(payload)
        index = end
    return items


def _balanced_json_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _plan_from_payload(item: Any, *, index: int, schema: DatasetSchema) -> RecordPlanInstance | None:
    if not isinstance(item, dict):
        return None
    event_type = str(item.get("event_type") or "").strip()
    if event_type not in schema.event_roles:
        return None
    record_id = str(item.get("record_id") or f"R{index}").strip() or f"R{index}"
    anchors_raw = item.get("anchors") or {}
    anchors: dict[str, list[str]] = {}
    if isinstance(anchors_raw, dict):
        for role, values in anchors_raw.items():
            role_text = str(role).strip()
            if role_text not in schema.event_roles[event_type]:
                continue
            value_list = values if isinstance(values, list) else [values]
            clean_values = [_normalize_text(value) for value in value_list]
            clean_values = [value for value in clean_values if value]
            if clean_values:
                anchors[role_text] = clean_values
    return RecordPlanInstance(record_id=record_id, event_type=event_type, anchors=anchors)


def _evaluate_plans(
    predicted_by_doc: dict[str, list[RecordPlanInstance]],
    oracle_by_doc: dict[str, list[RecordPlanInstance]],
) -> dict[str, Any]:
    doc_count = len(oracle_by_doc)
    record_count_exact = 0
    type_multiset_exact = 0
    anchor_matches = 0
    oracle_records = 0
    predicted_records = 0
    for doc_id, oracle in oracle_by_doc.items():
        predicted = predicted_by_doc.get(doc_id, [])
        oracle_records += len(oracle)
        predicted_records += len(predicted)
        if len(predicted) == len(oracle):
            record_count_exact += 1
        if Counter(plan.event_type for plan in predicted) == Counter(plan.event_type for plan in oracle):
            type_multiset_exact += 1
        remaining = list(oracle)
        for plan in predicted:
            match_index = next(
                (
                    idx
                    for idx, oracle_plan in enumerate(remaining)
                    if plan.event_type == oracle_plan.event_type
                    and _anchor_signature(plan) == _anchor_signature(oracle_plan)
                ),
                None,
            )
            if match_index is not None:
                anchor_matches += 1
                del remaining[match_index]
    return {
        "doc_count": doc_count,
        "oracle_records": oracle_records,
        "predicted_records": predicted_records,
        "record_count_exact_docs": record_count_exact,
        "record_count_accuracy": record_count_exact / doc_count if doc_count else 0.0,
        "event_type_multiset_exact_docs": type_multiset_exact,
        "event_type_multiset_accuracy": type_multiset_exact / doc_count if doc_count else 0.0,
        "anchor_exact_matches": anchor_matches,
        "anchor_exact_recall": anchor_matches / oracle_records if oracle_records else 0.0,
    }


def _anchor_signature(plan: RecordPlanInstance) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((role, tuple(_normalize_text(value) for value in values)) for role, values in sorted(plan.anchors.items()))


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _plan_qwen_config(
    *,
    model_path: Path,
    seed: int,
    max_new_tokens: int,
    max_train_steps: int,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "version": "record-plan-pilot",
        "run": {
            "profile": "record_plan_pilot",
            "dry_run": False,
            "real_run": True,
            "real_run_resource_monitor": {"enabled": False},
        },
        "getm": {
            "backend": "qwen",
            "output_format": "minimal_text",
            "qwen": {
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "model_path": str(model_path),
                "adapter_path": str(adapter_path) if adapter_path is not None else None,
                "quantization": "4-bit NF4",
                "double_quantization": True,
                "compute_dtype": "bf16",
                "lora": {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "training": {
                    "num_train_epochs": 1,
                    "learning_rate": 0.0002,
                    "logging_steps": 1,
                    "optimizer": "paged_adamw_8bit",
                    "micro_batch_size": 1,
                    "gradient_accumulation": 4,
                    "max_seq_len": 4096,
                    "gradient_checkpointing": True,
                    "max_train_steps": int(max_train_steps),
                    "seed": int(seed),
                },
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
                "response_prefix": PLAN_RESPONSE_PREFIX,
                "seed": int(seed),
                "deterministic": True,
                "deterministic_warn_only": True,
                "enable_balanced_json_stopping": True,
                "stop_after_balanced_events_json": False,
            },
        },
    }


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def _default_run_name(dataset: str, split: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("_") or "dataset"
    return f"predicted_record_plan_pilot_{safe_dataset}_{split}_{timestamp}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
