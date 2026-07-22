"""Run inference on a merged BF16 checkpoint via vLLM.

Mirrors scripts/infer_checkpoint.py CLI surface but swaps the backend
to VllmGetmBackend. Requires a pre-merged BF16 ckpt (no PEFT adapter
at runtime); use scripts/merge_lora_to_bf16.py to produce one.

  python scripts/infer_checkpoint_vllm.py \
    --merged /data/TJK/DEE/SARGE/runs/merged_models/qwen3_4b_chfinann_ep2_s13 \
    --dataset ChFinAnn-Doc2EDAG --split dev --limit 500
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data"

from sarge.data.schema import load_schema  # noqa: E402
from sarge.data.staging import stage_dataset  # noqa: E402
from sarge.generation.schema_decoding import build_dataset_json_schema  # noqa: E402
from sarge.models.vllm_backend import VllmGetmBackend  # noqa: E402
from sarge.pipeline.infer import run_inference  # noqa: E402


def _build_generation_config(args) -> dict[str, Any]:
    if args.sample:
        generation = {
            "k_candidates": args.k,
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "use_chat_template": True,
            "use_response_prefix": True,
            "response_prefix": '{"events":',
            "seed": args.seed,
            "enable_balanced_json_stopping": True,
            "stop_after_balanced_events_json": True,
        }
    else:
        generation = {
            "k_candidates": args.k,
            "max_new_tokens": 1024,
            "do_sample": False,
            "temperature": None,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "use_chat_template": True,
            "use_response_prefix": True,
            "response_prefix": '{"events":',
            "seed": args.seed,
            "deterministic": True,
            "deterministic_warn_only": True,
            "enable_balanced_json_stopping": True,
            "stop_after_balanced_events_json": True,
        }
    if getattr(args, "sacd", False):
        generation["sacd_strict"] = bool(getattr(args, "sacd_strict", False))
        generation["use_response_prefix"] = False
        generation["response_prefix"] = ""
    return generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", required=True, help="Merged BF16 ckpt directory")
    parser.add_argument("--dataset", default="ChFinAnn-Doc2EDAG")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--processed", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--slot-train-limit", type=int, default=50)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--source-commit",
        default=None,
        help="source git commit to record when running from a non-git server copy",
    )

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=None)

    parser.add_argument("--batch-mode", choices=("per_prompt", "prefilled"), default="prefilled",
                        help="prefilled = pre-batch all prompts before pipeline; per_prompt = one call per doc")

    parser.add_argument("--sacd", action="store_true",
                        help="enable Schema-Aware Constrained Decoding via vLLM guided_decoding")
    parser.add_argument("--sacd-strict", action="store_true",
                        help="strict SACD: oneOf per event_type with role-name closure (slower compile, tighter constraints)")
    parser.add_argument("--sacd-backend", default=None,
                        help="vLLM guided_decoding backend (e.g. xgrammar, outlines, lm-format-enforcer)")

    args = parser.parse_args()
    if args.k is None:
        args.k = 4 if args.sample else 1
    if not Path(args.merged).is_dir():
        parser.error(f"merged ckpt not found: {args.merged}")

    staging = Path(tempfile.mkdtemp(prefix="sarge_infer_vllm_"))
    try:
        _run_inference(args, staging)
    finally:
        import shutil
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return 0


def _run_inference(args, staging):
    stage_dataset(
        dataset=args.dataset,
        processed_root=args.processed,
        output_root=staging,
        splits=("train",),
        limit=args.slot_train_limit,
    )
    stage_dataset(dataset=args.dataset, processed_root=args.processed, output_root=staging, splits=(args.split,), limit=args.limit)

    qwen_cfg = {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_path": args.merged,
        "compute_dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    generation_cfg = _build_generation_config(args)
    if args.sacd:
        schema = load_schema(args.dataset, data_root=staging)
        generation_cfg["sacd_json_schema"] = build_dataset_json_schema(
            schema, strict=args.sacd_strict,
        )
        if args.sacd_backend:
            generation_cfg["sacd_backend"] = args.sacd_backend
    mode_label = "sampling" if generation_cfg["do_sample"] else "greedy"
    sacd_label = "+SACD" + ("(strict)" if args.sacd_strict else "(lax)") if args.sacd else ""
    print(f"[mode] {mode_label}{sacd_label}  k={generation_cfg['k_candidates']}  batch={args.batch_mode}", flush=True)

    config = {
        "version": "v1",
        "run": {"profile": "eval", "dry_run": False, "real_run": True, "real_run_resource_monitor": {"enabled": False}},
        "getm": {
            "backend": "vllm", "output_format": "minimal_text",
            "prompt": {
                "max_surface_candidates": 20, "candidate_context_chars": 0, "candidate_render_mode": "compact",
                "enable_candidate_filtering": True, "max_candidates_per_type": 6, "dedupe_surface_candidates": True,
                "drop_low_value_company_fragments": True, "prompt_token_budget": 4096,
                "fail_on_prompt_token_limit": False,
            },
            "qwen": qwen_cfg,
            "generation": generation_cfg,
        },
    }

    backend = VllmGetmBackend(config=config)

    if args.batch_mode == "prefilled":
        # Build prompts up-front so vllm can batch them all at once.
        from sarge.data.loader import load_documents
        from sarge.surface_memory.candidate_builder import build_surface_memories
        from sarge.surface_memory.builder import build_surface_memory
        from sarge.generation.prompt import build_getm_prompt_result
        from sarge.slot_planning.baseline import TrainPriorPlanner

        schema = load_schema(args.dataset, data_root=staging)
        documents = load_documents(args.dataset, args.split, data_root=staging, mode="predict", limit=args.limit)
        train_docs = load_documents(args.dataset, "train", data_root=staging, mode="train")
        planner = TrainPriorPlanner.fit(schema, train_docs)
        slot_plans = {p.doc_id: p for p in planner.predict(documents)}
        memories = {m.doc_id: m for m in build_surface_memories(documents)}

        prompts_to_run: list[tuple[str, int, str]] = []
        from sarge.generation.candidate_generator import _backend_prompt_options
        prompt_options = _backend_prompt_options(backend.generation_metadata)
        for doc in documents:
            mem = memories.get(doc.doc_id) or build_surface_memory(doc.input)
            sc = list(mem.candidates)
            sp = slot_plans.get(doc.doc_id)
            pr = build_getm_prompt_result(
                dataset=args.dataset, schema=schema, document=doc.input,
                surface_candidates=sc, slot_plan=sp, **prompt_options,
            )
            for ci in range(int(generation_cfg["k_candidates"])):
                prompts_to_run.append((doc.doc_id, ci, pr.prompt))

        print(f"[prebatch] running vllm on {len(prompts_to_run)} prompts ...", flush=True)
        t_pre = time.monotonic()
        backend.preload_prompts(prompts_to_run)
        print(f"[prebatch] vllm batch done in {time.monotonic()-t_pre:.0f}s", flush=True)

    t0 = time.monotonic()
    result = run_inference(
        dataset=args.dataset, split=args.split, data_root=staging, out_root=args.out,
        limit=args.limit, seed=args.seed, k=generation_cfg["k_candidates"],
        command_infer=" ".join([sys.executable, *sys.argv]),
        source_commit=args.source_commit,
        backend=backend,
    )
    elapsed = time.monotonic() - t0
    rows = [json.loads(line) for line in result.prediction_path.read_text().splitlines() if line.strip()]
    events = sum(len(r.get("events", [])) for r in rows)
    print(f"DONE pred={result.prediction_path} docs={len(rows)} events={events} pipeline_time={elapsed:.0f}s")


if __name__ == "__main__":
    main()
