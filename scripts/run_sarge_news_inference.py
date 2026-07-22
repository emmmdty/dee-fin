#!/usr/bin/env python
"""SARGE inference over exported news JSONL — the closed loop's first hop (D1).

SARGE's own inference scripts stage datasets through per-benchmark converters
(ChFinAnn / DuEE-Fin / DocFEE), so arbitrary news can't enter that path. This
driver keeps SARGE untouched and instead materialises the staged layout its
loader already understands: the event schema and the slot-plan train rows are
borrowed from a source benchmark (DuEE-Fin by default — the schema the merged
checkpoint was trained on), and the exported news rows
(`preprocess_datasets.py --export-sarge-input`) become the predict split. From
there it mirrors `external/sarge/scripts/infer_checkpoint_vllm.py`'s prefilled
path: TrainPriorPlanner slot plans + surface memories + one big vLLM batch.

Staging is idempotent (a pre-staged source dir is reused), and ``--dry-run``
stops after staging — the CPU-testable half. GPU half (server, one card):

    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    uv run --extra llm --extra serve python scripts/run_sarge_news_inference.py \\
      --news-jsonl data/processed/astock/sarge_input.jsonl \\
      --merged /data/TJK/DEE/SARGE/runs/merged_models/qwen3_4b_dueefin_ep2_s13 \\
      --sarge-data /data/TJK/DEE/SARGE/data \\
      --staging-root runs/sarge_news_staging --out runs/sarge_astock --limit 500
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SARGE_SRC = REPO_ROOT / "external" / "sarge" / "src"
if str(SARGE_SRC) not in sys.path:
    sys.path.insert(0, str(SARGE_SRC))


def _read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _stage_source(args) -> Path:
    """Ensure `<staging>/<schema-dataset>/{schema.json,train.jsonl}` exists.

    Reuses a pre-staged dir when present (idempotent re-runs, CPU tests);
    otherwise stages it from the SARGE processed data via `stage_dataset`.
    """
    source_dir = Path(args.staging_root) / args.schema_dataset
    if (source_dir / "schema.json").exists() and (source_dir / "train.jsonl").exists():
        return source_dir
    from sarge.data.staging import stage_dataset  # noqa: PLC0415 — needs sarge data

    return stage_dataset(
        dataset=args.schema_dataset,
        processed_root=args.sarge_data,
        output_root=args.staging_root,
        splits=("train",),
        limit=args.slot_train_limit,
    )


def _stage_news(args) -> tuple[Path, int]:
    """Write `<staging>/<dataset-name>/{schema.json,train.jsonl,test.jsonl}`."""
    source_dir = _stage_source(args)
    news_dir = Path(args.staging_root) / args.dataset_name
    news_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "schema.json", news_dir / "schema.json")
    shutil.copyfile(source_dir / "train.jsonl", news_dir / "train.jsonl")

    n_docs = 0
    with (news_dir / "test.jsonl").open("w", encoding="utf-8") as out:
        for row in _read_jsonl(Path(args.news_jsonl)):
            text = str(row.get("text") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            if not text or not doc_id:
                continue
            meta = {
                key: row[key]
                for key in ("date", "stock", "stock_name", "split", "label")
                if row.get(key) not in (None, "")
            }
            staged = {"doc_id": doc_id, "content": text, "split": "test", "meta": meta}
            out.write(json.dumps(staged, ensure_ascii=False) + "\n")
            n_docs += 1
    return news_dir, n_docs


def _run_vllm(args) -> int:
    """Mirror infer_checkpoint_vllm.py's prefilled path over the staged news."""
    from sarge.data.loader import load_documents
    from sarge.data.schema import load_schema
    from sarge.generation.candidate_generator import _backend_prompt_options
    from sarge.generation.prompt import build_getm_prompt_result
    from sarge.models.vllm_backend import VllmGetmBackend
    from sarge.pipeline.infer import run_inference
    from sarge.slot_planning.baseline import TrainPriorPlanner
    from sarge.surface_memory.builder import build_surface_memory
    from sarge.surface_memory.candidate_builder import build_surface_memories

    generation_cfg = {
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
    config = {
        "version": "v1",
        "run": {
            "profile": "eval",
            "dry_run": False,
            "real_run": True,
            "real_run_resource_monitor": {"enabled": False},
        },
        "getm": {
            "backend": "vllm",
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
            },
            "qwen": {
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "model_path": args.merged,
                "compute_dtype": "bfloat16",
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
            "generation": generation_cfg,
        },
    }
    backend = VllmGetmBackend(config=config)

    staging = Path(args.staging_root)
    schema = load_schema(args.dataset_name, data_root=staging)
    documents = load_documents(
        args.dataset_name, "test", data_root=staging, mode="predict", limit=args.limit
    )
    train_docs = load_documents(args.dataset_name, "train", data_root=staging, mode="train")
    planner = TrainPriorPlanner.fit(schema, train_docs)
    slot_plans = {p.doc_id: p for p in planner.predict(documents)}
    memories = {m.doc_id: m for m in build_surface_memories(documents)}

    prompt_options = _backend_prompt_options(backend.generation_metadata)
    prompts_to_run: list[tuple[str, int, str]] = []
    for doc in documents:
        memory = memories.get(doc.doc_id) or build_surface_memory(doc.input)
        result = build_getm_prompt_result(
            dataset=args.dataset_name,
            schema=schema,
            document=doc.input,
            surface_candidates=list(memory.candidates),
            slot_plan=slot_plans.get(doc.doc_id),
            **prompt_options,
        )
        for candidate in range(args.k):
            prompts_to_run.append((doc.doc_id, candidate, result.prompt))

    print(f"[prebatch] running vllm on {len(prompts_to_run)} prompts ...", flush=True)
    t_batch = time.monotonic()
    backend.preload_prompts(prompts_to_run)
    print(f"[prebatch] vllm batch done in {time.monotonic() - t_batch:.0f}s", flush=True)

    t_run = time.monotonic()
    result = run_inference(
        dataset=args.dataset_name,
        split="test",
        data_root=staging,
        out_root=args.out,
        limit=args.limit,
        seed=args.seed,
        k=args.k,
        command_infer=" ".join([sys.executable, *sys.argv]),
        source_commit=args.source_commit,
        backend=backend,
    )
    rows = [json.loads(line) for line in result.prediction_path.read_text().splitlines() if line]
    events = sum(len(r.get("events", [])) for r in rows)
    print(
        f"DONE pred={result.prediction_path} docs={len(rows)} events={events} "
        f"pipeline_time={time.monotonic() - t_run:.0f}s"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--news-jsonl", required=True, type=Path, help="exported news JSONL")
    parser.add_argument("--dataset-name", default="Astock-news", help="staged dataset name")
    parser.add_argument(
        "--schema-dataset", default="DuEE-Fin-dev500",
        help="source benchmark providing the schema + slot-plan train rows",
    )
    parser.add_argument("--sarge-data", type=Path, help="SARGE processed data root")
    parser.add_argument("--staging-root", type=Path, default=Path("runs/sarge_news_staging"))
    parser.add_argument("--slot-train-limit", type=int, default=50)
    parser.add_argument("--merged", help="merged BF16 checkpoint directory")
    parser.add_argument("--out", default="runs/sarge_news")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--dry-run", action="store_true", help="stage only (CPU), no vLLM")
    args = parser.parse_args()

    news_dir, n_docs = _stage_news(args)
    print(f"[stage] {news_dir} docs={n_docs} (schema from {args.schema_dataset})")
    if args.dry_run:
        return 0
    if not args.merged or not Path(args.merged).is_dir():
        parser.error(f"merged ckpt not found: {args.merged}")
    return _run_vllm(args)


if __name__ == "__main__":
    raise SystemExit(main())
