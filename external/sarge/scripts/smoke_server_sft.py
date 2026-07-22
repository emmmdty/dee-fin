"""Server-side smoke test for the full SARGE SFT-then-infer cycle.

Verifies the training half of the pipeline before committing GPU hours to
full-corpus training:

1. Stage 16 train docs + 5 dev docs from DuEE-Fin-dev500.
2. Build GETM SFT samples from the 16 train docs.
3. Train a Qwen3-4B LoRA adapter for 1 epoch (~5-10 min on a single 4090).
4. Reload the adapter, run inference on the 5 dev docs, and dump canonical
   predictions.
5. Print per-doc event summaries so we can sanity-check that SFT produced
   *more* structured output than the base-model smoke
   (``scripts/smoke_server_qwen.py``) reported.

If this smoke succeeds, the full W3 training run (full ChFinAnn or
DuEE-Fin train, multi-hour) becomes a same-script-different-config affair.

Example (run on server cwd /data/TJK/DEE/SARGE):
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \\
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
        /data/TJK/envs/sarge_vllm_full/bin/python scripts/smoke_server_sft.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sarge.data.loader import load_documents  # noqa: E402
from sarge.data.schema import load_schema  # noqa: E402
from sarge.data.staging import stage_dataset  # noqa: E402
from sarge.models.qwen_backend import QwenGetmBackend, train_sft  # noqa: E402
from sarge.models.sft_dataset import audit_sft_targets, build_getm_sft_sample  # noqa: E402
from sarge.pipeline.infer import run_inference  # noqa: E402
from sarge.surface_memory.builder import build_surface_memory  # noqa: E402

DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen" / "Qwen3-4B-Instruct-2507"


def build_sft_config(model_path: Path, dataset: str, *, max_train_steps: int) -> dict:
    """Smoke-scale SFT config: 16 train docs, 1 epoch, real run."""
    return {
        "version": "sarge-sft-smoke-0.1",
        "run": {
            "profile": "smoke_server_4090_sft",
            "dry_run": False,
            "real_run": True,
            "real_run_resource_monitor": {"enabled": False},
        },
        "data": {
            "dataset": dataset,
            "train_split": "train",
            "max_train_docs": 16,
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
            },
            "qwen": {
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "model_path": str(model_path),
                "adapter_path": None,
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
                },
            },
            "generation": {
                "k_candidates": 4,
                "max_new_tokens": 1024,
                "do_sample": False,
                "temperature": None,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "use_chat_template": True,
                "use_response_prefix": True,
                "response_prefix": '{"events":',
                "enable_balanced_json_stopping": True,
                "stop_after_balanced_events_json": True,
            },
        },
        "training_budget": {"max_train_steps": max_train_steps},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--dev-limit", type=int, default=5)
    parser.add_argument("--max-train-steps", type=int, default=8,
                        help="upper-bound trainer steps; smoke default 8")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--staging-root", type=Path, default=None)
    args = parser.parse_args()

    if not args.model_path.is_dir():
        print(f"[sft-smoke] ERROR: model path missing: {args.model_path}", file=sys.stderr)
        return 2

    staging_root = args.staging_root or Path(tempfile.mkdtemp(prefix="sarge_sft_smoke_"))
    print(f"[sft-smoke] dataset={args.dataset}")
    print(f"[sft-smoke] train_limit={args.train_limit} dev_limit={args.dev_limit}")
    print(f"[sft-smoke] max_train_steps={args.max_train_steps}")
    print(f"[sft-smoke] staging_root={staging_root}")

    # 1. Stage data
    stage_dataset(
        dataset=args.dataset,
        processed_root=args.processed_root,
        output_root=staging_root,
        splits=("train",),
        limit=args.train_limit,
    )
    stage_dataset(
        dataset=args.dataset,
        processed_root=args.processed_root,
        output_root=staging_root,
        splits=("dev",),
        limit=args.dev_limit,
    )

    schema = load_schema(args.dataset, data_root=staging_root)
    train_documents = load_documents(
        args.dataset, "train", data_root=staging_root, mode="train", limit=args.train_limit
    )
    print(f"[sft-smoke] loaded {len(train_documents)} train docs")

    # 2. Build SFT samples
    config = build_sft_config(args.model_path, args.dataset, max_train_steps=args.max_train_steps)
    prompt_options = dict(((config.get("getm") or {}).get("prompt") or {}))
    output_format = str(((config.get("getm") or {}).get("output_format") or "minimal_text"))
    sft_rows: list[dict] = []
    for doc in train_documents:
        memory = build_surface_memory(doc.input)
        sft_rows.append(
            build_getm_sft_sample(
                doc,
                schema,
                surface_candidates=memory.candidates,
                slot_plan=None,
                output_format=output_format,
                prompt_options=prompt_options,
            )
        )
    audit = audit_sft_targets(sft_rows, schema)
    print(f"[sft-smoke] built {len(sft_rows)} SFT samples; audit summary keys: {list(audit.keys())[:5]}")

    # 3. Train SFT
    train_out_dir = args.out_root / f"sft_smoke_{int(time.time())}"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sft-smoke] training out_dir={train_out_dir}")
    t0 = time.monotonic()
    backend_manifest = train_sft(config, sft_rows, train_out_dir)
    train_secs = time.monotonic() - t0
    print(f"[sft-smoke] training done in {train_secs:.1f}s")
    print(f"[sft-smoke] backend_manifest={backend_manifest['backend_manifest_path']}")
    adapter_dir = backend_manifest.get("adapter_dir")
    if not adapter_dir:
        print("[sft-smoke] ERROR: train_sft did not produce an adapter_dir", file=sys.stderr)
        return 3
    print(f"[sft-smoke] adapter_dir={adapter_dir}")

    # 4. Reload + inference with adapter
    infer_config = build_sft_config(args.model_path, args.dataset, max_train_steps=args.max_train_steps)
    infer_config["getm"]["qwen"]["adapter_path"] = str(adapter_dir)
    backend = QwenGetmBackend(config=infer_config)

    print("[sft-smoke] running inference with trained adapter...")
    t0 = time.monotonic()
    result = run_inference(
        dataset=args.dataset,
        split="dev",
        data_root=staging_root,
        out_root=args.out_root,
        limit=args.dev_limit,
        seed=args.seed,
        k=infer_config["getm"]["generation"]["k_candidates"],
        backend=backend,
    )
    infer_secs = time.monotonic() - t0
    print(f"[sft-smoke] inference done in {infer_secs:.1f}s ({infer_secs / max(args.dev_limit, 1):.1f}s/doc)")

    rows = [json.loads(line) for line in result.prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[sft-smoke] run_id={result.run_id}")
    print(f"[sft-smoke] prediction_rows={len(rows)}")
    total_events = 0
    total_roles = 0
    for row in rows:
        events = row.get("events", [])
        total_events += len(events)
        per_event = []
        for event in events:
            args_n = sum(len(values) for values in event.get("arguments", {}).values())
            total_roles += args_n
            per_event.append(f"{event['event_type']}(n_args={args_n})")
        summary = ", ".join(per_event) or "<no events>"
        print(f"[sft-smoke]   doc {row['doc_id'][:12]}...: {summary}")
    print(f"[sft-smoke] total: {total_events} events, {total_roles} argument values across {len(rows)} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
