"""Production SFT trainer for SARGE.

Stages the copied processed data, builds GETM SFT samples from the full
train split, trains a Qwen3-4B LoRA adapter, reloads it, and runs
inference on the dev split.

Replace the ad-hoc smoke scripts; called with --dataset and --gpu for
multi-GPU parallel training.

Example (server cwd /data/TJK/DEE/SARGE):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \\
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
        /data/TJK/envs/sarge_vllm_full/bin/python scripts/train_sft.py \\
            --dataset DuEE-Fin-dev500 --epochs 3 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
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


def _sanitize_run_name_part(value: object) -> str:
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "na"


def build_run_name(args: argparse.Namespace) -> str:
    ds_slug = _sanitize_run_name_part(args.dataset)
    parts = [f"sarge_sft_{ds_slug}", f"s{args.seed}", f"ep{args.epochs}", f"gpu{args.gpu}"]
    if args.output_format != "minimal_text":
        parts.append(_sanitize_run_name_part(args.output_format))
    if args.max_train_docs is not None:
        parts.append(f"train{args.max_train_docs}")
    if args.max_train_steps is not None:
        parts.append(f"steps{args.max_train_steps}")
    if args.dev_limit is not None:
        parts.append(f"dev{args.dev_limit}")
    if args.max_new_tokens != 1024:
        parts.append(f"gen{args.max_new_tokens}")
    if args.run_suffix:
        parts.append(_sanitize_run_name_part(args.run_suffix))
    return "_".join(parts)


def build_config(
    *,
    dataset: str,
    model_path: Path,
    epochs: int,
    max_train_docs: int | None,
    seed: int,
    grad_accum: int = 16,
    lr: float = 2e-4,
    quantization: str | None = "4-bit NF4",
    output_format: str = "minimal_text",
    max_train_steps: int | None = None,
    max_new_tokens: int = 1024,
) -> dict:
    use_response_prefix = output_format != "record_plan"
    return {
        "version": "sarge-sft-train-v1",
        "run": {
            "profile": "sarge_train_4090",
            "dry_run": False,
            "real_run": True,
            "real_run_resource_monitor": {"enabled": False},
        },
        "data": {
            "dataset": dataset,
            "train_split": "train",
            "max_train_docs": max_train_docs,
        },
        "getm": {
            "backend": "qwen",
            "output_format": output_format,
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
                "quantization": quantization,
                "double_quantization": quantization is not None,
                "compute_dtype": "bf16",
                "lora": {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "training": {
                    "num_train_epochs": epochs,
                    "learning_rate": lr,
                    "logging_steps": 50,
                    "optimizer": "paged_adamw_8bit",
                    "micro_batch_size": 1,
                    "gradient_accumulation": grad_accum,
                    "max_seq_len": 4096,
                    "gradient_checkpointing": True,
                    "max_train_steps": max_train_steps,
                    "seed": int(seed),
                },
            },
            "generation": {
                "k_candidates": 4,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "temperature": None,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "use_chat_template": True,
                "use_response_prefix": use_response_prefix,
                "response_prefix": '{"events":' if use_response_prefix else "",
                "seed": int(seed),
                "deterministic": True,
                "deterministic_warn_only": True,
                "enable_balanced_json_stopping": True,
                "stop_after_balanced_events_json": True,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-docs", type=int, default=None, help="cap on train docs")
    parser.add_argument("--max-train-steps", type=int, default=None, help="cap optimizer steps for pilot runs")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", help="use BF16 full precision (no 4-bit quantization)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-format", choices=("minimal_text", "argument_object", "record_plan"), default="minimal_text")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gpu", type=int, default=0, help="used for logging only; set via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--run-suffix", default=None, help="extra suffix for repeated pilot runs")
    parser.add_argument("--dev-limit", type=int, default=None, help="cap on dev docs for eval")
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true", help="skip inference + eval after training")
    args = parser.parse_args()

    # Apply seed at process entry. The Trainer-level seed (passed via the
    # config below) only covers training; this covers SFT row construction,
    # surface-memory shuffling, and any module-init RNG before Trainer runs.
    random.seed(args.seed)
    try:
        import numpy as np  # noqa: WPS433 - optional dep, guarded
        np.random.seed(args.seed)
    except Exception:
        pass
    try:
        import torch  # noqa: WPS433
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    ds_slug = _sanitize_run_name_part(args.dataset)
    run_name = build_run_name(args)
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{run_name}] {msg}", flush=True)
    log(f"started at {datetime.now(timezone.utc).isoformat()}")
    log(f"dataset={args.dataset} epochs={args.epochs} seed={args.seed} gpu={args.gpu}")
    log(f"max_train_docs={args.max_train_docs or 'all'}")
    log(f"max_train_steps={args.max_train_steps or 'all'} output_format={args.output_format}")
    log(f"dev_limit={args.dev_limit or 'all'} max_new_tokens={args.max_new_tokens}")
    log(f"run_dir={run_dir}")

    if not args.model_path.is_dir():
        log(f"ERROR: model_path missing: {args.model_path}")
        return 2

    # 1. Stage full data
    staging_root = args.staging_root or Path(tempfile.mkdtemp(prefix=f"sarge_train_{ds_slug}_"))
    log(f"staging_root={staging_root}")
    t0_stage = time.monotonic()
    stage_dataset(
        dataset=args.dataset,
        processed_root=args.processed_root,
        output_root=staging_root,
        splits=("train",),
        limit=args.max_train_docs,
    )
    stage_dataset(
        dataset=args.dataset,
        processed_root=args.processed_root,
        output_root=staging_root,
        splits=("dev",),
        limit=args.dev_limit,
    )
    log(f"staging done in {time.monotonic() - t0_stage:.1f}s")

    # 2. Load train docs + build SFT samples
    schema = load_schema(args.dataset, data_root=staging_root)
    train_docs = load_documents(
        args.dataset, "train", data_root=staging_root, mode="train", limit=args.max_train_docs
    )
    log(f"loaded {len(train_docs)} train docs")

    quant = None if args.bf16 else "4-bit NF4"
    config = build_config(
        dataset=args.dataset,
        model_path=args.model_path,
        epochs=args.epochs,
        max_train_docs=args.max_train_docs,
        seed=args.seed,
        grad_accum=args.grad_accum,
        lr=args.lr,
        quantization=quant,
        output_format=args.output_format,
        max_train_steps=args.max_train_steps,
        max_new_tokens=args.max_new_tokens,
    )
    prompt_options = dict(((config.get("getm") or {}).get("prompt") or {}))
    output_format = str(((config.get("getm") or {}).get("output_format") or "minimal_text"))

    t0_samples = time.monotonic()
    sft_rows: list[dict] = []
    for doc in train_docs:
        memory = build_surface_memory(doc.input)
        sft_rows.append(
            build_getm_sft_sample(
                doc, schema,
                surface_candidates=memory.candidates,
                slot_plan=None,
                output_format=output_format,
                prompt_options=prompt_options,
            )
        )
    audit = audit_sft_targets(sft_rows, schema)
    log(f"built {len(sft_rows)} SFT samples in {time.monotonic() - t0_samples:.1f}s; audit: {json.dumps({k: v for k, v in list(audit.items())[:6]}, ensure_ascii=False)}")

    # 3. Train
    t0_train = time.monotonic()
    backend_manifest = train_sft(config, sft_rows, run_dir)
    train_secs = time.monotonic() - t0_train
    log(f"training done in {train_secs:.1f}s ({train_secs/60:.1f} min)")
    adapter_dir = backend_manifest.get("adapter_dir")
    if not adapter_dir:
        log("ERROR: train_sft did not produce an adapter_dir")
        return 3
    log(f"adapter_dir={adapter_dir}")

    # Discover per-epoch checkpoints saved under trainer_state/.
    epoch_checkpoints: list[str] = []
    trainer_state_dir = run_dir / "artifacts" / "trainer_state"
    if trainer_state_dir.is_dir():
        for child in sorted(trainer_state_dir.iterdir()):
            if child.is_dir() and child.name.startswith("checkpoint-"):
                epoch_checkpoints.append(str(child))
        for cp in epoch_checkpoints:
            log(f"epoch_ckpt={cp}")

    # Save training summary
    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "output_format": args.output_format,
        "max_train_docs": args.max_train_docs,
        "max_train_steps": args.max_train_steps,
        "dev_limit": args.dev_limit,
        "max_new_tokens": args.max_new_tokens,
        "run_suffix": args.run_suffix,
        "seed": args.seed,
        "gpu": args.gpu,
        "train_docs": len(train_docs),
        "sft_rows": len(sft_rows),
        "train_secs": round(train_secs, 1),
        "adapter_dir": str(adapter_dir),
        "epoch_checkpoints": epoch_checkpoints,
        "target_audit": {k: v for k, v in audit.items() if k in ("row_count", "event_count", "target_schema_valid", "empty_target_count", "missing_argument_count")},
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    (run_dir / "summary_train.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.skip_eval:
        log("--skip-eval set; stopping after training.")
        return 0

    # 4. Inference with trained adapter
    log("running dev inference with trained adapter...")
    infer_config = build_config(
        dataset=args.dataset,
        model_path=args.model_path,
        epochs=args.epochs,
        max_train_docs=args.max_train_docs,
        seed=args.seed,
        grad_accum=args.grad_accum,
        lr=args.lr,
        quantization=quant,
        output_format=args.output_format,
        max_train_steps=args.max_train_steps,
        max_new_tokens=args.max_new_tokens,
    )
    infer_config["getm"]["qwen"]["adapter_path"] = str(adapter_dir)
    backend = QwenGetmBackend(config=infer_config)

    t0_infer = time.monotonic()
    infer_result = run_inference(
        dataset=args.dataset,
        split="dev",
        data_root=staging_root,
        out_root=run_dir,
        limit=args.dev_limit,
        seed=args.seed,
        k=infer_config["getm"]["generation"]["k_candidates"],
        backend=backend,
        run_id=f"{run_name}_infer",
    )
    infer_secs = time.monotonic() - t0_infer

    rows = [json.loads(line) for line in infer_result.prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_events = sum(len(row.get("events", [])) for row in rows)
    total_args = sum(
        sum(len(values) for values in event.get("arguments", {}).values())
        for row in rows for event in row.get("events", [])
    )
    log(f"inference done in {infer_secs:.1f}s ({infer_secs/60:.1f} min); {total_events} events, {total_args} args across {len(rows)} docs")

    infer_summary = {
        **summary,
        "infer_secs": round(infer_secs, 1),
        "pred_docs": len(rows),
        "pred_events": total_events,
        "pred_args": total_args,
        "prediction_path": str(infer_result.prediction_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(infer_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"summary: {json.dumps(infer_summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
