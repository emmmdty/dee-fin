#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL_PATHS = {
    "chinese-roberta-wwm-ext": "/data/TJK/DEE/dee-fin/models/chinese-roberta-wwm-ext_safetensors",
    "lawformer": "/data/TJK/DEE/dee-fin/models/thunlp_Lawformer_safetensors",
    "longformer-chinese": "/data/TJK/DEE/dee-fin/models/schen_longformer-chinese-base-4096_safetensors_custom",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure P1 encoder forward-pass memory.")
    parser.add_argument("--models", default="chinese-roberta-wwm-ext,lawformer,longformer-chinese")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        model_path = MODEL_PATHS.get(name, name)
        started = time.time()
        status: dict[str, Any] = {
            "name": name,
            "model_path": model_path,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "device": str(device),
            "status": "pending",
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True).to(device)
            model.eval()
            encoded = tokenizer(
                ["显存测量输入"] * args.batch_size,
                padding="max_length",
                truncation=True,
                max_length=args.sequence_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = model(**encoded)
            if torch.cuda.is_available():
                status["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                status["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            status["status"] = "ok"
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = repr(exc)
        finally:
            status["elapsed_seconds"] = round(time.time() - started, 3)
            results.append(status)

    payload = {
        "phase": "P1",
        "purpose": "encoder_forward_memory_measurement",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.executable,
        "git_commit": _git_commit(),
        "offline_env": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        _write_markdown(Path(args.markdown_out), payload)
    return 0 if all(result["status"] == "ok" for result in results) else 1


def _git_commit() -> str:
    process = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        "| Model | Status | Seq Len | Batch | Peak Allocated GB | Peak Reserved GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        allocated = result.get("peak_allocated_bytes")
        reserved = result.get("peak_reserved_bytes")
        rows.append(
            "| {name} | {status} | {seq} | {batch} | {alloc} | {reserved} |".format(
                name=result["name"],
                status=result["status"],
                seq=result["sequence_length"],
                batch=result["batch_size"],
                alloc=f"{allocated / 1024**3:.3f}" if allocated is not None else "NA",
                reserved=f"{reserved / 1024**3:.3f}" if reserved is not None else "NA",
            )
        )
    text = "\n".join(
        [
            "# P1 Memory Measurement",
            "",
            "> Status: measured evidence from P1. Do not edit manually without rerunning the command.",
            "",
            f"- Host: `{payload['host']}`",
            f"- Git commit: `{payload['git_commit']}`",
            f"- Python: `{payload['python']}`",
            "",
            *rows,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
