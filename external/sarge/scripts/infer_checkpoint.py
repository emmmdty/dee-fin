"""Run inference on a specific LoRA checkpoint, for epoch comparison.

Also supports --no-adapter for a pure base Qwen3-4B baseline (no SFT).

Decoding modes:
  (default)  greedy, k=1  — deterministic, fast, direct output
  --sample    sampling, k=4 — diverse candidates, MRS selection
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen" / "Qwen3-4B-Instruct-2507"

from sarge.data.staging import stage_dataset  # noqa: E402
from sarge.models.qwen_backend import QwenGetmBackend  # noqa: E402
from sarge.pipeline.infer import run_inference  # noqa: E402


def _build_generation_config(args) -> dict[str, Any]:
    """Return the 'generation' sub-dict for the GETM config.

    Two modes, selectable via --sample:
      * greedy (default): do_sample=False, k=1, seed+deterministic
      * sampling:         do_sample=True, k defaults to 4, temperature/top_p active
    """
    use_response_prefix = args.output_format != "record_plan"
    if args.sample:
        gen: dict[str, Any] = {
            "k_candidates": args.k,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "use_chat_template": True,
            "use_response_prefix": use_response_prefix,
            "response_prefix": '{"events":' if use_response_prefix else "",
            "seed": args.seed,
            "enable_balanced_json_stopping": True,
            "stop_after_balanced_events_json": True,
        }
    else:
        gen = {
            "k_candidates": args.k,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "use_chat_template": True,
            "use_response_prefix": use_response_prefix,
            "response_prefix": '{"events":' if use_response_prefix else "",
            "seed": args.seed,
            "deterministic": True,
            "deterministic_warn_only": True,
            "enable_balanced_json_stopping": True,
            "stop_after_balanced_events_json": True,
        }
    return gen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None, help="LoRA checkpoint path; not required with --no-adapter")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processed", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-format", choices=("minimal_text", "argument_object", "record_plan"), default="minimal_text")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--slot-train-limit", type=int, default=50)
    parser.add_argument(
        "--source-commit",
        default=None,
        help="source git commit to record when running from a non-git server copy",
    )
    parser.add_argument("--no-adapter", action="store_true")

    # Decoding strategy
    parser.add_argument("--sample", action="store_true",
                        help="Enable sampling (do_sample=True).  Default is greedy (k=1).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (only with --sample, default 0.7)")
    parser.add_argument("--k", type=int, default=None,
                        help="Candidates per doc (default: 1 greedy, 4 sampling)")

    args = parser.parse_args()

    # Set k default based on mode
    if args.k is None:
        args.k = 4 if args.sample else 1

    if not args.no_adapter and args.ckpt is None:
        parser.error("--ckpt is required unless --no-adapter is set")
    if not args.no_adapter and not Path(args.ckpt).is_dir():
        parser.error(f"adapter checkpoint not found: {args.ckpt}")
    if not Path(args.model).is_dir():
        parser.error(f"model path not found: {args.model}")

    staging = Path(tempfile.mkdtemp(prefix="sarge_infer_"))
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

    qwen_cfg = {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "model_path": args.model, "quantization": "4-bit NF4", "double_quantization": True, "compute_dtype": "bf16"}
    if not args.no_adapter:
        qwen_cfg["adapter_path"] = args.ckpt

    generation_cfg = _build_generation_config(args)
    mode_label = "sampling" if generation_cfg["do_sample"] else "greedy"
    mode_detail = (f"k={generation_cfg['k_candidates']} T={generation_cfg['temperature']}"
                   if generation_cfg["do_sample"]
                   else f"k={generation_cfg['k_candidates']} deterministic")
    print(f"[mode] {mode_label}  {mode_detail}", flush=True)

    config = {
        "version": "v1", "run": {"profile": "eval", "dry_run": False, "real_run": True, "real_run_resource_monitor": {"enabled": False}},
        "getm": {
            "backend": "qwen", "output_format": args.output_format,
            "prompt": {"max_surface_candidates": 20, "candidate_context_chars": 0, "candidate_render_mode": "compact", "enable_candidate_filtering": True, "max_candidates_per_type": 6, "dedupe_surface_candidates": True, "drop_low_value_company_fragments": True, "prompt_token_budget": 4096, "fail_on_prompt_token_limit": False},
            "qwen": qwen_cfg,
            "generation": generation_cfg,
        },
    }

    backend = QwenGetmBackend(config=config)
    t0 = time.monotonic()
    result = run_inference(
        dataset=args.dataset,
        split=args.split,
        data_root=staging,
        out_root=args.out,
        limit=args.limit,
        seed=args.seed,
        k=generation_cfg["k_candidates"],
        command_infer=" ".join([sys.executable, *sys.argv]),
        source_commit=args.source_commit,
        backend=backend,
    )
    elapsed = time.monotonic() - t0
    rows = [json.loads(line) for line in result.prediction_path.read_text().splitlines() if line.strip()]
    events = sum(len(r.get("events", [])) for r in rows)
    print(f"DONE pred={result.prediction_path} docs={len(rows)} events={events} time={elapsed:.0f}s")


if __name__ == "__main__":
    main()
