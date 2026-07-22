"""Server-side smoke test for real Qwen3-4B inference through SARGE.

Stages a tiny slice of DuEE-Fin-dev500 dev into a tempdir, loads Qwen3-4B
base weights from the project model registry, builds a QwenGetmBackend in
real-run mode (no LoRA adapter), and runs the full SARGE inference pipeline
end-to-end. Verifies on the server that:

* Qwen3-4B loads under bitsandbytes 4-bit NF4 quantization on a single 4090
* The schema-aware prompt + role-safe contract generate parseable output
* The full pipeline (CSG → LESP → GETM → MRS → postprocess → export)
  terminates cleanly with canonical-schema-conformant predictions

No SFT adapter is loaded. The performance number will be far below the
previous baseline (which used a trained LoRA adapter); the goal is *only*
to verify infrastructure works before committing GPU hours on full training.

Example (run on server cwd /data/TJK/DEE/SARGE):
    PYTHONPATH=src /data/TJK/envs/sarge_vllm_full/bin/python scripts/smoke_server_qwen.py --limit 5
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

from sarge.data.staging import stage_dataset  # noqa: E402
from sarge.models.qwen_backend import QwenGetmBackend  # noqa: E402
from sarge.pipeline.infer import run_inference  # noqa: E402

DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen" / "Qwen3-4B-Instruct-2507"


def build_qwen_config(model_path: Path, dataset: str) -> dict:
    """Minimal config to drive QwenGetmBackend in real-run inference mode."""
    return {
        "version": "sarge-qwen-smoke-0.1",
        "run": {
            "profile": "smoke_server_4090",
            "dry_run": False,
            "real_run": True,
            "real_run_resource_monitor": {
                "enabled": False,
            },
        },
        "data": {
            "dataset": dataset,
            "predict_split": "dev",
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--limit", type=int, default=5, help="document count cap; default 5 for smoke")
    parser.add_argument("--train-limit", type=int, default=50, help="train cap for slot-plan prior fit")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--staging-root", type=Path, default=None)
    args = parser.parse_args()

    if not args.model_path.is_dir():
        print(f"[qwen-smoke] ERROR: model path does not exist: {args.model_path}", file=sys.stderr)
        return 2

    staging_root = args.staging_root or Path(tempfile.mkdtemp(prefix="sarge_qwen_smoke_"))
    print(f"[qwen-smoke] dataset={args.dataset} split={args.split} limit={args.limit}")
    print(f"[qwen-smoke] processed_root={args.processed_root}")
    print(f"[qwen-smoke] model_path={args.model_path}")
    print(f"[qwen-smoke] staging_root={staging_root}")

    print("[qwen-smoke] staging data...")
    t0 = time.monotonic()
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
        splits=(args.split,),
        limit=args.limit,
    )
    print(f"[qwen-smoke] staging done in {time.monotonic() - t0:.1f}s")

    print("[qwen-smoke] building QwenGetmBackend (real_run=True, base model only)...")
    config = build_qwen_config(args.model_path, args.dataset)
    backend = QwenGetmBackend(config=config)

    print("[qwen-smoke] running inference...")
    t0 = time.monotonic()
    result = run_inference(
        dataset=args.dataset,
        split=args.split,
        data_root=staging_root,
        out_root=args.out_root,
        limit=args.limit,
        seed=args.seed,
        k=args.k,
        backend=backend,
    )
    elapsed = time.monotonic() - t0
    print(f"[qwen-smoke] inference done in {elapsed:.1f}s ({elapsed / max(args.limit, 1):.1f}s/doc)")

    pred_path = result.prediction_path
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[qwen-smoke] run_id={result.run_id}")
    print(f"[qwen-smoke] prediction={pred_path}")
    print(f"[qwen-smoke] prediction_rows={len(rows)}")
    for row in rows:
        event_summary = ", ".join(
            f"{e['event_type']}(n_roles={len(e['arguments'])})" for e in row.get("events", [])
        ) or "<no events>"
        print(f"[qwen-smoke]   doc {row['doc_id'][:12]}...: {event_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
