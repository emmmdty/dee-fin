"""Local smoke test for the SARGE inference pipeline.

Stages a tiny slice of the copied processed data into SARGE canonical layout,
then runs ``sarge.pipeline.infer.run_inference`` end-to-end with the mock
GETM backend. No GPU, no Qwen weights, no network. Intended to verify the
pipeline mechanics (CSG → LESP → GETM → MRS → postprocess → export) after
refactoring before committing GPU time on the full corpus.

Example:
    python scripts/smoke_local.py --limit 3
    python scripts/smoke_local.py --dataset DuEE-Fin-dev500 --split dev --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sarge.data.staging import stage_dataset  # noqa: E402
from sarge.pipeline.infer import run_inference  # noqa: E402

DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--limit", type=int, default=3, help="document count cap; default 3 for smoke")
    parser.add_argument("--train-limit", type=int, default=50, help="train cap for slot-plan prior fit")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--k", type=int, default=4, help="candidate generation k")
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="staging dir to write SARGE-canonical jsonl; default: tempdir",
    )
    args = parser.parse_args()

    staging_root = args.staging_root or Path(tempfile.mkdtemp(prefix="sarge_smoke_"))
    print(f"[smoke] dataset={args.dataset} split={args.split} limit={args.limit}")
    print(f"[smoke] processed_root={args.processed_root}")
    print(f"[smoke] staging_root={staging_root}")

    # Stage train (for slot-plan prior) + the chosen split with separate limits.
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

    result = run_inference(
        dataset=args.dataset,
        split=args.split,
        data_root=staging_root,
        out_root=args.out_root,
        limit=args.limit,
        seed=args.seed,
        k=args.k,
    )

    pred_path = result.prediction_path
    docs_written = sum(1 for _ in pred_path.open(encoding="utf-8") if _.strip())
    summary = json.loads((result.run_root / "diagnostics" / "pipeline_summary.json").read_text(encoding="utf-8"))

    print(f"[smoke] run_id={result.run_id}")
    print(f"[smoke] run_root={result.run_root}")
    print(f"[smoke] prediction={pred_path}")
    print(f"[smoke] prediction_lines={docs_written}")
    print(f"[smoke] handoff_script_exists={result.handoff_script_exists}")
    print(f"[smoke] handoff_command={result.handoff_command}")
    print(f"[smoke] pipeline_summary={summary['final_prediction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
