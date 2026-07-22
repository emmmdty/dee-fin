"""Convert SARGE canonical predictions to evaluator format, filter gold to the
predicted doc_id subset, then run all three evaluator tracks. Designed for
smoke / baseline runs with --limit subsets where gold has more docs than pred.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TRACKS = ("legacy-doc2edag", "unified-strict", "docfee-official")
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sarge.evaluation.evaluator_adapter import convert_sarge_predictions_to_evaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path,
                        help="e.g. /data/TJK/DEE/SARGE/runs/sarge_infer_...")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--processed-root", type=Path,
                        default=REPO_ROOT / "data")
    parser.add_argument("--project-root", type=Path,
                        default=REPO_ROOT,
                        help="SARGE project root containing the copied evaluator/ package")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    pred_path = args.run_root / "predictions" / args.dataset / f"{args.split}.canonical.pred.jsonl"
    if not pred_path.exists():
        print(f"ERROR: pred missing: {pred_path}", file=sys.stderr)
        return 2

    eval_root = args.run_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    pred_eval_path = args.run_root / "predictions" / args.dataset / f"{args.split}.evaluator.pred.jsonl"
    gold_path = args.processed_root / args.dataset / f"{args.split}.jsonl"
    if not gold_path.exists():
        gold_path = args.processed_root / args.dataset / f"{args.split}.json"
    gold_filtered_path = args.run_root / "predictions" / args.dataset / f"{args.split}.gold.filtered.jsonl"
    schema_path = args.processed_root / args.dataset / "schema.json"

    # Step A: convert pred to evaluator format
    # Adapter conversion is implemented in sarge.evaluation.evaluator_adapter.
    report = convert_sarge_predictions_to_evaluator(pred_path, pred_eval_path)
    print(f"[convert] {json.dumps(report.to_dict(), ensure_ascii=False)}")

    # Step B: filter gold to pred doc_ids
    pred_ids = set()
    with pred_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id") or doc.get("document_id")
            if doc_id:
                pred_ids.add(str(doc_id))

    # Step B: filter gold to pred doc_ids
    gold_is_jsonl = gold_path.suffix == ".jsonl"
    kept = 0
    if gold_is_jsonl:
        with gold_path.open(encoding="utf-8") as gin, gold_filtered_path.open("w", encoding="utf-8") as gout:
            for line in gin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = row.get("doc_id") or row.get("document_id") or row.get("id")
                if str(doc_id) in pred_ids:
                    gout.write(line + "\n")
                    kept += 1
        eval_gold_path = gold_filtered_path
    else:
        # JSON array format (ChFinAnn: [[doc_id, payload], ...])
        gold_array = json.loads(gold_path.read_text(encoding="utf-8"))
        filtered = [item for item in gold_array if str(item[0]) in pred_ids]
        kept = len(filtered)
        # Write as JSONL: one [doc_id, payload] pair per line
        with gold_filtered_path.open("w", encoding="utf-8") as gout:
            for item in filtered:
                gout.write(json.dumps(item, ensure_ascii=False) + "\n")
        eval_gold_path = gold_filtered_path
    print(f"[filter_gold] kept={kept} pred_doc_count={len(pred_ids)}")
    if kept != len(pred_ids):
        print(f"[filter_gold] WARNING: gold subset ({kept}) != pred docs ({len(pred_ids)})")

    # Step C: run 3 tracks
    results: dict[str, dict] = {}
    for track in TRACKS:
        out_path = eval_root / f"eval_{track.replace('-', '_')}.json"
        cmd = [
            args.python, "-B", "-m", "evaluator", track,
            "--dataset", args.dataset,
            "--gold", str(eval_gold_path),
            "--schema", str(schema_path),
            "--pred", str(pred_eval_path),
            "--out", str(out_path),
        ]
        if track == "legacy-doc2edag" and not gold_is_jsonl:
            cmd.extend(["--input-format", "canonical-jsonl"])
        print(f"[{track}] running ...")
        proc = subprocess.run(cmd, cwd=str(args.project_root), capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[{track}] FAILED rc={proc.returncode}")
            print(proc.stdout)
            print(proc.stderr)
            continue
        if out_path.exists():
            with out_path.open(encoding="utf-8") as handle:
                results[track] = json.load(handle)
        else:
            print(f"[{track}] no output JSON at {out_path}")

    # Step D: print summary
    print("\n========== SUMMARY ==========")
    print(f"pred_path={pred_path}")
    if gold_is_jsonl:
        print(f"gold_filtered={gold_filtered_path} (kept {kept} docs)")
    else:
        print(f"gold={gold_path} (JSON array, used as-is)")
    print()
    header = ('track', 'F1', 'P', 'R', 'multi_F1', 'single_F1')
    print('{:20s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}'.format(*header))
    print('-' * 78)
    for track, data in results.items():
        overall = data.get('overall') or {}
        multi = (data.get('subset_metrics') or {}).get('multi_event') or {}
        single = (data.get('subset_metrics') or {}).get('single_event') or {}
        print('{:20s}  {:8.4f}  {:8.4f}  {:8.4f}  {:10.4f}  {:10.4f}'.format(
            track,
            overall.get('f1', 0),
            overall.get('precision', 0),
            overall.get('recall', 0),
            multi.get('f1', 0),
            single.get('f1', 0),
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
