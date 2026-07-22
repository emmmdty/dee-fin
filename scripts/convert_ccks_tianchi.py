#!/usr/bin/env python
"""Convert the CCKS-2021 Tianchi raw format to the causal_pairs jsonl expected by
load_ccks_causal().

Tianchi fields:
  reason_type / reason_product / reason_region / reason_industry  ->  cause
  result_type  / result_product  / result_region  / result_industry ->  effect

Span offsets are not provided in the raw export; cause_span / effect_span are
left empty (the loader's _event_node() handles missing spans via the
`(span + [0, 0])[:2]` fallback).

Usage:
    python scripts/convert_ccks_tianchi.py \\
        --train  ccks_task2_train.txt \\
        --eval   ccks_task2_eval_data.txt \\
        --outdir data/processed/ccks_fin_causal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _convert_record(rec: dict) -> dict:
    pairs = []
    for r in rec.get("result", []):
        pairs.append(
            {
                "cause": {
                    "type": r.get("reason_type", ""),
                    "region": r.get("reason_region", ""),
                    "product": r.get("reason_product", ""),
                    "industry": r.get("reason_industry", ""),
                },
                "effect": {
                    "type": r.get("result_type", ""),
                    "region": r.get("result_region", ""),
                    "product": r.get("result_product", ""),
                    "industry": r.get("result_industry", ""),
                },
                "cause_span": [],
                "effect_span": [],
            }
        )
    return {
        "text_id": str(rec.get("text_id", "")),
        "text": rec.get("text", ""),
        "causal_pairs": pairs,
    }


def convert(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst.open("w", encoding="utf-8") as out:
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.write(json.dumps(_convert_record(rec), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="ccks_task2_train.txt")
    parser.add_argument("--eval", required=True, type=Path, help="ccks_task2_eval_data.txt")
    parser.add_argument("--outdir", required=True, type=Path, help="output directory")
    args = parser.parse_args()

    n_train = convert(args.train, args.outdir / "train.jsonl")
    print(f"train: {n_train} records -> {args.outdir}/train.jsonl")
    n_eval = convert(args.eval, args.outdir / "eval.jsonl")
    print(f"eval:  {n_eval} records -> {args.outdir}/eval.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
