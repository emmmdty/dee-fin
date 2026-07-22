#!/usr/bin/env python
"""M3b controlled cross-stage sweep on CGEP: composed coverage vs reachability loss.

Sweeps a controlled reachability loss over real reasoning ranks and reports each
CS-CRP variant's composed coverage, so the collapse of `naive` conformal and the
robustness of the conditional recycler (`cs_crp_cond`) are visible on one table.
Ranks come either from a pre-dumped SeDGPL run (`--ranks-file`, the real reasoner)
or from the torch-free `frequency`/`random` baseline computed on CPU.

Why the loss is controlled: the MAVEN causal+subevent extractor recovers ~0.4% /
0% of gold edges, so a real constructed ECG is degenerate (reachability loss
~1.0); the loss is therefore a design variable (see succession/cross_stage.py).

    uv run python scripts/evaluate_cgep_cross_stage.py --predictor frequency
    uv run python scripts/evaluate_cgep_cross_stage.py --ranks-file runs/cgep/sedgpl_ranks.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from finekg.succession.cross_stage import DEFAULT_LOSSES, cross_stage_sweep

_MAVEN = Path("data/processed/maven_ere")


def _ranks_from_predictor(
    name: str, cal_ratio: float, seed: int
) -> tuple[list[float], list[float]]:
    """Fit a torch-free baseline on train and read gold ranks on a valid cal/test split."""
    from finekg.succession.data.cgep import build_cgep, iter_documents
    from finekg.succession.predictor import successor_predictors
    from finekg.succession.selective import cgep_gold_ranks

    train, _ = build_cgep(iter_documents([str(_MAVEN / "train.jsonl")]))
    valid, _ = build_cgep(iter_documents([str(_MAVEN / "valid.jsonl")]))
    order = list(valid)
    random.Random(seed).shuffle(order)
    cut = int(len(order) * cal_ratio)
    predictor = successor_predictors.create(name)
    predictor.fit(train)
    cal_ranks, _ = cgep_gold_ranks(predictor, order[:cut])
    test_ranks, _ = cgep_gold_ranks(predictor, order[cut:])
    return cal_ranks, test_ranks


def _load_ranks(path: Path) -> tuple[list[float], list[float], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    to_f = lambda xs: [math.inf if r is None else float(r) for r in xs]  # noqa: E731
    return to_f(data["cal_ranks"]), to_f(data["test_ranks"]), data.get("predictor", path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks-file", type=Path,
                        help="JSON {cal_ranks,test_ranks} from a SeDGPL run")
    parser.add_argument("--predictor", default="frequency", choices=("random", "frequency"),
                        help="compute ranks on CPU when no --ranks-file is given")
    parser.add_argument("--cal-ratio", type=float, default=0.5)
    parser.add_argument("--alpha-total", type=float, default=0.2)
    parser.add_argument("--edge-share", type=float, default=0.5)
    parser.add_argument("--losses", type=float, nargs="+", default=list(DEFAULT_LOSSES))
    parser.add_argument("--modes", nargs="+", default=["random", "hardest"],
                        choices=("random", "hardest"))
    parser.add_argument("--seed", type=int, default=209)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.ranks_file:
        cal_ranks, test_ranks, source = _load_ranks(args.ranks_file)
    else:
        cal_ranks, test_ranks = _ranks_from_predictor(args.predictor, args.cal_ratio, args.seed)
        source = args.predictor

    reports: dict[str, dict] = {}
    for mode in args.modes:
        out = cross_stage_sweep(
            cal_ranks, test_ranks, losses=args.losses, alpha_total=args.alpha_total,
            edge_share=args.edge_share, mode=mode, seed=args.seed,
        )
        reports[mode] = out
        print(f"\n{source} / mode={mode} / alpha_total={args.alpha_total} "
              f"(target {1 - args.alpha_total:.2f})  "
              f"n_cal={len(cal_ranks)} n_test={len(test_ranks)}\n")
        print(f"{'loss':>6}{'reach':>7}{'naive':>9}{'bonf':>9}{'cs_crp':>9}{'cs_cond':>9}")
        for row in out["curve"]:
            print(f"{row['loss']:>6.2f}{row['reachable_rate']:>7.2f}"
                  f"{row['naive_coverage']:>9.3f}{row['bonferroni_coverage']:>9.3f}"
                  f"{row['cs_crp_coverage']:>9.3f}{row['cs_crp_cond_coverage']:>9.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"source": source, "alpha_total": args.alpha_total, "reports": reports},
                       indent=2),
            encoding="utf-8",
        )
        print(f"\n[cross-stage] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
