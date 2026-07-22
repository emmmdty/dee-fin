#!/usr/bin/env python
"""Selective conformal head over CGEP (M3a): calibrated prediction sets + curve.

The reasoning-stage selective predictor of CS-CRP (SPEC 4.3). Fits a predictor on
train, calibrates a conformal rank threshold on a held-out slice of valid, and
reports the risk-coverage curve on the rest: coverage >= 1 - alpha (the formal
guarantee, independent of MRR) traded against prediction-set size. The head is
predictor-agnostic, so `frequency` traces the whole curve on CPU exactly as
`sedgpl` does on GPU.

    uv run python scripts/evaluate_cgep_selective.py --dataset maven --predictor frequency
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate_cgep_selective.py \
      --dataset maven --predictor sedgpl --model-path <roberta-base> --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from finekg.succession.data.cgep import CgepInstance, build_cgep, iter_documents
from finekg.succession.linearize import EDGE_BUDGET
from finekg.succession.predictor import successor_predictors
from finekg.succession.selective import DEFAULT_ALPHAS, cgep_gold_ranks, selective_report


def _dump_ranks(path: Path, predictor: str, cal: list[float], test: list[float]) -> None:
    """Persist gold ranks (inf -> null) for the M3b cross-stage sweep."""
    def jsonable(xs: list[float]) -> list[float | None]:
        return [None if r == math.inf else r for r in xs]

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictor": predictor, "cal_ranks": jsonable(cal), "test_ranks": jsonable(test)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[cgep-selective] wrote gold ranks -> {path}")

_MAVEN = Path("data/processed/maven_ere")


def _split(
    instances: list[CgepInstance], cal_ratio: float, seed: int
) -> tuple[list[CgepInstance], list[CgepInstance]]:
    """Held-out (calibration, test) split of valid; conformal needs them disjoint."""
    order = list(instances)
    random.Random(seed).shuffle(order)
    cut = int(len(order) * cal_ratio)
    return order[:cut], order[cut:]


def _build(args, train: list[CgepInstance], valid: list[CgepInstance]):
    """`sedgpl` registers on import; the baselines take no arguments.

    The `<a_i>` vocabulary is transductive, so it must span train *and* every
    instance scored later (all of valid), exactly as in `evaluate_cgep`.
    """
    if args.predictor != "sedgpl":
        return successor_predictors.create(args.predictor)
    if not args.model_path:
        raise SystemExit("--predictor sedgpl needs --model-path")
    import finekg.succession.sedgpl  # noqa: F401 - registers "sedgpl"
    from finekg.succession.linearize import EventVocabulary

    return successor_predictors.create(
        "sedgpl", model_path=args.model_path,
        vocabulary=EventVocabulary.build([*train, *valid]),
        epochs=args.epochs, sample_rate=args.sample_rate, device=args.device, lr=args.lr,
        edge_selector=args.edge_selector, max_edges=args.max_edges,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="maven", choices=("maven",))
    parser.add_argument("--predictor", default="frequency",
                        choices=("random", "frequency", "sedgpl"))
    parser.add_argument("--cal-ratio", type=float, default=0.5,
                        help="held-out valid share used to calibrate")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--reasoning", nargs="+", default=["split", "aci"],
                        choices=("split", "aci", "weighted"),
                        help="one or more conformal calibrators, each reported from one fit")
    parser.add_argument("--strict", action="store_true", help="pessimistic tie-break ranks")
    parser.add_argument("--seed", type=int, default=209)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-ranks", type=Path,
                        help="dump cal/test gold ranks for the M3b cross-stage sweep")
    # sedgpl only
    parser.add_argument("--model-path", help="roberta-base checkpoint, for sedgpl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample-rate", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--edge-selector", default="sedgpl", choices=("sedgpl", "distance"))
    parser.add_argument("--max-edges", type=int, default=EDGE_BUDGET)
    args = parser.parse_args()

    train, _ = build_cgep(iter_documents([str(_MAVEN / "train.jsonl")]))
    valid, _ = build_cgep(iter_documents([str(_MAVEN / "valid.jsonl")]))
    cal, test = _split(valid, args.cal_ratio, args.seed)

    predictor = _build(args, train, valid)
    predictor.fit(train)

    if args.dump_ranks:
        cal_ranks, _ = cgep_gold_ranks(predictor, cal, strict=args.strict)
        test_ranks, _ = cgep_gold_ranks(predictor, test, strict=args.strict)
        _dump_ranks(args.dump_ranks, args.predictor, cal_ranks, test_ranks)

    tag = " / strict" if args.strict else ""
    reports: dict[str, dict] = {}
    for reasoning in args.reasoning:
        report = selective_report(
            predictor, cal, test,
            alphas=args.alphas, reasoning=reasoning, strict=args.strict,
        )
        reports[reasoning] = report
        print(f"\n{args.dataset} / {args.predictor} / {reasoning}{tag}  "
              f"n_cal={report['n_cal']} n_test={report['n_test']}\n")
        print(f"{'alpha':>7}{'target':>9}{'coverage':>10}{'set_size':>10}{'drift_gap':>11}")
        for row in report["curve"]:
            size = row["mean_set_size"]
            size_str = f"{size:.2f}" if size != float("inf") else "inf"
            print(f"{row['alpha']:>7.2f}{row['target']:>9.2f}{row['coverage']:>10.4f}"
                  f"{size_str:>10}{row['drift_gap']:>11.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"predictor": args.predictor, "reports": reports}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[cgep-selective] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
