#!/usr/bin/env python
"""Rebuild the CGEP corpus from MAVEN-ERE and check it against SeDGPL's stats.

SeDGPL (arXiv:2409.17480) never released its MAVEN build (`MAVENSubWoRe.npy`),
so its published CGEP-MAVEN numbers are not same-data comparable and the corpus
must be rebuilt. `finekg.succession.data.cgep` documents the protocol; this
script runs it and prints the acceptance table.

    uv run python scripts/build_cgep.py --split train+valid --report-stats
    uv run python scripts/build_cgep.py --split train --output runs/cgep/train.jsonl

The ECG count is a known, reported deviation (3743 vs the paper's 5308): no
reconstruction reproduces it while also matching the node and edge averages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from finekg.succession.data.cgep import CANDIDATE_SET_SIZE, build_cgep, iter_documents

_PROCESSED = Path("data/processed/maven_ere")

# Paper statistics (SeDGPL Table 1) and what our protocol yields on train+valid.
# `tolerance` is None where the target is recorded rather than enforced.
_ACCEPTANCE: tuple[tuple[str, str, float, float, float | None], ...] = (
    ("documents", "documents", 3015.0, 2994.0, 0.02),  # relative
    ("nodes_per_ecg", "nodes / ECG", 8.4, 8.82, 0.5),  # absolute
    ("edges_per_ecg", "edges / ECG", 12.9, 13.21, 0.5),  # absolute
    ("ecgs", "ECGs", 5308.0, 3743.0, None),
    ("instances", "instances", 12200.0, 10116.0, None),
)


def _split_paths(split: str) -> list[Path]:
    paths = [_PROCESSED / f"{name}.jsonl" for name in split.split("+")]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing split file(s): {', '.join(str(p) for p in missing)}")
    return paths


def _report(stats: dict[str, float], split: str) -> bool:
    # Triggers collide, so the nominal 512-way choice hides a smaller answer set.
    # SeDGPL's released ESC build shows the same ratio: 178.0 distinct of 256.
    print(
        f"\n[cgep] split={split}  candidate pool={int(stats['candidate_pool'])} event nodes"
        f"  |  distinct answers/instance={stats['distinct_answers']:.1f}\n"
    )
    header = f"{'metric':<16}{'paper':>10}{'expected':>11}{'measured':>11}   verdict"
    print(header)
    print("-" * len(header))
    ok = True
    for key, label, paper, expected, tolerance in _ACCEPTANCE:
        measured = stats[key]
        if tolerance is None:
            verdict = "recorded (known deviation)"
        elif key == "documents":
            passed = abs(measured - expected) <= tolerance * expected
            ok &= passed
            verdict = "PASS" if passed else "FAIL"
        else:
            passed = abs(measured - expected) <= tolerance
            ok &= passed
            verdict = "PASS" if passed else "FAIL"
        print(f"{label:<16}{paper:>10.1f}{expected:>11.2f}{measured:>11.2f}   {verdict}")
    print()
    return ok


def _write(path: Path, instances: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for instance in instances:
            # `asdict` recurses into the tuples of `CgepNode`s for free.
            handle.write(json.dumps(asdict(instance), ensure_ascii=False) + "\n")
    print(f"[cgep] wrote {len(instances)} instances -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="maven", choices=("maven",))
    parser.add_argument("--split", default="train+valid", help="e.g. train, valid, train+valid")
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--no-subevent", action="store_true", help="causal edges only")
    parser.add_argument("--candidates", type=int, default=CANDIDATE_SET_SIZE)
    parser.add_argument("--seed", type=int, default=209, help="SeDGPL's seed")
    parser.add_argument("--report-stats", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = _split_paths(args.split)
    instances, stats = build_cgep(
        iter_documents(str(p) for p in paths),
        min_nodes=args.min_nodes,
        include_subevent=not args.no_subevent,
        n_candidates=args.candidates,
        seed=args.seed,
    )

    if args.output:
        _write(args.output, instances)

    if args.report_stats:
        default_protocol = (
            args.split == "train+valid" and args.min_nodes == 4 and not args.no_subevent
        )
        if not default_protocol:
            print("[cgep] non-default protocol: acceptance targets do not apply")
            print(f"[cgep] {stats}")
            return 0
        return 0 if _report(stats, args.split) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
