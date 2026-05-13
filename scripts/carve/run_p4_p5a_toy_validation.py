#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carve.allocation import p4_toy_validation_summary, p5a_toy_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CARVE P4/P5a toy validation summary.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = {
        "status": "toy_behavior_only",
        "p4": p4_toy_validation_summary(),
        "p5a": p5a_toy_comparison(),
        "non_goals": ["no dev scoring", "no model training", "no paper main-table claim"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
