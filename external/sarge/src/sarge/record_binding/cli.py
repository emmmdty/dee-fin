from __future__ import annotations

import argparse

from sarge.record_binding.run import run_record_binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply schema-safe record binding to a SARGE run root.")
    parser.add_argument("--input-run-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    summary = run_record_binding(
        input_run_root=args.input_run_root,
        output_run_root=args.output_run_root,
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        threshold=args.threshold,
    )
    print(
        "record_binding "
        f"docs={summary['docs']} "
        f"events_before={summary['events_before']} "
        f"events_after={summary['events_after']} "
        f"merges={summary['merge_count']} "
        f"blocked={summary['blocked_count']} "
        f"pred={summary['output_prediction_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
