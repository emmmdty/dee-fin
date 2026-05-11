from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from evaluator.canonical.loaders import load_documents
from evaluator.canonical.schema import EventSchema
from evaluator.canonical.stats import event_type, record_unit_count
from evaluator.docfee_official.metric import evaluate_docfee_official
from evaluator.legacy_doc2edag.metric import evaluate_legacy_doc2edag, evaluate_legacy_doc2edag_native_table
from evaluator.legacy_doc2edag.native_table import load_native_event_table
from evaluator.unified_strict.metric import evaluate_unified_strict


EvaluatorFn = Callable[..., dict[str, Any]]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluator",
        description="Reproducible evaluators for Chinese financial document-level event extraction.",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_legacy_doc2edag_command(subparsers)
    _add_eval_command(
        subparsers,
        "docfee-official",
        "DocFEE official-style event_type grouped role:value evaluator.",
        evaluate_docfee_official,
    )
    _add_eval_command(
        subparsers,
        "unified-strict",
        "Cross-dataset strict canonical role-value evaluator.",
        evaluate_unified_strict,
    )

    inspect_parser = subparsers.add_parser("inspect-gold", help="Inspect a gold file after canonical adaptation.")
    inspect_parser.add_argument("--dataset", required=True)
    inspect_parser.add_argument("--gold", required=True)
    inspect_parser.add_argument("--schema")
    inspect_parser.add_argument("--out")
    inspect_parser.set_defaults(handler=_handle_inspect_gold)
    return parser


def _add_legacy_doc2edag_command(subparsers) -> None:
    help_text = "Native Doc2EDAG/ProcNet-compatible fixed-slot evaluator for historical comparison."
    command = subparsers.add_parser("legacy-doc2edag", help=help_text, description=help_text)
    command.add_argument(
        "--input-format",
        choices=("canonical-jsonl", "native-event-table"),
        default="canonical-jsonl",
        help="Input representation. Defaults to canonical-jsonl for backward compatibility.",
    )
    command.add_argument("--dataset")
    command.add_argument("--gold")
    command.add_argument("--pred")
    command.add_argument("--schema")
    command.add_argument("--native-table")
    command.add_argument("--out")
    command.set_defaults(handler=_handle_legacy_doc2edag)


def _add_eval_command(subparsers, name: str, help_text: str, evaluator: EvaluatorFn) -> None:
    command = subparsers.add_parser(name, help=help_text, description=help_text)
    command.add_argument("--dataset", required=True)
    command.add_argument("--gold", required=True)
    command.add_argument("--pred", required=True)
    command.add_argument("--schema")
    command.add_argument("--out")
    command.set_defaults(handler=lambda args, evaluator=evaluator: _handle_eval(args, evaluator))


def _handle_legacy_doc2edag(args: argparse.Namespace) -> None:
    if args.input_format == "native-event-table":
        _handle_legacy_doc2edag_native_table(args)
    else:
        _handle_legacy_doc2edag_canonical(args)


def _handle_legacy_doc2edag_canonical(args: argparse.Namespace) -> None:
    if args.native_table:
        raise SystemExit("--native-table is only valid with --input-format native-event-table")
    missing = [name for name in ("dataset", "gold", "pred") if not getattr(args, name)]
    if missing:
        raise SystemExit(
            "canonical-jsonl mode requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    report = _evaluate_canonical_jsonl(
        args,
        evaluate_legacy_doc2edag,
        input_format="canonical-jsonl",
    )
    _write_json(report, args.out)


def _handle_legacy_doc2edag_native_table(args: argparse.Namespace) -> None:
    incompatible = [name for name in ("dataset", "gold", "pred", "schema") if getattr(args, name)]
    if incompatible:
        raise SystemExit(
            ", ".join(f"--{name.replace('_', '-')}" for name in incompatible)
            + " not compatible with --input-format native-event-table"
        )
    if not args.native_table:
        raise SystemExit("--native-table is required with --input-format native-event-table")

    native_table = load_native_event_table(args.native_table)
    report = evaluate_legacy_doc2edag_native_table(native_table, input_path=args.native_table)
    _write_json(report, args.out)


def _handle_eval(args: argparse.Namespace, evaluator: EvaluatorFn) -> None:
    report = _evaluate_canonical_jsonl(args, evaluator)
    _write_json(report, args.out)


def _evaluate_canonical_jsonl(
    args: argparse.Namespace,
    evaluator: EvaluatorFn,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    gold = load_documents(args.gold, dataset=args.dataset)
    pred = load_documents(args.pred, dataset=args.dataset)
    schema = EventSchema.from_file(args.schema) if args.schema else None
    return evaluator(
        gold.documents,
        pred.documents,
        dataset=args.dataset,
        schema=schema,
        input_paths={"gold": args.gold, "pred": args.pred},
        schema_path=args.schema,
        loader_diagnostics=_prefix_loader_diagnostics(gold.diagnostics, pred.diagnostics),
        **extra_kwargs,
    )


def _handle_inspect_gold(args: argparse.Namespace) -> None:
    loaded = load_documents(args.gold, dataset=args.dataset)
    schema = EventSchema.from_file(args.schema) if args.schema else None
    event_counter: Counter[str] = Counter()
    role_value_units = 0
    for document in loaded.documents:
        for record in document.records:
            event_counter[event_type(record)] += 1
            role_value_units += record_unit_count(record)
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "document_count": len(loaded.documents),
        "event_record_count": sum(event_counter.values()),
        "role_value_unit_count": role_value_units,
        "event_record_counts": dict(sorted(event_counter.items())),
        "schema_path": args.schema,
        "schema_event_count": len(schema.event_roles) if schema else None,
        "diagnostics": loaded.diagnostics,
    }
    _write_json(report, args.out)


def _prefix_loader_diagnostics(gold: dict[str, int], pred: dict[str, int]) -> dict[str, int]:
    merged = {}
    for prefix, diagnostics in (("gold", gold), ("pred", pred)):
        for key, value in diagnostics.items():
            merged[f"{prefix}_{key}"] = value
            if key == "parse_failure_count":
                merged[key] = merged.get(key, 0) + value
            if key == "empty_prediction_count" and prefix == "pred":
                merged[key] = value
    return merged


def _write_json(report: dict[str, Any], out: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
