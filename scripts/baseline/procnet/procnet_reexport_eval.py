#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.baseline.procnet.collect_metrics import write_summary
from scripts.baseline.procnet.procnet_data_adapters import (
    canonical_predictions_from_native_table,
    export_canonical_gold_from_source,
    export_canonical_split,
)
from scripts.baseline.procnet.procnet_eval_runner import build_evaluator_commands, run_evaluator_commands
from scripts.baseline.procnet.procnet_value_decode import ProcNetTokenSlotDecoder, load_procnet_tokenizer
from scripts.baseline.procnet.procnet_wrapper import DATASETS, write_json


DEFAULT_MODEL_NAME = os.environ.get(
    "PROCNET_MODEL_NAME",
    "/data/TJK/DEE/models/chinese-roberta-wwm-ext",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-export ProcNet canonical artifacts and re-run evaluators.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="runs/baseline/procnet")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--decode-token-slots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser()
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_dir = output_root / args.experiment_name

    if args.dry_run:
        _print_dry_run(
            project_root=project_root,
            run_dir=run_dir,
            dataset=args.dataset,
            splits=args.splits,
            python_bin=args.python_bin,
            model_name=args.model_name,
            decode_token_slots=args.decode_token_slots,
        )
        return 0

    reexport_and_evaluate(
        project_root=project_root,
        run_dir=run_dir,
        dataset=args.dataset,
        python_bin=args.python_bin,
        splits=args.splits,
        model_name=args.model_name,
        decode_token_slots=args.decode_token_slots,
    )
    return 0


def reexport_and_evaluate(
    *,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    python_bin: str,
    splits: list[str],
    model_name: str,
    decode_token_slots: bool,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    decoder = None
    if decode_token_slots:
        tokenizer = load_procnet_tokenizer(model_name)
        decoder = ProcNetTokenSlotDecoder(tokenizer)

    canonical_reports = rewrite_canonical_artifacts(
        project_root=project_root,
        run_dir=run_dir,
        dataset=dataset,
        splits=splits,
        value_transform=decoder,
    )

    schema_path = _schema_path(run_dir, dataset)
    evaluator_commands = build_evaluator_commands(
        project_root=project_root,
        run_dir=run_dir,
        dataset=dataset,
        schema_path=schema_path,
        splits=splits,
        python_bin=python_bin,
    )
    eval_results = run_evaluator_commands(evaluator_commands, cwd=project_root)
    write_json(run_dir / "eval" / "command_results.json", eval_results)
    _write_export_drift_reports(run_dir, splits=splits)
    _refresh_summary(
        run_dir=run_dir,
        dataset=dataset,
        splits=splits,
        evaluator_commands=[command.argv for command in evaluator_commands],
    )
    report = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "splits": splits,
        "decode_token_slots": decode_token_slots,
        "model_name": model_name,
        "decoder": decoder.report() if decoder else None,
        "canonical": canonical_reports,
        "evaluator_command_count": len(evaluator_commands),
    }
    write_json(run_dir / "eval" / "reexport_recovery.json", report)
    return report


def rewrite_canonical_artifacts(
    *,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    splits: list[str],
    value_transform: Any = None,
) -> dict[str, Any]:
    reports = {}
    for split in splits:
        native_path = run_dir / "native_event_tables" / f"{split}.procnet_native_event_table_v1.json"
        if not native_path.exists():
            raise FileNotFoundError(f"missing native event table: {native_path}")
        native_payload = json.loads(native_path.read_text(encoding="utf-8"))
        source_path = _source_path(project_root, dataset, split)
        gold_path = export_canonical_gold_from_source(
            source_path=source_path,
            output_root=run_dir / "canonical",
            dataset=dataset,
            split=split,
        )
        pred_documents = canonical_predictions_from_native_table(
            native_payload,
            value_transform=value_transform,
        )
        _, pred_path = export_canonical_split(
            run_dir / "canonical",
            split=split,
            gold_documents=[json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line],
            pred_documents=pred_documents,
        )
        reports[split] = {
            "native_table": str(native_path),
            "gold": str(gold_path),
            "pred": str(pred_path),
            "document_count": len(pred_documents),
        }
    return reports


def _source_path(project_root: Path, dataset: str, split: str) -> Path:
    suffix = "jsonl" if dataset == "DuEE-Fin-dev500" else "json"
    path = project_root / "data" / "processed" / dataset / f"{split}.{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"missing processed source split: {path}")
    return path


def _schema_path(run_dir: Path, dataset: str) -> Path:
    path = run_dir / "staged_data" / dataset / "schema.json"
    if not path.exists():
        raise FileNotFoundError(f"missing staged schema: {path}")
    return path


def _refresh_summary(
    *,
    run_dir: Path,
    dataset: str,
    splits: list[str],
    evaluator_commands: list[list[str]],
) -> None:
    existing = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    write_summary(
        run_dir=run_dir,
        dataset=dataset,
        seed=int(existing.get("seed", 42)),
        splits=splits,
        early_stopping=existing.get("early_stopping", {}),
        warnings=existing.get("warnings", []),
        evaluator_commands=evaluator_commands,
    )


def _write_export_drift_reports(run_dir: Path, *, splits: list[str]) -> None:
    for split in splits:
        native_path = run_dir / "eval" / f"legacy_native_event_table.{split}.json"
        canonical_path = run_dir / "eval" / f"legacy_canonical.{split}.json"
        unified_path = run_dir / "eval" / f"unified_strict.{split}.json"
        native = _load_report(native_path)
        canonical = _load_report(canonical_path)
        unified = _load_report(unified_path)
        native_f1 = _overall_f1(native)
        canonical_f1 = _overall_f1(canonical)
        unified_f1 = _overall_f1(unified)
        write_json(
            run_dir / "eval" / f"export_drift.{split}.json",
            {
                "split": split,
                "status": "computed",
                "native_replay_f1": native_f1,
                "exported_legacy_f1": canonical_f1,
                "unified_strict_f1": unified_f1,
                "native_to_exported_f1_gap": None if native_f1 is None or canonical_f1 is None else native_f1 - canonical_f1,
                "exported_to_unified_f1_gap": None if canonical_f1 is None or unified_f1 is None else canonical_f1 - unified_f1,
                "inputs": {
                    "native": str(native_path),
                    "canonical": str(canonical_path),
                    "unified": str(unified_path),
                },
            },
        )


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _overall_f1(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    overall = report.get("overall")
    if not isinstance(overall, dict):
        return None
    value = overall.get("f1")
    return float(value) if value is not None else None


def _print_dry_run(
    *,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    splits: list[str],
    python_bin: str,
    model_name: str,
    decode_token_slots: bool,
) -> None:
    print("DRY RUN: no canonical artifacts or evaluator reports will be rewritten.")
    print(f"project_root: {project_root}")
    print(f"run_dir: {run_dir}")
    print(f"dataset: {dataset}")
    print(f"splits: {', '.join(splits)}")
    print(f"decode_token_slots: {decode_token_slots}")
    print(f"model_name: {model_name}")
    schema_path = run_dir / "staged_data" / dataset / "schema.json"
    for command in build_evaluator_commands(
        project_root=project_root,
        run_dir=run_dir,
        dataset=dataset,
        schema_path=schema_path,
        splits=splits,
        python_bin=python_bin,
    ):
        print(" ".join(command.argv))


if __name__ == "__main__":
    raise SystemExit(main())
