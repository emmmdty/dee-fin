#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
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
    load_procnet_schema,
    stage_dataset,
)
from scripts.baseline.procnet.procnet_eval_runner import build_evaluator_commands, run_evaluator_commands
from scripts.baseline.procnet.procnet_event_table_dump import (
    matrices_from_raw_results,
    write_native_event_table,
)
from scripts.baseline.procnet.procnet_training import build_early_stopping_trainer_class
from scripts.baseline.procnet.procnet_wrapper import (
    DATASETS,
    build_run_config,
    ensure_run_layout,
    format_command,
    set_seed,
    write_json,
    write_run_metadata,
)


DEFAULT_MODEL_NAME = os.environ.get(
    "PROCNET_MODEL_NAME",
    "/data/TJK/DEE/models/chinese-roberta-wwm-ext",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run script-side ProcNet reproduction wrappers.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="runs/baseline/procnet")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server-mode", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser()
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    config = build_run_config(
        project_root=project_root,
        dataset=args.dataset,
        experiment_name=args.experiment_name,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpu=str(args.gpu),
        python_bin=args.python_bin,
        output_root=output_root,
        server_mode=args.server_mode,
    )
    if args.dry_run:
        _print_dry_run(config)
        return 0
    run_reproduction(config, command=sys.argv)
    return 0


def run_reproduction(config: Any, *, command: list[str]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ensure_run_layout(config.run_dir)
    seed_metadata = set_seed(config.seed)
    write_run_metadata(config, command=command, seed_metadata=seed_metadata)
    staged = stage_dataset(config.project_root, config.run_dir, config.dataset)
    baseline_runtime = config.run_dir / "procnet_runtime"
    if (baseline_runtime / "Result").exists():
        shutil.rmtree(baseline_runtime / "Result")
    (baseline_runtime / "Result").mkdir(parents=True, exist_ok=True)

    stdout_path = config.run_dir / "stdout.log"
    stderr_path = config.run_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = _run_procnet_training(config, baseline_runtime, staged["schema"])

    _copy_baseline_results(baseline_runtime / "Result", config.run_dir / "baseline_result")
    splits = ["dev", "test"]
    native_payloads = _write_native_and_canonical_artifacts(config, result, staged, splits)
    evaluator_commands = build_evaluator_commands(
        project_root=config.project_root,
        run_dir=config.run_dir,
        dataset=config.dataset,
        schema_path=staged["schema"],
        splits=splits,
        python_bin=config.python_bin,
    )
    eval_results = run_evaluator_commands(evaluator_commands, cwd=config.project_root)
    write_json(config.run_dir / "eval" / "command_results.json", eval_results)
    write_summary(
        run_dir=config.run_dir,
        dataset=config.dataset,
        seed=config.seed,
        splits=splits,
        early_stopping=result["early_stopping"],
        warnings=result["warnings"],
        evaluator_commands=[command.argv for command in evaluator_commands],
    )
    _write_export_drift_placeholders(config.run_dir, native_payloads)


def _run_procnet_training(config: Any, baseline_runtime: Path, schema_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(config.project_root / "baseline" / "procnet"))
    import torch
    from procnet.conf.DocEE_conf import DocEEConfig
    from procnet.conf.global_config_manager import GlobalConfigManager
    from procnet.data_preparer.DocEE_preparer import DocEEPreparer
    from procnet.data_processor.DocEE_processor import DocEEProcessor
    from procnet.metric.DocEE_metric import DocEEMetric
    from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
    from procnet.optimizer.basic_optimizer import BasicOptimizer

    GlobalConfigManager.current_path = baseline_runtime
    dee_config = DocEEConfig()
    dee_config.model_save_name = config.experiment_name
    dee_config.node_size = 512
    dee_config.proxy_slot_num = 16
    dee_config.gradient_accumulation_steps = 32
    dee_config.max_epochs = config.max_epochs
    dee_config.data_loader_shuffle = True
    dee_config.model_name = DEFAULT_MODEL_NAME
    dee_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DocEEProcessor(False)
    processor.SCHEMA = load_procnet_schema(schema_path)
    processor.SCHEMA_KEY_ENG_CHN = None
    processor.SCHEMA_KEY_CHN_ENG = None
    preparer = DocEEPreparer(config=dee_config, processor=processor)
    train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader = (
        preparer.get_loader_for_flattened_fragment_before_event()
    )
    del train_dataset, dev_dataset, test_dataset
    metric = DocEEMetric(preparer=preparer)
    model = DocEEProxyNodeModel(config=dee_config, preparer=preparer)
    model.to(dee_config.device)
    optimizer = BasicOptimizer(config=dee_config, model=model)
    EarlyStoppingDocEETrainer = build_early_stopping_trainer_class()
    trainer = EarlyStoppingDocEETrainer(
        config=dee_config,
        model=model,
        optimizer=optimizer,
        preparer=preparer,
        metric=metric,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        patience=config.patience,
    )
    train_result = trainer.train()
    return {
        "metric": metric,
        "raw_results": train_result["raw_results"],
        "early_stopping": train_result["early_stopping"],
        "warnings": [],
    }


def _write_native_and_canonical_artifacts(
    config: Any,
    result: dict[str, Any],
    staged: dict[str, Path],
    splits: list[str],
) -> dict[str, dict[str, Any]]:
    payloads = {}
    for split in splits:
        raw_results = result["raw_results"][split]
        native = matrices_from_raw_results(result["metric"], raw_results)
        native_payload = write_native_event_table(
            config.run_dir / "native_event_tables" / f"{split}.procnet_native_event_table_v1.json",
            dataset=config.dataset,
            split=split,
            seed=config.seed,
            epoch=result["early_stopping"].get("best_epoch"),
            event_types=native["event_types"],
            event_type_fields=native["event_type_fields"],
            documents=native["documents"],
            metadata={
                "experiment_name": config.experiment_name,
                "max_epochs": config.max_epochs,
                "patience": config.patience,
            },
        )
        gold_path = export_canonical_gold_from_source(
            source_path=staged[f"source_{split}"],
            output_root=config.run_dir / "canonical",
            dataset=config.dataset,
            split=split,
        )
        pred_documents = canonical_predictions_from_native_table(native_payload)
        _, pred_path = export_canonical_split(
            config.run_dir / "canonical",
            split=split,
            gold_documents=[json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line],
            pred_documents=pred_documents,
        )
        del pred_path
        payloads[split] = native_payload
    return payloads


def _copy_baseline_results(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    if source.exists():
        shutil.copytree(source, target)
    else:
        target.mkdir(parents=True, exist_ok=True)


def _write_export_drift_placeholders(run_dir: Path, native_payloads: dict[str, dict[str, Any]]) -> None:
    for split, payload in native_payloads.items():
        out = run_dir / "eval" / f"export_drift.{split}.json"
        write_json(
            out,
            {
                "split": split,
                "status": "not_computed",
                "reason": "canonical export is an exchange format and is evaluated separately from native replay",
                "native_document_count": len(payload.get("documents", [])),
            },
        )


def _print_dry_run(config: Any) -> None:
    print("DRY RUN: no training, staging, evaluator execution, or run directory creation will be performed.")
    print(f"project_root: {config.project_root}")
    print(f"run_dir: {config.run_dir}")
    print(f"dataset: {config.dataset}")
    print(f"seed: {config.seed}")
    print(f"gpu: {config.gpu}")
    print(f"--max-epochs {config.max_epochs}")
    print(f"--patience {config.patience}")
    evaluator_commands = build_evaluator_commands(
        project_root=config.project_root,
        run_dir=config.run_dir,
        dataset=config.dataset,
        schema_path=config.run_dir / "staged_data" / config.dataset / "schema.json",
        splits=["dev", "test"],
        python_bin=config.python_bin,
    )
    for command in evaluator_commands:
        print(format_command(command.argv))


if __name__ == "__main__":
    raise SystemExit(main())
