from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluatorCommand:
    name: str
    split: str
    argv: list[str]
    out: Path


def build_evaluator_commands(
    *,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    schema_path: Path,
    splits: list[str],
    python_bin: str,
) -> list[EvaluatorCommand]:
    del project_root
    commands: list[EvaluatorCommand] = []
    for split in splits:
        native_table = run_dir / "native_event_tables" / f"{split}.procnet_native_event_table_v1.json"
        canonical_gold = run_dir / "canonical" / f"{split}.canonical.gold.jsonl"
        canonical_pred = run_dir / "canonical" / f"{split}.canonical.pred.jsonl"
        native_out = run_dir / "eval" / f"legacy_native_event_table.{split}.json"
        legacy_out = run_dir / "eval" / f"legacy_canonical.{split}.json"
        unified_out = run_dir / "eval" / f"unified_strict.{split}.json"
        commands.append(
            EvaluatorCommand(
                name="legacy_native_event_table",
                split=split,
                out=native_out,
                argv=[
                    python_bin,
                    "-m",
                    "evaluator",
                    "legacy-doc2edag",
                    "--input-format",
                    "native-event-table",
                    "--native-table",
                    str(native_table),
                    "--out",
                    str(native_out),
                ],
            )
        )
        commands.append(
            EvaluatorCommand(
                name="legacy_canonical",
                split=split,
                out=legacy_out,
                argv=[
                    python_bin,
                    "-m",
                    "evaluator",
                    "legacy-doc2edag",
                    "--input-format",
                    "canonical-jsonl",
                    "--dataset",
                    dataset,
                    "--gold",
                    str(canonical_gold),
                    "--pred",
                    str(canonical_pred),
                    "--schema",
                    str(schema_path),
                    "--out",
                    str(legacy_out),
                ],
            )
        )
        commands.append(
            EvaluatorCommand(
                name="unified_strict",
                split=split,
                out=unified_out,
                argv=[
                    python_bin,
                    "-m",
                    "evaluator",
                    "unified-strict",
                    "--dataset",
                    dataset,
                    "--gold",
                    str(canonical_gold),
                    "--pred",
                    str(canonical_pred),
                    "--schema",
                    str(schema_path),
                    "--out",
                    str(unified_out),
                ],
            )
        )
    return commands


def run_evaluator_commands(commands: list[EvaluatorCommand], *, cwd: Path) -> list[dict[str, object]]:
    results = []
    for command in commands:
        command.out.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(
            command.argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        results.append(
            {
                "name": command.name,
                "split": command.split,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "out": str(command.out),
                "argv": command.argv,
            }
        )
        if process.returncode != 0:
            raise RuntimeError(f"evaluator command failed: {' '.join(command.argv)}\n{process.stderr}")
    return results
