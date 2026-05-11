from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.baseline.procnet.procnet_wrapper import expected_artifacts_for_split, write_json


def write_summary(
    *,
    run_dir: Path,
    dataset: str,
    seed: int,
    splits: list[str],
    early_stopping: dict[str, Any],
    warnings: list[str],
    evaluator_commands: list[list[str]],
) -> tuple[Path, Path]:
    split_artifacts = {split: expected_artifacts_for_split(run_dir, split) for split in splits}
    eval_reports = _collect_eval_reports(run_dir)
    summary = {
        "dataset": dataset,
        "seed": seed,
        "splits": splits,
        "artifacts": split_artifacts,
        "early_stopping": early_stopping,
        "warnings": warnings,
        "evaluator_commands": evaluator_commands,
        "eval_reports": eval_reports,
    }
    json_path = run_dir / "summary.json"
    md_path = run_dir / "summary.md"
    write_json(json_path, summary)
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _collect_eval_reports(run_dir: Path) -> dict[str, Any]:
    reports = {}
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return reports
    for path in sorted(eval_dir.glob("*.json")):
        try:
            reports[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reports[path.name] = {"error": "invalid-json"}
    return reports


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# ProcNet Reproduction Summary: {summary['dataset']}",
        "",
        f"- seed: {summary['seed']}",
        f"- splits: {', '.join(summary['splits'])}",
        "",
        "## Artifacts",
        "",
    ]
    for split, artifacts in summary["artifacts"].items():
        lines.append(f"### {split}")
        for key in (
            "native_gold_path",
            "native_pred_path",
            "canonical_gold_path",
            "canonical_pred_path",
        ):
            lines.append(f"- {key}: `{artifacts[key]}`")
        lines.append("")
    lines.extend(["## Early Stopping", ""])
    for key, value in summary["early_stopping"].items():
        lines.append(f"- {key}: {value}")
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)
