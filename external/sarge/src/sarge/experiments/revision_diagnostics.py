from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalPaths:
    legacy: Path
    unified: Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


EXPORT_RUNS: tuple[tuple[str, str, str, EvalPaths], ...] = (
    (
        "ChFinAnn",
        "Reported canonical export",
        "server:runs/sarge_hf_ChFinAnn-Doc2EDAG_test_seed13_4bitNF4_k1_backend_ablation_20260520T165638Z/sarge_infer_ChFinAnn-Doc2EDAG_test_20260520T165641Z",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_test_seed13_hf4bin_k1/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_test_seed13_hf4bin_k1/eval/eval_unified_strict.json",
        ),
    ),
    (
        "ChFinAnn",
        "Direct canonical export",
        "server:runs/sarge_export_guard_ablation_ChFinAnn_seed13_pass_through_20260525",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_export_pass_through/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_export_pass_through/eval/eval_unified_strict.json",
        ),
    ),
    (
        "ChFinAnn",
        "Exact dedup only",
        "server:runs/sarge_export_guard_ablation_ChFinAnn_seed13_dedup_only_20260525",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_export_dedup_only/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_export_dedup_only/eval/eval_unified_strict.json",
        ),
    ),
    (
        "ChFinAnn",
        "Conservative split/merge guard",
        "server:runs/sarge_export_guard_ablation_ChFinAnn_seed13_conservative_assembler_20260525",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_export_conservative/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_export_conservative/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "Reported canonical export",
        "server:runs/sarge_infer_DuEE-Fin-dev500_test_seed13_safe_anchor_source_f56a0d3_20260520T003122Z/sarge_infer_DuEE-Fin-dev500_test_20260520T003122Z",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_k1_no_lrd/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_k1_no_lrd/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "Direct canonical export",
        "server:runs/sarge_export_guard_ablation_DuEEFin_seed13_pass_through_20260525",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_export_pass_through/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_export_pass_through/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "Exact dedup only",
        "server:runs/sarge_export_guard_ablation_DuEEFin_seed13_dedup_only_20260525",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_export_dedup_only/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_export_dedup_only/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "Conservative split/merge guard",
        "server:runs/sarge_export_guard_ablation_DuEEFin_seed13_conservative_assembler_20260525",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_export_conservative/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_export_conservative/eval/eval_unified_strict.json",
        ),
    ),
)


VALIDITY_RUNS: tuple[tuple[str, str, str, EvalPaths], ...] = (
    (
        "ChFinAnn",
        "SARGE seed 13",
        "test",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_test_seed13_hf4bin_k1/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_test_seed13_hf4bin_k1/eval/eval_unified_strict.json",
        ),
    ),
    (
        "ChFinAnn",
        "Base no-SFT",
        "test",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_test_seed13_vllm_bf16_no_sft/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/chfinann_test_seed13_vllm_bf16_no_sft/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "SARGE seed 13",
        "test",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_k1_no_lrd/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_k1_no_lrd/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "Base no-SFT",
        "test",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_no_sft/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_no_sft/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "DeepSeek flash",
        "dev500",
        EvalPaths(
            PROJECT_ROOT / "runs/sarge_deepseek_api_DuEE-Fin-dev500_dev_limit500_deepseek-v4-flash_role_safe_surface_memory_c100_20260522T161310Z/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "runs/sarge_deepseek_api_DuEE-Fin-dev500_dev_limit500_deepseek-v4-flash_role_safe_surface_memory_c100_20260522T161310Z/eval/eval_unified_strict.json",
        ),
    ),
    (
        "DuEE-Fin",
        "DeepSeek pro",
        "dev500",
        EvalPaths(
            PROJECT_ROOT / "runs/sarge_deepseek_api_DuEE-Fin-dev500_dev_limit500_deepseek-v4-pro_role_safe_surface_memory_c100_20260522T161441Z/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "runs/sarge_deepseek_api_DuEE-Fin-dev500_dev_limit500_deepseek-v4-pro_role_safe_surface_memory_c100_20260522T161441Z/eval/eval_unified_strict.json",
        ),
    ),
)


MODULE_RUNS: tuple[tuple[str, EvalPaths], ...] = (
    (
        "Full",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_k1_no_lrd/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_k1_no_lrd/eval/eval_unified_strict.json",
        ),
    ),
    (
        "w/o Surface Memory",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_ablation_no_surface_memory/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_ablation_no_surface_memory/eval/eval_unified_strict.json",
        ),
    ),
    (
        "w/o Slot Plan",
        EvalPaths(
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_ablation_no_slot_plan/eval/eval_legacy_doc2edag.json",
            PROJECT_ROOT / "paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_ablation_no_slot_plan/eval/eval_unified_strict.json",
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build revision diagnostic tables for the CCKS paper.")
    parser.add_argument(
        "--failure-csv",
        default=str(PROJECT_ROOT / "runs/sarge_record_diagnostics_seed13_20260525/eval/exact_record_failures.csv"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs/sarge_revision_diagnostics_20260525"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    export_rows = build_export_rows()
    validity_rows = build_validity_rows()
    taxonomy_rows = build_taxonomy_rows(Path(args.failure_csv))
    module_rows = build_module_rows()

    _write_csv(eval_dir / "export_guard_ablation.csv", export_rows)
    _write_csv(eval_dir / "output_validity.csv", validity_rows)
    _write_csv(eval_dir / "exact_record_failure_taxonomy.csv", taxonomy_rows)
    _write_csv(eval_dir / "module_exact_record.csv", module_rows)

    (eval_dir / "export_guard_ablation.tex").write_text(render_export_table(export_rows) + "\n", encoding="utf-8")
    (eval_dir / "output_validity.tex").write_text(render_validity_table(validity_rows) + "\n", encoding="utf-8")
    (eval_dir / "exact_record_failure_taxonomy.tex").write_text(
        render_taxonomy_table(taxonomy_rows) + "\n", encoding="utf-8"
    )
    (eval_dir / "module_exact_record.tex").write_text(render_module_table(module_rows) + "\n", encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "export_guard_ablation_csv": str(eval_dir / "export_guard_ablation.csv"),
            "export_guard_ablation_tex": str(eval_dir / "export_guard_ablation.tex"),
            "output_validity_csv": str(eval_dir / "output_validity.csv"),
            "output_validity_tex": str(eval_dir / "output_validity.tex"),
            "exact_record_failure_taxonomy_csv": str(eval_dir / "exact_record_failure_taxonomy.csv"),
            "exact_record_failure_taxonomy_tex": str(eval_dir / "exact_record_failure_taxonomy.tex"),
            "module_exact_record_csv": str(eval_dir / "module_exact_record.csv"),
            "module_exact_record_tex": str(eval_dir / "module_exact_record.tex"),
        },
        "export_guard_ablation": export_rows,
        "output_validity": validity_rows,
        "exact_record_failure_taxonomy": taxonomy_rows,
        "module_exact_record": module_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json")}, ensure_ascii=False, indent=2))
    return 0


def build_export_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, mode, source, paths in EXPORT_RUNS:
        legacy = _load_json(paths.legacy)
        unified = _load_json(paths.unified)
        rows.append(
            {
                "dataset": dataset,
                "export_mode": mode,
                "legacy_f1": _pct(legacy["overall"]["f1"]),
                "legacy_precision": _pct(legacy["overall"]["precision"]),
                "legacy_recall": _pct(legacy["overall"]["recall"]),
                "exact_record": _exact_record_pct(unified),
                "record_exact_match_count": _diag(unified, "record_exact_match_count"),
                "validated_record_count": _diag(unified, "validated_record_count"),
                "parse_failures": _diag(legacy, "parse_failure_count"),
                "invalid_types": _diag(legacy, "invalid_event_type_count"),
                "invalid_roles": _diag(legacy, "invalid_role_count"),
                "source": source,
            }
        )
    return rows


def build_validity_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, setting, split, paths in VALIDITY_RUNS:
        legacy = _load_json(paths.legacy)
        unified = _load_json(paths.unified)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "setting": setting,
                "legacy_f1": _pct(legacy["overall"]["f1"]),
                "schema_valid": _pct(_diag(legacy, "schema_valid_rate")),
                "parse_failures": _diag(legacy, "parse_failure_count"),
                "invalid_types": _diag(legacy, "invalid_event_type_count"),
                "invalid_roles": _diag(legacy, "invalid_role_count"),
                "empty_predictions": _diag(legacy, "empty_prediction_count"),
                "validated_records": _diag(unified, "validated_record_count"),
                "exact_record": _exact_record_pct(unified),
            }
        )
    return rows


def build_module_rows() -> list[dict[str, Any]]:
    rows = []
    for setting, paths in MODULE_RUNS:
        legacy = _load_json(paths.legacy)
        unified = _load_json(paths.unified)
        rows.append(
            {
                "setting": setting,
                "legacy_f1": _pct(legacy["overall"]["f1"]),
                "exact_record": _exact_record_pct(unified),
                "empty_predictions": _diag(legacy, "empty_prediction_count"),
                "record_exact_match_count": _diag(unified, "record_exact_match_count"),
                "validated_record_count": _diag(unified, "validated_record_count"),
            }
        )
    return rows


def build_taxonomy_rows(failure_csv: Path) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    with failure_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = str(row["dataset"]).replace("-Doc2EDAG", "").replace("-dev500", "")
            counters[dataset][_taxonomy_label(row)] += 1

    order = (
        "over-count / split-like",
        "under-count / merge-like",
        "binding or mixed-value mismatch",
        "missing-value only",
        "extra-value only",
        "duplicate predicted record",
        "empty/anchor/schema failure",
    )
    rows: list[dict[str, Any]] = []
    for dataset in sorted(counters):
        total = sum(counters[dataset].values())
        for label in order:
            count = counters[dataset][label]
            rows.append(
                {
                    "dataset": dataset,
                    "category": label,
                    "count": count,
                    "share": round((count / total * 100.0) if total else 0.0, 1),
                    "total_failures": total,
                }
            )
    return rows


def render_export_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Export guard ablation on seed-13 predictions. Direct export keeps the selected schema-valid canonical records; dedup and conservative rows re-export the same parsed predictions offline.}",
        r"\label{tab:export-guard-ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Export mode & Legacy-FS & Exact-Record \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['dataset'])} & {_tex(row['export_mode'])} & {_fmt(row['legacy_f1'])} & {_fmt(row['exact_record'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_validity_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Output validity diagnostics. API-only rows use the DuEE-Fin internal dev500 prompt-interface split and are not ranking evidence.}",
        r"\label{tab:output-validity}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Dataset & Split & Setting & F1 & Schema & Parse & Empty & Exact \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex(row["dataset"]),
                    _tex(row["split"]),
                    _tex(row["setting"]),
                    _fmt(row["legacy_f1"]),
                    _fmt(row["schema_valid"]),
                    str(row["parse_failures"]),
                    str(row["empty_predictions"]),
                    _fmt(row["exact_record"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_taxonomy_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["count"]:
            grouped[str(row["dataset"])].append(row)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Automatic Exact-Record failure taxonomy for seed-13 main runs. Categories are heuristic diagnostics computed from count mismatch and role FP/FN patterns.}",
        r"\label{tab:exact-record-failure-taxonomy}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Failure category & Count & Share \\",
        r"\midrule",
    ]
    for dataset in sorted(grouped):
        for row in grouped[dataset]:
            lines.append(
                f"{_tex(dataset)} & {_tex(row['category'])} & {row['count']} & {_fmt(row['share'])} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_module_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DuEE-Fin HF prompt-module diagnostics with Exact-Record. Rows are single-seed ablations.}",
        r"\label{tab:module-exact-record}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Setting & Legacy-FS & Exact-Record & Empty pred. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['setting'])} & {_fmt(row['legacy_f1'])} & {_fmt(row['exact_record'])} & {row['empty_predictions']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _taxonomy_label(row: dict[str, str]) -> str:
    if row.get("schema_parse_error_flag") == "1" or row.get("empty_predicted_record_flag") == "1":
        return "empty/anchor/schema failure"
    if row.get("missing_configured_anchor_flag") == "1":
        return "empty/anchor/schema failure"
    if row.get("duplicate_predicted_record_flag") == "1":
        return "duplicate predicted record"
    gold_count = int(row["gold_record_count"])
    pred_count = int(row["pred_record_count"])
    if pred_count > gold_count:
        return "over-count / split-like"
    if pred_count < gold_count:
        return "under-count / merge-like"
    fp = int(row["role_fp"])
    fn = int(row["role_fn"])
    if fn and not fp:
        return "missing-value only"
    if fp and not fn:
        return "extra-value only"
    return "binding or mixed-value mismatch"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _diag(report: dict[str, Any], key: str) -> Any:
    return (report.get("diagnostics") or {}).get(key, 0)


def _exact_record_pct(unified_report: dict[str, Any]) -> float:
    diag = unified_report.get("diagnostics") or {}
    exact = float(diag.get("record_exact_match_count") or 0)
    denom = float(diag.get("validated_record_count") or 0)
    return round(2.0 * exact / denom * 100.0, 1) if denom else 0.0


def _pct(value: Any) -> float:
    return round(float(value) * 100.0, 1)


def _fmt(value: Any) -> str:
    return f"{float(value):.1f}"


def _tex(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("&", r"\&")


if __name__ == "__main__":
    raise SystemExit(main())
