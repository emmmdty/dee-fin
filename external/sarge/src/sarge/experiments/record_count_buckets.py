from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator.canonical.grouping import all_document_ids, group_by_event_type, merge_documents
from evaluator.canonical.loaders import load_documents
from evaluator.canonical.stats import (
    Counts,
    count_record_pair,
    count_unmatched_gold,
    count_unmatched_pred,
    exact_record_match,
    record_unit_count,
    role_value_sets,
)
from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord
from evaluator.unified_strict.matcher import match_records
from sarge.postprocess.rule_planner import ANCHOR_ROLES_BY_EVENT_TYPE


BUCKET_LABELS = ("1", "2", "3", ">=4")
FAILURE_FLAG_COLUMNS = (
    "count_mismatch_flag",
    "duplicate_predicted_record_flag",
    "empty_predicted_record_flag",
    "missing_configured_anchor_flag",
    "schema_parse_error_flag",
)


@dataclass(frozen=True)
class DiagnosticInput:
    dataset: str
    gold_path: Path
    pred_path: Path
    eval_json_path: Path
    train_manifest_path: Path
    pipeline_summary_path: Path


@dataclass
class BucketStats:
    group_count: int = 0
    record_count_correct: int = 0
    gold_record_count: int = 0
    pred_record_count: int = 0
    exact_record_matches: int = 0
    counts: Counts = field(default_factory=Counts)

    def to_report(self) -> dict[str, float | int]:
        metrics = self.counts.to_metrics()
        exact_denominator = self.gold_record_count + self.pred_record_count
        record_count_accuracy = self.record_count_correct / self.group_count if self.group_count else 0.0
        return {
            "instances": self.group_count,
            "group_count": self.group_count,
            "gold_record_count": self.gold_record_count,
            "pred_record_count": self.pred_record_count,
            "role_tp": self.counts.tp,
            "role_fp": self.counts.fp,
            "role_fn": self.counts.fn,
            "role_precision": float(metrics["precision"]),
            "role_recall": float(metrics["recall"]),
            "role_f1": float(metrics["f1"]),
            "exact_record_matches": self.exact_record_matches,
            "exact_record_f1": (2.0 * self.exact_record_matches / exact_denominator) if exact_denominator else 0.0,
            "record_count_accuracy": record_count_accuracy,
        }


@dataclass
class OverallStats:
    counts: Counts = field(default_factory=Counts)
    exact_record_matches: int = 0
    matched_record_pairs: int = 0
    gold_record_count: int = 0
    pred_record_count: int = 0
    group_count: int = 0

    def add_group(self, gold_group: list[CanonicalEventRecord], pred_group: list[CanonicalEventRecord]) -> None:
        group_scores = _score_group(gold_group, pred_group)
        self.counts.add(group_scores["counts"])
        self.exact_record_matches += int(group_scores["exact_matches"])
        self.matched_record_pairs += int(group_scores["matched_pairs"])
        self.gold_record_count += len(gold_group)
        self.pred_record_count += len(pred_group)
        self.group_count += 1

    def to_report(self) -> dict[str, float | int]:
        metrics = self.counts.to_metrics()
        exact_denominator = self.gold_record_count + self.pred_record_count
        return {
            "group_count": self.group_count,
            "gold_record_count": self.gold_record_count,
            "pred_record_count": self.pred_record_count,
            "role_tp": self.counts.tp,
            "role_fp": self.counts.fp,
            "role_fn": self.counts.fn,
            "role_precision": float(metrics["precision"]),
            "role_recall": float(metrics["recall"]),
            "role_f1": float(metrics["f1"]),
            "exact_record_matches": self.exact_record_matches,
            "matched_record_pairs": self.matched_record_pairs,
            "exact_record_f1": (2.0 * self.exact_record_matches / exact_denominator) if exact_denominator else 0.0,
        }


def compute_bucket_metrics(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
) -> dict[str, dict[str, float | int]]:
    return compute_bucket_report(gold_documents, pred_documents)["buckets"]


def compute_bucket_report(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    bucket_stats = {label: BucketStats() for label in BUCKET_LABELS}
    diagnostics: dict[str, int] = defaultdict(int)

    gold_by_id = merge_documents(gold_documents)
    pred_by_id = merge_documents(pred_documents)

    for document_id in all_document_ids(gold_documents, pred_documents):
        gold_by_event = group_by_event_type(gold_by_id.get(document_id, []))
        pred_by_event = group_by_event_type(pred_by_id.get(document_id, []))
        for event_name in sorted(set(gold_by_event) | set(pred_by_event)):
            gold_group = gold_by_event.get(event_name, [])
            pred_group = pred_by_event.get(event_name, [])
            gold_count = len(gold_group)
            pred_count = len(pred_group)
            if gold_count == 0:
                if pred_count:
                    diagnostics["spurious_pred_group_count"] += 1
                    diagnostics["spurious_pred_record_count"] += pred_count
                continue

            bucket = _bucket_label(gold_count)
            group_scores = _score_group(gold_group, pred_group)
            stats = bucket_stats[bucket]
            stats.group_count += 1
            stats.gold_record_count += gold_count
            stats.pred_record_count += pred_count
            stats.record_count_correct += int(gold_count == pred_count)
            stats.exact_record_matches += int(group_scores["exact_matches"])
            stats.counts.add(group_scores["counts"])

    return {
        "dataset": dataset,
        "metric_family": "unified_strict_bucketed_by_gold_same_event_record_count",
        "buckets": {label: bucket_stats[label].to_report() for label in BUCKET_LABELS},
        "diagnostics": dict(sorted(diagnostics.items())),
    }


def compute_failure_rows(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
    *,
    dataset: str,
    eval_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gold_by_id = merge_documents(gold_documents)
    pred_by_id = merge_documents(pred_documents)
    schema_parse_flag_available = _has_positive_schema_or_parse_diagnostic(eval_diagnostics or {})

    for document_id in all_document_ids(gold_documents, pred_documents):
        gold_by_event = group_by_event_type(gold_by_id.get(document_id, []))
        pred_by_event = group_by_event_type(pred_by_id.get(document_id, []))
        for event_name in sorted(set(gold_by_event) | set(pred_by_event)):
            gold_group = gold_by_event.get(event_name, [])
            pred_group = pred_by_event.get(event_name, [])
            if not gold_group and not pred_group:
                continue

            group_scores = _score_group(gold_group, pred_group)
            exact_matches = int(group_scores["exact_matches"])
            gold_count = len(gold_group)
            pred_count = len(pred_group)
            exact_record_failed = not (gold_count == pred_count == exact_matches)
            if not exact_record_failed:
                continue

            flags = {
                "count_mismatch_flag": gold_count != pred_count,
                "duplicate_predicted_record_flag": _has_duplicate_predicted_record(pred_group),
                "empty_predicted_record_flag": any(record_unit_count(record) == 0 for record in pred_group),
                "missing_configured_anchor_flag": _missing_configured_anchor(event_name, gold_group, pred_group),
                "schema_parse_error_flag": schema_parse_flag_available,
            }
            automatic_flags = [name.removesuffix("_flag") for name, enabled in flags.items() if enabled]
            rows.append(
                {
                    "dataset": dataset,
                    "doc_id": document_id,
                    "event_type": event_name,
                    "gold_record_count": gold_count,
                    "pred_record_count": pred_count,
                    "exact_record_matches": exact_matches,
                    "role_tp": group_scores["counts"].tp,
                    "role_fp": group_scores["counts"].fp,
                    "role_fn": group_scores["counts"].fn,
                    **flags,
                    "automatic_error_flags": ";".join(automatic_flags) if automatic_flags else "role_value_mismatch",
                    "suggested_error_type": _suggest_error_type(gold_count, pred_count, flags),
                    "_gold_records": _records_for_csv(gold_group),
                    "_pred_records": _records_for_csv(pred_group),
                }
            )
    return rows


def analyze_dataset(spec: DiagnosticInput) -> dict[str, Any]:
    missing = [str(path) for path in _input_paths(spec) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing diagnostic inputs: " + ", ".join(missing))

    gold_loaded = load_documents(spec.gold_path, dataset=spec.dataset)
    pred_loaded = load_documents(spec.pred_path, dataset=spec.dataset)
    eval_json = _load_json(spec.eval_json_path)
    train_manifest = _load_json(spec.train_manifest_path)
    pipeline_summary = _load_json(spec.pipeline_summary_path)
    eval_diagnostics = eval_json.get("diagnostics") if isinstance(eval_json.get("diagnostics"), dict) else {}

    bucket_report = compute_bucket_report(gold_loaded.documents, pred_loaded.documents, dataset=spec.dataset)
    failure_rows = compute_failure_rows(
        gold_loaded.documents,
        pred_loaded.documents,
        dataset=spec.dataset,
        eval_diagnostics=eval_diagnostics,
    )
    overall = compute_overall_report(gold_loaded.documents, pred_loaded.documents)
    consistency = _metric_consistency(overall, eval_json)

    return {
        "dataset": spec.dataset,
        "input_paths": {
            "gold": str(spec.gold_path),
            "pred": str(spec.pred_path),
            "eval_json": str(spec.eval_json_path),
            "train_manifest": str(spec.train_manifest_path),
            "pipeline_summary": str(spec.pipeline_summary_path),
            "pipeline_final_prediction": pipeline_summary.get("final_prediction"),
            "pipeline_run_root": pipeline_summary.get("run_root"),
        },
        "loader_diagnostics": {
            "gold": gold_loaded.diagnostics,
            "pred": pred_loaded.diagnostics,
        },
        "document_counts": {
            "gold": len(gold_loaded.documents),
            "pred": len(pred_loaded.documents),
        },
        "bucket_report": bucket_report,
        "failure_rows": failure_rows,
        "failure_summary": summarize_failures(failure_rows),
        "runtime_metrics": extract_runtime_metrics(spec.dataset, train_manifest),
        "computed_overall": overall,
        "metric_consistency": consistency,
        "metric_consistency_passed": (
            consistency["role_f1_delta"] <= 0.001 and consistency["exact_record_f1_delta"] <= 0.001
        ),
    }


def compute_overall_report(
    gold_documents: list[CanonicalDocument],
    pred_documents: list[CanonicalDocument],
) -> dict[str, float | int]:
    stats = OverallStats()
    gold_by_id = merge_documents(gold_documents)
    pred_by_id = merge_documents(pred_documents)

    for document_id in all_document_ids(gold_documents, pred_documents):
        gold_by_event = group_by_event_type(gold_by_id.get(document_id, []))
        pred_by_event = group_by_event_type(pred_by_id.get(document_id, []))
        for event_name in sorted(set(gold_by_event) | set(pred_by_event)):
            stats.add_group(gold_by_event.get(event_name, []), pred_by_event.get(event_name, []))
    return stats.to_report()


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts = {flag: sum(1 for row in rows if row.get(flag)) for flag in FAILURE_FLAG_COLUMNS}
    suggested_counts = Counter(str(row["suggested_error_type"]) for row in rows)
    bucket_counts = Counter(_bucket_label(int(row["gold_record_count"])) if int(row["gold_record_count"]) else "0" for row in rows)
    return {
        "failure_instance_count": len(rows),
        "flag_counts": flag_counts,
        "suggested_error_type_counts": dict(sorted(suggested_counts.items())),
        "gold_record_bucket_counts": dict(sorted(bucket_counts.items())),
    }


def extract_runtime_metrics(dataset: str, train_manifest: dict[str, Any]) -> dict[str, Any]:
    memory = train_manifest.get("torch_cuda_memory") if isinstance(train_manifest.get("torch_cuda_memory"), dict) else {}
    training_time = train_manifest.get("train_runtime", train_manifest.get("train_secs"))
    return {
        "dataset": dataset,
        "training_time_seconds": training_time if training_time is not None else "unavailable",
        "peak_gpu_memory_allocated_gb": memory.get("max_memory_allocated_gb", "unavailable"),
        "peak_gpu_memory_reserved_gb": memory.get("max_memory_reserved_gb", "unavailable"),
        "inference_speed_docs_per_second": "unavailable",
        "average_generated_tokens": "unavailable",
        "validation_time_seconds": "unavailable",
        "source_notes": (
            "training time and peak memory from training_manifest; "
            "inference speed, generated tokens, and validation time unavailable in copied metadata"
        ),
    }


def write_outputs(analyses: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "summary_json": out_dir / "summary.json",
        "record_count_buckets_csv": eval_dir / "record_count_buckets.csv",
        "record_count_buckets_tex": eval_dir / "record_count_buckets.tex",
        "exact_record_failures_csv": eval_dir / "exact_record_failures.csv",
        "exact_record_error_summary_tex": eval_dir / "exact_record_error_summary.tex",
        "exact_record_failure_cases_50_csv": eval_dir / "exact_record_failure_cases_50.csv",
        "runtime_metrics_csv": eval_dir / "runtime_metrics.csv",
    }

    bucket_rows = _bucket_csv_rows(analyses)
    failure_rows = [row for analysis in analyses for row in analysis["failure_rows"]]
    representative_rows = _representative_failure_rows(failure_rows, limit=50)
    runtime_rows = [analysis["runtime_metrics"] for analysis in analyses]

    _write_csv(artifacts["record_count_buckets_csv"], bucket_rows, _bucket_csv_fields())
    artifacts["record_count_buckets_tex"].write_text(render_bucket_latex(analyses) + "\n", encoding="utf-8")
    _write_csv(artifacts["exact_record_failures_csv"], failure_rows, _failure_csv_fields())
    artifacts["exact_record_error_summary_tex"].write_text(render_error_summary_latex(analyses) + "\n", encoding="utf-8")
    _write_csv(artifacts["exact_record_failure_cases_50_csv"], representative_rows, _manual_case_csv_fields())
    _write_csv(artifacts["runtime_metrics_csv"], runtime_rows, _runtime_csv_fields())

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_family": "unified_strict_record_count_and_exact_record_diagnostics",
        "outputs": {name: str(path) for name, path in artifacts.items()},
        "analysis": [_summary_analysis(analysis) for analysis in analyses],
    }
    artifacts["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {name: str(path) for name, path in artifacts.items()}


def render_bucket_latex(analyses: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Record-count bucket diagnostics on the seed-13 main runs. Buckets group document--event-type instances by gold same-type record count.}",
        r"\label{tab:record-count-buckets}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Gold rec. & Inst. & Role P. & Role R. & Role F1 & Exact / Count \\",
        r"\midrule",
    ]
    for analysis in analyses:
        dataset = _latex_escape(str(analysis["dataset"]).replace("-dev500", ""))
        for bucket in BUCKET_LABELS:
            row = analysis["bucket_report"]["buckets"][bucket]
            lines.append(
                " & ".join(
                    [
                        dataset,
                        _latex_bucket(bucket),
                        str(row["instances"]),
                        _fmt_pct(row["role_precision"]),
                        _fmt_pct(row["role_recall"]),
                        _fmt_pct(row["role_f1"]),
                        f"{_fmt_pct(row['exact_record_f1'])} / {_fmt_pct(row['record_count_accuracy'])}",
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_error_summary_latex(analyses: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Automatic Exact-Record failure diagnostics for seed-13 main runs.}",
        r"\label{tab:exact-record-error-summary}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Fail & Count & Dup. & Empty & Anchor & Schema \\",
        r"\midrule",
    ]
    for analysis in analyses:
        summary = analysis["failure_summary"]
        flags = summary["flag_counts"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(analysis["dataset"]).replace("-dev500", "")),
                    str(summary["failure_instance_count"]),
                    str(flags["count_mismatch_flag"]),
                    str(flags["duplicate_predicted_record_flag"]),
                    str(flags["empty_predicted_record_flag"]),
                    str(flags["missing_configured_anchor_flag"]),
                    str(flags["schema_parse_error_flag"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build seed-13 record-count and Exact-Record diagnostics.")
    parser.add_argument(
        "--input",
        action="append",
        nargs=6,
        metavar=("DATASET", "GOLD", "PRED", "EVAL_JSON", "TRAIN_MANIFEST", "PIPELINE_SUMMARY"),
        required=True,
        help="Dataset label plus gold, prediction, eval JSON, training manifest, and pipeline summary files.",
    )
    parser.add_argument("--out-dir", required=True, help="Output run directory for summary.json and eval/*.csv/*.tex.")
    args = parser.parse_args(argv)

    specs = [
        DiagnosticInput(
            dataset=dataset,
            gold_path=Path(gold_path),
            pred_path=Path(pred_path),
            eval_json_path=Path(eval_json_path),
            train_manifest_path=Path(train_manifest_path),
            pipeline_summary_path=Path(pipeline_summary_path),
        )
        for dataset, gold_path, pred_path, eval_json_path, train_manifest_path, pipeline_summary_path in args.input
    ]
    analyses = [analyze_dataset(spec) for spec in specs]
    artifacts = write_outputs(analyses, Path(args.out_dir))
    print(json.dumps({"outputs": artifacts}, ensure_ascii=False, indent=2))
    return 0


def _score_group(
    gold_group: list[CanonicalEventRecord],
    pred_group: list[CanonicalEventRecord],
) -> dict[str, Any]:
    pairs, _ = match_records(pred_group, gold_group)
    counts = Counts()
    exact_matches = 0
    used_pred = set()
    used_gold = set()

    for pred_index, gold_index in pairs:
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        pred_record = pred_group[pred_index]
        gold_record = gold_group[gold_index]
        counts.add(count_record_pair(pred_record, gold_record))
        exact_matches += int(exact_record_match(pred_record, gold_record))

    for pred_index, pred_record in enumerate(pred_group):
        if pred_index not in used_pred:
            counts.add(count_unmatched_pred(pred_record))
    for gold_index, gold_record in enumerate(gold_group):
        if gold_index not in used_gold:
            counts.add(count_unmatched_gold(gold_record))

    return {"counts": counts, "exact_matches": exact_matches, "matched_pairs": len(pairs)}


def _bucket_label(gold_count: int) -> str:
    if gold_count <= 1:
        return "1"
    if gold_count == 2:
        return "2"
    if gold_count == 3:
        return "3"
    return ">=4"


def _record_signature(record: CanonicalEventRecord) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    return (
        str(record.event_type),
        tuple((role, tuple(sorted(values))) for role, values in sorted(role_value_sets(record).items())),
    )


def _has_duplicate_predicted_record(pred_group: list[CanonicalEventRecord]) -> bool:
    signatures = [_record_signature(record) for record in pred_group]
    return len(signatures) != len(set(signatures))


def _missing_configured_anchor(
    event_name: str,
    gold_group: list[CanonicalEventRecord],
    pred_group: list[CanonicalEventRecord],
) -> bool:
    anchor_roles = ANCHOR_ROLES_BY_EVENT_TYPE.get(event_name, ())
    if not anchor_roles:
        return False
    gold_has_anchor = any(_record_has_any_role(record, anchor_roles) for record in gold_group)
    if not gold_has_anchor:
        return False
    if not pred_group:
        return True
    return any(not _record_has_any_role(record, anchor_roles) for record in pred_group)


def _record_has_any_role(record: CanonicalEventRecord, roles: tuple[str, ...]) -> bool:
    values_by_role = role_value_sets(record)
    return any(values_by_role.get(role) for role in roles)


def _has_positive_schema_or_parse_diagnostic(diagnostics: dict[str, Any]) -> bool:
    keys = (
        "parse_failure_count",
        "pred_parse_failure_count",
        "invalid_event_type_count",
        "invalid_role_count",
    )
    return any(int(diagnostics.get(key) or 0) > 0 for key in keys)


def _suggest_error_type(gold_count: int, pred_count: int, flags: dict[str, bool]) -> str:
    if flags["schema_parse_error_flag"]:
        return "schema_or_parse_error"
    if gold_count != pred_count:
        return "under_predicted_record_count" if pred_count < gold_count else "over_predicted_record_count"
    if flags["duplicate_predicted_record_flag"]:
        return "duplicate_predicted_record"
    if flags["empty_predicted_record_flag"]:
        return "empty_predicted_record"
    if flags["missing_configured_anchor_flag"]:
        return "missing_configured_anchor"
    return "role_value_mismatch"


def _metric_consistency(overall: dict[str, Any], eval_json: dict[str, Any]) -> dict[str, float | None]:
    eval_overall = eval_json.get("overall") if isinstance(eval_json.get("overall"), dict) else {}
    eval_diagnostics = eval_json.get("diagnostics") if isinstance(eval_json.get("diagnostics"), dict) else {}
    eval_role_f1 = eval_overall.get("f1")
    exact = eval_diagnostics.get("record_exact_match_count")
    exact_denominator = eval_diagnostics.get("validated_record_count")
    eval_exact_f1 = (2.0 * float(exact) / float(exact_denominator)) if exact is not None and exact_denominator else None
    computed_role_f1 = float(overall["role_f1"])
    computed_exact_f1 = float(overall["exact_record_f1"])
    return {
        "computed_role_f1": computed_role_f1,
        "eval_role_f1": float(eval_role_f1) if eval_role_f1 is not None else None,
        "role_f1_delta": abs(computed_role_f1 - float(eval_role_f1)) if eval_role_f1 is not None else float("inf"),
        "computed_exact_record_f1": computed_exact_f1,
        "eval_exact_record_f1": eval_exact_f1,
        "exact_record_f1_delta": abs(computed_exact_f1 - eval_exact_f1) if eval_exact_f1 is not None else float("inf"),
    }


def _bucket_csv_rows(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for bucket in BUCKET_LABELS:
            row = dict(analysis["bucket_report"]["buckets"][bucket])
            row.update({"dataset": analysis["dataset"], "gold_record_count_bucket": bucket})
            rows.append(row)
    return rows


def _bucket_csv_fields() -> list[str]:
    return [
        "dataset",
        "gold_record_count_bucket",
        "instances",
        "gold_record_count",
        "pred_record_count",
        "role_tp",
        "role_fp",
        "role_fn",
        "role_precision",
        "role_recall",
        "role_f1",
        "exact_record_matches",
        "exact_record_f1",
        "record_count_accuracy",
    ]


def _failure_csv_fields() -> list[str]:
    return [
        "dataset",
        "doc_id",
        "event_type",
        "gold_record_count",
        "pred_record_count",
        "exact_record_matches",
        "role_tp",
        "role_fp",
        "role_fn",
        *FAILURE_FLAG_COLUMNS,
        "automatic_error_flags",
        "suggested_error_type",
    ]


def _manual_case_csv_fields() -> list[str]:
    return [
        "dataset",
        "doc_id",
        "event_type",
        "gold_records",
        "pred_records",
        "automatic_error_flags",
        "suggested_error_type",
    ]


def _runtime_csv_fields() -> list[str]:
    return [
        "dataset",
        "training_time_seconds",
        "peak_gpu_memory_allocated_gb",
        "peak_gpu_memory_reserved_gb",
        "inference_speed_docs_per_second",
        "average_generated_tokens",
        "validation_time_seconds",
        "source_notes",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _representative_failure_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["dataset"], item["suggested_error_type"], item["doc_id"], item["event_type"])):
        bucket = _bucket_label(int(row["gold_record_count"])) if int(row["gold_record_count"]) else "0"
        rows_by_key[(str(row["dataset"]), str(row["suggested_error_type"]), bucket)].append(row)

    selected: list[dict[str, Any]] = []
    keys = sorted(rows_by_key)
    while keys and len(selected) < limit:
        next_keys = []
        for key in keys:
            group = rows_by_key[key]
            if group and len(selected) < limit:
                selected.append(group.pop(0))
            if group:
                next_keys.append(key)
        keys = next_keys

    return [_manual_case_row(row) for row in selected]


def _manual_case_row(row: dict[str, Any]) -> dict[str, Any]:
    gold_records = row.get("_gold_records", [])
    pred_records = row.get("_pred_records", [])
    return {
        "dataset": row["dataset"],
        "doc_id": row["doc_id"],
        "event_type": row["event_type"],
        "gold_records": gold_records,
        "pred_records": pred_records,
        "automatic_error_flags": row["automatic_error_flags"],
        "suggested_error_type": row["suggested_error_type"],
    }


def _records_for_csv(records: list[CanonicalEventRecord]) -> list[dict[str, Any]]:
    return [
        {
            "event_type": record.event_type,
            "arguments": {role: sorted(values) for role, values in sorted(role_value_sets(record).items())},
        }
        for record in records
    ]


def _summary_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": analysis["dataset"],
        "input_paths": analysis["input_paths"],
        "loader_diagnostics": analysis["loader_diagnostics"],
        "document_counts": analysis["document_counts"],
        "bucket_report": analysis["bucket_report"],
        "failure_summary": analysis["failure_summary"],
        "runtime_metrics": analysis["runtime_metrics"],
        "computed_overall": analysis["computed_overall"],
        "metric_consistency": analysis["metric_consistency"],
        "metric_consistency_passed": analysis["metric_consistency_passed"],
    }


def _input_paths(spec: DiagnosticInput) -> tuple[Path, ...]:
    return (
        spec.gold_path,
        spec.pred_path,
        spec.eval_json_path,
        spec.train_manifest_path,
        spec.pipeline_summary_path,
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _fmt_pct(value: float | int) -> str:
    return f"{float(value) * 100.0:.1f}"


def _latex_bucket(bucket: str) -> str:
    if bucket == ">=4":
        return r"$\geq$4"
    return bucket


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


if __name__ == "__main__":
    raise SystemExit(main())
