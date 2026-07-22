from __future__ import annotations

from pathlib import Path

from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord

from sarge.experiments.record_count_buckets import (
    compute_bucket_metrics,
    compute_bucket_report,
    compute_failure_rows,
    compute_overall_report,
    extract_runtime_metrics,
    summarize_failures,
    write_outputs,
)


def _record(doc_id: str, event_type: str, value: str, role: str = "role") -> CanonicalEventRecord:
    return CanonicalEventRecord(
        document_id=doc_id,
        event_type=event_type,
        arguments={role: [value]},
    )


def test_compute_bucket_metrics_separates_same_event_record_counts() -> None:
    gold = [
        CanonicalDocument("doc-1", [_record("doc-1", "Event", "x")]),
        CanonicalDocument("doc-2", [_record("doc-2", "Event", "a"), _record("doc-2", "Event", "b")]),
        CanonicalDocument(
            "doc-3",
            [_record("doc-3", "Event", "u"), _record("doc-3", "Event", "v"), _record("doc-3", "Event", "w")],
        ),
        CanonicalDocument(
            "doc-4",
            [
                _record("doc-4", "Event", "p"),
                _record("doc-4", "Event", "q"),
                _record("doc-4", "Event", "r"),
                _record("doc-4", "Event", "s"),
            ],
        ),
    ]
    pred = [
        CanonicalDocument("doc-1", [_record("doc-1", "Event", "x")]),
        CanonicalDocument("doc-2", [_record("doc-2", "Event", "a"), _record("doc-2", "Event", "c")]),
        CanonicalDocument("doc-3", [_record("doc-3", "Event", "u"), _record("doc-3", "Event", "v")]),
        CanonicalDocument(
            "doc-4",
            [
                _record("doc-4", "Event", "p"),
                _record("doc-4", "Event", "q"),
                _record("doc-4", "Event", "r"),
                _record("doc-4", "Event", "s"),
            ],
        ),
    ]

    report = compute_bucket_metrics(gold, pred)

    assert report["1"]["group_count"] == 1
    assert report["1"]["role_f1"] == 1.0
    assert report["1"]["exact_record_f1"] == 1.0
    assert report["1"]["record_count_accuracy"] == 1.0

    assert report["2"]["group_count"] == 1
    assert report["2"]["role_f1"] == 0.5
    assert report["2"]["exact_record_f1"] == 0.5
    assert report["2"]["record_count_accuracy"] == 1.0

    assert report["3"]["group_count"] == 1
    assert report["3"]["role_f1"] == 0.8
    assert report["3"]["exact_record_f1"] == 0.8
    assert report["3"]["record_count_accuracy"] == 0.0

    assert report[">=4"]["group_count"] == 1
    assert report[">=4"]["role_f1"] == 1.0
    assert report[">=4"]["exact_record_f1"] == 1.0
    assert report[">=4"]["record_count_accuracy"] == 1.0


def test_compute_failure_rows_sets_automatic_flags() -> None:
    gold = [
        CanonicalDocument(
            "doc-empty",
            [CanonicalEventRecord("doc-empty", "质押", {"质押方": ["甲公司"], "质押物": ["A股"]})],
        ),
        CanonicalDocument(
            "doc-under",
            [_record("doc-under", "Event", "a"), _record("doc-under", "Event", "b")],
        ),
        CanonicalDocument("doc-over", [_record("doc-over", "Event", "z")]),
    ]
    pred = [
        CanonicalDocument("doc-empty", [CanonicalEventRecord("doc-empty", "质押", {})]),
        CanonicalDocument("doc-under", [_record("doc-under", "Event", "a")]),
        CanonicalDocument("doc-over", [_record("doc-over", "Event", "z"), _record("doc-over", "Event", "z")]),
    ]

    rows = compute_failure_rows(gold, pred, dataset="Demo")
    by_doc = {row["doc_id"]: row for row in rows}

    assert by_doc["doc-empty"]["count_mismatch_flag"] is False
    assert by_doc["doc-empty"]["empty_predicted_record_flag"] is True
    assert by_doc["doc-empty"]["missing_configured_anchor_flag"] is True
    assert by_doc["doc-empty"]["suggested_error_type"] == "empty_predicted_record"

    assert by_doc["doc-under"]["count_mismatch_flag"] is True
    assert by_doc["doc-under"]["suggested_error_type"] == "under_predicted_record_count"

    assert by_doc["doc-over"]["count_mismatch_flag"] is True
    assert by_doc["doc-over"]["duplicate_predicted_record_flag"] is True
    assert by_doc["doc-over"]["suggested_error_type"] == "over_predicted_record_count"


def test_write_outputs_exports_requested_artifacts(tmp_path: Path) -> None:
    gold = [CanonicalDocument("doc-1", [_record("doc-1", "Event", "x")])]
    pred = [CanonicalDocument("doc-1", [_record("doc-1", "Event", "y")])]
    failures = compute_failure_rows(gold, pred, dataset="Demo")
    analysis = {
        "dataset": "Demo",
        "input_paths": {},
        "loader_diagnostics": {},
        "document_counts": {"gold": 1, "pred": 1},
        "bucket_report": compute_bucket_report(gold, pred, dataset="Demo"),
        "failure_rows": failures,
        "failure_summary": summarize_failures(failures),
        "runtime_metrics": extract_runtime_metrics(
            "Demo",
            {
                "train_runtime": 12.5,
                "torch_cuda_memory": {
                    "max_memory_allocated_gb": 3.25,
                    "max_memory_reserved_gb": 4.5,
                },
            },
        ),
        "computed_overall": compute_overall_report(gold, pred),
        "metric_consistency": {"role_f1_delta": 0.0, "exact_record_f1_delta": 0.0},
        "metric_consistency_passed": True,
    }

    artifacts = write_outputs([analysis], tmp_path / "sarge_record_diagnostics_seed13_20260525")

    for path in artifacts.values():
        assert Path(path).exists()
    assert "gold_record_count_bucket" in Path(artifacts["record_count_buckets_csv"]).read_text(encoding="utf-8")
    assert "suggested_error_type" in Path(artifacts["exact_record_failure_cases_50_csv"]).read_text(encoding="utf-8")
    assert "unavailable" in Path(artifacts["runtime_metrics_csv"]).read_text(encoding="utf-8")
