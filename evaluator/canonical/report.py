from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluator.canonical.stats import Counts

NORMALIZATION_POLICY = {
    "unicode": "NFKC",
    "whitespace": "strip_and_collapse",
    "empty_values": "ignored",
    "semantic_matching": False,
    "embedding_matching": False,
    "llm_judge": False,
}


def counts_dict_to_metrics(counts_by_key: dict[str, Counts]) -> dict[str, dict[str, float | int]]:
    return {key: counts.to_metrics() for key, counts in sorted(counts_by_key.items())}


def build_report(
    *,
    dataset: str,
    metric_family: str,
    overall: Counts,
    per_event: dict[str, Counts],
    subset_metrics: dict[str, Counts] | None,
    diagnostics: dict[str, Any] | None,
    matching_policy: dict[str, Any],
    input_paths: dict[str, str | None] | None = None,
    schema_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset": dataset,
        "metric_family": metric_family,
        "overall": overall.to_metrics(),
        "per_event": counts_dict_to_metrics(per_event),
        "subset_metrics": {
            "single_event": (subset_metrics or {}).get("single_event", Counts()).to_metrics(),
            "multi_event": (subset_metrics or {}).get("multi_event", Counts()).to_metrics(),
        },
        "diagnostics": diagnostics or {},
        "input_paths": input_paths or {"gold": None, "pred": None},
        "schema_path": schema_path,
        "normalization_policy": NORMALIZATION_POLICY,
        "matching_policy": matching_policy,
    }
    if extra:
        report.update(extra)
    return report


def add_diagnostics(*diagnostic_maps: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = defaultdict(int)
    for diagnostics in diagnostic_maps:
        if not diagnostics:
            continue
        for key, value in diagnostics.items():
            if isinstance(value, int):
                merged[key] += value
            else:
                merged[key] = value
    return dict(merged)
