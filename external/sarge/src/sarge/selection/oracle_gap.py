from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_oracle_gap_rows(
    *,
    selected_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_doc = {str(row.get("doc_id", "")): str(row.get("candidate_id", "")) for row in selected_rows}
    rewards_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        rewards_by_doc[str(row.get("doc_id", ""))].append(row)

    rows: list[dict[str, Any]] = []
    for doc_id in sorted(rewards_by_doc):
        rewards = rewards_by_doc[doc_id]
        greedy = min(rewards, key=lambda row: (int(row.get("candidate_index", 0)), str(row.get("candidate_id", ""))))
        oracle = max(rewards, key=lambda row: (float(row.get("reward", 0.0)), -int(row.get("candidate_index", 0))))
        selected_candidate_id = selected_by_doc.get(doc_id)
        selected = next(
            (row for row in rewards if str(row.get("candidate_id", "")) == selected_candidate_id),
            greedy,
        )
        greedy_score = float(greedy.get("reward", 0.0))
        selected_score = float(selected.get("reward", 0.0))
        oracle_score = float(oracle.get("reward", 0.0))
        rows.append(
            {
                "doc_id": doc_id,
                "greedy_candidate_id": str(greedy.get("candidate_id", "")),
                "selected_candidate_id": str(selected.get("candidate_id", "")),
                "oracle_candidate_id": str(oracle.get("candidate_id", "")),
                "greedy_score": greedy_score,
                "selected_score": selected_score,
                "oracle_best_score": oracle_score,
                "oracle_gap": round(oracle_score - selected_score, 12),
                "mrs_gain": round(selected_score - greedy_score, 12),
            }
        )
    return rows


def summarize_oracle_gap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "doc_count": 0,
            "mean_greedy_score": 0.0,
            "mean_selected_score": 0.0,
            "mean_oracle_best_score": 0.0,
            "mean_oracle_gap": 0.0,
            "mean_mrs_gain": 0.0,
        }
    return {
        "doc_count": len(rows),
        "mean_greedy_score": _mean(rows, "greedy_score"),
        "mean_selected_score": _mean(rows, "selected_score"),
        "mean_oracle_best_score": _mean(rows, "oracle_best_score"),
        "mean_oracle_gap": _mean(rows, "oracle_gap"),
        "mean_mrs_gain": _mean(rows, "mrs_gain"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
