from __future__ import annotations

from functools import lru_cache
from typing import Any

from evaluator.canonical.stats import record_overlap
from evaluator.canonical.types import CanonicalEventRecord


def match_records(
    pred_records: list[CanonicalEventRecord],
    gold_records: list[CanonicalEventRecord],
    *,
    exact_limit: int = 12,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Return pred/gold index pairs maximizing strict role-value overlap."""
    if not pred_records or not gold_records:
        return [], {"algorithm": "none", "fallback_used": False}

    if len(pred_records) <= exact_limit and len(gold_records) <= exact_limit:
        return _match_exact_dp(pred_records, gold_records), {"algorithm": "deterministic_exact_dp", "fallback_used": False}

    scipy_pairs = _match_with_scipy(pred_records, gold_records)
    if scipy_pairs is not None:
        return scipy_pairs, {"algorithm": "scipy_linear_sum_assignment", "fallback_used": False}

    return _match_greedy(pred_records, gold_records), {"algorithm": "deterministic_greedy_fallback", "fallback_used": True}


def _match_exact_dp(pred_records: list[CanonicalEventRecord], gold_records: list[CanonicalEventRecord]) -> list[tuple[int, int]]:
    overlaps = [
        [record_overlap(pred_record, gold_record) for gold_record in gold_records]
        for pred_record in pred_records
    ]

    @lru_cache(maxsize=None)
    def solve(pred_index: int, used_mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if pred_index >= len(pred_records):
            return 0, ()

        best_score, best_pairs = solve(pred_index + 1, used_mask)
        for gold_index in range(len(gold_records)):
            if used_mask & (1 << gold_index):
                continue
            overlap = overlaps[pred_index][gold_index]
            if overlap <= 0:
                continue
            tail_score, tail_pairs = solve(pred_index + 1, used_mask | (1 << gold_index))
            candidate_score = overlap + tail_score
            candidate_pairs = ((pred_index, gold_index),) + tail_pairs
            if candidate_score > best_score or (candidate_score == best_score and candidate_pairs < best_pairs):
                best_score = candidate_score
                best_pairs = candidate_pairs
        return best_score, best_pairs

    return list(solve(0, 0)[1])


def _match_with_scipy(
    pred_records: list[CanonicalEventRecord],
    gold_records: list[CanonicalEventRecord],
) -> list[tuple[int, int]] | None:
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None

    size = max(len(pred_records), len(gold_records))
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for pred_index, pred_record in enumerate(pred_records):
        for gold_index, gold_record in enumerate(gold_records):
            matrix[pred_index][gold_index] = -record_overlap(pred_record, gold_record)

    row_indices, col_indices = linear_sum_assignment(matrix)
    pairs = []
    for pred_index, gold_index in zip(row_indices, col_indices):
        if pred_index < len(pred_records) and gold_index < len(gold_records):
            if record_overlap(pred_records[pred_index], gold_records[gold_index]) > 0:
                pairs.append((int(pred_index), int(gold_index)))
    return sorted(pairs)


def _match_greedy(pred_records: list[CanonicalEventRecord], gold_records: list[CanonicalEventRecord]) -> list[tuple[int, int]]:
    unused_gold = set(range(len(gold_records)))
    pairs: list[tuple[int, int]] = []
    for pred_index, pred_record in enumerate(pred_records):
        best_gold = None
        best_overlap = 0
        for gold_index in sorted(unused_gold):
            overlap = record_overlap(pred_record, gold_records[gold_index])
            if overlap > best_overlap:
                best_overlap = overlap
                best_gold = gold_index
        if best_gold is not None:
            pairs.append((pred_index, best_gold))
            unused_gold.remove(best_gold)
    return pairs
