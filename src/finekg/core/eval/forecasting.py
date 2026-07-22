"""Legacy-compatible temporal link-prediction metrics: MRR and Hits@k.

Given gold answers and ranked candidate lists (one per query), compute Mean
Reciprocal Rank and Hits@k — the standard TKG-forecasting metrics used by
ICEWS / FinDKG. They are retained for archived TKG artifacts; v4 CGEP metrics live
in ``finekg.succession.metrics``. Queries whose gold answer is absent count as
rank = infinity (reciprocal rank 0), matching the conventional "raw" setting.
"""

from __future__ import annotations

from collections.abc import Sequence

from finekg.core.schema import Prediction

__all__ = ["rank_of", "mrr_hits", "evaluate_predictions"]


def rank_of(gold: str, ranked_objects: Sequence[str]) -> int | None:
    """1-based rank of `gold` in `ranked_objects`, or None if absent."""
    for i, obj in enumerate(ranked_objects):
        if obj == gold:
            return i + 1
    return None


def mrr_hits(
    gold_answers: Sequence[str],
    rankings: Sequence[Sequence[str]],
    ks: Sequence[int] = (1, 3, 10),
    filter_sets: Sequence[set[str]] | None = None,
) -> dict[str, float]:
    """MRR and Hits@k over aligned (gold, ranking) pairs.

    With ``filter_sets`` (the **time-aware filtered** TKG protocol), each query's
    ``filter_sets[i]`` lists *other* objects known true for the same (subject,
    relation, timestamp); they are removed from ``rankings[i]`` before locating
    the gold (which is never removed), so competing correct answers don't push the
    gold down. Filtered MRR is the standard reported by RE-GCN/TiRGN/etc.; the raw
    setting (``filter_sets=None``) keeps every candidate and scores lower.
    """
    if len(gold_answers) != len(rankings):
        raise ValueError("gold_answers and rankings must be the same length")
    if filter_sets is not None and len(filter_sets) != len(gold_answers):
        raise ValueError("filter_sets must align with gold_answers")
    n = len(gold_answers)
    if n == 0:
        return {"mrr": 0.0, **{f"hits@{k}": 0.0 for k in ks}}

    reciprocal = 0.0
    hits = {k: 0 for k in ks}
    for i, (gold, ranking) in enumerate(zip(gold_answers, rankings, strict=True)):
        if filter_sets is not None and filter_sets[i]:
            blocked = filter_sets[i]
            ranking = [o for o in ranking if o == gold or o not in blocked]
        r = rank_of(gold, ranking)
        if r is None:
            continue
        reciprocal += 1.0 / r
        for k in ks:
            if r <= k:
                hits[k] += 1
    out = {"mrr": reciprocal / n}
    out.update({f"hits@{k}": hits[k] / n for k in ks})
    return out


def evaluate_predictions(
    predictions: Sequence[Prediction],
    gold_answers: Sequence[str],
    ks: Sequence[int] = (1, 3, 10),
    filter_sets: Sequence[set[str]] | None = None,
) -> dict[str, float]:
    """Convenience wrapper computing MRR/Hits@k from `Prediction` objects.

    Pass ``filter_sets`` for the time-aware filtered protocol (see `mrr_hits`).
    """
    rankings = [[c.object for c in p.ranked] for p in predictions]
    return mrr_hits(gold_answers, rankings, ks, filter_sets=filter_sets)
