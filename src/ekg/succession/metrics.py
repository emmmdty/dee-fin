"""Ranking metrics for successor-event prediction, in SeDGPL's exact dialect.

`tools.calculate` ranks the gold answer like this::

    predtCandi = prediction[i][candiSet[i]].tolist()
    label      = candiSet[i].index(labels[i])
    labelScore = predtCandi[label]
    predtCandi.sort(reverse=True)
    rank       = predtCandi.index(labelScore)

`list.index` on the descending list returns the *first* slot holding that score,
so every candidate tied with gold is ranked behind it. That is an optimistic
tie-break, and it is not cosmetic: SeDGPL scores a candidate by the token id of
its mention string, and candidates sharing a mention therefore score identically.
In its released ESC build only ~178.0 of 256 candidates are distinct answers, and
gold shares its string with another candidate in 468 of 1192 instances. Reporting
`strict_rank` instead would move the published numbers.

So `sedgpl_rank` reproduces the original and `strict_rank` states what the
optimistic count hides. Report both; never silently swap one for the other.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["HITS_AT", "cgep_metrics", "sedgpl_rank", "strict_rank"]

HITS_AT = (1, 3, 10, 20, 50)


def sedgpl_rank(scores: Sequence[float], label: int) -> int:
    """0-based rank of `label`, ties resolved in gold's favour (SeDGPL's rule)."""
    gold = scores[label]
    return sum(1 for score in scores if score > gold)


def strict_rank(scores: Sequence[float], label: int) -> int:
    """0-based rank of `label`, ties resolved against gold (the pessimistic bound)."""
    gold = scores[label]
    return sum(
        1 for i, score in enumerate(scores) if score > gold or (score == gold and i != label)
    )


def cgep_metrics(ranks: Sequence[int], hits_at: Sequence[int] = HITS_AT) -> dict[str, float]:
    """MRR and Hit@k over 0-based ranks. Empty input yields zeros, not NaNs."""
    if not ranks:
        return {"mrr": 0.0, **{f"hits@{k}": 0.0 for k in hits_at}, "n": 0.0}
    n = len(ranks)
    return {
        "mrr": sum(1.0 / (rank + 1) for rank in ranks) / n,
        **{f"hits@{k}": sum(rank < k for rank in ranks) / n for k in hits_at},
        "n": float(n),
    }
