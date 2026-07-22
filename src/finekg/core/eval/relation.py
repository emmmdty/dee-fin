"""Precision / recall / F1 for event relation extraction.

Edges are matched by their identity key (head, tail, relation_type, subtype).
We report per-relation-family scores and a micro average over all families,
which is the standard reporting for MAVEN-ERE-style benchmarks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from finekg.core.graph import close_pairs
from finekg.core.schema import STRICT_TEMPORAL_SUBTYPES, RelationEdge, RelationType

__all__ = ["PRF", "relation_prf"]

EdgeKey = tuple[str, str, str, str]


class PRF(dict):
    """A small dict subclass holding precision/recall/f1 plus match counts."""

    @classmethod
    def from_counts(cls, tp: int, n_pred: int, n_gold: int) -> PRF:
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return cls(
            precision=precision, recall=recall, f1=f1, tp=tp, n_pred=n_pred, n_gold=n_gold
        )


def _normalize(edge: RelationEdge) -> tuple[str, str, str, str]:
    """Identity key; symmetric relations are made order-invariant."""
    head, tail = edge.head_id, edge.tail_id
    if not edge.directed and head > tail:
        head, tail = tail, head
    return (head, tail, edge.relation_type.value, edge.subtype)


def _edge_keys(edges: Iterable[RelationEdge], temporal_closure: bool) -> set[EdgeKey]:
    """Normalized identity keys, optionally closing strict-order temporal edges.

    With ``temporal_closure`` the directed strict-temporal (BEFORE) edges are
    transitively closed and collapsed onto a canonical ``BEFORE`` subtype, so a
    sparse but correct ordering chain earns credit for every implied pair — the
    fair comparison against MAVEN-ERE gold, which ships already transitively
    closed. Non-strict temporal (OVERLAP …) and other families are untouched.
    """
    if not temporal_closure:
        return {_normalize(e) for e in edges}
    keys: set[EdgeKey] = set()
    strict_pairs: list[tuple[str, str]] = []
    for e in edges:
        if (
            e.relation_type is RelationType.TEMPORAL
            and e.directed
            and e.subtype in STRICT_TEMPORAL_SUBTYPES
            and e.head_id != e.tail_id
        ):
            strict_pairs.append((e.head_id, e.tail_id))
        else:
            keys.add(_normalize(e))
    for h, t in close_pairs(strict_pairs):
        keys.add((h, t, RelationType.TEMPORAL.value, "BEFORE"))
    return keys


def relation_prf(
    predicted: Iterable[RelationEdge],
    gold: Iterable[RelationEdge],
    *,
    temporal_closure: bool = False,
) -> dict[str, PRF]:
    """Per-type and micro P/R/F1.

    Returns a dict keyed by relation family value (``"temporal"`` …) plus a
    ``"micro"`` entry aggregating across families. Pass ``temporal_closure=True``
    to score strict-order temporal relations after transitive closure (see
    `_edge_keys`); the default is exact pairwise matching.
    """
    pred_by_type: dict[str, set[EdgeKey]] = defaultdict(set)
    gold_by_type: dict[str, set[EdgeKey]] = defaultdict(set)
    for k in _edge_keys(predicted, temporal_closure):
        pred_by_type[k[2]].add(k)
    for k in _edge_keys(gold, temporal_closure):
        gold_by_type[k[2]].add(k)

    results: dict[str, PRF] = {}
    tp_all = n_pred_all = n_gold_all = 0
    for rel in RelationType:
        p, g = pred_by_type.get(rel.value, set()), gold_by_type.get(rel.value, set())
        if not p and not g:
            continue
        tp = len(p & g)
        results[rel.value] = PRF.from_counts(tp, len(p), len(g))
        tp_all += tp
        n_pred_all += len(p)
        n_gold_all += len(g)

    results["micro"] = PRF.from_counts(tp_all, n_pred_all, n_gold_all)
    return results
