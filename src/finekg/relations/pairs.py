"""Pair-classification harness: same-setting comparability for relation models.

Published MAVEN-ERE numbers (supervised pair classifiers, e.g. RoBERTa micro F1
51.8) assume the *pair-classification* setting: gold mentions are given and the
model labels every candidate mention pair. Our generative extractors emit edge
lists instead, so their raw `relation_prf` is not apples-to-apples. This module
is the bridge: it projects any edge list onto the candidate-pair universe of a
document and scores per-pair labels, so a generative LLM, a discriminative
encoder and the published baselines are read off the same ruler.

Three deliberate semantics:

- **Universe** = all ordered mention pairs (both directions; symmetric
  coreference labels both), optionally windowed by textual distance. Mention
  order comes from trigger spans (sentence, char), *not* node list order — the
  loader appends mentions grouped by event.
- **Hallucinations count**: a predicted edge whose endpoint is not a gold
  mention can never match, so it stays in ``n_pred`` (a false positive) *and*
  is reported separately — the generative failure mode made measurable.
- **Window diagnostics** separate the structural recall ceiling of windowed
  extraction (`window_recall_ceiling`, gold pairs a K-mention window can ever
  see) from genuine model misses; `pair_prf(max_distance=...)` scores the
  windowed universe for like-for-like comparison with window-limited models.

Pure Python / CPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finekg.core.eval.relation import PRF
from finekg.core.schema import RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument

__all__ = [
    "PairExample",
    "mention_order",
    "candidate_pairs",
    "edges_to_pair_labels",
    "pair_prf",
    "pair_examples",
    "window_recall_ceiling",
]

_FAMILIES = (
    RelationType.COREFERENCE,
    RelationType.TEMPORAL,
    RelationType.CAUSAL,
    RelationType.SUBEVENT,
)
_COREF_LABEL = "COREF"

PairKey = tuple[str, str]


@dataclass(frozen=True)
class PairExample:
    """One candidate pair with its gold labels (one subtype per family)."""

    doc_id: str
    head_id: str
    tail_id: str
    distance: int
    labels: dict[str, str]


def mention_order(doc: RelationDocument) -> dict[str, int]:
    """Mention id -> textual position (sentence, char start of the trigger)."""

    def key(node) -> tuple[int, int, str]:
        span = node.trigger_evidence[0] if node.trigger_evidence else None
        sent = span.sent_id if span is not None and span.sent_id is not None else 10**9
        start = span.char_start if span is not None else 10**9
        return (sent, start, node.event_id)

    ordered = sorted(doc.nodes, key=key)
    return {node.event_id: i for i, node in enumerate(ordered)}


def candidate_pairs(doc: RelationDocument, max_distance: int | None = None) -> list[PairKey]:
    """All ordered mention pairs, optionally capped by textual distance."""
    order = mention_order(doc)
    ids = sorted(order, key=order.__getitem__)
    pairs: list[PairKey] = []
    for i, head in enumerate(ids):
        for j, tail in enumerate(ids):
            if i == j:
                continue
            if max_distance is not None and abs(i - j) > max_distance:
                continue
            pairs.append((head, tail))
    return pairs


def edges_to_pair_labels(
    edges: Iterable[RelationEdge], *, family: RelationType
) -> dict[PairKey, str]:
    """Project one family's edges onto pair labels (confidence-max dedup).

    Directed edges label their (head, tail) key; symmetric edges label both
    directions. Duplicate labels for a key keep the highest-confidence one —
    the projection adapter that makes generative edge lists pair-comparable.
    """
    best: dict[PairKey, tuple[float, str]] = {}
    for edge in edges:
        if edge.relation_type is not family:
            continue
        label = _COREF_LABEL if family is RelationType.COREFERENCE else edge.subtype
        confidence = float(edge.confidence)
        keys: tuple[PairKey, ...] = ((edge.head_id, edge.tail_id),)
        if not edge.directed:
            keys = ((edge.head_id, edge.tail_id), (edge.tail_id, edge.head_id))
        for key in keys:
            if key not in best or confidence > best[key][0]:
                best[key] = (confidence, label)
    return {key: label for key, (_, label) in best.items()}


def pair_prf(
    predicted: Iterable[RelationEdge],
    doc: RelationDocument,
    *,
    max_distance: int | None = None,
) -> dict:
    """Per-family and micro P/R/F1 in the pair-classification setting.

    Returns family PRFs plus ``"micro"`` and a ``"diagnostics"`` entry:
    ``hallucinated_pred_pairs`` (endpoint not a gold mention; penalised in
    precision), ``out_of_window_gold`` / ``out_of_window_pred`` (excluded from
    the windowed universe) and ``n_universe``.
    """
    predicted = list(predicted)
    universe = set(candidate_pairs(doc, max_distance))
    known = {node.event_id for node in doc.nodes}

    results: dict = {}
    hallucinated = out_gold = out_pred = 0
    micro_tp = micro_pred = micro_gold = 0
    for family in _FAMILIES:
        pred = edges_to_pair_labels(predicted, family=family)
        gold = edges_to_pair_labels(doc.gold_edges, family=family)

        ghost_keys = {k for k in pred if not (k[0] in known and k[1] in known)}
        pred_out = {k for k in pred if k not in ghost_keys and k not in universe}
        gold_out = {k for k in gold if k not in universe}
        hallucinated += len(ghost_keys)
        out_pred += len(pred_out)
        out_gold += len(gold_out)

        scored_pred = {k: v for k, v in pred.items() if k in universe or k in ghost_keys}
        scored_gold = {k: v for k, v in gold.items() if k in universe}
        tp = sum(1 for k, v in scored_pred.items() if scored_gold.get(k) == v)
        results[family.value] = PRF.from_counts(tp, len(scored_pred), len(scored_gold))
        micro_tp += tp
        micro_pred += len(scored_pred)
        micro_gold += len(scored_gold)

    results["micro"] = PRF.from_counts(micro_tp, micro_pred, micro_gold)
    results["diagnostics"] = {
        "hallucinated_pred_pairs": hallucinated,
        "out_of_window_gold": out_gold,
        "out_of_window_pred": out_pred,
        "n_universe": len(universe),
    }
    return results


def pair_examples(doc: RelationDocument, max_distance: int | None = None) -> list[PairExample]:
    """The candidate universe with gold labels — pair-classifier training rows."""
    order = mention_order(doc)
    by_family = {
        family: edges_to_pair_labels(doc.gold_edges, family=family) for family in _FAMILIES
    }
    examples: list[PairExample] = []
    for head, tail in candidate_pairs(doc, max_distance):
        labels = {
            family.value: table[(head, tail)]
            for family, table in by_family.items()
            if (head, tail) in table
        }
        examples.append(
            PairExample(
                doc_id=doc.doc_id,
                head_id=head,
                tail_id=tail,
                distance=abs(order[head] - order[tail]),
                labels=labels,
            )
        )
    return examples


def window_recall_ceiling(
    docs: Iterable[RelationDocument], window_events: int
) -> dict[str, float]:
    """Share of gold edges a window of ``window_events`` mentions can ever see.

    The structural recall ceiling of windowed (generative) extraction: a gold
    pair farther apart than the window is unreachable no matter how good the
    model — report it as a ceiling, not a model miss.
    """
    reachable = total = 0
    for doc in docs:
        order = mention_order(doc)
        for edge in doc.gold_edges:
            total += 1
            head, tail = order.get(edge.head_id), order.get(edge.tail_id)
            if head is None or tail is None:
                continue
            reachable += int(abs(head - tail) < window_events)
    return {
        "reachable_gold": reachable,
        "total_gold": total,
        "ceiling": reachable / total if total else 0.0,
    }
