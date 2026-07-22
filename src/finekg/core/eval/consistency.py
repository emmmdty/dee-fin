"""Global consistency diagnostics for a constructed event graph.

These quantify exactly the failure mode the relation stage targets: pairwise
predictions that, taken together, violate global structure. They are reported
before vs. after the consistency solver to show its effect.

- temporal: a strict "before"-style order must be acyclic and transitively
  closed; cycles are contradictions, the closure gap measures incompleteness.
- causal: a causal subgraph must be acyclic; cycles are contradictions.
- coreference: clusters are transitive by definition, so we measure how far the
  *predicted edges* are from being closed within their connected components.
"""

from __future__ import annotations

from finekg.core.graph import (
    coreference_clusters,
    edge_pair_set,
    find_cycles,
    transitive_closure_pairs,
)
from finekg.core.schema import STRICT_TEMPORAL_SUBTYPES, EventGraph, RelationType

__all__ = ["consistency_report"]


def _temporal_metrics(graph: EventGraph) -> dict[str, float]:
    # Only strict-order subtypes (BEFORE) are subject to acyclicity/closure;
    # counting OVERLAP-style edges here would report false contradictions.
    strict = [
        e
        for e in graph.edges_of_type(RelationType.TEMPORAL)
        if e.subtype in STRICT_TEMPORAL_SUBTYPES
    ]
    closure = transitive_closure_pairs(
        graph, RelationType.TEMPORAL, subtypes=STRICT_TEMPORAL_SUBTYPES
    )
    observed = edge_pair_set(strict)
    cycles = find_cycles(graph, RelationType.TEMPORAL, subtypes=STRICT_TEMPORAL_SUBTYPES)
    missing = closure - observed
    return {
        "temporal_cycle_count": float(len(cycles)),
        "temporal_closure_gap": (len(missing) / len(closure)) if closure else 0.0,
    }


def _causal_metrics(graph: EventGraph) -> dict[str, float]:
    cycles = find_cycles(graph, RelationType.CAUSAL)
    return {"causal_cycle_count": float(len(cycles))}


def _coreference_metrics(graph: EventGraph) -> dict[str, float]:
    clusters = coreference_clusters(graph)
    observed = {
        (e.head_id, e.tail_id) for e in graph.edges_of_type(RelationType.COREFERENCE)
    } | {(e.tail_id, e.head_id) for e in graph.edges_of_type(RelationType.COREFERENCE)}
    # Fully-closed clusters would contain every within-cluster ordered pair.
    expected = sum(len(c) * (len(c) - 1) for c in clusters)
    present = len(observed)
    gap = 1.0 - (present / expected) if expected else 0.0
    return {"coref_closure_gap": max(0.0, gap)}


def consistency_report(graph: EventGraph) -> dict[str, float]:
    """All consistency diagnostics in one flat dict (lower is better)."""
    report: dict[str, float] = {}
    report.update(_temporal_metrics(graph))
    report.update(_causal_metrics(graph))
    report.update(_coreference_metrics(graph))
    return report
