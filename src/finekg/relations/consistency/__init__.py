"""Global consistency solvers for the constructed event graph.

The relation extractor predicts edges (largely) independently, so the raw graph
can contain causal loops, temporal contradictions, and unclosed coreference.
A consistency solver repairs these into a globally coherent graph:

- causal & temporal: break directed cycles by dropping the lowest-confidence
  edge in each cycle until the subgraph is acyclic (temporal acyclicity is
  enforced over the strict-order subtypes only — OVERLAP-style edges carry no
  order and pass through);
- temporal: optionally add transitively-implied BEFORE edges (closure over the
  strict-order subtypes);
- coreference: deduplicate and (optionally) close clusters.

`identity` is a no-op used for the "-consistency" ablation. New strategies
(e.g. an ILP solver) register here without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx

from finekg.core.registry import Registry
from finekg.core.schema import (
    STRICT_TEMPORAL_SUBTYPES,
    EventGraph,
    RelationEdge,
    RelationType,
)

__all__ = [
    "ConsistencySolver",
    "consistency_solvers",
    "GreedyConsistencySolver",
    "IdentityConsistencySolver",
]


class ConsistencySolver(ABC):
    @abstractmethod
    def solve(self, graph: EventGraph) -> EventGraph:
        """Return a new graph with consistent relations."""


consistency_solvers: Registry[ConsistencySolver] = Registry("consistency_solver")


@consistency_solvers.register("identity")
class IdentityConsistencySolver(ConsistencySolver):
    """No-op solver (ablation baseline)."""

    def solve(self, graph: EventGraph) -> EventGraph:
        return graph.model_copy(deep=True)


def _dedup_by_key(edges: list[RelationEdge]) -> list[RelationEdge]:
    """Collapse exact duplicates (same identity key) to the most confident copy.

    Parallel edges that differ in subtype (e.g. CAUSE and PRECONDITION on the
    same pair) are distinct keys and survive — only true duplicates collapse.
    """
    best: dict[tuple[str, str, str, str], RelationEdge] = {}
    for e in edges:
        k = e.key()
        if k not in best or e.confidence > best[k].confidence:
            best[k] = e
    return list(best.values())


def _break_cycles(edges: list[RelationEdge]) -> list[RelationEdge]:
    """Greedily drop the weakest pair in each directed cycle until acyclic.

    Cycle structure lives at the (head, tail) pair level, so parallel edges of
    different subtypes on one pair stand or fall together (the pair's strength
    is its strongest edge); they are not silently collapsed into one edge.
    """
    by_pair: dict[tuple[str, str], list[RelationEdge]] = {}
    for e in _dedup_by_key(edges):
        by_pair.setdefault((e.head_id, e.tail_id), []).append(e)
    g: nx.DiGraph = nx.DiGraph()
    for (h, t), parallel in by_pair.items():
        g.add_edge(h, t, confidence=max(e.confidence for e in parallel))
    while True:
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda c: g[c[0]][c[1]]["confidence"])
        g.remove_edge(weakest[0], weakest[1])
        by_pair.pop((weakest[0], weakest[1]), None)
    return [e for h, t in g.edges() for e in by_pair[(h, t)]]


@consistency_solvers.register("greedy")
class GreedyConsistencySolver(ConsistencySolver):
    def __init__(self, close_temporal: bool = True) -> None:
        self.close_temporal = close_temporal

    def solve(self, graph: EventGraph) -> EventGraph:
        out = EventGraph(nodes=dict(graph.nodes), metadata=dict(graph.metadata))

        # Coreference: keep the most confident undirected edge per unordered
        # pair (the same winner rule `_dedup_by_key` applies to typed edges).
        best_coref: dict[frozenset[str], RelationEdge] = {}
        for e in graph.edges_of_type(RelationType.COREFERENCE):
            pair = frozenset((e.head_id, e.tail_id))
            if pair not in best_coref or e.confidence > best_coref[pair].confidence:
                best_coref[pair] = e
        for e in best_coref.values():
            out.add_edge(e)

        # Causal: enforce acyclicity (cycles across CAUSE/PRECONDITION are
        # contradictions regardless of subtype).
        for e in _break_cycles(graph.edges_of_type(RelationType.CAUSAL)):
            out.add_edge(e)

        # Temporal: only the strict-order subtypes (BEFORE) define an order, so
        # only they participate in cycle breaking and closure; symmetric
        # subtypes (OVERLAP, ...) pass through deduplicated — running them
        # through the order machinery would break false "cycles" and emit
        # mislabeled BEFORE edges via closure.
        temporal_all = graph.edges_of_type(RelationType.TEMPORAL)
        strict = [e for e in temporal_all if e.subtype in STRICT_TEMPORAL_SUBTYPES]
        loose = [e for e in temporal_all if e.subtype not in STRICT_TEMPORAL_SUBTYPES]
        temporal = _break_cycles(strict)
        for e in temporal + _dedup_by_key(loose):
            out.add_edge(e)
        if self.close_temporal:
            out.edges.extend(self._temporal_closure(temporal))

        # Subevent passes through deduplicated (rare in this domain).
        for e in _dedup_by_key(graph.edges_of_type(RelationType.SUBEVENT)):
            out.add_edge(e)
        return out

    @staticmethod
    def _temporal_closure(temporal: list[RelationEdge]) -> list[RelationEdge]:
        g: nx.DiGraph = nx.DiGraph()
        for e in temporal:
            g.add_edge(e.head_id, e.tail_id)
        observed = set(g.edges())
        closure = nx.transitive_closure(g, reflexive=False)
        return [
            RelationEdge(
                head_id=h,
                tail_id=t,
                relation_type=RelationType.TEMPORAL,
                subtype="BEFORE",
                directed=True,
                confidence=0.5,
                rationale="transitively implied",
            )
            for h, t in closure.edges()
            if h != t and (h, t) not in observed
        ]
