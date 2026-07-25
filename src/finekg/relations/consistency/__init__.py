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
from dataclasses import dataclass

import networkx as nx

from finekg.core.eval.consistency import consistency_report
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
    "RepairEdit",
    "RepairTrace",
]


@dataclass(frozen=True)
class RepairEdit:
    """One structured edit the solver made to reach global consistency.

    ``violation`` names the inconsistency the edit resolves (``causal_cycle`` /
    ``temporal_cycle`` / ``temporal_closure`` / ``coref_dedup`` / ...);
    ``edge_key`` is the affected edge's ``RelationEdge.key()``.
    """

    action: str  # "drop" | "add"
    violation: str
    edge_key: tuple[str, str, str, str]
    confidence: float


def _edit(action: str, violation: str, edge: RelationEdge) -> RepairEdit:
    return RepairEdit(action, violation, edge.key(), edge.confidence)


@dataclass
class RepairTrace:
    """An auditable record of a solver run: every edit plus before/after diags.

    ``before`` / ``after`` are `consistency_report` snapshots of the input and
    output graphs; ``counts`` holds edge totals and edit tallies. The trace is a
    diagnostic artifact, deliberately kept out of the frozen `core.schema`
    contracts (nothing downstream depends on it).
    """

    edits: list[RepairEdit]
    before: dict[str, float]
    after: dict[str, float]
    counts: dict[str, int]

    @property
    def dropped(self) -> list[RepairEdit]:
        return [e for e in self.edits if e.action == "drop"]

    @property
    def added(self) -> list[RepairEdit]:
        return [e for e in self.edits if e.action == "add"]

    @classmethod
    def of(
        cls, before_graph: EventGraph, after_graph: EventGraph, edits: list[RepairEdit]
    ) -> RepairTrace:
        return cls(
            edits=list(edits),
            before=consistency_report(before_graph),
            after=consistency_report(after_graph),
            counts={
                "edges_before": len(before_graph.edges),
                "edges_after": len(after_graph.edges),
                "dropped": sum(1 for e in edits if e.action == "drop"),
                "added": sum(1 for e in edits if e.action == "add"),
            },
        )


class ConsistencySolver(ABC):
    @abstractmethod
    def solve(self, graph: EventGraph) -> EventGraph:
        """Return a new graph with consistent relations."""

    def solve_with_trace(self, graph: EventGraph) -> tuple[EventGraph, RepairTrace]:
        """Solve and emit an audit trace of the repairs made.

        The base implementation records no edits — correct for no-op solvers
        (e.g. `identity`). Solvers that edit the graph override this to log each
        drop/add; `solve` itself is left byte-for-byte unchanged.
        """
        out = self.solve(graph)
        return out, RepairTrace.of(graph, out, [])


consistency_solvers: Registry[ConsistencySolver] = Registry("consistency_solver")


@consistency_solvers.register("identity")
class IdentityConsistencySolver(ConsistencySolver):
    """No-op solver (ablation baseline)."""

    def solve(self, graph: EventGraph) -> EventGraph:
        return graph.model_copy(deep=True)


def _dedup_by_key_traced(
    edges: list[RelationEdge],
) -> tuple[list[RelationEdge], list[RelationEdge]]:
    """Collapse exact duplicates (same identity key) to the most confident copy.

    Parallel edges that differ in subtype (e.g. CAUSE and PRECONDITION on the
    same pair) are distinct keys and survive — only true duplicates collapse.
    Returns ``(kept, dropped)``; the dropped losers feed the repair trace.
    """
    best: dict[tuple[str, str, str, str], RelationEdge] = {}
    dropped: list[RelationEdge] = []
    for e in edges:
        k = e.key()
        cur = best.get(k)
        if cur is None or e.confidence > cur.confidence:
            if cur is not None:
                dropped.append(cur)
            best[k] = e
        else:
            dropped.append(e)
    return list(best.values()), dropped


def _dedup_by_key(edges: list[RelationEdge]) -> list[RelationEdge]:
    return _dedup_by_key_traced(edges)[0]


def _break_cycles_traced(
    edges: list[RelationEdge],
) -> tuple[list[RelationEdge], list[RelationEdge]]:
    """Greedily drop the weakest pair in each directed cycle until acyclic.

    Cycle structure lives at the (head, tail) pair level, so parallel edges of
    different subtypes on one pair stand or fall together (the pair's strength
    is its strongest edge); they are not silently collapsed into one edge.
    Returns ``(kept, dropped)``; the dropped edges are the cycle-breaking edits.
    """
    by_pair: dict[tuple[str, str], list[RelationEdge]] = {}
    for e in _dedup_by_key(edges):
        by_pair.setdefault((e.head_id, e.tail_id), []).append(e)
    g: nx.DiGraph = nx.DiGraph()
    for (h, t), parallel in by_pair.items():
        g.add_edge(h, t, confidence=max(e.confidence for e in parallel))
    dropped: list[RelationEdge] = []
    while True:
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda c: g[c[0]][c[1]]["confidence"])
        g.remove_edge(weakest[0], weakest[1])
        dropped.extend(by_pair.pop((weakest[0], weakest[1]), []))
    kept = [e for h, t in g.edges() for e in by_pair[(h, t)]]
    return kept, dropped


@consistency_solvers.register("greedy")
class GreedyConsistencySolver(ConsistencySolver):
    def __init__(self, close_temporal: bool = True) -> None:
        self.close_temporal = close_temporal

    def solve(self, graph: EventGraph) -> EventGraph:
        return self._solve(graph, None)

    def solve_with_trace(self, graph: EventGraph) -> tuple[EventGraph, RepairTrace]:
        edits: list[RepairEdit] = []
        out = self._solve(graph, edits)
        return out, RepairTrace.of(graph, out, edits)

    def _solve(self, graph: EventGraph, edits: list[RepairEdit] | None) -> EventGraph:
        """Shared solver body; records edits into ``edits`` when it is not None.

        With ``edits=None`` this is exactly the original `solve` (same output,
        same edge order) — the trace path is a strict extension, never a change
        to the default behavior.
        """
        out = EventGraph(nodes=dict(graph.nodes), metadata=dict(graph.metadata))

        # Coreference: keep the most confident undirected edge per unordered
        # pair (the same winner rule `_dedup_by_key` applies to typed edges).
        best_coref: dict[frozenset[str], RelationEdge] = {}
        for e in graph.edges_of_type(RelationType.COREFERENCE):
            pair = frozenset((e.head_id, e.tail_id))
            cur = best_coref.get(pair)
            if cur is None or e.confidence > cur.confidence:
                if cur is not None and edits is not None:
                    edits.append(_edit("drop", "coref_dedup", cur))
                best_coref[pair] = e
            elif edits is not None:
                edits.append(_edit("drop", "coref_dedup", e))
        for e in best_coref.values():
            out.add_edge(e)

        # Causal: enforce acyclicity (cycles across CAUSE/PRECONDITION are
        # contradictions regardless of subtype).
        causal_kept, causal_dropped = _break_cycles_traced(
            graph.edges_of_type(RelationType.CAUSAL)
        )
        if edits is not None:
            edits.extend(_edit("drop", "causal_cycle", e) for e in causal_dropped)
        for e in causal_kept:
            out.add_edge(e)

        # Temporal: only the strict-order subtypes (BEFORE) define an order, so
        # only they participate in cycle breaking and closure; symmetric
        # subtypes (OVERLAP, ...) pass through deduplicated — running them
        # through the order machinery would break false "cycles" and emit
        # mislabeled BEFORE edges via closure.
        temporal_all = graph.edges_of_type(RelationType.TEMPORAL)
        strict = [e for e in temporal_all if e.subtype in STRICT_TEMPORAL_SUBTYPES]
        loose = [e for e in temporal_all if e.subtype not in STRICT_TEMPORAL_SUBTYPES]
        temporal, temporal_dropped = _break_cycles_traced(strict)
        loose_kept, loose_dropped = _dedup_by_key_traced(loose)
        if edits is not None:
            edits.extend(_edit("drop", "temporal_cycle", e) for e in temporal_dropped)
            edits.extend(_edit("drop", "temporal_dedup", e) for e in loose_dropped)
        for e in temporal + loose_kept:
            out.add_edge(e)
        if self.close_temporal:
            closure = self._temporal_closure(temporal)
            out.edges.extend(closure)
            if edits is not None:
                edits.extend(_edit("add", "temporal_closure", e) for e in closure)

        # Subevent passes through deduplicated (rare in this domain).
        sub_kept, sub_dropped = _dedup_by_key_traced(graph.edges_of_type(RelationType.SUBEVENT))
        if edits is not None:
            edits.extend(_edit("drop", "subevent_dedup", e) for e in sub_dropped)
        for e in sub_kept:
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
