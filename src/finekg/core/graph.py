"""Generic graph algorithms over an `EventGraph`.

These are reusable primitives (networkx conversion, coreference clustering,
cycle detection, transitive closure) shared by the consistency solver and the
evaluation metrics. The *policy* of how to repair inconsistencies lives in
`finekg.relations.consistency`; here we only compute structural facts.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

import networkx as nx

from finekg.core.schema import EventGraph, RelationEdge, RelationType

__all__ = [
    "to_networkx",
    "coreference_clusters",
    "find_cycles",
    "transitive_closure_pairs",
    "close_pairs",
    "is_acyclic",
    "edge_pair_set",
]


def to_networkx(
    graph: EventGraph, relation_type: RelationType | None = None
) -> nx.MultiDiGraph:
    """Build a networkx graph; optionally restrict to one relation family.

    Symmetric edges (e.g. coreference, where `directed=False`) are added in both
    directions so undirected algorithms behave correctly.
    """
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        if relation_type is not None and edge.relation_type != relation_type:
            continue
        g.add_edge(edge.head_id, edge.tail_id, key=edge.subtype or edge.relation_type.value)
        if not edge.directed:
            g.add_edge(edge.tail_id, edge.head_id, key=edge.subtype or edge.relation_type.value)
    return g


def coreference_clusters(graph: EventGraph, min_size: int = 1) -> list[set[str]]:
    """Connected components over coreference edges = event identity clusters.

    `min_size=2` drops singleton clusters — the CoNLL scoring convention, where
    unclustered mentions must not earn B³/CEAFe credit.
    """
    ug: nx.Graph = nx.Graph()
    ug.add_nodes_from(graph.nodes)
    for edge in graph.edges_of_type(RelationType.COREFERENCE):
        ug.add_edge(edge.head_id, edge.tail_id)
    return [set(c) for c in nx.connected_components(ug) if len(c) >= min_size]


def _directed_subgraph(
    graph: EventGraph,
    relation_type: RelationType,
    subtypes: Collection[str] | None = None,
) -> nx.DiGraph:
    """Directed edges of one family, optionally restricted to some subtypes.

    The restriction matters for temporal relations: only strict-order subtypes
    (BEFORE) form a partial order, so cycle/closure arguments must not mix in
    symmetric ones like OVERLAP (see `schema.STRICT_TEMPORAL_SUBTYPES`).
    """
    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(graph.nodes)
    for edge in graph.edges_of_type(relation_type):
        if edge.directed and (subtypes is None or edge.subtype in subtypes):
            g.add_edge(edge.head_id, edge.tail_id)
    return g


def find_cycles(
    graph: EventGraph,
    relation_type: RelationType,
    subtypes: Collection[str] | None = None,
) -> list[list[str]]:
    """Directed cycles within one relation family (e.g. causal loops)."""
    g = _directed_subgraph(graph, relation_type, subtypes)
    return [cycle for cycle in nx.simple_cycles(g)]


def is_acyclic(
    graph: EventGraph,
    relation_type: RelationType,
    subtypes: Collection[str] | None = None,
) -> bool:
    return nx.is_directed_acyclic_graph(_directed_subgraph(graph, relation_type, subtypes))


def transitive_closure_pairs(
    graph: EventGraph,
    relation_type: RelationType,
    subtypes: Collection[str] | None = None,
) -> set[tuple[str, str]]:
    """All (head, tail) pairs implied by transitively closing a relation family.

    For temporal "BEFORE"-style relations a consistent graph should be closed:
    if a->b and b->c then a->c. The difference between this set and the observed
    edges quantifies transitivity violations (see `eval.consistency`). Pass
    `subtypes` to close only the strict-order subtypes.
    """
    g = _directed_subgraph(graph, relation_type, subtypes)
    closure = nx.transitive_closure(g, reflexive=False)
    return {(h, t) for h, t in closure.edges() if h != t}


def close_pairs(pairs: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """Transitive closure of a set of directed ``(head, tail)`` pairs.

    A thin pairs-level companion to `transitive_closure_pairs` (which needs a
    full `EventGraph`): the evaluation metric holds plain edge lists, not graphs.
    Reflexive edges are not returned, so closing the empty set yields the empty set.
    """
    g: nx.DiGraph = nx.DiGraph()
    g.add_edges_from(pairs)
    return {(h, t) for h, t in nx.transitive_closure(g, reflexive=False).edges() if h != t}


def edge_pair_set(edges: list[RelationEdge]) -> set[tuple[str, str]]:
    return {(e.head_id, e.tail_id) for e in edges if e.directed}
