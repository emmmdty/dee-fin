"""Distance-sensitive graph linearisation (DsGL), SeDGPL's prompt builder.

Faithful to `tools.getTemplate` / `util.getDistance`, because M1 (risk-aware
linearisation) is only a meaningful comparison against the real baseline.

Two steps, and they are not the same criterion:

1. **Truncate** to `max_edges` = ``len_arg // 10`` = 20 edges. SeDGPL keeps the
   *first* 20 in stored order -- not the 20 nearest. The query edge is always
   excluded (it is stored last, and the graph is only truncated when it holds
   more than 20 edges, so slicing the head can never capture it).
2. **Order** the survivors by hop distance from the query edge, *descending*, so
   the nearest edges sit adjacent to the `<mask>`. Distance is a BFS over the
   *undirected* graph seeded from both endpoints of the query edge; unreachable
   edges get 16, as in the original.

Step 1 is the limitation SeDGPL admits to ("we have to discard triples due to the
input length limit"): the discarded edges are chosen by nothing at all. That is
what M1's `select_nearest_edges` replaces -- a distance-budgeted selection that
keeps the edges nearest the query. (On a *constructed* ECG, whose edges carry a
confidence, the same seam would take an admission-scored selector instead; the
gold ECG the main table runs on has no such score, so distance is M1 there.)

Events are rendered as single vocabulary tokens (`<a_0>`, `<a_1>`, ...), one per
distinct mention string and one per event type, mirroring `util.collect_mult_event`.
A multi-word trigger would otherwise break the original's invariant that the text
and type templates tokenize to the same length.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from finekg.core.registry import Registry
from finekg.succession.data.cgep import RELATION_SURFACE, CgepInstance, CgepNode

__all__ = [
    "EDGE_BUDGET",
    "UNREACHABLE_DISTANCE",
    "EdgeSelector",
    "EventVocabulary",
    "Linearization",
    "edge_distances",
    "edge_selectors",
    "linearize",
    "select_nearest_edges",
    "truncate_edges",
]

# `parameter.py` sets len_arg=200 and `getTemplate` divides it by 10.
EDGE_BUDGET = 20

# `util.find_distances` hardcodes this for nodes the BFS never reaches.
UNREACHABLE_DISTANCE = 16

Edge = tuple[int, str, int]
EdgeSelector = Callable[[Sequence[Edge], int], list[Edge]]


class EventVocabulary:
    """`<a_i>` token per distinct mention string, then per event type.

    Mirrors `util.collect_mult_event`: mentions from nodes and candidate sets
    first, in encounter order, then any event type not already claimed by a
    mention. `to_add` maps each token to the string it should be mean-initialised
    from, which is what `model.handler` consumes.
    """

    def __init__(self) -> None:
        self._token: dict[str, str] = {}

    @classmethod
    def build(cls, instances: Iterable[CgepInstance]) -> EventVocabulary:
        vocab = cls()
        instances = list(instances)
        for instance in instances:
            for node in (*instance.nodes, *instance.candidates):
                vocab._claim(node.trigger)
        for instance in instances:
            for node in (*instance.nodes, *instance.candidates):
                vocab._claim(node.event_type)
        return vocab

    def _claim(self, surface: str) -> str:
        if surface not in self._token:
            self._token[surface] = f"<a_{len(self._token)}>"
        return self._token[surface]

    def __len__(self) -> int:
        return len(self._token)

    def token(self, surface: str) -> str:
        """The `<a_i>` standing for `surface`; unseen strings pass through."""
        return self._token.get(surface, surface)

    @property
    def to_add(self) -> dict[str, str]:
        return {token: surface for surface, token in self._token.items()}


@dataclass(frozen=True)
class Linearization:
    """A prompt template and its type-only twin, plus the edges they came from."""

    template: str
    type_template: str
    edges: tuple[Edge, ...]  # ordered template edges, then the query edge
    dropped: int  # edges the budget discarded

    @property
    def query_edge(self) -> Edge:
        return self.edges[-1]


def truncate_edges(edges: Sequence[Edge], max_edges: int = EDGE_BUDGET) -> list[Edge]:
    """SeDGPL's budget: the first `max_edges` in stored order, query excluded."""
    if len(edges) <= max_edges:
        return list(edges[:-1])
    return list(edges[:max_edges])


def edge_distances(
    edges: Sequence[Edge], query: Edge, unreachable: int = UNREACHABLE_DISTANCE
) -> list[int]:
    """Hop distance of each edge from the query edge, over the undirected graph.

    An edge's distance is the smaller of its endpoints' distances; both endpoints
    of the query edge start at 0. The graph spans `edges` *and* `query`, matching
    `getDistance`, which passes the query edge into the adjacency it builds.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    for head, _, tail in (*edges, query):
        adjacency[head].add(tail)
        adjacency[tail].add(head)

    distance: dict[int, int] = {}
    queue = deque([(query[0], 0), (query[2], 0)])
    seen = {query[0], query[2]}
    while queue:
        node, hops = queue.popleft()
        distance[node] = hops
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, hops + 1))

    return [
        min(distance.get(head, unreachable), distance.get(tail, unreachable))
        for head, _, tail in edges
    ]


def select_nearest_edges(edges: Sequence[Edge], max_edges: int = EDGE_BUDGET) -> list[Edge]:
    """M1's risk-aware budget: keep the `max_edges` edges nearest the query.

    `truncate_edges` slices the stored head and discards the rest by nothing at
    all. This scores every template edge by hop distance to the query over the
    *whole* graph (not the truncated head SeDGPL would keep) and retains the
    nearest `max_edges`, spending the budget on the context closest to what is
    predicted. Ties break by stored order, and a graph already within budget is
    returned unchanged -- so on the instances the budget never bites, M1 and
    SeDGPL are identical.
    """
    template = list(edges[:-1])
    if len(template) <= max_edges:
        return template
    distances = edge_distances(template, edges[-1])
    nearest = sorted(range(len(template)), key=lambda i: (distances[i], i))[:max_edges]
    keep = set(nearest)
    return [edge for i, edge in enumerate(template) if i in keep]


def _surface(relation: str) -> str:
    """Readable form of a relation, or the relation verbatim.

    MAVEN's subtypes (`CAUSE`, `PRECONDITION`, `SUBEVENT_OF`) get a surface form;
    anything else passes through untouched. ESC edges are already the literal
    string `'cause'` that SeDGPL splices into its template, and the reproduction
    anchor is only worth anything if we splice the same one.
    """
    return RELATION_SURFACE.get(relation, relation)


def _render(pairs: Sequence[tuple[str, str, str]], anchor: str, relation: str) -> str:
    body = "".join(f"{head} {rel} {tail} , " for head, rel, tail in pairs)
    return f"{body}{anchor} {relation} <mask> ."


def linearize(
    instance: CgepInstance,
    vocab: EventVocabulary | None = None,
    *,
    max_edges: int = EDGE_BUDGET,
    selector: EdgeSelector | None = None,
) -> Linearization:
    """Render `instance` into SeDGPL's masked template and its type twin.

    `selector` picks which edges survive the budget -- SeDGPL's storage-order
    `truncate_edges` by default, `select_nearest_edges` for M1. The distance
    ordering that follows is unchanged, so only the surviving *set* differs.
    """
    select = selector or truncate_edges
    template_edges = select(instance.edges, max_edges)
    query = instance.query_edge

    distances = edge_distances(template_edges, query)
    # `sorted(zip(distance, relation), reverse=True)`: far edges first, so the
    # nearest context ends up adjacent to the mask. Ties fall back to the edge
    # tuple, which keeps the order deterministic.
    weighted = sorted(zip(distances, template_edges, strict=True), reverse=True)
    ordered = [edge for _, edge in weighted]

    def text(node: CgepNode) -> str:
        return vocab.token(node.trigger) if vocab else node.trigger

    def kind(node: CgepNode) -> str:
        return vocab.token(node.event_type) if vocab else node.event_type

    nodes = instance.nodes
    text_pairs = [(text(nodes[h]), _surface(r), text(nodes[t])) for h, r, t in ordered]
    type_pairs = [(kind(nodes[h]), _surface(r), kind(nodes[t])) for h, r, t in ordered]

    return Linearization(
        template=_render(text_pairs, text(nodes[query[0]]), _surface(query[1])),
        type_template=_render(type_pairs, kind(nodes[query[0]]), _surface(query[1])),
        edges=(*ordered, query),
        dropped=len(instance.edges) - 1 - len(template_edges),
    )


edge_selectors: Registry[EdgeSelector] = Registry("edge_selector")


@edge_selectors.register("sedgpl")
def _sedgpl_selector() -> EdgeSelector:
    """SeDGPL's storage-order truncation: the baseline, and the default."""
    return truncate_edges


@edge_selectors.register("distance")
def _distance_selector() -> EdgeSelector:
    """M1 risk-aware truncation: keep the edges nearest the query."""
    return select_nearest_edges
