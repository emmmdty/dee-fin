"""Rebuild CGEP instances from MAVEN-ERE.

SeDGPL (EMNLP'24 Findings, arXiv:2409.17480) introduced CGEP but shipped only
its ESC build (`ESCSubWoRe.npy`); the MAVEN build its `load_data.py` reads
(`data/MAVENSubWoRe.npy`) was never released. Its published CGEP-MAVEN numbers
are therefore *not* same-data comparable, and the corpus has to be rebuilt from
MAVEN-ERE under an explicitly stated protocol.

Protocol (each choice is checked against the statistics SeDGPL reports, and
against what its `tools.py` template builder actually requires):

* **ECG** = one weakly-connected component of a document's causal
  (``CAUSE`` + ``PRECONDITION``) *and* ``subevent`` edges, with at least
  ``min_nodes`` events. Temporal ``BEFORE`` edges are excluded from the
  topology: adding them takes the component density to ~235 edges/ECG, which no
  20-edge linearisation budget can carry. They re-enter downstream as structural
  features, not as graph structure.
* **Coreference** is already collapsed by MAVEN-ERE (one ``events[i]`` is one
  cluster); the first mention represents the cluster, matching SeDGPL's
  `processNode` ("选择第一个共指事件").
* **Query edge** = an edge whose tail has out-degree 0 *and* in-degree 1. This
  is forced by the prompt format rather than chosen: SeDGPL renders every
  template edge as a pair of event tokens, so a gold successor appearing in any
  other edge would print its own answer token into the prompt. Occurring solely
  in the query edge is exactly ``outdeg == 0 and indeg == 1``.
* **Candidate set** = ``n_candidates`` distinct event nodes drawn uniformly from
  the split, always including the gold successor and *not* excluding the source
  ECG's own nodes. This is read off the one build SeDGPL did release
  (``ESCSubWoRe.npy``): its candidate sets hold 256 distinct event identities,
  gold is always present, draws are near-uniform corpus-wide, and roughly 17% of
  a graph's own nodes reappear as candidates -- exactly what uniform sampling
  without exclusion produces. Its precise eligibility rule is not recoverable
  (161 of its candidates never occur in any released ECG), so uniform sampling
  over the split is ours and must be cited as such.

  Candidates are *nodes*, not trigger strings, so two candidates may share a
  trigger. That is faithful: SeDGPL keys its added vocabulary by mention string
  and scores candidates by token id, so colliding triggers score identically. In
  the released ESC build only ~178.0 of the 256 candidates are distinct answers.

Measured against the paper (train+valid, ``min_nodes=4``): 2994 documents
(paper 3015), 8.82 nodes/ECG (8.4), 13.21 edges/ECG (12.9), 10116 instances
(12200). The ECG count (3743 vs 5308) does **not** reconcile and is reported as
a known deviation rather than tuned away.

The query-edge rule above is not a guess: on all 1192 instances of the released
ESC build, the gold successor has out-degree 0, in-degree 1, and never occurs in
a template edge.

Note on edge order: SeDGPL's `getTemplate` truncates by *stored order* before
sorting the survivors by graph distance, so the order edges are emitted in here
is load-bearing for any faithful baseline. It follows document order.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from string import punctuation

from finekg.core.schema import RelationType
from finekg.relations.data.maven_ere import RelationDocument

__all__ = [
    "CANDIDATE_SET_SIZE",
    "CgepInstance",
    "CgepNode",
    "CandidatePool",
    "EventCausalityGraph",
    "RELATION_SURFACE",
    "build_cgep",
    "extract_ecgs",
    "query_edge_indices",
]

# SeDGPL uses 512 candidates on MAVEN and 256 on ESC.
CANDIDATE_SET_SIZE = 512

# Causal subtypes that carry ECG topology. `subevent` is added separately.
_CAUSAL_SUBTYPES = ("CAUSE", "PRECONDITION")

# Surface forms used when a graph is linearised into a prompt template.
RELATION_SURFACE = {
    "CAUSE": "causes",
    "PRECONDITION": "precondition of",
    "SUBEVENT_OF": "subevent of",
}


@dataclass(frozen=True)
class CgepNode:
    """One event, coreference-collapsed to its representative mention.

    `token_span` locates the trigger among the whitespace tokens of `sentence`.
    SeDGPL's `doReplace` swaps those tokens for the event's `<a_i>` token before
    encoding, so the sentence encoder reads the same symbol the template does; a
    plain string replace would also hit other occurrences of the same word.
    """

    node_id: str
    event_type: str
    trigger: str
    sentence: str
    sent_id: int | None = None
    token_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class EventCausalityGraph:
    """A connected causal+subevent component of one document."""

    doc_id: str
    nodes: tuple[CgepNode, ...]
    edges: tuple[tuple[int, str, int], ...]  # (head_index, subtype, tail_index)

    def out_degrees(self) -> Counter[int]:
        return Counter(head for head, _, _ in self.edges)

    def in_degrees(self) -> Counter[int]:
        return Counter(tail for _, _, tail in self.edges)


@dataclass(frozen=True)
class CgepInstance:
    """One (ECG, anchor, gold successor, candidate set) prediction problem.

    `edges[-1]` is the query edge; the rest form the prompt template. Candidates
    are event nodes drawn corpus-wide; because SeDGPL keys its added vocabulary
    by mention string and scores by token id, candidates sharing a trigger score
    identically, so the set of *distinct answers* is smaller than `candidates`.
    """

    instance_id: str
    doc_id: str
    nodes: tuple[CgepNode, ...]
    edges: tuple[tuple[int, str, int], ...]
    candidates: tuple[CgepNode, ...]
    label: int

    @property
    def query_edge(self) -> tuple[int, str, int]:
        return self.edges[-1]

    @property
    def template_edges(self) -> tuple[tuple[int, str, int], ...]:
        return self.edges[:-1]

    @property
    def anchor_index(self) -> int:
        return self.edges[-1][0]

    @property
    def gold_index(self) -> int:
        return self.edges[-1][2]

    @property
    def gold_trigger(self) -> str:
        return self.candidates[self.label].trigger

    @property
    def distinct_answers(self) -> int:
        """Candidates collapse to this many scoreable answers (trigger tokens)."""
        return len({c.trigger for c in self.candidates})


def _sentences(doc: RelationDocument) -> list[str]:
    return doc.doc_text.split("\n") if doc.doc_text else []


def token_span(sentence: str, trigger: str) -> tuple[int, int] | None:
    """Whitespace-token span of `trigger` inside `sentence`, or None if absent.

    Punctuation-insensitive, then case-insensitive. MAVEN's sentences are raw
    text, so a trigger is routinely glued to a comma or full stop ("died.") and
    sometimes differs in case from the sentence ("revolution" / "Revolution").
    Exact whitespace matching misses 3390 of its 33017 events; both fallbacks
    together bring that down to 6 (possessives like "game's"), reaching 14 of
    10116 instances. Those are rejected by `encode_instance` rather than encoded
    at the wrong position.
    """
    words = sentence.split()
    target = [core for core in (w.strip(punctuation) for w in trigger.split()) if core]
    if not target:
        return None
    cores = [w.strip(punctuation) for w in words]

    for candidate in (cores, [c.lower() for c in cores]):
        needle = target if candidate is cores else [t.lower() for t in target]
        for start in range(len(candidate) - len(needle) + 1):
            if candidate[start : start + len(needle)] == needle:
                return (start, start + len(needle))
    return None


def _representatives(doc: RelationDocument) -> dict[str, CgepNode]:
    """Representative mention per coreference cluster, keyed by node id."""
    lines = _sentences(doc)
    by_id = {node.event_id: node for node in doc.nodes}
    out: dict[str, CgepNode] = {}
    for node_id in doc.representative.values():
        node = by_id.get(node_id)
        if node is None:
            continue
        sent_id = node.trigger_evidence[0].sent_id if node.trigger_evidence else None
        sentence = lines[sent_id] if sent_id is not None and sent_id < len(lines) else ""
        out[node_id] = CgepNode(
            node_id=node_id,
            event_type=node.event_type,
            trigger=node.trigger,
            sentence=sentence,
            sent_id=sent_id,
            token_span=token_span(sentence, node.trigger),
        )
    return out


def _topology_edges(doc: RelationDocument, *, include_subevent: bool) -> list[tuple[str, str, str]]:
    """Causal (+ optionally subevent) edges between representative mentions."""
    edges: list[tuple[str, str, str]] = []
    for edge in doc.gold_edges:
        if edge.relation_type is RelationType.CAUSAL and edge.subtype in _CAUSAL_SUBTYPES:
            edges.append((edge.head_id, edge.subtype, edge.tail_id))
        elif include_subevent and edge.relation_type is RelationType.SUBEVENT:
            edges.append((edge.head_id, edge.subtype or "SUBEVENT_OF", edge.tail_id))
    return edges


def extract_ecgs(
    doc: RelationDocument,
    *,
    min_nodes: int = 4,
    include_subevent: bool = True,
) -> list[EventCausalityGraph]:
    """Weakly-connected causal+subevent components with >= `min_nodes` events."""
    edges = _topology_edges(doc, include_subevent=include_subevent)
    if not edges:
        return []
    reps = _representatives(doc)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for head, _, tail in edges:
        adjacency[head].add(tail)
        adjacency[tail].add(head)

    # Deterministic component order: first appearance in document edge order.
    order = [node for edge in edges for node in (edge[0], edge[2])]
    graphs: list[EventCausalityGraph] = []
    seen: set[str] = set()
    for start in order:
        if start in seen:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component
        if len(component) < min_nodes or not component <= reps.keys():
            continue
        member_edges = [e for e in edges if e[0] in component]
        # Node indices follow first appearance in the component's edge order.
        index: dict[str, int] = {}
        for head, _, tail in member_edges:
            for node in (head, tail):
                if node not in index:
                    index[node] = len(index)
        graphs.append(
            EventCausalityGraph(
                doc_id=doc.doc_id,
                nodes=tuple(reps[node] for node in index),
                edges=tuple((index[h], rel, index[t]) for h, rel, t in member_edges),
            )
        )
    return graphs


def query_edge_indices(ecg: EventCausalityGraph) -> list[int]:
    """Edges whose tail occurs *only* there: out-degree 0 and in-degree 1.

    Any looser rule leaks the answer, because every other edge the tail takes
    part in would render the gold event's own token into the prompt template.
    """
    out_degree, in_degree = ecg.out_degrees(), ecg.in_degrees()
    return [
        i
        for i, (_, _, tail) in enumerate(ecg.edges)
        if out_degree[tail] == 0 and in_degree[tail] == 1
    ]


class CandidatePool:
    """The split's event nodes, drawn from uniformly and without exclusion."""

    def __init__(self, nodes: Iterable[CgepNode]) -> None:
        unique: dict[str, CgepNode] = {}
        for node in nodes:
            if node.trigger:
                unique.setdefault(node.node_id, node)
        self._nodes = list(unique.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def sample(
        self,
        gold: CgepNode,
        rng: random.Random,
        size: int = CANDIDATE_SET_SIZE,
    ) -> tuple[tuple[CgepNode, ...], int]:
        """`size`-1 negatives + gold, shuffled. Returns (candidates, gold index).

        Negatives are distinct *nodes*; their triggers may collide with each
        other or with gold, as they do in SeDGPL's own build. Yields fewer than
        `size` candidates only when the pool is smaller, i.e. on fixtures.
        """
        wanted = min(size - 1, max(len(self._nodes) - 1, 0))
        picks: list[CgepNode] = []
        # One oversampled draw absorbs the gold collision without a retry loop.
        for node in rng.sample(self._nodes, min(len(self._nodes), wanted + 1)):
            if node.node_id == gold.node_id:
                continue
            picks.append(node)
            if len(picks) == wanted:
                break
        candidates = [*picks, gold]
        rng.shuffle(candidates)
        label = next(i for i, c in enumerate(candidates) if c.node_id == gold.node_id)
        return tuple(candidates), label


def build_cgep(
    docs: Iterable[RelationDocument],
    *,
    min_nodes: int = 4,
    include_subevent: bool = True,
    n_candidates: int = CANDIDATE_SET_SIZE,
    seed: int = 209,
) -> tuple[list[CgepInstance], dict[str, float]]:
    """Build every CGEP instance in `docs`, plus the statistics to check it.

    `seed` defaults to SeDGPL's. Instances are deterministic given document
    order; the candidate pool is drawn from all `docs`, so pass one split.
    """
    graphs = [
        ecg
        for doc in docs
        for ecg in extract_ecgs(doc, min_nodes=min_nodes, include_subevent=include_subevent)
    ]
    pool = CandidatePool(node for ecg in graphs for node in ecg.nodes)
    rng = random.Random(seed)

    instances: list[CgepInstance] = []
    for ecg in graphs:
        for position in query_edge_indices(ecg):
            query = ecg.edges[position]
            gold = ecg.nodes[query[2]]
            if not gold.trigger:
                continue
            candidates, label = pool.sample(gold, rng, size=n_candidates)
            template = tuple(e for i, e in enumerate(ecg.edges) if i != position)
            if not template:
                continue
            instances.append(
                CgepInstance(
                    instance_id=f"{ecg.doc_id}::{query[0]}-{query[2]}",
                    doc_id=ecg.doc_id,
                    nodes=ecg.nodes,
                    edges=(*template, query),
                    candidates=candidates,
                    label=label,
                )
            )

    n_graphs = len(graphs) or 1
    n_instances = len(instances) or 1
    stats = {
        "documents": float(len({ecg.doc_id for ecg in graphs})),
        "ecgs": float(len(graphs)),
        "nodes_per_ecg": sum(len(g.nodes) for g in graphs) / n_graphs,
        "edges_per_ecg": sum(len(g.edges) for g in graphs) / n_graphs,
        "instances": float(len(instances)),
        "candidate_pool": float(len(pool)),
        # Triggers collide, so a nominally 512-way choice is easier than it looks.
        "distinct_answers": sum(i.distinct_answers for i in instances) / n_instances,
    }
    return instances, stats


def iter_documents(paths: Iterable[str]) -> Iterator[RelationDocument]:
    """Chain `load_maven_ere` over several split files."""
    from finekg.relations.data.maven_ere import load_maven_ere

    for path in paths:
        yield from load_maven_ere(path)
