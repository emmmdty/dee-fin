"""Deterministic, torch-free relation extractor.

A transparent rule baseline that exercises the full pipeline on CPU and serves
as the lower-bound reference in ablations. It reads only `EventNode` fields, so
it needs no model and no raw text:

- coreference: same event_type + same subject + same time_anchor, or argument
  overlap above a threshold.
- temporal: order events by `time_anchor`; every event of one anchor group is
  BEFORE every event of the next group.
- causal: a same-subject earlier->later pair whose (type, type) appears in a
  small, configurable financial cue table.

Every edge is grounded in the union of its endpoints' evidence spans.

`temporal_scope` decides what "the next group" means. The default `"subject"`
chains inside one subject's own timeline, which is what a BEFORE edge is supposed
to mean. `"corpus"` chains date groups across the whole node list, and only stays
near-linear when each date holds a handful of events. On a multi-company news
corpus it does not: Astock has 15623 events over 980 dates (median 12/date), so
corpus chaining emits 331k BEFORE edges of which **99.93% link two different
companies** — "some firm acted the day before some other firm did", which is date
adjacency, not a relation. Transitive closure over that layered DAG is quadratic
(~1.2e8 edges). `"corpus"` therefore has to be asked for explicitly; it remains
available for single-subject corpora and for reproducing the older graphs.
"""

from __future__ import annotations

from itertools import combinations

from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)

__all__ = ["HeuristicRelationExtractor"]

_TEMPORAL_SCOPES = ("corpus", "subject")

# A minimal, illustrative causal prior over financial event types. Extend via
# config; this is a baseline cue table, not a learned model.
_DEFAULT_CAUSAL_CUES: set[tuple[str, str]] = {
    ("EquityPledge", "EquityFreeze"),
    ("EquityFreeze", "EquityRepurchase"),
    ("ShareReduction", "EquityPledge"),
}


def _node_spans(node: EventNode) -> list[EvidenceSpan]:
    spans = list(node.trigger_evidence)
    for role_spans in node.argument_evidence.values():
        spans.extend(role_spans)
    return spans


def _argument_overlap(a: EventNode, b: EventNode) -> float:
    va, vb = set(a.arguments.values()), set(b.arguments.values())
    if not va or not vb:
        return 0.0
    return len(va & vb) / len(va | vb)


@relation_extractors.register("heuristic")
class HeuristicRelationExtractor(RelationExtractor):
    def __init__(
        self,
        coref_overlap_threshold: float = 0.6,
        causal_cues: set[tuple[str, str]] | None = None,
        temporal_scope: str = "subject",
    ) -> None:
        if temporal_scope not in _TEMPORAL_SCOPES:
            raise ValueError(
                f"unknown temporal_scope {temporal_scope!r}; expected one of {_TEMPORAL_SCOPES}"
            )
        self.coref_overlap_threshold = coref_overlap_threshold
        self.causal_cues = causal_cues if causal_cues is not None else _DEFAULT_CAUSAL_CUES
        self.temporal_scope = temporal_scope

    def extract(
        self, nodes: list[EventNode], context: ExtractionContext | None = None
    ) -> list[RelationEdge]:
        edges: list[RelationEdge] = []
        edges.extend(self._coreference(nodes))
        edges.extend(self._temporal(nodes))
        edges.extend(self._causal(nodes))
        return edges

    def _coreference(self, nodes: list[EventNode]) -> list[RelationEdge]:
        edges: list[RelationEdge] = []
        for a, b in combinations(nodes, 2):
            # Both branches require matching types, so skip the overlap
            # computation otherwise: at corpus scale this loop is O(n^2) pairs
            # (15623 nodes = 122M) and the set algebra dominates the runtime.
            if a.event_type != b.event_type:
                continue
            same_slot = (
                a.subject is not None
                and a.subject == b.subject
                and a.time_anchor is not None
                and a.time_anchor == b.time_anchor
            )
            overlap = _argument_overlap(a, b)
            if same_slot or overlap >= self.coref_overlap_threshold:
                edges.append(
                    RelationEdge(
                        head_id=a.event_id,
                        tail_id=b.event_id,
                        relation_type=RelationType.COREFERENCE,
                        directed=False,
                        confidence=max(overlap, 0.5),
                        evidence=_node_spans(a) + _node_spans(b),
                        rationale="same type/subject/time or high argument overlap",
                    )
                )
        return edges

    def _temporal(self, nodes: list[EventNode]) -> list[RelationEdge]:
        """Chain consecutive time-anchor groups: every event of one anchor is
        BEFORE every event of the next anchor.

        Linking whole groups (not adjacent nodes) keeps events that share an
        anchor connected to the chain — an adjacent-pair chain silently drops
        all but the last of them — while transitive closure still recovers the
        non-consecutive pairs.

        With `temporal_scope="subject"` (default) the chain runs inside each
        company's own timeline; with `"corpus"` it runs across every node, which
        only stays near-linear when dates hold few events. See the module docstring.
        """
        dated = [n for n in nodes if n.time_anchor]
        if self.temporal_scope == "subject":
            by_subject: dict[str, list[EventNode]] = {}
            for node in dated:
                by_subject.setdefault(node.subject or "", []).append(node)
            timelines = list(by_subject.values())
        else:
            timelines = [dated]

        edges: list[RelationEdge] = []
        for timeline in timelines:
            timeline.sort(key=lambda n: (n.time_anchor or "", n.event_id))
            groups: list[list[EventNode]] = []
            for node in timeline:
                if groups and groups[-1][0].time_anchor == node.time_anchor:
                    groups[-1].append(node)
                else:
                    groups.append([node])
            for earlier_group, later_group in zip(groups, groups[1:], strict=False):
                for earlier in earlier_group:
                    for later in later_group:
                        edges.append(
                            RelationEdge(
                                head_id=earlier.event_id,
                                tail_id=later.event_id,
                                relation_type=RelationType.TEMPORAL,
                                subtype="BEFORE",
                                directed=True,
                                confidence=0.7,
                                evidence=_node_spans(earlier) + _node_spans(later),
                                rationale=f"time {earlier.time_anchor} < {later.time_anchor}",
                            )
                        )
        return edges

    def _causal(self, nodes: list[EventNode]) -> list[RelationEdge]:
        edges: list[RelationEdge] = []
        for a, b in combinations(nodes, 2):
            if a.subject is None or a.subject != b.subject:
                continue
            ta, tb = a.time_anchor or "", b.time_anchor or ""
            if ta != tb:
                ordered = [(a, b)] if ta < tb else [(b, a)]
            else:
                # Tied (or missing) anchors carry no temporal order, so the cue
                # table decides the direction. Both orientations are checked —
                # otherwise the emitted edge set would depend on the input
                # order of `nodes`, breaking determinism.
                ordered = [(a, b), (b, a)]
            for earlier, later in ordered:
                if (earlier.event_type, later.event_type) in self.causal_cues:
                    edges.append(
                        RelationEdge(
                            head_id=earlier.event_id,
                            tail_id=later.event_id,
                            relation_type=RelationType.CAUSAL,
                            subtype="CAUSE",
                            directed=True,
                            confidence=0.55,
                            evidence=_node_spans(earlier) + _node_spans(later),
                            rationale="same subject + causal cue over event types",
                        )
                    )
        return edges
