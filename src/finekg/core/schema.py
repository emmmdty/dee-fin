"""Frozen cross-stage data contracts for the FinEKG pipeline.

These models are the stable interface between every stage:

    event mentions/canonical records -> EventNode
    relation extraction & graph      -> RelationEdge / EventGraph
    reliable downstream reasoning    -> Prediction

Downstream code depends only on these types. Methods may be swapped freely
(heuristic baseline <-> neural) as long as they consume and produce these
contracts, so upgrading a method never requires reworking the framework.

Design rule: extend by adding optional fields; never repurpose an existing
field. Every node, edge and prediction carries `evidence` for provenance.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

__all__ = [
    "EvidenceSpan",
    "EventNode",
    "RelationType",
    "STRICT_TEMPORAL_SUBTYPES",
    "RelationEdge",
    "EventGraph",
    "TemporalQuad",
    "ForecastQuery",
    "RankedCandidate",
    "EvidenceLink",
    "Prediction",
]


class EvidenceSpan(BaseModel):
    """A pointer back to the source text — the unit of provenance.

    Character offsets are relative to the document `doc_id`; `sent_id` is
    optional and only used when the source is sentence-segmented.
    """

    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    sent_id: int | None = Field(default=None, ge=0)
    text: str = ""

    def is_valid(self) -> bool:
        return self.char_end >= self.char_start


# --------------------------------------------------------------------------- #
# Nodes: an event record produced upstream (e.g. MAVEN pipeline or SARGE application).
# --------------------------------------------------------------------------- #
class EventNode(BaseModel):
    """A single event record = one node of the event graph.

    `event_id` must be stable and unique within a corpus so edges and
    forecasting queries can reference it. `arguments` maps an argument role to
    its surface value; `argument_evidence` maps the same role to the spans that
    support it.
    """

    event_id: str
    event_type: str
    doc_id: str
    trigger: str = ""
    trigger_evidence: list[EvidenceSpan] = Field(default_factory=list)
    arguments: dict[str, str] = Field(default_factory=dict)
    argument_evidence: dict[str, list[EvidenceSpan]] = Field(default_factory=dict)
    # Coarse anchors that make a record a first-class graph node.
    time_anchor: str | None = None  # normalized date/time, e.g. "2021-03-15"
    subject: str | None = None  # normalized company / actor, for cross-doc grouping
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Edges: relations between events = the graph the relation stage builds.
# --------------------------------------------------------------------------- #
class RelationType(str, Enum):
    """The four relation families covered by MAVEN-ERE."""

    COREFERENCE = "coreference"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SUBEVENT = "subevent"


# Temporal subtypes that assert a *strict order* and therefore must be acyclic
# and may be transitively closed. Symmetric/interval subtypes (OVERLAP,
# SIMULTANEOUS, ...) carry no order, so mixing them into cycle breaking or
# closure would manufacture contradictions and mislabeled BEFORE edges. The
# empty string is included because extractors default untyped temporal edges
# to the BEFORE reading.
STRICT_TEMPORAL_SUBTYPES: frozenset[str] = frozenset({"", "BEFORE"})


class RelationEdge(BaseModel):
    """A typed, evidence-grounded relation between two event nodes.

    `subtype` carries the fine-grained label (e.g. "BEFORE"/"OVERLAP" for
    temporal, "CAUSE"/"PRECONDITION" for causal). `directed=False` marks
    symmetric relations such as coreference. `evidence` grounds the edge in the
    source text; `rationale` is an optional free-text justification.
    """

    head_id: str
    tail_id: str
    relation_type: RelationType
    subtype: str = ""
    directed: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    rationale: str = ""
    # Verifier-stage annotations (optional; set by the grounding/faithfulness
    # agent). `faithfulness` is the calibrated evidence-faithfulness score;
    # `admitted=False` marks an edge the verifier abstained on.
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    admitted: bool = True

    def key(self) -> tuple[str, str, str, str]:
        """Identity used for set-style comparison and dedup in evaluation."""
        return (self.head_id, self.tail_id, self.relation_type.value, self.subtype)


class EventGraph(BaseModel):
    """A container of event nodes and the typed relations between them.

    Pure data; graph algorithms (networkx conversion, transitive closure,
    cycle detection) live in `finekg.core.graph` so this contract stays stable.
    """

    nodes: dict[str, EventNode] = Field(default_factory=dict)
    edges: list[RelationEdge] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def add_node(self, node: EventNode) -> None:
        self.nodes[node.event_id] = node

    def add_edge(self, edge: RelationEdge) -> None:
        self.edges.append(edge)

    def edges_of_type(self, relation_type: RelationType) -> list[RelationEdge]:
        return [e for e in self.edges if e.relation_type == relation_type]


# --------------------------------------------------------------------------- #
# Forecasting: queries and predictions over a temporal event graph.
# --------------------------------------------------------------------------- #
class TemporalQuad(BaseModel):
    """Legacy-compatible timestamped fact (subject, relation, object, timestamp).

    Retained so archived TKG experiments and old artifacts remain readable. The v4
    headline uses CGEP event instances from ``finekg.succession`` instead.
    `timestamp` is an integer time index (days/quarters) for chronological
    ordering.
    """

    subject: str
    relation: str
    object: str
    timestamp: int = Field(ge=0)

    def ref(self) -> str:
        """Compact reference used in evidence chains."""
        return f"({self.subject}, {self.relation}, {self.object}, t={self.timestamp})"


class ForecastQuery(BaseModel):
    """Legacy-compatible link-prediction query: (subject, relation, ?, timestamp).

    `gold_object` is the held-out answer for evaluation. `candidates` optionally
    restricts the answer space (else the full entity vocabulary is ranked).
    """

    subject: str
    relation: str
    timestamp: int = Field(ge=0)
    gold_object: str | None = None
    candidates: list[str] | None = None
    query_id: str = ""


class RankedCandidate(BaseModel):
    object: str
    score: float


class EvidenceLink(BaseModel):
    """One step of an evidence chain: a graph element plus why it matters."""

    reference: str  # event_id, edge key, or quad string
    spans: list[EvidenceSpan] = Field(default_factory=list)
    reason: str = ""
    # How much ablating this link degrades the forecast — the unit of
    # intervention-based path faithfulness (set by the faithfulness verifier).
    intervention_delta: float | None = None


class Prediction(BaseModel):
    """A ranked forecast with an interpretable evidence chain."""

    query_id: str
    ranked: list[RankedCandidate] = Field(default_factory=list)
    evidence_chain: list[EvidenceLink] = Field(default_factory=list)
    # Selective-prediction annotations (optional; set by the calibrator agent).
    # `faithfulness` is the path-faithfulness driving abstention; `abstained`
    # marks a withheld forecast; `coverage_set` is the conformal answer set.
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool = False
    coverage_set: list[str] | None = None

    @property
    def top1(self) -> str | None:
        return self.ranked[0].object if self.ranked else None
