"""finekg.core — frozen cross-stage contracts, graph algorithms, metrics, registry.

This is the stable foundation every stage depends on. Import data contracts and
shared utilities from here.
"""

from finekg.core.registry import Registry
from finekg.core.schema import (
    EventGraph,
    EventNode,
    EvidenceLink,
    EvidenceSpan,
    ForecastQuery,
    Prediction,
    RankedCandidate,
    RelationEdge,
    RelationType,
    TemporalQuad,
)

__all__ = [
    "EvidenceSpan",
    "EventNode",
    "RelationType",
    "RelationEdge",
    "EventGraph",
    "TemporalQuad",
    "ForecastQuery",
    "RankedCandidate",
    "EvidenceLink",
    "Prediction",
    "Registry",
]

__version__ = "0.1.0"
