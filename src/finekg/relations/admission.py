"""Risk-controlled edge admission via Conformal Risk Control.

Turns graph construction into a guarantee-carrying decision. Picking an admission
threshold ``tau`` on edge scores so the constructed graph provably retains at
least ``1 - alpha`` of the gold relations (false-negative rate ``<= alpha``) is
the recall analogue of the FNR example in Conformal Risk Control (Angelopoulos
et al., 2022), and the relation-stage half of the verifier's third duty — risk
control — that `core.calibration` brings to forecasting.

The threshold is calibrated on held-out documents: for every gold edge we read
the model's score (its `confidence`, or the verifier `faithfulness`; ``0`` if
the edge was never proposed), then choose the *tightest* ``tau`` whose
CRC-bounded FNR is ``<= alpha`` — the most selective graph that still meets the
recall guarantee. At inference, every proposed edge with score ``>= tau`` is
admitted; the rest are dropped, lifting precision without breaching the bound.

Pure-Python / CPU; selected by registry name (`edge_admission`) like the other
swappable components, so adding a guarantee is a config change, not a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from finekg.core.calibration import conformal_risk_threshold
from finekg.core.eval.relation import relation_prf
from finekg.core.registry import Registry
from finekg.core.schema import EventGraph, RelationEdge

__all__ = [
    "edge_admission",
    "EdgeAdmission",
    "CRCEdgeAdmission",
    "PassthroughAdmission",
    "edge_score",
    "gold_edge_scores",
    "admission_report",
]

edge_admission: Registry[EdgeAdmission] = Registry("edge_admission")


def _key(edge: RelationEdge) -> tuple[str, str, str, str]:
    """Order-invariant identity for symmetric edges (matches `relation_prf`)."""
    head, tail = edge.head_id, edge.tail_id
    if not edge.directed and head > tail:
        head, tail = tail, head
    return (head, tail, edge.relation_type.value, edge.subtype)


def edge_score(edge: RelationEdge, field: str = "confidence") -> float:
    """Read an edge's admission score; a missing/None score reads as 0."""
    value = getattr(edge, field, None)
    return float(value) if value is not None else 0.0


def gold_edge_scores(
    predicted: Iterable[RelationEdge],
    gold: Iterable[RelationEdge],
    field: str = "confidence",
) -> list[float]:
    """The model's score for each gold edge (0 if it was never proposed).

    These are the calibration nonconformity inputs: a gold edge the model scored
    highly is easy to retain, a low/zero one is a miss waiting to happen.
    """
    best: dict[tuple[str, str, str, str], float] = {}
    for e in predicted:
        k = _key(e)
        best[k] = max(best.get(k, 0.0), edge_score(e, field))
    return [best.get(_key(g), 0.0) for g in gold]


class EdgeAdmission(ABC):
    """Selects an admission threshold and applies it to a graph."""

    score_field: str = "confidence"

    @abstractmethod
    def fit(self, gold_scores: Sequence[float]) -> EdgeAdmission:
        """Calibrate the threshold from per-gold-edge model scores; returns self."""

    @abstractmethod
    def threshold(self) -> float:
        """The calibrated admission threshold ``tau``."""

    def apply(self, graph: EventGraph) -> EventGraph:
        """Return a copy keeping only edges scoring ``>= tau`` (marked admitted).

        Immutable in spirit: the input graph is untouched; admitted edges are
        copies with ``admitted=True`` and the rest are dropped.
        """
        tau = self.threshold()
        kept = [
            e.model_copy(update={"admitted": True})
            for e in graph.edges
            if edge_score(e, self.score_field) >= tau
        ]
        return EventGraph(
            nodes=dict(graph.nodes),
            edges=kept,
            metadata={
                **graph.metadata,
                "admission": type(self).__name__,
                "admission_tau": f"{tau:.4f}",
                "edges_dropped_low_score": str(len(graph.edges) - len(kept)),
            },
        )


@edge_admission.register("crc")
class CRCEdgeAdmission(EdgeAdmission):
    """Conformal Risk Control admission: bound the gold-edge FNR at ``alpha``."""

    def __init__(self, alpha: float = 0.1, score_field: str = "confidence") -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.score_field = score_field
        self._tau = 0.0

    def fit(self, gold_scores: Sequence[float]) -> CRCEdgeAdmission:
        scores = [float(s) for s in gold_scores]
        n = len(scores)
        if n == 0:
            self._tau = 0.0  # no calibration data -> admit all (no guarantee)
            return self
        # Candidate thresholds = distinct gold scores, tight (high) -> loose (low);
        # FNR(tau) = fraction of gold below tau, non-increasing along that order.
        lambdas = sorted(set(scores), reverse=True)
        mean_losses = [sum(1.0 for s in scores if s < tau) / n for tau in lambdas]
        self._tau = conformal_risk_threshold(lambdas, mean_losses, n, self.alpha, 1.0)
        return self

    def threshold(self) -> float:
        return self._tau


@edge_admission.register("none")
class PassthroughAdmission(EdgeAdmission):
    """Admit every proposed edge (no guarantee) — the ablation baseline."""

    def __init__(self, score_field: str = "confidence") -> None:
        self.score_field = score_field

    def fit(self, gold_scores: Sequence[float]) -> PassthroughAdmission:
        _ = gold_scores
        return self

    def threshold(self) -> float:
        return 0.0


def admission_report(
    admitted: Iterable[RelationEdge], gold: Iterable[RelationEdge]
) -> dict[str, float]:
    """Precision / recall / FNR of an admitted edge set against gold."""
    micro = relation_prf(admitted, gold)["micro"]
    return {
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "fnr": 1.0 - float(micro["recall"]),
        "f1": float(micro["f1"]),
    }
