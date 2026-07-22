"""Risk-controlled edge admission must carry a recall guarantee: the calibrated
threshold keeps the gold false-negative rate at or below alpha, while dropping
low-confidence edges to lift precision. Checked deterministically on CPU.
"""

from __future__ import annotations

from finekg.core.schema import EventGraph, EventNode, RelationEdge, RelationType
from finekg.relations.admission import (
    CRCEdgeAdmission,
    PassthroughAdmission,
    admission_report,
    edge_admission,
    gold_edge_scores,
)


def _edge(head: str, tail: str, conf: float, subtype: str = "CAUSE") -> RelationEdge:
    return RelationEdge(
        head_id=head,
        tail_id=tail,
        relation_type=RelationType.CAUSAL,
        subtype=subtype,
        confidence=conf,
    )


def test_registry_exposes_admission_strategies() -> None:
    assert set(edge_admission.available()) == {"crc", "none"}


def test_crc_picks_tightest_threshold_meeting_the_fnr_bound() -> None:
    # 20 gold scores: 16 strong, 2 mid, 2 weak. alpha=0.2.
    # tau=0.9 -> FNR 0.20 -> bound 0.238 (fail); tau=0.5 -> FNR 0.10 -> bound 0.143 (pass).
    gold_scores = [0.9] * 16 + [0.5] * 2 + [0.1] * 2
    crc = CRCEdgeAdmission(alpha=0.2).fit(gold_scores)
    assert crc.threshold() == 0.5
    fnr = sum(1 for s in gold_scores if s < crc.threshold()) / len(gold_scores)
    assert fnr <= 0.2


def test_apply_retains_recall_and_lifts_precision() -> None:
    # Gold edges, all proposed with high confidence; plus fabricated low-confidence
    # edges absent from gold. Admission should drop the fabrications.
    gold = [_edge(f"e{i}", f"e{i + 1}", 0.9) for i in range(10)]
    fabricated = [_edge(f"x{i}", f"y{i}", 0.2) for i in range(10)]
    nodes = {
        n: EventNode(event_id=n, event_type="Acq", doc_id="d")
        for e in gold + fabricated
        for n in (e.head_id, e.tail_id)
    }
    graph = EventGraph(nodes=nodes, edges=gold + fabricated)

    crc = CRCEdgeAdmission(alpha=0.1).fit(gold_edge_scores(gold + fabricated, gold))
    admitted_graph = crc.apply(graph)

    before = admission_report(graph.edges, gold)
    after = admission_report(admitted_graph.edges, gold)
    assert after["recall"] >= 1.0 - 0.1  # recall guarantee holds
    assert after["precision"] > before["precision"]  # fabrications dropped
    assert all(e.admitted for e in admitted_graph.edges)
    # the original graph is untouched (immutable application)
    assert len(graph.edges) == 20


def test_gold_edge_scores_marks_unproposed_gold_as_zero() -> None:
    gold = [_edge("a", "b", 0.9), _edge("c", "d", 0.9)]
    predicted = [_edge("a", "b", 0.8)]  # only the first gold edge was proposed
    scores = gold_edge_scores(predicted, gold)
    assert scores == [0.8, 0.0]


def test_passthrough_admits_everything() -> None:
    edges = [_edge("a", "b", 0.1), _edge("c", "d", 0.9)]
    nodes = {n: EventNode(event_id=n, event_type="x", doc_id="d") for n in ("a", "b", "c", "d")}
    graph = EventGraph(nodes=nodes, edges=edges)
    out = PassthroughAdmission().fit([]).apply(graph)
    assert len(out.edges) == 2


def test_empty_calibration_admits_all() -> None:
    assert CRCEdgeAdmission(alpha=0.1).fit([]).threshold() == 0.0
