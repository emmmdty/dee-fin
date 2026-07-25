"""ECG reconstructability: does the *constructed* (predicted) graph still let Ch4
pose its successor-prediction queries, and how faithfully?

R1 (reachability) is the per-query-edge flag CS-CRP reserves budget for; R2
(query-edge fidelity) is the precision-aware view where repair/admission can help.
Checked deterministically on CPU against the authoritative CGEP construction.
"""

from __future__ import annotations

import pytest

from finekg.core.schema import EventGraph, EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument
from finekg.succession.reconstruction import ecg_reachable_flags, reconstruction_report


def _node(doc_id: str, key: str, trigger: str, sent_id: int) -> EventNode:
    return EventNode(
        event_id=f"{doc_id}::{key}",
        event_type=f"Type_{trigger}",
        doc_id=doc_id,
        trigger=trigger,
        trigger_evidence=[
            EvidenceSpan(doc_id=doc_id, char_start=0, char_end=len(trigger),
                         sent_id=sent_id, text=trigger)
        ],
    )


def _edge(doc_id: str, head: str, tail: str, kind: RelationType, subtype: str) -> RelationEdge:
    return RelationEdge(
        head_id=f"{doc_id}::{head}", tail_id=f"{doc_id}::{tail}",
        relation_type=kind, subtype=subtype,
    )


# Gold doc: the ALPHA topology from test_cgep_build. Its single ECG query edge is
# m2 -CAUSE-> m4 ("arrest": the pendant sink, out-degree 0 / in-degree 1).
_GOLD_TOPOLOGY = [
    ("m1", "m2", RelationType.CAUSAL, "CAUSE"),
    ("m2", "m3", RelationType.CAUSAL, "PRECONDITION"),
    ("m2", "m4", RelationType.CAUSAL, "CAUSE"),
    ("m3", "m5", RelationType.SUBEVENT, "SUBEVENT_OF"),
    ("m1", "m5", RelationType.CAUSAL, "CAUSE"),
]


def _document() -> RelationDocument:
    triggers = ["attack", "riot", "march", "arrest", "trial"]
    keys = [f"m{i}" for i in range(1, 6)]
    nodes = [_node("docA", k, t, i) for i, (k, t) in enumerate(zip(keys, triggers, strict=True))]
    edges = [_edge("docA", h, t, k, s) for h, t, k, s in _GOLD_TOPOLOGY]
    edges.append(_edge("docA", "m1", "m3", RelationType.TEMPORAL, "BEFORE"))  # out of topology
    return RelationDocument(
        doc_id="docA", nodes=nodes, gold_edges=edges,
        doc_text="\n".join(f"sentence {i} mentions {t}" for i, t in enumerate(triggers)),
        representative={f"E{i}": n.event_id for i, n in enumerate(nodes)},
    )


def _pred_graph(doc: RelationDocument, triples: list) -> EventGraph:
    return EventGraph(
        nodes={n.event_id: n for n in doc.nodes},
        edges=[_edge(doc.doc_id, h, t, k, s) for h, t, k, s in triples],
    )


def test_full_topology_reconstructs_every_query_edge() -> None:
    doc = _document()
    predicted = _pred_graph(doc, _GOLD_TOPOLOGY)

    report = reconstruction_report(doc, predicted)

    assert report["n_gold_queries"] == 1
    assert report["r1_reachability_rate"] == pytest.approx(1.0)
    assert ecg_reachable_flags(doc, predicted) == [True]
    # predicted topology == gold topology -> identical query edges
    assert report["r2_query_prf"]["f1"] == pytest.approx(1.0)


def test_missing_query_edge_is_unreachable() -> None:
    doc = _document()
    # Drop exactly the query edge m2 -CAUSE-> m4.
    predicted = _pred_graph(doc, [t for t in _GOLD_TOPOLOGY if t[:2] != ("m2", "m4")])

    report = reconstruction_report(doc, predicted)

    assert report["r1_reachability_rate"] == pytest.approx(0.0)
    assert ecg_reachable_flags(doc, predicted) == [False]
    # the gold query edge is not recovered
    assert report["r2_query_prf"]["recall"] == pytest.approx(0.0)


def test_reachable_flags_length_matches_query_count_and_partial_rate() -> None:
    # A second pendant sink m6 off m2 gives the ECG two query edges (m4, m6).
    doc = _document()
    doc.nodes.append(_node("docA", "m6", "verdict", 5))
    doc.representative["E5"] = "docA::m6"
    doc.gold_edges.append(_edge("docA", "m2", "m6", RelationType.CAUSAL, "CAUSE"))

    predicted = _pred_graph(doc, _GOLD_TOPOLOGY)  # has m2->m4 but not m2->m6

    flags = ecg_reachable_flags(doc, predicted)
    assert len(flags) == 2  # one reachable flag per gold query edge (CS-CRP bridge)
    assert sum(flags) == 1
    assert reconstruction_report(doc, predicted)["r1_reachability_rate"] == pytest.approx(0.5)
