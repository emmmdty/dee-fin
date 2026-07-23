"""CPU tests for the discriminative supervised relation extractor (Phase A).

Everything here runs without torch: registry wiring, lazy-import discipline,
document-level candidate enumeration, and edge construction from injected
scores (the model itself is torch-guarded and tested under `pytest.importorskip`).
"""

from __future__ import annotations

import finekg.relations.extractor.supervised as sup
from finekg.core.schema import EventNode, EvidenceSpan
from finekg.relations.extractor import relation_extractors


def _node(eid: str, sent: int, start: int, etype: str = "Attack") -> EventNode:
    return EventNode(
        event_id=eid,
        event_type=etype,
        doc_id="d1",
        trigger=eid,
        trigger_evidence=[
            EvidenceSpan(doc_id="d1", char_start=start, char_end=start + 1, sent_id=sent)
        ],
    )


def test_supervised_registered():
    assert "supervised" in relation_extractors


def test_module_imports_without_torch():
    # Lazy-import discipline: the module must not bind torch at import time,
    # so the whole package imports on a CPU-only machine without the llm extra.
    assert not hasattr(sup, "torch")


def test_candidate_pairs_document_level_all_ordered_pairs():
    ex = relation_extractors.create("supervised")
    nodes = [_node("a", 0, 0), _node("b", 0, 5), _node("c", 1, 0)]
    pairs = ex._candidate_pairs(nodes)
    # Document-level = all ordered mention pairs (both directions), no self-pairs.
    assert len(pairs) == 6
    assert ("a", "b") in pairs and ("b", "a") in pairs
    assert all(h != t for h, t in pairs)


def test_extract_builds_grounded_edges_from_scores(monkeypatch):
    ex = relation_extractors.create("supervised")
    nodes = [_node("a", 0, 0), _node("b", 1, 0)]

    def fake_scores(self, ns, pairs, context):
        return {("a", "b"): {"causal": ("CAUSE", 0.9)}}  # only a->b causal

    monkeypatch.setattr(sup.SupervisedRelationExtractor, "_score_pairs", fake_scores)
    edges = ex.extract(nodes)
    assert len(edges) == 1
    e = edges[0]
    assert (e.head_id, e.tail_id) == ("a", "b")
    assert e.relation_type.value == "causal" and e.subtype == "CAUSE"
    assert e.directed is True
    assert abs(e.confidence - 0.9) < 1e-6
    assert len(e.evidence) >= 1  # grounded in the endpoints' trigger spans


def test_extract_no_prediction_yields_no_edge(monkeypatch):
    ex = relation_extractors.create("supervised")
    monkeypatch.setattr(
        sup.SupervisedRelationExtractor,
        "_score_pairs",
        lambda self, ns, pairs, context: {},
    )
    assert ex.extract([_node("a", 0, 0), _node("b", 1, 0)]) == []
