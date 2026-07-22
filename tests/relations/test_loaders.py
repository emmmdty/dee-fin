"""Tests for the relation dataset loaders against the bundled fixtures."""

from __future__ import annotations

from finekg.core.eval import relation_prf
from finekg.core.schema import RelationType
from finekg.relations.data import load_ccks_causal, load_maven_ere


def test_load_maven_ere_fixture(fixtures_dir) -> None:
    docs = list(load_maven_ere(fixtures_dir / "maven_ere" / "sample.jsonl"))
    assert len(docs) == 2
    doc1 = docs[0]
    types = {e.relation_type for e in doc1.gold_edges}
    # doc1 has coreference (EV1 has 2 mentions), temporal, causal and subevent.
    assert RelationType.COREFERENCE in types
    assert RelationType.TEMPORAL in types
    assert RelationType.CAUSAL in types
    # Gold-vs-gold is a perfect score (loader + metric integrate cleanly).
    assert relation_prf(doc1.gold_edges, doc1.gold_edges)["micro"]["f1"] == 1.0


def test_load_ccks_causal_fixture(fixtures_dir) -> None:
    docs = list(load_ccks_causal(fixtures_dir / "ccks_fin_causal" / "sample.jsonl"))
    assert len(docs) == 2
    assert all(
        any(e.relation_type is RelationType.CAUSAL for e in d.gold_edges) for d in docs
    )
    assert docs[0].nodes[0].doc_id == docs[0].doc_id
