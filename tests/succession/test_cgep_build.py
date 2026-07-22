"""CGEP rebuild contract.

The load-bearing invariant is that the gold successor may occur *only* in the
query edge: SeDGPL renders every template edge as a pair of event tokens, so a
gold event appearing elsewhere would print its own answer token into the prompt.
That is what pins the query-edge rule to ``outdeg == 0 and indeg == 1``.
"""

from __future__ import annotations

import pytest

from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.data.maven_ere import RelationDocument, load_maven_ere
from finekg.succession.data.cgep import build_cgep, extract_ecgs, query_edge_indices


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
        head_id=f"{doc_id}::{head}",
        tail_id=f"{doc_id}::{tail}",
        relation_type=kind,
        subtype=subtype,
    )


def _document(doc_id: str, triggers: list[str]) -> RelationDocument:
    """5-event doc. n5 is a sink with in-degree 2, so it must never be gold.

    n1 -CAUSE-> n2 -PRECONDITION-> n3 -SUBEVENT_OF-> n5
                n2 -CAUSE-------> n4          n1 -CAUSE-> n5
    plus a TEMPORAL BEFORE edge that must stay out of the topology.
    """
    keys = [f"m{i}" for i in range(1, 6)]
    nodes = [_node(doc_id, k, t, i) for i, (k, t) in enumerate(zip(keys, triggers, strict=True))]
    edges = [
        _edge(doc_id, "m1", "m2", RelationType.CAUSAL, "CAUSE"),
        _edge(doc_id, "m2", "m3", RelationType.CAUSAL, "PRECONDITION"),
        _edge(doc_id, "m2", "m4", RelationType.CAUSAL, "CAUSE"),
        _edge(doc_id, "m3", "m5", RelationType.SUBEVENT, "SUBEVENT_OF"),
        _edge(doc_id, "m1", "m5", RelationType.CAUSAL, "CAUSE"),
        _edge(doc_id, "m1", "m3", RelationType.TEMPORAL, "BEFORE"),
    ]
    return RelationDocument(
        doc_id=doc_id,
        nodes=nodes,
        gold_edges=edges,
        doc_text="\n".join(f"sentence {i} mentions {t}" for i, t in enumerate(triggers)),
        representative={f"E{i}": n.event_id for i, n in enumerate(nodes)},
    )


ALPHA = _document("docA", ["attack", "riot", "march", "arrest", "trial"])
BETA = _document("docB", ["flood", "evacuate", "rescue", "rebuild", "inquiry"])


def test_temporal_edges_stay_out_of_the_topology():
    (ecg,) = extract_ecgs(ALPHA)
    assert len(ecg.nodes) == 5
    assert {rel for _, rel, _ in ecg.edges} == {"CAUSE", "PRECONDITION", "SUBEVENT_OF"}
    assert len(ecg.edges) == 5  # the BEFORE edge is dropped


def test_query_edge_is_the_only_occurrence_of_its_tail():
    (ecg,) = extract_ecgs(ALPHA)
    triggers = [n.trigger for n in ecg.nodes]
    positions = query_edge_indices(ecg)
    assert len(positions) == 1
    _, _, tail = ecg.edges[positions[0]]
    # "arrest" is the pendant sink; "trial" is a sink too but has in-degree 2.
    assert triggers[tail] == "arrest"


def test_gold_never_appears_in_a_template_edge():
    instances, _ = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=4)
    assert instances
    for instance in instances:
        gold = instance.gold_index
        assert all(gold not in (h, t) for h, _, t in instance.template_edges)
        assert instance.query_edge is instance.edges[-1]
        assert instance.nodes[gold].trigger == instance.gold_trigger


def test_candidates_hold_gold_and_do_not_exclude_the_source_graph():
    # SeDGPL's released ESC build draws candidates corpus-wide *without* removing
    # the source graph's own nodes: ~17% of a graph's nodes reappear as
    # candidates, which is what uniform sampling produces. Only gold is pinned.
    instances, _ = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=512)
    alpha = next(i for i in instances if i.doc_id == "docA")
    assert alpha.candidates[alpha.label].node_id == alpha.nodes[alpha.gold_index].node_id
    ids = [c.node_id for c in alpha.candidates]
    assert len(ids) == len(set(ids))  # distinct node identities
    # Pool is both documents' nodes; the source graph's are eligible negatives.
    pool_ids = {n.event_id for n in ALPHA.nodes} | {n.event_id for n in BETA.nodes}
    assert set(ids) <= pool_ids
    assert set(ids) & {n.event_id for n in ALPHA.nodes}


def test_candidate_triggers_may_collide_so_answers_are_fewer_than_candidates():
    # Two events share the trigger "flood"; SeDGPL scores by mention token id, so
    # they collapse to one answer. The instance must expose that, not hide it.
    twin = _document("docC", ["flood", "flood", "siren", "curfew", "review"])
    instances, stats = build_cgep([twin], min_nodes=4, n_candidates=512)
    inst = instances[0]
    assert len(inst.candidates) == 5
    assert inst.distinct_answers == 4
    assert stats["distinct_answers"] == pytest.approx(4.0)


def test_build_is_deterministic_under_a_seed():
    first, _ = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=4, seed=7)
    second, _ = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=4, seed=7)
    other, _ = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=4, seed=8)
    assert [i.candidates for i in first] == [i.candidates for i in second]
    assert [i.candidates for i in first] != [i.candidates for i in other]


def test_candidate_set_degrades_gracefully_when_the_pool_is_exhausted():
    instances, _ = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=512)
    alpha = next(i for i in instances if i.doc_id == "docA")
    # Pool is the 10 nodes of both documents; a 512-way set cannot be filled.
    assert len(alpha.candidates) == 10
    assert alpha.candidates[alpha.label].trigger == alpha.gold_trigger


def test_min_nodes_filters_small_components():
    assert extract_ecgs(ALPHA, min_nodes=5)
    assert not extract_ecgs(ALPHA, min_nodes=6)


def test_dropping_subevent_changes_the_topology():
    (with_sub,) = extract_ecgs(ALPHA)
    (without,) = extract_ecgs(ALPHA, include_subevent=False)
    assert len(with_sub.edges) == 5
    assert len(without.edges) == 4
    assert "SUBEVENT_OF" not in {rel for _, rel, _ in without.edges}


def test_stats_are_reported_per_ecg_not_per_document():
    _, stats = build_cgep([ALPHA, BETA], min_nodes=4, n_candidates=4)
    assert stats["documents"] == 2.0
    assert stats["ecgs"] == 2.0
    assert stats["nodes_per_ecg"] == pytest.approx(5.0)
    assert stats["edges_per_ecg"] == pytest.approx(5.0)


def test_real_fixture_parses_through_the_loader(fixtures_dir):
    docs = list(load_maven_ere(fixtures_dir / "maven_ere" / "sample_with_text.jsonl"))
    assert docs
    # The 2-line fixture is far below the real threshold; only the parse path
    # is under test here, so relax `min_nodes` to reach any component at all.
    graphs = [g for doc in docs for g in extract_ecgs(doc, min_nodes=2)]
    assert graphs
    for graph in graphs:
        assert all(node.trigger for node in graph.nodes)
        assert all(0 <= h < len(graph.nodes) and 0 <= t < len(graph.nodes)
                   for h, _, t in graph.edges)
