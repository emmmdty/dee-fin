"""DsGL must match SeDGPL exactly, or M1's comparison against it means nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from finekg.succession.data.cgep import CgepInstance, CgepNode
from finekg.succession.data.esc import load_cgep_esc
from finekg.succession.linearize import (
    EDGE_BUDGET,
    UNREACHABLE_DISTANCE,
    EventVocabulary,
    edge_distances,
    edge_selectors,
    linearize,
    select_nearest_edges,
    truncate_edges,
)

REAL_ESC = Path("data/raw/sedgpl_esc/ESCSubWoRe.npy")


def _node(i: int) -> CgepNode:
    return CgepNode(node_id=f"n{i}", event_type=f"T{i}", trigger=f"e{i}", sentence=f"s{i}")


def _instance(edges: list[tuple[int, str, int]], n_nodes: int = 6) -> CgepInstance:
    nodes = tuple(_node(i) for i in range(n_nodes))
    return CgepInstance(
        instance_id="x",
        doc_id="d",
        nodes=nodes,
        edges=tuple(edges),
        candidates=(nodes[edges[-1][2]],),
        label=0,
    )


def test_truncation_takes_the_first_k_in_stored_order_not_the_nearest():
    # SeDGPL slices the head of the edge list. Nothing about distance enters here
    # -- that is precisely the arbitrariness M1 replaces.
    edges = [(i, "cause", i + 1) for i in range(25)]
    kept = truncate_edges(edges, max_edges=EDGE_BUDGET)
    assert kept == edges[:20]
    assert edges[-1] not in kept  # the query edge is never in the template


def test_truncation_drops_only_the_query_edge_when_under_budget():
    edges = [(0, "cause", 1), (1, "cause", 2), (2, "cause", 3)]
    assert truncate_edges(edges, max_edges=EDGE_BUDGET) == edges[:-1]


def test_distance_is_bfs_from_both_endpoints_of_the_query_edge():
    #  0 -> 1 -> 2 -> 3 (query: 2 -> 3).  Endpoints 2 and 3 sit at distance 0.
    template = [(0, "cause", 1), (1, "cause", 2)]
    query = (2, "cause", 3)
    # edge (1,2): min(d[1], d[2]) = min(1, 0) = 0.  edge (0,1): min(2, 1) = 1.
    assert edge_distances(template, query) == [1, 0]


def test_unreachable_edges_get_the_hardcoded_sentinel():
    template = [(4, "cause", 5)]  # a component disjoint from the query edge
    assert edge_distances(template, (0, "cause", 1)) == [UNREACHABLE_DISTANCE]


def test_nearest_edges_land_adjacent_to_the_mask():
    instance = _instance([(0, "CAUSE", 1), (1, "CAUSE", 2), (2, "CAUSE", 3)])
    result = linearize(instance)
    # Ordered far -> near, then the query edge last.
    assert result.edges == ((0, "CAUSE", 1), (1, "CAUSE", 2), (2, "CAUSE", 3))
    assert result.template == "e0 causes e1 , e1 causes e2 , e2 causes <mask> ."
    assert result.type_template == "T0 causes T1 , T1 causes T2 , T2 causes <mask> ."
    assert result.dropped == 0


def test_unmapped_relations_are_spliced_verbatim():
    # ESC stores the literal 'cause'; SeDGPL splices `rel[1]` straight into the
    # template. Rewriting it to "causes" would silently desync the anchor.
    instance = _instance([(0, "cause", 1), (1, "cause", 2)])
    assert linearize(instance).template == "e0 cause e1 , e1 cause <mask> ."


def test_dropped_count_reports_what_the_budget_discarded():
    edges = [(i, "cause", i + 1) for i in range(25)]
    result = linearize(_instance(edges, n_nodes=26), max_edges=EDGE_BUDGET)
    assert len(result.edges) == 21  # 20 template + query
    assert result.dropped == 24 - 20


def test_vocabulary_gives_one_token_per_distinct_mention_then_per_type():
    instance = _instance([(0, "cause", 1), (1, "cause", 2)])
    vocab = EventVocabulary.build([instance])
    assert vocab.token("e0") == "<a_0>"
    # Types are claimed only after every mention has a token.
    assert vocab.token("T0").startswith("<a_")
    assert vocab.token("e0") != vocab.token("T0")
    assert vocab.to_add[vocab.token("e0")] == "e0"
    assert vocab.token("never seen") == "never seen"


def test_vocabulary_renders_multiword_triggers_as_single_tokens():
    # SeDGPL asserts the text and type templates tokenize to equal length. A raw
    # multi-word trigger ("checks into") would break that; a `<a_i>` cannot.
    nodes = (
        CgepNode(node_id="n0", event_type="ACTION", trigger="checks into ", sentence=""),
        CgepNode(node_id="n1", event_type="STATE", trigger="stays ", sentence=""),
        CgepNode(node_id="n2", event_type="ACTION", trigger="leaves ", sentence=""),
    )
    instance = CgepInstance(
        instance_id="x", doc_id="d", nodes=nodes,
        edges=((0, "cause", 1), (1, "cause", 2)), candidates=(nodes[2],), label=0,
    )
    vocab = EventVocabulary.build([instance])
    result = linearize(instance, vocab)
    assert len(result.template.split(" ")) == len(result.type_template.split(" "))
    assert "checks into" not in result.template


@pytest.mark.skipif(not REAL_ESC.exists(), reason="ESCSubWoRe.npy not downloaded")
def test_linearizing_the_released_esc_build_upholds_sedgpls_own_assertion():
    instances = [i for topic in load_cgep_esc(REAL_ESC).values() for i in topic]
    vocab = EventVocabulary.build(instances)
    truncated = 0
    for instance in instances:
        result = linearize(instance, vocab)
        # `tools.getTemplate`: assert len(template.split(' ')) == len(templateType.split(' '))
        assert len(result.template.split(" ")) == len(result.type_template.split(" "))
        # `assert data['edge'][-1] not in relation`
        assert result.query_edge == instance.query_edge
        assert result.query_edge not in result.edges[:-1]
        assert len(result.edges) <= EDGE_BUDGET + 1
        truncated += result.dropped > 0
    # 463 of 1192 ESC instances (38.8%) lose edges to the 20-edge budget, and
    # which ones go is decided by storage order alone. That is the limitation M1
    # exists to address; pin the rate so a regression in DsGL stays visible.
    assert 0.35 < truncated / len(instances) < 0.42


# --- M1: risk-aware linearisation (distance-selected truncation) ---------------


def test_distance_selector_keeps_the_edges_nearest_the_query():
    # A chain 0->1->...->25 with the query at the tail. SeDGPL's storage-order
    # slice keeps the 20 *farthest* edges and discards the 5 nearest the query;
    # M1 keeps the 20 nearest instead, over the same budget.
    edges = [(i, "cause", i + 1) for i in range(25)]
    kept = select_nearest_edges(edges, max_edges=EDGE_BUDGET)
    assert kept == edges[4:24]  # (4,5)..(23,24): the 20 nearest the query (24,25)
    assert truncate_edges(edges, max_edges=EDGE_BUDGET) == edges[:20]  # the farthest 20
    assert edges[-1] not in kept  # the query edge is never in the template


def test_distance_selector_drops_only_the_query_edge_when_under_budget():
    # Under budget there is nothing to choose, so it matches SeDGPL exactly.
    edges = [(0, "cause", 1), (1, "cause", 2), (2, "cause", 3)]
    assert select_nearest_edges(edges, max_edges=EDGE_BUDGET) == edges[:-1]


def test_distance_selection_scores_over_the_whole_graph_not_the_stored_head():
    # A near edge stored *last* must still survive: distance is measured over the
    # full template, not the truncated head SeDGPL would have kept.
    far = [(i, "cause", i + 1) for i in range(2, 22)]  # 20 far edges, stored first
    near = (1, "cause", 2)  # distance 0 from the query, stored last
    query = (0, "cause", 1)
    edges = [*far, near, query]
    kept = select_nearest_edges(edges, max_edges=EDGE_BUDGET)
    assert near in kept  # survives despite being last in stored order
    assert near not in truncate_edges(edges, max_edges=EDGE_BUDGET)  # slice drops it
    assert query not in kept


def test_linearize_threads_the_selector_and_swaps_the_survivors():
    edges = [(i, "cause", i + 1) for i in range(25)]
    instance = _instance(edges, n_nodes=26)
    near = linearize(instance, max_edges=EDGE_BUDGET, selector=select_nearest_edges)
    default = linearize(instance, max_edges=EDGE_BUDGET)  # SeDGPL storage-order slice
    assert set(near.edges[:-1]) != set(default.edges[:-1])  # different survivors
    assert near.query_edge == default.query_edge == instance.query_edge
    assert near.dropped == default.dropped == 24 - 20  # same budget, same drop count


def test_edge_selector_registry_resolves_both_policies_by_name():
    # config selects the policy by name, like every other swappable component.
    assert edge_selectors.create("sedgpl") is truncate_edges
    assert edge_selectors.create("distance") is select_nearest_edges
    assert {"sedgpl", "distance"} <= set(edge_selectors.available())
