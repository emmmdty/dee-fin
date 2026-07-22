"""M2 structural signal: does each event causally reach the anchor?

`reach_anchor` is the fourth EeCE stream (SPEC 4.2): 1 if an event reaches the
query anchor over the *directed* surviving edges -- i.e. it is upstream causal
evidence for what is being predicted -- else 0. Pure graph facts, CPU-only, so
these pin the semantics without torch.
"""

from __future__ import annotations

from finekg.succession.structure import event_reach_anchor, reaches_anchor


def test_reaches_anchor_collects_upstream_events_and_the_anchor_itself():
    # 0 -> 2, 1 -> 2, 2 -> 3.  Reaching anchor 2: 0, 1, and 2 itself.
    edges = ((0, "CAUSE", 2), (1, "CAUSE", 2), (2, "CAUSE", 3))
    assert reaches_anchor(edges, anchor=2) == {0, 1, 2}


def test_reaches_anchor_excludes_events_downstream_of_the_anchor():
    # 3 sits *after* the anchor (2 -> 3), so it does not reach 2.
    edges = ((2, "CAUSE", 3),)
    assert reaches_anchor(edges, anchor=2) == {2}


def test_event_reach_anchor_flags_each_token_by_its_node():
    edges = ((0, "CAUSE", 2), (1, "CAUSE", 2), (2, "CAUSE", 3))
    # token order carries repeats and a downstream node (3); each maps to its own flag
    nodes = [0, 2, 1, 2, 3]
    assert event_reach_anchor(edges, nodes, anchor=2) == [1, 1, 1, 1, 0]


def test_event_reach_anchor_follows_multi_hop_causal_chains():
    # 0 -> 1 -> 2: node 0 reaches anchor 2 only transitively.
    edges = ((0, "CAUSE", 1), (1, "CAUSE", 2))
    assert event_reach_anchor(edges, [0, 1, 2], anchor=2) == [1, 1, 1]


def test_anchor_with_only_outgoing_edges_flags_only_itself():
    edges = ((2, "CAUSE", 3), (2, "PRECONDITION", 4))
    assert event_reach_anchor(edges, [2, 3, 4], anchor=2) == [1, 0, 0]


def test_reachability_ignores_the_relation_label():
    # subevent / precondition / cause all carry the same head->tail direction.
    edges = ((0, "SUBEVENT_OF", 1), (1, "PRECONDITION", 2))
    assert event_reach_anchor(edges, [0, 1], anchor=2) == [1, 1]
