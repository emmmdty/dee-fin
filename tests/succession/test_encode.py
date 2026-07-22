"""Batch encoding: the i-th event token must carry the i-th event's sentence.

SeDGPL only checks `len(sentId) - 1 == len(ePosition)`, a count that passes even
when the pairing is shifted by one. These tests check the identity.
"""

from __future__ import annotations

import pytest

from finekg.succession.data.cgep import CgepInstance, CgepNode, token_span
from finekg.succession.encode import (
    encode_instance,
    event_token_nodes,
    replace_mention,
)
from finekg.succession.linearize import EventVocabulary, linearize, select_nearest_edges


def _node(i: int, trigger: str, sentence: str) -> CgepNode:
    return CgepNode(
        node_id=f"n{i}", event_type=f"T{i}", trigger=trigger,
        sentence=sentence, token_span=token_span(sentence, trigger),
    )


def _instance() -> CgepInstance:
    nodes = (
        _node(0, "flood", "a flood hit the town ."),
        _node(1, "evacuate", "they evacuate the town ."),
        _node(2, "rescue", "crews rescue the last family ."),
        _node(3, "rebuild", "the town will rebuild ."),
    )
    return CgepInstance(
        instance_id="x", doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1), (1, "CAUSE", 2), (2, "CAUSE", 3)),
        candidates=(nodes[3], nodes[0]), label=0,
    )


class _FakeTokenizer:
    """Whitespace tokenizer that honours added tokens, enough to pin the wiring."""

    mask_token_id = 4

    def __init__(self) -> None:
        self._vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3, "<mask>": 4}

    def add_tokens(self, tokens):
        for token in tokens:
            self._vocab.setdefault(token, len(self._vocab))

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, 3)

    def __call__(self, text, add_special_tokens=True, padding=None,
                 max_length=None, truncation=False):
        # Ordinary words map to <unk>; only added tokens sit at or above <a_0>,
        # which is the boundary `encode_instance` uses to find event positions.
        ids = [self._vocab.get(w, 3) for w in text.split()]
        if add_special_tokens:
            ids = [0, *ids, 1]
        if truncation and max_length:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and max_length:
            pad = max_length - len(ids)
            ids, mask = ids + [2] * pad, mask + [0] * pad
        return {"input_ids": ids, "attention_mask": mask}


def _tokenizer(vocab: EventVocabulary) -> _FakeTokenizer:
    tokenizer = _FakeTokenizer()
    tokenizer.add_tokens([f"<a_{i}>" for i in range(len(vocab))])
    return tokenizer


def test_event_token_nodes_are_both_endpoints_then_the_anchor():
    instance = _instance()
    vocab = EventVocabulary.build([instance])
    nodes = event_token_nodes(linearize(instance, vocab))
    # edges are 0->1, 1->2, then the query 2->3; the gold (3) contributes <mask>.
    assert nodes == [0, 1, 1, 2, 2]


def test_replace_mention_keeps_trailing_punctuation_as_its_own_token():
    assert replace_mention("In Victoria 47 people died.", (4, 5), "<a_7>") == \
        "In Victoria 47 people <a_7> ."


def test_replace_mention_widens_over_the_whole_span():
    assert replace_mention("they keep a hold on it", (1, 4), "<a_3>") == "they <a_3> on it"


def test_replace_mention_leaves_the_sentence_alone_when_the_span_is_unusable():
    assert replace_mention("nothing to do here", None, "<a_0>") == "nothing to do here"
    assert replace_mention("short", (3, 9), "<a_0>") == "short"


def test_each_event_token_is_bound_to_its_own_events_sentence():
    instance = _instance()
    vocab = EventVocabulary.build([instance])
    tokenizer = _tokenizer(vocab)
    encoded = encode_instance(instance, tokenizer, vocab, max_length=32)

    nodes = event_token_nodes(linearize(instance, vocab))
    assert len(encoded.event_positions) == len(nodes)
    for slot, node_index in enumerate(nodes):
        # The template token at this position ...
        position = encoded.event_positions[slot]
        expected = tokenizer.convert_tokens_to_ids(vocab.token(instance.nodes[node_index].trigger))
        assert encoded.template_ids[position] == expected
        # ... and the sentence row it points at hold the *same* event token.
        row = encoded.event_rows[slot]
        assert encoded.sentence_ids[row][encoded.sentence_positions[slot]] == expected


def test_the_mask_position_is_the_masked_slot():
    instance = _instance()
    vocab = EventVocabulary.build([instance])
    tokenizer = _tokenizer(vocab)
    encoded = encode_instance(instance, tokenizer, vocab, max_length=32)
    assert encoded.template_ids[encoded.mask_index] == tokenizer.mask_token_id


def test_candidates_that_share_a_trigger_share_a_token_id():
    nodes = (
        _node(0, "flood", "a flood hit ."),
        _node(1, "evacuate", "they evacuate ."),
        _node(2, "rescue", "crews rescue ."),
        _node(3, "flood", "a second flood came ."),  # same trigger as node 0
    )
    instance = CgepInstance(
        instance_id="x", doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1), (1, "CAUSE", 2), (2, "CAUSE", 3)),
        candidates=(nodes[3], nodes[0], nodes[1]), label=0,
    )
    vocab = EventVocabulary.build([instance])
    encoded = encode_instance(instance, _tokenizer(vocab), vocab, max_length=32)
    ids = encoded.candidate_token_ids
    assert ids[0] == ids[1] != ids[2]  # "flood" twice, then "evacuate"


def test_the_gold_nodes_sentence_is_never_encoded():
    # Gold contributes `<mask>`, not a token, so an unlocatable gold is harmless.
    # This is why `token_span` gaps only matter for template nodes.
    nodes = (
        _node(0, "flood", "a flood hit ."),
        _node(1, "evacuate", "they evacuate ."),
        CgepNode(node_id="n2", event_type="T2", trigger="rescue", sentence="unrelated text ."),
    )
    instance = CgepInstance(
        instance_id="gold-unlocatable", doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1), (1, "CAUSE", 2)), candidates=(nodes[2],), label=0,
    )
    vocab = EventVocabulary.build([instance])
    encoded = encode_instance(instance, _tokenizer(vocab), vocab, max_length=32)
    assert len(encoded.sentence_ids) == 2  # only nodes 0 and 1


def test_an_unlocatable_template_mention_is_rejected_not_encoded_at_position_zero():
    nodes = (
        _node(0, "flood", "a flood hit ."),
        # Trigger absent from its sentence, and this node *is* in the template.
        CgepNode(node_id="n1", event_type="T1", trigger="evacuate", sentence="unrelated text ."),
        _node(2, "rescue", "crews rescue ."),
    )
    instance = CgepInstance(
        instance_id="broken", doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1), (1, "CAUSE", 2)), candidates=(nodes[2],), label=0,
    )
    vocab = EventVocabulary.build([instance])
    with pytest.raises(ValueError, match="missing from the sentence"):
        encode_instance(instance, _tokenizer(vocab), vocab, max_length=32)


def test_truncating_away_an_event_token_is_rejected():
    instance = _instance()
    vocab = EventVocabulary.build([instance])
    with pytest.raises(ValueError, match="raise max_length or lower max_edges"):
        encode_instance(instance, _tokenizer(vocab), vocab, max_length=6)


def test_encode_instance_flags_events_that_reach_the_anchor():
    # M2: reach_anchor is 1 for events upstream of the anchor, 0 downstream.
    # 0 -> 1(anchor) -> 2 branches off, and 1 -> 3(gold/mask): node 2 is
    # downstream of the anchor, so its token must be flagged 0.
    from finekg.succession.structure import event_reach_anchor

    nodes = (
        _node(0, "spark", "a spark began ."),
        _node(1, "fire", "the fire spread ."),
        _node(2, "smoke", "then smoke rose ."),
        _node(3, "alarm", "an alarm rang ."),
    )
    instance = CgepInstance(
        instance_id="x", doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1), (1, "CAUSE", 2), (1, "CAUSE", 3)),
        candidates=(nodes[3], nodes[0]), label=0,
    )
    vocab = EventVocabulary.build([instance])
    encoded = encode_instance(instance, _tokenizer(vocab), vocab, max_length=32)

    lin = linearize(instance, vocab)
    reference = event_reach_anchor(lin.edges[:-1], event_token_nodes(lin), lin.query_edge[0])
    assert encoded.reach_anchor == reference
    assert len(encoded.reach_anchor) == len(encoded.event_positions)
    assert 0 in encoded.reach_anchor and 1 in encoded.reach_anchor


def test_encode_instance_forwards_the_edge_selector():
    # Under a tight budget the two selectors keep different edges (distance keeps
    # those nearest the query), so a dropped `selector=` would make the encoded
    # templates identical -- which is exactly what this guards against.
    nodes = tuple(_node(i, f"e{i}", f"an e{i} occurred .") for i in range(6))
    instance = CgepInstance(
        instance_id="x", doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1), (1, "CAUSE", 2), (2, "CAUSE", 3),
               (3, "CAUSE", 4), (4, "CAUSE", 5)),
        candidates=(nodes[5],), label=0,
    )
    vocab = EventVocabulary.build([instance])
    tokenizer = _tokenizer(vocab)
    near = encode_instance(instance, tokenizer, vocab, max_length=32, max_edges=2,
                           selector=select_nearest_edges)
    base = encode_instance(instance, tokenizer, vocab, max_length=32, max_edges=2)
    assert near.template_ids != base.template_ids
