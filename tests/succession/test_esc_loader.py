"""ESC loader contract: the whitelist must hold, and the real file must parse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finekg.succession.data.esc import (
    ESC_CANDIDATE_SET_SIZE,
    load_cgep_esc,
    load_npy_object,
    topic_folds,
)

REAL_ESC = Path("data/raw/sedgpl_esc/ESCSubWoRe.npy")


def _node(topic, doc, idx, etype, mention, sentence, sent_id, place="_1"):
    # `place` is the trigger's *token* index in `sentence`, not the node index.
    return (0, topic, doc, idx, etype, mention, sentence, sent_id, place)


def _instance(topic="1", doc="1_1"):
    nodes = [
        _node(topic, doc, 0, "ACTION_OCCURRENCE", "leaves ", "she leaves rehab", 0),
        _node(topic, doc, 1, "ACTION_OCCURRENCE", "checks ", "she checks in", 1),
        _node(topic, doc, 2, "ACTION_STATE", "stays ", "she stays there", 2),
    ]
    return {
        "node": nodes,
        "edge": [(0, "cause", 1), (1, "cause", 2)],
        "adja": np.zeros((3, 3)),
        "candiSet": [nodes[2], nodes[0]],
        "label": 0,  # candiSet[0] is node 2, the tail of the query edge
    }


def _write_npy(path: Path, obj) -> Path:
    np.save(path, np.array(obj, dtype=object), allow_pickle=True)
    return path.with_suffix(".npy") if path.suffix != ".npy" else path


def test_load_npy_object_reads_a_pickled_dict(tmp_path):
    path = _write_npy(tmp_path / "ok.npy", {"1": {"1_1": [_instance()]}})
    data = load_npy_object(path)
    assert set(data) == {"1"}
    assert data["1"]["1_1"][0]["label"] == 0


def test_load_npy_object_blocks_a_non_numpy_import(tmp_path):
    # A pickle that reconstructs `os.system` is exactly what `allow_pickle=True`
    # would happily execute. The loader must refuse to import it at all.
    import pickle

    payload = pickle.dumps({"evil": eval})  # noqa: S301 - constructing the attack
    path = tmp_path / "evil.npy"
    header = b"\x93NUMPY\x01\x00"
    meta = b"{'descr': '|O', 'fortran_order': False, 'shape': (), }"
    path.write_bytes(header + len(meta).to_bytes(2, "little") + meta + payload)

    with pytest.raises(pickle.UnpicklingError, match="blocked pickle import"):
        load_npy_object(path)


def test_load_npy_object_rejects_a_non_npy_file(tmp_path):
    path = tmp_path / "not.npy"
    path.write_bytes(b"just bytes")
    with pytest.raises(ValueError, match="not a .npy file"):
        load_npy_object(path)


def test_load_cgep_esc_maps_onto_the_shared_instance_type(tmp_path):
    path = _write_npy(tmp_path / "esc.npy", {"1": {"1_1": [_instance()]}})
    by_topic = load_cgep_esc(path)
    assert set(by_topic) == {"1"}
    (instance,) = by_topic["1"]
    assert instance.query_edge == (1, "cause", 2)
    assert instance.anchor_index == 1 and instance.gold_index == 2
    assert instance.candidates[instance.label].node_id.endswith("::2")
    assert instance.gold_trigger == "stays "  # raw spacing preserved
    assert instance.nodes[2].token_span == (1, 2)


def test_discontiguous_mentions_are_widened_the_way_docorrect_widens_them(tmp_path):
    # 27 of ESC's 2824 mentions skip interior words. SeDGPL widens the mention to
    # its contiguous span so the `<a_i>` token stands for text that really occurs.
    nodes = [
        _node("1", "1_1", 0, "ACTION_OCCURRENCE", "leaves ", "she leaves rehab", 0),
        _node("1", "1_1", 1, "ACTION_OCCURRENCE", "checks ", "she checks in", 1),
        _node("1", "1_1", 2, "ACTION_STATE", "keep hold ", "they keep a hold on it", 2, "_1_2_3"),
    ]
    raw = {"node": nodes, "edge": [(0, "cause", 1), (1, "cause", 2)],
           "adja": np.zeros((3, 3)), "candiSet": [nodes[2]], "label": 0}
    path = _write_npy(tmp_path / "wide.npy", {"1": {"1_1": [raw]}})

    (instance,) = load_cgep_esc(path)["1"]
    widened = instance.nodes[2]
    assert widened.trigger == "keep a hold "
    assert widened.token_span == (1, 4)
    # Widening must not desync the label from the gold node.
    assert instance.candidates[instance.label].trigger == widened.trigger


def test_topic_folds_partition_topics_without_overlap():
    topics = [str(i) for i in range(1, 11)]
    folds = list(topic_folds(topics, n_folds=5))
    assert len(folds) == 5
    seen: list[str] = []
    for train, test in folds:
        assert not set(train) & set(test)
        assert sorted(train + test, key=int) == sorted(topics, key=int)
        seen.extend(test)
    assert sorted(seen, key=int) == sorted(topics, key=int)  # every topic tested once


def test_topic_folds_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        list(topic_folds(["1", "2"], n_folds=1))
    with pytest.raises(ValueError, match="cannot fill"):
        list(topic_folds(["1", "2"], n_folds=5))


@pytest.mark.skipif(not REAL_ESC.exists(), reason="ESCSubWoRe.npy not downloaded")
def test_released_esc_build_holds_the_invariants_we_rebuilt_maven_on():
    """The reproduction anchor. These are the numbers our MAVEN protocol copies."""
    by_topic = load_cgep_esc(REAL_ESC)
    instances = [i for topic in by_topic.values() for i in topic]

    assert len(by_topic) == 22
    assert len({i.doc_id for i in instances}) == 244  # paper says 243
    assert len(instances) == 1192  # paper says 1191
    assert {len(i.candidates) for i in instances} == {ESC_CANDIDATE_SET_SIZE}

    for instance in instances:
        gold = instance.gold_index
        out_degree = sum(1 for h, _, _ in instance.edges if h == gold)
        in_degree = sum(1 for _, _, t in instance.edges if t == gold)
        # The rule we derived from `getTemplate` and rebuilt CGEP-MAVEN on.
        assert out_degree == 0 and in_degree == 1
        assert all(gold not in (h, t) for h, _, t in instance.template_edges)
        assert instance.candidates[instance.label].trigger == instance.nodes[gold].trigger

    # Triggers collide, so 256 candidates offer far fewer distinct answers.
    distinct = sum(i.distinct_answers for i in instances) / len(instances)
    assert 177.5 < distinct < 178.5
