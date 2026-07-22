"""Tests for the verifiable reward components (verifier-as-reward, CPU)."""

from __future__ import annotations

import json

import pytest

from finekg.core.schema import EventNode, RelationEdge, RelationType
from finekg.relations.data import load_maven_ere
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.rl.rewards import (
    ConsistencyReward,
    FormatReward,
    GroundingReward,
    TaskF1Reward,
    build_relation_reward,
    relation_reward_components,
)


def _doc() -> RelationDocument:
    nodes = [
        EventNode(event_id="e0", event_type="盈利预警", doc_id="d"),
        EventNode(event_id="e1", event_type="股价下跌", doc_id="d"),
        EventNode(event_id="e2", event_type="股东减持", doc_id="d"),
    ]
    gold = [
        RelationEdge(
            head_id="e0", tail_id="e1", relation_type=RelationType.CAUSAL, subtype="CAUSE"
        )
    ]
    return RelationDocument(
        doc_id="d",
        nodes=nodes,
        gold_edges=gold,
        doc_text="甲公司发布盈利预警，导致股价大幅下跌，随后乙股东宣布减持。",
    )


def _completion(relations: list[dict]) -> str:
    return json.dumps({"relations": relations}, ensure_ascii=False)


_GOLD_ITEM = {
    "head": 0,
    "tail": 1,
    "type": "causal",
    "subtype": "CAUSE",
    "evidence_quote": "导致股价大幅下跌",
}


def test_all_components_registered() -> None:
    for name in ("format", "grounding", "consistency", "task_f1"):
        assert name in relation_reward_components


def test_perfect_completion_maxes_every_component() -> None:
    doc, completion = _doc(), _completion([_GOLD_ITEM])
    assert FormatReward()(completion, doc) == 1.0
    assert GroundingReward()(completion, doc) == 1.0
    assert ConsistencyReward()(completion, doc) == 1.0
    assert TaskF1Reward()(completion, doc) == 1.0


def test_unparseable_completion_scores_zero_everywhere() -> None:
    doc = _doc()
    for component in (FormatReward(), GroundingReward(), ConsistencyReward(), TaskF1Reward()):
        assert component("这不是 JSON", doc) == 0.0


def test_empty_relations_list_is_well_formed_but_earns_nothing_else() -> None:
    doc, completion = _doc(), _completion([])
    assert FormatReward()(completion, doc) == 1.0
    assert GroundingReward()(completion, doc) == 0.0
    assert ConsistencyReward()(completion, doc) == 0.0
    assert TaskF1Reward()(completion, doc) == 0.0


def test_empty_prediction_is_the_correct_answer_on_a_no_gold_window() -> None:
    base = _doc()
    no_gold = RelationDocument(
        doc_id="d-nogold", nodes=base.nodes, gold_edges=[], doc_text=base.doc_text
    )
    empty = _completion([])
    for component in (FormatReward(), GroundingReward(), ConsistencyReward(), TaskF1Reward()):
        assert component(empty, no_gold) == 1.0
    # Honest silence must outearn a grounded-but-hallucinated edge there:
    # the hallucination keeps format/grounding/consistency but loses task F1.
    hallucinated = _completion([_GOLD_ITEM])
    assert TaskF1Reward()(hallucinated, no_gold) == 0.0


def test_format_penalizes_invalid_items() -> None:
    doc = _doc()
    out_of_range = {"head": 0, "tail": 9, "type": "causal"}
    self_loop = {"head": 1, "tail": 1, "type": "causal"}
    completion = _completion([_GOLD_ITEM, out_of_range, self_loop])
    assert FormatReward()(completion, doc) == pytest.approx(1 / 3)


def test_fabricated_quote_earns_no_grounding() -> None:
    doc = _doc()
    fabricated = dict(_GOLD_ITEM, evidence_quote="子虚乌有的证据片段")
    assert GroundingReward()(_completion([fabricated]), doc) == 0.0


def test_oversized_quote_counts_as_ungrounded() -> None:
    doc = _doc()
    assert GroundingReward(max_quote_chars=4)(_completion([_GOLD_ITEM]), doc) == 0.0
    assert GroundingReward(max_quote_chars=60)(_completion([_GOLD_ITEM]), doc) == 1.0


def test_causal_cycle_lowers_consistency_and_repair_restores_it() -> None:
    doc = _doc()
    cyclic = _completion(
        [
            {"head": 0, "tail": 1, "type": "causal", "subtype": "CAUSE"},
            {"head": 1, "tail": 2, "type": "causal", "subtype": "CAUSE"},
            {"head": 2, "tail": 0, "type": "causal", "subtype": "CAUSE"},
        ]
    )
    acyclic = _completion(
        [
            {"head": 0, "tail": 1, "type": "causal", "subtype": "CAUSE"},
            {"head": 1, "tail": 2, "type": "causal", "subtype": "CAUSE"},
        ]
    )
    assert ConsistencyReward()(cyclic, doc) == pytest.approx(2 / 3)
    assert ConsistencyReward()(acyclic, doc) == 1.0


def test_task_f1_zero_for_disjoint_predictions() -> None:
    doc = _doc()
    wrong = _completion([{"head": 1, "tail": 2, "type": "temporal", "subtype": "BEFORE"}])
    assert TaskF1Reward()(wrong, doc) == 0.0


def test_composite_from_config_specs_traces_components() -> None:
    composite = build_relation_reward(
        [
            {"name": "format", "weight": 0.1},
            {"name": "grounding", "weight": 0.3, "kwargs": {"max_quote_chars": 60}},
            {"name": "consistency", "weight": 0.2},
            {"name": "task_f1", "weight": 0.4},
        ]
    )
    trace = composite.score(_completion([_GOLD_ITEM]), _doc())
    assert trace.total == pytest.approx(1.0)
    assert set(trace.components) == {"format", "grounding", "consistency", "task_f1"}


def test_maven_loader_fills_doc_text_and_grounding_works(fixtures_dir) -> None:
    docs = list(load_maven_ere(fixtures_dir / "maven_ere" / "sample_with_text.jsonl"))
    doc = docs[0]
    assert "The assault injured" in doc.doc_text
    completion = _completion(
        [
            {
                "head": 0,
                "tail": 2,
                "type": "temporal",
                "subtype": "BEFORE",
                "evidence_quote": "The assault injured",
            }
        ]
    )
    assert GroundingReward()(completion, doc) == 1.0


def test_maven_loader_without_text_degrades_to_empty(fixtures_dir) -> None:
    docs = list(load_maven_ere(fixtures_dir / "maven_ere" / "sample.jsonl"))
    assert docs[0].doc_text == ""
