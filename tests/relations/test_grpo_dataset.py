"""Tests for GRPO dataset construction and the TRL reward adapter (CPU)."""

from __future__ import annotations

import json

import pytest

from finekg.core.schema import EventNode, RelationEdge, RelationType
from finekg.relations.data import load_maven_ere
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.llm import build_relation_prompt
from finekg.relations.rl.dataset import build_grpo_dataset, to_rows, window_document
from finekg.relations.rl.rewards import build_relation_reward
from finekg.relations.rl.trl_adapter import TrlRewardAdapter


def _big_doc(n_nodes: int = 30) -> RelationDocument:
    nodes = [
        EventNode(event_id=f"e{i}", event_type="EventType", doc_id="big") for i in range(n_nodes)
    ]
    gold = [
        # In-window pair (window 0) and a cross-window pair that must be dropped.
        RelationEdge(
            head_id="e0", tail_id="e1", relation_type=RelationType.TEMPORAL, subtype="BEFORE"
        ),
        RelationEdge(
            head_id="e0", tail_id="e29", relation_type=RelationType.TEMPORAL, subtype="BEFORE"
        ),
    ]
    return RelationDocument(doc_id="big", nodes=nodes, gold_edges=gold, doc_text="text")


def test_windowing_bounds_size_and_localizes_gold() -> None:
    windows = window_document(_big_doc(30), window_events=12)
    assert [len(w.nodes) for w in windows] == [12, 12, 6]
    assert all(w.doc_text == "text" for w in windows)
    first = windows[0]
    assert [e.key() for e in first.gold_edges] == [("e0", "e1", "temporal", "BEFORE")]
    assert all(not w.gold_edges for w in windows[1:])  # cross-window edge dropped


def test_windowing_skips_singleton_tail_and_rejects_tiny_windows() -> None:
    doc = _big_doc(13)
    windows = window_document(doc, window_events=12)
    assert [len(w.nodes) for w in windows] == [12]  # the 1-node tail is no task
    with pytest.raises(ValueError, match="at least 2"):
        window_document(doc, window_events=1)


def test_build_grpo_dataset_prompts_and_store(fixtures_dir) -> None:
    docs = list(load_maven_ere(fixtures_dir / "maven_ere" / "sample_with_text.jsonl"))
    samples, store = build_grpo_dataset(docs, window_events=12)
    assert len(samples) == len(store) == 2  # both fixture docs fit one window each
    for sample in samples:
        doc = store.get(sample.doc_key)
        assert sample.prompt == build_relation_prompt(doc.nodes, doc_text=doc.doc_text)
        assert doc.doc_text and doc.doc_text[:40] in sample.prompt  # the model can quote
        assert sample.difficulty == float(len(doc.nodes))
    rows = to_rows(samples)
    assert set(rows[0]) == {"prompt", "doc_key", "difficulty"}
    with pytest.raises(KeyError, match="doc_key"):
        store.get("nonexistent")


def test_trl_adapter_scores_string_and_chat_completions(fixtures_dir) -> None:
    docs = list(load_maven_ere(fixtures_dir / "maven_ere" / "sample_with_text.jsonl"))
    samples, store = build_grpo_dataset(docs, window_events=12)
    composite = build_relation_reward(
        [{"name": "format", "weight": 0.5}, {"name": "grounding", "weight": 0.5}]
    )
    adapter = TrlRewardAdapter(composite, store)

    good = json.dumps(
        {
            "relations": [
                {
                    "head": 0,
                    "tail": 2,
                    "type": "temporal",
                    "subtype": "BEFORE",
                    "evidence_quote": "The assault injured",
                }
            ]
        },
        ensure_ascii=False,
    )
    completions = [good, [{"role": "assistant", "content": "not json"}]]
    doc_keys = [samples[0].doc_key, samples[0].doc_key]
    rewards = adapter(prompts=["p", "p"], completions=completions, doc_key=doc_keys)

    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == 0.0
    means = adapter.component_means()
    assert set(means) == {"format", "grounding", "total"}
    assert means["total"] == pytest.approx(0.5)
    assert adapter.__name__ == "verifiable_composite"


def test_trl_adapter_requires_doc_key_column() -> None:
    composite = build_relation_reward([{"name": "format", "weight": 1.0}])
    adapter = TrlRewardAdapter(composite, build_grpo_dataset([], window_events=12)[1])
    with pytest.raises(ValueError, match="doc_key"):
        adapter(completions=["x"])
