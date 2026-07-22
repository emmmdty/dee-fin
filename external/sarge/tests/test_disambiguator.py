"""Unit tests for the LRD core modules (no GPU / no RoBERTa weights required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sarge.models.disambiguator import jaccard_overlap_matrix


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "lrd"


@pytest.fixture(autouse=True)
def _seed() -> None:
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)


def test_jaccard_overlap_matrix_identical():
    rec = {"event_type": "质押", "arguments": {"质押方": [{"text": "甲公司"}]}}
    m = jaccard_overlap_matrix([rec, rec])
    assert m[0, 1] == 1.0
    assert m[0, 0] == 1.0


def test_jaccard_overlap_matrix_disjoint():
    a = {"event_type": "质押", "arguments": {"质押方": [{"text": "甲公司"}]}}
    b = {"event_type": "质押", "arguments": {"质押方": [{"text": "乙公司"}]}}
    m = jaccard_overlap_matrix([a, b])
    assert m[0, 1] == 0.0


def test_jaccard_overlap_matrix_partial():
    a = {"event_type": "质押", "arguments": {"质押方": [{"text": "甲"}], "质押物": [{"text": "A 股票"}]}}
    b = {"event_type": "质押", "arguments": {"质押方": [{"text": "甲"}], "质押物": [{"text": "B 股票"}]}}
    m = jaccard_overlap_matrix([a, b])
    assert 0.3 < m[0, 1] < 0.5  # 1 shared role-value out of 3 total


def test_pairwise_scorer_shape():
    """Instantiate PairwiseScorer and check output shape."""
    import torch
    from sarge.models.disambiguator import PairwiseScorer

    scorer = PairwiseScorer(input_dim=64, hidden_dim=32)
    n = 4
    embeddings = torch.randn(n, 64)
    overlap = torch.rand(n, n)
    logits = scorer(embeddings, overlap)
    assert logits.shape == (n, n)


def test_pairwise_scorer_upper_tri_mask():
    """Ensure pair_mask filters scores correctly."""
    import torch
    from sarge.models.disambiguator import PairwiseScorer

    scorer = PairwiseScorer(input_dim=64, hidden_dim=32)
    n = 4
    embeddings = torch.randn(n, 64)
    overlap = torch.rand(n, n)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    logits = scorer(embeddings, overlap, pair_mask=mask)
    assert (logits[~mask] == -1e9).all()


def test_lrd_planner_instantiation():
    """LRDPlanner builds without crashing on a minimal schema."""
    from sarge.data.schema import DatasetSchema
    from sarge.models.encoder import ArgumentEncodingConfig
    from sarge.postprocess.lrd_planner import LRDConfig, LRDPlanner

    schema = DatasetSchema(
        dataset_id="fixture",
        schema_dataset="fixture",
        schema_path=Path("/dev/null"),
        canonical_version=None,
        event_roles={"质押": ("质押方", "质押物"), "中标": ("中标公司",)},
        role_to_event_types={"质押方": ("质押",), "质押物": ("质押",), "中标公司": ("中标",)},
    )
    roles = ["质押方", "质押物", "中标公司"]
    enc_cfg = ArgumentEncodingConfig(model_path="/nonexistent", hidden_dim=64, role_embedding_dim=16)
    lrd_cfg = LRDConfig(encoder_config=enc_cfg, role_vocabulary=roles)
    planner = LRDPlanner(lrd_cfg, schema)
    assert planner is not None
    assert hasattr(planner, "merge_thresholds")


def test_lrd_planner_disambiguate_noop():
    """Single-record input passes through."""
    from sarge.data.schema import DatasetSchema
    from sarge.models.encoder import ArgumentEncodingConfig
    from sarge.postprocess.lrd_planner import LRDConfig, LRDPlanner
    from sarge.postprocess.rule_planner import EventRecord

    schema = DatasetSchema(
        dataset_id="fixture", schema_dataset="fixture",
        schema_path=Path("/dev/null"), canonical_version=None,
        event_roles={"质押": ("质押方",)},
        role_to_event_types={"质押方": ("质押",)},
    )
    roles = ["质押方"]
    enc_cfg = ArgumentEncodingConfig(model_path="/nonexistent", hidden_dim=64, role_embedding_dim=16)
    planner = LRDPlanner(LRDConfig(encoder_config=enc_cfg, role_vocabulary=roles), schema)
    rec = EventRecord(event_type="质押", arguments={"质押方": [{"text": "甲"}]})
    planned, diag = planner.disambiguate([rec], doc_text="甲")
    assert len(planned) == 1
    assert diag.events_before == diag.events_after == 1


def test_lrd_planner_groups_records_by_event_type_before_disambiguation():
    from sarge.data.schema import DatasetSchema
    from sarge.models.encoder import ArgumentEncodingConfig
    from sarge.postprocess.lrd_planner import LRDConfig, LRDPlanner
    from sarge.postprocess.rule_planner import EventRecord

    schema = DatasetSchema(
        dataset_id="fixture",
        schema_dataset="fixture",
        schema_path=Path("/dev/null"),
        canonical_version=None,
        event_roles={"质押": ("质押方",), "中标": ("中标公司",)},
        role_to_event_types={"质押方": ("质押",), "中标公司": ("中标",)},
    )
    roles = ["质押方", "中标公司"]
    enc_cfg = ArgumentEncodingConfig(model_path="/nonexistent", hidden_dim=64, role_embedding_dim=16)
    planner = LRDPlanner(LRDConfig(encoder_config=enc_cfg, role_vocabulary=roles), schema)

    seen: list[list[str]] = []

    def fake_disambiguate_group(records, doc_text, diagnostics):
        del doc_text, diagnostics
        seen.append([rec.event_type for rec in records])
        return records

    planner._disambiguate_group = fake_disambiguate_group  # type: ignore[method-assign]
    records = [
        EventRecord(event_type="质押", arguments={"质押方": [{"text": "甲"}]}),
        EventRecord(event_type="质押", arguments={"质押方": [{"text": "乙"}]}),
        EventRecord(event_type="中标", arguments={"中标公司": [{"text": "丙"}]}),
    ]

    planned, diag = planner.disambiguate(records, doc_text="甲乙丙")

    assert planned == records
    assert diag.events_before == diag.events_after == 3
    assert seen == [["质押", "质押"]]


def test_lrd_cluster_merge_preserves_incompatible_anchor_records():
    from sarge.postprocess.lrd_planner import LRDPlanner
    from sarge.postprocess.rule_planner import EventRecord, PlannerDiagnostics

    records = [
        EventRecord(
            event_type="被约谈",
            arguments={
                "公司名称": [{"text": "爱奇艺"}],
                "约谈机构": [{"text": "浙江省消费者权益保护委员会"}],
                "被约谈时间": [{"text": "8日"}],
            },
        ),
        EventRecord(
            event_type="被约谈",
            arguments={
                "公司名称": [{"text": "腾讯视频"}],
                "约谈机构": [{"text": "浙江省消费者权益保护委员会"}],
                "被约谈时间": [{"text": "8日"}],
            },
        ),
    ]
    diagnostics = PlannerDiagnostics(mode="lrd", events_before=len(records))

    planned = LRDPlanner._merge_clusters(records, [[0, 1]], diagnostics)

    assert planned == records
    assert diagnostics.decisions == []
