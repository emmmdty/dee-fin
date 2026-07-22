"""Tests for the LRD training objective helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from sarge.data.schema import DatasetSchema
from sarge.models.encoder import ArgumentEncodingConfig
from sarge.postprocess.lrd_planner import LRDConfig, LRDPlanner


def _make_planner() -> LRDPlanner:
    schema = DatasetSchema(
        dataset_id="fixture",
        schema_dataset="fixture",
        schema_path=Path("/dev/null"),
        canonical_version=None,
        event_roles={"质押": ("质押方",)},
        role_to_event_types={"质押方": ("质押",)},
    )
    enc_cfg = ArgumentEncodingConfig(model_path="/nonexistent", hidden_dim=4, role_embedding_dim=2)
    planner = LRDPlanner(LRDConfig(encoder_config=enc_cfg, role_vocabulary=["质押方"], scorer_hidden_dim=8, scorer_dropout=0.0), schema)
    planner.encoder.project_span = lambda pooled, roles: pooled
    return planner


def test_train_step_has_no_proxy_reward_term() -> None:
    from scripts.train_lrd import _train_step

    planner = _make_planner()
    batch = [
        {
            "records": [
                {"arg_pooled": [[0.1, 0.2, 0.3, 0.4]], "role_indices": [0], "role_mask": [1.0]},
                {"arg_pooled": [[0.5, 0.6, 0.7, 0.8]], "role_indices": [0], "role_mask": [1.0]},
            ],
            "pairs": [{"i": 0, "j": 1, "label": 1}],
        }
    ]

    total, pair_loss, reward_loss = _train_step(planner, batch, device=planner.merge_thresholds.device, reward_weight=1.0)

    assert reward_loss.item() == pytest.approx(0.0)
    assert total.item() == pytest.approx(pair_loss.item())
