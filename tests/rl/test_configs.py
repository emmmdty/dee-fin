"""Every RL experiment config parses and references only registered components."""

from __future__ import annotations

from pathlib import Path

import pytest

from finekg.core.config import load_config
from finekg.relations.rl.rewards import relation_reward_components
from finekg.rl.curriculum import phases_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]

GRPO_CONFIGS = sorted((REPO_ROOT / "configs" / "relations").glob("*grpo*.yaml"))


def test_expected_config_files_exist() -> None:
    assert [p.name for p in GRPO_CONFIGS] == [
        "ablation_grpo_no_consistency.yaml",
        "ablation_grpo_no_grounding.yaml",
        "grpo_rlvr.yaml",
        "grpo_rlvr_colocate.yaml",
        "grpo_rlvr_easy.yaml",
        "grpo_rlvr_hf.yaml",
        "grpo_rlvr_smoke.yaml",
        "grpo_rlvr_smoke_vllm.yaml",
    ]


@pytest.mark.parametrize("path", GRPO_CONFIGS, ids=lambda p: p.name)
def test_grpo_configs_reference_registered_rewards(path: Path) -> None:
    cfg = load_config(path)
    section = cfg["relations_rl"]
    assert section["base_model"]
    names = [spec["name"] for spec in section["rewards"]]
    assert names, "a GRPO config must keep at least one reward component"
    for name in names:
        assert name in relation_reward_components
    phases = phases_from_config(section["curriculum"]["phases"])
    assert all(p.steps > 0 for p in phases)
    assert cfg["data"]["loader"] in ("maven_ere", "ccks_causal")
    assert section["rollout"]["backend"] in ("vllm_server", "vllm_colocate", "hf")
