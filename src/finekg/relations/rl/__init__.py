"""GRPO-RLVR post-training for the relation extractor.

The verifiers that gate extraction at inference time (evidence grounding,
global consistency) become the training reward here, on top of the task F1 —
the relations-stage face of verifier-as-reward. CPU pieces (rewards, dataset,
TRL adapter) import eagerly; the GPU trainer stays behind a lazy import in
`finekg.relations.rl.trainer`.
"""

from finekg.relations.rl.dataset import (
    DocStore,
    GrpoSample,
    build_grpo_dataset,
    to_rows,
    window_document,
)
from finekg.relations.rl.rewards import build_relation_reward, relation_reward_components
from finekg.relations.rl.trl_adapter import TrlRewardAdapter

__all__ = [
    "GrpoSample",
    "DocStore",
    "window_document",
    "build_grpo_dataset",
    "to_rows",
    "relation_reward_components",
    "build_relation_reward",
    "TrlRewardAdapter",
]
