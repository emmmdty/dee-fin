"""Stage-agnostic reinforcement-learning base: the verifier-as-reward substrate.

The same verifier kernels that gate outputs at inference time (grounding,
consistency, intervention faithfulness) are recast here as *training* signals:

- `reward`     composable, weighted verifiable rewards with per-component traces;
- `advantage`  group-relative advantages (GRPO-style, no value network);
- `shaping`    potential-based reward shaping (policy-invariant densification);
- `curriculum` easy-to-hard progressive sample mixing.

Everything in this package is pure CPU (no torch) so rewards, advantages and
curricula are unit-testable locally; the GPU trainer in `relations.rl`
consumes these primitives lazily.
"""

from ekg.rl.advantage import group_relative_advantage
from ekg.rl.curriculum import CurriculumPhase, phase_indices, phases_from_config, seeded_order
from ekg.rl.reward import CompositeReward, RewardTrace, WeightedComponent, build_composite
from ekg.rl.shaping import shaping_increments, shaping_sum

__all__ = [
    "CompositeReward",
    "RewardTrace",
    "WeightedComponent",
    "build_composite",
    "group_relative_advantage",
    "CurriculumPhase",
    "phases_from_config",
    "phase_indices",
    "seeded_order",
    "shaping_increments",
    "shaping_sum",
]
