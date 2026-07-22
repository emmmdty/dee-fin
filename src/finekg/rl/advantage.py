"""Group-relative advantage estimation (the GRPO baseline, value-network-free).

For a group of rewards obtained from the *same* prompt/query, the advantage of
each sample is its reward standardized within the group:

    A_i = (r_i - mean(group)) / max(std(group), std_floor)

This replaces a learned critic with a Monte-Carlo group baseline — the core
trick of GRPO. The current consumer is the retained relation-extraction baseline;
the old path-RL consumer exists only on tag ``frozen-tkg-line``. Pure function,
exactly unit-testable on CPU.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from statistics import fmean, pstdev

__all__ = ["group_relative_advantage"]


def group_relative_advantage(
    rewards: Sequence[float],
    *,
    group_size: int | None = None,
    group_ids: Sequence[Hashable] | None = None,
    std_floor: float = 1e-4,
) -> list[float]:
    """Standardize each reward within its group.

    Exactly one of `group_size` (consecutive, equally-sized groups) or
    `group_ids` (explicit per-sample group labels) must be given. A constant
    group yields zero advantages (no signal), never NaN: the denominator is
    floored at `std_floor`.
    """
    if (group_size is None) == (group_ids is None):
        raise ValueError("provide exactly one of group_size or group_ids")
    if std_floor <= 0:
        raise ValueError("std_floor must be positive")

    if group_size is not None:
        if group_size <= 0 or len(rewards) % group_size:
            raise ValueError(
                f"group_size {group_size} must evenly divide {len(rewards)} rewards"
            )
        group_ids = [i // group_size for i in range(len(rewards))]
    assert group_ids is not None
    if len(group_ids) != len(rewards):
        raise ValueError("group_ids must align with rewards")

    by_group: dict[Hashable, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        by_group[gid].append(i)

    advantages = [0.0] * len(rewards)
    for indices in by_group.values():
        values = [rewards[i] for i in indices]
        mean = fmean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        if std == 0.0:
            continue  # constant (or singleton) group: no learning signal
        denom = max(std, std_floor)
        for i in indices:
            advantages[i] = (rewards[i] - mean) / denom
    return advantages
