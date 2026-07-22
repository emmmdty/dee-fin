"""Potential-based reward shaping (Ng, Harada & Russell, 1999).

Sparse terminal rewards (did the path hit the gold object?) are densified with
the shaping term F(s, s') = gamma * Phi(s') - Phi(s). Because the increments
telescope — with gamma = 1 the undiscounted sum is exactly
Phi(s_T) - Phi(s_0), independent of the path taken — shaping changes the
*learning signal*, never the optimal policy.

The potential Phi itself is domain-specific (the path-RL stage uses a
recency-frequency score of the current entity); this module only owns the
policy-invariant difference form.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["shaping_increments", "shaping_sum"]


def shaping_increments(potentials: Sequence[float], gamma: float = 1.0) -> list[float]:
    """Per-step shaping rewards from the potentials of states s_0 .. s_T.

    Returns T increments for T+1 potentials; a single-state trajectory gets no
    shaping.
    """
    if not potentials:
        raise ValueError("potentials must contain at least the initial state")
    return [
        gamma * potentials[t + 1] - potentials[t] for t in range(len(potentials) - 1)
    ]


def shaping_sum(potentials: Sequence[float], gamma: float = 1.0) -> float:
    """Total shaping reward along a trajectory (telescopes when gamma = 1)."""
    return sum(shaping_increments(potentials, gamma))
