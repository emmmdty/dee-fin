"""Easy-to-hard curriculum with progressive sample mixing.

Training on hard samples (long documents, many events) from step one
destabilizes small-batch GRPO. Instead, samples are bucketed by a difficulty
score and phases are *cumulative*: phase k contains every sample whose
difficulty is at most the phase cap, so later phases mix earlier (easy)
samples back in rather than abandoning them — the progressive sample-mixing
recipe. Phase membership and ordering are deterministic given a seed, so
curricula are reproducible across runs and machines.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["CurriculumPhase", "phases_from_config", "phase_indices", "seeded_order"]


@dataclass(frozen=True)
class CurriculumPhase:
    """One curriculum stage: a difficulty cap plus its training-step budget."""

    max_difficulty: float
    steps: int = 0


def phases_from_config(specs: Sequence[dict[str, Any]]) -> list[CurriculumPhase]:
    """Parse `[{"max_difficulty": ..., "steps": ...}, ...]` config entries."""
    phases = [
        CurriculumPhase(
            max_difficulty=float(spec["max_difficulty"]), steps=int(spec.get("steps", 0))
        )
        for spec in specs
    ]
    if not phases:
        raise ValueError("curriculum needs at least one phase")
    caps = [p.max_difficulty for p in phases]
    if caps != sorted(caps):
        raise ValueError(f"phase difficulty caps must be non-decreasing, got {caps}")
    return phases


def phase_indices(
    difficulties: Sequence[float], phases: Sequence[CurriculumPhase]
) -> list[list[int]]:
    """Sample indices admitted to each phase (cumulative / progressive mixing)."""
    if not phases:
        raise ValueError("curriculum needs at least one phase")
    return [
        [i for i, d in enumerate(difficulties) if d <= phase.max_difficulty]
        for phase in phases
    ]


def seeded_order(n: int, seed: int) -> list[int]:
    """A deterministic shuffle of range(n) — reproducible curriculum ordering."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order
