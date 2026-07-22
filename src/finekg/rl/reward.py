"""Composable verifiable rewards — the verifier-as-reward contract.

A reward component is any callable mapping a domain sample (an LLM completion
plus its source document, a trajectory plus its query, ...) to a score in
[0, 1]. `CompositeReward` applies named, weighted components and returns a
`RewardTrace` keeping every component value, so reward hacking stays visible in
training curves (one component saturating while the total still climbs).

Each domain owns a `Registry` of component factories (see
`finekg.relations.rl.rewards`); ablating a component is deleting its line from
the experiment config — no code changes, mirroring how extractors and
forecasters are swapped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from finekg.core.registry import Registry

__all__ = ["RewardFn", "RewardTrace", "WeightedComponent", "CompositeReward", "build_composite"]

RewardFn = Callable[..., float]


@dataclass(frozen=True)
class RewardTrace:
    """The total reward plus the raw value of every component."""

    total: float
    components: dict[str, float]


@dataclass(frozen=True)
class WeightedComponent:
    name: str
    weight: float
    fn: RewardFn


class CompositeReward:
    """Weighted sum of named reward components with per-component tracing."""

    def __init__(self, components: list[WeightedComponent]) -> None:
        if not components:
            raise ValueError("CompositeReward needs at least one component")
        names = [c.name for c in components]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate reward component names: {names}")
        self.components = list(components)

    def score(self, *args: Any, **kwargs: Any) -> RewardTrace:
        """Apply every component to the same sample and combine."""
        values = {c.name: float(c.fn(*args, **kwargs)) for c in self.components}
        total = sum(c.weight * values[c.name] for c in self.components)
        return RewardTrace(total=total, components=values)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.score(*args, **kwargs).total

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.components]


def build_composite(registry: Registry[RewardFn], specs: list[dict[str, Any]]) -> CompositeReward:
    """Assemble a `CompositeReward` from config specs.

    Each spec is ``{"name": <registered component>, "weight": <float>,
    "kwargs": {...}}``. Removing a spec line is the per-component ablation.
    """
    components = [
        WeightedComponent(
            name=spec["name"],
            weight=float(spec.get("weight", 1.0)),
            fn=registry.create(spec["name"], **spec.get("kwargs", {})),
        )
        for spec in specs
    ]
    return CompositeReward(components)
