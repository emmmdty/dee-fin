"""Tests for the composable verifiable-reward base."""

from __future__ import annotations

import pytest

from finekg.core.registry import Registry
from finekg.rl.reward import CompositeReward, RewardFn, WeightedComponent, build_composite


def _component(value: float) -> RewardFn:
    return lambda *args, **kwargs: value


def test_weighted_sum_and_trace() -> None:
    composite = CompositeReward(
        [
            WeightedComponent("a", 0.25, _component(1.0)),
            WeightedComponent("b", 0.75, _component(0.5)),
        ]
    )
    trace = composite.score("sample")
    assert trace.total == pytest.approx(0.25 * 1.0 + 0.75 * 0.5)
    assert trace.components == {"a": 1.0, "b": 0.5}
    assert composite("sample") == pytest.approx(trace.total)


def test_zero_weight_component_is_traced_but_does_not_affect_total() -> None:
    with_zero = CompositeReward(
        [
            WeightedComponent("a", 1.0, _component(0.4)),
            WeightedComponent("b", 0.0, _component(0.9)),
        ]
    )
    without = CompositeReward([WeightedComponent("a", 1.0, _component(0.4))])
    assert with_zero("x") == pytest.approx(without("x"))
    assert with_zero.score("x").components["b"] == 0.9


def test_empty_or_duplicate_components_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeReward([])
    with pytest.raises(ValueError, match="duplicate"):
        CompositeReward(
            [
                WeightedComponent("a", 1.0, _component(0.0)),
                WeightedComponent("a", 1.0, _component(1.0)),
            ]
        )


def test_build_composite_from_specs_and_ablation_by_deleting_a_line() -> None:
    registry: Registry[RewardFn] = Registry("test_reward_component")

    @registry.register("constant")
    class ConstantReward:
        def __init__(self, value: float = 1.0) -> None:
            self.value = value

        def __call__(self, *args: object, **kwargs: object) -> float:
            return self.value

    full_specs = [
        {"name": "constant", "weight": 0.5, "kwargs": {"value": 0.8}},
    ]
    composite = build_composite(registry, full_specs)
    assert composite.names == ["constant"]
    assert composite("sample") == pytest.approx(0.5 * 0.8)

    with pytest.raises(KeyError):
        build_composite(registry, [{"name": "missing"}])
