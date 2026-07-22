"""Tests for group-relative advantage estimation."""

from __future__ import annotations

import pytest

from finekg.rl.advantage import group_relative_advantage


def test_matches_hand_computed_values() -> None:
    # Two groups of two: mean 0.5, population std 0.5 -> advantages +-1.
    adv = group_relative_advantage([1.0, 0.0, 0.0, 1.0], group_size=2)
    assert adv == pytest.approx([1.0, -1.0, -1.0, 1.0])


def test_zero_mean_within_each_group() -> None:
    rewards = [0.3, 0.9, 0.1, 0.5, 0.2, 0.8, 0.4, 0.7]
    adv = group_relative_advantage(rewards, group_size=4)
    assert sum(adv[:4]) == pytest.approx(0.0, abs=1e-9)
    assert sum(adv[4:]) == pytest.approx(0.0, abs=1e-9)


def test_constant_group_yields_zero_advantages_not_nan() -> None:
    adv = group_relative_advantage([0.7, 0.7, 0.7], group_size=3)
    assert adv == [0.0, 0.0, 0.0]


def test_singleton_groups_yield_zero() -> None:
    adv = group_relative_advantage([0.2, 0.9], group_ids=["q1", "q2"])
    assert adv == [0.0, 0.0]


def test_group_ids_equivalent_to_group_size() -> None:
    rewards = [1.0, 0.0, 0.5, 0.25]
    by_size = group_relative_advantage(rewards, group_size=2)
    by_ids = group_relative_advantage(rewards, group_ids=["a", "a", "b", "b"])
    assert by_size == pytest.approx(by_ids)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        group_relative_advantage([1.0], group_size=1, group_ids=[0])
    with pytest.raises(ValueError, match="exactly one"):
        group_relative_advantage([1.0])
    with pytest.raises(ValueError, match="evenly divide"):
        group_relative_advantage([1.0, 2.0, 3.0], group_size=2)
    with pytest.raises(ValueError, match="align"):
        group_relative_advantage([1.0, 2.0], group_ids=[0])
    with pytest.raises(ValueError, match="std_floor"):
        group_relative_advantage([1.0, 2.0], group_size=2, std_floor=0.0)
