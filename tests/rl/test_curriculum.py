"""Tests for the easy-to-hard curriculum with progressive mixing."""

from __future__ import annotations

import pytest

from finekg.rl.curriculum import (
    CurriculumPhase,
    phase_indices,
    phases_from_config,
    seeded_order,
)


def test_phases_parse_from_config() -> None:
    phases = phases_from_config(
        [{"max_difficulty": 6, "steps": 200}, {"max_difficulty": 12, "steps": 300}]
    )
    assert phases == [CurriculumPhase(6.0, 200), CurriculumPhase(12.0, 300)]


def test_non_monotone_caps_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        phases_from_config([{"max_difficulty": 12}, {"max_difficulty": 6}])
    with pytest.raises(ValueError, match="at least one"):
        phases_from_config([])


def test_progressive_mixing_phases_are_cumulative_supersets() -> None:
    difficulties = [2.0, 5.0, 8.0, 11.0, 20.0]
    phases = phases_from_config(
        [{"max_difficulty": 6}, {"max_difficulty": 12}, {"max_difficulty": 24}]
    )
    buckets = phase_indices(difficulties, phases)
    assert buckets[0] == [0, 1]
    assert buckets[1] == [0, 1, 2, 3]  # easy samples mixed back in
    assert buckets[2] == [0, 1, 2, 3, 4]
    assert set(buckets[0]) <= set(buckets[1]) <= set(buckets[2])


def test_seeded_order_is_deterministic_and_a_permutation() -> None:
    a = seeded_order(10, seed=42)
    b = seeded_order(10, seed=42)
    c = seeded_order(10, seed=13)
    assert a == b
    assert sorted(a) == list(range(10))
    assert a != c  # different seed, different order
