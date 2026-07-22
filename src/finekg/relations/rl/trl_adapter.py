"""Adapter exposing a CompositeReward under TRL's reward-function convention.

TRL's GRPOTrainer calls a reward function as
``fn(prompts=..., completions=..., **columns) -> list[float]`` where every
extra dataset column (here ``doc_key``) arrives as a list aligned with the
completions; conversational datasets deliver each completion as a
``[{"role": ..., "content": ...}]`` message list. This module follows that
*calling convention only* — it never imports trl — so the adapter is fully
unit-testable on CPU and the heavy stack stays confined to `trainer`.

The adapter also keeps per-component telemetry: composite rewards are only
trustworthy when each component's trajectory is visible (a grounding term
collapsing while the total climbs is reward hacking, not progress). Three
views are exposed — cumulative means (`component_means`), phase-local means
(`phase_means` / `mark_phase`), and a windowed time series (`curve`), which is
what actually shows whether the four components rise together.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from finekg.relations.rl.dataset import DocStore
from finekg.rl.reward import CompositeReward

__all__ = ["TrlRewardAdapter"]


def _completion_text(completion: Any) -> str:
    """Extract plain text from a string or chat-format completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        last = completion[-1] if completion else None
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return "" if last is None else str(last)
    return str(completion)


class TrlRewardAdapter:
    def __init__(
        self, composite: CompositeReward, store: DocStore, window_size: int = 64
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.composite = composite
        self.store = store
        self.window_size = window_size
        names = [*composite.names, "total"]
        self._sums: dict[str, float] = dict.fromkeys(names, 0.0)
        self._count = 0
        self._mark_sums: dict[str, float] = dict(self._sums)
        self._mark_count = 0
        self._window_sums: dict[str, float] = dict.fromkeys(names, 0.0)
        self._window_count = 0
        self._curve: list[dict[str, float]] = []
        # TRL logs reward functions by __name__.
        self.__name__ = "verifiable_composite"

    def __call__(
        self,
        prompts: Sequence[Any] | None = None,
        completions: Sequence[Any] | None = None,
        doc_key: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        if completions is None or doc_key is None:
            raise ValueError("TrlRewardAdapter needs `completions` and the `doc_key` column")
        rewards: list[float] = []
        for completion, key in zip(completions, doc_key, strict=True):
            trace = self.composite.score(_completion_text(completion), self.store.get(key))
            rewards.append(trace.total)
            for name, value in [*trace.components.items(), ("total", trace.total)]:
                self._sums[name] += value
                self._window_sums[name] += value
            self._count += 1
            self._window_count += 1
            if self._window_count >= self.window_size:
                self._flush_window()
        return rewards

    def _flush_window(self) -> None:
        point = {"n_scored": float(self._count)}
        point.update(
            {name: value / self._window_count for name, value in self._window_sums.items()}
        )
        self._curve.append(point)
        self._window_sums = dict.fromkeys(self._window_sums, 0.0)
        self._window_count = 0

    def component_means(self) -> dict[str, float]:
        """Running mean of every component (and the total) since construction."""
        if not self._count:
            return dict.fromkeys(self._sums, 0.0)
        return {name: value / self._count for name, value in self._sums.items()}

    def phase_means(self) -> dict[str, float]:
        """Means since the last `mark_phase()` — phase-local, not cumulative."""
        count = self._count - self._mark_count
        if count <= 0:
            return dict.fromkeys(self._sums, 0.0)
        return {
            name: (value - self._mark_sums[name]) / count
            for name, value in self._sums.items()
        }

    def mark_phase(self) -> None:
        """Snapshot the counters so the next `phase_means()` starts fresh."""
        self._mark_sums = dict(self._sums)
        self._mark_count = self._count

    def curve(self) -> list[dict[str, float]]:
        """Windowed per-component means in scoring order — the hacking watch.

        Each point is the mean over `window_size` consecutively scored
        completions plus `n_scored`, the cumulative completion count at flush
        time. This is the time series the W3-4 exit criterion ("all components
        rising together") is read from.
        """
        return list(self._curve)
