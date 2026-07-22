"""Successor-event predictors: rank a candidate set for one anchor event.

Deliberately not `finekg.forecasting.Forecaster`, whose contract is
`predict(query, history: list[TemporalQuad])` -- an entity-centric temporal quad
`(subject, relation, ?, t)`. CGEP ranks *events* out of an explicit candidate
set given a graph, and squeezing it into quads would misstate the task.

`evaluate` reports the optimistic (SeDGPL) and strict rankings side by side. The
gap between them is the share of the score that duplicate triggers hand over for
free, and it is large on real data, so no caller gets to see only one of them.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from finekg.core.registry import Registry
from finekg.succession.data.cgep import CgepInstance
from finekg.succession.metrics import HITS_AT, cgep_metrics, sedgpl_rank, strict_rank

__all__ = [
    "FrequencySuccessorPredictor",
    "RandomSuccessorPredictor",
    "SuccessorPredictor",
    "UnscorableInstance",
    "evaluate",
    "successor_predictors",
]


class UnscorableInstance(Exception):
    """A predictor could not score an instance at all.

    Raise this instead of returning flat scores. Under SeDGPL's optimistic
    tie-break a fully tied score gives gold rank 0, so a predictor that quietly
    failed on every instance would report a perfect MRR of 1.0. `evaluate` counts
    these as worst-rank misses and reports how many there were.
    """


class SuccessorPredictor(ABC):
    @abstractmethod
    def fit(self, instances: Sequence[CgepInstance]) -> None:
        """Learn from training instances; may be a no-op."""

    @abstractmethod
    def score(self, instance: CgepInstance) -> list[float]:
        """One score per entry of `instance.candidates`, higher is better.

        Candidates sharing a trigger must score identically: SeDGPL scores the
        token id of the mention string, so a faithful implementation cannot tell
        two candidates with the same trigger apart.
        """


successor_predictors: Registry[SuccessorPredictor] = Registry("successor_predictor")


@successor_predictors.register("frequency")
class FrequencySuccessorPredictor(SuccessorPredictor):
    """Rank candidates by how often their trigger was a gold successor in training.

    The torch-free lower bound. It ignores the graph entirely, so it measures how
    much of CGEP is answerable from the marginal popularity of an event mention
    -- the number any graph-aware model has to beat to have earned its structure.
    """

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def fit(self, instances: Sequence[CgepInstance]) -> None:
        self._counts = Counter(instance.gold_trigger for instance in instances)

    def score(self, instance: CgepInstance) -> list[float]:
        return [float(self._counts[candidate.trigger]) for candidate in instance.candidates]


@successor_predictors.register("random")
class RandomSuccessorPredictor(SuccessorPredictor):
    """Score each trigger by a seeded hash: chance, but obeying the tie contract.

    The reference every other predictor is measured against. Scoring by trigger
    rather than by slot keeps duplicate triggers tied, exactly as a token-id
    scorer would, so its ranks are comparable with the real models'.
    """

    def __init__(self, seed: int = 209) -> None:
        self.seed = seed

    def fit(self, instances: Sequence[CgepInstance]) -> None:
        """Nothing to learn."""

    def score(self, instance: CgepInstance) -> list[float]:
        scores = []
        for candidate in instance.candidates:
            key = f"{self.seed}:{instance.instance_id}:{candidate.trigger}".encode()
            digest = hashlib.blake2b(key, digest_size=8).digest()
            scores.append(int.from_bytes(digest, "big") / float(1 << 64))
        return scores


def evaluate(
    predictor: SuccessorPredictor,
    instances: Sequence[CgepInstance],
    hits_at: Sequence[int] = HITS_AT,
) -> dict[str, float]:
    """Rank every instance, reporting both tie-break conventions.

    `mrr` follows SeDGPL (ties go to gold) so published numbers are comparable;
    `mrr_strict` charges gold for every tie. Report both.

    Instances the predictor cannot score at all count as worst-rank misses and
    are reported as `n_unscorable`. They are never dropped from the denominator:
    a pipeline that fails to encode an instance has not answered it.
    """
    optimistic: list[int] = []
    strict: list[int] = []
    unscorable = 0
    for instance in instances:
        try:
            scores = predictor.score(instance)
        except UnscorableInstance:
            unscorable += 1
            worst = len(instance.candidates) - 1
            optimistic.append(worst)
            strict.append(worst)
            continue
        if len(scores) != len(instance.candidates):
            raise ValueError(
                f"{type(predictor).__name__} scored {len(scores)} of "
                f"{len(instance.candidates)} candidates for {instance.instance_id}"
            )
        optimistic.append(sedgpl_rank(scores, instance.label))
        strict.append(strict_rank(scores, instance.label))

    metrics = cgep_metrics(optimistic, hits_at)
    tight = cgep_metrics(strict, hits_at)
    metrics.update({f"{k}_strict": v for k, v in tight.items() if k != "n"})
    metrics["n_unscorable"] = float(unscorable)
    return metrics
