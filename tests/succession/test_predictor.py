"""Predictor contract: tie-preserving scores, and both rankings reported."""

from __future__ import annotations

import pytest

from finekg.succession.data.cgep import CgepInstance, CgepNode
from finekg.succession.predictor import (
    FrequencySuccessorPredictor,
    RandomSuccessorPredictor,
    SuccessorPredictor,
    UnscorableInstance,
    evaluate,
    successor_predictors,
)


def _node(name: str, trigger: str | None = None) -> CgepNode:
    return CgepNode(node_id=name, event_type="T", trigger=trigger or name, sentence="")


def _instance(candidates: list[CgepNode], label: int, iid: str = "i0") -> CgepInstance:
    nodes = (_node("a"), _node("b"))
    return CgepInstance(
        instance_id=iid, doc_id="d", nodes=nodes,
        edges=((0, "CAUSE", 1),), candidates=tuple(candidates), label=label,
    )


def test_registry_exposes_the_torch_free_baselines():
    assert {"frequency", "random"} <= set(successor_predictors.available())
    assert isinstance(successor_predictors.create("frequency"), SuccessorPredictor)


def test_frequency_ranks_by_how_often_a_trigger_was_a_successor():
    train = [
        _instance([_node("x")], 0),
        _instance([_node("x")], 0),
        _instance([_node("y")], 0),
    ]
    predictor = FrequencySuccessorPredictor()
    predictor.fit(train)
    scores = predictor.score(_instance([_node("y"), _node("x"), _node("z")], 0))
    assert scores == [1.0, 2.0, 0.0]


def test_candidates_sharing_a_trigger_must_score_identically():
    # SeDGPL scores the token id of the mention, so it cannot tell them apart.
    # Any predictor that does would be ranking on information the model lacks.
    duplicates = [_node("n1", "flood"), _node("n2", "flood"), _node("n3", "fire")]
    instance = _instance(duplicates, label=0)

    frequency = FrequencySuccessorPredictor()
    frequency.fit([_instance([_node("n9", "flood")], 0)])
    scores = frequency.score(instance)
    assert scores[0] == scores[1] != scores[2]

    scores = RandomSuccessorPredictor().score(instance)
    assert scores[0] == scores[1]


def test_random_is_deterministic_per_seed_and_varies_across_seeds():
    instance = _instance([_node("p"), _node("q"), _node("r")], 0)
    assert RandomSuccessorPredictor(seed=1).score(instance) == \
        RandomSuccessorPredictor(seed=1).score(instance)
    assert RandomSuccessorPredictor(seed=1).score(instance) != \
        RandomSuccessorPredictor(seed=2).score(instance)


def test_random_scores_differ_across_instances_for_the_same_trigger():
    # Otherwise every instance would share one global ordering of triggers,
    # which is a systematic ranking, not chance.
    candidates = [_node("p"), _node("q")]
    first = RandomSuccessorPredictor().score(_instance(candidates, 0, iid="i0"))
    second = RandomSuccessorPredictor().score(_instance(candidates, 0, iid="i1"))
    assert first != second


class _Oracle(SuccessorPredictor):
    def fit(self, instances):  # noqa: D102 - trivial
        pass

    def score(self, instance):  # noqa: D102 - trivial
        return [1.0 if i == instance.label else 0.0 for i in range(len(instance.candidates))]


def test_evaluate_reports_both_tie_break_conventions():
    instances = [_instance([_node("a"), _node("b"), _node("c")], label=1)]
    metrics = evaluate(_Oracle(), instances, hits_at=(1, 3))
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["mrr_strict"] == pytest.approx(1.0)
    assert metrics["hits@1"] == pytest.approx(1.0)
    assert metrics["n"] == 1.0
    assert "n_strict" not in metrics


def test_evaluate_separates_the_conventions_when_triggers_collide():
    # Gold ties with one other candidate: optimistic rank 0, strict rank 1.
    class _Flat(SuccessorPredictor):
        def fit(self, instances):
            pass

        def score(self, instance):
            return [1.0, 1.0, 0.0]

    instances = [_instance([_node("a"), _node("b"), _node("c")], label=0)]
    metrics = evaluate(_Flat(), instances, hits_at=(1,))
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["mrr_strict"] == pytest.approx(0.5)
    assert metrics["hits@1"] == 1.0
    assert metrics["hits@1_strict"] == 0.0


def test_a_flat_scorer_would_look_perfect_which_is_why_unscorable_exists():
    # The trap: under the optimistic tie-break a fully tied score gives gold rank
    # 0, so a predictor that failed on every instance reports MRR 1.0. This is
    # not hypothetical -- the first SeDGPL smoke run did exactly that.
    class _Flat(SuccessorPredictor):
        def fit(self, instances):
            pass

        def score(self, instance):
            return [0.0] * len(instance.candidates)

    instances = [_instance([_node("a"), _node("b"), _node("c")], label=2)]
    metrics = evaluate(_Flat(), instances, hits_at=(1,))
    assert metrics["mrr"] == pytest.approx(1.0)  # the lie
    assert metrics["mrr_strict"] == pytest.approx(1 / 3)  # the tell
    assert metrics["n_unscorable"] == 0.0


def test_unscorable_instances_count_as_worst_rank_and_are_reported():
    class _Broken(SuccessorPredictor):
        def fit(self, instances):
            pass

        def score(self, instance):
            raise UnscorableInstance(instance.instance_id)

    instances = [_instance([_node("a"), _node("b"), _node("c")], label=0)]
    metrics = evaluate(_Broken(), instances, hits_at=(1,))
    assert metrics["n_unscorable"] == 1.0
    assert metrics["mrr"] == pytest.approx(1 / 3)  # worst of three candidates
    assert metrics["mrr_strict"] == pytest.approx(1 / 3)
    assert metrics["hits@1"] == 0.0
    assert metrics["n"] == 1.0  # never dropped from the denominator


def test_evaluate_rejects_a_predictor_that_scores_the_wrong_arity():
    class _Short(SuccessorPredictor):
        def fit(self, instances):
            pass

        def score(self, instance):
            return [0.0]

    instances = [_instance([_node("a"), _node("b")], label=0)]
    with pytest.raises(ValueError, match="scored 1 of 2 candidates"):
        evaluate(_Short(), instances)
