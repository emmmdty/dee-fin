"""CPU tests for the supervised trainer's data preparation.

Only the pure-Python helpers are exercised here (the training loop needs a GPU).
The script is loaded by path because `scripts/` is not on `pythonpath` — the same
pattern as `test_evaluate_relation_pairs.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from finekg.relations.pairs import PairExample

_REPO = Path(__file__).resolve().parents[2]


def _load_script():
    path = _REPO / "scripts" / "train_supervised_relations.py"
    spec = importlib.util.spec_from_file_location("train_supervised_relations", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tr = _load_script()


def _example(index: int, labels: dict[str, str]) -> PairExample:
    return PairExample(
        doc_id="d1", head_id=f"h{index}", tail_id=f"t{index}", distance=1, labels=labels
    )


def test_downsample_negatives_is_deterministic_and_hits_the_ratio():
    rows = [_example(i, {}) for i in range(100)]
    rows += [_example(1000 + i, {"causal": "CAUSE"}) for i in range(4)]
    first = tr.downsample_negatives(rows, ratio=3.0, seed=13)
    second = tr.downsample_negatives(rows, ratio=3.0, seed=13)
    assert first == second  # same seed -> same subset
    assert sum(1 for r in first if r.labels) == 4  # every positive kept
    assert sum(1 for r in first if not r.labels) == 12  # 3 negatives per positive


def test_downsample_negatives_refuses_when_there_are_no_positives():
    # Training on NONE only would silently learn the majority class -- fail loudly.
    with pytest.raises(ValueError):
        tr.downsample_negatives([_example(i, {}) for i in range(10)], ratio=3.0)


def test_class_weights_downweight_the_dominant_none_class():
    rows = [_example(i, {}) for i in range(8)]
    rows += [_example(100 + i, {"causal": "CAUSE"}) for i in range(2)]
    causal = tr.class_weights(rows)["causal"]  # (NONE, CAUSE, PRECONDITION)
    assert causal[0] < causal[1]  # frequent NONE weighted below the sparse CAUSE
    assert causal[2] == 0.0  # never seen -> no weight
