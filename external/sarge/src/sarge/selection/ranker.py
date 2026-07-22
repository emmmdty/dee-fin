from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sarge.selection.features import FEATURE_NAMES

DEFAULT_WEIGHTS = {
    "bias": 0.0,
    "schema_valid_rate": 1.0,
    "role_coverage": 1.2,
    "duplicate_argument_rate": -1.0,
    "unknown_event_type_count": -1.0,
    "unknown_role_count": -0.7,
    "empty_prediction": -0.8,
    "candidate_length": 0.03,
    "avg_logprob": 0.05,
    "grounding_confidence": 0.8,
    "lesp_event_count_agreement": 0.6,
    "self_consistency_argument_jaccard": 0.4,
    "self_consistency_event_type_jaccard": 0.2,
}


def train_ranker(
    pair_rows: list[dict[str, Any]],
    *,
    mode: str = "weighted_linear",
    epochs: int = 5,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    if mode == "weighted_linear":
        return train_weighted_linear(pair_rows, epochs=epochs, learning_rate=learning_rate)
    if mode == "sklearn_logistic":
        return train_sklearn_logistic(pair_rows)
    if mode == "rule_based":
        return default_rule_based_model(training_pairs=len(pair_rows))
    raise ValueError(f"unsupported MRS ranker mode: {mode}")


def train_weighted_linear(
    pair_rows: list[dict[str, Any]],
    *,
    epochs: int = 5,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    weights = dict(DEFAULT_WEIGHTS)
    if not pair_rows:
        model = default_rule_based_model(training_pairs=0)
        model["fallback_reason"] = "no_pairwise_preferences"
        return model

    updates = 0
    for _ in range(max(1, epochs)):
        for pair in pair_rows:
            preferred = pair.get("preferred_features") or {}
            rejected = pair.get("rejected_features") or {}
            if score_features(weights, preferred) <= score_features(weights, rejected):
                weights["bias"] = weights.get("bias", 0.0) + learning_rate
                for name in FEATURE_NAMES:
                    weights[name] = weights.get(name, 0.0) + learning_rate * (
                        float(preferred.get(name, 0.0)) - float(rejected.get(name, 0.0))
                    )
                updates += 1
    return {
        "version": "mrs_simple_ranker_v0",
        "mode": "weighted_linear",
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "training_pairs": len(pair_rows),
        "epochs": max(1, epochs),
        "updates": updates,
        "fallback_rule_based": False,
    }


def train_sklearn_logistic(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not pair_rows:
        return default_rule_based_model(training_pairs=0)
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - depends on optional environment
        model = default_rule_based_model(training_pairs=len(pair_rows))
        model["fallback_reason"] = f"sklearn_unavailable:{type(exc).__name__}"
        return model

    x_rows = []
    y_rows = []
    for pair in pair_rows:
        preferred = pair.get("preferred_features") or {}
        rejected = pair.get("rejected_features") or {}
        diff = [float(preferred.get(name, 0.0)) - float(rejected.get(name, 0.0)) for name in FEATURE_NAMES]
        x_rows.append(diff)
        y_rows.append(1)
        x_rows.append([-value for value in diff])
        y_rows.append(0)
    classifier = LogisticRegression(random_state=0, solver="liblinear")
    classifier.fit(x_rows, y_rows)
    weights = dict(DEFAULT_WEIGHTS)
    for name, value in zip(FEATURE_NAMES, classifier.coef_[0], strict=True):
        weights[name] = float(value)
    weights["bias"] = float(classifier.intercept_[0])
    return {
        "version": "mrs_simple_ranker_v0",
        "mode": "sklearn_logistic",
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "training_pairs": len(pair_rows),
        "fallback_rule_based": False,
    }


def default_rule_based_model(*, training_pairs: int = 0) -> dict[str, Any]:
    return {
        "version": "mrs_simple_ranker_v0",
        "mode": "rule_based",
        "feature_names": list(FEATURE_NAMES),
        "weights": dict(DEFAULT_WEIGHTS),
        "training_pairs": training_pairs,
        "fallback_rule_based": True,
    }


def score_with_model(model: dict[str, Any] | None, features: dict[str, Any]) -> float:
    weights = (model or default_rule_based_model()).get("weights") or DEFAULT_WEIGHTS
    if not isinstance(weights, dict):
        weights = DEFAULT_WEIGHTS
    return score_features(weights, features)


def score_features(weights: dict[str, Any], features: dict[str, Any]) -> float:
    score = float(weights.get("bias", 0.0))
    for name in FEATURE_NAMES:
        score += float(weights.get(name, 0.0)) * float(features.get(name, 0.0))
    return score


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if model_path.is_dir():
        model_path = model_path / "model.json"
    with model_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"MRS model must be a JSON object: {model_path}")
    return payload


def save_model(model: dict[str, Any], path: str | Path) -> Path:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return model_path
