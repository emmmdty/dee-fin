from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_pairwise_rows(
    reward_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    min_delta: float = 0.0,
) -> list[dict[str, Any]]:
    features_by_candidate = {str(row.get("candidate_id", "")): row.get("features") or {} for row in feature_rows}
    rewards_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        rewards_by_doc[str(row.get("doc_id", ""))].append(row)

    pairs: list[dict[str, Any]] = []
    for doc_id in sorted(rewards_by_doc):
        rows = sorted(
            rewards_by_doc[doc_id],
            key=lambda row: (-float(row.get("reward", 0.0)), str(row.get("candidate_id", ""))),
        )
        for preferred_index, preferred in enumerate(rows):
            for rejected in rows[preferred_index + 1 :]:
                preferred_reward = float(preferred.get("reward", 0.0))
                rejected_reward = float(rejected.get("reward", 0.0))
                reward_delta = round(preferred_reward - rejected_reward, 12)
                if reward_delta <= min_delta:
                    continue
                preferred_candidate_id = str(preferred.get("candidate_id", ""))
                rejected_candidate_id = str(rejected.get("candidate_id", ""))
                pairs.append(
                    {
                        "doc_id": doc_id,
                        "preferred_candidate_id": preferred_candidate_id,
                        "rejected_candidate_id": rejected_candidate_id,
                        "preferred_reward": preferred_reward,
                        "rejected_reward": rejected_reward,
                        "reward_delta": reward_delta,
                        "preferred_features": dict(features_by_candidate.get(preferred_candidate_id, {})),
                        "rejected_features": dict(features_by_candidate.get(rejected_candidate_id, {})),
                        "metric_source": str(preferred.get("metric_source") or rejected.get("metric_source") or ""),
                    }
                )
    return pairs
