from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import torch

from evaluator.canonical.normalize import normalize_text
from evaluator.canonical.types import CanonicalEventRecord


@dataclass(frozen=True)
class CandidateMention:
    event_type: str
    role: str
    value: str
    start: int = -1
    end: int = -1
    source: str = "text"
    oracle_injected: bool = False


@dataclass(frozen=True)
class AllocationBatch:
    event_type: str
    role: str
    records: list[CanonicalEventRecord]
    candidates: list[CandidateMention]
    target: torch.Tensor
    share_labels: list[bool]


def build_allocation_targets(
    *,
    records: Iterable[CanonicalEventRecord],
    event_type: str,
    role: str,
    candidates: Iterable[CandidateMention],
    oracle_inject: bool,
) -> AllocationBatch:
    normalized_event_type = normalize_text(event_type)
    normalized_role = normalize_text(role)
    sorted_records = _sort_records(
        record for record in records if normalize_text(record.event_type) == normalized_event_type
    )
    candidate_list = [
        _normalize_candidate(candidate, normalized_event_type, normalized_role)
        for candidate in candidates
        if normalize_text(candidate.event_type) == normalized_event_type and normalize_text(candidate.role) == normalized_role
    ]
    if oracle_inject:
        candidate_list = _with_oracle_candidates(candidate_list, sorted_records, normalized_event_type, normalized_role)

    rows: list[list[float]] = []
    share_labels: list[bool] = []
    for candidate in candidate_list:
        positives = []
        for index, record in enumerate(sorted_records):
            values = {normalize_text(value) for value in record.arguments.get(normalized_role, [])}
            if normalize_text(candidate.value) in values:
                positives.append(index)
        row = [0.0] * (len(sorted_records) + 1)
        if positives:
            for index in positives:
                row[index] = 1.0
        else:
            row[-1] = 1.0
        rows.append(row)
        share_labels.append(len(positives) > 1)

    target = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, len(sorted_records) + 1))
    return AllocationBatch(
        event_type=normalized_event_type,
        role=normalized_role,
        records=sorted_records,
        candidates=candidate_list,
        target=target,
        share_labels=share_labels,
    )


def sinkhorn(logits: torch.Tensor, *, iterations: int = 20, eps: float = 1e-8) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.softmax(logits, dim=-1)
    matrix = torch.exp(logits - logits.max())
    for _ in range(iterations):
        matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(eps)
        matrix = matrix / matrix.sum(dim=0, keepdim=True).clamp_min(eps)
    return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(eps)


def l_alloc(probs: torch.Tensor, target: torch.Tensor, *, positive_coverage_mu: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    if probs.shape != target.shape:
        raise ValueError(f"shape mismatch: probs={tuple(probs.shape)} target={tuple(target.shape)}")
    if probs.numel() == 0:
        return probs.sum()
    positive_mass = (probs * target).sum(dim=1).clamp_min(eps)
    loss = -torch.log(positive_mass).mean()
    if positive_coverage_mu:
        positive_entries = target > 0
        coverage = -torch.log(probs[positive_entries].clamp_min(eps)).mean()
        loss = loss + positive_coverage_mu * coverage
    return loss


def p5a_toy_comparison() -> dict[str, object]:
    baseline_scores = {"wrong-record": 0.72, "correct-record": 0.65}
    allocation_scores = {"wrong-record": 0.18, "correct-record": 0.91}
    baseline_choice = max(baseline_scores, key=baseline_scores.get)
    allocation_choice = max(allocation_scores, key=allocation_scores.get)
    return {
        "baseline_choice": baseline_choice,
        "allocation_aware_choice": allocation_choice,
        "baseline_scores": baseline_scores,
        "allocation_scores": allocation_scores,
        "allocation_margin": allocation_scores["correct-record"] - allocation_scores["wrong-record"],
        "status": "toy_behavior_only",
    }


def _sort_records(records: Iterable[CanonicalEventRecord]) -> list[CanonicalEventRecord]:
    return sorted(records, key=_record_sort_key)


def _record_sort_key(record: CanonicalEventRecord) -> tuple[str, str]:
    arguments = json.dumps(record.arguments, ensure_ascii=False, sort_keys=True)
    return (record.record_id or "", arguments)


def _normalize_candidate(candidate: CandidateMention, event_type: str, role: str) -> CandidateMention:
    return CandidateMention(
        event_type=event_type,
        role=role,
        value=normalize_text(candidate.value),
        start=candidate.start,
        end=candidate.end,
        source=candidate.source,
        oracle_injected=candidate.oracle_injected,
    )


def _with_oracle_candidates(
    candidates: list[CandidateMention],
    records: list[CanonicalEventRecord],
    event_type: str,
    role: str,
) -> list[CandidateMention]:
    seen = {normalize_text(candidate.value) for candidate in candidates}
    result = list(candidates)
    for record in records:
        for raw_value in record.arguments.get(role, []):
            value = normalize_text(raw_value)
            if value and value not in seen:
                result.append(
                    CandidateMention(
                        event_type=event_type,
                        role=role,
                        value=value,
                        source="oracle_train_only",
                        oracle_injected=True,
                    )
                )
                seen.add(value)
    return result
