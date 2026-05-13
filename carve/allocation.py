from __future__ import annotations

import json
import math
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
    raw_span: str | None = None


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


def p4_toy_validation_summary() -> dict[str, object]:
    records = [
        CanonicalEventRecord(
            document_id="doc1",
            event_type="质押",
            record_id="r2",
            arguments={"质押方": ["甲公司"], "质权方": ["乙银行"]},
        ),
        CanonicalEventRecord(
            document_id="doc1",
            event_type="质押",
            record_id="r1",
            arguments={"质押方": ["甲公司"], "质权方": ["丙银行"]},
        ),
    ]
    shared_candidates = [
        CandidateMention(event_type="质押", role="质押方", value="甲公司", start=3, end=6),
        CandidateMention(event_type="质押", role="质押方", value="无关方", start=9, end=12),
    ]
    shared_batch = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质押方",
        candidates=shared_candidates,
        oracle_inject=False,
    )
    repeated_shared_batch = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质押方",
        candidates=shared_candidates,
        oracle_inject=False,
    )
    single_batch = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质权方",
        candidates=[CandidateMention(event_type="质押", role="质权方", value="丙银行", start=13, end=16)],
        oracle_inject=False,
    )
    oracle_training = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质权方",
        candidates=[],
        oracle_inject=True,
    )
    oracle_inference = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质权方",
        candidates=[],
        oracle_inject=False,
    )

    logits = torch.tensor([[2.0, 2.0, -2.0], [-1.0, -1.0, 3.0]], dtype=torch.float32)
    probs = sinkhorn(logits, iterations=30)
    target = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    marginal_loss = -torch.log((probs * target).sum(dim=1).clamp_min(1e-8)).mean()
    mu_zero_loss = l_alloc(probs, target, positive_coverage_mu=0.0)
    coverage_loss = l_alloc(probs, target, positive_coverage_mu=0.5)

    deterministic = _batch_signature(shared_batch) == _batch_signature(repeated_shared_batch)
    training_oracle_values = [candidate.value for candidate in oracle_training.candidates if candidate.oracle_injected]
    inference_candidates = [candidate.value for candidate in oracle_inference.candidates]
    target_cases = {
        "shared_surface": {
            "candidate": shared_batch.candidates[0].value,
            "target_row": shared_batch.target.tolist()[0],
            "share_label": shared_batch.share_labels[0],
        },
        "null_candidate": {
            "candidate": shared_batch.candidates[1].value,
            "target_row": shared_batch.target.tolist()[1],
            "share_label": shared_batch.share_labels[1],
        },
        "single_positive": {
            "candidate": single_batch.candidates[0].value,
            "target_row": single_batch.target.tolist()[0],
            "share_label": single_batch.share_labels[0],
        },
    }
    acceptance_checks = {
        "repeated_surface_shared_by_two_records": target_cases["shared_surface"]["target_row"] == [1.0, 1.0, 0.0],
        "non_shared_value_single_positive_column": target_cases["single_positive"]["target_row"] == [1.0, 0.0, 0.0],
        "unmatched_candidate_assigned_to_null": target_cases["null_candidate"]["target_row"] == [0.0, 0.0, 1.0],
        "deterministic_repeated_execution": deterministic,
        "oracle_injected_candidate_train_only": bool(training_oracle_values) and not inference_candidates,
        "mu_zero_disables_positive_coverage": bool(torch.allclose(mu_zero_loss, marginal_loss)),
        "share_gate_surface_only": shared_batch.share_labels == [True, False] and single_batch.share_labels == [False],
    }
    return {
        "status": "toy_behavior_only",
        "record_order": [record.record_id for record in shared_batch.records],
        "target_cases": target_cases,
        "oracle_injection": {
            "training_oracle_values": training_oracle_values,
            "inference_candidates": inference_candidates,
        },
        "sinkhorn": {
            "matrix": probs.tolist(),
            "row_sums": probs.sum(dim=1).tolist(),
            "row_sums_close_to_one": bool(torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4)),
        },
        "loss": {
            "mu_zero": float(mu_zero_loss),
            "marginal_only": float(marginal_loss),
            "with_positive_coverage": float(coverage_loss),
            "mu_zero_equals_marginal": bool(torch.allclose(mu_zero_loss, marginal_loss)),
        },
        "acceptance_checks": acceptance_checks,
        "accepted": all(acceptance_checks.values()),
        "non_goals": ["no dev scoring", "no test scoring", "no model training", "no paper main-table claim"],
    }


def p5a_toy_comparison() -> dict[str, object]:
    records = [
        CanonicalEventRecord(
            document_id="doc-p5a",
            event_type="质押",
            record_id="r2",
            arguments={"质押方": ["甲公司"], "质权方": ["乙银行"]},
        ),
        CanonicalEventRecord(
            document_id="doc-p5a",
            event_type="质押",
            record_id="r1",
            arguments={"质押方": ["甲公司"], "质权方": ["丙银行"]},
        ),
    ]
    candidates = [
        CandidateMention(event_type="质押", role="质押方", value="甲公司", start=3, end=6),
        CandidateMention(event_type="质押", role="质权方", value="丙银行", start=18, end=21),
    ]
    shared_batch = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质押方",
        candidates=candidates,
        oracle_inject=False,
    )
    disputed_batch = build_allocation_targets(
        records=records,
        event_type="质押",
        role="质权方",
        candidates=candidates,
        oracle_inject=False,
    )

    record_order = [record.record_id for record in disputed_batch.records]
    base_scores_by_record = {"r1": 0.65, "r2": 0.72}
    allocation_prior_by_record = {"r1": 0.91, "r2": 0.18}
    lambda_alloc = 0.5
    toy_input_signature = {
        "event_type": disputed_batch.event_type,
        "record_order": record_order,
        "candidates": [
            {
                "role": candidate.role,
                "value": candidate.value,
                "start": candidate.start,
                "end": candidate.end,
            }
            for candidate in candidates
        ],
    }

    baseline_route = _p5a_route_summary(
        route_name="no_allocation_baseline",
        toy_input_signature=toy_input_signature,
        score_components={
            record_id: {
                "base": base_scores_by_record[record_id],
                "allocation_log": 0.0,
                "share_adjustment": 0.0,
                "total": base_scores_by_record[record_id],
            }
            for record_id in record_order
        },
    )
    allocation_route = _p5a_route_summary(
        route_name="allocation_prior_share_gate",
        toy_input_signature=toy_input_signature,
        score_components={
            record_id: {
                "base": base_scores_by_record[record_id],
                "allocation_log": math.log(allocation_prior_by_record[record_id]),
                "share_adjustment": 0.0,
                "total": base_scores_by_record[record_id]
                + lambda_alloc * math.log(allocation_prior_by_record[record_id]),
            }
            for record_id in record_order
        },
    )

    label_by_record = {"r1": "correct-record", "r2": "wrong-record"}
    baseline_scores = {
        label_by_record[record_id]: baseline_route["score_components"][record_id]["total"]
        for record_id in record_order
    }
    allocation_scores = {
        label_by_record[record_id]: allocation_route["score_components"][record_id]["total"]
        for record_id in record_order
    }
    baseline_choice = label_by_record[baseline_route["choice"]["record_id"]]
    allocation_choice = label_by_record[allocation_route["choice"]["record_id"]]
    share_gate = {
        "shared_surface_role": shared_batch.candidates[0].role,
        "shared_surface_value": shared_batch.candidates[0].value,
        "shared_surface_share_label": shared_batch.share_labels[0],
        "disputed_role": disputed_batch.candidates[0].role,
        "disputed_value": disputed_batch.candidates[0].value,
        "disputed_value_share_label": disputed_batch.share_labels[0],
    }
    acceptance_checks = {
        "same_toy_inputs": baseline_route["toy_input_signature"] == allocation_route["toy_input_signature"],
        "explicit_deterministic_misallocation": record_order == ["r1", "r2"]
        and disputed_batch.target.tolist()[0] == [1.0, 0.0, 0.0]
        and baseline_choice == "wrong-record",
        "allocation_changes_preference": allocation_choice == "correct-record"
        and allocation_scores["correct-record"] > allocation_scores["wrong-record"],
        "toy_behavior_only": True,
        "oracle_injection_disabled": not any(candidate.oracle_injected for candidate in disputed_batch.candidates),
    }
    return {
        "status": "toy_behavior_only",
        "accepted": all(acceptance_checks.values()),
        "oracle_inject": False,
        "records": [
            {"record_id": record.record_id, "arguments": record.arguments} for record in disputed_batch.records
        ],
        "expected_misallocation": {
            "role": disputed_batch.candidates[0].role,
            "value": disputed_batch.candidates[0].value,
            "wrong_record_id": "r2",
            "correct_record_id": "r1",
            "target_row": disputed_batch.target.tolist()[0],
        },
        "share_gate": share_gate,
        "baseline_route": baseline_route,
        "allocation_aware_route": allocation_route,
        "baseline_choice": baseline_choice,
        "allocation_aware_choice": allocation_choice,
        "baseline_scores": baseline_scores,
        "allocation_scores": allocation_scores,
        "allocation_margin": allocation_scores["correct-record"] - allocation_scores["wrong-record"],
        "acceptance_checks": acceptance_checks,
        "non_goals": ["no dev scoring", "no test scoring", "no model training", "no verifier pruning"],
    }


def _p5a_route_summary(
    *,
    route_name: str,
    toy_input_signature: dict[str, object],
    score_components: dict[str, dict[str, float]],
) -> dict[str, object]:
    choice_record_id = max(score_components, key=lambda record_id: score_components[record_id]["total"])
    return {
        "route": route_name,
        "toy_input_signature": toy_input_signature,
        "score_components": score_components,
        "choice": {
            "record_id": choice_record_id,
            "score": score_components[choice_record_id]["total"],
        },
    }


def _sort_records(records: Iterable[CanonicalEventRecord]) -> list[CanonicalEventRecord]:
    return sorted(records, key=_record_sort_key)


def _record_sort_key(record: CanonicalEventRecord) -> tuple[str, str]:
    arguments = json.dumps(record.arguments, ensure_ascii=False, sort_keys=True)
    return (record.record_id or "", arguments)


def _batch_signature(batch: AllocationBatch) -> tuple[object, ...]:
    return (
        [record.record_id for record in batch.records],
        [(candidate.value, candidate.start, candidate.end, candidate.source) for candidate in batch.candidates],
        batch.target.tolist(),
        batch.share_labels,
    )


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
