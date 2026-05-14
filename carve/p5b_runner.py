from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from carve.allocation import AllocationBatch, CandidateMention, build_allocation_targets, l_alloc, sinkhorn
from carve.datasets import (
    DueeDocument,
    build_candidate_lexicon,
    generate_inference_candidates,
    load_duee_documents,
    multi_event_subset,
    write_canonical_jsonl,
)
from carve.encoder import build_encoder
from carve.p3_planner import RecordPlanner, truncated_poisson_argmax
from carve.text_segmentation import split_sentences
from evaluator.canonical.normalize import normalize_text
from evaluator.canonical.schema import EventSchema


PLANNER_FEATURE_MODES = ("global_only", "evidence", "evidence_lexical")


class PlannerGate:
    """Wraps a trained R3 RecordPlanner so P5b can replace its rule-based type gate and count head.

    Loaded only when --planner-checkpoint is provided. Otherwise P5b retains the existing
    string-match _type_gate and trigger-count _estimate_record_count behavior.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        encoder: nn.Module | None,
        feature_mode: str,
        presence_threshold: float | None,
        device: torch.device,
    ) -> None:
        if feature_mode not in PLANNER_FEATURE_MODES:
            raise ValueError(f"planner feature mode must be in {PLANNER_FEATURE_MODES}, got {feature_mode!r}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = payload.get("planner_metadata") or {}
        checkpoint_mode = str(metadata.get("encoder_feature_mode") or "")
        if checkpoint_mode and checkpoint_mode != feature_mode:
            raise ValueError(
                f"planner feature mode {feature_mode!r} does not match checkpoint mode {checkpoint_mode!r}"
            )
        event_types = list(metadata.get("event_types") or [])
        if not event_types:
            raise ValueError("planner checkpoint metadata missing event_types")
        hidden_size = int(metadata.get("hidden_size") or 0)
        if hidden_size <= 0:
            raise ValueError("planner checkpoint metadata missing hidden_size")
        k_clip = int(metadata.get("k_clip") or 1)
        self.event_types = event_types
        self.type_to_id = {event_type: index for index, event_type in enumerate(event_types)}
        self.feature_mode = feature_mode
        self.k_clip = max(k_clip, 1)
        self.encoder = encoder
        self.device = device
        self.hidden_size = hidden_size
        if presence_threshold is None:
            presence_threshold = float(
                metadata.get("presence_threshold_all_dev")
                or metadata.get("presence_threshold_multi_event_dev")
                or metadata.get("presence_threshold")
                or 0.5
            )
        self.presence_threshold = float(presence_threshold)
        self.metadata = metadata
        count_head_mode = str(metadata.get("count_head_mode") or "document")
        self.count_head_mode = count_head_mode
        self.planner = RecordPlanner(
            hidden_size=hidden_size,
            num_event_types=len(event_types),
            k_max=self.k_clip,
            count_head_mode=count_head_mode,
        ).to(device)
        self.planner.load_state_dict(payload["planner"])
        self.planner.eval()
        if feature_mode in {"evidence", "evidence_lexical"} and encoder is None:
            raise ValueError("encoder is required for evidence / evidence_lexical feature modes")

    @torch.no_grad()
    def predict(self, document: DueeDocument, event_type: str) -> tuple[bool, int]:
        type_id = self.type_to_id.get(event_type)
        if type_id is None:
            return False, 0
        global_repr, sentence_repr, sentence_mask = self._encode(document)
        if self.feature_mode in {"evidence", "evidence_lexical"} or self.count_head_mode == "sentence":
            sentence_kwarg = sentence_repr
            mask_kwarg = sentence_mask
        else:
            sentence_kwarg = None
            mask_kwarg = None
        if self.feature_mode == "evidence_lexical":
            lex_value = 1.0 if _type_gate(document, event_type) else 0.0
            lexical_kwarg = torch.tensor([[lex_value]], dtype=torch.float32, device=self.device)
        else:
            lexical_kwarg = None
        type_tensor = torch.tensor([type_id], dtype=torch.long, device=self.device)
        presence_logit = self.planner.presence_logit(
            global_repr,
            type_tensor,
            sentence_repr=sentence_kwarg,
            sentence_mask=mask_kwarg,
            lexical_hit=lexical_kwarg,
        )
        present_prob = float(torch.sigmoid(presence_logit).reshape(-1)[0].item())
        if present_prob < self.presence_threshold:
            return False, 0
        if self.count_head_mode == "sentence":
            if sentence_kwarg is None or mask_kwarg is None:
                raise ValueError("sentence_repr required for sentence count head in PlannerGate")
            count = int(self.planner.expected_count(type_tensor, sentence_kwarg, mask_kwarg).reshape(-1)[0].item())
        else:
            log_lambda = self.planner.count_log_lambda(
                global_repr,
                type_tensor,
                sentence_repr=sentence_kwarg,
                sentence_mask=mask_kwarg,
                lexical_hit=lexical_kwarg,
            )
            count = int(truncated_poisson_argmax(log_lambda, k_clip=self.k_clip).reshape(-1)[0].item())
        return True, max(count, 1)

    @torch.no_grad()
    def _encode(self, document: DueeDocument) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            global_repr = torch.zeros((1, self.hidden_size), dtype=torch.float32, device=self.device)
            sentence_repr = torch.zeros((1, 0, self.hidden_size), dtype=torch.float32, device=self.device)
            sentence_mask = torch.zeros((1, 0), dtype=torch.bool, device=self.device)
            return global_repr, sentence_repr, sentence_mask
        text = _planner_document_text(document)
        sentences = split_sentences(text)
        if not sentences:
            global_repr = torch.zeros((1, self.hidden_size), dtype=torch.float32, device=self.device)
            sentence_repr = torch.zeros((1, 0, self.hidden_size), dtype=torch.float32, device=self.device)
            sentence_mask = torch.zeros((1, 0), dtype=torch.bool, device=self.device)
            return global_repr, sentence_repr, sentence_mask
        encoded = self.encoder.encode_document(text, sentences)
        global_repr = encoded.global_repr.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        sentence_repr = encoded.sentence_repr.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        sentence_mask = torch.ones((1, sentence_repr.shape[1]), dtype=torch.bool, device=self.device)
        return global_repr, sentence_repr, sentence_mask


def _planner_document_text(document: DueeDocument) -> str:
    title = (document.title or "").strip()
    text = document.text or ""
    if title and not text.startswith(title):
        return f"{title}\n{text}"
    return text


@dataclass(frozen=True)
class TrainingGroup:
    document: DueeDocument
    batch: AllocationBatch


class AllocationDiagnosticModel(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.allocation = nn.Linear(feature_dim, 1)
        self.share = nn.Linear(feature_dim - 2, 1)

    def allocation_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.allocation(features).squeeze(-1)

    def share_logits(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.share(candidate_features).squeeze(-1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CARVE P5b DuEE-Fin diagnostic training.")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--routes", default="baseline,carve")
    parser.add_argument("--share-threshold", type=float, default=0.50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--planner-checkpoint", default="", help="path to R3 planner checkpoint (r3_planner.pt). When set, P5b replaces _type_gate and _estimate_record_count with the trained planner.")
    parser.add_argument("--planner-encoder-path", default="", help="path to encoder safetensors directory; required when --planner-checkpoint is set unless --planner-feature-mode is global_only.")
    parser.add_argument("--planner-feature-mode", choices=PLANNER_FEATURE_MODES, default="evidence_lexical", help="must match the encoder_feature_mode used when the planner checkpoint was trained.")
    parser.add_argument("--planner-presence-threshold", type=float, default=None, help="override presence threshold; default uses checkpoint metadata.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_p5b(args)
    return 0


def run_p5b(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    _set_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", vars(args))
    _write_json(run_dir / "environment.json", _environment(args))

    data_root = Path(args.data_root)
    schema = EventSchema.from_file(args.schema)
    print(json.dumps({"stage": "load_data", "data_root": str(data_root)}, ensure_ascii=False), flush=True)
    train_docs = multi_event_subset(load_duee_documents(data_root / "train.jsonl", dataset=args.dataset))
    dev_docs = multi_event_subset(load_duee_documents(data_root / "dev.jsonl", dataset=args.dataset))
    if args.limit_docs:
        train_docs = train_docs[: args.limit_docs]
        dev_docs = dev_docs[: args.limit_docs]
    if args.smoke:
        train_docs = train_docs[: min(len(train_docs), 16)]
        dev_docs = dev_docs[: min(len(dev_docs), 16)]
    print(
        json.dumps(
            {"stage": "data_loaded", "train_documents": len(train_docs), "dev_documents": len(dev_docs)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    lexicon = build_candidate_lexicon(train_docs, min_count=1)
    print(json.dumps({"stage": "lexicon_built", "event_types": len(lexicon)}, ensure_ascii=False), flush=True)
    groups = _build_training_groups(train_docs, schema, lexicon)
    if not groups:
        raise RuntimeError("no P5b training groups were constructed")
    print(json.dumps({"stage": "training_groups_built", "groups": len(groups)}, ensure_ascii=False), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner_gate: PlannerGate | None = None
    if args.planner_checkpoint:
        planner_encoder = None
        if args.planner_feature_mode in {"evidence", "evidence_lexical"}:
            encoder_path = args.planner_encoder_path or args.model_path
            if not encoder_path:
                raise RuntimeError("--planner-encoder-path is required for evidence / evidence_lexical feature modes")
            planner_encoder = build_encoder(encoder_path, device=device).to(device)
            planner_encoder.eval()
        planner_gate = PlannerGate(
            checkpoint_path=args.planner_checkpoint,
            encoder=planner_encoder,
            feature_mode=args.planner_feature_mode,
            presence_threshold=args.planner_presence_threshold,
            device=device,
        )
        print(
            json.dumps(
                {
                    "stage": "planner_gate_loaded",
                    "checkpoint": str(args.planner_checkpoint),
                    "feature_mode": planner_gate.feature_mode,
                    "presence_threshold": planner_gate.presence_threshold,
                    "k_clip": planner_gate.k_clip,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    model = AllocationDiagnosticModel(feature_dim=9).to(device)
    history = _train(
        model,
        groups,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    _write_json(run_dir / "diagnostics" / "train_history.json", history)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "checkpoints" / "allocation_diagnostic.pt")

    candidate_recall = _compute_candidate_recall(dev_docs, lexicon)
    _write_json(run_dir / "diagnostics" / "candidate_recall.json", candidate_recall)
    print(json.dumps({"stage": "candidate_recall", **candidate_recall}, ensure_ascii=False), flush=True)

    routes = [route.strip() for route in args.routes.split(",") if route.strip()]
    gold_path = run_dir / "canonical" / "dev.gold.jsonl"
    write_canonical_jsonl(gold_path, dev_docs)

    route_reports = {}
    for route in routes:
        pred_rows, route_diagnostics = _predict_route(
            route=route,
            model=model,
            documents=dev_docs,
            schema=schema,
            lexicon=lexicon,
            device=device,
            share_threshold=args.share_threshold,
            planner_gate=planner_gate,
        )
        pred_path = run_dir / "canonical" / f"dev.{route}.pred.jsonl"
        write_canonical_jsonl(pred_path, pred_rows)
        eval_path = run_dir / "eval" / f"dev.{route}.unified_strict.json"
        _run_unified_strict(
            python_bin=args.python_bin,
            dataset=args.dataset,
            gold_path=gold_path,
            pred_path=pred_path,
            schema_path=Path(args.schema),
            out_path=eval_path,
        )
        eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
        route_reports[route] = {
            "prediction_path": str(pred_path),
            "eval_path": str(eval_path),
            "diagnostics": route_diagnostics,
            "unified_strict_overall": eval_report.get("overall", {}),
        }

    report = {
        "dataset": args.dataset,
        "split": "dev",
        "status": "dev_diagnostic_only",
        "routes": route_reports,
        "train_group_count": len(groups),
        "train_document_count": len(train_docs),
        "dev_document_count": len(dev_docs),
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    _write_json(run_dir / "summary.json", report)
    _write_json(run_dir / "diagnostics" / "p5b_duee_fin_decision_row.json", _decision_row(report))
    return report


def _build_training_groups(
    documents: Iterable[DueeDocument],
    schema: EventSchema,
    lexicon: dict[str, dict[str, dict[str, int]]],
) -> list[TrainingGroup]:
    groups = []
    for document in documents:
        by_event: dict[str, list] = defaultdict(list)
        for record in document.records:
            by_event[normalize_text(record.event_type)].append(record)
        for event_type, records in by_event.items():
            roles = schema.roles_for(event_type)
            for role in roles:
                candidates = _training_candidates(document, lexicon, event_type=event_type, role=role)
                batch = build_allocation_targets(
                    records=records,
                    event_type=event_type,
                    role=role,
                    candidates=candidates,
                    oracle_inject=True,
                )
                if batch.candidates:
                    groups.append(TrainingGroup(document=document, batch=batch))
    return groups


def _training_candidates(
    document: DueeDocument,
    lexicon: dict[str, dict[str, dict[str, int]]],
    *,
    event_type: str,
    role: str,
) -> list[CandidateMention]:
    return generate_inference_candidates(document, lexicon, event_type=event_type, role=role)


def _train(
    model: AllocationDiagnosticModel,
    groups: list[TrainingGroup],
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    grad_accum: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale = 0
    batch_size = max(batch_size, 1)
    grad_accum = max(grad_accum, 1)
    for epoch in range(1, max_epochs + 1):
        random.shuffle(groups)
        total = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)
        pending_steps = 0
        for start in range(0, len(groups), batch_size):
            mini_batch = groups[start : start + batch_size]
            loss = torch.stack([_group_loss(model, group, device) for group in mini_batch]).mean()
            (loss / grad_accum).backward()
            pending_steps += 1
            if pending_steps % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * len(mini_batch)
            count += len(mini_batch)
        if pending_steps % grad_accum:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        avg = total / max(count, 1)
        history.append({"epoch": float(epoch), "loss": avg})
        if avg + 1e-5 < best_loss:
            best_loss = avg
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "loss": avg}, ensure_ascii=False), flush=True)
        if stale >= patience:
            break
    return history


def _group_loss(model: AllocationDiagnosticModel, group: TrainingGroup, device: torch.device) -> torch.Tensor:
    batch = group.batch
    features = _allocation_feature_matrix(group.document, batch).to(device)
    logits = model.allocation_logits(features).reshape(batch.target.shape)
    probs = sinkhorn(logits)
    target = batch.target.to(device)
    alloc_loss = l_alloc(probs, target, positive_coverage_mu=0.05)
    candidate_features = _candidate_feature_matrix(group.document, batch.candidates).to(device)
    share_logits = model.share_logits(candidate_features)
    share_target = torch.tensor(batch.share_labels, dtype=torch.float32, device=device)
    share_loss = nn.functional.binary_cross_entropy_with_logits(share_logits, share_target)
    return alloc_loss + 0.1 * share_loss


def _predict_route(
    *,
    route: str,
    model: AllocationDiagnosticModel,
    documents: list[DueeDocument],
    schema: EventSchema,
    lexicon: dict[str, dict[str, dict[str, int]]],
    device: torch.device,
    share_threshold: float = 0.50,
    planner_gate: PlannerGate | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    candidate_total = 0
    predicted_record_total = 0
    predicted_by_event_type: dict[str, int] = defaultdict(int)
    arguments_by_event_type: dict[str, int] = defaultdict(int)
    for document in documents:
        events = []
        for event_type, roles in schema.event_roles.items():
            if planner_gate is not None:
                present, neural_count = planner_gate.predict(document, event_type)
                if not present:
                    continue
            else:
                neural_count = None
                if not _type_gate(document, event_type):
                    continue
            role_candidates = {
                role: generate_inference_candidates(document, lexicon, event_type=event_type, role=role)
                for role in roles
            }
            role_candidates = {role: candidates for role, candidates in role_candidates.items() if candidates}
            if not role_candidates:
                continue
            record_count = neural_count if neural_count is not None else _estimate_record_count(document, event_type)
            records = [defaultdict(list) for _ in range(record_count)]
            for role, candidates in role_candidates.items():
                candidate_total += len(candidates)
                if route == "baseline":
                    records[0][role].append(candidates[0].value)
                    continue
                batch = build_allocation_targets(
                    records=[],
                    event_type=event_type,
                    role=role,
                    candidates=candidates,
                    oracle_inject=False,
                )
                probs = _predict_allocation_probs(model, document, batch, record_count, device)
                share_probs = torch.sigmoid(
                    model.share_logits(_candidate_feature_matrix(document, candidates).to(device))
                ).detach().cpu()
                proposals: dict[int, tuple[str, float]] = {}
                for row_index, candidate in enumerate(candidates):
                    row = probs[row_index]
                    null_score = row[-1]
                    best_index = int(torch.argmax(row[:-1]).item()) if record_count else 0
                    if row[best_index] <= null_score:
                        continue
                    if share_probs[row_index] >= 0.5:
                        for column_index, score in enumerate(row[:-1]):
                            if score >= share_threshold:
                                value_score = float(score)
                                existing = proposals.get(column_index)
                                if existing is None or value_score > existing[1]:
                                    proposals[column_index] = (candidate.value, value_score)
                    else:
                        value_score = float(row[best_index])
                        existing = proposals.get(best_index)
                        if existing is None or value_score > existing[1]:
                            proposals[best_index] = (candidate.value, value_score)
                for column_index, (value, _) in proposals.items():
                    records[column_index][role].append(value)
            for record_index, arguments in enumerate(records):
                compact = {role: sorted(set(values)) for role, values in arguments.items() if values}
                if compact:
                    predicted_record_total += 1
                    predicted_by_event_type[event_type] += 1
                    arguments_by_event_type[event_type] += sum(len(values) for values in compact.values())
                    events.append(
                        {
                            "event_type": event_type,
                            "record_id": f"{route}:{event_type}:{record_index}",
                            "arguments": compact,
                        }
                    )
        rows.append({"document_id": document.document_id, "events": events})
    return rows, {
        "candidate_count": candidate_total,
        "predicted_record_count": predicted_record_total,
        "predicted_by_event_type": dict(predicted_by_event_type),
        "arguments_by_event_type": dict(arguments_by_event_type),
    }


def _predict_allocation_probs(
    model: AllocationDiagnosticModel,
    document: DueeDocument,
    batch: AllocationBatch,
    record_count: int,
    device: torch.device,
) -> torch.Tensor:
    pseudo_target = torch.zeros((len(batch.candidates), record_count + 1), dtype=torch.float32)
    pseudo_batch = AllocationBatch(
        event_type=batch.event_type,
        role=batch.role,
        records=[],
        candidates=batch.candidates,
        target=pseudo_target,
        share_labels=[False] * len(batch.candidates),
    )
    features = _allocation_feature_matrix(document, pseudo_batch, record_count=record_count).to(device)
    logits = model.allocation_logits(features).reshape(pseudo_target.shape)
    return sinkhorn(logits).detach().cpu()


def _allocation_feature_matrix(
    document: DueeDocument,
    batch: AllocationBatch,
    *,
    record_count: int | None = None,
) -> torch.Tensor:
    n_columns = (record_count + 1) if record_count is not None else batch.target.shape[1]
    candidate_features = _candidate_feature_matrix(document, batch.candidates)
    rows = []
    denom = max(n_columns - 1, 1)
    for candidate_index in range(candidate_features.shape[0]):
        for column_index in range(n_columns):
            is_null = 1.0 if column_index == n_columns - 1 else 0.0
            record_position = -1.0 if is_null else column_index / denom
            rows.append(torch.cat([candidate_features[candidate_index], torch.tensor([record_position, is_null])]))
    return torch.stack(rows) if rows else torch.zeros((0, 9), dtype=torch.float32)


def _candidate_feature_matrix(document: DueeDocument, candidates: list[CandidateMention]) -> torch.Tensor:
    title = document.title
    text = f"{document.title}\n{document.text}"
    features = []
    for candidate in candidates:
        value = candidate.value
        features.append(
            [
                1.0,
                min(len(value), 30) / 30.0,
                float(any(char.isdigit() for char in value)),
                float("%" in value or "％" in value),
                float(value in title),
                float(candidate.oracle_injected),
                min(text.count(value), 5) / 5.0 if value else 0.0,
            ]
        )
    return torch.tensor(features, dtype=torch.float32) if features else torch.zeros((0, 7), dtype=torch.float32)


def _compute_candidate_recall(
    documents: list[DueeDocument],
    lexicon: dict[str, dict[str, dict[str, int]]],
) -> dict[str, object]:
    total = 0
    covered = 0
    for document in documents:
        for record in document.records:
            event_type = normalize_text(record.event_type)
            for role, values in record.arguments.items():
                normalized_role = normalize_text(role)
                candidates = generate_inference_candidates(
                    document, lexicon, event_type=event_type, role=normalized_role
                )
                candidate_values = {c.value for c in candidates}
                for value in values:
                    total += 1
                    if normalize_text(value) in candidate_values:
                        covered += 1
    return {
        "candidate_recall": round(covered / max(total, 1), 6),
        "covered_gold_args": covered,
        "total_gold_args": total,
    }


def _estimate_record_count(document: DueeDocument, event_type: str) -> int:
    trigger_count = document.text.count(event_type) + document.title.count(event_type)
    return max(1, min(3, trigger_count))


def _type_gate(document: DueeDocument, event_type: str) -> bool:
    haystack = normalize_text(f"{document.title}\n{document.text}")
    return normalize_text(event_type) in haystack


def _run_unified_strict(
    *,
    python_bin: str,
    dataset: str,
    gold_path: Path,
    pred_path: Path,
    schema_path: Path,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        "-m",
        "evaluator",
        "unified-strict",
        "--dataset",
        dataset,
        "--gold",
        str(gold_path),
        "--pred",
        str(pred_path),
        "--schema",
        str(schema_path),
        "--out",
        str(out_path),
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"unified-strict failed: {' '.join(command)}\n{process.stderr}")


def _decision_row(report: dict[str, Any]) -> dict[str, Any]:
    baseline = report["routes"].get("baseline", {}).get("unified_strict_overall", {})
    carve = report["routes"].get("carve", {}).get("unified_strict_overall", {})
    baseline_f1 = float(baseline.get("f1", 0.0) or 0.0)
    carve_f1 = float(carve.get("f1", 0.0) or 0.0)
    if carve_f1 >= baseline_f1 + 0.01:
        label = "Weak"
    else:
        label = "No support"
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "status": "dev_diagnostic_only_not_final_test",
        "baseline_unified_strict_f1": baseline_f1,
        "carve_unified_strict_f1": carve_f1,
        "support_label": label,
        "decision_rule": "first DuEE-Fin diagnostic row only; full P5b requires all datasets and final table",
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": os.environ.get("DEE_FIN_GIT_COMMIT"),
        "model_path_recorded": args.model_path,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
