from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from carve.datasets import load_duee_documents, multi_event_subset
from carve.encoder import build_encoder
from carve.p2_heads import EvidenceHead, PointerHead, build_evidence_labels, evidence_bce_loss, pointer_mi_loss
from carve.text_segmentation import Sentence, split_sentences
from evaluator.canonical.schema import EventSchema


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CARVE P2 evidence and pointer-head training.")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4, help="lr for heads and embeddings")
    parser.add_argument("--encoder-lr", type=float, default=2e-5, help="lr for encoder fine-tune")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="max_norm for grad clipping; 0 disables")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="linear warmup fraction of total steps")
    parser.add_argument("--lambda-ground-mi", type=float, default=0.5)
    parser.add_argument("--grad-accum", type=int, default=8, help="accumulate gradients over this many docs before optimizer step")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_p2(build_arg_parser().parse_args(argv))
    return 0


def run_p2(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    _set_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", vars(args))
    _write_json(run_dir / "environment.json", _environment(args))

    schema = EventSchema.from_file(args.schema)
    train_docs = multi_event_subset(load_duee_documents(Path(args.data_root) / "train.jsonl", dataset=args.dataset))
    dev_docs = multi_event_subset(load_duee_documents(Path(args.data_root) / "dev.jsonl", dataset=args.dataset))
    if args.limit_docs:
        train_docs = train_docs[: args.limit_docs]
        dev_docs = dev_docs[: args.limit_docs]
    if args.smoke:
        train_docs = train_docs[: min(len(train_docs), 16)]
        dev_docs = dev_docs[: min(len(dev_docs), 16)]

    device = torch.device("cuda" if torch.cuda.is_available() and args.model_path != "__toy__" else "cpu")
    encoder = build_encoder(args.model_path, device=device).to(device)
    hidden_size = int(getattr(encoder, "hidden_size"))
    max_roles = max((len(roles) for roles in schema.event_roles.values()), default=1)
    evidence_head = EvidenceHead(hidden_size, len(schema.event_roles), max_roles).to(device)
    pointer_head = PointerHead(hidden_size).to(device)
    type_embedding = nn.Embedding(max(len(schema.event_roles), 1), hidden_size).to(device)
    role_embedding = nn.Embedding(max(max_roles, 1), hidden_size).to(device)
    head_params = (
        list(evidence_head.parameters())
        + list(pointer_head.parameters())
        + list(type_embedding.parameters())
        + list(role_embedding.parameters())
    )
    encoder_params = list(encoder.parameters())
    parameters = encoder_params + head_params
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr, "initial_lr": args.encoder_lr},
            {"params": head_params, "lr": args.lr, "initial_lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    total_steps = max(args.max_epochs * len(train_docs), 1)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)

    def _lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    grad_accum = max(args.grad_accum, 1)
    global_step = 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.max_epochs + 1):
        random.shuffle(train_docs)
        total = {"loss": 0.0, "evidence_bce": 0.0, "pointer_mi": 0.0, "documents": 0.0}
        accum_count = 0
        for document in train_docs:
            # encode_document batches all sentences of this doc in one GPU forward.
            metrics = _p2_document_loss(
                encoder, evidence_head, pointer_head, type_embedding, role_embedding,
                document, schema, lambda_ground_mi=args.lambda_ground_mi, device=device,
            )
            if metrics is None:
                continue
            (metrics["loss"] / grad_accum).backward()
            accum_count += 1
            total["loss"] += float(metrics["loss"].detach().cpu())
            total["evidence_bce"] += float(metrics["evidence_bce"].cpu())
            total["pointer_mi"] += float(metrics["pointer_mi"].cpu())
            total["documents"] += 1.0
            if accum_count % grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.grad_clip)
                scale = _lr_scale(global_step)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        if accum_count % grad_accum != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.grad_clip)
            scale = _lr_scale(global_step)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        denom = max(total["documents"], 1.0)
        row = {
            "epoch": epoch,
            "loss": total["loss"] / denom,
            "evidence_bce": total["evidence_bce"] / denom,
            "pointer_mi": total["pointer_mi"] / denom,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        _write_json(run_dir / "diagnostics" / "p2_train_history.json", history)
    _write_json(run_dir / "diagnostics" / "p2_train_history.json", history)
    evidence_metrics = _evaluate_p2(encoder, evidence_head, pointer_head, type_embedding, role_embedding, dev_docs, schema, device)
    _write_json(run_dir / "diagnostics" / "evidence_metrics.json", evidence_metrics)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "evidence_head": evidence_head.state_dict(),
            "pointer_head": pointer_head.state_dict(),
            "type_embedding": type_embedding.state_dict(),
            "role_embedding": role_embedding.state_dict(),
        },
        run_dir / "checkpoints" / "p2.pt",
    )
    report = {
        "status": "p2_smoke" if args.smoke else "p2_diagnostic",
        "dataset": args.dataset,
        "train_documents": len(train_docs),
        "dev_documents": len(dev_docs),
        "history": history,
        "evidence_metrics": evidence_metrics,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "non_goals": ["no unified-strict dev scoring", "no hidden-test", "no paper main-table claim"],
    }
    _write_json(run_dir / "summary.json", report)
    return report


def _p2_doc_loss_from_sent_repr(
    sentence_repr: torch.Tensor,
    evidence_head: EvidenceHead,
    pointer_head: PointerHead,
    type_embedding: nn.Embedding,
    role_embedding: nn.Embedding,
    document,
    sentences: list[Sentence],
    schema: EventSchema,
    *,
    lambda_ground_mi: float,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    labels = build_evidence_labels(document, sentences, schema)
    type_logits, role_logits = evidence_head(sentence_repr)
    ev_loss = evidence_bce_loss(type_logits, role_logits, _labels_to_device(labels, device))
    log_rows = []
    pos_rows = []
    event_index = {event_type: index for index, event_type in enumerate(labels.event_types)}
    for (event_type, role, value), indices in labels.pos_sent.items():
        type_id = event_index[event_type]
        role_id = labels.roles_by_event_type[event_type].index(role)
        value_repr = _value_repr(value, sentence_repr.shape[-1], device)
        log_rows.append(pointer_head(sentence_repr, type_embedding.weight[type_id], role_embedding.weight[role_id], value_repr))
        pos_rows.append(indices)
    pointer_loss = pointer_mi_loss(torch.stack(log_rows), pos_rows) if log_rows else sentence_repr.sum() * 0.0
    loss = ev_loss + lambda_ground_mi * pointer_loss
    return {"loss": loss, "evidence_bce": ev_loss.detach(), "pointer_mi": pointer_loss.detach()}


def _p2_document_loss(
    encoder: nn.Module,
    evidence_head: EvidenceHead,
    pointer_head: PointerHead,
    type_embedding: nn.Embedding,
    role_embedding: nn.Embedding,
    document,
    schema: EventSchema,
    *,
    lambda_ground_mi: float,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    text = _document_text(document)
    sentences = split_sentences(text)
    if not sentences:
        return None
    labels = build_evidence_labels(document, sentences, schema)
    encoded = encoder.encode_document(text, sentences)
    sentence_repr = encoded.sentence_repr.to(device)
    type_logits, role_logits = evidence_head(sentence_repr)
    ev_loss = evidence_bce_loss(type_logits, role_logits, _labels_to_device(labels, device))
    log_rows = []
    pos_rows = []
    event_index = {event_type: index for index, event_type in enumerate(labels.event_types)}
    for (event_type, role, value), indices in labels.pos_sent.items():
        type_id = event_index[event_type]
        role_id = labels.roles_by_event_type[event_type].index(role)
        value_repr = _value_repr(value, sentence_repr.shape[-1], device)
        log_rows.append(pointer_head(sentence_repr, type_embedding.weight[type_id], role_embedding.weight[role_id], value_repr))
        pos_rows.append(indices)
    if log_rows:
        pointer_loss = pointer_mi_loss(torch.stack(log_rows), pos_rows)
    else:
        pointer_loss = sentence_repr.sum() * 0.0
    loss = ev_loss + lambda_ground_mi * pointer_loss
    return {"loss": loss, "evidence_bce": ev_loss.detach(), "pointer_mi": pointer_loss.detach()}


def _evaluate_p2(
    encoder: nn.Module,
    evidence_head: EvidenceHead,
    pointer_head: PointerHead,
    type_embedding: nn.Embedding,
    role_embedding: nn.Embedding,
    documents,
    schema: EventSchema,
    device: torch.device,
) -> dict[str, float]:
    total_unalignable = 0
    total_args = 0
    total_ev = 0.0
    total_pointer = 0.0
    count = 0
    with torch.no_grad():
        for document in documents:
            text = _document_text(document)
            sentences = split_sentences(text)
            if not sentences:
                continue
            labels = build_evidence_labels(document, sentences, schema)
            total_unalignable += len(labels.unalignable)
            total_args += len(labels.unalignable) + len(labels.pos_sent)
            metrics = _p2_document_loss(
                encoder,
                evidence_head,
                pointer_head,
                type_embedding,
                role_embedding,
                document,
                schema,
                lambda_ground_mi=0.5,
                device=device,
            )
            if metrics is None:
                continue
            total_ev += float(metrics["evidence_bce"].cpu())
            total_pointer += float(metrics["pointer_mi"].cpu())
            count += 1
    return {
        "evidence_bce": round(total_ev / max(count, 1), 6),
        "pointer_mi": round(total_pointer / max(count, 1), 6),
        "unalignable_rate": round(total_unalignable / max(total_args, 1), 6),
        "unalignable_args": total_unalignable,
        "aligned_args": max(total_args - total_unalignable, 0),
    }


def _labels_to_device(labels, device: torch.device):
    return type(labels)(
        y_ev_type=labels.y_ev_type.to(device),
        y_ev_role=labels.y_ev_role.to(device),
        role_mask=labels.role_mask.to(device),
        event_types=labels.event_types,
        roles_by_event_type=labels.roles_by_event_type,
        pos_sent=labels.pos_sent,
        unalignable=labels.unalignable,
    )


def _value_repr(value: str, hidden_size: int, device: torch.device) -> torch.Tensor:
    vector = torch.zeros((hidden_size,), dtype=torch.float32, device=device)
    if not value:
        return vector
    for index, char in enumerate(value):
        vector[(ord(char) + index) % hidden_size] += 1.0
    return vector / max(float(len(value)), 1.0)


def _document_text(document) -> str:
    return f"{document.title}\n{document.text}" if document.title else document.text


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
        "model_path_recorded": args.model_path,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
