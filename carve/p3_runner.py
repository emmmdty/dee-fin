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
from carve.encoder import EncoderOutput, build_encoder
from carve.p2_heads import EvidenceHead, PointerHead, build_evidence_labels, evidence_bce_loss, pointer_mi_loss
from carve.p2_runner import _document_text, _labels_to_device, _value_repr
from carve.p3_mention_crf import MentionCRF, build_bio_labels
from carve.p3_planner import RecordPlanner, presence_loss, truncated_poisson_argmax, truncated_poisson_nll
from carve.text_segmentation import Sentence, split_sentences
from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.schema import EventSchema


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CARVE P3 mention CRF and record planner training.")
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
    parser.add_argument("--lambda-mention", type=float, default=1.0)
    parser.add_argument("--lambda-plan", type=float, default=1.0, help="deprecated; kept for old command compatibility")
    parser.add_argument("--lambda-presence", type=float, default=1.0)
    parser.add_argument("--lambda-count", type=float, default=1.0)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--grad-accum", type=int, default=8, help="accumulate gradients over this many docs before optimizer step")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_p3(build_arg_parser().parse_args(argv))
    return 0


def run_p3(args: argparse.Namespace) -> dict[str, Any]:
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
    planner_train_stats = _planner_train_stats(train_docs, schema)
    k_clip = int(planner_train_stats["k_clip"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.model_path != "__toy__" else "cpu")
    encoder = build_encoder(args.model_path, device=device).to(device)
    hidden_size = int(getattr(encoder, "hidden_size"))
    max_roles = max((len(roles) for roles in schema.event_roles.values()), default=1)
    evidence_head = EvidenceHead(hidden_size, len(schema.event_roles), max_roles).to(device)
    pointer_head = PointerHead(hidden_size).to(device)
    type_embedding = nn.Embedding(max(len(schema.event_roles), 1), hidden_size).to(device)
    role_embedding = nn.Embedding(max(max_roles, 1), hidden_size).to(device)
    mention_crf = MentionCRF(hidden_size=hidden_size).to(device)
    planner = RecordPlanner(hidden_size=hidden_size, num_event_types=max(len(schema.event_roles), 1), k_max=k_clip).to(device)
    head_params = (
        list(evidence_head.parameters())
        + list(pointer_head.parameters())
        + list(type_embedding.parameters())
        + list(role_embedding.parameters())
        + list(mention_crf.parameters())
        + list(planner.parameters())
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
    presence_pos_weight = torch.tensor(float(planner_train_stats["presence_pos_weight"]), dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.max_epochs + 1):
        random.shuffle(train_docs)
        totals = {"loss": 0.0, "p2": 0.0, "mention": 0.0, "planner": 0.0, "presence": 0.0, "count": 0.0, "documents": 0.0}
        accum_count = 0
        for document in train_docs:
            # encode_document batches all sentences of this doc in one GPU forward.
            metrics = _p3_document_loss(
                encoder, evidence_head, pointer_head, type_embedding, role_embedding,
                mention_crf, planner, document, schema,
                lambda_ground_mi=args.lambda_ground_mi,
                lambda_mention=args.lambda_mention,
                lambda_presence=args.lambda_presence,
                lambda_count=args.lambda_count,
                presence_pos_weight=presence_pos_weight,
                k_clip=k_clip,
                device=device,
            )
            if metrics is None:
                continue
            (metrics["loss"] / grad_accum).backward()
            accum_count += 1
            for key in ("loss", "p2", "mention", "planner", "presence", "count"):
                totals[key] += float(metrics[key].detach().cpu())
            totals["documents"] += 1.0
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
        denom = max(totals["documents"], 1.0)
        row = {key: totals[key] / denom for key in ("loss", "p2", "mention", "planner")}
        row["presence_loss"] = totals["presence"] / denom
        row["count_loss"] = totals["count"] / denom
        row["presence_pos_frac"] = planner_train_stats["presence_pos_frac"]
        row["presence_pos_weight"] = planner_train_stats["presence_pos_weight"]
        row["k_clip"] = k_clip
        row["epoch"] = epoch
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        _write_json(run_dir / "diagnostics" / "p3_train_history.json", history)

    mention_metrics = _evaluate_mentions(encoder, mention_crf, dev_docs, schema, device)
    planner_metrics = _evaluate_planner(encoder, planner, dev_docs, schema, device, k_clip)
    _write_json(run_dir / "diagnostics" / "p3_train_history.json", history)
    _write_json(run_dir / "diagnostics" / "p3_mention_metrics.json", mention_metrics)
    _write_json(run_dir / "diagnostics" / "p3_planner_metrics.json", planner_metrics)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "evidence_head": evidence_head.state_dict(),
            "pointer_head": pointer_head.state_dict(),
            "type_embedding": type_embedding.state_dict(),
            "role_embedding": role_embedding.state_dict(),
            "mention_crf": mention_crf.state_dict(),
            "planner": planner.state_dict(),
            "planner_metadata": {
                "k_clip": k_clip,
                "presence_pos_weight": planner_train_stats["presence_pos_weight"],
                "presence_threshold": planner_metrics["presence_threshold"],
            },
        },
        run_dir / "checkpoints" / "p3.pt",
    )
    report = {
        "status": "p3_smoke" if args.smoke else "p3_diagnostic",
        "dataset": args.dataset,
        "train_documents": len(train_docs),
        "dev_documents": len(dev_docs),
        "history": history,
        "mention_metrics": mention_metrics,
        "planner_metrics": planner_metrics,
        "planner_train_stats": planner_train_stats,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "non_goals": ["no unified-strict dev scoring", "no hidden-test", "no paper main-table claim"],
    }
    _write_json(run_dir / "summary.json", report)
    return report


def _p3_doc_loss_from_encoded(
    sentence_repr: torch.Tensor,
    encoded: EncoderOutput,
    evidence_head: EvidenceHead,
    pointer_head: PointerHead,
    type_embedding: nn.Embedding,
    role_embedding: nn.Embedding,
    mention_crf: MentionCRF,
    planner: RecordPlanner,
    document,
    sentences: list[Sentence],
    schema: EventSchema,
    *,
    lambda_ground_mi: float,
    lambda_mention: float,
    lambda_presence: float,
    lambda_count: float,
    presence_pos_weight: torch.Tensor,
    k_clip: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    labels = build_evidence_labels(document, sentences, schema)
    device_labels = _labels_to_device(labels, device)
    type_logits, role_logits = evidence_head(sentence_repr)
    ev_loss = evidence_bce_loss(type_logits, role_logits, device_labels)
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
    p2_loss = ev_loss + lambda_ground_mi * pointer_loss
    mention_losses = []
    for sentence in _selected_gold_sentences(labels, sentences):
        token_repr = _sentence_token_repr(encoded, sentence, sentence_repr.shape[-1], device)
        if token_repr.shape[0] == 0:
            continue
        tokens = _sentence_tokens(encoded, sentence)
        gold_values = _gold_values_for_sentence(document, sentence, schema)
        bio = torch.tensor([build_bio_labels(sentence, tokens, gold_values)], dtype=torch.long, device=device)
        mask = torch.ones_like(bio, dtype=torch.bool)
        mention_losses.append(mention_crf.forward_loss(token_repr.unsqueeze(0), bio, mask))
    mention_loss = torch.stack(mention_losses).mean() if mention_losses else sentence_repr.sum() * 0.0
    presence_logits = []
    presence_targets = []
    count_log_lambdas = []
    count_targets = []
    global_repr = encoded.global_repr.to(device).unsqueeze(0)
    for type_id, event_type in enumerate(schema.event_roles):
        type_tensor = torch.tensor([type_id], device=device)
        gold_count = RecordPlanner.gold_n_t(document.records, event_type)
        presence_logits.append(planner.presence_logit(global_repr, type_tensor).reshape(1))
        presence_targets.append(1.0 if gold_count > 0 else 0.0)
        if gold_count > 0:
            count_log_lambdas.append(planner.count_log_lambda(global_repr, type_tensor).reshape(1))
            count_targets.append(float(gold_count))
    if presence_logits:
        presence_logit_tensor = torch.cat(presence_logits)
        presence_target_tensor = torch.tensor(presence_targets, dtype=torch.float32, device=device)
        presence_loss_value = presence_loss(
            presence_logit_tensor,
            presence_target_tensor,
            pos_weight=presence_pos_weight,
        )
    else:
        presence_loss_value = sentence_repr.sum() * 0.0
    if count_log_lambdas:
        count_log_lambda_tensor = torch.cat(count_log_lambdas)
        count_target_tensor = torch.tensor(count_targets, dtype=torch.float32, device=device)
        count_loss_value = truncated_poisson_nll(count_log_lambda_tensor, count_target_tensor)
    else:
        count_loss_value = sentence_repr.sum() * 0.0
    plan_loss = lambda_presence * presence_loss_value + lambda_count * count_loss_value
    total = p2_loss + lambda_mention * mention_loss + plan_loss
    return {
        "loss": total,
        "p2": p2_loss.detach(),
        "mention": mention_loss.detach(),
        "planner": plan_loss.detach(),
        "presence": presence_loss_value.detach(),
        "count": count_loss_value.detach(),
    }


def _p3_document_loss(
    encoder: nn.Module,
    evidence_head: EvidenceHead,
    pointer_head: PointerHead,
    type_embedding: nn.Embedding,
    role_embedding: nn.Embedding,
    mention_crf: MentionCRF,
    planner: RecordPlanner,
    document,
    schema: EventSchema,
    *,
    lambda_ground_mi: float,
    lambda_mention: float,
    lambda_presence: float,
    lambda_count: float,
    presence_pos_weight: torch.Tensor,
    k_clip: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    text = _document_text(document)
    sentences = split_sentences(text)
    if not sentences:
        return None
    encoded = encoder.encode_document(text, sentences)
    sentence_repr = encoded.sentence_repr.to(device)
    return _p3_doc_loss_from_encoded(
        sentence_repr, encoded, evidence_head, pointer_head,
        type_embedding, role_embedding, mention_crf, planner,
        document, sentences, schema,
        lambda_ground_mi=lambda_ground_mi,
        lambda_mention=lambda_mention,
        lambda_presence=lambda_presence,
        lambda_count=lambda_count,
        presence_pos_weight=presence_pos_weight,
        k_clip=k_clip,
        device=device,
    )


def _evaluate_mentions(
    encoder: nn.Module,
    mention_crf: MentionCRF,
    documents,
    schema: EventSchema,
    device: torch.device,
) -> dict[str, float]:
    gold_total = 0
    pred_total = 0
    matched = 0
    with torch.no_grad():
        for document in documents:
            text = _document_text(document)
            sentences = split_sentences(text)
            if not sentences:
                continue
            encoded = encoder.encode_document(text, sentences)
            for sentence in sentences:
                gold_values = _gold_values_for_sentence(document, sentence, schema)
                gold_spans = {(event_type, role, value) for event_type, role, value in gold_values}
                gold_total += len(gold_spans)
                token_repr = _sentence_token_repr(encoded, sentence, int(getattr(encoder, "hidden_size")), device)
                if token_repr.shape[0] == 0:
                    continue
                mask = torch.ones((1, token_repr.shape[0]), dtype=torch.bool, device=device)
                spans = mention_crf.decode(token_repr.unsqueeze(0), mask)[0]
                tokens = _sentence_tokens(encoded, sentence)
                predicted_values = set()
                for start, end in spans:
                    if start < len(tokens) and end <= len(tokens):
                        raw = "".join(token.text for token in tokens[start:end])
                        value = normalize_optional_text(raw)
                        if value:
                            predicted_values.add(value)
                pred_total += len(predicted_values)
                gold_value_set = {value for _event_type, _role, value in gold_spans}
                matched += len(predicted_values & gold_value_set)
    precision = matched / max(pred_total, 1)
    recall = matched / max(gold_total, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "mention_precision": round(precision, 6),
        "mention_recall": round(recall, 6),
        "mention_f1": round(f1, 6),
        "matched_mentions": matched,
        "predicted_mentions": pred_total,
        "gold_mentions": gold_total,
    }


def _planner_train_stats(documents, schema: EventSchema) -> dict[str, float | int]:
    total = 0
    positive = 0
    max_n = 0
    for document in documents:
        for event_type in schema.event_roles:
            n_t = RecordPlanner.gold_n_t(document.records, event_type)
            total += 1
            if n_t > 0:
                positive += 1
            max_n = max(max_n, n_t)
    negative = total - positive
    return {
        "event_type_pairs": total,
        "presence_positive": positive,
        "presence_negative": negative,
        "presence_pos_frac": round(positive / max(total, 1), 6),
        "presence_pos_weight": round(negative / positive, 6) if positive else 1.0,
        "train_max_n_t": max_n,
        "k_clip": max(max_n, 1),
    }


def _evaluate_planner(
    encoder: nn.Module,
    planner: RecordPlanner,
    documents,
    schema: EventSchema,
    device: torch.device,
    k_clip: int,
) -> dict[str, float]:
    rows = []
    positive_abs = 0.0
    positive_count = 0
    truncated = 0
    with torch.no_grad():
        for document in documents:
            text = _document_text(document)
            sentences = split_sentences(text)
            if not sentences:
                continue
            encoded = encoder.encode_document(text, sentences)
            global_repr = encoded.global_repr.to(device)
            for type_id, event_type in enumerate(schema.event_roles):
                gold = RecordPlanner.gold_n_t(document.records, event_type)
                if gold > k_clip:
                    truncated += 1
                type_tensor = torch.tensor([type_id], device=device)
                presence_prob = float(torch.sigmoid(planner.presence_logit(global_repr.unsqueeze(0), type_tensor)).reshape(-1)[0].cpu())
                count_pred = int(truncated_poisson_argmax(planner.count_log_lambda(global_repr.unsqueeze(0), type_tensor), k_clip=k_clip).reshape(-1)[0].cpu())
                rows.append({"gold": gold, "presence_prob": presence_prob, "count_pred": count_pred})
                if gold > 0:
                    positive_abs += abs(count_pred - gold)
                    positive_count += 1
    labels = [1 if row["gold"] > 0 else 0 for row in rows]
    scores = [float(row["presence_prob"]) for row in rows]
    gate_metrics = _presence_threshold_metrics(labels, scores)
    threshold = gate_metrics["threshold"]
    total_abs = 0.0
    positive_gold = 0
    positive_recalled = 0
    for row in rows:
        pred = int(row["count_pred"]) if float(row["presence_prob"]) >= threshold else 0
        gold = int(row["gold"])
        total_abs += abs(pred - gold)
        if gold > 0:
            positive_gold += 1
            if pred > 0:
                positive_recalled += 1
    return {
        "planner_mae": round(total_abs / max(len(rows), 1), 6),
        "type_gate_recall": round(positive_recalled / max(positive_gold, 1), 6),
        "type_gate_auc": round(_binary_auc(labels, scores), 6),
        "type_gate_f1_youden": round(gate_metrics["f1"], 6),
        "type_gate_precision_youden": round(gate_metrics["precision"], 6),
        "type_gate_recall_youden": round(gate_metrics["recall"], 6),
        "presence_threshold": round(threshold, 6),
        "count_mae_positive": round(positive_abs / max(positive_count, 1), 6),
        "evaluated_event_types": len(rows),
        "positive_gold_event_types": positive_count,
        "truncated_gold_event_types": truncated,
        "truncation_rate": round(truncated / max(len(rows), 1), 6),
        "k_clip": k_clip,
    }


def _presence_threshold_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if not labels:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    positives = sum(labels)
    negatives = len(labels) - positives
    best = {"threshold": 0.5, "youden_j": -2.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in sorted(set(scores + [0.5]), reverse=True):
        tp = fp = tn = fn = 0
        for label, score in zip(labels, scores):
            pred = score >= threshold
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif not pred and label:
                fn += 1
            else:
                tn += 1
        tpr = tp / max(positives, 1)
        fpr = fp / max(negatives, 1)
        precision = tp / max(tp + fp, 1)
        recall = tpr
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        youden_j = tpr - fpr
        if (youden_j, f1, threshold) > (best["youden_j"], best["f1"], best["threshold"]):
            best = {
                "threshold": float(threshold),
                "youden_j": youden_j,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
    return best


def _binary_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    rank = 1
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        rank += j - i + 1
        i = j + 1
    positive_rank_sum = sum(rank_value for rank_value, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _selected_gold_sentences(labels, sentences: list[Sentence]) -> list[Sentence]:
    selected = []
    for index, sentence in enumerate(sentences):
        if labels.y_ev_type[index].sum().item() > 0:
            selected.append(sentence)
    return selected


def _gold_values_for_sentence(document, sentence: Sentence, schema: EventSchema) -> list[tuple[str, str, str]]:
    sentence_text = normalize_text(sentence.text)
    values: list[tuple[str, str, str]] = []
    for event_type, roles in schema.event_roles.items():
        for role in roles:
            for record in document.records:
                if normalize_text(record.event_type) != event_type:
                    continue
                role_values = []
                for record_role, record_values in record.arguments.items():
                    if normalize_text(record_role) == role:
                        role_values.extend(record_values)
                for value in role_values:
                    normalized_value = normalize_optional_text(value)
                    if normalized_value and normalized_value in sentence_text:
                        values.append((event_type, role, normalized_value))
    return values


def _sentence_token_repr(
    encoded: EncoderOutput,
    sentence: Sentence,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    indices = _sentence_token_indices(encoded, sentence)
    if not indices:
        return torch.zeros((0, hidden_size), dtype=torch.float32, device=device)
    return encoded.token_repr[torch.tensor(indices, dtype=torch.long, device=encoded.token_repr.device)].to(device)


def _sentence_tokens(encoded: EncoderOutput, sentence: Sentence):
    return [encoded.token_offsets[index] for index in _sentence_token_indices(encoded, sentence)]


def _sentence_token_indices(encoded: EncoderOutput, sentence: Sentence) -> list[int]:
    return [
        index
        for index, token in enumerate(encoded.token_offsets)
        if token.char_start >= sentence.char_start and token.char_end <= sentence.char_end
    ]


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
