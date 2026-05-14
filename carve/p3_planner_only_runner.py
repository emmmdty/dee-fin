from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from carve.datasets import DueeDocument, load_duee_documents, multi_event_subset
from carve.encoder import build_encoder
from carve.p2_runner import _document_text, _environment, _set_seed, _write_json
from carve.p3_planner import RecordPlanner, presence_loss, truncated_poisson_argmax, truncated_poisson_nll
from carve.p3_runner import _binary_auc, _presence_threshold_metrics
from carve.p5b_runner import _estimate_record_count, _type_gate
from carve.text_segmentation import split_sentences
from evaluator.canonical.schema import EventSchema


FEATURE_MODES = ("global_only", "evidence", "evidence_lexical")
TRAIN_POPULATIONS = ("all_train", "multi_event_train")
ACCEPTANCE_POPULATIONS: tuple[str, ...] = ("multi_event_dev", "all_dev")


@dataclass(frozen=True)
class PlannerFeatureCache:
    name: str
    document_ids: list[str]
    documents: list[DueeDocument]
    global_repr: torch.Tensor
    sentence_repr: torch.Tensor
    sentence_mask: torch.Tensor
    lexical_hit: torch.Tensor
    counts: torch.Tensor
    event_types: list[str]
    truncated_sentence_documents: int
    max_sentences: int

    @property
    def documents_count(self) -> int:
        return int(self.global_repr.shape[0])

    @property
    def event_type_pairs(self) -> int:
        return int(self.counts.numel())

    @property
    def truncation_rate(self) -> float:
        denom = max(self.documents_count, 1)
        return round(self.truncated_sentence_documents / denom, 6)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CARVE R3 planner-only training.")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lambda-presence", type=float, default=1.0)
    parser.add_argument("--lambda-count", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--train-population", choices=TRAIN_POPULATIONS, default="all_train")
    parser.add_argument("--encoder-feature-mode", choices=FEATURE_MODES, default="evidence_lexical")
    parser.add_argument("--max-sentences", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_r3_planner_only(build_arg_parser().parse_args(argv))
    return 0


def run_r3_planner_only(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    _set_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", vars(args))
    _write_json(run_dir / "environment.json", _environment(args))

    schema = EventSchema.from_file(args.schema)
    data_root = Path(args.data_root)
    train_docs_all = load_duee_documents(data_root / "train.jsonl", dataset=args.dataset)
    dev_docs_all = load_duee_documents(data_root / "dev.jsonl", dataset=args.dataset)
    if args.train_population == "multi_event_train":
        train_docs = multi_event_subset(train_docs_all)
        train_cache_name = "multi_event_train"
    else:
        train_docs = list(train_docs_all)
        train_cache_name = "all_train"
    dev_docs_multi = multi_event_subset(dev_docs_all)
    if args.limit_docs:
        train_docs = train_docs[: args.limit_docs]
        dev_docs_multi = dev_docs_multi[: args.limit_docs]
        dev_docs_all = dev_docs_all[: args.limit_docs]
    if args.smoke:
        train_docs = train_docs[: min(len(train_docs), 16)]
        dev_docs_multi = dev_docs_multi[: min(len(dev_docs_multi), 16)]
        dev_docs_all = dev_docs_all[: min(len(dev_docs_all), 16)]
    if not train_docs:
        raise RuntimeError("no R3 planner-only train documents were selected")

    device = torch.device("cuda" if torch.cuda.is_available() and args.model_path != "__toy__" else "cpu")
    encoder = build_encoder(args.model_path, device=device).to(device)
    encoder.eval()
    hidden_size = int(getattr(encoder, "hidden_size"))
    event_types = list(schema.event_roles)
    if not event_types:
        raise RuntimeError("schema has no event types")

    max_sentences = max(int(args.max_sentences), 1)
    cache_dir = run_dir / "cache"
    train_cache = _build_feature_cache(
        train_cache_name, train_docs, encoder, schema, hidden_size=hidden_size, max_sentences=max_sentences,
    )
    dev_multi_cache = _build_feature_cache(
        "multi_event_dev", dev_docs_multi, encoder, schema, hidden_size=hidden_size, max_sentences=max_sentences,
    )
    dev_all_cache = _build_feature_cache(
        "all_dev", dev_docs_all, encoder, schema, hidden_size=hidden_size, max_sentences=max_sentences,
    )
    for cache in (train_cache, dev_multi_cache, dev_all_cache):
        _write_feature_cache(cache_dir / f"{cache.name}.pt", cache)

    train_stats = _population_stats(train_cache)
    k_clip = int(train_stats["k_clip"])
    planner = RecordPlanner(hidden_size=hidden_size, num_event_types=len(event_types), k_max=k_clip).to(device)
    history = _train_two_stage_planner(planner, train_cache, args, device)

    metrics = {
        "multi_event_dev": _evaluate_two_stage_planner(planner, dev_multi_cache, k_clip, device, args),
        "all_dev": _evaluate_two_stage_planner(planner, dev_all_cache, k_clip, device, args),
    }
    legacy_model = _train_legacy_single_softmax_model(train_cache, schema, k_clip, args, device)
    baselines = {
        "multi_event_dev": _baseline_report(dev_multi_cache, k_clip, legacy_model, device),
        "all_dev": _baseline_report(dev_all_cache, k_clip, legacy_model, device),
    }
    acceptance_checks = _acceptance_checks(metrics, baselines, history)

    _write_json(run_dir / "diagnostics" / "r3_planner_train_history.json", history)
    _write_json(run_dir / "diagnostics" / "r3_planner_metrics.json", metrics)
    _write_json(run_dir / "diagnostics" / "r3_planner_baselines.json", baselines)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "planner": planner.state_dict(),
            "planner_metadata": {
                "event_types": event_types,
                "hidden_size": hidden_size,
                "k_clip": k_clip,
                "presence_pos_weight": train_stats["presence_pos_weight"],
                "presence_threshold_multi_event_dev": metrics["multi_event_dev"]["presence_threshold"],
                "presence_threshold_all_dev": metrics["all_dev"]["presence_threshold"],
                "acceptance_population": list(ACCEPTANCE_POPULATIONS),
                "train_population": train_cache_name,
                "encoder_feature_mode": args.encoder_feature_mode,
                "max_sentences": max_sentences,
            },
        },
        run_dir / "checkpoints" / "r3_planner.pt",
    )

    report = {
        "status": "r3_planner_only_smoke" if args.smoke else "r3_planner_only_diagnostic",
        "dataset": args.dataset,
        "acceptance_population": list(ACCEPTANCE_POPULATIONS),
        "diagnostic_populations": [],
        "encoder_feature_mode": args.encoder_feature_mode,
        "train_population": train_stats,
        "dev_populations": {
            "multi_event_dev": _population_stats(dev_multi_cache),
            "all_dev": _population_stats(dev_all_cache),
        },
        "history": history,
        "metrics": metrics,
        "baselines": baselines,
        "acceptance_checks": acceptance_checks,
        "accepted": all(check["passed"] for check in acceptance_checks.values()),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "non_goals": [
            "no P2 acceptance claim",
            "no P3 acceptance claim",
            "no P5b behavior change",
            "no hidden-test",
            "no paper main-table claim",
        ],
    }
    _write_json(run_dir / "summary.json", report)
    return report


def _build_feature_cache(
    name: str,
    documents: list[DueeDocument],
    encoder: nn.Module,
    schema: EventSchema,
    *,
    hidden_size: int,
    max_sentences: int,
) -> PlannerFeatureCache:
    event_types = list(schema.event_roles)
    global_rows: list[torch.Tensor] = []
    sentence_rows: list[torch.Tensor] = []
    count_rows: list[list[int]] = []
    lexical_rows: list[list[int]] = []
    document_ids: list[str] = []
    truncated = 0
    with torch.no_grad():
        for document in documents:
            text = _document_text(document)
            sentences = split_sentences(text)
            if sentences:
                encoded = encoder.encode_document(text, sentences)
                global_repr = encoded.global_repr.detach().cpu().to(dtype=torch.float32)
                sentence_repr = encoded.sentence_repr.detach().cpu().to(dtype=torch.float32)
            else:
                global_repr = torch.zeros((hidden_size,), dtype=torch.float32)
                sentence_repr = torch.zeros((0, hidden_size), dtype=torch.float32)
            if sentence_repr.shape[0] > max_sentences:
                sentence_repr = sentence_repr[:max_sentences]
                truncated += 1
            global_rows.append(global_repr)
            sentence_rows.append(sentence_repr)
            count_rows.append([RecordPlanner.gold_n_t(document.records, event_type) for event_type in event_types])
            lexical_rows.append([1 if _type_gate(document, event_type) else 0 for event_type in event_types])
            document_ids.append(document.document_id)

    n_docs = len(global_rows)
    if n_docs == 0:
        global_tensor = torch.zeros((0, hidden_size), dtype=torch.float32)
        sentence_tensor = torch.zeros((0, 0, hidden_size), dtype=torch.float32)
        mask_tensor = torch.zeros((0, 0), dtype=torch.bool)
        lexical_tensor = torch.zeros((0, len(event_types)), dtype=torch.float32)
        count_tensor = torch.zeros((0, len(event_types)), dtype=torch.long)
    else:
        n_sent_max = max(int(row.shape[0]) for row in sentence_rows)
        n_sent_max = max(n_sent_max, 1)
        global_tensor = torch.stack(global_rows)
        sentence_tensor = torch.zeros((n_docs, n_sent_max, hidden_size), dtype=torch.float32)
        mask_tensor = torch.zeros((n_docs, n_sent_max), dtype=torch.bool)
        for index, row in enumerate(sentence_rows):
            length = int(row.shape[0])
            if length:
                sentence_tensor[index, :length] = row
                mask_tensor[index, :length] = True
        lexical_tensor = torch.tensor(lexical_rows, dtype=torch.float32) if lexical_rows else torch.zeros((0, len(event_types)), dtype=torch.float32)
        count_tensor = torch.tensor(count_rows, dtype=torch.long) if count_rows else torch.zeros((0, len(event_types)), dtype=torch.long)
    return PlannerFeatureCache(
        name=name,
        document_ids=document_ids,
        documents=documents,
        global_repr=global_tensor,
        sentence_repr=sentence_tensor,
        sentence_mask=mask_tensor,
        lexical_hit=lexical_tensor,
        counts=count_tensor,
        event_types=event_types,
        truncated_sentence_documents=truncated,
        max_sentences=max_sentences,
    )


def _write_feature_cache(path: Path, cache: PlannerFeatureCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": cache.name,
            "document_ids": cache.document_ids,
            "global_repr": cache.global_repr,
            "sentence_repr": cache.sentence_repr,
            "sentence_mask": cache.sentence_mask,
            "lexical_hit": cache.lexical_hit,
            "counts": cache.counts,
            "event_types": cache.event_types,
            "documents": cache.documents_count,
            "event_type_pairs": cache.event_type_pairs,
            "max_sentences": cache.max_sentences,
            "truncated_sentence_documents": cache.truncated_sentence_documents,
        },
        path,
    )


def _train_two_stage_planner(
    planner: RecordPlanner,
    cache: PlannerFeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | int]]:
    doc_indices, type_ids, targets, lexical_hits = _pair_tensors(cache)
    if targets.numel() == 0:
        raise RuntimeError("no R3 planner-only train pairs were constructed")
    feature_mode = str(args.encoder_feature_mode)
    use_evidence = feature_mode in {"evidence", "evidence_lexical"}
    use_lexical = feature_mode == "evidence_lexical"
    batch_size = max(int(args.batch_size), 1)
    steps_per_epoch = max(math.ceil(targets.numel() / batch_size), 1)
    total_steps = max(int(args.max_epochs) * steps_per_epoch, 1)
    warmup_steps = max(int(total_steps * float(args.warmup_ratio)), 1)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=float(args.lr), weight_decay=1e-4)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    pos_weight = torch.tensor(
        float(_population_stats(cache)["presence_pos_weight"]),
        dtype=torch.float32,
        device=device,
    )
    global_repr_device = cache.global_repr.to(device)
    sentence_repr_device = cache.sentence_repr.to(device) if use_evidence else None
    sentence_mask_device = cache.sentence_mask.to(device) if use_evidence else None
    lexical_hit_device = lexical_hits.to(device) if use_lexical else None
    targets_device = targets.to(device)
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)
    history: list[dict[str, float | int]] = []
    global_step = 0
    for epoch in range(1, int(args.max_epochs) + 1):
        order = torch.randperm(targets.numel(), device=device)
        totals = {"loss": 0.0, "presence_loss": 0.0, "count_loss": 0.0, "pairs": 0.0}
        for batch_start in range(0, targets.numel(), batch_size):
            batch_ids = order[batch_start : batch_start + batch_size]
            doc_idx_batch = doc_indices_device[batch_ids]
            type_batch = type_ids_device[batch_ids]
            global_batch = global_repr_device[doc_idx_batch]
            target_batch = targets_device[batch_ids]
            sent_batch = sentence_repr_device[doc_idx_batch] if sentence_repr_device is not None else None
            mask_batch = sentence_mask_device[doc_idx_batch] if sentence_mask_device is not None else None
            lex_batch = lexical_hit_device[batch_ids].unsqueeze(-1) if lexical_hit_device is not None else None

            presence_target = (target_batch > 0).to(dtype=torch.float32)
            presence_logits = planner.presence_logit(
                global_batch,
                type_batch,
                sentence_repr=sent_batch,
                sentence_mask=mask_batch,
                lexical_hit=lex_batch,
            )
            presence_loss_value = presence_loss(presence_logits, presence_target, pos_weight=pos_weight)
            positive_mask = target_batch > 0
            if bool(positive_mask.any().item()):
                pos_sent = sent_batch[positive_mask] if sent_batch is not None else None
                pos_mask = mask_batch[positive_mask] if mask_batch is not None else None
                pos_lex = lex_batch[positive_mask] if lex_batch is not None else None
                count_log_lambda = planner.count_log_lambda(
                    global_batch[positive_mask],
                    type_batch[positive_mask],
                    sentence_repr=pos_sent,
                    sentence_mask=pos_mask,
                    lexical_hit=pos_lex,
                )
                target_pos = target_batch[positive_mask].to(dtype=torch.float32)
                sample_weights = torch.log(target_pos.clamp_min(1.0)) + 1.0
                count_loss_value = truncated_poisson_nll(
                    count_log_lambda,
                    target_pos,
                    sample_weights=sample_weights,
                )
            else:
                count_loss_value = presence_logits.sum() * 0.0
            loss = float(args.lambda_presence) * presence_loss_value + float(args.lambda_count) * count_loss_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=float(args.grad_clip))
            scale = _lr_scale(global_step, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            optimizer.step()
            global_step += 1
            batch_pairs = float(batch_ids.numel())
            totals["loss"] += float(loss.detach().cpu()) * batch_pairs
            totals["presence_loss"] += float(presence_loss_value.detach().cpu()) * batch_pairs
            totals["count_loss"] += float(count_loss_value.detach().cpu()) * batch_pairs
            totals["pairs"] += batch_pairs
        denom = max(totals["pairs"], 1.0)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / denom,
            "presence_loss": totals["presence_loss"] / denom,
            "count_loss": totals["count_loss"] / denom,
            "event_type_pairs": int(totals["pairs"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return history


def _evaluate_two_stage_planner(
    planner: RecordPlanner,
    cache: PlannerFeatureCache,
    k_clip: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    doc_indices, type_ids, targets, lexical_hits = _pair_tensors(cache)
    if targets.numel() == 0:
        return _empty_metrics(cache.name, k_clip)
    feature_mode = str(args.encoder_feature_mode)
    use_evidence = feature_mode in {"evidence", "evidence_lexical"}
    use_lexical = feature_mode == "evidence_lexical"
    eval_batch_size = max(int(getattr(args, "eval_batch_size", 0) or args.batch_size), 1)
    global_repr_device = cache.global_repr.to(device)
    sentence_repr_device = cache.sentence_repr.to(device) if use_evidence else None
    sentence_mask_device = cache.sentence_mask.to(device) if use_evidence else None
    lexical_hit_device = lexical_hits.to(device) if use_lexical else None
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)
    presence_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    total = int(targets.numel())
    with torch.no_grad():
        for start in range(0, total, eval_batch_size):
            stop = min(start + eval_batch_size, total)
            doc_idx_batch = doc_indices_device[start:stop]
            type_batch = type_ids_device[start:stop]
            global_batch = global_repr_device[doc_idx_batch]
            sent_batch = sentence_repr_device[doc_idx_batch] if sentence_repr_device is not None else None
            mask_batch = sentence_mask_device[doc_idx_batch] if sentence_mask_device is not None else None
            lex_batch = lexical_hit_device[start:stop].unsqueeze(-1) if lexical_hit_device is not None else None
            presence_probs = torch.sigmoid(
                planner.presence_logit(
                    global_batch,
                    type_batch,
                    sentence_repr=sent_batch,
                    sentence_mask=mask_batch,
                    lexical_hit=lex_batch,
                )
            )
            count_preds = truncated_poisson_argmax(
                planner.count_log_lambda(
                    global_batch,
                    type_batch,
                    sentence_repr=sent_batch,
                    sentence_mask=mask_batch,
                    lexical_hit=lex_batch,
                ),
                k_clip=k_clip,
            )
            presence_chunks.append(presence_probs.detach().cpu())
            count_chunks.append(count_preds.detach().cpu())
    presence_scores = torch.cat(presence_chunks) if presence_chunks else torch.zeros((0,))
    count_predictions = torch.cat(count_chunks) if count_chunks else torch.zeros((0,), dtype=torch.long)
    return _prediction_metrics(
        population=cache.name,
        documents=cache.documents_count,
        targets=targets,
        presence_scores=[float(score) for score in presence_scores.tolist()],
        count_predictions=[int(value) for value in count_predictions.tolist()],
        k_clip=k_clip,
    )


def _baseline_report(
    cache: PlannerFeatureCache,
    k_clip: int,
    legacy_model: nn.Module,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    return {
        "predict_one": _predict_one_baseline(cache),
        "p5b_lexical_trigger": _p5b_lexical_trigger_baseline(cache, k_clip),
        "legacy_single_softmax": _legacy_single_softmax_baseline(legacy_model, cache, k_clip, device),
    }


def _predict_one_baseline(cache: PlannerFeatureCache) -> dict[str, float | int | str]:
    positive = cache.counts[cache.counts > 0]
    if positive.numel() == 0:
        mae = 0.0
    else:
        mae = float(torch.abs(positive.to(dtype=torch.float32) - 1.0).mean().item())
    return {
        "population": cache.name,
        "positive_gold_event_types": int(positive.numel()),
        "count_mae_positive": round(mae, 6),
        "description": "positive-only count baseline: always predict n_t=1 for gold-positive event types",
    }


def _p5b_lexical_trigger_baseline(cache: PlannerFeatureCache, k_clip: int) -> dict[str, float | int | str]:
    scores: list[float] = []
    predictions: list[int] = []
    for document in cache.documents:
        for event_type in cache.event_types:
            present = _type_gate(document, event_type)
            scores.append(1.0 if present else 0.0)
            predictions.append(_estimate_record_count(document, event_type) if present else 0)
    return _prediction_metrics(
        population=cache.name,
        documents=cache.documents_count,
        targets=cache.counts.reshape(-1),
        presence_scores=scores,
        count_predictions=predictions,
        k_clip=k_clip,
    )


def _train_legacy_single_softmax_model(
    cache: PlannerFeatureCache,
    schema: EventSchema,
    k_clip: int,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    model = _LegacySoftmaxPlanner(
        hidden_size=int(cache.global_repr.shape[1]) if cache.global_repr.numel() else 1,
        num_event_types=len(schema.event_roles),
        k_clip=k_clip,
    ).to(device)
    _train_legacy_single_softmax(model, cache, args, device)
    return model


def _legacy_single_softmax_baseline(
    model: nn.Module,
    cache: PlannerFeatureCache,
    k_clip: int,
    device: torch.device,
) -> dict[str, Any]:
    if cache.event_type_pairs == 0:
        metrics = _empty_metrics(cache.name, k_clip)
        metrics["diagnostic_only"] = True
        return metrics
    metrics = _evaluate_legacy_single_softmax(model, cache, k_clip, device)
    metrics["diagnostic_only"] = True
    metrics["trained_on_population"] = "train_cache_population"
    return metrics


def _train_legacy_single_softmax(
    model: nn.Module,
    cache: PlannerFeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    doc_indices, type_ids, targets, _ = _pair_tensors(cache)
    if targets.numel() == 0:
        return
    batch_size = max(int(args.batch_size), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    features = cache.global_repr.to(device)
    targets_device = targets.to(device)
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)
    for _epoch in range(1, int(args.max_epochs) + 1):
        order = torch.randperm(targets.numel(), device=device)
        for batch_start in range(0, targets.numel(), batch_size):
            batch_ids = order[batch_start : batch_start + batch_size]
            logits = model(features[doc_indices_device[batch_ids]], type_ids_device[batch_ids])
            loss = nn.functional.cross_entropy(logits, targets_device[batch_ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def _evaluate_legacy_single_softmax(
    model: nn.Module,
    cache: PlannerFeatureCache,
    k_clip: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    doc_indices, type_ids, targets, _ = _pair_tensors(cache)
    features = cache.global_repr.to(device)
    with torch.no_grad():
        logits = model(features[doc_indices.to(device)], type_ids.to(device))
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        presence_scores = (1.0 - probs[:, 0]).tolist()
        predictions = torch.argmax(probs, dim=-1).tolist()
    return _prediction_metrics(
        population=cache.name,
        documents=cache.documents_count,
        targets=targets,
        presence_scores=[float(score) for score in presence_scores],
        count_predictions=[int(value) for value in predictions],
        k_clip=k_clip,
    )


def _prediction_metrics(
    *,
    population: str,
    documents: int,
    targets: torch.Tensor,
    presence_scores: list[float],
    count_predictions: list[int],
    k_clip: int,
) -> dict[str, float | int | str]:
    labels = [1 if int(target) > 0 else 0 for target in targets.tolist()]
    gate_metrics = _presence_threshold_metrics(labels, presence_scores)
    threshold = float(gate_metrics["threshold"])
    total_abs = 0.0
    positive_abs = 0.0
    positive_count = 0
    positive_recalled = 0
    truncated = 0
    tp = fp = fn = 0
    for gold_value, score, count_pred in zip(targets.tolist(), presence_scores, count_predictions):
        gold = int(gold_value)
        if gold > k_clip:
            truncated += 1
        present = score >= threshold
        pred = int(count_pred) if present else 0
        total_abs += abs(pred - gold)
        if gold > 0:
            positive_count += 1
            positive_abs += abs(int(count_pred) - gold)
            if pred > 0:
                positive_recalled += 1
            if present:
                tp += 1
            else:
                fn += 1
        elif present:
            fp += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    event_type_pairs = int(targets.numel())
    return {
        "population": population,
        "documents": documents,
        "evaluated_event_types": event_type_pairs,
        "positive_gold_event_types": positive_count,
        "zero_rate": round((event_type_pairs - positive_count) / max(event_type_pairs, 1), 6),
        "planner_mae": round(total_abs / max(event_type_pairs, 1), 6),
        "type_gate_recall": round(positive_recalled / max(positive_count, 1), 6),
        "type_gate_auc": round(_binary_auc(labels, presence_scores), 6),
        "type_gate_f1_youden": round(float(gate_metrics["f1"]), 6),
        "type_gate_precision_youden": round(float(gate_metrics["precision"]), 6),
        "type_gate_recall_youden": round(float(gate_metrics["recall"]), 6),
        "presence_threshold": round(threshold, 6),
        "count_mae_positive": round(positive_abs / max(positive_count, 1), 6),
        "truncated_gold_event_types": truncated,
        "truncation_rate": round(truncated / max(event_type_pairs, 1), 6),
        "k_clip": k_clip,
        "type_gate_f1_at_threshold": round(f1, 6),
    }


def _empty_metrics(population: str, k_clip: int) -> dict[str, float | int | str]:
    return {
        "population": population,
        "documents": 0,
        "evaluated_event_types": 0,
        "positive_gold_event_types": 0,
        "zero_rate": 0.0,
        "planner_mae": 0.0,
        "type_gate_recall": 0.0,
        "type_gate_auc": 0.5,
        "type_gate_f1_youden": 0.0,
        "type_gate_precision_youden": 0.0,
        "type_gate_recall_youden": 0.0,
        "presence_threshold": 0.5,
        "count_mae_positive": 0.0,
        "truncated_gold_event_types": 0,
        "truncation_rate": 0.0,
        "k_clip": k_clip,
        "type_gate_f1_at_threshold": 0.0,
    }


def _population_stats(cache: PlannerFeatureCache) -> dict[str, float | int | str]:
    positive = int((cache.counts > 0).sum().item())
    total = cache.event_type_pairs
    negative = total - positive
    max_n = int(cache.counts.max().item()) if cache.counts.numel() else 0
    records_per_doc = [len(document.records) for document in cache.documents]
    return {
        "name": cache.name,
        "documents": cache.documents_count,
        "records_mean": round(sum(records_per_doc) / max(len(records_per_doc), 1), 6),
        "records_max": max(records_per_doc) if records_per_doc else 0,
        "event_type_pairs": total,
        "presence_positive": positive,
        "presence_negative": negative,
        "presence_pos_frac": round(positive / max(total, 1), 6),
        "presence_pos_weight": round(negative / max(positive, 1), 6) if positive else 1.0,
        "zero_rate": round(negative / max(total, 1), 6),
        "max_n_t": max_n,
        "k_clip": max(max_n, 1),
        "max_sentences": cache.max_sentences,
        "sentence_truncation_rate": cache.truncation_rate,
    }


def _pair_tensors(cache: PlannerFeatureCache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_docs, num_types = cache.counts.shape if cache.counts.numel() else (0, len(cache.event_types))
    if num_docs == 0:
        return (
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
        )
    doc_indices = torch.arange(num_docs, dtype=torch.long).repeat_interleave(num_types)
    type_ids = torch.arange(num_types, dtype=torch.long).repeat(num_docs)
    counts_flat = cache.counts.reshape(-1)
    lexical_flat = cache.lexical_hit.to(dtype=torch.float32).reshape(-1)
    return doc_indices, type_ids, counts_flat, lexical_flat


def _acceptance_checks(
    metrics_by_population: dict[str, dict[str, Any]],
    baselines_by_population: dict[str, dict[str, Any]],
    history: list[dict[str, float | int]],
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    metric_specs = (
        ("type_gate_auc", "absolute_ge", 0.80, 0.05, "max"),
        ("type_gate_f1_youden", "absolute_ge", 0.55, 0.05, "max"),
        ("count_mae_positive", "absolute_le", 0.5, 0.05, "min"),
    )
    for population in ACCEPTANCE_POPULATIONS:
        metrics = metrics_by_population.get(population, {})
        baselines = baselines_by_population.get(population, {})
        for metric_name, abs_op, abs_threshold, rel_margin, aggregator in metric_specs:
            value = float(metrics.get(metric_name, 0.0))
            best_baseline, contributing = _best_baseline_for(metric_name, baselines, aggregator)
            if abs_op == "absolute_ge":
                abs_pass = value >= float(abs_threshold)
                baseline_threshold = best_baseline + float(rel_margin)
                rel_pass = value >= baseline_threshold
                threshold_repr = f">= {float(abs_threshold):.6f}"
                baseline_threshold_repr = f">= {baseline_threshold:.6f}"
            else:
                abs_pass = value <= float(abs_threshold)
                baseline_threshold = best_baseline - float(rel_margin)
                rel_pass = value <= baseline_threshold
                threshold_repr = f"<= {float(abs_threshold):.6f}"
                baseline_threshold_repr = f"<= {baseline_threshold:.6f}"
            checks[f"{population}/{metric_name}"] = {
                "value": round(value, 6),
                "absolute_threshold": threshold_repr,
                "best_baseline": round(best_baseline, 6),
                "best_baseline_sources": contributing,
                "baseline_relative_threshold": baseline_threshold_repr,
                "baseline_relative_margin": float(rel_margin),
                "passed_absolute": bool(abs_pass),
                "passed_baseline_relative": bool(rel_pass),
                "passed": bool(abs_pass and rel_pass),
            }
    checks["training/presence_loss_trend"] = {
        "value": _trend_summary(history, "presence_loss"),
        "threshold": "first/last ratio >= 2.0 over at least 10 epochs",
        "passed": _has_required_overall_decrease(history, "presence_loss"),
    }
    checks["training/count_loss_trend"] = {
        "value": _trend_summary(history, "count_loss"),
        "threshold": "first/last ratio >= 2.0 over at least 10 epochs",
        "passed": _has_required_overall_decrease(history, "count_loss"),
    }
    return checks


def _best_baseline_for(
    metric_name: str,
    baselines: dict[str, Any],
    aggregator: str,
) -> tuple[float, dict[str, float]]:
    candidate_sources = ("predict_one", "p5b_lexical_trigger", "legacy_single_softmax")
    contributing: dict[str, float] = {}
    for key in candidate_sources:
        source = baselines.get(key, {})
        if metric_name in source:
            try:
                contributing[key] = float(source[metric_name])
            except (TypeError, ValueError):
                continue
    if not contributing:
        return (0.0 if aggregator == "max" else float("inf"), {})
    if aggregator == "min":
        best = min(contributing.values())
    else:
        best = max(contributing.values())
    return best, {key: round(value, 6) for key, value in contributing.items()}


def _has_required_overall_decrease(
    history: list[dict[str, float | int]],
    key: str,
    *,
    min_ratio: float = 2.0,
    min_epochs: int = 10,
) -> bool:
    values = [float(row[key]) for row in history if key in row]
    if len(values) < min_epochs:
        return False
    first = values[0]
    last = values[-1]
    if first <= 0 or last <= 0:
        return False
    return (first / last) >= min_ratio


def _trend_summary(history: list[dict[str, float | int]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in history if key in row]
    if not values:
        return {"epochs": 0, "first": None, "last": None, "decrease_ratio": None}
    ratio = values[0] / values[-1] if values[-1] != 0 else None
    return {
        "epochs": len(values),
        "first": round(values[0], 6),
        "last": round(values[-1], 6),
        "decrease_ratio": round(ratio, 6) if ratio is not None else None,
    }


def _lr_scale(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0


class _LegacySoftmaxPlanner(nn.Module):
    def __init__(self, *, hidden_size: int, num_event_types: int, k_clip: int) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(max(num_event_types, 1), hidden_size)
        self.proj = nn.Linear(hidden_size * 2, k_clip + 1)

    def forward(self, global_repr: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        if type_id.dim() == 0:
            type_id = type_id.unsqueeze(0)
        if global_repr.dim() == 1:
            global_repr = global_repr.unsqueeze(0)
        if global_repr.shape[0] == 1 and type_id.shape[0] > 1:
            global_repr = global_repr.expand(type_id.shape[0], -1)
        type_emb = self.type_embedding(type_id.to(device=self.type_embedding.weight.device, dtype=torch.long))
        return self.proj(torch.cat([global_repr.to(type_emb.device), type_emb], dim=-1))
