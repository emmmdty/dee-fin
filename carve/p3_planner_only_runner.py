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
from carve.p3_mention_crf import MentionCRF
from carve.p3_planner import (
    ArgumentCoreferenceHead,
    RecordPlanner,
    SentenceLevelCountPlanner,
    coref_pair_loss,
    predict_clusters,
    presence_loss,
    sentence_count_loss,
    truncated_poisson_argmax,
    truncated_poisson_nll,
)
from carve.p3_runner import _binary_auc, _presence_threshold_metrics
from carve.p3_apcc_v4 import (
    CANDIDATE_THRESHOLD_GRID,
    DEFAULT_MAX_MENTIONS,
    STATIC_REFERENCE_BASELINES,
    MentionSpan,
    bcubed_f1,
    build_coref_pair_labels,
    build_gold_partition,
    extract_gold_mention_spans,
    extract_predicted_mentions,
    load_p3_mention_crf,
    pad_mentions_to_tensors,
    pair_pos_weight,
)
from carve.p5b_runner import _estimate_record_count, _type_gate
from carve.text_segmentation import split_sentences
from evaluator.canonical.normalize import normalize_optional_text, normalize_text
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
    sentence_record_label: torch.Tensor | None  # [N, S_max, num_event_types] uint8; None for document mode
    event_types: list[str]
    truncated_sentence_documents: int
    max_sentences: int
    # v4 (coref) fields; None for document/sentence modes
    span_repr: torch.Tensor | None = None      # [N, M_max, H]
    span_sent_idx: torch.Tensor | None = None  # [N, M_max] long
    span_role_id: torch.Tensor | None = None   # [N, M_max] long
    span_mask: torch.Tensor | None = None      # [N, M_max] bool
    span_normalized_values: list[list[str]] | None = None  # [N][M]
    max_mentions: int = 0

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
    parser.add_argument(
        "--count-head-mode",
        choices=("document", "sentence", "coref"),
        default="document",
        help="document (v2.1, default), sentence (v3), coref (v4 APCC)",
    )
    parser.add_argument(
        "--sentence-label-min-hits",
        type=int,
        default=1,
        help="minimum argument hits per sentence for a positive sentence label (v3 only)",
    )
    parser.add_argument(
        "--sentence-bce-pos-weight-cap",
        type=float,
        default=20.0,
        help="cap on BCE pos_weight for sentence count loss (v3 only)",
    )
    parser.add_argument(
        "--mention-source",
        choices=("crf", "gold"),
        default="crf",
        help="v4 only: 'crf' loads --p3-crf-checkpoint and runs CRF (acceptance path); "
             "'gold' uses gold-record argument values (oracle, ablation only)",
    )
    parser.add_argument(
        "--p3-crf-checkpoint",
        type=str,
        default="",
        help="v4 only: path to a trained P3 checkpoint containing 'mention_crf' state dict",
    )
    parser.add_argument(
        "--max-mentions",
        type=int,
        default=DEFAULT_MAX_MENTIONS,
        help="v4 only: max candidate mentions per document",
    )
    parser.add_argument(
        "--coref-pos-weight-cap",
        type=float,
        default=20.0,
        help="v4 only: cap on BCE pos_weight for coref pair loss",
    )
    parser.add_argument(
        "--coref-threshold-grid",
        type=str,
        default=",".join(f"{t:.2f}" for t in CANDIDATE_THRESHOLD_GRID),
        help="v4 only: comma-separated thresholds searched on train for clustering cutoff",
    )
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
    count_head_mode = str(args.count_head_mode)
    sentence_min_hits = max(int(args.sentence_label_min_hits), 1)
    mention_source = str(getattr(args, "mention_source", "crf"))
    max_mentions = max(int(getattr(args, "max_mentions", DEFAULT_MAX_MENTIONS)), 1)

    mention_crf: MentionCRF | None = None
    if count_head_mode == "coref" and mention_source == "crf":
        ckpt_path = str(getattr(args, "p3_crf_checkpoint", ""))
        if not ckpt_path:
            raise RuntimeError(
                "--p3-crf-checkpoint is required when --count-head-mode coref --mention-source crf"
            )
        mention_crf = load_p3_mention_crf(ckpt_path, hidden_size).to(device)
        mention_crf.eval()

    cache_dir = run_dir / "cache"
    train_cache = _build_feature_cache(
        train_cache_name, train_docs, encoder, schema, hidden_size=hidden_size, max_sentences=max_sentences,
        count_head_mode=count_head_mode, sentence_label_min_hits=sentence_min_hits,
        mention_crf=mention_crf, mention_source=mention_source, max_mentions=max_mentions,
    )
    dev_multi_cache = _build_feature_cache(
        "multi_event_dev", dev_docs_multi, encoder, schema, hidden_size=hidden_size, max_sentences=max_sentences,
        count_head_mode=count_head_mode, sentence_label_min_hits=sentence_min_hits,
        mention_crf=mention_crf, mention_source=mention_source, max_mentions=max_mentions,
    )
    dev_all_cache = _build_feature_cache(
        "all_dev", dev_docs_all, encoder, schema, hidden_size=hidden_size, max_sentences=max_sentences,
        count_head_mode=count_head_mode, sentence_label_min_hits=sentence_min_hits,
        mention_crf=mention_crf, mention_source=mention_source, max_mentions=max_mentions,
    )
    for cache in (train_cache, dev_multi_cache, dev_all_cache):
        _write_feature_cache(cache_dir / f"{cache.name}.pt", cache)

    train_stats = _population_stats(train_cache)
    k_clip = int(train_stats["k_clip"])
    planner = RecordPlanner(
        hidden_size=hidden_size,
        num_event_types=len(event_types),
        k_max=k_clip,
        count_head_mode=count_head_mode,
        num_roles=0,
        max_sentence_pos=max_sentences,
    ).to(device)
    if count_head_mode == "coref":
        history = _train_coref_planner(planner, train_cache, args, device)
    else:
        history = _train_two_stage_planner(planner, train_cache, args, device)

    noise_diagnostics: dict[str, dict[str, Any]] = {}
    if count_head_mode == "sentence":
        for cache_name, cache_ref in (
            ("train", train_cache),
            ("multi_event_dev", dev_multi_cache),
            ("all_dev", dev_all_cache),
        ):
            noise_diagnostics[cache_name] = _sentence_label_diagnostics(cache_ref)

    coref_threshold = 0.5
    coref_threshold_grid: dict[str, float] = {}
    coref_diagnostics: dict[str, dict[str, Any]] = {}
    if count_head_mode == "coref":
        grid_str = str(getattr(args, "coref_threshold_grid", "")).strip()
        grid: tuple[float, ...]
        if grid_str:
            grid = tuple(float(x) for x in grid_str.split(",") if x.strip())
        else:
            grid = CANDIDATE_THRESHOLD_GRID
        coref_threshold, coref_threshold_grid = _tune_coref_threshold(planner, train_cache, grid, device)
        metrics = {
            "multi_event_dev": _evaluate_coref_planner(planner, dev_multi_cache, device, args, coref_threshold),
            "all_dev": _evaluate_coref_planner(planner, dev_all_cache, device, args, coref_threshold),
        }
        for cache_name, cache_ref in (
            ("train", train_cache),
            ("multi_event_dev", dev_multi_cache),
            ("all_dev", dev_all_cache),
        ):
            coref_diagnostics[cache_name] = _coref_pair_diagnostics(cache_ref)
    else:
        metrics = {
            "multi_event_dev": _evaluate_two_stage_planner(planner, dev_multi_cache, k_clip, device, args),
            "all_dev": _evaluate_two_stage_planner(planner, dev_all_cache, k_clip, device, args),
        }
    legacy_model = _train_legacy_single_softmax_model(train_cache, schema, k_clip, args, device)
    baselines = {
        "multi_event_dev": _baseline_report(dev_multi_cache, k_clip, legacy_model, device),
        "all_dev": _baseline_report(dev_all_cache, k_clip, legacy_model, device),
    }
    if count_head_mode == "coref":
        for population in ACCEPTANCE_POPULATIONS:
            baselines[population]["v2_1_poisson_static"] = {
                "count_mae_positive": STATIC_REFERENCE_BASELINES["v2_1_poisson"][population]["count_mae_positive"],
                "diagnostic_only": True,
                "source": "docs/measurements/r3_planner_only_duee_fin_seed42_v2_1.md",
            }
            baselines[population]["v3_sentence_static"] = {
                "count_mae_positive": STATIC_REFERENCE_BASELINES["v3_sentence"][population]["count_mae_positive"],
                "diagnostic_only": True,
                "source": "docs/measurements/r3_planner_only_duee_fin_seed42_v3.md",
            }
    acceptance_checks = _acceptance_checks(metrics, baselines, history, count_head_mode=count_head_mode)

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
                "count_head_mode": count_head_mode,
                "sentence_label_min_hits": sentence_min_hits if count_head_mode == "sentence" else None,
                "coref_threshold": coref_threshold if count_head_mode == "coref" else None,
                "mention_source": mention_source if count_head_mode == "coref" else None,
                "max_mentions": max_mentions if count_head_mode == "coref" else None,
                "p3_crf_checkpoint": str(getattr(args, "p3_crf_checkpoint", "")) if count_head_mode == "coref" else None,
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
        "count_head_mode": count_head_mode,
        "noise_diagnostics": noise_diagnostics,
        "coref_threshold": coref_threshold if count_head_mode == "coref" else None,
        "coref_threshold_grid_train_mae": coref_threshold_grid if count_head_mode == "coref" else {},
        "coref_diagnostics": coref_diagnostics,
        "mention_source": mention_source if count_head_mode == "coref" else None,
        "max_mentions": max_mentions if count_head_mode == "coref" else 0,
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
    count_head_mode: str = "document",
    sentence_label_min_hits: int = 1,
    mention_crf: "MentionCRF | None" = None,
    mention_source: str = "crf",
    max_mentions: int = DEFAULT_MAX_MENTIONS,
) -> PlannerFeatureCache:
    event_types = list(schema.event_roles)
    build_sentence_labels = count_head_mode == "sentence"
    build_coref_spans = count_head_mode == "coref"
    global_rows: list[torch.Tensor] = []
    sentence_rows: list[torch.Tensor] = []
    count_rows: list[list[int]] = []
    lexical_rows: list[list[int]] = []
    sentence_label_rows: list[list[list[int]]] = []  # [doc][sentence][type]
    per_doc_spans: list[list[MentionSpan]] = []
    document_ids: list[str] = []
    truncated = 0
    if build_coref_spans and mention_source == "crf" and mention_crf is None:
        raise RuntimeError(
            "count_head_mode='coref' with mention_source='crf' requires a loaded MentionCRF; "
            "pass --p3-crf-checkpoint to load one."
        )
    with torch.no_grad():
        for document in documents:
            text = _document_text(document)
            sentences = split_sentences(text)
            if sentences:
                encoded = encoder.encode_document(text, sentences)
                global_repr = encoded.global_repr.detach().cpu().to(dtype=torch.float32)
                sentence_repr = encoded.sentence_repr.detach().cpu().to(dtype=torch.float32)
            else:
                encoded = None
                global_repr = torch.zeros((hidden_size,), dtype=torch.float32)
                sentence_repr = torch.zeros((0, hidden_size), dtype=torch.float32)
            if sentence_repr.shape[0] > max_sentences:
                sentence_repr = sentence_repr[:max_sentences]
                truncated += 1
            n_sent = int(sentence_repr.shape[0])
            global_rows.append(global_repr)
            sentence_rows.append(sentence_repr)
            count_rows.append([RecordPlanner.gold_n_t(document.records, event_type) for event_type in event_types])
            lexical_rows.append([1 if _type_gate(document, event_type) else 0 for event_type in event_types])
            if build_sentence_labels:
                doc_labels = _build_sentence_labels(
                    document, sentences[:n_sent], event_types, min_hits=sentence_label_min_hits,
                )
                sentence_label_rows.append(doc_labels)
            if build_coref_spans:
                if encoded is None or not sentences:
                    per_doc_spans.append([])
                else:
                    sents_used = sentences[:n_sent]
                    if mention_source == "crf":
                        assert mention_crf is not None
                        spans = extract_predicted_mentions(
                            encoded, sents_used, mention_crf,
                            hidden_size=hidden_size, max_mentions=max_mentions,
                        )
                    else:
                        spans = extract_gold_mention_spans(
                            encoded, sents_used, document, schema,
                            hidden_size=hidden_size, max_mentions=max_mentions,
                        )
                    # clip sent_idx to be safe under sentence truncation
                    spans = [s for s in spans if s.sent_idx < n_sent]
                    per_doc_spans.append(spans)
            document_ids.append(document.document_id)

    n_docs = len(global_rows)
    if n_docs == 0:
        global_tensor = torch.zeros((0, hidden_size), dtype=torch.float32)
        sentence_tensor = torch.zeros((0, 0, hidden_size), dtype=torch.float32)
        mask_tensor = torch.zeros((0, 0), dtype=torch.bool)
        lexical_tensor = torch.zeros((0, len(event_types)), dtype=torch.float32)
        count_tensor = torch.zeros((0, len(event_types)), dtype=torch.long)
        sentence_label_tensor = None
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
        if build_sentence_labels:
            sentence_label_tensor = torch.zeros(
                (n_docs, n_sent_max, len(event_types)), dtype=torch.uint8,
            )
            for doc_idx, doc_labels in enumerate(sentence_label_rows):
                n = min(len(doc_labels), n_sent_max)
                if n:
                    sentence_label_tensor[doc_idx, :n] = torch.tensor(doc_labels[:n], dtype=torch.uint8)
        else:
            sentence_label_tensor = None

    span_repr_tensor: torch.Tensor | None = None
    span_sent_idx_tensor: torch.Tensor | None = None
    span_role_id_tensor: torch.Tensor | None = None
    span_mask_tensor: torch.Tensor | None = None
    span_values: list[list[str]] | None = None
    effective_max_mentions = 0
    if build_coref_spans and n_docs > 0:
        observed_max = max((len(s) for s in per_doc_spans), default=0)
        effective_max_mentions = max(min(observed_max, max_mentions), 1)
        packed = pad_mentions_to_tensors(
            per_doc_spans, hidden_size=hidden_size, max_mentions=effective_max_mentions,
        )
        span_repr_tensor = packed["span_repr"]
        span_sent_idx_tensor = packed["span_sent_idx"]
        span_role_id_tensor = packed["span_role_id"]
        span_mask_tensor = packed["span_mask"]
        span_values = [
            [s.normalized_value for s in per_doc_spans[d][:effective_max_mentions]]
            for d in range(n_docs)
        ]

    return PlannerFeatureCache(
        name=name,
        document_ids=document_ids,
        documents=documents,
        global_repr=global_tensor,
        sentence_repr=sentence_tensor,
        sentence_mask=mask_tensor,
        lexical_hit=lexical_tensor,
        counts=count_tensor,
        sentence_record_label=sentence_label_tensor,
        event_types=event_types,
        truncated_sentence_documents=truncated,
        max_sentences=max_sentences,
        span_repr=span_repr_tensor,
        span_sent_idx=span_sent_idx_tensor,
        span_role_id=span_role_id_tensor,
        span_mask=span_mask_tensor,
        span_normalized_values=span_values,
        max_mentions=effective_max_mentions,
    )


def _build_sentence_labels(
    document: DueeDocument,
    sentences: Any,  # list[Sentence] — duck-typed .text
    event_types: list[str],
    *,
    min_hits: int = 1,
) -> list[list[int]]:
    """Derive per-(sentence, event_type) binary labels from gold record arguments.

    For each sentence and event type, check whether any gold record of that type
    has at least `min_hits` argument values whose normalised form appears in the
    normalised sentence text.  Returns list-of-lists indexed [sentence][event_type].
    """
    labels: list[list[int]] = []
    s_texts = [normalize_text(s.text) for s in sentences]
    type_arg_sets: list[list[set[str]]] = []
    for event_type in event_types:
        nt = normalize_text(event_type)
        arg_sets: list[set[str]] = []
        for record in document.records:
            if normalize_text(record.event_type) != nt:
                continue
            arg_values: set[str] = set()
            for role_values in record.arguments.values():
                for value in role_values:
                    nv = normalize_optional_text(value)
                    if nv:
                        arg_values.add(nv)
            if arg_values:
                arg_sets.append(arg_values)
        type_arg_sets.append(arg_sets)
    for s_text in s_texts:
        sent_labels: list[int] = []
        for t_idx in range(len(event_types)):
            hits = 0
            for arg_set in type_arg_sets[t_idx]:
                if any(arg_value in s_text for arg_value in arg_set):
                    hits += 1
            sent_labels.append(1 if hits >= min_hits else 0)
        labels.append(sent_labels)
    return labels


def _write_feature_cache(path: Path, cache: PlannerFeatureCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
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
    }
    if cache.sentence_record_label is not None:
        payload["sentence_record_label"] = cache.sentence_record_label
    torch.save(payload, path)


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
    count_head_mode = str(planner.count_head_mode)
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
    sentence_repr_device = cache.sentence_repr.to(device) if (use_evidence or count_head_mode == "sentence") else None
    sentence_mask_device = cache.sentence_mask.to(device) if (use_evidence or count_head_mode == "sentence") else None
    lexical_hit_device = lexical_hits.to(device) if use_lexical else None
    targets_device = targets.to(device)
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)
    sentence_label_device: torch.Tensor | None = None
    bce_pos_weight = torch.ones((), dtype=torch.float32, device=device)
    if count_head_mode == "sentence":
        if cache.sentence_record_label is None:
            raise RuntimeError("sentence_record_label is None but count_head_mode='sentence'")
        sentence_label_device = cache.sentence_record_label.to(device)
        bce_pos_weight_value = _sentence_bce_pos_weight(cache, cap=float(args.sentence_bce_pos_weight_cap))
        bce_pos_weight = torch.tensor(bce_pos_weight_value, dtype=torch.float32, device=device)
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
                if count_head_mode == "sentence":
                    if sent_batch is None or mask_batch is None:
                        raise RuntimeError("sentence_repr required for sentence count head")
                    assert sentence_label_device is not None
                    pos_doc_idx = doc_idx_batch[positive_mask]
                    pos_type = type_batch[positive_mask]
                    pos_sent_labels = sentence_label_device[pos_doc_idx, :, pos_type]    # [B_pos, S_max]
                    pos_mask_batch = mask_batch[positive_mask]                            # [B_pos, S_max]
                    pos_sent_logits = planner.sentence_count_logits(pos_type, sent_batch[positive_mask])  # [B_pos, S_max]
                    count_loss_value = sentence_count_loss(
                        pos_sent_logits,
                        pos_sent_labels,
                        pos_mask_batch,
                        pos_weight=bce_pos_weight,
                    )
                else:
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
    count_head_mode = str(planner.count_head_mode)
    eval_batch_size = max(int(getattr(args, "eval_batch_size", 0) or args.batch_size), 1)
    global_repr_device = cache.global_repr.to(device)
    sentence_repr_device = cache.sentence_repr.to(device) if (use_evidence or count_head_mode == "sentence") else None
    sentence_mask_device = cache.sentence_mask.to(device) if (use_evidence or count_head_mode == "sentence") else None
    lexical_hit_device = lexical_hits.to(device) if use_lexical else None
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)
    presence_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    sentence_score_chunks: list[torch.Tensor] = []
    sentence_label_chunks: list[torch.Tensor] = []
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
            if count_head_mode == "sentence":
                if sent_batch is None or mask_batch is None:
                    raise RuntimeError("sentence_repr required for sentence count head evaluation")
                count_preds = planner.expected_count(type_batch, sent_batch, mask_batch)
                sent_logits = planner.sentence_count_logits(type_batch, sent_batch)  # [B, S]
                sent_scores = torch.sigmoid(sent_logits)
                if cache.sentence_record_label is not None:
                    sent_labels = cache.sentence_record_label[
                        doc_idx_batch.cpu(), :, type_batch.cpu()
                    ].to(device=device, dtype=torch.float32)
                else:
                    sent_labels = torch.zeros_like(sent_scores)
                sent_mask_float = mask_batch.float()
                for i in range(int(sent_batch.shape[0])):
                    n = int(sent_mask_float[i].sum().item())
                    if n:
                        sentence_score_chunks.append(sent_scores[i, :n].detach().cpu())
                        sentence_label_chunks.append(sent_labels[i, :n].detach().cpu())
            else:
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
    result = _prediction_metrics(
        population=cache.name,
        documents=cache.documents_count,
        targets=targets,
        presence_scores=[float(score) for score in presence_scores.tolist()],
        count_predictions=[int(value) for value in count_predictions.tolist()],
        k_clip=k_clip,
    )
    if count_head_mode == "sentence" and sentence_score_chunks:
        all_scores = torch.cat(sentence_score_chunks).tolist()
        all_labels = torch.cat(sentence_label_chunks).tolist()
        result["sentence_score_auc"] = round(_binary_auc([int(lbl > 0.5) for lbl in all_labels], [float(s) for s in all_scores]), 6)
    else:
        result["sentence_score_auc"] = 0.5
    return result


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
        "sentence_score_auc": 0.5,
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
        "sentence_score_auc": 0.5,
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


def _sentence_bce_pos_weight(cache: PlannerFeatureCache, *, cap: float = 20.0) -> float:
    """Compute BCE pos_weight from sentence-level labels in the cache."""
    if cache.sentence_record_label is None:
        return 1.0
    label = cache.sentence_record_label.float()
    mask = cache.sentence_mask.float().unsqueeze(-1)  # [N, S, 1]
    positives = (label * mask).sum().item()
    negatives = ((1.0 - label) * mask).sum().item()
    if positives <= 0:
        return 1.0
    return min(negatives / positives, cap)


def _sentence_label_diagnostics(cache: PlannerFeatureCache) -> dict[str, Any]:
    """Compute noise diagnostics for sentence-level heuristic labels."""
    if cache.sentence_record_label is None or cache.counts is None:
        return {"gold_record_sentence_recall": None, "mean_sentence_label_count_over_gold": None}
    label = cache.sentence_record_label.float()  # [N, S_max, T]
    mask = cache.sentence_mask.float()           # [N, S_max]
    counts = cache.counts.float()                # [N, T]
    # Per (doc, type): does any sentence have label=1?
    any_sentence_label = (label.sum(dim=1) > 0).float()  # [N, T]
    positive_gold = (counts > 0).float()
    # gold_record_sentence_recall
    recall_num = (any_sentence_label * positive_gold).sum().item()
    recall_den = positive_gold.sum().clamp_min(1).item()
    # mean_sentence_label_count_over_gold
    sent_count_per_pair = (label * mask.unsqueeze(-1)).sum(dim=1)  # [N, T]
    positive_sent_count = sent_count_per_pair[counts > 0]
    gold_positive_counts = counts[counts > 0]
    if gold_positive_counts.numel():
        ratios = positive_sent_count / gold_positive_counts.clamp_min(1.0)
        mean_ratio = float(ratios.mean().item())
    else:
        mean_ratio = 0.0
    return {
        "gold_record_sentence_recall": round(recall_num / max(recall_den, 1), 6),
        "mean_sentence_label_count_over_gold": round(mean_ratio, 6),
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
    *,
    count_head_mode: str = "document",
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    # TypeGate: unchanged from v2.1
    type_gate_specs = (
        ("type_gate_auc", "absolute_ge", 0.80, 0.05, "max"),
        ("type_gate_f1_youden", "absolute_ge", 0.55, 0.05, "max"),
    )
    # Count: in sentence mode baseline-relative-only (dynamic threshold from predict-1 minus margin)
    count_margins = {"multi_event_dev": 0.05, "all_dev": 0.02}
    for population in ACCEPTANCE_POPULATIONS:
        metrics = metrics_by_population.get(population, {})
        baselines = baselines_by_population.get(population, {})
        for metric_name, _, abs_threshold, rel_margin, aggregator in type_gate_specs:
            value = float(metrics.get(metric_name, 0.0))
            best_baseline, contributing = _best_baseline_for(metric_name, baselines, aggregator)
            abs_pass = value >= float(abs_threshold)
            baseline_threshold = best_baseline + float(rel_margin)
            rel_pass = value >= baseline_threshold
            checks[f"{population}/{metric_name}"] = {
                "value": round(value, 6),
                "absolute_threshold": f">= {float(abs_threshold):.6f}",
                "best_baseline": round(best_baseline, 6),
                "best_baseline_sources": contributing,
                "baseline_relative_threshold": f">= {baseline_threshold:.6f}",
                "baseline_relative_margin": float(rel_margin),
                "passed_absolute": bool(abs_pass),
                "passed_baseline_relative": bool(rel_pass),
                "passed": bool(abs_pass and rel_pass),
            }
        # count_mae_positive
        count_value = float(metrics.get("count_mae_positive", float("inf")))
        predict_one = baselines.get("predict_one", {}).get("count_mae_positive", float("inf"))
        margin = float(count_margins.get(population, 0.05))
        dynamic_threshold = predict_one - margin
        rel_pass_count = count_value <= dynamic_threshold
        if count_head_mode == "sentence":
            # baseline-relative-only: set absolute_threshold = dynamic_threshold so both converge
            abs_pass_count = rel_pass_count
            abs_threshold_repr = f"<= {dynamic_threshold:.6f}"
        else:
            abs_pass_count = count_value <= 0.50
            abs_threshold_repr = "<= 0.500000"
        checks[f"{population}/count_mae_positive"] = {
            "value": round(count_value, 6),
            "absolute_threshold": abs_threshold_repr,
            "best_baseline": round(predict_one, 6),
            "best_baseline_sources": {"predict_one": round(predict_one, 6)},
            "baseline_relative_threshold": f"<= {dynamic_threshold:.6f}",
            "baseline_relative_margin": margin,
            "passed_absolute": bool(abs_pass_count),
            "passed_baseline_relative": bool(rel_pass_count),
            "passed": bool(abs_pass_count and rel_pass_count),
        }
        # sentence_score_auc (only in sentence mode)
        if count_head_mode == "sentence":
            sent_auc = float(metrics.get("sentence_score_auc", 0.5))
            checks[f"{population}/sentence_score_auc"] = {
                "value": round(sent_auc, 6),
                "absolute_threshold": ">= 0.750000",
                "passed_absolute": sent_auc >= 0.75,
                "passed": sent_auc >= 0.75,
            }
    # trend checks
    checks["training/presence_loss_trend"] = {
        "value": _trend_summary(history, "presence_loss"),
        "threshold": "first/last ratio >= 2.0 over at least 10 epochs",
        "passed": _has_required_overall_decrease(history, "presence_loss"),
    }
    if count_head_mode == "sentence":
        checks["training/sentence_count_loss_trend"] = {
            "value": _trend_summary(history, "count_loss"),
            "threshold": "first/last ratio >= 1.5 over at least 10 epochs",
            "passed": _has_required_overall_decrease(history, "count_loss", min_ratio=1.5),
        }
    elif count_head_mode == "coref":
        checks["training/coref_loss_trend"] = {
            "value": _trend_summary(history, "count_loss"),
            "threshold": "first/last ratio >= 1.5 over at least 10 epochs",
            "passed": _has_required_overall_decrease(history, "count_loss", min_ratio=1.5),
        }
    else:
        checks["training/count_loss_trend"] = {
            "value": _trend_summary(history, "count_loss"),
            "threshold": "first/last ratio >= 2.0 over at least 10 epochs",
            "passed": _has_required_overall_decrease(history, "count_loss"),
        }

    # v4 adds pair AUC and cluster B³ checks per population, and a static-baselines
    # minimum for count_mae_positive (predict_one ∧ p5b_lexical_trigger ∧ v2.1 ∧ v3).
    if count_head_mode == "coref":
        coref_count_margins = {"multi_event_dev": 0.05, "all_dev": 0.02}
        for population in ACCEPTANCE_POPULATIONS:
            metrics = metrics_by_population.get(population, {})
            baselines = baselines_by_population.get(population, {})
            count_value = float(metrics.get("count_mae_positive", float("inf")))
            candidate_baselines: dict[str, float] = {}
            for key in ("predict_one", "p5b_lexical_trigger", "legacy_single_softmax",
                        "v2_1_poisson_static", "v3_sentence_static"):
                v = baselines.get(key, {}).get("count_mae_positive")
                if v is not None:
                    try:
                        candidate_baselines[key] = float(v)
                    except (TypeError, ValueError):
                        continue
            best_count_baseline = min(candidate_baselines.values()) if candidate_baselines else float("inf")
            margin = float(coref_count_margins.get(population, 0.05))
            dynamic_threshold = best_count_baseline - margin
            count_pass = count_value <= dynamic_threshold
            checks[f"{population}/count_mae_positive"] = {
                "value": round(count_value, 6),
                "absolute_threshold": f"<= {dynamic_threshold:.6f}",
                "best_baseline": round(best_count_baseline, 6),
                "best_baseline_sources": {k: round(v, 6) for k, v in candidate_baselines.items()},
                "baseline_relative_threshold": f"<= {dynamic_threshold:.6f}",
                "baseline_relative_margin": margin,
                "passed_absolute": bool(count_pass),
                "passed_baseline_relative": bool(count_pass),
                "passed": bool(count_pass),
            }
            for metric_name, threshold in (
                ("pair_auc", 0.75),
                ("cluster_b3_f1", 0.65),
            ):
                value = float(metrics.get(metric_name, 0.0))
                passed = value >= float(threshold)
                checks[f"{population}/{metric_name}"] = {
                    "value": round(value, 6),
                    "absolute_threshold": f">= {float(threshold):.6f}",
                    "passed_absolute": bool(passed),
                    "passed": bool(passed),
                }
        # Ambiguity audit: presence of the field is itself the gate (missing = fail)
        ambig_present = all(
            "ambiguous_pair_rate" in metrics_by_population.get(p, {})
            for p in ACCEPTANCE_POPULATIONS
        )
        checks["ambiguity_audit"] = {
            "value": {
                p: metrics_by_population.get(p, {}).get("ambiguous_pair_rate")
                for p in ACCEPTANCE_POPULATIONS
            },
            "threshold": "ambiguous_pair_rate field must be present for both populations",
            "passed": bool(ambig_present),
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


# ---------------------------------------------------------------------------
# v4 coref-mode helpers
# ---------------------------------------------------------------------------


def _coref_training_pairs(cache: PlannerFeatureCache) -> list[tuple[int, int]]:
    """Return (doc_idx, type_id) tuples whose gold_n_t > 0 — eligible for APCC training."""
    if cache.counts.numel() == 0:
        return []
    nonzero = (cache.counts > 0).nonzero(as_tuple=False).tolist()
    return [(int(d), int(t)) for d, t in nonzero]


def _train_coref_planner(
    planner: RecordPlanner,
    cache: PlannerFeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Unified single-backward training for coref mode.

    Each step processes a batch of (doc, type) pairs, computes presence loss on the
    whole batch and coref loss on its positive subset (mirrors the v2.1 pattern in
    `_train_two_stage_planner`). This gives a single, well-conditioned optimization
    trajectory rather than alternating presence/coref passes with disjoint warmup.
    """
    if cache.span_repr is None or cache.span_normalized_values is None:
        raise RuntimeError("coref training requires span features; check cache build path")
    feature_mode = str(args.encoder_feature_mode)
    use_evidence = feature_mode in {"evidence", "evidence_lexical"}
    use_lexical = feature_mode == "evidence_lexical"

    doc_indices, type_ids, presence_targets, lexical_hits = _pair_tensors(cache)
    if presence_targets.numel() == 0:
        raise RuntimeError("no (doc, type) pairs to train on")
    train_pairs = _coref_training_pairs(cache)
    if not train_pairs:
        raise RuntimeError("no positive (doc, type) pairs to train APCC on")

    presence_pos_weight = torch.tensor(
        float(_population_stats(cache)["presence_pos_weight"]),
        dtype=torch.float32, device=device,
    )

    m_max = int(cache.span_repr.shape[1]) if cache.span_repr is not None else 1

    # Pre-pad pair labels & eligibility to global M_max once; index by (doc, type).
    pair_label_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    eligible_pairs_count = 0
    for doc_idx, type_id in train_pairs:
        labels, eligible, _ = build_coref_pair_labels(
            cache.span_normalized_values[doc_idx],
            cache.documents[doc_idx],
            cache.event_types[type_id],
        )
        if int(eligible.sum().item()) == 0:
            continue
        labels_padded = torch.zeros((m_max, m_max), dtype=labels.dtype)
        eligible_padded = torch.zeros((m_max, m_max), dtype=eligible.dtype)
        n = min(int(labels.shape[0]), m_max)
        labels_padded[:n, :n] = labels[:n, :n]
        eligible_padded[:n, :n] = eligible[:n, :n]
        pair_label_cache[(int(doc_idx), int(type_id))] = (labels_padded, eligible_padded)
        eligible_pairs_count += 1
    if not pair_label_cache:
        raise RuntimeError("no eligible pairs found for coref training (all ambiguous)")

    coref_pw_value = pair_pos_weight(
        [
            [
                MentionSpan(0, 0, 0, v, torch.zeros(1))
                for v in cache.span_normalized_values[d]
            ]
            for d in range(cache.documents_count)
        ],
        cache.documents,
        cache.event_types,
        cap=float(getattr(args, "coref_pos_weight_cap", 20.0)),
    )
    coref_pos_weight = torch.tensor(coref_pw_value, dtype=torch.float32, device=device)

    global_repr_device = cache.global_repr.to(device)
    sentence_repr_device = cache.sentence_repr.to(device) if use_evidence else None
    sentence_mask_device = cache.sentence_mask.to(device) if use_evidence else None
    lexical_hit_device = lexical_hits.to(device) if use_lexical else None
    span_repr_device = cache.span_repr.to(device)
    span_sent_idx_device = cache.span_sent_idx.to(device) if cache.span_sent_idx is not None else torch.zeros(span_repr_device.shape[:2], dtype=torch.long, device=device)
    span_role_id_device = cache.span_role_id.to(device) if cache.span_role_id is not None else torch.zeros(span_repr_device.shape[:2], dtype=torch.long, device=device)
    span_mask_device = cache.span_mask.to(device) if cache.span_mask is not None else torch.ones(span_repr_device.shape[:2], dtype=torch.bool, device=device)

    optimizer = torch.optim.AdamW(planner.parameters(), lr=float(args.lr), weight_decay=1e-4)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    presence_targets_device = presence_targets.to(device)
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)
    batch_size = max(int(args.batch_size), 1)

    steps_per_epoch = max(math.ceil(presence_targets.numel() / batch_size), 1)
    total_steps = max(int(args.max_epochs) * steps_per_epoch, 1)
    warmup_steps = max(int(total_steps * float(args.warmup_ratio)), 1)

    lambda_presence = float(args.lambda_presence)
    lambda_count = float(args.lambda_count)

    history: list[dict[str, float | int]] = []
    global_step = 0
    for epoch in range(1, int(args.max_epochs) + 1):
        totals = {"loss": 0.0, "presence_loss": 0.0, "count_loss": 0.0, "pairs": 0.0, "coref_batches": 0.0, "coref_loss_sum": 0.0}
        order = torch.randperm(presence_targets.numel(), device=device)
        for batch_start in range(0, presence_targets.numel(), batch_size):
            batch_ids = order[batch_start:batch_start + batch_size]
            doc_idx_batch = doc_indices_device[batch_ids]
            type_batch = type_ids_device[batch_ids]
            global_batch = global_repr_device[doc_idx_batch]
            target_batch = presence_targets_device[batch_ids]
            sent_batch = sentence_repr_device[doc_idx_batch] if sentence_repr_device is not None else None
            mask_batch = sentence_mask_device[doc_idx_batch] if sentence_mask_device is not None else None
            lex_batch = lexical_hit_device[batch_ids].unsqueeze(-1) if lexical_hit_device is not None else None

            presence_target = (target_batch > 0).to(dtype=torch.float32)
            presence_logits = planner.presence_logit(
                global_batch, type_batch,
                sentence_repr=sent_batch, sentence_mask=mask_batch, lexical_hit=lex_batch,
            )
            p_loss = presence_loss(presence_logits, presence_target, pos_weight=presence_pos_weight)

            # Coref loss on the positive subset that has eligible pairs
            positive_mask = target_batch > 0
            c_loss = presence_logits.sum() * 0.0
            coref_keys: list[tuple[int, int]] = []
            if bool(positive_mask.any().item()):
                pos_local_idx = positive_mask.nonzero(as_tuple=False).reshape(-1)
                for li in pos_local_idx.tolist():
                    d = int(doc_idx_batch[li].item())
                    t = int(type_batch[li].item())
                    if (d, t) in pair_label_cache:
                        coref_keys.append((d, t))
                if coref_keys:
                    coref_doc_idx = torch.tensor([d for d, _ in coref_keys], dtype=torch.long, device=device)
                    coref_type = torch.tensor([t for _, t in coref_keys], dtype=torch.long, device=device)
                    span_repr_batch = span_repr_device[coref_doc_idx]
                    span_role_batch = span_role_id_device[coref_doc_idx]
                    span_sent_batch = span_sent_idx_device[coref_doc_idx]
                    span_mask_batch = span_mask_device[coref_doc_idx]
                    label_stack = torch.stack([pair_label_cache[k][0] for k in coref_keys]).to(device)
                    eligible_stack = torch.stack([pair_label_cache[k][1] for k in coref_keys]).to(device)
                    affinity = planner.coref_affinity(
                        span_repr_batch, span_role_batch, span_sent_batch, coref_type,
                        span_mask=span_mask_batch,
                    )
                    c_loss = coref_pair_loss(affinity, label_stack, eligible_stack, pos_weight=coref_pos_weight)

            loss = lambda_presence * p_loss + lambda_count * c_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=float(args.grad_clip))
            scale = _lr_scale(global_step, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            optimizer.step()
            global_step += 1

            n = float(batch_ids.numel())
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["presence_loss"] += float(p_loss.detach().cpu()) * n
            totals["pairs"] += n
            if coref_keys:
                totals["coref_batches"] += 1.0
                totals["coref_loss_sum"] += float(c_loss.detach().cpu())

        pair_denom = max(totals["pairs"], 1.0)
        coref_denom = max(totals["coref_batches"], 1.0)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / pair_denom,
            "presence_loss": totals["presence_loss"] / pair_denom,
            "count_loss": totals["coref_loss_sum"] / coref_denom,
            "event_type_pairs": int(totals["pairs"]),
            "coref_batches": int(totals["coref_batches"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return history


def _evaluate_coref_planner(
    planner: RecordPlanner,
    cache: PlannerFeatureCache,
    device: torch.device,
    args: argparse.Namespace,
    coref_threshold: float,
) -> dict[str, float | int | str]:
    if cache.span_repr is None:
        return _empty_metrics(cache.name, 1)
    feature_mode = str(args.encoder_feature_mode)
    use_evidence = feature_mode in {"evidence", "evidence_lexical"}
    use_lexical = feature_mode == "evidence_lexical"
    doc_indices, type_ids, targets, lexical_hits = _pair_tensors(cache)
    if targets.numel() == 0:
        return _empty_metrics(cache.name, 1)

    global_repr_device = cache.global_repr.to(device)
    sentence_repr_device = cache.sentence_repr.to(device) if use_evidence else None
    sentence_mask_device = cache.sentence_mask.to(device) if use_evidence else None
    lexical_hit_device = lexical_hits.to(device) if use_lexical else None
    span_repr_device = cache.span_repr.to(device)
    span_sent_idx_device = cache.span_sent_idx.to(device) if cache.span_sent_idx is not None else torch.zeros(span_repr_device.shape[:2], dtype=torch.long, device=device)
    span_role_id_device = cache.span_role_id.to(device) if cache.span_role_id is not None else torch.zeros(span_repr_device.shape[:2], dtype=torch.long, device=device)
    span_mask_device = cache.span_mask.to(device) if cache.span_mask is not None else torch.ones(span_repr_device.shape[:2], dtype=torch.bool, device=device)
    doc_indices_device = doc_indices.to(device)
    type_ids_device = type_ids.to(device)

    presence_scores: list[float] = []
    count_predictions: list[int] = []
    pair_score_collect: list[float] = []
    pair_label_collect: list[int] = []
    bcubed_p: list[float] = []
    bcubed_r: list[float] = []
    ambiguous_pairs = 0
    eligible_pairs = 0

    with torch.no_grad():
        # presence first (all pairs)
        for start in range(0, int(targets.numel()), max(int(args.eval_batch_size or args.batch_size), 1)):
            stop = min(start + max(int(args.eval_batch_size or args.batch_size), 1), int(targets.numel()))
            doc_idx_batch = doc_indices_device[start:stop]
            type_batch = type_ids_device[start:stop]
            global_batch = global_repr_device[doc_idx_batch]
            sent_batch = sentence_repr_device[doc_idx_batch] if sentence_repr_device is not None else None
            mask_batch = sentence_mask_device[doc_idx_batch] if sentence_mask_device is not None else None
            lex_batch = lexical_hit_device[start:stop].unsqueeze(-1) if lexical_hit_device is not None else None
            presence_probs = torch.sigmoid(
                planner.presence_logit(
                    global_batch, type_batch,
                    sentence_repr=sent_batch, sentence_mask=mask_batch, lexical_hit=lex_batch,
                )
            ).detach().cpu()
            presence_scores.extend([float(v) for v in presence_probs.tolist()])

        # presence threshold via Youden's J (consistent with existing _prediction_metrics path)
        labels = [1 if int(v) > 0 else 0 for v in targets.tolist()]
        gate_metrics = _presence_threshold_metrics(labels, presence_scores)
        threshold = float(gate_metrics["threshold"])

        # coref clustering per (doc, type) pair
        for idx in range(int(targets.numel())):
            d = int(doc_indices_device[idx].item())
            t = int(type_ids_device[idx].item())
            present = presence_scores[idx] >= threshold
            mention_values = cache.span_normalized_values[d] if cache.span_normalized_values is not None else []
            mask_d = span_mask_device[d]
            if not present:
                count_predictions.append(0)
                # still collect pair stats for AUC/B³ on positive types
                if int(targets[idx].item()) > 0:
                    self_eval = _evaluate_coref_pair_metrics(
                        planner, span_repr_device[d:d + 1], span_role_id_device[d:d + 1],
                        span_sent_idx_device[d:d + 1], span_mask_device[d:d + 1],
                        torch.tensor([t], device=device),
                        cache.documents[d], cache.event_types[t],
                        mention_values, coref_threshold,
                    )
                    if self_eval is not None:
                        pair_score_collect.extend(self_eval["pair_scores"])
                        pair_label_collect.extend(self_eval["pair_labels"])
                        bcubed_p.append(self_eval["bcubed_p"])
                        bcubed_r.append(self_eval["bcubed_r"])
                        ambiguous_pairs += self_eval["ambiguous_pairs"]
                        eligible_pairs += self_eval["eligible_pairs"]
                continue
            if not bool(mask_d.to(torch.bool).any().item()):
                count_predictions.append(1)
                continue
            self_eval = _evaluate_coref_pair_metrics(
                planner, span_repr_device[d:d + 1], span_role_id_device[d:d + 1],
                span_sent_idx_device[d:d + 1], span_mask_device[d:d + 1],
                torch.tensor([t], device=device),
                cache.documents[d], cache.event_types[t],
                mention_values, coref_threshold,
            )
            if self_eval is None:
                count_predictions.append(1)
                continue
            count_predictions.append(self_eval["n_clusters"])
            pair_score_collect.extend(self_eval["pair_scores"])
            pair_label_collect.extend(self_eval["pair_labels"])
            bcubed_p.append(self_eval["bcubed_p"])
            bcubed_r.append(self_eval["bcubed_r"])
            ambiguous_pairs += self_eval["ambiguous_pairs"]
            eligible_pairs += self_eval["eligible_pairs"]

    result = _prediction_metrics(
        population=cache.name,
        documents=cache.documents_count,
        targets=targets,
        presence_scores=presence_scores,
        count_predictions=count_predictions,
        k_clip=1,
    )

    pair_auc = _binary_auc(pair_label_collect, pair_score_collect) if pair_score_collect else 0.5
    if bcubed_p:
        mean_p = sum(bcubed_p) / len(bcubed_p)
        mean_r = sum(bcubed_r) / len(bcubed_r)
        f1 = 0.0 if mean_p + mean_r == 0 else 2 * mean_p * mean_r / (mean_p + mean_r)
    else:
        mean_p = mean_r = f1 = 0.0
    total_pair_candidates = eligible_pairs + ambiguous_pairs
    result["pair_auc"] = round(pair_auc, 6)
    result["cluster_b3_precision"] = round(mean_p, 6)
    result["cluster_b3_recall"] = round(mean_r, 6)
    result["cluster_b3_f1"] = round(f1, 6)
    result["ambiguous_pair_rate"] = round(ambiguous_pairs / max(total_pair_candidates, 1), 6)
    result["coref_threshold"] = coref_threshold
    return result


def _evaluate_coref_pair_metrics(
    planner: RecordPlanner,
    span_repr: torch.Tensor,
    span_role: torch.Tensor,
    span_sent: torch.Tensor,
    span_mask: torch.Tensor,
    type_id: torch.Tensor,
    document: DueeDocument,
    event_type: str,
    mention_values: list[str],
    coref_threshold: float,
) -> dict[str, Any] | None:
    affinity = planner.coref_affinity(span_repr, span_role, span_sent, type_id, span_mask=span_mask)
    affinity_one = affinity[0]
    mask_one = span_mask[0]
    if not bool(mask_one.to(torch.bool).any().item()):
        return None
    clusters = predict_clusters(affinity_one, mask_one, threshold=coref_threshold)
    n_clusters = max(len(clusters), 1)

    pred_assignment: list[int | None] = [None] * len(mention_values)
    for cid, members in enumerate(clusters):
        for m in members:
            if 0 <= m < len(mention_values):
                pred_assignment[m] = cid

    gold_partition = build_gold_partition(mention_values, document, event_type)
    bp, br, _ = bcubed_f1(pred_assignment, gold_partition)

    labels, eligible, stats = build_coref_pair_labels(mention_values, document, event_type)
    pair_scores: list[float] = []
    pair_labels: list[int] = []
    M = labels.shape[0]
    probs = torch.sigmoid(affinity_one).detach().cpu()
    for i in range(M):
        for j in range(i + 1, M):
            if not bool(eligible[i, j].item()):
                continue
            pair_scores.append(float(probs[i, j].item()))
            pair_labels.append(int(labels[i, j].item()))
    return {
        "n_clusters": n_clusters,
        "pair_scores": pair_scores,
        "pair_labels": pair_labels,
        "bcubed_p": bp,
        "bcubed_r": br,
        "ambiguous_pairs": int(stats["ambiguous_mentions"]),
        "eligible_pairs": stats["positive_pairs"] + stats["negative_pairs"],
    }


def _tune_coref_threshold(
    planner: RecordPlanner,
    train_cache: PlannerFeatureCache,
    grid: tuple[float, ...],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    if train_cache.span_repr is None or train_cache.span_normalized_values is None:
        return 0.5, {}
    span_repr_d = train_cache.span_repr.to(device)
    span_sent_d = train_cache.span_sent_idx.to(device) if train_cache.span_sent_idx is not None else torch.zeros(span_repr_d.shape[:2], dtype=torch.long, device=device)
    span_role_d = train_cache.span_role_id.to(device) if train_cache.span_role_id is not None else torch.zeros(span_repr_d.shape[:2], dtype=torch.long, device=device)
    span_mask_d = train_cache.span_mask.to(device) if train_cache.span_mask is not None else torch.ones(span_repr_d.shape[:2], dtype=torch.bool, device=device)
    positive_pairs = (train_cache.counts > 0).nonzero(as_tuple=False).tolist()
    if not positive_pairs:
        return 0.5, {}

    affinities: dict[tuple[int, int], torch.Tensor] = {}
    with torch.no_grad():
        for d, t in positive_pairs:
            d = int(d); t = int(t)
            type_id = torch.tensor([t], dtype=torch.long, device=device)
            affinity = planner.coref_affinity(
                span_repr_d[d:d + 1], span_role_d[d:d + 1], span_sent_d[d:d + 1], type_id,
                span_mask=span_mask_d[d:d + 1],
            )[0].detach().cpu()
            affinities[(d, t)] = affinity

    mae_by_tau: dict[str, float] = {}
    best_tau = 0.5
    best_mae = float("inf")
    for tau in grid:
        diff = 0.0
        n = 0
        for (d, t), affinity in affinities.items():
            gold_n = int(train_cache.counts[d, t].item())
            mask = train_cache.span_mask[d] if train_cache.span_mask is not None else torch.ones(affinity.shape[0], dtype=torch.bool)
            if not bool(mask.to(torch.bool).any().item()):
                pred_n = 1
            else:
                clusters = predict_clusters(affinity, mask, threshold=float(tau))
                pred_n = max(len(clusters), 1)
            diff += abs(pred_n - gold_n)
            n += 1
        mae = diff / max(n, 1)
        mae_by_tau[f"{tau:.2f}"] = round(mae, 6)
        if mae < best_mae:
            best_mae = mae
            best_tau = float(tau)
    return best_tau, mae_by_tau


def _coref_pair_diagnostics(cache: PlannerFeatureCache) -> dict[str, Any]:
    if cache.span_normalized_values is None:
        return {"matched_mentions": None, "ambiguous_mentions": None, "positive_pairs": None, "negative_pairs": None}
    matched = 0
    ambiguous = 0
    pos_pairs = 0
    neg_pairs = 0
    for d in range(cache.documents_count):
        for t, event_type in enumerate(cache.event_types):
            if int(cache.counts[d, t].item()) <= 0:
                continue
            _, _, stats = build_coref_pair_labels(
                cache.span_normalized_values[d], cache.documents[d], event_type,
            )
            matched += stats["matched_mentions"]
            ambiguous += stats["ambiguous_mentions"]
            pos_pairs += stats["positive_pairs"]
            neg_pairs += stats["negative_pairs"]
    return {
        "matched_mentions": matched,
        "ambiguous_mentions": ambiguous,
        "positive_pairs": pos_pairs,
        "negative_pairs": neg_pairs,
    }
