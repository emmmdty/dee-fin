"""R3 v4 (APCC) helpers: mention extraction, span pooling, pair labels, B³ F1.

This module isolates v4-specific logic from `carve/p3_planner_only_runner.py`,
which keeps v1/v2.1/v3 paths byte-identical. The runner imports these helpers
only when `--count-head-mode coref` is set.

Mention sources:
    crf  - load a trained P3 mention CRF checkpoint and decode per sentence.
    gold - use gold record argument values found in sentences (oracle ablation).

Static reference baselines below come from prior non-smoke measurements; they
let v4 acceptance gates report against v2.1 and v3 without retraining.
Source: docs/measurements/r3_planner_only_duee_fin_seed42_v2_1.md,
        docs/measurements/r3_planner_only_duee_fin_seed42_v3.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from carve.datasets import DueeDocument
from carve.encoder import EncoderOutput
from carve.p3_mention_crf import MentionCRF
from carve.text_segmentation import Sentence, split_sentences
from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.schema import EventSchema


STATIC_REFERENCE_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "v2_1_poisson": {
        "multi_event_dev": {"count_mae_positive": 0.8702},
        "all_dev": {"count_mae_positive": 0.4040},
    },
    "v3_sentence": {
        "multi_event_dev": {"count_mae_positive": 5.6058},
        "all_dev": {"count_mae_positive": 4.6580},
    },
}

CANDIDATE_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_MAX_MENTIONS = 64


@dataclass(frozen=True)
class MentionSpan:
    char_start: int
    char_end: int
    sent_idx: int
    normalized_value: str
    span_repr: torch.Tensor  # [H], CPU float32


def load_p3_mention_crf(checkpoint_path: str | Path, hidden_size: int) -> MentionCRF:
    """Load a trained P3 mention CRF from a P3 checkpoint dict."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "mention_crf" not in payload:
        raise ValueError(
            f"P3 checkpoint at {checkpoint_path} does not contain 'mention_crf' key; "
            f"available keys: {list(payload.keys())}"
        )
    crf = MentionCRF(hidden_size=hidden_size)
    crf.load_state_dict(payload["mention_crf"])
    crf.eval()
    return crf


def _sentence_token_indices(encoded: EncoderOutput, sentence: Sentence) -> list[int]:
    return [
        index
        for index, token in enumerate(encoded.token_offsets)
        if token.char_start >= sentence.char_start and token.char_end <= sentence.char_end
    ]


def extract_predicted_mentions(
    encoded: EncoderOutput,
    sentences: list[Sentence],
    mention_crf: MentionCRF,
    *,
    hidden_size: int,
    max_mentions: int = DEFAULT_MAX_MENTIONS,
) -> list[MentionSpan]:
    """Run mention CRF on each sentence; return MentionSpans with mean-pooled span_repr."""
    spans: list[MentionSpan] = []
    crf_device = next(mention_crf.parameters()).device
    for sent_idx, sentence in enumerate(sentences):
        indices = _sentence_token_indices(encoded, sentence)
        if not indices:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=encoded.token_repr.device)
        token_repr = encoded.token_repr[idx_tensor]
        if token_repr.shape[0] == 0:
            continue
        crf_input = token_repr.unsqueeze(0).to(crf_device)
        mask = torch.ones((1, crf_input.shape[1]), dtype=torch.bool, device=crf_device)
        with torch.no_grad():
            decoded = mention_crf.decode(crf_input, mask)[0]
        tokens = [encoded.token_offsets[i] for i in indices]
        for token_start, token_end in decoded:
            if not (0 <= token_start < token_end <= len(tokens)):
                continue
            raw = "".join(tok.text for tok in tokens[token_start:token_end])
            value = normalize_optional_text(raw)
            if not value:
                continue
            span_h = token_repr[token_start:token_end].mean(dim=0).detach().to(dtype=torch.float32).cpu()
            spans.append(
                MentionSpan(
                    char_start=int(tokens[token_start].char_start),
                    char_end=int(tokens[token_end - 1].char_end),
                    sent_idx=sent_idx,
                    normalized_value=value,
                    span_repr=span_h,
                )
            )
            if len(spans) >= max_mentions:
                return spans
    return spans


def extract_gold_mention_spans(
    encoded: EncoderOutput,
    sentences: list[Sentence],
    document: DueeDocument,
    schema: EventSchema,
    *,
    hidden_size: int,
    max_mentions: int = DEFAULT_MAX_MENTIONS,
) -> list[MentionSpan]:
    """Oracle mention extractor: each gold argument value found in a sentence is a mention.

    Use only as an ablation upper bound — must not be the acceptance path.
    """
    spans: list[MentionSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for sent_idx, sentence in enumerate(sentences):
        sent_text_norm = normalize_text(sentence.text)
        indices = _sentence_token_indices(encoded, sentence)
        if not indices:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=encoded.token_repr.device)
        token_repr = encoded.token_repr[idx_tensor]
        tokens = [encoded.token_offsets[i] for i in indices]
        # collect (event_type, role, normalized_value) tuples that appear in this sentence
        for record in document.records:
            for role, values in record.arguments.items():
                for v in values:
                    nv = normalize_optional_text(v)
                    if not nv or nv not in sent_text_norm:
                        continue
                    idx = sentence.text.find(v) if v in sentence.text else -1
                    if idx < 0:
                        idx = sentence.text.find(nv)
                    if idx < 0:
                        continue
                    char_start = sentence.char_start + idx
                    char_end = char_start + len(nv)
                    tok_ids = [
                        i for i, t in enumerate(tokens)
                        if t.char_start >= char_start and t.char_end <= char_end
                    ]
                    if not tok_ids:
                        continue
                    key = (char_start, char_end, nv)
                    if key in seen:
                        continue
                    seen.add(key)
                    span_h = token_repr[tok_ids[0]:tok_ids[-1] + 1].mean(dim=0).detach().to(dtype=torch.float32).cpu()
                    spans.append(
                        MentionSpan(
                            char_start=char_start,
                            char_end=char_end,
                            sent_idx=sent_idx,
                            normalized_value=nv,
                            span_repr=span_h,
                        )
                    )
                    if len(spans) >= max_mentions:
                        return spans
    return spans


def build_coref_pair_labels(
    mention_values: list[str],
    document: DueeDocument,
    event_type: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build pairwise gold labels for one (document, event_type).

    Returns:
        labels:    [M, M] float (0 or 1), only meaningful where eligible is True
        eligible:  [M, M] bool (excludes diagonal, ambiguous mentions, and unmatched pairs)
        stats:     dict with positive_pairs, negative_pairs, ambiguous_mentions, matched_mentions
    """
    M = len(mention_values)
    labels = torch.zeros((M, M), dtype=torch.float32)
    eligible = torch.zeros((M, M), dtype=torch.bool)
    stats = {"positive_pairs": 0, "negative_pairs": 0, "ambiguous_mentions": 0, "matched_mentions": 0}
    if M == 0:
        return labels, eligible, stats

    nt = normalize_text(event_type)
    relevant_records: list[set[str]] = []
    for record in document.records:
        if normalize_text(record.event_type) != nt:
            continue
        arg_values: set[str] = set()
        for role_values in record.arguments.values():
            for v in role_values:
                nv = normalize_optional_text(v)
                if nv:
                    arg_values.add(nv)
        relevant_records.append(arg_values)

    if not relevant_records:
        return labels, eligible, stats

    mention_to_records: list[list[int]] = []
    for value in mention_values:
        matched = [k for k, rec in enumerate(relevant_records) if value in rec]
        mention_to_records.append(matched)

    for i in range(M):
        ri = mention_to_records[i]
        if not ri:
            continue
        if len(ri) > 1:
            stats["ambiguous_mentions"] += 1
            continue
        stats["matched_mentions"] += 1

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            ri, rj = mention_to_records[i], mention_to_records[j]
            if not ri or not rj:
                continue
            if len(ri) > 1 or len(rj) > 1:
                continue
            same = ri[0] == rj[0]
            eligible[i, j] = True
            labels[i, j] = 1.0 if same else 0.0
            if i < j:
                if same:
                    stats["positive_pairs"] += 1
                else:
                    stats["negative_pairs"] += 1
    return labels, eligible, stats


def build_gold_partition(
    mention_values: list[str],
    document: DueeDocument,
    event_type: str,
) -> list[list[int]]:
    """For B³: gold partition of NON-AMBIGUOUS matched mentions into record-id clusters."""
    M = len(mention_values)
    if M == 0:
        return []
    nt = normalize_text(event_type)
    relevant_records: list[set[str]] = []
    for record in document.records:
        if normalize_text(record.event_type) != nt:
            continue
        arg_values: set[str] = set()
        for role_values in record.arguments.values():
            for v in role_values:
                nv = normalize_optional_text(v)
                if nv:
                    arg_values.add(nv)
        relevant_records.append(arg_values)
    if not relevant_records:
        return []
    partition: list[list[int]] = [[] for _ in relevant_records]
    for i, value in enumerate(mention_values):
        matched = [k for k, rec in enumerate(relevant_records) if value in rec]
        if len(matched) != 1:
            continue
        partition[matched[0]].append(i)
    return [c for c in partition if c]


def bcubed_f1(
    pred_assignment: list[int | None],
    gold_partition: list[list[int]],
) -> tuple[float, float, float]:
    """B-cubed precision/recall/F1 over the subset of mentions present in both pred and gold.

    pred_assignment[i] is the predicted cluster id for mention i (or None).
    gold_partition is a list of clusters; absent mentions are unsupervised.
    """
    M = len(pred_assignment)
    if M == 0:
        return 0.0, 0.0, 0.0
    gold_id: list[int | None] = [None] * M
    for cid, cluster in enumerate(gold_partition):
        for m in cluster:
            if 0 <= m < M:
                gold_id[m] = cid
    eligible = [i for i in range(M) if gold_id[i] is not None and pred_assignment[i] is not None]
    if not eligible:
        return 0.0, 0.0, 0.0
    pred_clusters: dict[int, set[int]] = {}
    for i, pid in enumerate(pred_assignment):
        if pid is None:
            continue
        pred_clusters.setdefault(pid, set()).add(i)
    gold_clusters: dict[int, set[int]] = {}
    for i, gid in enumerate(gold_id):
        if gid is None:
            continue
        gold_clusters.setdefault(gid, set()).add(i)
    precisions: list[float] = []
    recalls: list[float] = []
    for i in eligible:
        p_set = pred_clusters[pred_assignment[i]]
        g_set = gold_clusters[gold_id[i]]
        intersect = len(p_set & g_set)
        precisions.append(intersect / max(len(p_set), 1))
        recalls.append(intersect / max(len(g_set), 1))
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def pad_mentions_to_tensors(
    per_doc_spans: list[list[MentionSpan]],
    *,
    hidden_size: int,
    max_mentions: int,
) -> dict[str, torch.Tensor]:
    """Pack per-doc lists of MentionSpan into padded tensors for batch processing."""
    n_docs = len(per_doc_spans)
    if n_docs == 0 or max_mentions <= 0:
        return {
            "span_repr": torch.zeros((0, max(max_mentions, 1), hidden_size), dtype=torch.float32),
            "span_sent_idx": torch.zeros((0, max(max_mentions, 1)), dtype=torch.long),
            "span_role_id": torch.zeros((0, max(max_mentions, 1)), dtype=torch.long),
            "span_mask": torch.zeros((0, max(max_mentions, 1)), dtype=torch.bool),
        }
    span_repr = torch.zeros((n_docs, max_mentions, hidden_size), dtype=torch.float32)
    span_sent_idx = torch.zeros((n_docs, max_mentions), dtype=torch.long)
    span_role_id = torch.zeros((n_docs, max_mentions), dtype=torch.long)
    span_mask = torch.zeros((n_docs, max_mentions), dtype=torch.bool)
    for d, spans in enumerate(per_doc_spans):
        for i, span in enumerate(spans[:max_mentions]):
            span_repr[d, i] = span.span_repr
            span_sent_idx[d, i] = span.sent_idx
            span_role_id[d, i] = 0  # v4: no role assignment (see plan §Mention 来源)
            span_mask[d, i] = True
    return {
        "span_repr": span_repr,
        "span_sent_idx": span_sent_idx,
        "span_role_id": span_role_id,
        "span_mask": span_mask,
    }


def pair_pos_weight(
    per_doc_spans: list[list[MentionSpan]],
    documents: list[DueeDocument],
    event_types: list[str],
    *,
    cap: float = 20.0,
) -> float:
    """pos_weight = clamp(#neg / #pos, max=cap) over all eligible pairs in train cache."""
    positives = 0
    negatives = 0
    for doc_idx, document in enumerate(documents):
        spans = per_doc_spans[doc_idx]
        mention_values = [s.normalized_value for s in spans]
        for event_type in event_types:
            _, _, stats = build_coref_pair_labels(mention_values, document, event_type)
            positives += stats["positive_pairs"]
            negatives += stats["negative_pairs"]
    if positives <= 0:
        return 1.0
    return float(min(negatives / positives, cap))


def grid_search_threshold_on_train(
    *,
    affinity_per_pair: dict[tuple[int, int], torch.Tensor],
    span_mask_per_doc: torch.Tensor,
    per_doc_spans: list[list[MentionSpan]],
    documents: list[DueeDocument],
    event_types: list[str],
    grid: tuple[float, ...] = CANDIDATE_THRESHOLD_GRID,
    type_gate_present: dict[tuple[int, int], bool] | None = None,
) -> tuple[float, dict[float, float]]:
    """Choose τ minimising train count_mae_positive.

    affinity_per_pair[(doc_idx, type_id)] is the [M, M] affinity tensor (only for
    (doc, type) pairs the caller wishes to consider; typically all of train).
    Returns the best threshold and the full {τ: mae} map.
    """
    from carve.p3_planner import predict_clusters

    mae_by_tau: dict[float, float] = {}
    best_tau = grid[len(grid) // 2]
    best_mae = float("inf")
    for tau in grid:
        diff_sum = 0.0
        n = 0
        for (doc_idx, type_id), affinity in affinity_per_pair.items():
            gold_n = sum(
                1 for r in documents[doc_idx].records
                if normalize_text(r.event_type) == normalize_text(event_types[type_id])
            )
            if gold_n <= 0:
                continue
            if type_gate_present is not None and not type_gate_present.get((doc_idx, type_id), True):
                pred_n = 0
            else:
                mask = span_mask_per_doc[doc_idx]
                if not bool(mask.to(torch.bool).any().item()):
                    pred_n = 1
                else:
                    clusters = predict_clusters(affinity, mask, threshold=tau)
                    pred_n = max(len(clusters), 1)
            diff_sum += abs(pred_n - gold_n)
            n += 1
        mae = diff_sum / max(n, 1)
        mae_by_tau[tau] = round(mae, 6)
        if mae < best_mae:
            best_mae = mae
            best_tau = float(tau)
    return best_tau, mae_by_tau


def static_baseline_for(metric_name: str, baseline_key: str, population: str) -> float | None:
    """Look up a static reference baseline value (v2_1_poisson or v3_sentence)."""
    table = STATIC_REFERENCE_BASELINES.get(baseline_key, {})
    pop = table.get(population, {})
    return pop.get(metric_name)
