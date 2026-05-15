from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from scipy.cluster.hierarchy import fcluster as _scipy_fcluster
    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.spatial.distance import squareform as _scipy_squareform

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from evaluator.canonical.normalize import normalize_text
from evaluator.canonical.types import CanonicalEventRecord


class TypeGate(nn.Module):
    def __init__(self, hidden_size: int, num_event_types: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.type_embedding = nn.Embedding(max(num_event_types, 1), hidden_size)
        self.proj = nn.Linear(hidden_size * 3 + 1, 1)

    def forward(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = _planner_features(
            global_repr,
            type_id,
            self.type_embedding,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )
        return self.proj(features).squeeze(-1)

    def predict_present(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        *,
        threshold: float = 0.5,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> bool:
        logit = self.forward(
            global_repr,
            type_id,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )
        return bool((torch.sigmoid(logit).reshape(-1)[0] >= threshold).item())


class CountPlanner(nn.Module):
    def __init__(self, hidden_size: int, num_event_types: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.type_embedding = nn.Embedding(max(num_event_types, 1), hidden_size)
        self.proj = nn.Linear(hidden_size * 3 + 1, 1)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = _planner_features(
            global_repr,
            type_id,
            self.type_embedding,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )
        return self.proj(features).squeeze(-1)

    def predict_count(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        *,
        k_clip: int,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> int:
        log_lambda = self.forward(
            global_repr,
            type_id,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )
        counts = truncated_poisson_argmax(log_lambda, k_clip=k_clip)
        return int(counts.reshape(-1)[0].item())


class SentenceLevelCountPlanner(nn.Module):
    """Per-sentence count head for R3 v3.

    Predicts a per-sentence logit for "this sentence contains a record of type X",
    using [sentence_repr; type_emb; sentence_repr * type_emb] as the feature triple.
    Expected count n_t is obtained as `sum_s sigmoid(logit_st) * mask_s`.
    """

    def __init__(self, hidden_size: int, num_event_types: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.type_embedding = nn.Embedding(max(num_event_types, 1), hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, type_id: torch.Tensor, sentence_repr: torch.Tensor) -> torch.Tensor:
        """Return per-sentence logits.

        Args:
            type_id: [B] long tensor, event type indices.
            sentence_repr: [B, S, H] float tensor, sentence-level representations.

        Returns:
            [B, S] float tensor of per-sentence logits.
        """
        device = next(self.parameters()).device
        if type_id.dim() == 0:
            type_id = type_id.unsqueeze(0)
        if sentence_repr.dim() == 2:
            sentence_repr = sentence_repr.unsqueeze(0)
        type_id = type_id.to(device=device, dtype=torch.long)
        sentence_repr = sentence_repr.to(device=device)
        batch_size, n_sent, _ = sentence_repr.shape
        if type_id.shape[0] == 1 and batch_size > 1:
            type_id = type_id.expand(batch_size)
        type_emb = self.type_embedding(type_id)                      # [B, H]
        type_emb_s = type_emb.unsqueeze(1).expand(-1, n_sent, -1)    # [B, S, H]
        features = torch.cat([sentence_repr, type_emb_s, sentence_repr * type_emb_s], dim=-1)
        return self.scorer(features).squeeze(-1)                      # [B, S]

    def expected_count(
        self,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregated expected record count n_t = round(Σ_s σ(logit_st) · mask_s), clamped ≥ 1.

        Returns [B] long tensor.
        """
        logits = self.forward(type_id, sentence_repr)                 # [B, S]
        probs = torch.sigmoid(logits)
        mask = sentence_mask.to(device=probs.device, dtype=probs.dtype)
        expected = (probs * mask).sum(dim=-1)                         # [B]
        return expected.clamp(min=1.0).round().to(torch.long)


class ArgumentCoreferenceHead(nn.Module):
    """Pairwise argument coreference head for R3 v4 (APCC).

    For each candidate mention pair (m_i, m_j) of a given event type, predicts
    `P(same_record | m_i, m_j, t)`. At inference time, `predict_clusters` thresholds
    the affinity matrix and returns connected components; `n_t = #components`.

    Inputs (forward):
        span_repr:     [B, M_max, H] float — span pooled encoder representations.
        span_role_id:  [B, M_max] long   — role index per mention; 0 = padding / no-role.
        type_id:       [B] long          — event type id per document.
        sent_pos:      [B, M_max] long   — sentence index of each mention (for pos embedding).
        span_mask:     [B, M_max] bool   — valid mention mask.

    Output:
        affinity:      [B, M_max, M_max] float logits, symmetric, with diag forced to a
                       very negative value (self-pair not used in BCE).
    """

    def __init__(
        self,
        hidden_size: int,
        num_event_types: int,
        num_roles: int,
        *,
        max_sentence_pos: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_roles = num_roles
        self.type_embedding = nn.Embedding(max(num_event_types, 1), hidden_size)
        self.role_embedding = nn.Embedding(max(num_roles + 1, 1), hidden_size)  # 0 reserved for padding
        self.sent_pos_embedding = nn.Embedding(max(max_sentence_pos + 1, 1), hidden_size)  # 0 reserved
        self.span_proj = nn.Linear(hidden_size, hidden_size)
        self.feature_proj = nn.Linear(hidden_size * 3, hidden_size)  # [span; role; sent_pos]
        self.same_role_bias = nn.Parameter(torch.zeros(1))
        self.same_sentence_bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.span_proj.bias)
        nn.init.zeros_(self.feature_proj.bias)

    def _build_query(
        self,
        span_repr: torch.Tensor,
        span_role_id: torch.Tensor,
        sent_pos: torch.Tensor,
        type_id: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        span_repr = span_repr.to(device=device, dtype=self.feature_proj.weight.dtype)
        span_role_id = span_role_id.to(device=device, dtype=torch.long)
        sent_pos = sent_pos.to(device=device, dtype=torch.long)
        type_id = type_id.to(device=device, dtype=torch.long)
        if span_repr.dim() == 2:
            span_repr = span_repr.unsqueeze(0)
            span_role_id = span_role_id.unsqueeze(0)
            sent_pos = sent_pos.unsqueeze(0)
        if type_id.dim() == 0:
            type_id = type_id.unsqueeze(0)
        batch_size, m_max, _ = span_repr.shape
        if type_id.shape[0] == 1 and batch_size > 1:
            type_id = type_id.expand(batch_size)

        # clamp into legal embedding ranges to be robust to caller off-by-one
        span_role_id = span_role_id.clamp(min=0, max=self.role_embedding.num_embeddings - 1)
        sent_pos = sent_pos.clamp(min=0, max=self.sent_pos_embedding.num_embeddings - 1)

        span_h = self.span_proj(span_repr)                                   # [B, M, H]
        role_h = self.role_embedding(span_role_id)                            # [B, M, H]
        sent_h = self.sent_pos_embedding(sent_pos)                            # [B, M, H]
        type_h = self.type_embedding(type_id).unsqueeze(1).expand(-1, m_max, -1)  # [B, M, H]

        features = torch.cat([span_h, role_h, sent_h], dim=-1)                # [B, M, 3H]
        q = self.feature_proj(features) + type_h                              # [B, M, H]
        return q

    def forward(
        self,
        span_repr: torch.Tensor,
        span_role_id: torch.Tensor,
        sent_pos: torch.Tensor,
        type_id: torch.Tensor,
        span_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._build_query(span_repr, span_role_id, sent_pos, type_id)
        scale = math.sqrt(max(self.hidden_size, 1))
        affinity = torch.einsum("bih,bjh->bij", q, q) / scale                 # symmetric in (i,j)

        # additive priors that exploit known structural signals (model can ignore by zeroing weight)
        role_i = span_role_id.to(q.device).long().unsqueeze(2)
        role_j = span_role_id.to(q.device).long().unsqueeze(1)
        same_role = (role_i == role_j).to(dtype=affinity.dtype)
        affinity = affinity + same_role * self.same_role_bias

        sent_i = sent_pos.to(q.device).long().unsqueeze(2)
        sent_j = sent_pos.to(q.device).long().unsqueeze(1)
        same_sentence = (sent_i == sent_j).to(dtype=affinity.dtype)
        affinity = affinity + same_sentence * self.same_sentence_bias

        # diagonal is not a meaningful pair
        diag_mask = torch.eye(affinity.shape[-1], dtype=torch.bool, device=affinity.device).unsqueeze(0)
        affinity = affinity.masked_fill(diag_mask, -1e9)

        if span_mask is not None:
            mask = span_mask.to(device=affinity.device).bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            invalid_i = (~mask).unsqueeze(2)
            invalid_j = (~mask).unsqueeze(1)
            affinity = affinity.masked_fill(invalid_i | invalid_j, -1e9)

        return affinity


def predict_clusters_agglomerative(
    affinity: torch.Tensor,
    span_mask: torch.Tensor,
    *,
    threshold: float,
    temperature: float = 1.0,
) -> list[set[int]]:
    """Average-linkage agglomerative clustering over calibrated pairwise affinities.

    Two clusters merge only when their *average* inter-cluster sigmoid probability
    exceeds `threshold`, preventing a single false-positive edge from triggering a
    cascade merge (the connected-components weakness fixed in v5).

    Falls back to `predict_clusters` (connected-components) if scipy is not available.

    Args:
        affinity: [M, M] logit matrix for a single document.
        span_mask: [M] boolean mask; False positions are padding.
        threshold: probability cutoff used as the merge criterion (same search space as v4).
        temperature: divisor applied to logits before sigmoid for calibration (default 1.0 = no-op).
    """
    if not _SCIPY_AVAILABLE:
        return predict_clusters(affinity, span_mask, threshold=threshold)

    if affinity.dim() != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"affinity must be [M, M]; got {tuple(affinity.shape)}")
    m_max = int(affinity.shape[0])
    valid = span_mask.to(torch.bool).reshape(-1).tolist()
    if len(valid) != m_max:
        raise ValueError("span_mask length must match affinity dimension")
    valid_indices = [i for i, v in enumerate(valid) if v]
    if not valid_indices:
        return []
    if len(valid_indices) == 1:
        return [{valid_indices[0]}]

    probs = torch.sigmoid(affinity.detach().cpu() / max(float(temperature), 1e-6)).numpy()

    n = len(valid_indices)
    idx_map = {vi: li for li, vi in enumerate(valid_indices)}

    # Build condensed distance matrix for valid mentions only
    import numpy as np
    dist_condensed = []
    for li in range(n):
        for lj in range(li + 1, n):
            vi, vj = valid_indices[li], valid_indices[lj]
            p = float((probs[vi, vj] + probs[vj, vi]) / 2.0)
            dist_condensed.append(max(1.0 - p, 0.0))

    dist_arr = np.array(dist_condensed, dtype=np.float64)
    Z = _scipy_linkage(dist_arr, method="average")
    # fcluster with criterion='distance': merge clusters whose average inter-cluster
    # distance < (1 - threshold), i.e., average probability > threshold
    cut = max(1.0 - float(threshold), 0.0)
    raw_labels = _scipy_fcluster(Z, t=cut, criterion="distance")

    groups: dict[int, set[int]] = {}
    for local_i, cluster_id in enumerate(raw_labels):
        groups.setdefault(int(cluster_id), set()).add(valid_indices[local_i])
    return list(groups.values())


def predict_clusters(
    affinity: torch.Tensor,
    span_mask: torch.Tensor,
    *,
    threshold: float,
) -> list[set[int]]:
    """Connected components over `sigmoid(affinity) >= threshold` adjacency.

    `affinity` is a single doc's [M, M] logit matrix. Pads are filtered via `span_mask`.
    Returns a list of clusters (sets of valid mention indices). Singletons are included.
    """
    if affinity.dim() != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"affinity must be [M, M]; got {tuple(affinity.shape)}")
    m_max = int(affinity.shape[0])
    valid = span_mask.to(torch.bool).reshape(-1).tolist()
    if len(valid) != m_max:
        raise ValueError("span_mask length must match affinity dimension")
    if not any(valid):
        return []

    probs = torch.sigmoid(affinity.detach().cpu()).numpy()
    parent = list(range(m_max))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(m_max):
        if not valid[i]:
            continue
        for j in range(i + 1, m_max):
            if not valid[j]:
                continue
            if probs[i, j] >= threshold:
                union(i, j)

    groups: dict[int, set[int]] = {}
    for i in range(m_max):
        if not valid[i]:
            continue
        root = find(i)
        groups.setdefault(root, set()).add(i)
    return list(groups.values())


def coref_pair_loss(
    affinity: torch.Tensor,
    pair_labels: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor | float,
) -> torch.Tensor:
    """Masked BCE over upper-triangle eligible pairs.

    `affinity`, `pair_labels`, `eligible_mask` are all [B, M, M]. The function only
    considers the strict upper triangle (i < j) to avoid double-counting and the
    diagonal. Ambiguous pairs (those not in eligible_mask) contribute zero loss.
    """
    if affinity.shape != pair_labels.shape or affinity.shape != eligible_mask.shape:
        raise ValueError("affinity / pair_labels / eligible_mask must share shape")
    if affinity.dim() != 3:
        raise ValueError("expected [B, M, M] tensors")
    m_max = int(affinity.shape[-1])
    device = affinity.device
    upper = torch.triu(torch.ones((m_max, m_max), dtype=torch.bool, device=device), diagonal=1)
    upper = upper.unsqueeze(0).expand_as(affinity)
    mask = eligible_mask.to(dtype=torch.bool, device=device) & upper
    if not bool(mask.any().item()):
        return affinity.sum() * 0.0
    weight = torch.as_tensor(pos_weight, dtype=affinity.dtype, device=device)
    labels = pair_labels.to(device=device, dtype=affinity.dtype)
    raw = F.binary_cross_entropy_with_logits(
        affinity, labels, pos_weight=weight, reduction="none",
    )
    return (raw * mask.to(dtype=affinity.dtype)).sum() / mask.to(dtype=affinity.dtype).sum().clamp_min(1.0)


class RecordPlanner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_event_types: int,
        k_max: int = 10,
        *,
        count_head_mode: str = "document",
        num_roles: int = 0,
        max_sentence_pos: int = 256,
    ) -> None:
        super().__init__()
        self.k_max = k_max
        self.count_head_mode = count_head_mode
        self.type_gate = TypeGate(hidden_size, num_event_types)
        if count_head_mode == "sentence":
            self.count_planner = SentenceLevelCountPlanner(hidden_size, num_event_types)
        elif count_head_mode in ("coref", "coref_v5"):
            self.count_planner = ArgumentCoreferenceHead(
                hidden_size,
                num_event_types,
                num_roles=num_roles,
                max_sentence_pos=max_sentence_pos,
            )
        else:
            self.count_planner = CountPlanner(hidden_size, num_event_types)

    def forward(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
        span_repr: torch.Tensor | None = None,
        span_role_id: torch.Tensor | None = None,
        span_sent_pos: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {
            "presence_logit": self.presence_logit(
                global_repr,
                type_id,
                sentence_repr=sentence_repr,
                sentence_mask=sentence_mask,
                lexical_hit=lexical_hit,
            ),
        }
        if self.count_head_mode == "sentence":
            if sentence_repr is None:
                raise ValueError("sentence_repr is required for sentence count head mode")
            result["sentence_count_logits"] = self.sentence_count_logits(type_id, sentence_repr)
        elif self.count_head_mode == "coref":
            if span_repr is None or span_role_id is None or span_sent_pos is None:
                raise ValueError(
                    "span_repr, span_role_id, span_sent_pos are required for coref count head mode"
                )
            result["coref_affinity"] = self.coref_affinity(
                span_repr,
                span_role_id,
                span_sent_pos,
                type_id,
                span_mask=span_mask,
            )
        else:
            result["log_lambda"] = self.count_log_lambda(
                global_repr,
                type_id,
                sentence_repr=sentence_repr,
                sentence_mask=sentence_mask,
                lexical_hit=lexical_hit,
            )
        return result

    def coref_affinity(
        self,
        span_repr: torch.Tensor,
        span_role_id: torch.Tensor,
        span_sent_pos: torch.Tensor,
        type_id: torch.Tensor,
        span_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.count_head_mode not in ("coref", "coref_v5"):
            raise RuntimeError(
                f"coref_affinity is only available when count_head_mode='coref' or 'coref_v5', "
                f"got {self.count_head_mode!r}"
            )
        return self.count_planner(span_repr, span_role_id, span_sent_pos, type_id, span_mask=span_mask)

    def presence_logit(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.type_gate(
            global_repr,
            type_id,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )

    def sentence_count_logits(
        self,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor,
    ) -> torch.Tensor:
        if self.count_head_mode != "sentence":
            raise RuntimeError(
                f"sentence_count_logits is only available when count_head_mode='sentence', "
                f"got {self.count_head_mode!r}"
            )
        return self.count_planner(type_id, sentence_repr)

    def expected_count(
        self,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.count_head_mode != "sentence":
            raise RuntimeError(
                f"expected_count is only available when count_head_mode='sentence', "
                f"got {self.count_head_mode!r}"
            )
        return self.count_planner.expected_count(type_id, sentence_repr, sentence_mask)

    def count_log_lambda(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.count_head_mode == "sentence":
            raise RuntimeError(
                "count_log_lambda is not available when count_head_mode='sentence'; "
                "use sentence_count_logits + expected_count instead"
            )
        if self.count_head_mode in ("coref", "coref_v5"):
            raise RuntimeError(
                "count_log_lambda is not available when count_head_mode='coref'/'coref_v5'; "
                "use coref_affinity + predict_clusters instead"
            )
        return self.count_planner(
            global_repr,
            type_id,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )

    def predict_n_t(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        *,
        threshold: float = 0.5,
        k_clip: int | None = None,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
        span_repr: torch.Tensor | None = None,
        span_role_id: torch.Tensor | None = None,
        span_sent_pos: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
        coref_threshold: float = 0.5,
    ) -> int:
        if not self.type_gate.predict_present(
            global_repr,
            type_id,
            threshold=threshold,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        ):
            return 0
        if self.count_head_mode == "sentence":
            if sentence_repr is None or sentence_mask is None:
                raise ValueError("sentence_repr and sentence_mask are required for sentence count head mode")
            count = self.expected_count(type_id, sentence_repr, sentence_mask)
            return int(count.reshape(-1)[0].item())
        if self.count_head_mode in ("coref", "coref_v5"):
            if span_repr is None or span_role_id is None or span_sent_pos is None or span_mask is None:
                raise ValueError(
                    "span_repr, span_role_id, span_sent_pos, span_mask are required for coref count head mode"
                )
            affinity = self.coref_affinity(
                span_repr,
                span_role_id,
                span_sent_pos,
                type_id,
                span_mask=span_mask,
            )
            # take the first (and only) batch row
            if affinity.dim() == 3:
                affinity_one = affinity[0]
                mask_one = span_mask if span_mask.dim() == 1 else span_mask[0]
            else:
                affinity_one = affinity
                mask_one = span_mask
            if not bool(mask_one.to(torch.bool).any().item()):
                # TypeGate fired but no candidate mentions extracted; can not contradict presence -> return 1.
                return 1
            if self.count_head_mode == "coref_v5":
                clusters = predict_clusters_agglomerative(affinity_one, mask_one, threshold=coref_threshold)
            else:
                clusters = predict_clusters(affinity_one, mask_one, threshold=coref_threshold)
            return max(len(clusters), 1)
        return self.count_planner.predict_count(
            global_repr,
            type_id,
            k_clip=k_clip or self.k_max,
            sentence_repr=sentence_repr,
            sentence_mask=sentence_mask,
            lexical_hit=lexical_hit,
        )

    @staticmethod
    def gold_n_t(records: list[CanonicalEventRecord], event_type: str) -> int:
        normalized_event_type = normalize_text(event_type)
        return sum(1 for record in records if normalize_text(record.event_type) == normalized_event_type)

    @staticmethod
    def gold_presence(records: list[CanonicalEventRecord], event_type: str) -> int:
        return int(RecordPlanner.gold_n_t(records, event_type) > 0)


def presence_loss(logits: torch.Tensor, targets: torch.Tensor, *, pos_weight: torch.Tensor | float) -> torch.Tensor:
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits.reshape_as(targets.to(device=logits.device, dtype=logits.dtype)),
        targets.to(device=logits.device, dtype=logits.dtype),
        pos_weight=weight,
    )


def sentence_count_loss(
    sent_logits: torch.Tensor,
    sent_labels: torch.Tensor,
    sent_mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor | float,
) -> torch.Tensor:
    """Masked sentence-level BCE loss for the v3 sentence count head.

    Args:
        sent_logits: [B, S] float logits.
        sent_labels: [B, S] float (0/1) labels.
        sent_mask: [B, S] bool mask (True = valid sentence).
        pos_weight: positive class weight for BCE.

    Returns:
        Scalar mean loss over all valid sentence positions.
    """
    if sent_logits.numel() == 0:
        return sent_logits.sum() * 0.0
    weight = torch.as_tensor(pos_weight, dtype=sent_logits.dtype, device=sent_logits.device)
    labels = sent_labels.to(device=sent_logits.device, dtype=sent_logits.dtype)
    mask = sent_mask.to(device=sent_logits.device, dtype=sent_logits.dtype)
    raw = F.binary_cross_entropy_with_logits(sent_logits, labels, pos_weight=weight, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1e-8)


def truncated_poisson_nll(
    log_lambda: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = targets.to(device=log_lambda.device, dtype=log_lambda.dtype)
    if targets.numel() == 0:
        return log_lambda.sum() * 0.0
    if bool((targets <= 0).any().item()):
        raise ValueError("zero-truncated Poisson targets must be positive")
    log_lambda, targets = torch.broadcast_tensors(log_lambda, targets)
    lambda_ = torch.exp(log_lambda).clamp_min(torch.finfo(log_lambda.dtype).tiny)
    log_nonzero_prob = _log1mexp_neg(lambda_)
    nll = lambda_ - targets * log_lambda + torch.lgamma(targets + 1.0) + log_nonzero_prob
    if sample_weights is None:
        return nll.mean()
    weights = sample_weights.to(device=nll.device, dtype=nll.dtype).reshape_as(nll)
    return (nll * weights).sum() / weights.sum().clamp_min(1e-8)


def truncated_poisson_argmax(log_lambda: torch.Tensor, *, k_clip: int) -> torch.Tensor:
    if k_clip < 1:
        raise ValueError("k_clip must be at least 1")
    original_shape = log_lambda.shape
    flat_log_lambda = log_lambda.reshape(-1)
    counts = torch.arange(1, k_clip + 1, dtype=log_lambda.dtype, device=log_lambda.device)
    scores = flat_log_lambda.unsqueeze(-1) * counts - torch.lgamma(counts + 1.0)
    predictions = counts[torch.argmax(scores, dim=-1)].to(torch.long)
    return predictions.reshape(original_shape)


def planner_loss(logits: torch.Tensor, targets: torch.Tensor, *, k_max: int) -> tuple[torch.Tensor, int]:
    mask = targets <= k_max
    skipped = int((~mask).sum().item())
    if not mask.any():
        return logits.sum() * 0.0, skipped
    return nn.functional.cross_entropy(logits[mask], targets[mask].to(logits.device)), skipped


def _planner_features(
    global_repr: torch.Tensor,
    type_id: torch.Tensor,
    type_embedding: nn.Embedding,
    *,
    sentence_repr: torch.Tensor | None = None,
    sentence_mask: torch.Tensor | None = None,
    lexical_hit: torch.Tensor | None = None,
) -> torch.Tensor:
    device = type_embedding.weight.device
    if type_id.dim() == 0:
        type_id = type_id.unsqueeze(0)
    if global_repr.dim() == 1:
        global_repr = global_repr.unsqueeze(0)
    global_repr = global_repr.to(device)
    type_id = type_id.to(device=device, dtype=torch.long)
    if global_repr.shape[0] == 1 and type_id.shape[0] > 1:
        global_repr = global_repr.expand(type_id.shape[0], -1)
    if type_id.shape[0] == 1 and global_repr.shape[0] > 1:
        type_id = type_id.expand(global_repr.shape[0])
    if global_repr.shape[0] != type_id.shape[0]:
        raise ValueError("global_repr and type_id batch sizes must match")
    type_emb = type_embedding(type_id)
    batch_size = global_repr.shape[0]
    hidden_size = int(type_embedding.embedding_dim)
    dtype = global_repr.dtype

    if sentence_repr is None:
        evidence_vec = torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
    else:
        evidence_vec = _evidence_attention(
            type_emb,
            sentence_repr,
            sentence_mask,
            hidden_size=hidden_size,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    if lexical_hit is None:
        lex = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    else:
        lex = lexical_hit.to(device=device, dtype=dtype).reshape(-1, 1)
        if lex.shape[0] == 1 and batch_size > 1:
            lex = lex.expand(batch_size, -1)
        elif lex.shape[0] != batch_size:
            raise ValueError("lexical_hit batch size mismatch")

    return torch.cat([global_repr, type_emb, evidence_vec, lex], dim=-1)


def _evidence_attention(
    type_emb: torch.Tensor,
    sentence_repr: torch.Tensor,
    sentence_mask: torch.Tensor | None,
    *,
    hidden_size: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sentence_repr = sentence_repr.to(device=device, dtype=dtype)
    if sentence_repr.dim() == 2:
        sentence_repr = sentence_repr.unsqueeze(0)
    if sentence_repr.shape[0] == 1 and batch_size > 1:
        sentence_repr = sentence_repr.expand(batch_size, -1, -1)
    if sentence_repr.shape[0] != batch_size:
        raise ValueError("sentence_repr batch size mismatch")
    if sentence_repr.shape[1] == 0:
        return torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
    if sentence_mask is None:
        mask = torch.ones(sentence_repr.shape[:2], dtype=torch.bool, device=device)
    else:
        mask = sentence_mask.to(device=device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        mask = mask.bool()
    scale = math.sqrt(max(hidden_size, 1))
    logits = torch.einsum("bh,bnh->bn", type_emb, sentence_repr) / scale
    any_valid = mask.any(dim=-1, keepdim=True)
    safe_logits = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(safe_logits, dim=-1)
    weights = torch.where(any_valid, weights, torch.zeros_like(weights))
    return torch.einsum("bn,bnh->bh", weights, sentence_repr)


def _log1mexp_neg(lambda_: torch.Tensor) -> torch.Tensor:
    threshold = torch.tensor(math.log(2.0), dtype=lambda_.dtype, device=lambda_.device)
    return torch.where(
        lambda_ <= threshold,
        torch.log(-torch.expm1(-lambda_)),
        torch.log1p(-torch.exp(-lambda_)),
    )
