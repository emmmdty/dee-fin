from __future__ import annotations

import math

import torch
from torch import nn

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


class RecordPlanner(nn.Module):
    def __init__(self, hidden_size: int, num_event_types: int, k_max: int = 10) -> None:
        super().__init__()
        self.k_max = k_max
        self.type_gate = TypeGate(hidden_size, num_event_types)
        self.count_planner = CountPlanner(hidden_size, num_event_types)

    def forward(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "presence_logit": self.presence_logit(
                global_repr,
                type_id,
                sentence_repr=sentence_repr,
                sentence_mask=sentence_mask,
                lexical_hit=lexical_hit,
            ),
            "log_lambda": self.count_log_lambda(
                global_repr,
                type_id,
                sentence_repr=sentence_repr,
                sentence_mask=sentence_mask,
                lexical_hit=lexical_hit,
            ),
        }

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

    def count_log_lambda(
        self,
        global_repr: torch.Tensor,
        type_id: torch.Tensor,
        sentence_repr: torch.Tensor | None = None,
        sentence_mask: torch.Tensor | None = None,
        lexical_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
    return nn.functional.binary_cross_entropy_with_logits(
        logits.reshape_as(targets.to(device=logits.device, dtype=logits.dtype)),
        targets.to(device=logits.device, dtype=logits.dtype),
        pos_weight=weight,
    )


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
