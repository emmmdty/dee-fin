from __future__ import annotations

import torch
from torch import nn

from evaluator.canonical.normalize import normalize_text
from evaluator.canonical.types import CanonicalEventRecord


class RecordPlanner(nn.Module):
    def __init__(self, hidden_size: int, num_event_types: int, k_max: int = 10) -> None:
        super().__init__()
        self.k_max = k_max
        self.type_embedding = nn.Embedding(max(num_event_types, 1), hidden_size)
        self.proj = nn.Linear(hidden_size * 2, k_max + 1)

    def forward(self, global_repr: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        if type_id.dim() == 0:
            type_id = type_id.unsqueeze(0)
        type_emb = self.type_embedding(type_id.to(global_repr.device))
        if global_repr.dim() == 1:
            global_repr = global_repr.unsqueeze(0)
        return self.proj(torch.cat([global_repr, type_emb], dim=-1))

    def predict_n_t(self, global_repr: torch.Tensor, type_id: torch.Tensor) -> int:
        logits = self.forward(global_repr, type_id)
        return int(torch.argmax(logits, dim=-1)[0].item())

    @staticmethod
    def gold_n_t(records: list[CanonicalEventRecord], event_type: str) -> int:
        normalized_event_type = normalize_text(event_type)
        return sum(1 for record in records if normalize_text(record.event_type) == normalized_event_type)


def planner_loss(logits: torch.Tensor, targets: torch.Tensor, *, k_max: int) -> tuple[torch.Tensor, int]:
    mask = targets <= k_max
    skipped = int((~mask).sum().item())
    if not mask.any():
        return logits.sum() * 0.0, skipped
    return nn.functional.cross_entropy(logits[mask], targets[mask].to(logits.device)), skipped
