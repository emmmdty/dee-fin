from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from carve.datasets import DueeDocument
from carve.text_segmentation import Sentence
from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.schema import EventSchema


@dataclass(frozen=True)
class EvidenceLabels:
    y_ev_type: torch.Tensor
    y_ev_role: torch.Tensor
    role_mask: torch.Tensor
    event_types: tuple[str, ...]
    roles_by_event_type: dict[str, tuple[str, ...]]
    pos_sent: dict[tuple[str, str, str], list[int]]
    unalignable: list[tuple[str, str, str]]


def build_evidence_labels(
    document: DueeDocument,
    sentences: list[Sentence],
    schema: EventSchema,
) -> EvidenceLabels:
    event_types = tuple(schema.event_roles.keys())
    roles_by_event_type = {event_type: tuple(roles) for event_type, roles in schema.event_roles.items()}
    max_roles = max((len(roles) for roles in roles_by_event_type.values()), default=0)
    y_ev_type = torch.zeros((len(sentences), len(event_types)), dtype=torch.float32)
    y_ev_role = torch.zeros((len(sentences), len(event_types), max_roles), dtype=torch.float32)
    role_mask = torch.zeros((len(event_types), max_roles), dtype=torch.bool)
    event_index = {event_type: index for index, event_type in enumerate(event_types)}
    role_index: dict[tuple[str, str], int] = {}
    for event_type, roles in roles_by_event_type.items():
        for index, role in enumerate(roles):
            role_mask[event_index[event_type], index] = True
            role_index[(event_type, role)] = index

    normalized_sentences = [normalize_text(sentence.text) for sentence in sentences]
    pos_sets: dict[tuple[str, str, str], set[int]] = {}
    unalignable: list[tuple[str, str, str]] = []
    for record in document.records:
        event_type = normalize_text(record.event_type)
        if event_type not in event_index:
            continue
        type_id = event_index[event_type]
        for role, values in record.arguments.items():
            normalized_role = normalize_text(role)
            role_id = role_index.get((event_type, normalized_role))
            if role_id is None:
                continue
            for value in values:
                normalized_value = normalize_optional_text(value)
                if not normalized_value:
                    continue
                matched = [
                    sent_id
                    for sent_id, sentence_text in enumerate(normalized_sentences)
                    if normalized_value in sentence_text
                ]
                key = (event_type, normalized_role, normalized_value)
                if not matched:
                    unalignable.append(key)
                    continue
                pos_sets.setdefault(key, set()).update(matched)
                for sent_id in matched:
                    y_ev_role[sent_id, type_id, role_id] = 1.0
                    y_ev_type[sent_id, type_id] = 1.0

    pos_sent = {key: sorted(indices) for key, indices in sorted(pos_sets.items())}
    return EvidenceLabels(
        y_ev_type=y_ev_type,
        y_ev_role=y_ev_role,
        role_mask=role_mask,
        event_types=event_types,
        roles_by_event_type=roles_by_event_type,
        pos_sent=pos_sent,
        unalignable=unalignable,
    )


class EvidenceHead(nn.Module):
    def __init__(self, hidden_size: int, num_event_types: int, max_roles: int) -> None:
        super().__init__()
        self.type_classifier = nn.Linear(hidden_size, num_event_types)
        self.role_classifier = nn.Linear(hidden_size, num_event_types * max_roles)
        self.num_event_types = num_event_types
        self.max_roles = max_roles

    def forward(self, sentence_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        type_logits = self.type_classifier(sentence_repr)
        role_logits = self.role_classifier(sentence_repr).reshape(
            sentence_repr.shape[0], self.num_event_types, self.max_roles
        )
        return type_logits, role_logits


class PointerHead(nn.Module):
    """Role-value-conditioned sentence attention for backward grounding."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size * 3, hidden_size)

    def forward(
        self,
        sentence_repr: torch.Tensor,
        type_emb: torch.Tensor,
        role_emb: torch.Tensor,
        value_repr: torch.Tensor,
    ) -> torch.Tensor:
        if type_emb.dim() == 1:
            type_emb = type_emb.unsqueeze(0)
        if role_emb.dim() == 1:
            role_emb = role_emb.unsqueeze(0)
        if value_repr.dim() == 1:
            value_repr = value_repr.unsqueeze(0)
        query = self.query(torch.cat([type_emb, role_emb, value_repr], dim=-1)).squeeze(0)
        scores = sentence_repr @ query
        return torch.log_softmax(scores, dim=-1)


def evidence_bce_loss(
    type_logits: torch.Tensor,
    role_logits: torch.Tensor,
    labels: EvidenceLabels,
    role_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    type_loss = nn.functional.binary_cross_entropy_with_logits(type_logits, labels.y_ev_type.to(type_logits.device))
    mask = (role_mask if role_mask is not None else labels.role_mask).to(role_logits.device)
    expanded_mask = mask.unsqueeze(0).expand_as(role_logits)
    if expanded_mask.any():
        role_loss = nn.functional.binary_cross_entropy_with_logits(
            role_logits[expanded_mask],
            labels.y_ev_role.to(role_logits.device)[expanded_mask],
        )
    else:
        role_loss = role_logits.sum() * 0.0
    return type_loss + role_loss


def pointer_mi_loss(log_p_sent: torch.Tensor, pos_sent_indices: list[list[int]]) -> torch.Tensor:
    losses = []
    for row, indices in zip(log_p_sent, pos_sent_indices):
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=row.device)
        losses.append(-torch.logsumexp(row[index_tensor], dim=0))
    if not losses:
        return log_p_sent.sum() * 0.0
    return torch.stack(losses).mean()
