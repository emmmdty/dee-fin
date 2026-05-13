from __future__ import annotations

import re
import unicodedata

import torch
from torch import nn

from carve.text_segmentation import Sentence
from carve.tokenization import TokenOffset
from evaluator.canonical.normalize import normalize_optional_text, normalize_text


TAG_O = 0
TAG_B = 1
TAG_I = 2


class MentionCRF(nn.Module):
    def __init__(self, hidden_size: int, num_tags: int = 3) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.emission = nn.Linear(hidden_size, num_tags)
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    def forward_loss(self, token_repr: torch.Tensor, bio_labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = self.emission(token_repr)
        log_z = self._log_partition(emissions, mask)
        gold = self._gold_score(emissions, bio_labels, mask)
        return (log_z - gold).mean()

    def decode(self, token_repr: torch.Tensor, mask: torch.Tensor) -> list[list[tuple[int, int]]]:
        emissions = self.emission(token_repr)
        decoded: list[list[tuple[int, int]]] = []
        for batch_id in range(emissions.shape[0]):
            length = int(mask[batch_id].sum().item())
            if length == 0:
                decoded.append([])
                continue
            tags = self._viterbi(emissions[batch_id, :length])
            decoded.append(_tags_to_spans(tags))
        return decoded

    def _log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = emissions[:, 0]
        for step in range(1, emissions.shape[1]):
            next_score = torch.logsumexp(score.unsqueeze(2) + self.transitions.unsqueeze(0), dim=1) + emissions[:, step]
            score = torch.where(mask[:, step].unsqueeze(1), next_score, score)
        return torch.logsumexp(score, dim=1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        score = emissions[batch, 0, tags[:, 0]]
        for step in range(1, emissions.shape[1]):
            transition = self.transitions[tags[:, step - 1], tags[:, step]]
            emission = emissions[batch, step, tags[:, step]]
            score = score + torch.where(mask[:, step], transition + emission, torch.zeros_like(emission))
        return score

    def _viterbi(self, emissions: torch.Tensor) -> list[int]:
        score = emissions[0]
        backpointers: list[torch.Tensor] = []
        for step in range(1, emissions.shape[0]):
            next_score = score.unsqueeze(1) + self.transitions
            best_score, best_tag = next_score.max(dim=0)
            score = best_score + emissions[step]
            backpointers.append(best_tag)
        tags = [int(score.argmax().item())]
        for backpointer in reversed(backpointers):
            tags.append(int(backpointer[tags[-1]].item()))
        tags.reverse()
        return tags


def build_bio_labels(
    sentence: Sentence,
    tokens_with_offsets: list[TokenOffset],
    gold_values: list[tuple[str, str, str]],
) -> list[int]:
    labels = [TAG_O] * len(tokens_with_offsets)
    occupied = [False] * len(tokens_with_offsets)
    for _event_type, _role, value in gold_values:
        normalized_value = normalize_optional_text(value)
        if not normalized_value:
            continue
        for start, end in _find_occurrences(sentence.text, normalized_value):
            token_ids = [
                index
                for index, token in enumerate(tokens_with_offsets)
                if token.char_start >= sentence.char_start + start and token.char_end <= sentence.char_start + end
            ]
            if not token_ids or any(occupied[index] for index in token_ids):
                continue
            labels[token_ids[0]] = TAG_B
            occupied[token_ids[0]] = True
            for token_id in token_ids[1:]:
                labels[token_id] = TAG_I
                occupied[token_id] = True
    return labels


def _find_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    normalized_text, offset_map = _normalized_offset_map(text)
    spans = []
    for match in re.finditer(re.escape(normalize_text(value)), normalized_text):
        if match.start() < match.end():
            spans.append((offset_map[match.start()][0], offset_map[match.end() - 1][1]))
    return spans


def _normalized_offset_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    previous_space = False
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char)
        for normalized_char in normalized:
            if normalized_char.isspace():
                if previous_space:
                    continue
                chars.append(" ")
                offsets.append((index, index + 1))
                previous_space = True
            else:
                chars.append(normalized_char)
                offsets.append((index, index + 1))
                previous_space = False
    while chars and chars[0] == " ":
        chars.pop(0)
        offsets.pop(0)
    while chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def _tags_to_spans(tags: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, tag in enumerate(tags):
        if tag == TAG_B:
            if start is not None:
                spans.append((start, index))
            start = index
        elif tag == TAG_I:
            if start is None:
                start = index
        else:
            if start is not None:
                spans.append((start, index))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans
