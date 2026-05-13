from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from carve.text_segmentation import Sentence
from carve.tokenization import TokenOffset, tokenize_with_offsets


@dataclass(frozen=True)
class EncoderOutput:
    token_repr: torch.Tensor
    token_offsets: list[TokenOffset]
    sentence_repr: torch.Tensor
    global_repr: torch.Tensor


class ToySentenceEncoder(nn.Module):
    def __init__(self, hidden_size: int = 32, vocab_size: int = 4096) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def encode_document(self, text: str, sentences: list[Sentence]) -> EncoderOutput:
        tokens = tokenize_with_offsets(text)
        device = self.embedding.weight.device
        if tokens:
            token_ids = torch.tensor(
                [ord(token.text) % self.embedding.num_embeddings for token in tokens],
                dtype=torch.long,
                device=device,
            )
            token_repr = self.embedding(token_ids)
        else:
            token_repr = torch.zeros((0, self.hidden_size), dtype=torch.float32, device=device)
        sentence_reprs = []
        for sentence in sentences:
            indices = [
                index
                for index, token in enumerate(tokens)
                if token.char_start >= sentence.char_start and token.char_end <= sentence.char_end
            ]
            if indices:
                sentence_reprs.append(token_repr[torch.tensor(indices, dtype=torch.long, device=device)].mean(dim=0))
            else:
                sentence_reprs.append(torch.zeros((self.hidden_size,), dtype=token_repr.dtype, device=token_repr.device))
        if sentence_reprs:
            sentence_repr = torch.stack(sentence_reprs)
            global_repr = sentence_repr.mean(dim=0)
        else:
            sentence_repr = torch.zeros((0, self.hidden_size), dtype=torch.float32, device=device)
            global_repr = torch.zeros((self.hidden_size,), dtype=torch.float32, device=device)
        return EncoderOutput(token_repr=token_repr, token_offsets=tokens, sentence_repr=sentence_repr, global_repr=global_repr)


class RobertaSentenceEncoder(nn.Module):
    def __init__(self, model_path: str | Path, *, device: torch.device | None = None) -> None:
        super().__init__()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer

        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.model.parameters(recurse=recurse)

    def encode_document(self, text: str, sentences: list[Sentence]) -> EncoderOutput:
        token_reprs = []
        token_offsets: list[TokenOffset] = []
        sentence_reprs = []
        for sentence in sentences:
            encoded = self.tokenizer(
                sentence.text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded).last_hidden_state[0]
            valid_indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and encoded["attention_mask"][0, index].item()
            ]
            if valid_indices:
                index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=output.device)
                sentence_tokens = output[index_tensor]
                sentence_reprs.append(sentence_tokens.mean(dim=0))
                for token_id, (start, end) in zip(valid_indices, [offsets[index] for index in valid_indices]):
                    token_offsets.append(
                        TokenOffset(
                            text=sentence.text[start:end],
                            char_start=sentence.char_start + start,
                            char_end=sentence.char_start + end,
                        )
                    )
                    token_reprs.append(output[token_id])
            else:
                sentence_reprs.append(torch.zeros((self.hidden_size,), dtype=output.dtype, device=output.device))
        if token_reprs:
            token_repr = torch.stack(token_reprs)
        else:
            token_repr = torch.zeros((0, self.hidden_size), dtype=torch.float32, device=self.device)
        if sentence_reprs:
            sentence_repr = torch.stack(sentence_reprs)
            global_repr = sentence_repr.mean(dim=0)
        else:
            sentence_repr = torch.zeros((0, self.hidden_size), dtype=torch.float32, device=self.device)
            global_repr = torch.zeros((self.hidden_size,), dtype=torch.float32, device=self.device)
        return EncoderOutput(token_repr=token_repr, token_offsets=token_offsets, sentence_repr=sentence_repr, global_repr=global_repr)


def build_encoder(model_path: str, *, device: torch.device | None = None) -> nn.Module:
    if model_path == "__toy__":
        return ToySentenceEncoder()
    return RobertaSentenceEncoder(model_path, device=device)
