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

    def encode_many_sentences(
        self,
        sentences: list[Sentence],
        *,
        max_length: int = 192,
        chunk_size: int = 64,
    ) -> list[EncoderOutput]:
        result = []
        for s in sentences:
            doc_out = self.encode_document("", [s])
            # sentence_repr from encode_document is [1, H]; squeeze to [H]
            sent_repr = doc_out.sentence_repr[0] if doc_out.sentence_repr.shape[0] > 0 else torch.zeros(self.hidden_size, dtype=torch.float32, device=doc_out.sentence_repr.device)
            result.append(EncoderOutput(
                token_repr=doc_out.token_repr,
                token_offsets=doc_out.token_offsets,
                sentence_repr=sent_repr,
                global_repr=sent_repr,
            ))
        return result


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
        """Encode all sentences of one document; returned as a single EncoderOutput."""
        per_sent = self.encode_many_sentences(sentences)
        return _assemble_doc_output(per_sent, self.hidden_size, self.device)

    def encode_many_sentences(
        self,
        sentences: list[Sentence],
        *,
        max_length: int = 192,
        chunk_size: int = 64,
    ) -> list[EncoderOutput]:
        """Encode a flat list of sentences in GPU batches; returns one EncoderOutput per sentence.

        Uses ``chunk_size`` sentences per RoBERTa forward to fill tensor cores without
        excessive padding. ``max_length=192`` covers p95 of DuEE-Fin sentence lengths
        (~121 chars, which is ~121 Chinese tokens + 2 special tokens).
        """
        if not sentences:
            return []
        per_sent: list[EncoderOutput] = []
        for chunk_start in range(0, len(sentences), chunk_size):
            chunk = sentences[chunk_start : chunk_start + chunk_size]
            encoded = self.tokenizer(
                [s.text for s in chunk],
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            all_offsets = encoded.pop("offset_mapping").tolist()
            attn_cpu = encoded["attention_mask"].tolist()
            encoded_gpu = {k: v.to(self.device) for k, v in encoded.items()}
            all_outputs = self.model(**encoded_gpu).last_hidden_state  # [B, seq, H]
            for i, sentence in enumerate(chunk):
                offsets = all_offsets[i]
                output = all_outputs[i]
                attn = attn_cpu[i]
                valid = [j for j, (s, e) in enumerate(offsets) if e > s and attn[j]]
                if valid:
                    idx_t = torch.tensor(valid, dtype=torch.long, device=output.device)
                    toks = output[idx_t]
                    sent_repr = toks.mean(dim=0)
                    tok_offsets = [
                        TokenOffset(
                            text=sentence.text[offsets[j][0] : offsets[j][1]],
                            char_start=sentence.char_start + offsets[j][0],
                            char_end=sentence.char_start + offsets[j][1],
                        )
                        for j in valid
                    ]
                else:
                    toks = torch.zeros((0, self.hidden_size), dtype=output.dtype, device=output.device)
                    sent_repr = torch.zeros((self.hidden_size,), dtype=output.dtype, device=output.device)
                    tok_offsets = []
                per_sent.append(EncoderOutput(
                    token_repr=toks,
                    token_offsets=tok_offsets,
                    sentence_repr=sent_repr,
                    global_repr=sent_repr,  # assembled by caller
                ))
        return per_sent


def _assemble_doc_output(per_sent: list[EncoderOutput], hidden_size: int, device: torch.device) -> EncoderOutput:
    if not per_sent:
        empty = torch.zeros((0, hidden_size), dtype=torch.float32, device=device)
        return EncoderOutput(
            token_repr=empty,
            token_offsets=[],
            sentence_repr=empty,
            global_repr=torch.zeros((hidden_size,), dtype=torch.float32, device=device),
        )
    sentence_repr = torch.stack([o.sentence_repr for o in per_sent])
    token_parts = [o.token_repr for o in per_sent if o.token_repr.shape[0] > 0]
    token_repr = torch.cat(token_parts) if token_parts else torch.zeros((0, hidden_size), dtype=sentence_repr.dtype, device=device)
    token_offsets = [off for o in per_sent for off in o.token_offsets]
    global_repr = sentence_repr.mean(dim=0)
    return EncoderOutput(token_repr=token_repr, token_offsets=token_offsets, sentence_repr=sentence_repr, global_repr=global_repr)


def build_encoder(model_path: str, *, device: torch.device | None = None) -> nn.Module:
    if model_path == "__toy__":
        return ToySentenceEncoder()
    return RobertaSentenceEncoder(model_path, device=device)
