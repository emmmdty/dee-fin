"""Contextual argument encoder for the LRD pairwise scorer.

Loads Chinese-RoBERTa-wwm-ext in offline mode and encodes each
``(record, role, argument_text)`` triple as a pooled embedding
vector (argument span + sentence window + learned role embedding).

Used by :mod:`sarge.postprocess.lrd_planner` at both training and
inference time.  Kept separate from the Qwen backbone so LRD can
run on its own encoder without interfering with the candidate-generator
weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ArgumentEncodingConfig:
    """Hyperparameters for the RoBERTa argument encoder."""

    model_path: str  # path to Chinese-RoBERTa-wwm-ext safetensors
    hidden_dim: int = 768
    role_embedding_dim: int = 64
    context_window: int = 64  # chars on each side of the argument span
    max_seq_len: int = 512
    pooling: str = "mean"  # "mean" | "cls"


class ArgumentEncoder(nn.Module):
    """Encode ``(record, role, argument_text)`` triples into embeddings.

    For each argument value the encoder:
    1. Locates the argument span in the document text.
    2. Extracts a ±context_window character window.
    3. Runs the window through a frozen RoBERTa, mean-pools the token
       representations that fall inside the argument span.
    4. Concatenates the pooled vector with a learned per-role embedding.

    Record embeddings are obtained by mean-pooling all argument
    embeddings of a record, then concatenating a binary role-mask
    that indicates which schema roles are covered by the record.
    """

    def __init__(self, config: ArgumentEncodingConfig, role_vocabulary: list[str]):
        super().__init__()
        self.config = config
        self.role_vocabulary = role_vocabulary
        self.role_to_idx = {role: i for i, role in enumerate(role_vocabulary)}

        # Frozen RoBERTa backbone (loaded lazily to avoid import overhead).
        self._encoder: nn.Module | None = None

        self.role_embed = nn.Embedding(len(role_vocabulary), config.role_embedding_dim)

        # Projection from RoBERTa + role → hidden_dim.
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim + config.role_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

    def _ensure_encoder(self) -> nn.Module:
        if self._encoder is not None:
            return self._encoder
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        encoder = AutoModel.from_pretrained(
            self.config.model_path,
            use_safetensors=True,
            local_files_only=True,
        )
        # Freeze backbone.
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        # Move to the same device as the rest of the planner.
        device = next(self.projection.parameters()).device
        encoder.to(device)
        # Store tokenizer as attribute for encoding convenience.
        self._tokenizer = tokenizer
        self._encoder = encoder
        return encoder

    @property
    def encoder(self) -> nn.Module:
        return self._ensure_encoder()

    @property
    def tokenizer(self) -> Any:
        self._ensure_encoder()
        return self._tokenizer

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_masks: torch.Tensor,
        role_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of argument spans.

        Args:
            input_ids: [B, L] tokenised document windows.
            attention_mask: [B, L].
            span_masks: [B, L] binary mask marking tokens inside the
                        argument span.
            role_indices: [B] indices into ``role_vocabulary``.

        Returns:
            [B, hidden_dim] pooled argument embeddings.
        """
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # [B, L, D]

        span_mask_float = span_masks.float().unsqueeze(-1)  # [B, L, 1]
        if self.config.pooling == "mean":
            pooled = (token_embeddings * span_mask_float).sum(dim=1) / span_mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = token_embeddings[:, 0, :]  # CLS token

        role_emb = self.role_embed(role_indices)  # [B, D_role]
        concat = torch.cat([pooled, role_emb], dim=-1)  # [B, D + D_role]
        return self.projection(concat)

    def encode_span_raw(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen RoBERTa + mean-pool span, NO role emb or projection.

        Returns [B, hidden_dim] pooled span vectors — the intermediate
        representation that can be cached and later combined with a
        learnable role embedding + projection on GPU.
        """
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # [B, L, D]

        span_mask_float = span_masks.float().unsqueeze(-1)  # [B, L, 1]
        if self.config.pooling == "mean":
            pooled = (token_embeddings * span_mask_float).sum(dim=1) / span_mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = token_embeddings[:, 0, :]
        return pooled  # [B, D]

    def project_span(
        self,
        pooled: torch.Tensor,  # [B, hidden_dim]
        role_indices: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Combine a cached pooled span with role embedding + projection."""
        role_emb = self.role_embed(role_indices)  # [B, D_role]
        concat = torch.cat([pooled, role_emb], dim=-1)  # [B, D + D_role]
        return self.projection(concat)

    def record_embedding(
        self,
        arg_embeddings: torch.Tensor,  # [N_args, hidden_dim]
        role_mask: torch.Tensor,  # [n_unique_roles] bool
    ) -> torch.Tensor:
        """Pool per-argument embeddings into a record-level vector.

        Args:
            arg_embeddings: stacked argument embeddings for a single record.
            role_mask: [n_unique_roles] binary tensor; 1 where the record
                       has a non-empty value for that role.

        Returns:
            [hidden_dim] record embedding.
        """
        arg_pooled = arg_embeddings.mean(dim=0)  # [D]
        return torch.cat([arg_pooled, role_mask.float().to(arg_pooled.device)])
