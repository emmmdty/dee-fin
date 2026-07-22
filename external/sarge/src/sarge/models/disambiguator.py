"""Pairwise record-scoring head for learned record disambiguation.

Given a set of candidate records (each represented as a pooled
argument-context embedding), produces the probability that two records
belong to the same gold event record.  Used inside
:mod:`sarge.postprocess.lrd_planner` as the core scoring signal to
replace the rule-based conservative-split / conservative-merge /
near-dedup steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseScorer(nn.Module):
    """Compute ``P(same_record | record_i, record_j)``.

    Input features (per pair):
    - record_i embedding H_i  [D]
    - record_j embedding H_j  [D]
    - elementwise product H_i ⊙ H_j  [D]
    - Jaccard overlap over argument surface forms (scalar, precomputed)

    Architecture: 3-layer MLP with LayerNorm and GELU, output logit.

    Used both in training (BCE loss) and inference (agglomerative
    linkage on 1 - sigmoid(logit)).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # 3 * input_dim (concat + hadamard) + 1 (surface overlap)
        self.mlp = nn.Sequential(
            nn.Linear(3 * input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        record_embeddings: torch.Tensor,  # [N, D]
        surface_overlap: torch.Tensor,  # [N, N] jaccard scores
        pair_mask: torch.Tensor | None = None,  # [N, N] bool (upper-tri for training)
    ) -> torch.Tensor:
        """Score all pairs in a batch.

        Args:
            record_embeddings: N × D per-record embeddings.
            surface_overlap: N × N precomputed Jaccard overlap matrix.
            pair_mask: optional N × N mask selecting which pairs to score.

        Returns:
            N × N matrix of pairwise same-record logits.
        """
        n = record_embeddings.shape[0]
        # [N, D] → [N, N, D]
        h_i = record_embeddings.unsqueeze(1).expand(-1, n, -1)
        h_j = record_embeddings.unsqueeze(0).expand(n, -1, -1)
        hadamard = h_i * h_j
        overlap = surface_overlap.unsqueeze(-1)  # [N, N, 1]

        features = torch.cat([h_i, h_j, hadamard, overlap], dim=-1)  # [N, N, 3D+1]
        logits = self.mlp(features).squeeze(-1)  # [N, N]

        if pair_mask is not None:
            logits = logits.masked_fill(~pair_mask, -1e9)
        return logits

    def score_pair(self, h_i: torch.Tensor, h_j: torch.Tensor, jaccard: float) -> torch.Tensor:
        """Score a single pair (used during agglomerative clustering)."""
        hadamard = h_i * h_j
        surface = torch.tensor([jaccard], device=h_i.device, dtype=h_i.dtype)
        features = torch.cat([h_i, h_j, hadamard, surface], dim=-1).unsqueeze(0)
        return self.mlp(features).squeeze()


def jaccard_overlap_matrix(records: list[dict]) -> torch.Tensor:
    """Precompute pairwise argument-surface Jaccard overlap.

    Args:
        records: list of ``EventRecord``-like dicts with ``arguments``.

    Returns:
        [N, N] float tensor where entry (i, j) = |A_i ∩ A_j| / |A_i ∪ A_j|
        over normalised argument-text sets.
    """
    n = len(records)

    def _norm(text: str) -> str:
        return "".join(text.split())

    arg_sets: list[set[str]] = []
    for rec in records:
        args = set()
        for role, values in (rec.get("arguments") or {}).items():
            for value in values or []:
                text = _norm(str(value.get("text", "")))
                if text:
                    args.add(f"{role}:{text}")
        arg_sets.append(args)

    overlap = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                overlap[i, j] = 1.0
                continue
            a_i, a_j = arg_sets[i], arg_sets[j]
            if not a_i or not a_j:
                overlap[i, j] = 0.0
            else:
                overlap[i, j] = len(a_i & a_j) / len(a_i | a_j)
    return overlap
