"""Train LRD scorer on GPU using CPU-precomputed argument span embeddings.

Loads mean-pooled RoBERTa span vectors from a pre-encode cache (created
by ``preencode_lrd.py``) and trains only the projection MLP, role
embeddings, and pairwise scorer on GPU.  The frozen RoBERTa backbone is
never loaded — training runs in seconds per epoch instead of minutes.

The current objective is pairwise BCE only.  A record-level reward term is
intentionally disabled until a principled differentiable/estimable clustering
reward is implemented and validated.

Example:
    python scripts/train_lrd.py \\
        --preencoded runs/lrd/preencoded.pt \\
        --schema data/processed/ChFinAnn-Doc2EDAG/schema.json \\
        --out runs/lrd/train_seed13 \\
        --epochs 5 --seed 13
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from sarge.data.schema import load_schema
from sarge.models.encoder import ArgumentEncodingConfig
from sarge.postprocess.lrd_planner import LRDConfig, LRDPlanner


class PreencodedDataset:
    """Wraps a pre-encode cache for GPU training."""

    def __init__(self, path: str | Path):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        self.role_vocabulary: list[str] = cache["role_vocabulary"]
        self.hidden_dim: int = cache["hidden_dim"]
        self.role_embedding_dim: int = cache["role_embedding_dim"]
        self.docs: dict[str, dict] = cache["docs"]
        self.doc_ids: list[str] = sorted(self.docs.keys())

        total_args = 0
        total_pairs = 0
        for d in self.docs.values():
            for r in d["records"]:
                total_args += len(r["arg_pooled"])
            total_pairs += len(d.get("pairs", []))
        print(
            f"Loaded preencoded cache: {len(self.doc_ids)} docs, "
            f"{total_args} arguments, {total_pairs} pairs"
        )

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        did = self.doc_ids[idx]
        return {"doc_id": did, **self.docs[did]}


def _collate(batch: list[dict]) -> list[dict]:
    return list(batch)


def _build_model(config: LRDConfig, schema: Any, device: torch.device) -> LRDPlanner:
    """Create an LRDPlanner but never load the frozen RoBERTa."""
    planner = LRDPlanner(config, schema)
    planner.to(device)
    planner.train()
    return planner


def _forward_doc(
    planner: LRDPlanner,
    doc: dict,
    device: torch.device,
) -> torch.Tensor | None:
    """Reconstruct record embeddings from cached pooled spans."""
    encoder = planner.encoder
    cached_records = doc["records"]
    n_rec = len(cached_records)

    if n_rec < 2:
        return None

    record_embs: list[torch.Tensor] = []
    role_masks: list[torch.Tensor] = []
    n_roles = len(encoder.role_vocabulary)

    for rec in cached_records:
        arg_pooled = rec.get("arg_pooled") or []
        role_indices = rec.get("role_indices") or []
        role_mask = torch.tensor(rec.get("role_mask") or [0.0] * n_roles, device=device)

        if arg_pooled:
            pooled_t = torch.tensor(arg_pooled, device=device, dtype=torch.float32)
            roles_t = torch.tensor(role_indices, device=device, dtype=torch.long)
            # project_span: role_emb + projection (trainable)
            arg_embs = encoder.project_span(pooled_t, roles_t)  # [M, D]
            rec_emb = encoder.record_embedding(arg_embs, role_mask)  # [D + n_roles]
        else:
            # Record with no encodable arguments (rare edge case).
            rec_emb = torch.zeros(
                encoder.config.hidden_dim + n_roles, device=device
            )
        record_embs.append(rec_emb)
        role_masks.append(role_mask)

    return torch.stack(record_embs)  # [N, D+n_roles]


def _train_step(
    planner: LRDPlanner,
    batch: list[dict],
    device: torch.device,
    *,
    reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the validated pairwise BCE objective.

    ``reward_weight`` is accepted for CLI compatibility, but the reward term
    is disabled instead of using an unvalidated proxy.
    """
    pair_losses: list[torch.Tensor] = []
    del reward_weight

    for doc in batch:
        record_embs = _forward_doc(planner, doc, device)
        if record_embs is None:
            continue

        pairs = doc.get("pairs") or []
        n_rec = len(doc["records"])

        # Pairwise BCE.
        for pair in pairs:
            i, j, label = pair["i"], pair["j"], pair["label"]
            if i < n_rec and j < n_rec:
                logit = planner.scorer.score_pair(record_embs[i], record_embs[j], 0.0)
                target = torch.tensor(float(label), device=device)
                pair_losses.append(
                    F.binary_cross_entropy_with_logits(logit, target)
                )

    if not pair_losses:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    pair_loss = torch.stack(pair_losses).mean()
    reward_loss = torch.tensor(0.0, device=device)
    total = pair_loss
    return total, pair_loss.detach(), reward_loss.detach()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preencoded", required=True, help=".pt cache from preencode_lrd.py")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--reward-weight",
        type=float,
        default=0.0,
        help="Reserved for a future validated record-level reward; currently ignored.",
    )
    args = parser.parse_args()

    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load schema.
    schema_path = Path(args.schema).resolve()
    dataset_name = schema_path.parent.name
    data_root = schema_path.parent.parent
    schema = load_schema(dataset_name, data_root=data_root)

    # Load precomputed dataset.
    dataset = PreencodedDataset(args.preencoded)
    role_vocab = dataset.role_vocabulary

    # Build model (NO RoBERTa loaded — projection + role_embed + scorer only).
    encoder_cfg = ArgumentEncodingConfig(
        model_path="__unused__",  # RoBERTa is never loaded on GPU
        hidden_dim=dataset.hidden_dim,
        role_embedding_dim=dataset.role_embedding_dim,
    )
    lrd_cfg = LRDConfig(
        encoder_config=encoder_cfg,
        role_vocabulary=role_vocab,
    )
    planner = _build_model(lrd_cfg, schema, device)
    print(f"Model on {device}: {sum(p.numel() for p in planner.parameters()):,} trainable params")

    # DataLoader.
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )

    optimizer = torch.optim.AdamW(
        list(planner.scorer.parameters())
        + list(planner.encoder.projection.parameters())
        + list(planner.encoder.role_embed.parameters())
        + [planner.merge_thresholds],
        lr=args.lr,
    )

    print(
        f"train docs: {len(dataset)}  batches: {len(train_loader)}  "
        f"device: {device}  epochs: {args.epochs}"
    )

    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            loss, pair_loss, reward_loss = _train_step(
                planner, batch, device, reward_weight=args.reward_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(train_loader), 1)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}")

    train_secs = time.monotonic() - t0
    print(f"training done in {train_secs:.0f}s")

    # Save checkpoint.
    ckpt_path = out_dir / "checkpoints" / "lrd_planner.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scorer": planner.scorer.state_dict(),
            "encoder_projection": planner.encoder.projection.state_dict(),
            "encoder_role_embed": planner.encoder.role_embed.state_dict(),
            "merge_thresholds": planner.merge_thresholds,
            "config": lrd_cfg,
            "role_vocabulary": role_vocab,
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")

    summary = {
        "train_docs": len(dataset),
        "epochs": args.epochs,
        "train_secs": round(train_secs, 1),
        "checkpoint": str(ckpt_path),
        "seed": args.seed,
        "preencoded": args.preencoded,
        "objective": "pairwise_bce",
        "reward_term_enabled": False,
        "reward_weight_requested": args.reward_weight,
    }
    (out_dir / "summary_train.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
