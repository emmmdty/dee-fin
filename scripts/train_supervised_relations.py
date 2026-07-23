#!/usr/bin/env python
"""Train the discriminative supervised relation extractor on MAVEN-ERE (server / CUDA).

Builds pair-classification rows from gold mentions (`relations.pairs.pair_examples`),
downsamples the dominant NONE class, and trains a RoBERTa encoder + per-family linear
heads with class-weighted cross-entropy. Saves encoder, tokenizer and heads to
`--output`, which `configs/relations/supervised.yaml` then loads.

This is the v4 Phase A *discriminative* trainer — not `train_relation_extractor.py`,
which is the retained v3 generative LoRA baseline.

Data preparation (`build_training_rows` / `downsample_negatives` / `class_weights`) is
pure Python and unit-tested on CPU; training needs the `llm` extra + a GPU:

    uv run --extra llm python scripts/train_supervised_relations.py \
        --train data/processed/maven_ere/train.jsonl \
        --model roberta-base \
        --output runs/relations/supervised_maven

`train_smoke.jsonl` / `valid_smoke.jsonl` are the small subsets for a quick check.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from finekg.relations.data.maven_ere import load_maven_ere
from finekg.relations.extractor.supervised import FAMILY_SUBTYPES
from finekg.relations.pairs import PairExample, pair_examples


def build_training_rows(docs, max_distance: int | None = None) -> list[PairExample]:
    """Every document's labelled candidate universe, flattened.

    `pair_examples` already carries exactly what a pair classifier trains on
    (endpoint ids + one gold subtype per family, empty labels = negative), and it is
    the same universe evaluation scores against — so the rows *are* its output.
    """
    rows: list[PairExample] = []
    for doc in docs:
        rows.extend(pair_examples(doc, max_distance))
    return rows


def downsample_negatives(
    rows: list[PairExample], ratio: float, seed: int = 13
) -> list[PairExample]:
    """Keep every positive pair, subsample negatives to `ratio` per positive.

    Deterministic for a given seed. Raises when there is no positive at all:
    training on NONE only silently learns the majority class — exactly the failure
    behind the 0.4% causal recall — so this fails loudly instead of hiding it.
    """
    positives = [r for r in rows if r.labels]
    negatives = [r for r in rows if not r.labels]
    if not positives:
        raise ValueError("no positive pairs in training rows -- refusing to train on NONE only")
    keep = min(len(negatives), int(len(positives) * ratio))
    return positives + random.Random(seed).sample(negatives, keep)


def class_weights(rows: list[PairExample], alpha: float = 1.0) -> dict[str, list[float]]:
    """Inverse-frequency weight per label per family, tempered by `alpha`.

    The weight is `(total / (k * count)) ** alpha`: alpha=1 is plain inverse
    frequency, alpha=0 is uniform (off), alpha=0.5 the usual middle ground.

    Tempering matters because the families differ in sparsity by ~39:3.4:1
    (temporal:causal:subevent gold). Full inverse weighting makes the dense
    families over-predict (precision collapses); dropping it entirely buries the
    sparsest one (subevent recall collapses). A single global setting cannot
    satisfy both ends of that range, so the strength is a dial, not a switch.
    """
    weights: dict[str, list[float]] = {}
    for family, subtypes in FAMILY_SUBTYPES.items():
        index = {s: i for i, s in enumerate(subtypes)}
        counts = [0] * len(subtypes)
        for row in rows:
            # No gold label for this family = the negative class. An *unknown*
            # subtype must not be silently folded into NONE — that is how positives
            # go missing — so the lookup raises instead.
            counts[index[row.labels.get(family, "NONE")]] += 1
        total = sum(counts)
        weights[family] = [
            (total / (len(subtypes) * c)) ** alpha if c else 0.0 for c in counts
        ]
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", required=True, type=Path, help="MAVEN-ERE train jsonl")
    parser.add_argument("--model", required=True, type=str, help="base RoBERTa (name or path)")
    parser.add_argument("--output", required=True, type=Path, help="checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--neg-ratio", type=float, default=3.0, help="negatives per positive")
    parser.add_argument(
        "--weight-alpha",
        type=float,
        default=1.0,
        help="class-imbalance dial (PHASE_A ablation): CE weight = inverse_freq ** alpha. "
        "1.0 = plain inverse (dense families over-predict, precision collapses), "
        "0.0 = off (the sparsest family, subevent, gets buried), 0.5 = middle ground.",
    )
    parser.add_argument("--max-distance", type=int, default=None, help="None = document-level")
    parser.add_argument(
        "--max-length", type=int, default=512, help="512 covers the longest sentence (322 tokens)"
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    # torch-only imports stay inside main so the data helpers above import on CPU.
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from finekg.relations.extractor.supervised import (
        PairClassifier,
        _pair_features,
        encode_trigger_reps,
    )

    docs = list(load_maven_ere(args.train))
    rows = downsample_negatives(
        build_training_rows(docs, args.max_distance), args.neg_ratio, args.seed
    )
    weights = class_weights(rows, args.weight_alpha) if args.weight_alpha > 0 else None
    docs_by_id = {d.doc_id: d for d in docs}
    rows_by_doc: dict[str, list[PairExample]] = {}
    for row in rows:
        rows_by_doc.setdefault(row.doc_id, []).append(row)
    print(f"[train] {len(docs)} docs, {len(rows)} rows after downsampling", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(device)
    counts = {fam: len(subs) for fam, subs in FAMILY_SUBTYPES.items()}
    heads = PairClassifier(encoder.config.hidden_size, counts).to(device)
    weight_tensors = (
        {f: torch.tensor(w, device=device) for f, w in weights.items()} if weights else {}
    )
    label_index = {f: {s: i for i, s in enumerate(subs)} for f, subs in FAMILY_SUBTYPES.items()}
    optimiser = torch.optim.AdamW([*encoder.parameters(), *heads.parameters()], lr=args.lr)

    encoder.train()
    heads.train()
    for epoch in range(args.epochs):
        doc_ids = list(rows_by_doc)
        random.Random(args.seed + epoch).shuffle(doc_ids)
        running = 0.0
        for seen, doc_id in enumerate(doc_ids, start=1):
            doc = docs_by_id[doc_id]
            embs = encode_trigger_reps(
                encoder, tokenizer, doc.nodes, doc.doc_text, args.max_length, device
            )
            doc_rows = rows_by_doc[doc_id]
            # One batched pair feature per document: per-pair construction launches
            # a kernel per candidate (thousands in a single document).
            head_emb = torch.stack([embs[r.head_id] for r in doc_rows])
            tail_emb = torch.stack([embs[r.tail_id] for r in doc_rows])
            logits = heads(_pair_features(head_emb, tail_emb))
            loss = torch.zeros((), device=device)
            for family in FAMILY_SUBTYPES:
                target = torch.tensor(
                    [label_index[family][r.labels.get(family, "NONE")] for r in doc_rows],
                    device=device,
                )
                loss = loss + F.cross_entropy(
                    logits[family], target, weight=weight_tensors.get(family)
                )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss)
            if seen % 500 == 0:  # long run: report progress inside the epoch too
                print(
                    f"[train] epoch {epoch} {seen}/{len(doc_ids)} docs "
                    f"running_loss={running / seen:.4f}",
                    flush=True,
                )
        print(f"[train] epoch {epoch} mean_loss={running / max(1, len(doc_ids)):.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    torch.save(heads.state_dict(), args.output / "heads.pt")
    print(f"[train] saved encoder + heads to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
