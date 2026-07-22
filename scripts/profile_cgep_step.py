#!/usr/bin/env python
"""Time one SeDGPL training step before committing GPU hours to a full run.

SeDGPL's cost is not the template pass. `tools.getSentence` encodes *every event's
sentence separately*, padded to `len_arg`, and `parameter.py` fixes batch_size=1.
On the released ESC build that is a mean of 10.3 sentence encodings per instance
(p90 17, max 24), so one instance costs ~12.3 RoBERTa-base forwards of 200 tokens,
not one. This script measures that with synthetic tensors of the real shapes, so
we learn the throughput without first writing the batching glue.

    uv run --extra llm python scripts/profile_cgep_step.py \
        --model-path /data/TJK/Fin-EKG/models/roberta-base --steps 12

Reports seconds/instance and the projected wall time of a 5-fold ESC run.
"""

from __future__ import annotations

import argparse
import time

from finekg.succession.model import TORCH_AVAILABLE, build_sedgpl

# Measured on data/raw/sedgpl_esc/ESCSubWoRe.npy via `succession.linearize`.
ESC_INSTANCES = 1192
ESC_MEAN_SENTENCES = 10.3
ESC_EVENT_TOKENS = 27  # mean event tokens in a template
ESC_ADDED_TOKENS = 1008
CANDIDATES = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=12, help="timed steps after warmup")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="SeDGPL uses 1")
    parser.add_argument("--sentences", type=int, default=round(ESC_MEAN_SENTENCES))
    parser.add_argument("--seq-len", type=int, default=200, help="parameter.py len_arg")
    parser.add_argument("--epochs", type=int, default=10, help="for the projection only")
    parser.add_argument("--folds", type=int, default=5, help="for the projection only")
    parser.add_argument("--instances", type=int, default=ESC_INSTANCES, help="corpus size")
    parser.add_argument("--train-share", type=float, default=0.8, help="fraction trained on")
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("needs the `llm` extra: uv sync --extra llm")

    import torch
    from torch.nn.functional import cross_entropy

    device = torch.device(args.device)
    vocab_size = 50265 + ESC_ADDED_TOKENS
    model = build_sedgpl(args.model_path, vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)

    seq, n_sent, bs = args.seq_len, args.sentences, args.batch_size
    embeddings = model.template_model.roberta.embeddings.word_embeddings

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for _ in range(bs):
            template_ids = torch.randint(0, vocab_size, (1, seq), device=device)
            type_ids = torch.randint(0, vocab_size, (1, seq), device=device)
            sentence_ids = torch.randint(0, vocab_size, (n_sent, seq), device=device)
            mask = torch.ones_like(template_ids)

            # The expensive part: one encoder pass per event sentence.
            sent_out = model.sentence_model.roberta(
                sentence_ids, attention_mask=torch.ones_like(sentence_ids)
            )[0]
            type_out = model.type_model.roberta(type_ids, attention_mask=mask)[0]
            word_emb = embeddings(template_ids)

            positions = torch.arange(1, ESC_EVENT_TOKENS + 1, device=device)
            inst = word_emb[0, positions]
            sent = sent_out[positions % n_sent, 0]
            kind = type_out[0, positions]
            word_emb = word_emb.clone()
            word_emb[0, positions] = model.eece(inst, sent, kind)

            hidden = model.template_model.roberta(attention_mask=mask, inputs_embeds=word_emb)[0]
            mask_emb = hidden[:, 0]

            candidate_ids = torch.randint(0, vocab_size, (args.candidates,), device=device)
            logits = model.scep(mask_emb, model.template_model.lm_head, candidate_ids)
            gold = torch.zeros(1, dtype=torch.long, device=device)
            loss = loss + cross_entropy(logits, gold)
            loss = loss + 0.5 * model.scep.similarity_loss(mask_emb, embeddings(candidate_ids), 0)
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_step = elapsed / args.steps
    per_instance = per_step / bs
    epoch = per_instance * args.instances * args.train_share
    total = epoch * args.epochs * args.folds

    print(f"\ndevice={args.device}  batch_size={bs}  sentences/instance={n_sent}  seq_len={seq}")
    print(f"  {per_step * 1000:8.1f} ms / step")
    print(f"  {per_instance * 1000:8.1f} ms / instance")
    n_train = int(args.instances * args.train_share)
    print(f"  {epoch:8.1f} s  / epoch (~{n_train} train instances)")
    print(f"  {total / 3600:8.2f} h  / {args.folds}-fold x {args.epochs}-epoch run")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {peak:8.2f} GiB peak allocated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
