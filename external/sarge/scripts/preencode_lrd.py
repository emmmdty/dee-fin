"""CPU pre-encoding: freeze RoBERTa → mean-pooled argument spans for GPU training.

Runs the frozen Chinese-RoBERTa-wwm-ext backbone on CPU to pre-compute
mean-pooled span vectors for every argument in the LRD training pairs.
The output cache is consumed by ``train_lrd.py --preencoded <path>``,
which only trains the projection + scorer on GPU (seconds per epoch).

Example:
    python scripts/preencode_lrd.py \\
        --pairs runs/lrd/chfinann_train_pairs_text.jsonl \\
        --roberta models/chinese-roberta-wwm-ext_safetensors \\
        --schema data/processed/ChFinAnn-Doc2EDAG/schema.json \\
        --out runs/lrd/preencoded.pt
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

from sarge.data.schema import DatasetSchema, load_schema
from sarge.models.encoder import ArgumentEncodingConfig, ArgumentEncoder


def _char_to_token_map(text: str, tokenizer: Any, input_ids: torch.Tensor) -> list[tuple[int, int]]:
    """Map each token in *input_ids* to its (start_char, end_char) in *text*."""
    mapping: list[tuple[int, int]] = []
    offset = 0
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    for token in tokens:
        if token in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            mapping.append((0, 0))
            continue
        clean = token.lstrip("##")
        pos = text.find(clean, offset)
        if pos == -1:
            mapping.append((offset, offset))
        else:
            end = pos + len(clean)
            mapping.append((pos, end))
            offset = end
    return mapping


def preencode_doc(
    doc: dict,
    encoder: ArgumentEncoder,
) -> dict:
    """Pre-encode all argument spans in a single document on CPU.

    Returns:
        {doc_id: {"records": [...], "pairs": [...]}}
    """
    doc_id = doc["doc_id"]
    doc_text = doc.get("text") or ""
    records = doc.get("records") or []

    if not doc_text or not records:
        return {}

    tokenizer = encoder.tokenizer
    full_tokenized = tokenizer(
        doc_text, return_tensors="pt", truncation=True,
        max_length=encoder.config.max_seq_len,
    )
    full_input_ids = full_tokenized["input_ids"]  # [1, L] on CPU

    encoded_records: list[dict] = []

    for rec in records:
        arg_pooled: list[list[float]] = []
        role_indices: list[int] = []
        role_mask = [0.0] * len(encoder.role_vocabulary)

        for role, values in rec.get("arguments", {}).items():
            if role not in encoder.role_to_idx:
                continue
            role_idx = encoder.role_to_idx[role]
            role_mask[role_idx] = 1.0

            for value in values:
                arg_text = str(value if isinstance(value, str) else value.get("text", "")).strip()
                if not arg_text:
                    continue

                # Locate argument span.
                pos = doc_text.find(arg_text)
                if pos == -1:
                    continue

                cw = encoder.config.context_window
                start = max(0, pos - cw)
                end = min(len(doc_text), pos + len(arg_text) + cw)
                window = doc_text[start:end]
                arg_start_in_window = pos - start
                arg_end_in_window = arg_start_in_window + len(arg_text)

                tokenized = tokenizer(
                    window, return_tensors="pt", truncation=True,
                    max_length=encoder.config.max_seq_len,
                )
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # Build span mask on CPU (cheap) — char_to_token map is a
                # text-side operation that does not need to touch the model.
                char_to_token = _char_to_token_map(window, tokenizer, input_ids[0])
                span_mask = torch.zeros_like(input_ids, dtype=torch.float)
                for tok_idx, (c_start, c_end) in enumerate(char_to_token):
                    if c_start >= arg_start_in_window and c_end <= arg_end_in_window:
                        span_mask[0, tok_idx] = 1.0

                if span_mask.sum() == 0:
                    span_mask[0, 0] = 1.0  # fall back to CLS

                # Run frozen RoBERTa + mean-pool (no role emb, no projection).
                model_device = next(encoder.encoder.parameters()).device
                pooled = encoder.encode_span_raw(
                    input_ids.to(model_device),
                    attention_mask.to(model_device),
                    span_mask.to(model_device),
                )
                # pooled is [1, hidden_dim]
                arg_pooled.append(pooled[0].tolist())
                role_indices.append(role_idx)

        if arg_pooled:
            encoded_records.append({
                "arg_pooled": arg_pooled,
                "role_indices": role_indices,
                "role_mask": role_mask,
            })
        else:
            # Record with no encodable arguments → zero vector.
            encoded_records.append({
                "arg_pooled": [],
                "role_indices": [],
                "role_mask": role_mask,
            })

    return {
        doc_id: {
            "records": encoded_records,
            "pairs": doc.get("pairs") or [],
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="enriched training pairs jsonl (with text field)")
    parser.add_argument("--roberta", required=True, help="path to Chinese-RoBERTa-wwm-ext safetensors")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True, help="output .pt cache file")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--device", default="cpu",
                        help="device for the frozen RoBERTa forward pass (cpu or cuda)")
    args = parser.parse_args()

    # Resolve schema.
    schema_path = Path(args.schema).resolve()
    dataset_name = schema_path.parent.name
    data_root = schema_path.parent.parent
    schema = load_schema(dataset_name, data_root=data_root)
    role_vocab = sorted(schema.unique_roles)

    # Build encoder; move projection params to the requested device so
    # ``_ensure_encoder`` co-locates the frozen RoBERTa backbone there.
    encoder_cfg = ArgumentEncodingConfig(
        model_path=args.roberta,
        hidden_dim=768,
        role_embedding_dim=64,
    )
    encoder = ArgumentEncoder(encoder_cfg, role_vocab)
    encoder.to(torch.device(args.device))
    encoder._ensure_encoder()
    print(f"RoBERTa loaded on {args.device}, role vocab size: {len(role_vocab)}")

    # Load pairs.
    docs: list[dict] = []
    with Path(args.pairs).open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.max_docs is not None and idx >= args.max_docs:
                break
            if line.strip():
                docs.append(json.loads(line))
    print(f"Loaded {len(docs)} training docs")

    # Pre-encode all docs.
    cache: dict[str, dict] = {}
    t0 = time.monotonic()
    for idx, doc in enumerate(docs):
        result = preencode_doc(doc, encoder)
        cache.update(result)
        if (idx + 1) % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = (idx + 1) / elapsed
            eta = (len(docs) - idx - 1) / rate
            print(f"  pre-encoded {idx + 1}/{len(docs)} docs  ({rate:.1f} docs/s, ETA {eta:.0f}s)")

    elapsed = time.monotonic() - t0
    print(f"Pre-encoding done: {len(cache)} docs in {elapsed:.0f}s ({len(docs)/elapsed:.1f} docs/s)")

    # Save cache.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "role_vocabulary": role_vocab,
        "hidden_dim": encoder_cfg.hidden_dim,
        "role_embedding_dim": encoder_cfg.role_embedding_dim,
        "docs": cache,
    }, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    # Count total argument embeddings.
    total_args = sum(
        sum(len(r["arg_pooled"]) for r in d["records"])
        for d in cache.values()
    )
    print(f"Total pooled argument spans cached: {total_args}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
