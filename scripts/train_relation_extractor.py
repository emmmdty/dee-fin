#!/usr/bin/env python
"""Fine-tune the LoRA relation extractor on MAVEN-ERE (server / CUDA).

Turns each document into a supervised example: the all-pairs extraction prompt
as input, the gold relations (as the JSON the parser expects) as target. Trains
a LoRA adapter on a Qwen-style base model, reusing the Chapter-1 stack
(transformers + PEFT). The resulting adapter dir goes into the `adapter_path`
of `configs/relations/llm_grounded_consistent.yaml`.

GPU + the `llm` extra required:
    uv run --extra llm python scripts/train_relation_extractor.py \
        --train data/processed/maven_ere/train.jsonl \
        --model models/Qwen/Qwen3-4B-Instruct-2507 \
        --output runs/relation_extractor_lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finekg.core.schema import EventNode
from finekg.relations.data import load_maven_ere
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.llm import build_relation_prompt
from finekg.relations.rl.dataset import window_document

# Keep synthesized quotes within the grounding reward's quote-length cap
# (configs use max_quote_chars: 60), or SFT would teach quotes the verifier
# refuses to ground.
QUOTE_CHARS = 60


def _gold_quote(doc: RelationDocument, node: EventNode) -> str:
    """A verbatim, groundable quote around the node's trigger.

    The model can only learn to emit verifiable evidence if the SFT targets
    contain it: take the trigger's sentence (sent_id line of `doc_text`) and
    clip a window around the trigger to the grounding length cap.
    """
    if not doc.doc_text:
        return ""
    lines = doc.doc_text.split("\n")
    span = node.trigger_evidence[0] if node.trigger_evidence else None
    if span is None or span.sent_id is None or not (0 <= span.sent_id < len(lines)):
        return ""
    line = lines[span.sent_id]
    pos = line.find(node.trigger) if node.trigger else -1
    if pos < 0:
        return line[:QUOTE_CHARS]
    start = max(0, pos - (QUOTE_CHARS - len(node.trigger)) // 2)
    return line[start : start + QUOTE_CHARS]


def build_supervised_examples(path: Path, window_events: int = 24) -> list[dict[str, str]]:
    """One (prompt, target-JSON) pair per event window from gold relations.

    Windowing mirrors the GRPO dataset (`build_grpo_dataset`), so the SFT
    policy that later anchors the KL term saw the same prompt distribution.
    Targets carry an `evidence_quote` per relation — the groundable behaviour
    the verifier (and the grounding reward) checks for.
    """
    examples: list[dict[str, str]] = []
    for doc in load_maven_ere(path):
        for window in window_document(doc, window_events):
            index = {node.event_id: i for i, node in enumerate(window.nodes)}
            node_by_id = {node.event_id: node for node in window.nodes}
            relations = []
            for edge in window.gold_edges:
                item: dict[str, object] = {
                    "head": index[edge.head_id],
                    "tail": index[edge.tail_id],
                    "type": edge.relation_type.value,
                    "subtype": edge.subtype,
                }
                quote = _gold_quote(window, node_by_id[edge.head_id])
                if quote:
                    item["evidence_quote"] = quote
                relations.append(item)
            target = json.dumps({"relations": relations}, ensure_ascii=False)
            examples.append(
                {
                    "prompt": build_relation_prompt(window.nodes, doc_text=window.doc_text),
                    "target": target,
                }
            )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument(
        "--window-events",
        type=int,
        default=24,
        help="events per prompt window (keep equal to the GRPO curriculum's window_events)",
    )
    args = parser.parse_args()

    # Heavy, GPU-only imports kept local to the entry point.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            example["target"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        input_ids = (prompt_ids + target_ids)[: args.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: args.max_len]
        return {"input_ids": input_ids, "labels": labels}

    examples = [tokenize(e) for e in build_supervised_examples(args.train, args.window_events)]
    # A prompt at/over --max-len leaves no room for target tokens: every label
    # is -100, the example contributes nothing, and a batch of only such
    # examples makes the loss 0/0 = NaN. Drop them loudly instead.
    n_total = len(examples)
    examples = [e for e in examples if any(label != -100 for label in e["labels"])]
    if len(examples) < n_total:
        print(
            f"[train] dropped {n_total - len(examples)}/{n_total} examples whose prompt "
            f"left no target tokens within --max-len {args.max_len}; consider raising "
            f"--max-len or lowering --window-events"
        )
    print(f"[train] {len(examples)} supervised examples")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # A 4B base at --max-len 2048 OOMs a 24GB card on the first backward unless we
    # recompute activations instead of storing the full-depth graph. enable_input_require_grads
    # lets the frozen base propagate gradient into the LoRA params under checkpointing
    # (mirrors gradient_checkpointing: true already used by configs/relations/grpo_rlvr.yaml).
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    # DataCollatorForSeq2Seq pads and PRESERVES the labels above.
    # (DataCollatorForLanguageModeling would silently overwrite them with a
    # clone of input_ids, training on the prompt tokens and discarding the
    # -100 prompt mask.)
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=examples, data_collator=collator
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[train] saved LoRA adapter -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
