"""Minimal vLLM smoke: load merged BF16 ckpt, run one real prompt, print output.

Goal: validate vllm 0.8.5 + cu124 + Qwen3 merged ckpt produces non-empty,
structurally plausible JSON without integrating into the full sarge pipeline.

Run on server (sarge_vllm_full env):
  /data/TJK/envs/sarge_vllm_full/bin/python scripts/smoke_vllm_single.py
"""

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

DEFAULT_MERGED = "/data/TJK/DEE/SARGE/runs/merged_models/qwen3_4b_chfinann_ep2_s13"
DEFAULT_PROMPTS = (
    "/data/TJK/DEE/SARGE/runs/sarge_infer_ChFinAnn-Doc2EDAG_dev_20260518T153305Z"
    "/intermediate/getm/prompts.dev.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", default=DEFAULT_MERGED)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--n", type=int, default=1, help="prompts to run")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    raw_prompts = []
    with open(args.prompts) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_prompts.append((row["doc_id"], row["prompt"]))
            if len(raw_prompts) >= args.n:
                break
    print(f"[smoke] loaded {len(raw_prompts)} prompts")

    tok = AutoTokenizer.from_pretrained(args.merged, trust_remote_code=True)
    chat_texts = []
    for _, raw in raw_prompts:
        messages = [{"role": "user", "content": raw}]
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat = chat + '{"events":'
        chat_texts.append(chat)
    print(f"[smoke] chat_text[0] tail: ...{chat_texts[0][-200:]!r}")

    print(f"[smoke] loading vllm.LLM from {args.merged}")
    t0 = time.monotonic()
    llm = LLM(
        model=args.merged,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print(f"[smoke] vllm loaded in {time.monotonic()-t0:.1f}s")

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
    )
    t0 = time.monotonic()
    results = llm.generate(chat_texts, sp)
    elapsed = time.monotonic() - t0
    print(f"[smoke] generated {len(results)} in {elapsed:.1f}s ({elapsed/len(results):.2f}s/doc)")

    for (doc_id, _), result in zip(raw_prompts, results):
        out_text = result.outputs[0].text
        finish = result.outputs[0].finish_reason
        gen_tokens = len(result.outputs[0].token_ids)
        print(f"  [{doc_id}] tokens={gen_tokens} finish={finish}")
        print(f"    head: {out_text[:300]!r}")
        print(f"    tail: ...{out_text[-200:]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
