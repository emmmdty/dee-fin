"""Merge a PEFT LoRA adapter into the base model and save as BF16 full ckpt.

The output is a self-contained HF model directory consumable by vllm
(no --enable-lora, no bnb quantization). Run on a server with the base
model and adapter both accessible; needs ~10 GB peak GPU/CPU RAM for
a 4B model in BF16.

Usage:
  python scripts/merge_lora_to_bf16.py \
    --base /data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507 \
    --adapter /data/TJK/DEE/SARGE/runs/sarge_sft_ChFinAnn_Doc2EDAG_s13_ep2_gpu1/artifacts/model/adapter \
    --out /data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507-sarge-ChFinAnn-ep2-merged
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model directory")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory (contains adapter_config.json)")
    parser.add_argument("--out", required=True, help="Output directory for merged BF16 model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for merge math; cuda is much faster")
    args = parser.parse_args()

    base = Path(args.base)
    adapter = Path(args.adapter)
    out = Path(args.out)
    if not base.is_dir():
        print(f"ERROR: base not found: {base}", file=sys.stderr)
        return 2
    if not (adapter / "adapter_config.json").is_file():
        print(f"ERROR: adapter_config.json missing in: {adapter}", file=sys.stderr)
        return 2
    if out.exists():
        print(f"ERROR: output already exists: {out}", file=sys.stderr)
        return 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge] base    = {base}")
    print(f"[merge] adapter = {adapter}")
    print(f"[merge] out     = {out}")
    print(f"[merge] device  = {args.device}")
    print(f"[merge] torch   = {torch.__version__}  cuda_available={torch.cuda.is_available()}")

    print("[merge] loading base in BF16 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device if args.device == "cuda" else None,
        local_files_only=True,
        use_safetensors=True,
    )

    print("[merge] loading adapter ...")
    model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=False)

    print("[merge] merge_and_unload ...")
    merged = model.merge_and_unload()
    merged.eval()

    print(f"[merge] saving merged model to {out} ...")
    out.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")

    print("[merge] saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, local_files_only=True)
    tokenizer.save_pretrained(str(out))

    print("[merge] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
