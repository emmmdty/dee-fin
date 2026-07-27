"""GRPO trainer assembly for the relation extractor (server / CUDA).

The only module in the relations RL stack that touches trl / peft / torch —
everything upstream (rewards, dataset, adapter) is pure CPU. Imports are lazy
so this module loads on a CPU-only box; building a trainer requires the `llm`
and `rl` extras.

Rollout backends, in fallback order (config ``rollout.backend``):

- ``vllm_server``    `trl vllm-serve` on a dedicated GPU (two-card default);
- ``vllm_colocate``  vLLM shares the training GPU (single-card, tight);
- ``hf``             transformers `generate` — no vllm coupling, always works.

GRPOConfig fields are filtered against the installed TRL signature, so a
patch-level TRL bump inside the pin (>=0.17,<0.19) cannot crash the entry
point over a renamed knob; dropped knobs are reported instead.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_grpo_trainer", "resolve_rollout_kwargs"]


def resolve_rollout_kwargs(rollout: dict[str, Any]) -> dict[str, Any]:
    """Map the config's rollout section onto TRL GRPOConfig fields."""
    backend = str(rollout.get("backend", "vllm_server"))
    if backend == "hf":
        return {"use_vllm": False}
    if backend == "vllm_server":
        return {
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_host": str(rollout.get("vllm_server_host", "0.0.0.0")),
            "vllm_server_port": int(rollout.get("vllm_server_port", 8000)),
        }
    if backend == "vllm_colocate":
        return {
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": float(
                rollout.get("vllm_gpu_memory_utilization", 0.25)
            ),
        }
    raise ValueError(f"unknown rollout backend {backend!r} (vllm_server/vllm_colocate/hf)")


def build_grpo_trainer(
    model_name: str,
    rows: list[dict[str, Any]],
    reward_fn: Any,
    cfg: dict[str, Any],
    output_dir: str,
    sft_adapter_path: str | None = None,
    max_steps: int | None = None,
) -> Any:
    """Assemble a TRL GRPOTrainer over prompt rows and a verifiable reward.

    `cfg` is the ``relations_rl`` config section (``grpo:``, ``lora:``,
    ``rollout:``). With `sft_adapter_path` the policy warm-starts from the SFT
    LoRA adapter (recommended); otherwise a fresh LoRA is attached.
    """
    import inspect

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    grpo: dict[str, Any] = dict(cfg.get("grpo", {}))
    if max_steps is not None:
        grpo["max_steps"] = int(max_steps)
    grpo.setdefault("logging_steps", 10)
    grpo.setdefault("save_strategy", "no")  # phases checkpoint explicitly
    grpo.setdefault("report_to", [])
    grpo.update(resolve_rollout_kwargs(dict(cfg.get("rollout", {}))))

    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = sorted(k for k in grpo if k not in accepted)
    if dropped:
        print(f"[grpo] dropping knobs unknown to this TRL version: {dropped}")
    args = GRPOConfig(output_dir=output_dir, **{k: v for k, v in grpo.items() if k in accepted})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    peft_config = None
    if sft_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
    else:
        lora = dict(cfg.get("lora", {}))
        default_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", default_targets)),
            task_type="CAUSAL_LM",
        )

    return GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
