"""vLLM-backed GETM backend with the same interface as QwenGetmBackend.

Loads a merged BF16 ckpt via vllm.LLM and serves generate_one per call.
Reuses qwen_backend's prompt/generation helpers so chat-template, response
prefix, balanced-json stopping, and metadata stay identical between the two
backends.

This backend does not support adapters (PEFT) — merge LoRA into the base
ckpt first via scripts/merge_lora_to_bf16.py and point --model at the
merged dir.

Two modes:
  * batch=1 (default): one prompt per generate() call; safe drop-in,
    ~2-3x faster than HF+bnb via PagedAttention alone.
  * pre-batched: call .preload_prompts(...) before pipeline run; subsequent
    generate_one() calls are O(1) lookups against the prefilled cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from sarge.generation.diagnostics import DIAGNOSTIC_VERSION
from sarge.models.qwen_backend import (
    _apply_generation_stopping,
    _document_payload,
    _dry_run,
    _generation_config,
    _generation_metadata,
    _generation_parse_options,
    _prompt_config,
    _qwen_config,
    _reject_predict_gold_visible,
    _require_real_run,
)

ENV_VLLM_ENFORCE_EAGER = "SARGE_VLLM_ENFORCE_EAGER"
ENV_VLLM_MAX_NUM_SEQS = "SARGE_VLLM_MAX_NUM_SEQS"
ENV_VLLM_MAX_NUM_BATCHED_TOKENS = "SARGE_VLLM_MAX_NUM_BATCHED_TOKENS"


@dataclass
class VllmGetmBackend:
    config: dict[str, Any] = field(default_factory=dict)
    telemetry: Any | None = None
    _llm: Any | None = field(default=None, init=False, repr=False)
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _sampling_params_cls: Any | None = field(default=None, init=False, repr=False)
    _sampling_common_kwargs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _sampling_params_greedy: Any | None = field(default=None, init=False, repr=False)
    _sampling_params_sample: Any | None = field(default=None, init=False, repr=False)
    _guided_decoding: Any | None = field(default=None, init=False, repr=False)
    _base_seed: int = field(default=13, init=False, repr=False)
    _last_generation_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _prefilled_outputs: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def parse_options(self) -> dict[str, Any]:
        return _generation_parse_options(self.config)

    @property
    def generation_metadata(self) -> dict[str, Any]:
        meta = _generation_metadata(self.config, runtime=None)
        meta["backend_kind"] = "vllm"
        meta["vllm_engine"] = True
        engine_cfg = _resolved_vllm_engine_config(self.config)
        meta["vllm_engine_config"] = dict(engine_cfg)
        meta["vllm_dtype"] = engine_cfg["dtype"]
        meta["vllm_gpu_memory_utilization"] = engine_cfg["gpu_memory_utilization"]
        meta["vllm_max_model_len"] = engine_cfg["max_model_len"]
        meta["vllm_enforce_eager"] = engine_cfg["enforce_eager"]
        if engine_cfg["max_num_seqs"] is not None:
            meta["vllm_max_num_seqs"] = engine_cfg["max_num_seqs"]
        if engine_cfg["max_num_batched_tokens"] is not None:
            meta["vllm_max_num_batched_tokens"] = engine_cfg["max_num_batched_tokens"]
        guided = _vllm_guided_decoding_config(self.config)
        meta["sacd_enabled"] = guided is not None
        if guided is not None:
            meta["sacd_backend"] = guided["backend"]
            meta["sacd_strict"] = guided["strict"]
        return meta

    @property
    def last_generation_metadata(self) -> dict[str, Any]:
        return dict(self._last_generation_metadata)

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from transformers import AutoTokenizer

        qwen_cfg = _qwen_config(self.config)
        model_path = str(qwen_cfg.get("model_path") or qwen_cfg.get("base_model"))
        engine_cfg = _resolved_vllm_engine_config(self.config)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        llm_kwargs = {
            "model": model_path,
            "dtype": engine_cfg["dtype"],
            "gpu_memory_utilization": engine_cfg["gpu_memory_utilization"],
            "max_model_len": engine_cfg["max_model_len"],
            "trust_remote_code": True,
            "enforce_eager": engine_cfg["enforce_eager"],
        }
        if engine_cfg["max_num_seqs"] is not None:
            llm_kwargs["max_num_seqs"] = engine_cfg["max_num_seqs"]
        if engine_cfg["max_num_batched_tokens"] is not None:
            llm_kwargs["max_num_batched_tokens"] = engine_cfg["max_num_batched_tokens"]
        self._llm = LLM(**llm_kwargs)

        gen = _generation_config(self.config)
        # vllm SamplingParams equivalent of HF generate config
        common = {
            "max_tokens": int(gen["max_new_tokens"]),
            "repetition_penalty": float(gen["repetition_penalty"]),
        }
        self._sampling_params_cls = SamplingParams
        self._base_seed = int(gen.get("seed") or 13)
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None:
            common["stop_token_ids"] = [eos_id]
        # SACD: build GuidedDecodingParams from the dataset JSON schema
        # supplied via raw config (built by the caller from DatasetSchema).
        guided = _vllm_guided_decoding_config(self.config)
        if guided is not None:
            self._guided_decoding = GuidedDecodingParams(
                json=guided["json_schema"],
                backend=guided["backend"] if guided["backend"] else None,
            )
            common["guided_decoding"] = self._guided_decoding
        self._sampling_common_kwargs = dict(common)
        # Greedy
        self._sampling_params_greedy = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=int(gen.get("seed") or 13),
            **common,
        )
        # Sampling
        self._sampling_params_sample = SamplingParams(
            temperature=float(gen["temperature"]) if gen["temperature"] is not None else 0.7,
            top_p=float(gen.get("top_p") or 0.95),
            seed=int(gen.get("seed") or 13),
            **common,
        )

    def preload_prompts(
        self,
        prompts: list[tuple[str, int, str]],
    ) -> None:
        """Optional: pre-run vllm batched over a list of (doc_id, cand_idx, raw_prompt).

        Result is cached; generate_one looks it up by (doc_id, cand_idx).
        """
        if not prompts:
            return
        self._ensure_loaded()
        chat_texts = [self._render_chat(raw) for _, _, raw in prompts]
        gen = _generation_config(self.config)
        sp = self._sampling_params_sample if gen["do_sample"] else self._sampling_params_greedy
        if gen["do_sample"]:
            sp = [
                self._sampling_params_for_prompt(doc_id=doc_id, candidate_index=cand_idx, generation_cfg=gen)
                for doc_id, cand_idx, _ in prompts
            ]
        results = self._llm.generate(chat_texts, sp, use_tqdm=False)
        for (doc_id, cand_idx, _), result in zip(prompts, results):
            completion = result.outputs[0]
            self._prefilled_outputs[(doc_id, int(cand_idx))] = {
                "text": completion.text,
                "token_count": int(len(completion.token_ids)),
                "ended_with_eos": completion.finish_reason == "stop",
            }

    def generate_one(
        self,
        *,
        prompt: str,
        document: Any,
        schema: Any,
        surface_candidates: list[Any],
        slot_plan: Any,
        candidate_index: int,
    ) -> str:
        del schema, surface_candidates, slot_plan
        _reject_predict_gold_visible(_document_payload(document))
        generation_cfg = _generation_config(self.config)
        if _dry_run(self.config):
            output = json.dumps({"events": []}, ensure_ascii=False)
            metadata = self._fallback_metadata(prompt=prompt, output=output)
            output, self._last_generation_metadata = _apply_generation_stopping(
                output=output,
                metadata=metadata,
                generation_cfg=generation_cfg,
            )
            return output
        _require_real_run(self.config, operation="real vLLM GETM inference")

        doc_id = str(_document_payload(document).get("doc_id") or "")
        cache_key = (doc_id, int(candidate_index))
        if cache_key in self._prefilled_outputs:
            cached = self._prefilled_outputs[cache_key]
            output = self._wrap_output(
                raw=str(cached["text"]),
                generation_cfg=generation_cfg,
                prompt=prompt,
                token_count=int(cached.get("token_count") or 0),
                ended_with_eos=bool(cached.get("ended_with_eos")),
            )
            if self.telemetry is not None:
                self.telemetry.record_item(item_id=doc_id, item_type="candidate")
            return output

        self._ensure_loaded()
        chat_text = self._render_chat(prompt)
        sp = self._sampling_params_for_prompt(
            doc_id=doc_id,
            candidate_index=candidate_index,
            generation_cfg=generation_cfg,
        )
        results = self._llm.generate([chat_text], sp, use_tqdm=False)
        completion = results[0].outputs[0]
        output = self._wrap_output(
            raw=completion.text,
            generation_cfg=generation_cfg,
            prompt=prompt,
            token_count=int(len(completion.token_ids)),
            ended_with_eos=completion.finish_reason == "stop",
            vllm_result=results[0],
        )
        if self.telemetry is not None:
            self.telemetry.record_item(item_id=doc_id, item_type="candidate")
        return output

    def _render_chat(self, raw_prompt: str) -> str:
        messages = [{"role": "user", "content": raw_prompt}]
        chat = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen = _generation_config(self.config)
        if gen["use_response_prefix"] and gen["response_prefix"]:
            chat = chat + str(gen["response_prefix"])
        return chat

    def _sampling_params_for_prompt(
        self,
        *,
        doc_id: str,
        candidate_index: int,
        generation_cfg: dict[str, Any],
    ) -> Any:
        if self._sampling_params_cls is None:
            raise RuntimeError("vLLM sampling params class not initialized")
        if not generation_cfg["do_sample"]:
            if self._sampling_params_greedy is not None:
                return self._sampling_params_greedy
            return self._sampling_params_cls(
                temperature=0.0,
                top_p=1.0,
                seed=self._base_seed,
                **self._sampling_common_kwargs,
            )
        return self._sampling_params_cls(
            temperature=float(generation_cfg["temperature"]) if generation_cfg["temperature"] is not None else 0.7,
            top_p=float(generation_cfg.get("top_p") or 0.95),
            seed=_stable_candidate_seed(self._base_seed, doc_id, candidate_index),
            **self._sampling_common_kwargs,
        )

    def _wrap_output(
        self,
        *,
        raw: str,
        generation_cfg: dict[str, Any],
        prompt: str,
        token_count: int | None = None,
        ended_with_eos: bool | None = None,
        vllm_result: Any | None = None,
    ) -> str:
        # Recover the full "events" body by prepending the response prefix
        # the model continued from.
        if generation_cfg["use_response_prefix"] and generation_cfg["response_prefix"]:
            output = str(generation_cfg["response_prefix"]) + raw
        else:
            output = raw
        if vllm_result is not None:
            token_count = int(len(vllm_result.outputs[0].token_ids))
            ended_with_eos = vllm_result.outputs[0].finish_reason == "stop"
        token_count = int(token_count or 0)
        max_new_tokens = int(generation_cfg["max_new_tokens"])
        metadata = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "max_new_tokens": max_new_tokens,
            "prompt_char_count": len(prompt),
            "raw_output_char_count": len(output),
            "prompt_token_count": 0,
            "prompt_token_count_source": "vllm_unavailable",
            "prompt_token_budget": int(_prompt_config(self.config)["prompt_token_budget"]),
            "generated_token_count": token_count,
            "generated_token_count_source": "vllm_completion_output_token_ids",
            "hit_max_new_tokens": token_count >= max_new_tokens and ended_with_eos is not True,
            "hit_max_new_tokens_source": "vllm_completion_output_token_ids",
            "ended_with_eos": ended_with_eos,
            "ended_with_eos_source": "vllm_finish_reason",
            "ended_with_eos_reason": None,
            "raw_output": output,
        }
        output, self._last_generation_metadata = _apply_generation_stopping(
            output=output,
            metadata=metadata,
            generation_cfg=generation_cfg,
        )
        return output

    def _fallback_metadata(self, *, prompt: str, output: str) -> dict[str, Any]:
        return {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "max_new_tokens": int(_generation_config(self.config)["max_new_tokens"]),
            "prompt_char_count": len(prompt),
            "raw_output_char_count": len(output),
            "prompt_token_count": 0,
            "prompt_token_count_source": "dry_run",
            "prompt_token_budget": int(_prompt_config(self.config)["prompt_token_budget"]),
            "generated_token_count": 0,
            "generated_token_count_source": "dry_run",
            "hit_max_new_tokens": False,
            "hit_max_new_tokens_source": "dry_run",
            "ended_with_eos": True,
            "ended_with_eos_source": "dry_run",
            "ended_with_eos_reason": None,
            "raw_output": output,
        }


def _stable_candidate_seed(base_seed: int, doc_id: str, candidate_index: int) -> int:
    payload = f"{int(base_seed)}:{doc_id}:{int(candidate_index)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _resolved_vllm_engine_config(config: dict[str, Any]) -> dict[str, Any]:
    qwen_cfg = _qwen_config(config)
    dtype = _normalize_vllm_dtype(str(qwen_cfg.get("compute_dtype", "bfloat16")))
    return {
        "dtype": dtype,
        "gpu_memory_utilization": float(qwen_cfg.get("gpu_memory_utilization", 0.80)),
        "max_model_len": int(qwen_cfg.get("max_model_len", 8192)),
        "enforce_eager": _env_bool(ENV_VLLM_ENFORCE_EAGER, default=False),
        "max_num_seqs": _env_positive_int(ENV_VLLM_MAX_NUM_SEQS),
        "max_num_batched_tokens": _env_positive_int(ENV_VLLM_MAX_NUM_BATCHED_TOKENS),
    }


def _normalize_vllm_dtype(dtype: str) -> str:
    normalized = dtype.lower()
    if normalized in ("bf16", "bfloat16"):
        return "bfloat16"
    if normalized in ("fp16", "float16"):
        return "float16"
    return dtype


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def _env_positive_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _vllm_guided_decoding_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return compact vLLM guided-decoding config from raw generation fields."""
    raw = ((config.get("getm") or {}).get("generation") or {})
    schema = raw.get("sacd_json_schema")
    if not schema:
        return None
    return {
        "json_schema": schema,
        "backend": raw.get("sacd_backend"),
        "strict": bool(raw.get("sacd_strict", False)),
    }
