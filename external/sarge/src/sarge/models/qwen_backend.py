from __future__ import annotations

import importlib
import inspect
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sarge.data.jsonl import iter_jsonl, write_jsonl
from sarge.utils.gpu_monitor import GpuMemoryMonitor, ResourceMonitorConfig, VramLimitExceeded
from sarge.utils.time_eta import TimingTracker
from sarge.generation.diagnostics import (
    DIAGNOSTIC_VERSION,
    build_generation_diagnostics,
    generation_diagnostic_fields,
)
from sarge.generation.json_stopping import apply_balanced_json_stopping
from sarge.generation.prompt import (
    DEFAULT_GETM_CANDIDATE_RENDER_MODE,
    DEFAULT_GETM_OUTPUT_FORMAT,
    normalize_candidate_render_mode,
    normalize_output_format,
    normalize_prompt_baseline_mode,
)
from sarge.experiments.ablation import resolve_ablation_profile
from sarge.models.sft_dataset import build_sft_training_examples

TRAIN_STACK_REQUIRED_MODULES = ("torch", "transformers", "peft", "accelerate")
TRAIN_STACK_OPTIONAL_MODULES = ("bitsandbytes",)
PREDICT_FORBIDDEN_KEYS = frozenset(
    {
        "gold",
        "gold_path",
        "gold_template",
        "events",
        "events_gold",
        "output",
        "norm_text",
        "empty_roles",
        "event_id",
    }
)
DEFAULT_RESPONSE_PREFIX = '{"events":'
DEFAULT_PROMPT_DELIMITER = "### RESPONSE_JSON"
GENERATION_PROMPT_PREFIX_KEEP_TOKENS = 256


class QwenDependencyError(RuntimeError):
    """Raised when real Qwen execution is requested without the training stack."""


def detect_train_stack() -> dict[str, Any]:
    availability: dict[str, Any] = {"available": True, "modules": {}}
    for module_name in (*TRAIN_STACK_REQUIRED_MODULES, *TRAIN_STACK_OPTIONAL_MODULES):
        try:
            module = importlib.import_module(module_name)
            availability["modules"][module_name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:  # pragma: no cover - environment-dependent import probing
            if module_name in TRAIN_STACK_REQUIRED_MODULES:
                availability["available"] = False
            availability["modules"][module_name] = {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return availability


@dataclass
class QwenGetmBackend:
    config: dict[str, Any] = field(default_factory=dict)
    telemetry: Any | None = None
    _runtime: _QwenRuntime | None = field(default=None, init=False, repr=False)
    _last_generation_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @property
    def parse_options(self) -> dict[str, Any]:
        return _generation_parse_options(self.config)

    @property
    def generation_metadata(self) -> dict[str, Any]:
        return _generation_metadata(self.config, runtime=self._runtime)

    @property
    def last_generation_metadata(self) -> dict[str, Any]:
        return dict(self._last_generation_metadata)

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
        del schema, surface_candidates, slot_plan, candidate_index
        _reject_predict_gold_visible(_document_payload(document))
        generation_cfg = _generation_config(self.config)
        if _dry_run(self.config):
            if generation_cfg["use_response_prefix"]:
                output = "]}" if _compact_json_prefix(generation_cfg["response_prefix"]) == '{"events":[' else "[]}"
            else:
                output = json.dumps({"events": []}, ensure_ascii=False)
            metadata = _dry_run_generation_metadata(
                prompt=prompt,
                output=output,
                generation_cfg=generation_cfg,
                prompt_cfg=_prompt_config(self.config),
            )
            output, self._last_generation_metadata = _apply_generation_stopping(
                output=output,
                metadata=metadata,
                generation_cfg=generation_cfg,
            )
        else:
            _require_real_run(self.config, operation="real Qwen GETM inference")
            runtime = self._runtime or _load_model_for_generation(self.config)
            self._runtime = runtime
            output, metadata = _generate_text_with_metadata(runtime, prompt, self.config)
            self._last_generation_metadata = metadata
        if self.telemetry is not None:
            self.telemetry.record_item(
                item_id=str(_document_payload(document).get("doc_id") or ""),
                item_type="candidate",
            )
        return output


@dataclass(frozen=True)
class _QwenRuntime:
    torch: Any
    tokenizer: Any
    model: Any
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _SimpleTrainingDataset:
    rows: list[dict[str, list[int]]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class _DryRunSftTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        continue_final_message: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str | dict[str, list[int]]:
        rendered = "".join(f"<|{message['role']}|>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        if continue_final_message:
            rendered += "<|continue|>"
        if not tokenize:
            return rendered
        return {"input_ids": [ord(char) for char in rendered]}

    def __call__(self, text: str, *, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


class QwenTelemetryRun:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        output_dir: str | Path,
        operation: str,
        total_items: int | None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.operation = operation
        self.monitor_config = _resource_monitor_config(config)
        self.enabled = self.monitor_config.enabled
        self.real_run = bool(_run_config(config).get("real_run", False))
        self.dry_run = _dry_run(config)
        self.telemetry_dir = self.output_dir / "telemetry"
        self.timing: TimingTracker | None = None
        self.monitor: GpuMemoryMonitor | None = None
        self.monitor_started = False
        self.warnings: list[str] = []
        self.total_items = total_items
        self._started = False

    @property
    def manifest_path(self) -> Path:
        return self.telemetry_dir / "telemetry_manifest.json"

    def start(self) -> QwenTelemetryRun:
        if not self.enabled or self._started:
            return self
        self._started = True
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.timing = TimingTracker(total_items=self.total_items)
        if self.real_run and not self.dry_run:
            try:
                self.monitor = GpuMemoryMonitor(out_dir=self.telemetry_dir, config=self.monitor_config)
                self.monitor.start()
                self.monitor_started = True
            except Exception as exc:  # pragma: no cover - defensive telemetry path
                self.warnings.append(f"GPU monitor failed to start: {type(exc).__name__}: {exc}")
        return self

    def set_total_items(self, total_items: int | None) -> None:
        self.total_items = total_items
        if self.timing is not None:
            self.timing.set_total_items(total_items)

    def record_item(
        self,
        *,
        item_id: str | None = None,
        item_type: str = "item",
        duration_sec: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self.timing is None:
            self.start()
        if self.timing is None:
            return
        try:
            self.timing.record_item(item_id=item_id, item_type=item_type, duration_sec=duration_sec)
        except Exception as exc:  # pragma: no cover - defensive telemetry path
            self.warnings.append(f"Timing telemetry failed: {type(exc).__name__}: {exc}")

    def finish(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        timing_summary: dict[str, Any] | None = None
        gpu_summary: dict[str, Any] | None = None
        vram_error: VramLimitExceeded | None = None
        if self.timing is not None:
            try:
                timing_summary = self.timing.finish(self.telemetry_dir)
            except Exception as exc:  # pragma: no cover - defensive telemetry path
                self.warnings.append(f"Timing telemetry finalization failed: {type(exc).__name__}: {exc}")
        if self.monitor_started and self.monitor is not None:
            try:
                gpu_summary = self.monitor.finish()
                self.warnings.extend(self.monitor.warnings)
                self.monitor.enforce_vram_limit(real_run=self.real_run)
            except VramLimitExceeded as exc:
                vram_error = exc
                self.warnings.append(str(exc))
            except Exception as exc:  # pragma: no cover - defensive telemetry path
                self.warnings.append(f"GPU monitor finalization failed: {type(exc).__name__}: {exc}")
        manifest = {
            "operation": self.operation,
            "telemetry_enabled": self.enabled,
            "real_run": self.real_run,
            "dry_run": self.dry_run,
            "monitor_started": self.monitor_started,
            "resource_monitor": self.monitor_config.to_dict(),
            "timing_summary_path": (
                str(self.telemetry_dir / "timing_summary.json") if timing_summary is not None else None
            ),
            "timing_events_path": (
                str(self.telemetry_dir / "timing_events.jsonl") if timing_summary is not None else None
            ),
            "gpu_memory_samples_path": (
                str(self.telemetry_dir / "gpu_memory_samples.csv") if gpu_summary is not None else None
            ),
            "gpu_memory_summary_path": (
                str(self.telemetry_dir / "gpu_memory_summary.json") if gpu_summary is not None else None
            ),
            "timing_summary": timing_summary,
            "gpu_memory_summary": gpu_summary,
            "warnings": list(dict.fromkeys(self.warnings)),
            "vram_guard_failed": vram_error is not None,
        }
        if extra:
            manifest.update(extra)
        _write_json(self.manifest_path, manifest)
        if vram_error is not None:
            raise vram_error
        return manifest


def start_qwen_telemetry(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    operation: str,
    total_items: int | None,
) -> QwenTelemetryRun:
    return QwenTelemetryRun(config=config, output_dir=output_dir, operation=operation, total_items=total_items).start()


def build_model(config: dict[str, Any]) -> dict[str, Any]:
    """Build or describe the Qwen GETM model.

    In dry-run mode this returns a manifest and does not import torch/transformers.
    In real-run mode it validates dependencies and loads the base model plus an
    optional LoRA adapter for generation.
    """

    manifest = _base_model_manifest(config)
    if _dry_run(config):
        return {**manifest, "dry_run": True, "real_run": False, "train_stack_checked": False}
    _require_real_run(config, operation="real Qwen GETM model loading")
    runtime = _load_model_for_generation(config)
    return {**runtime.manifest, "dry_run": False, "real_run": True}


def train_sft(
    config: dict[str, Any],
    train_data: str | Path | Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train or dry-run Qwen GETM SFT from prebuilt GETM SFT rows."""

    rows = _load_rows(train_data)
    out_dir = Path(output_dir)
    telemetry = start_qwen_telemetry(config, out_dir, operation="train_sft", total_items=len(rows))
    torch_memory_stats: dict[str, Any] | None = None
    try:
        if _dry_run(config):
            for index, row in enumerate(rows, 1):
                telemetry.record_item(item_id=str(row.get("doc_id") or index), item_type="train_row")
            _, label_mask_audit = build_sft_training_examples(
                rows=rows,
                tokenizer=_DryRunSftTokenizer(),
                max_seq_len=int(_training_config(config)["max_seq_len"]),
                config=config,
            )
            manifest = {
                **_base_model_manifest(config),
                "operation": "train_sft",
                "dry_run": True,
                "real_run": False,
                "profile": _profile(config),
                "train_rows": len(rows),
                "max_seq_len": _training_config(config)["max_seq_len"],
                "micro_batch_size": _training_config(config)["micro_batch_size"],
                "gradient_accumulation": _training_config(config)["gradient_accumulation"],
                "gradient_checkpointing": _training_config(config)["gradient_checkpointing"],
                "max_train_steps": _training_config(config)["max_train_steps"],
                "sft_label_mask": {**label_mask_audit, "tokenizer_source": "dry_run_char_tokenizer"},
                "message": "Dry-run only; no Qwen weights, GPU, trainer, or checkpoint were loaded.",
            }
            if telemetry.enabled:
                manifest["telemetry_manifest_path"] = str(telemetry.manifest_path)
            _write_manifests(out_dir, manifest, training=True)
            return {
                **manifest,
                "training_manifest_path": str(out_dir / "training_manifest.json"),
                "backend_manifest_path": str(out_dir / "artifacts" / "backend_manifest.json"),
            }

        _require_real_run(config, operation="real Qwen GETM SFT training")
        _validate_train_rows(rows)
        runtime = _load_model_for_training(config)
        examples, label_mask_audit = build_sft_training_examples(
            rows=rows,
            tokenizer=runtime.tokenizer,
            max_seq_len=int(_training_config(config)["max_seq_len"]),
            config=config,
        )
        if not examples:
            raise RuntimeError("No trainable GETM SFT rows remained after max_seq_len filtering")

        transformers = importlib.import_module("transformers")
        training_cfg = _training_config(config)
        if training_cfg["max_train_steps"] is not None:
            telemetry.set_total_items(int(training_cfg["max_train_steps"]))
        else:
            telemetry.set_total_items(len(examples))
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer_state_dir = out_dir / "artifacts" / "trainer_state"
        training_kwargs = {
            "output_dir": str(trainer_state_dir),
            "overwrite_output_dir": True,
            "num_train_epochs": float(training_cfg["num_train_epochs"]),
            "learning_rate": float(training_cfg["learning_rate"]),
            "per_device_train_batch_size": int(training_cfg["micro_batch_size"]),
            "gradient_accumulation_steps": int(training_cfg["gradient_accumulation"]),
            "logging_steps": int(training_cfg["logging_steps"]),
            "save_strategy": "epoch",
            "save_total_limit": max(1, int(float(training_cfg.get("num_train_epochs", 3)))),
            "report_to": "none",
            "remove_unused_columns": False,
            "gradient_checkpointing": bool(training_cfg["gradient_checkpointing"]),
            "bf16": _compute_dtype(config) == "bf16",
            "fp16": _compute_dtype(config) == "fp16",
            "do_train": True,
            "optim": str(training_cfg["optimizer"]),
        }
        if training_cfg.get("seed") is not None:
            training_kwargs["seed"] = int(training_cfg["seed"])
            training_kwargs["data_seed"] = int(training_cfg["seed"])
        if training_cfg["max_train_steps"] is not None:
            training_kwargs["max_steps"] = int(training_cfg["max_train_steps"])
        training_args = transformers.TrainingArguments(
            **_filter_training_arguments_kwargs(transformers=transformers, kwargs=training_kwargs)
        )
        trainer = transformers.Trainer(
            model=runtime.model,
            args=training_args,
            train_dataset=_SimpleTrainingDataset(examples),
            data_collator=_causal_lm_collator(runtime.tokenizer),
            callbacks=_trainer_callbacks(transformers, telemetry),
        )
        train_result = trainer.train()

        adapter_dir = out_dir / "artifacts" / "model" / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        runtime.model.save_pretrained(adapter_dir)
        runtime.tokenizer.save_pretrained(adapter_dir)
        torch_memory_stats = _write_torch_cuda_memory_stats(runtime.torch, out_dir / "telemetry")

        metrics = getattr(train_result, "metrics", {}) or {}
        manifest = {
            **runtime.manifest,
            "operation": "train_sft",
            "dry_run": False,
            "real_run": True,
            "profile": _profile(config),
            "train_rows": len(rows),
            "train_examples": len(examples),
            "adapter_dir": str(adapter_dir),
            "train_runtime": metrics.get("train_runtime"),
            "train_samples_per_second": metrics.get("train_samples_per_second"),
            "train_steps_per_second": metrics.get("train_steps_per_second"),
            "train_loss": metrics.get("train_loss"),
            "max_train_steps": training_cfg["max_train_steps"],
            "sft_label_mask": label_mask_audit,
            "torch_cuda_memory": torch_memory_stats,
        }
        if telemetry.enabled:
            manifest["telemetry_manifest_path"] = str(telemetry.manifest_path)
        _write_manifests(out_dir, manifest, training=True)
        return {
            **manifest,
            "training_manifest_path": str(out_dir / "training_manifest.json"),
            "backend_manifest_path": str(out_dir / "artifacts" / "backend_manifest.json"),
        }
    finally:
        telemetry.finish(extra={"torch_cuda_memory": torch_memory_stats} if torch_memory_stats else None)


def generate_candidates(
    config: dict[str, Any],
    input_file: str | Path,
    output_dir: str | Path,
    k: int,
) -> dict[str, Any]:
    """Generate raw Qwen GETM outputs from prompt JSONL rows.

    The input file is a prediction artifact and must not include gold-visible
    fields such as `events`, `output`, or `norm_text`.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    rows = _load_rows(input_file)
    for row in rows:
        _reject_predict_gold_visible(row)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry = start_qwen_telemetry(config, out_dir, operation="generate_candidates", total_items=len(rows) * k)
    try:
        backend = QwenGetmBackend(config=config, telemetry=telemetry)
        raw_rows: list[dict[str, Any]] = []
        for row in rows:
            doc_id = str(row.get("doc_id") or "")
            if not doc_id:
                raise ValueError("prediction prompt row missing doc_id")
            prompt = str(row.get("prompt") or "")
            if not prompt:
                raise ValueError(f"prediction prompt row missing prompt: {doc_id}")
            for candidate_index in range(k):
                candidate_id = f"{doc_id}:qwen-getm:{candidate_index}"
                raw_output = backend.generate_one(
                    prompt=prompt,
                    document=row,
                    schema=None,
                    surface_candidates=[],
                    slot_plan=None,
                    candidate_index=candidate_index,
                )
                raw_diagnostics = _raw_generation_diagnostics(
                    raw_output=raw_output,
                    prompt=prompt,
                    surface_candidate_count=_surface_candidate_count(row),
                    generation_cfg=_generation_config(config),
                    prompt_cfg=_prompt_config(config),
                    token_metadata=backend.last_generation_metadata,
                )
                raw_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "doc_id": doc_id,
                        "candidate_index": candidate_index,
                        "backend": "qwen_getm",
                        "raw_output": raw_output,
                        **raw_diagnostics,
                    }
                )

        raw_outputs_path = write_jsonl(out_dir / "raw_outputs.jsonl", raw_rows)
        manifest = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            **_base_model_manifest(config),
            "operation": "generate_candidates",
            "dry_run": _dry_run(config),
            "real_run": bool(_run_config(config).get("real_run", False)),
            "profile": _profile(config),
            "input_file": str(input_file),
            "raw_outputs_path": str(raw_outputs_path),
            "prompt_rows": len(rows),
            "k": int(k),
            "candidate_count": len(raw_rows),
            "gold_visible": False,
            "generation": backend.generation_metadata,
        }
        if telemetry.enabled:
            manifest["telemetry_manifest_path"] = str(telemetry.manifest_path)
        _write_json(out_dir / "generation_manifest.json", manifest)
        return manifest
    finally:
        telemetry.finish()


def _raw_generation_diagnostics(
    *,
    raw_output: str,
    prompt: str,
    surface_candidate_count: int | None,
    generation_cfg: dict[str, Any],
    prompt_cfg: dict[str, Any],
    token_metadata: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = build_generation_diagnostics(
        raw_output=raw_output,
        prompt=prompt,
        surface_candidate_count=surface_candidate_count,
        max_new_tokens=int(generation_cfg["max_new_tokens"]),
        prompt_token_count=_metadata_int(token_metadata.get("prompt_token_count")),
        prompt_token_count_source=_metadata_str(token_metadata.get("prompt_token_count_source")),
        prompt_token_budget=_metadata_int(
            token_metadata.get("prompt_token_budget"),
            prompt_cfg.get("prompt_token_budget"),
        ),
        full_prompt_token_count=_metadata_int(token_metadata.get("full_prompt_token_count")),
        prompt_packing_strategy=_metadata_str(token_metadata.get("prompt_packing_strategy")),
        prompt_prefix_token_keep_count=_metadata_int(token_metadata.get("prompt_prefix_token_keep_count")),
        prompt_suffix_token_keep_count=_metadata_int(token_metadata.get("prompt_suffix_token_keep_count")),
        prompt_middle_token_drop_count=_metadata_int(token_metadata.get("prompt_middle_token_drop_count")),
        prompt_delimiter_present_after_packing=_metadata_bool(
            token_metadata.get("prompt_delimiter_present_after_packing")
        ),
        response_prefix_present_after_packing=_metadata_bool(
            token_metadata.get("response_prefix_present_after_packing")
        ),
        prompt_section_char_counts=_metadata_int_mapping(token_metadata.get("prompt_section_char_counts")),
        prompt_section_token_counts=_metadata_int_mapping(token_metadata.get("prompt_section_token_counts")),
        generated_token_count=_metadata_int(token_metadata.get("generated_token_count")),
        generated_token_count_source=_metadata_str(token_metadata.get("generated_token_count_source")),
        hit_max_new_tokens=_metadata_bool(token_metadata.get("hit_max_new_tokens")),
        hit_max_new_tokens_source=_metadata_str(token_metadata.get("hit_max_new_tokens_source")),
        ended_with_eos=_metadata_bool(token_metadata.get("ended_with_eos")),
        ended_with_eos_source=_metadata_str(token_metadata.get("ended_with_eos_source")),
        ended_with_eos_reason=_metadata_str(token_metadata.get("ended_with_eos_reason")),
    )
    return generation_diagnostic_fields(diagnostics)


def _surface_candidate_count(row: dict[str, Any]) -> int | None:
    candidates = row.get("surface_candidates")
    return len(candidates) if isinstance(candidates, list) else None


def _metadata_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _metadata_int_mapping(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    mapping: dict[str, int] = {}
    for key, raw_value in value.items():
        parsed = _metadata_int(raw_value)
        if parsed is not None:
            mapping[str(key)] = parsed
    return mapping or None


def _metadata_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _metadata_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _trainer_callbacks(transformers: Any, telemetry: QwenTelemetryRun) -> list[Any]:
    if not telemetry.enabled or not hasattr(transformers, "TrainerCallback"):
        return []

    class _StepTimingCallback(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self._last_step = time.monotonic()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            now = time.monotonic()
            telemetry.record_item(
                item_id=str(getattr(state, "global_step", "")),
                item_type="train_step",
                duration_sec=now - self._last_step,
            )
            self._last_step = now
            return control

    return [_StepTimingCallback()]


def _write_torch_cuda_memory_stats(torch: Any, telemetry_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"torch_cuda_available": False}
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return stats
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        allocated = int(cuda.max_memory_allocated())
        reserved = int(cuda.max_memory_reserved())
        stats = {
            "torch_cuda_available": True,
            "max_memory_allocated_bytes": allocated,
            "max_memory_reserved_bytes": reserved,
            "max_memory_allocated_gb": round(allocated / (1024**3), 6),
            "max_memory_reserved_gb": round(reserved / (1024**3), 6),
        }
        try:
            summary = cuda.memory_summary()
            summary_path = telemetry_dir / "torch_cuda_memory_summary.txt"
            summary_path.write_text(summary, encoding="utf-8")
            stats["memory_summary_path"] = str(summary_path)
        except Exception as exc:  # pragma: no cover - optional torch diagnostic
            stats["memory_summary_warning"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive telemetry path
        stats = {"torch_cuda_available": False, "warning": f"{type(exc).__name__}: {exc}"}
    return stats


def _load_model_for_training(config: dict[str, Any]) -> _QwenRuntime:
    modules = _ensure_train_stack(config)
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]

    training_cfg = _training_config(config)
    reproducibility_manifest = _apply_reproducibility_settings(
        torch=torch,
        seed=training_cfg.get("seed"),
        # Training-time strict determinism interacts badly with bitsandbytes
        # 4-bit kernels (no deterministic implementation); we still seed all
        # RNGs but do not flip on use_deterministic_algorithms here.
        deterministic=False,
        warn_only=True,
    )

    qwen_cfg = _qwen_config(config)
    model_path = str(qwen_cfg.get("model_path") or qwen_cfg.get("base_model"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _build_quantization_config(transformers=transformers, torch=torch, config=config)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **_model_dtype_kwargs(transformers=transformers, torch=torch, config=config),
        device_map="auto",
        quantization_config=quantization_config,
        local_files_only=True,
        use_safetensors=True,
    )
    if quantization_config is not None:
        model = peft.prepare_model_for_kbit_training(model)

    lora_cfg = _lora_config(config)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            target_modules=list(lora_cfg["target_modules"]),
        ),
    )
    if bool(_training_config(config)["gradient_checkpointing"]) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return _QwenRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        manifest={
            **_base_model_manifest(config),
            "train_stack": _stack_manifest(modules),
            "reproducibility": reproducibility_manifest,
        },
    )


def _load_model_for_generation(config: dict[str, Any]) -> _QwenRuntime:
    modules = _ensure_train_stack(config)
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    generation_cfg = _generation_config(config)
    reproducibility_manifest = _apply_reproducibility_settings(
        torch=torch,
        seed=generation_cfg["seed"],
        deterministic=bool(generation_cfg["deterministic"]),
        warn_only=bool(generation_cfg["deterministic_warn_only"]),
    )
    qwen_cfg = _qwen_config(config)
    model_path = str(qwen_cfg.get("model_path") or qwen_cfg.get("base_model"))
    adapter_path = qwen_cfg.get("adapter_path")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = _build_quantization_config(transformers=transformers, torch=torch, config=config)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **_model_dtype_kwargs(transformers=transformers, torch=torch, config=config),
        device_map="auto",
        quantization_config=quantization_config,
        local_files_only=True,
        use_safetensors=True,
    )
    model = peft.PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        local_files_only=True,
    ) if adapter_path else base_model
    model.eval()
    warmup_manifest = _warmup_generation_kernels(torch=torch, tokenizer=tokenizer, model=model)
    return _QwenRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        manifest={
            **_base_model_manifest(config),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "train_stack": _stack_manifest(modules),
            "reproducibility": reproducibility_manifest,
            "runtime_environment": _runtime_environment_manifest(
                torch=torch,
                transformers=transformers,
                model=model,
                config=config,
            ),
            "warmup": warmup_manifest,
        },
    )


def _warmup_generation_kernels(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    # Cold-start processes (e.g. infer_checkpoint.py) skip the trainer.train()
    # warmup that train_sft.py provides; the very first model.generate() can
    # then hang on bitsandbytes 4-bit JIT + accelerate hook install + dynamo
    # compile. Force lazy init here, not on the first real document.
    manifest: dict[str, Any] = {"attempted": False, "completed": False}
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        try:
            dynamo_config = getattr(dynamo, "config", None)
            if dynamo_config is not None:
                setattr(dynamo_config, "disable", True)
                setattr(dynamo_config, "suppress_errors", True)
                manifest["dynamo_disabled"] = True
        except Exception as exc:  # pragma: no cover - defensive
            manifest["dynamo_disable_warning"] = f"{type(exc).__name__}: {exc}"
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        manifest["skipped_reason"] = "cuda_unavailable"
        return manifest
    manifest["attempted"] = True
    try:
        t0 = time.monotonic()
        warmup_inputs = tokenizer("warmup", return_tensors="pt")
        warmup_inputs = {
            key: value.to(model.device) for key, value in warmup_inputs.items()
        }
        with torch.inference_mode():
            model.generate(
                **warmup_inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        manifest["completed"] = True
        manifest["warmup_secs"] = round(time.monotonic() - t0, 2)
    except Exception as exc:  # pragma: no cover - defensive warmup
        manifest["warning"] = f"{type(exc).__name__}: {exc}"
    return manifest


def _generate_text(runtime: _QwenRuntime, prompt: str, config: dict[str, Any]) -> str:
    output, _ = _generate_text_with_metadata(runtime, prompt, config)
    return output


def _generate_text_with_metadata(
    runtime: _QwenRuntime,
    prompt: str,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    generation_cfg = _generation_config(config)
    prompt_cfg = _prompt_config(config)
    model_inputs, token_metadata = _tokenize_generation_prompt_with_metadata(runtime.tokenizer, prompt, config)
    input_width = int(model_inputs["input_ids"].shape[-1])
    prompt_token_budget = int(prompt_cfg["prompt_token_budget"])
    if bool(prompt_cfg["fail_on_prompt_token_limit"]) and input_width >= prompt_token_budget:
        raise RuntimeError(
            f"GETM prompt token budget exceeded: prompt_token_count={input_width} "
            f">= prompt_token_budget={prompt_token_budget}"
        )
    model_inputs = {key: value.to(runtime.model.device) for key, value in model_inputs.items()}
    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": int(generation_cfg["max_new_tokens"]),
        "do_sample": bool(generation_cfg["do_sample"]),
        "num_beams": int(generation_cfg["num_beams"]),
        "num_return_sequences": int(generation_cfg["num_return_sequences"]),
        "pad_token_id": runtime.tokenizer.pad_token_id,
        "eos_token_id": runtime.tokenizer.eos_token_id,
        "repetition_penalty": float(generation_cfg["repetition_penalty"]),
    }
    if generation_kwargs["do_sample"]:
        if generation_cfg["temperature"] is not None:
            generation_kwargs["temperature"] = float(generation_cfg["temperature"])
        generation_kwargs["top_p"] = float(generation_cfg["top_p"])
        if generation_cfg["top_k"] is not None:
            generation_kwargs["top_k"] = int(generation_cfg["top_k"])
    if generation_cfg["use_cache"] is not None:
        generation_kwargs["use_cache"] = bool(generation_cfg["use_cache"])
    else:
        # PEFT-wrapped models do not always inherit base config.use_cache; explicit True
        # avoids long-sequence generate degradation when caller leaves it unspecified.
        generation_kwargs["use_cache"] = True
    with runtime.torch.inference_mode():
        generated_ids = runtime.model.generate(**generation_kwargs)
    generated_new_ids = generated_ids[0][input_width:]
    output = runtime.tokenizer.decode(generated_new_ids, skip_special_tokens=True).strip()
    metadata = _generated_token_metadata(
        prompt=prompt,
        output=output,
        input_width=input_width,
        prompt_token_budget=prompt_token_budget,
        generated_new_ids=generated_new_ids,
        eos_token_id=runtime.tokenizer.eos_token_id,
        max_new_tokens=int(generation_cfg["max_new_tokens"]),
        token_metadata=token_metadata,
    )
    return _apply_generation_stopping(output=output, metadata=metadata, generation_cfg=generation_cfg)


def _generated_token_metadata(
    *,
    prompt: str,
    output: str,
    input_width: int,
    prompt_token_budget: int,
    generated_new_ids: Any,
    eos_token_id: Any,
    max_new_tokens: int,
    token_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_token_count = int(len(generated_new_ids))
    ended_with_eos = _ended_with_eos(generated_new_ids, eos_token_id=eos_token_id)
    return {
        **dict(token_metadata or {}),
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "max_new_tokens": int(max_new_tokens),
        "prompt_char_count": len(prompt),
        "raw_output_char_count": len(output),
        "prompt_token_count": int(input_width),
        "prompt_token_count_source": "generation_input_ids_exact",
        "prompt_token_budget": int(prompt_token_budget),
        "generated_token_count": generated_token_count,
        "generated_token_count_source": "generated_ids_exact",
        "hit_max_new_tokens": generated_token_count >= int(max_new_tokens) and ended_with_eos is not True,
        "hit_max_new_tokens_source": "generated_ids_exact",
        "ended_with_eos": ended_with_eos,
        "ended_with_eos_source": "generated_ids_exact",
        "ended_with_eos_reason": None if ended_with_eos is not None else "tokenizer eos_token_id is unavailable",
    }


def _apply_generation_stopping(
    *,
    output: str,
    metadata: dict[str, Any],
    generation_cfg: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    result = apply_balanced_json_stopping(
        output,
        enabled=bool(generation_cfg["enable_balanced_json_stopping"]),
        stop_after_balanced_events_json=bool(generation_cfg["stop_after_balanced_events_json"]),
        response_prefix=generation_cfg["response_prefix"] if generation_cfg["use_response_prefix"] else None,
        response_prefix_used=bool(generation_cfg["use_response_prefix"]),
        hit_max_new_tokens=_metadata_bool(metadata.get("hit_max_new_tokens")),
        ended_with_eos=_metadata_bool(metadata.get("ended_with_eos")),
    )
    updated = {
        **metadata,
        "raw_output": result.raw_output,
        "stopped_output": result.stopped_output,
        "stop_reason": result.stop_reason,
        "balanced_stop_applied": result.balanced_stop_applied,
        "stopped_output_char_count": len(result.stopped_output),
        "enable_balanced_json_stopping": bool(generation_cfg["enable_balanced_json_stopping"]),
        "stop_after_balanced_events_json": bool(generation_cfg["stop_after_balanced_events_json"]),
    }
    return result.stopped_output, updated


def _dry_run_generation_metadata(
    *,
    prompt: str,
    output: str,
    generation_cfg: dict[str, Any],
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "max_new_tokens": int(generation_cfg["max_new_tokens"]),
        "prompt_char_count": len(prompt),
        "raw_output_char_count": len(output),
        "prompt_token_count": None,
        "prompt_token_count_source": None,
        "prompt_token_budget": int(prompt_cfg["prompt_token_budget"]),
        "generated_token_count": None,
        "generated_token_count_source": None,
        "hit_max_new_tokens": False,
        "hit_max_new_tokens_source": "dry_run_synthetic",
        "ended_with_eos": None,
        "ended_with_eos_source": None,
        "ended_with_eos_reason": "dry-run synthetic output has no model EOS or finish reason",
    }


def _ended_with_eos(generated_new_ids: Any, *, eos_token_id: Any) -> bool | None:
    if eos_token_id is None:
        return None
    if len(generated_new_ids) == 0:
        return False
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    last_token = generated_new_ids[-1]
    if hasattr(last_token, "item"):
        last_token = last_token.item()
    return int(last_token) in {int(token_id) for token_id in eos_ids}


def _tokenize_generation_prompt(tokenizer: Any, prompt: str, config: dict[str, Any]) -> Any:
    model_inputs, _ = _tokenize_generation_prompt_with_metadata(tokenizer, prompt, config)
    return model_inputs


def _tokenize_generation_prompt_with_metadata(
    tokenizer: Any,
    prompt: str,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    generation_cfg = _generation_config(config)
    max_length = int(_training_config(config)["max_seq_len"])
    if generation_cfg["use_chat_template"] and hasattr(tokenizer, "apply_chat_template"):
        if generation_cfg["use_response_prefix"]:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": generation_cfg["response_prefix"]},
            ]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
            )
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                continue_final_message=True,
                return_tensors="pt",
                return_dict=True,
                truncation=False,
            )
            return _pack_generation_model_inputs(
                tokenizer,
                model_inputs,
                max_length,
                generation_cfg,
                protocol_suffix_min_tokens=_protocol_suffix_token_count(
                    tokenizer,
                    str(rendered_prompt),
                    generation_cfg,
                ),
            )
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=False,
        )
        return _pack_generation_model_inputs(
            tokenizer,
            model_inputs,
            max_length,
            generation_cfg,
            protocol_suffix_min_tokens=_protocol_suffix_token_count(tokenizer, str(rendered_prompt), generation_cfg),
        )

    fallback_prompt = prompt
    if generation_cfg["use_response_prefix"]:
        fallback_prompt = f"{prompt}\n{generation_cfg['response_prefix']}"
    model_inputs = tokenizer(
        fallback_prompt,
        return_tensors="pt",
        truncation=False,
    )
    return _pack_generation_model_inputs(
        tokenizer,
        model_inputs,
        max_length,
        generation_cfg,
        protocol_suffix_min_tokens=_protocol_suffix_token_count(tokenizer, fallback_prompt, generation_cfg),
    )


def _pack_generation_model_inputs(
    tokenizer: Any,
    model_inputs: Any,
    max_length: int,
    generation_cfg: dict[str, Any],
    *,
    protocol_suffix_min_tokens: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    input_width = _token_width(input_ids)
    kept_indices, metadata = _generation_prompt_keep_indices(
        input_width,
        max_length=max_length,
        prefix_keep_tokens=GENERATION_PROMPT_PREFIX_KEEP_TOKENS,
        protocol_suffix_min_tokens=protocol_suffix_min_tokens,
    )
    if len(kept_indices) != input_width:
        model_inputs = {
            key: _select_token_positions(value, kept_indices, expected_width=input_width)
            for key, value in model_inputs.items()
        }
    packed_ids = _token_ids_to_list(model_inputs["input_ids"])
    packed_text = _decode_token_ids(tokenizer, packed_ids)
    metadata.update(
        {
            "prompt_token_count": len(packed_ids),
            "prompt_token_count_source": "generation_input_ids_exact",
            "prompt_token_budget": int(max_length),
            "prompt_delimiter_present_after_packing": str(generation_cfg["prompt_delimiter"]) in packed_text,
            "response_prefix_present_after_packing": str(generation_cfg["response_prefix"]) in packed_text
            if generation_cfg["use_response_prefix"]
            else None,
        }
    )
    return model_inputs, metadata


def _generation_prompt_keep_indices(
    token_count: int,
    *,
    max_length: int,
    prefix_keep_tokens: int,
    protocol_suffix_min_tokens: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    if token_count <= max_length:
        return list(range(token_count)), {
            "prompt_packing_strategy": "none",
            "full_prompt_token_count": int(token_count),
            "prompt_token_count": int(token_count),
            "prompt_prefix_token_keep_count": int(token_count),
            "prompt_suffix_token_keep_count": 0,
            "prompt_middle_token_drop_count": 0,
        }
    suffix_min = min(max(int(protocol_suffix_min_tokens), 0), int(max_length))
    prefix_keep = min(
        int(prefix_keep_tokens),
        max(int(max_length) // 4, 0),
        max(int(max_length) - suffix_min, 0),
    )
    suffix_keep = max(int(max_length) - prefix_keep, 0)
    if suffix_keep <= 0:
        prefix_keep = 0
        suffix_keep = int(max_length)
    kept_indices = list(range(prefix_keep)) + list(range(token_count - suffix_keep, token_count))
    return kept_indices, {
        "prompt_packing_strategy": "middle_truncate_keep_prefix_suffix",
        "full_prompt_token_count": int(token_count),
        "prompt_token_count": int(max_length),
        "prompt_prefix_token_keep_count": int(prefix_keep),
        "prompt_suffix_token_keep_count": int(suffix_keep),
        "prompt_middle_token_drop_count": int(token_count - max_length),
    }


def _pack_generation_token_ids(
    token_ids: list[int],
    *,
    max_length: int,
    prefix_keep_tokens: int = GENERATION_PROMPT_PREFIX_KEEP_TOKENS,
) -> tuple[list[int], dict[str, Any]]:
    kept_indices, metadata = _generation_prompt_keep_indices(
        len(token_ids),
        max_length=max_length,
        prefix_keep_tokens=prefix_keep_tokens,
    )
    return [token_ids[index] for index in kept_indices], metadata


def _protocol_suffix_token_count(tokenizer: Any, rendered_prompt: str, generation_cfg: dict[str, Any]) -> int:
    start = -1
    delimiter = str(generation_cfg.get("prompt_delimiter") or "")
    if delimiter:
        start = rendered_prompt.rfind(delimiter)
    if start < 0 and generation_cfg.get("use_response_prefix"):
        start = rendered_prompt.rfind(str(generation_cfg.get("response_prefix") or ""))
    if start < 0:
        return 0
    encoded = tokenizer(rendered_prompt[start:], add_special_tokens=False)
    return len(_token_ids_to_list(encoded["input_ids"]))


def _token_width(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if isinstance(value, list) and value and isinstance(value[0], list):
        return len(value[0])
    return len(value)


def _select_token_positions(value: Any, indices: list[int], *, expected_width: int) -> Any:
    try:
        if _token_width(value) != expected_width:
            return value
    except TypeError:
        return value
    if hasattr(value, "shape"):
        return value[..., indices]
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [[row[index] for index in indices] for row in value]
    if isinstance(value, list):
        return [value[index] for index in indices]
    return value


def _token_ids_to_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if len(tensor.shape) > 1:
            tensor = tensor[0]
        return [int(token_id) for token_id in tensor.tolist()]
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [int(token_id) for token_id in value[0]]
    return [int(token_id) for token_id in value]


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    return ""


def _validate_train_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, 1):
        if not row.get("prompt"):
            raise ValueError(f"train row {index} missing prompt")
        if "output" not in row:
            raise ValueError(f"train row {index} missing output")
        if "norm_text" in json.dumps(row, ensure_ascii=False):
            raise ValueError(f"train row {index} contains norm_text; GETM targets must use argument text")


def _causal_lm_collator(tokenizer: Any):
    def collate(rows: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        pad_id = tokenizer.pad_token_id or 0
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad)
            batch["labels"].append(row["labels"] + [-100] * pad)
        torch = importlib.import_module("torch")
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collate


def _build_quantization_config(*, transformers: Any, torch: Any, config: dict[str, Any]) -> Any | None:
    quantization = str(_qwen_config(config).get("quantization", "disabled")).strip().lower()
    if quantization in {"disabled", "none", "false"}:
        return None
    if quantization not in {"4-bit nf4", "4bit", "4-bit", "nf4"}:
        raise ValueError("GETM Qwen quantization must be disabled or 4-bit NF4")
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=bool(_qwen_config(config).get("double_quantization", True)),
        bnb_4bit_compute_dtype=_resolve_dtype(torch, config),
    )


def _ensure_train_stack(config: dict[str, Any]) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    missing: list[str] = []
    for module_name in TRAIN_STACK_REQUIRED_MODULES:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment-dependent
            missing.append(f"{module_name} ({type(exc).__name__}: {exc})")
    if _requires_bitsandbytes(config):
        try:
            modules["bitsandbytes"] = importlib.import_module("bitsandbytes")
        except Exception as exc:  # pragma: no cover - environment-dependent
            missing.append(f"bitsandbytes ({type(exc).__name__}: {exc})")
    if missing:
        raise QwenDependencyError(
            "Real GETM Qwen execution requires the server ML stack. Missing: "
            + "; ".join(missing)
            + ". Install the project [server] extras or use dry-run/mock backend."
        )
    return modules


def _requires_bitsandbytes(config: dict[str, Any]) -> bool:
    quantization = str(_qwen_config(config).get("quantization", "disabled")).strip().lower()
    return quantization in {"4-bit nf4", "4bit", "4-bit", "nf4"}


def _stack_manifest(modules: dict[str, Any]) -> dict[str, Any]:
    return {
        module_name: {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
        }
        for module_name, module in sorted(modules.items())
    }


def _resolve_dtype(torch: Any, config: dict[str, Any]) -> Any:
    dtype = _compute_dtype(config)
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported GETM Qwen compute_dtype: {dtype!r}")


def _model_dtype_kwargs(*, transformers: Any, torch: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Return the dtype keyword accepted by the installed Transformers major.

    Transformers 5 accepts ``dtype``. Transformers 4 still expects
    ``torch_dtype``; passing ``dtype`` leaks through to Qwen3ForCausalLM and
    breaks model construction.
    """
    version = str(getattr(transformers, "__version__", "0"))
    major_text = version.split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError:
        major = 0
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: _resolve_dtype(torch, config)}


def _compute_dtype(config: dict[str, Any]) -> str:
    return str(_qwen_config(config).get("compute_dtype", "bf16")).strip().lower()


def _filter_training_arguments_kwargs(*, transformers: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = inspect.signature(transformers.TrainingArguments.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def _load_rows(source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return list(iter_jsonl(source))
    return [dict(row) for row in source]


def _reject_predict_gold_visible(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in PREDICT_FORBIDDEN_KEYS:
                raise ValueError(f"predict input is gold-visible at {path}.{key_text}")
            _reject_predict_gold_visible(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_predict_gold_visible(child, path=f"{path}[{index}]")


def _document_payload(document: Any) -> dict[str, Any]:
    if hasattr(document, "to_dict"):
        return document.to_dict()
    if isinstance(document, dict):
        return dict(document)
    return {}


def _base_model_manifest(config: dict[str, Any]) -> dict[str, Any]:
    qwen_cfg = _qwen_config(config)
    lora_cfg = _lora_config(config)
    training_cfg = _training_config(config)
    return {
        "backend": "qwen_getm",
        "base_model": str(qwen_cfg.get("base_model", "Qwen/Qwen3-4B-Instruct-2507")),
        "model_path": qwen_cfg.get("model_path"),
        "adapter_path": qwen_cfg.get("adapter_path"),
        "quantization": qwen_cfg.get("quantization", "4-bit NF4"),
        "double_quantization": bool(qwen_cfg.get("double_quantization", True)),
        "compute_dtype": qwen_cfg.get("compute_dtype", "bf16"),
        "lora": {
            "rank": int(lora_cfg["rank"]),
            "alpha": int(lora_cfg["alpha"]),
            "dropout": float(lora_cfg["dropout"]),
            "target_modules": list(lora_cfg["target_modules"]),
        },
        "training": training_cfg,
        "generation": _generation_metadata(config),
    }


def _qwen_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict((config.get("getm") or {}).get("qwen") or {})


def _lora_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _qwen_config(config).get("lora") or {}
    return {
        "rank": int(raw.get("rank", raw.get("r", 16))),
        "alpha": int(raw.get("alpha", 32)),
        "dropout": float(raw.get("dropout", 0.05)),
        "target_modules": tuple(raw.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
    }


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _qwen_config(config).get("training") or {}
    budget = config.get("training_budget") or {}
    max_train_steps = raw.get("max_train_steps", budget.get("max_train_steps"))
    seed_raw = raw.get("seed", budget.get("seed"))
    return {
        "num_train_epochs": float(raw.get("num_train_epochs", 1.0)),
        "learning_rate": float(raw.get("learning_rate", 2e-4)),
        "logging_steps": int(raw.get("logging_steps", 5)),
        "optimizer": str(raw.get("optimizer", "paged_adamw_8bit")),
        "micro_batch_size": int(raw.get("micro_batch_size", budget.get("micro_batch_size", 1))),
        "gradient_accumulation": int(
            raw.get(
                "gradient_accumulation",
                raw.get("gradient_accumulation_steps", budget.get("gradient_accumulation_steps", 8)),
            )
        ),
        "max_seq_len": int(raw.get("max_seq_len", budget.get("max_seq_len", 4096))),
        "gradient_checkpointing": bool(raw.get("gradient_checkpointing", True)),
        "max_train_steps": int(max_train_steps) if max_train_steps is not None else None,
        "seed": int(seed_raw) if seed_raw is not None else None,
    }


def _generation_config(config: dict[str, Any]) -> dict[str, Any]:
    getm_cfg = config.get("getm") or {}
    raw = getm_cfg.get("generation") or {}
    output_format = normalize_output_format(
        getm_cfg.get("output_format", raw.get("output_format", DEFAULT_GETM_OUTPUT_FORMAT))
    )
    use_chat_template = _as_bool(raw.get("use_chat_template", True))
    default_use_response_prefix = output_format != "record_plan"
    use_response_prefix = _as_bool(raw.get("use_response_prefix", default_use_response_prefix))
    do_sample = _as_bool(raw.get("do_sample", False))
    temperature = _optional_float(raw.get("temperature", None))
    default_response_prefix = DEFAULT_RESPONSE_PREFIX if default_use_response_prefix else ""
    response_prefix = str(raw.get("response_prefix", default_response_prefix))
    add_generation_prompt = bool(use_chat_template and not use_response_prefix)
    continue_final_message = bool(use_chat_template and use_response_prefix)
    seed = raw.get("seed")
    top_k = raw.get("top_k")
    use_cache = raw.get("use_cache")
    return {
        "k_candidates": int(raw.get("k_candidates", 4)),
        "output_format": output_format,
        "max_new_tokens": int(raw.get("max_new_tokens", 1024)),
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": float(raw.get("top_p", 1.0)),
        "top_k": int(top_k) if top_k is not None else None,
        "num_beams": int(raw.get("num_beams", 1)),
        "num_return_sequences": int(raw.get("num_return_sequences", 1)),
        "repetition_penalty": float(raw.get("repetition_penalty", 1.05)),
        "seed": int(seed) if seed is not None else None,
        "deterministic": _as_bool(raw.get("deterministic", False)),
        "deterministic_warn_only": _as_bool(raw.get("deterministic_warn_only", True)),
        "record_resolved_generation_config": _as_bool(raw.get("record_resolved_generation_config", False)),
        "use_cache": _as_bool(use_cache) if use_cache is not None else None,
        "use_chat_template": use_chat_template,
        "add_generation_prompt": add_generation_prompt,
        "continue_final_message": continue_final_message,
        "use_response_prefix": use_response_prefix,
        "response_prefix": response_prefix,
        "prompt_delimiter": str(raw.get("prompt_delimiter", DEFAULT_PROMPT_DELIMITER)),
        "enable_balanced_json_stopping": _as_bool(raw.get("enable_balanced_json_stopping", True)),
        "stop_after_balanced_events_json": _as_bool(raw.get("stop_after_balanced_events_json", True)),
    }


def _prompt_config(config: dict[str, Any]) -> dict[str, Any]:
    getm_cfg = config.get("getm") or {}
    raw = getm_cfg.get("prompt") or {}
    max_candidates_per_type = raw.get("max_candidates_per_type")
    candidate_context_chars = raw.get("candidate_context_chars")
    raw_ablation_profile = str(
        raw.get("ablation_profile") or os.environ.get("SARGE_ABLATION_PROFILE") or ""
    ).strip()
    ablation_profile = resolve_ablation_profile(raw_ablation_profile) if raw_ablation_profile else None
    baseline_mode = (
        ablation_profile.prompt_baseline_mode if ablation_profile is not None else raw.get("baseline_mode")
    )
    prompt_config = {
        "max_surface_candidates": int(raw.get("max_surface_candidates", 40)),
        "candidate_context_chars": (
            int(candidate_context_chars) if candidate_context_chars is not None else None
        ),
        "candidate_render_mode": normalize_candidate_render_mode(
            raw.get("candidate_render_mode", DEFAULT_GETM_CANDIDATE_RENDER_MODE)
        ),
        "enable_candidate_filtering": _as_bool(raw.get("enable_candidate_filtering", False)),
        "max_candidates_per_type": (
            int(max_candidates_per_type) if max_candidates_per_type is not None else None
        ),
        "dedupe_surface_candidates": _as_bool(raw.get("dedupe_surface_candidates", False)),
        "drop_low_value_company_fragments": _as_bool(raw.get("drop_low_value_company_fragments", False)),
        "prompt_token_budget": int(raw.get("prompt_token_budget", _training_config(config)["max_seq_len"])),
        "fail_on_prompt_token_limit": _as_bool(raw.get("fail_on_prompt_token_limit", False)),
        "baseline_mode": normalize_prompt_baseline_mode(baseline_mode),
    }
    if ablation_profile:
        prompt_config["ablation_profile"] = ablation_profile.name
    return prompt_config


def _compact_json_prefix(prefix: Any) -> str:
    return "".join(str(prefix or "").split())


def _generation_metadata(config: dict[str, Any], runtime: _QwenRuntime | None = None) -> dict[str, Any]:
    generation_cfg = _generation_config(config)
    metadata = {
        **generation_cfg,
        **_prompt_config(config),
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "chat_template_used": bool(generation_cfg["use_chat_template"]),
        "response_prefix_used": bool(generation_cfg["use_response_prefix"]),
        "prompt_delimiter_used": bool(generation_cfg["prompt_delimiter"]),
        "attn_implementation": _qwen_config(config).get("attn_implementation"),
        "base_model": _qwen_config(config).get("base_model"),
        "model_path": _qwen_config(config).get("model_path"),
        "adapter_path": _qwen_config(config).get("adapter_path"),
        "quantization": _qwen_config(config).get("quantization"),
        "double_quantization": _qwen_config(config).get("double_quantization"),
        "compute_dtype": _qwen_config(config).get("compute_dtype", "bf16"),
    }
    pad_token_id = None
    eos_token_id = None
    if runtime is not None:
        pad_token_id = runtime.tokenizer.pad_token_id
        eos_token_id = runtime.tokenizer.eos_token_id
        metadata["pad_token_id"] = pad_token_id
        metadata["eos_token_id"] = eos_token_id
        metadata["runtime_environment"] = runtime.manifest.get("runtime_environment")
        metadata["reproducibility"] = runtime.manifest.get("reproducibility")
    metadata["resolved_generation_config"] = _resolved_generation_config(
        generation_cfg,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    return metadata


def _resolved_generation_config(
    generation_cfg: dict[str, Any],
    *,
    pad_token_id: Any = None,
    eos_token_id: Any = None,
) -> dict[str, Any]:
    resolved = {
        "max_new_tokens": int(generation_cfg["max_new_tokens"]),
        "do_sample": bool(generation_cfg["do_sample"]),
        "temperature": generation_cfg["temperature"],
        "top_p": float(generation_cfg["top_p"]),
        "top_k": generation_cfg["top_k"],
        "num_beams": int(generation_cfg["num_beams"]),
        "num_return_sequences": int(generation_cfg["num_return_sequences"]),
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "repetition_penalty": float(generation_cfg["repetition_penalty"]),
        "use_cache": generation_cfg["use_cache"],
    }
    return resolved


def _apply_reproducibility_settings(
    *,
    torch: Any,
    seed: int | None,
    deterministic: bool,
    warn_only: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "seed": seed,
        "seed_applied": seed is not None,
        "deterministic": bool(deterministic),
        "deterministic_warn_only": bool(warn_only),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "warnings": [],
    }
    if seed is not None:
        random.seed(int(seed))
        try:
            numpy = importlib.import_module("numpy")
            numpy.random.seed(int(seed))
            manifest["numpy_seed_applied"] = True
            manifest["numpy_version"] = getattr(numpy, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - optional dependency guard
            manifest["numpy_seed_applied"] = False
            manifest["warnings"].append(f"numpy seed skipped: {type(exc).__name__}: {exc}")
        torch.manual_seed(int(seed))
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "manual_seed_all"):
            cuda.manual_seed_all(int(seed))
            manifest["cuda_seed_applied"] = True
        else:
            manifest["cuda_seed_applied"] = False
    if deterministic:
        manifest["cublas_workspace_config_before"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        manifest["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        backends = getattr(torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False
            manifest["cuda_matmul_allow_tf32"] = False
        cudnn_backend = getattr(backends, "cudnn", None) if backends is not None else None
        if cudnn_backend is not None:
            if hasattr(cudnn_backend, "allow_tf32"):
                cudnn_backend.allow_tf32 = False
                manifest["cudnn_allow_tf32"] = False
            if hasattr(cudnn_backend, "benchmark"):
                cudnn_backend.benchmark = False
                manifest["cudnn_benchmark"] = False
            if hasattr(cudnn_backend, "deterministic"):
                cudnn_backend.deterministic = True
                manifest["cudnn_deterministic"] = True
        use_deterministic_algorithms = getattr(torch, "use_deterministic_algorithms", None)
        if callable(use_deterministic_algorithms):
            try:
                use_deterministic_algorithms(True, warn_only=bool(warn_only))
                manifest["use_deterministic_algorithms"] = True
            except TypeError:
                use_deterministic_algorithms(True)
                manifest["use_deterministic_algorithms"] = True
                manifest["warnings"].append("torch.use_deterministic_algorithms does not accept warn_only")
    cuda = getattr(torch, "cuda", None)
    try:
        manifest["cuda_available"] = bool(cuda is not None and cuda.is_available())
        if manifest["cuda_available"] and hasattr(cuda, "get_device_name"):
            manifest["device_name"] = cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment-dependent
        manifest["warnings"].append(f"cuda device inspection failed: {type(exc).__name__}: {exc}")
    return manifest


def _runtime_environment_manifest(
    *,
    torch: Any,
    transformers: Any,
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    qwen_cfg = _qwen_config(config)
    manifest: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "dtype": _compute_dtype(config),
        "attn_implementation": qwen_cfg.get("attn_implementation"),
        "device_map": qwen_cfg.get("device_map", "auto"),
        "model_path": qwen_cfg.get("model_path"),
        "adapter_path": qwen_cfg.get("adapter_path"),
    }
    try:
        manifest["model_device"] = str(getattr(model, "device", None))
    except Exception as exc:  # pragma: no cover - defensive
        manifest["model_device_warning"] = f"{type(exc).__name__}: {exc}"
    cuda = getattr(torch, "cuda", None)
    try:
        manifest["cuda_available"] = bool(cuda is not None and cuda.is_available())
        if manifest["cuda_available"] and hasattr(cuda, "get_device_name"):
            manifest["device_name"] = cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment-dependent
        manifest["cuda_warning"] = f"{type(exc).__name__}: {exc}"
    return manifest


def _generation_parse_options(config: dict[str, Any]) -> dict[str, Any]:
    generation_cfg = _generation_config(config)
    return {
        "response_prefix": generation_cfg["response_prefix"] if generation_cfg["use_response_prefix"] else None,
        "response_prefix_used": bool(generation_cfg["use_response_prefix"]),
        "output_format": generation_cfg["output_format"],
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _resource_monitor_config(config: dict[str, Any]) -> ResourceMonitorConfig:
    return ResourceMonitorConfig.from_mapping(config.get("resource_monitor") or {})


def _run_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("run") or {})


def _profile(config: dict[str, Any]) -> str:
    return str(_run_config(config).get("profile", "local_dry_run"))


def _dry_run(config: dict[str, Any]) -> bool:
    return bool(_run_config(config).get("dry_run", True))


def _require_real_run(config: dict[str, Any], *, operation: str) -> None:
    if not bool(_run_config(config).get("real_run", False)):
        raise RuntimeError(f"{operation} requires explicit --real-run")


def _write_manifests(out_dir: Path, manifest: dict[str, Any], *, training: bool) -> None:
    _write_json(out_dir / ("training_manifest.json" if training else "generation_manifest.json"), manifest)
    _write_json(out_dir / "artifacts" / "backend_manifest.json", manifest)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
