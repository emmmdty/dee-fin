from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATOR_VERSION = "eval-artifacts-v1.1"
PREDICTION_FORMAT = "canonical-jsonl"
METHOD_NAME = "SARGE"
METHOD_FAMILY = "SARGE"


def write_run_manifest(
    run_root: str | Path,
    *,
    run_id: str,
    dataset: str,
    split: str,
    seed: int,
    command_infer: str | None = None,
    notes: str | None = None,
    repo_root: str | Path | None = None,
    backend: str = "unknown",
    backend_metadata: dict[str, Any] | None = None,
    generation_metadata: dict[str, Any] | None = None,
    limit: int | None = None,
    document_count: int | None = None,
    source_commit: str | None = None,
    model_performance_evidence: bool | None = None,
) -> Path:
    output_path = Path(run_root) / "run_manifest.json"
    payload = build_run_manifest(
        run_id=run_id,
        dataset=dataset,
        split=split,
        seed=seed,
        command_infer=command_infer,
        notes=notes,
        repo_root=repo_root,
        backend=backend,
        backend_metadata=backend_metadata,
        generation_metadata=generation_metadata,
        limit=limit,
        document_count=document_count,
        source_commit=source_commit,
        model_performance_evidence=model_performance_evidence,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def build_run_manifest(
    *,
    run_id: str,
    dataset: str,
    split: str,
    seed: int,
    command_infer: str | None = None,
    notes: str | None = None,
    repo_root: str | Path | None = None,
    backend: str = "unknown",
    backend_metadata: dict[str, Any] | None = None,
    generation_metadata: dict[str, Any] | None = None,
    limit: int | None = None,
    document_count: int | None = None,
    source_commit: str | None = None,
    model_performance_evidence: bool | None = None,
) -> dict[str, Any]:
    metadata = _json_ready_dict(backend_metadata)
    generation = _generation_subset(generation_metadata or metadata)
    git_commit = source_commit or _git_commit(repo_root)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "method_name": METHOD_NAME,
        "method_family": METHOD_FAMILY,
        "dataset_version": dataset,
        "split_version": split,
        "evaluator_version": EVALUATOR_VERSION,
        "prediction_format": PREDICTION_FORMAT,
        "training_view": "evaluator_gold/train",
        "gold_view": f"processed/views/evaluator_gold/{dataset}",
        "seed": int(seed),
        "git_commit": git_commit,
        "git_commit_source": "source_commit" if source_commit else ("repo_root" if git_commit else None),
        "command_train": None,
        "command_infer": command_infer,
        "created_at": _created_at(),
        "backend": backend,
        "notes": notes or f"SARGE inference via {backend}",
    }
    payload["limit"] = int(limit) if limit is not None else None
    if document_count is not None:
        payload["document_count"] = int(document_count)
    if model_performance_evidence is not None:
        payload["model_performance_evidence"] = bool(model_performance_evidence)
    payload.update(_backend_identity_fields(backend=backend, metadata=metadata))
    if metadata:
        payload["backend_metadata"] = metadata
    if generation:
        payload["generation"] = generation
    return payload


def _created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit(repo_root: str | Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _backend_identity_fields(*, backend: str, metadata: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    backend_kind = metadata.get("backend_kind")
    if backend_kind is None:
        backend_lower = backend.lower()
        if "vllm" in backend_lower:
            backend_kind = "vllm"
        elif "qwen" in backend_lower:
            backend_kind = "qwen"
        elif "mock" in backend_lower:
            backend_kind = "mock"
    if backend_kind is not None:
        fields["backend_kind"] = backend_kind
    for key in (
        "base_model",
        "model_path",
        "adapter_path",
        "quantization",
        "double_quantization",
        "compute_dtype",
    ):
        if key in metadata:
            fields[key] = metadata[key]
    if fields.get("backend_kind") == "vllm" and metadata.get("model_path"):
        fields["merged_model_path"] = metadata["model_path"]
    return fields


def _generation_subset(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    generation: dict[str, Any] = {}
    for key in (
        "k_candidates",
        "do_sample",
        "temperature",
        "top_p",
        "repetition_penalty",
        "seed",
        "max_new_tokens",
        "deterministic",
        "use_chat_template",
        "use_response_prefix",
        "response_prefix",
        "prompt_delimiter",
        "output_format",
        "baseline_mode",
        "ablation_profile",
        "max_surface_candidates",
        "candidate_context_chars",
        "candidate_render_mode",
        "enable_candidate_filtering",
        "max_candidates_per_type",
        "dedupe_surface_candidates",
        "drop_low_value_company_fragments",
        "sacd_enabled",
        "sacd_backend",
        "sacd_strict",
    ):
        if key in metadata:
            generation[key] = _json_ready(metadata[key])
    return generation


def _json_ready_dict(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): _json_ready(value) for key, value in payload.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
