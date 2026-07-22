"""End-to-end pipeline smoke test.

Stages 3 DuEE-Fin-dev500 documents from the copied data snapshot into
SARGE canonical layout,
runs the full inference pipeline with the mock GETM backend, and asserts
that the canonical prediction file is well-formed (doc_id, events,
event_type, arguments, role, text). No GPU / no Qwen weights / no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sarge.data.canonical import (
    CANONICAL_ARGUMENT_KEYS,
    CANONICAL_DOCUMENT_KEYS,
    CANONICAL_EVENT_RECORD_KEYS,
)
from sarge.data.staging import stage_dataset
from sarge.data.schema import DatasetSchema
from sarge.pipeline.manifest import build_run_manifest
from sarge.pipeline.infer import InferenceResult, run_inference
from sarge.slot_planning.plan import SlotPlanDocument
from sarge.surface_memory.types import SurfaceCandidate

PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def staging_dir(tmp_path: Path) -> Path:
    return tmp_path / "staging"


@pytest.fixture
def out_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _stage_duee_fin(staging: Path, *, train_limit: int = 30, dev_limit: int = 3) -> None:
    stage_dataset(
        dataset="DuEE-Fin-dev500",
        processed_root=PROCESSED_ROOT,
        output_root=staging,
        splits=("train",),
        limit=train_limit,
    )
    stage_dataset(
        dataset="DuEE-Fin-dev500",
        processed_root=PROCESSED_ROOT,
        output_root=staging,
        splits=("dev",),
        limit=dev_limit,
    )


@pytest.mark.skipif(
    not (PROCESSED_ROOT / "DuEE-Fin-dev500" / "dev.jsonl").is_file(),
    reason="data/DuEE-Fin-dev500/dev.jsonl not present",
)
def test_pipeline_runs_end_to_end_with_mock_backend(staging_dir: Path, out_root: Path) -> None:
    _stage_duee_fin(staging_dir, train_limit=30, dev_limit=3)
    result: InferenceResult = run_inference(
        dataset="DuEE-Fin-dev500",
        split="dev",
        data_root=staging_dir,
        out_root=out_root,
        limit=3,
        seed=13,
        k=4,
    )

    assert result.prediction_path.is_file(), "canonical prediction file missing"
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((result.run_root / "diagnostics" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "MockGetmBackend"
    assert manifest["backend_kind"] == "mock"
    assert manifest["model_performance_evidence"] is False
    assert summary["model_performance_evidence"] is False
    assert "mock_backend_notice" in summary
    rows = [json.loads(line) for line in result.prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3, f"expected 3 prediction rows, got {len(rows)}"

    # Every prediction conforms to the frozen canonical schema.
    for row in rows:
        assert set(row.keys()) <= CANONICAL_DOCUMENT_KEYS | {"doc_id", "events"}
        assert "doc_id" in row and isinstance(row["doc_id"], str)
        assert "events" in row and isinstance(row["events"], list)
        for event in row["events"]:
            assert set(event.keys()) <= CANONICAL_EVENT_RECORD_KEYS
            assert isinstance(event["event_type"], str)
            assert isinstance(event["arguments"], dict)
            for role, values in event["arguments"].items():
                assert isinstance(role, str)
                assert isinstance(values, list)
                for value in values:
                    assert set(value.keys()) <= CANONICAL_ARGUMENT_KEYS
                    assert isinstance(value["text"], str)


@pytest.mark.skipif(
    not (PROCESSED_ROOT / "DuEE-Fin-dev500" / "dev.jsonl").is_file(),
    reason="data/DuEE-Fin-dev500/dev.jsonl not present",
)
def test_pipeline_records_real_backend_provenance(staging_dir: Path, out_root: Path) -> None:
    _stage_duee_fin(staging_dir, train_limit=10, dev_limit=2)
    result = run_inference(
        dataset="DuEE-Fin-dev500",
        split="dev",
        data_root=staging_dir,
        out_root=out_root,
        limit=2,
        seed=13,
        k=1,
        command_infer="python scripts/infer_checkpoint.py --source-commit abc123",
        source_commit="abc123",
        backend=_EmptyRealBackend(),
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((result.run_root / "diagnostics" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"] == "abc123"
    assert manifest["git_commit_source"] == "source_commit"
    assert manifest["backend"] == "_EmptyRealBackend"
    assert manifest["backend_kind"] == "qwen"
    assert manifest["model_path"] == "/models/qwen"
    assert manifest["adapter_path"] == "/runs/adapter"
    assert manifest["quantization"] == "4-bit NF4"
    assert manifest["generation"]["k_candidates"] == 1
    assert manifest["generation"]["do_sample"] is False
    assert manifest["limit"] == 2
    assert manifest["document_count"] == 2
    assert manifest["model_performance_evidence"] is True
    assert summary["backend"] == "_EmptyRealBackend"
    assert summary["model_performance_evidence"] is True
    assert "mock_backend_notice" not in summary


def test_run_manifest_always_contains_limit_key() -> None:
    payload = build_run_manifest(
        run_id="run-1",
        dataset="DuEE-Fin-dev500",
        split="dev",
        seed=13,
        backend="MockGetmBackend",
        model_performance_evidence=False,
        document_count=3,
    )

    assert "limit" in payload
    assert payload["limit"] is None


def test_run_manifest_keeps_compact_sacd_generation_metadata() -> None:
    payload = build_run_manifest(
        run_id="run-sacd",
        dataset="DuEE-Fin-dev500",
        split="dev",
        seed=13,
        backend="VllmGetmBackend",
        model_performance_evidence=True,
        document_count=3,
        generation_metadata={
            "k_candidates": 1,
            "do_sample": False,
            "sacd_enabled": True,
            "sacd_backend": "xgrammar",
            "sacd_strict": True,
            "sacd_json_schema": {"type": "object"},
        },
    )

    assert payload["generation"]["sacd_enabled"] is True
    assert payload["generation"]["sacd_backend"] == "xgrammar"
    assert payload["generation"]["sacd_strict"] is True
    assert "sacd_json_schema" not in payload["generation"]


@pytest.mark.skipif(
    not (PROCESSED_ROOT / "DuEE-Fin-dev500" / "dev.jsonl").is_file(),
    reason="data/DuEE-Fin-dev500/dev.jsonl not present",
)
def test_staging_writes_expected_schema_shape(staging_dir: Path) -> None:
    _stage_duee_fin(staging_dir, train_limit=10, dev_limit=3)
    schema_path = staging_dir / "DuEE-Fin-dev500" / "schema.json"
    assert schema_path.is_file()
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "DuEE-Fin-dev500"
    assert isinstance(payload["event_types"], list)
    assert all("event_type" in entry and "roles" in entry for entry in payload["event_types"])


class _EmptyRealBackend:
    @property
    def generation_metadata(self) -> dict[str, Any]:
        return {
            "backend_kind": "qwen",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "model_path": "/models/qwen",
            "adapter_path": "/runs/adapter",
            "quantization": "4-bit NF4",
            "compute_dtype": "bf16",
            "k_candidates": 1,
            "do_sample": False,
            "temperature": None,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "seed": 13,
            "max_new_tokens": 1024,
        }

    def generate_one(
        self,
        *,
        prompt: str,
        document: Any,
        schema: DatasetSchema,
        surface_candidates: list[SurfaceCandidate],
        slot_plan: SlotPlanDocument,
        candidate_index: int,
    ) -> str:
        del prompt, document, schema, surface_candidates, slot_plan, candidate_index
        return '{"events":[]}'
