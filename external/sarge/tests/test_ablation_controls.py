from __future__ import annotations

from pathlib import Path

from sarge.data.schema import DatasetSchema
from sarge.generation.prompt import build_getm_prompt_result
from sarge.slot_planning.plan import EventSlot, SlotPlanDocument
from sarge.surface_memory.types import SurfaceCandidate


def _schema() -> DatasetSchema:
    return DatasetSchema(
        dataset_id="fixture",
        schema_dataset="fixture",
        schema_path=Path("/dev/null"),
        canonical_version=None,
        event_roles={"质押": ("质押方", "质押物")},
        role_to_event_types={"质押方": ("质押",), "质押物": ("质押",)},
    )


def _document() -> dict:
    return {
        "doc_id": "doc-1",
        "dataset": "fixture",
        "dataset_id": "fixture",
        "split": "dev",
        "content": "甲公司质押乙公司股票。",
    }


def _candidate() -> SurfaceCandidate:
    return SurfaceCandidate(
        candidate_id="doc-1:csg:1",
        doc_id="doc-1",
        surface="甲公司",
        context="甲公司质押乙公司股票",
        chunk_id="chunk_0000",
    )


def _slot_plan() -> SlotPlanDocument:
    return SlotPlanDocument(
        doc_id="doc-1",
        dataset="fixture",
        slots=[
            EventSlot(
                event_type="质押",
                slot_id=0,
                count_confidence=0.8,
                role_prior={"质押方": 0.9},
            )
        ],
    )


def test_role_safe_surface_only_ablation_renders_candidates_without_slot_plan() -> None:
    result = build_getm_prompt_result(
        dataset="fixture",
        schema=_schema(),
        document=_document(),
        surface_candidates=[_candidate()],
        slot_plan=_slot_plan(),
        baseline_mode="role_safe_surface_only",
    )

    assert "[Schema]" in result.prompt
    assert "[Surface Candidates]" in result.prompt
    assert "甲公司" in result.prompt
    assert "[Event Slot Plan]" not in result.prompt
    assert result.prompt_metadata["baseline_mode"] == "role_safe_surface_only"
    assert result.prompt_metadata["selected_surface_candidate_count"] == 1


def test_role_safe_slot_plan_only_ablation_renders_slot_plan_without_candidates() -> None:
    result = build_getm_prompt_result(
        dataset="fixture",
        schema=_schema(),
        document=_document(),
        surface_candidates=[_candidate()],
        slot_plan=_slot_plan(),
        baseline_mode="role_safe_slot_plan_only",
    )

    assert "[Schema]" in result.prompt
    assert "[Surface Candidates]" not in result.prompt
    assert "[Event Slot Plan]" in result.prompt
    assert "质押 slot_id=0" in result.prompt
    assert result.prompt_metadata["baseline_mode"] == "role_safe_slot_plan_only"
    assert result.prompt_metadata["selected_surface_candidate_count"] == 0


def test_apply_ablation_profile_sets_prompt_mode_without_mutating_input() -> None:
    from sarge.experiments.ablation import apply_ablation_profile

    base_config = {
        "getm": {
            "prompt": {
                "baseline_mode": "role_safe_surface_memory",
                "max_surface_candidates": 20,
            }
        }
    }

    updated = apply_ablation_profile(base_config, "no_surface_memory")

    assert updated["getm"]["prompt"]["baseline_mode"] == "role_safe_slot_plan_only"
    assert updated["getm"]["prompt"]["ablation_profile"] == "no_surface_memory"
    assert base_config["getm"]["prompt"]["baseline_mode"] == "role_safe_surface_memory"


def test_ablation_prompt_metadata_is_manifest_visible() -> None:
    from sarge.experiments.ablation import apply_ablation_profile
    from sarge.models.qwen_backend import QwenGetmBackend
    from sarge.pipeline.manifest import build_run_manifest

    config = apply_ablation_profile(
        {
            "run": {"dry_run": True},
            "getm": {
                "prompt": {"max_surface_candidates": 20},
                "qwen": {"model_path": "/models/qwen"},
                "generation": {"k_candidates": 1, "do_sample": False, "seed": 13},
            },
        },
        "no_slot_plan",
    )

    payload = build_run_manifest(
        run_id="run-ablation",
        dataset="DuEE-Fin-dev500",
        split="dev",
        seed=13,
        backend="QwenGetmBackend",
        model_performance_evidence=True,
        document_count=1,
        generation_metadata=QwenGetmBackend(config=config).generation_metadata,
    )

    assert payload["generation"]["ablation_profile"] == "no_slot_plan"
    assert payload["generation"]["baseline_mode"] == "role_safe_surface_only"
    assert payload["generation"]["max_surface_candidates"] == 20


def test_ablation_profile_env_selects_prompt_mode(monkeypatch) -> None:
    from sarge.models.qwen_backend import QwenGetmBackend

    monkeypatch.setenv("SARGE_ABLATION_PROFILE", "no_surface_memory")
    backend = QwenGetmBackend(
        config={
            "run": {"dry_run": True},
            "getm": {
                "prompt": {"max_surface_candidates": 20},
                "qwen": {"model_path": "/models/qwen"},
                "generation": {"k_candidates": 1, "do_sample": False, "seed": 13},
            },
        }
    )

    metadata = backend.generation_metadata

    assert metadata["ablation_profile"] == "no_surface_memory"
    assert metadata["baseline_mode"] == "role_safe_slot_plan_only"
