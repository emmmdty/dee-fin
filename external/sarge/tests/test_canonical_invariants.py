"""Boundary invariants for SARGE.

Ensures the SARGE package keeps:
- a canonical surface-prediction schema with stable public keys;
- a clean module skeleton with no forbidden imports (training stacks must
  not leak into data/canonical contracts);
- no gold-matching or evaluator code embedded in the prediction path
  (gold matching belongs to the evaluator package).
"""

from __future__ import annotations

import ast
from pathlib import Path

SARGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "sarge"
DATA_ROOT = SARGE_ROOT / "data"

EXPECTED_PACKAGE_FILES = {
    SARGE_ROOT / "__init__.py",
    DATA_ROOT / "__init__.py",
    DATA_ROOT / "canonical.py",
    DATA_ROOT / "schema.py",
    SARGE_ROOT / "postprocess" / "__init__.py",
    SARGE_ROOT / "postprocess" / "rule_planner.py",
    SARGE_ROOT / "models" / "__init__.py",
    SARGE_ROOT / "generation" / "__init__.py",
    SARGE_ROOT / "selection" / "__init__.py",
    SARGE_ROOT / "surface_memory" / "__init__.py",
    SARGE_ROOT / "slot_planning" / "__init__.py",
    SARGE_ROOT / "pipeline" / "__init__.py",
    SARGE_ROOT / "evaluation" / "__init__.py",
    SARGE_ROOT / "utils" / "__init__.py",
}


def _python_files(root: Path) -> list[Path]:
    assert root.exists(), f"missing sarge root: {root}"
    return sorted(root.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_sarge_package_skeleton_exists() -> None:
    missing = sorted(str(path) for path in EXPECTED_PACKAGE_FILES if not path.exists())
    assert missing == []


def test_sarge_does_not_import_sibling_dee_data_or_evaluator() -> None:
    forbidden = {"dee_eval", "dee_data"}
    offenders = {
        str(path): sorted(_imported_roots(path) & forbidden)
        for path in _python_files(SARGE_ROOT)
        if _imported_roots(path) & forbidden
    }
    assert offenders == {}


def test_sarge_data_contracts_do_not_import_training_stacks() -> None:
    forbidden = {"torch", "transformers", "peft", "accelerate", "bitsandbytes"}
    offenders = {
        str(path): sorted(_imported_roots(path) & forbidden)
        for path in _python_files(DATA_ROOT)
        if _imported_roots(path) & forbidden
    }
    assert offenders == {}


def test_sarge_data_contracts_do_not_embed_evaluator_or_gold_matching_logic() -> None:
    forbidden_markers = {
        "score_event_records",
        "DocEEArgTableMicroF1",
        "EventRecordExactF1",
        "gold_matching",
        "match_gold",
        "offset_as_gold",
        "gold_offset",
        "span_label",
    }
    offenders: dict[str, list[str]] = {}
    for path in _python_files(DATA_ROOT):
        source = path.read_text(encoding="utf-8")
        hits = sorted(marker for marker in forbidden_markers if marker in source)
        if hits:
            offenders[str(path)] = hits
    assert offenders == {}


def test_sarge_canonical_schema_contains_only_surface_prediction_keys() -> None:
    from sarge.data.canonical import (
        CANONICAL_ARGUMENT_KEYS,
        CANONICAL_DOCUMENT_KEYS,
        CANONICAL_EVENT_RECORD_KEYS,
    )

    assert CANONICAL_DOCUMENT_KEYS == frozenset({"doc_id", "events"})
    assert CANONICAL_EVENT_RECORD_KEYS == frozenset({"event_type", "arguments"})
    assert CANONICAL_ARGUMENT_KEYS == frozenset({"text"})


def test_sarge_canonical_format_version_matches_project_wire_format() -> None:
    """Wire format string must be the canonical SARGE prediction format."""
    from sarge.data.canonical import CANONICAL_PREDICTION_FORMAT_VERSION

    assert CANONICAL_PREDICTION_FORMAT_VERSION == "sarge.canonical_prediction.v1"


def test_no_stage_named_files_in_sarge_package() -> None:
    """SARGE source must not contain stage-marker filenames (per AGENTS.md)."""
    stage_patterns = ("_v21", "_v22", "_v23", "_r3", "_r4", "_r5", "_r6", "_r7", "_r8", "_phase", "_smoke_", "_s4_")
    offenders = []
    for path in _python_files(SARGE_ROOT):
        name = path.name.lower()
        for pattern in stage_patterns:
            if pattern in name:
                offenders.append(str(path))
                break
    assert offenders == [], f"stage-named files leaked into src/sarge/: {offenders}"


def test_generation_scope_guard_uses_functional_names_not_legacy_stage_markers() -> None:
    source = (SARGE_ROOT / "generation" / "scope_guard.py").read_text(encoding="utf-8")
    forbidden_markers = ("sage_v2", "phase4", "Phase 4", "_s4_", "S4")
    offenders = [marker for marker in forbidden_markers if marker in source]
    assert offenders == []
