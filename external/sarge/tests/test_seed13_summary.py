from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "paper" / "exp" / "scripts" / "build_seed13_summary.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_seed13_summary", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_main_tables_are_test_only_and_include_sarge_dueefin() -> None:
    builder = _load_builder()

    document = builder.render_summary(PROJECT_ROOT)

    assert "| ChFinAnn | P | R | F1 | F1(S) | F1(M) |" in document
    assert "| DuEE-Fin | P | R | F1 | F1(S) | F1(M) |" in document
    assert "| SARGE | 84.4 | 87.7 | 86.0 | 89.9 | 81.8 |" in document
    assert "| SARGE | 76.6 | 79.3 | 78.0 | 79.3 | 77.5 |" in document
    assert "SARGE (dev)" not in document
    assert "test-only main comparison tables" in document


def test_missing_values_render_as_dash_not_nan() -> None:
    builder = _load_builder()

    document = builder.render_summary(PROJECT_ROOT)

    assert "| SEELE | - | - | 85.1 | - | - |" in document
    assert "| PTPCG | 71.0 | 61.7 | 66.0 | - | - |" in document
    assert "NaN" not in document


def test_exact_record_is_recomputed_from_counts() -> None:
    builder = _load_builder()

    assert builder.exact_record_f1({"record_exact_match_count": 662, "validated_record_count": 3090}) == 42.8
    assert builder.exact_record_f1({"record_exact_match_count": 2689, "validated_record_count": 9637}) == 55.8
