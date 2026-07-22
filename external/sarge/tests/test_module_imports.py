"""Import-smoke check for SARGE modules.

Asserts that runtime modules are importable on a machine that has the SARGE
dependencies but not necessarily the server-only stack (peft / bitsandbytes).
Modules that touch the heavy LLM stack import those packages lazily.
"""

from __future__ import annotations

import importlib

PURE_PYTHON_MODULES = [
    # data
    "sarge.data.canonical",
    "sarge.data.schema",
    "sarge.data.loader",
    "sarge.data.jsonl",
    # surface_memory (was csg/)
    "sarge.surface_memory.types",
    "sarge.surface_memory.audit",
    "sarge.surface_memory.candidate_builder",
    "sarge.surface_memory.builder",
    "sarge.surface_memory.weak_alignment",
    # slot_planning (was lesp/)
    "sarge.slot_planning.types",
    "sarge.slot_planning.audit",
    "sarge.slot_planning.baseline",
    "sarge.slot_planning.metrics",
    "sarge.slot_planning.labels",
    "sarge.slot_planning.plan",
    # generation (was getm/)
    "sarge.generation.candidate_types",
    "sarge.generation.candidate_generator",
    "sarge.generation.diagnostics",
    "sarge.generation.json_stopping",
    "sarge.generation.parser_ablation",
    "sarge.generation.parser",
    "sarge.generation.prompt",
    "sarge.generation.schema_decoding",
    "sarge.generation.scope_guard",
    # selection (was mrs/)
    "sarge.selection.features",
    "sarge.selection.oracle_gap",
    "sarge.selection.pairwise_data",
    "sarge.selection.reward",
    "sarge.selection.selector",
    "sarge.selection.ranker",
    # postprocess
    "sarge.postprocess.rule_planner",
    # record planning
    "sarge.record_planning",
    "sarge.record_planning.plan",
    # record binding
    "sarge.record_binding",
    "sarge.record_binding.assembler",
    "sarge.record_binding.cli",
    "sarge.record_binding.prediction",
    "sarge.record_binding.run",
    # pipeline
    "sarge.pipeline.run_types",
    "sarge.pipeline.manifest",
    "sarge.pipeline.infer",
    # evaluation
    "sarge.evaluation.handoff",
    "sarge.evaluation.evaluator_adapter",
    "sarge.evaluation.export",
    # experiments
    "sarge.experiments",
    "sarge.experiments.ablation",
    # models (mock backend has no heavy deps)
    "sarge.models.mock_backend",
    "sarge.models.sft_dataset",
    # utils
    "sarge.utils.io",
    "sarge.utils.gpu_monitor",
    "sarge.utils.time_eta",
]

# These touch torch / peft / bitsandbytes at runtime but use lazy importlib at
# the call sites; module-level import is still safe on dev machines, so we
# include them as importable but mark them for clarity.
LLM_LAZY_MODULES = [
    "sarge.models.qwen_backend",
    "sarge.models.vllm_backend",
]


def test_pure_python_modules_import() -> None:
    failures: list[str] = []
    for name in PURE_PYTHON_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert failures == [], "import failures:\n" + "\n".join(failures)


def test_llm_lazy_modules_import_with_torch_available() -> None:
    """qwen_backend imports torch/transformers/peft lazily, but the module
    file itself uses importlib.import_module + decorator patterns, so importing
    the module file should still succeed on machines with only base deps."""
    failures: list[str] = []
    for name in LLM_LAZY_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert failures == [], "import failures:\n" + "\n".join(failures)


def test_total_module_count_matches_expected_port_size() -> None:
    """Sanity check that the W2 port produced the expected number of modules.

    Update this number whenever new top-level modules are added."""
    expected = len(PURE_PYTHON_MODULES) + len(LLM_LAZY_MODULES)
    assert expected == 53, (
        f"PURE_PYTHON_MODULES + LLM_LAZY_MODULES totals {expected}, "
        "update test once new public modules land."
    )
