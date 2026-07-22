from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from sarge.models.qwen_backend import _generation_config, _model_dtype_kwargs
from sarge.models.vllm_backend import (
    VllmGetmBackend,
    _stable_candidate_seed,
    _vllm_guided_decoding_config,
)


def _config(*, do_sample: bool = False, sacd: bool = False) -> dict:
    generation = {
        "k_candidates": 4,
        "max_new_tokens": 1024,
        "do_sample": do_sample,
        "temperature": 0.7 if do_sample else None,
        "top_p": 0.95 if do_sample else 1.0,
        "repetition_penalty": 1.05,
        "use_chat_template": True,
        "use_response_prefix": True,
        "response_prefix": '{"events":',
        "seed": 13,
    }
    if sacd:
        generation.update(
            {
                "sacd_json_schema": {
                    "type": "object",
                    "properties": {"events": {"type": "array"}},
                    "required": ["events"],
                },
                "sacd_backend": "xgrammar",
                "sacd_strict": True,
            }
        )
    return {
        "run": {"dry_run": False, "real_run": True},
        "getm": {
            "backend": "vllm",
            "output_format": "minimal_text",
            "prompt": {"prompt_token_budget": 4096},
            "qwen": {"model_path": "/tmp/not-loaded"},
            "generation": generation,
        },
    }


def test_generation_config_enables_balanced_json_stopping_by_default() -> None:
    generation = _generation_config(_config())
    assert generation["enable_balanced_json_stopping"] is True
    assert generation["stop_after_balanced_events_json"] is True


def test_qwen_model_dtype_kwargs_match_transformers_major() -> None:
    torch = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
    config = {"getm": {"qwen": {"compute_dtype": "bf16"}}}

    assert _model_dtype_kwargs(transformers=SimpleNamespace(__version__="4.51.3"), torch=torch, config=config) == {
        "torch_dtype": "bf16"
    }
    assert _model_dtype_kwargs(transformers=SimpleNamespace(__version__="5.4.0"), torch=torch, config=config) == {
        "dtype": "bf16"
    }


def test_stable_candidate_seed_is_deterministic_and_candidate_specific() -> None:
    seed_a0 = _stable_candidate_seed(13, "doc-a", 0)
    seed_a0_again = _stable_candidate_seed(13, "doc-a", 0)
    seed_a1 = _stable_candidate_seed(13, "doc-a", 1)
    seed_b0 = _stable_candidate_seed(13, "doc-b", 0)
    assert seed_a0 == seed_a0_again
    assert len({seed_a0, seed_a1, seed_b0}) == 3


def test_sampling_params_use_distinct_per_candidate_seed() -> None:
    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    backend = VllmGetmBackend(config=_config(do_sample=True))
    backend._sampling_params_cls = Params
    backend._sampling_common_kwargs = {"max_tokens": 8, "repetition_penalty": 1.05}
    backend._base_seed = 13
    generation = _generation_config(backend.config)

    first = backend._sampling_params_for_prompt(doc_id="doc-a", candidate_index=0, generation_cfg=generation)
    second = backend._sampling_params_for_prompt(doc_id="doc-a", candidate_index=1, generation_cfg=generation)

    assert first.kwargs["seed"] != second.kwargs["seed"]
    assert first.kwargs["temperature"] == 0.7
    assert second.kwargs["top_p"] == 0.95


def test_prefilled_cache_preserves_token_metadata_and_balanced_stop() -> None:
    backend = VllmGetmBackend(config=_config())
    backend._prefilled_outputs[("doc-a", 2)] = {
        "text": '[{"event_type":"质押","arguments":{}}]} trailing text',
        "token_count": 7,
        "ended_with_eos": False,
    }

    output = backend.generate_one(
        prompt="prompt",
        document={"doc_id": "doc-a", "content": "公告文本"},
        schema=None,
        surface_candidates=[],
        slot_plan=None,
        candidate_index=2,
    )

    assert output == '{"events":[{"event_type":"质押","arguments":{}}]}'
    metadata = backend.last_generation_metadata
    assert metadata["generated_token_count"] == 7
    assert metadata["ended_with_eos"] is False
    assert metadata["balanced_stop_applied"] is True


def test_vllm_guided_decoding_config_preserves_sacd_raw_generation_fields() -> None:
    guided = _vllm_guided_decoding_config(_config(sacd=True))

    assert guided is not None
    assert guided["backend"] == "xgrammar"
    assert guided["json_schema"]["required"] == ["events"]
    assert guided["strict"] is True


def test_vllm_generation_metadata_records_sacd_without_full_json_schema() -> None:
    backend = VllmGetmBackend(config=_config(sacd=True))

    metadata = backend.generation_metadata

    assert metadata["sacd_enabled"] is True
    assert metadata["sacd_backend"] == "xgrammar"
    assert metadata["sacd_strict"] is True
    assert "sacd_json_schema" not in metadata


def test_vllm_generation_metadata_records_resolved_engine_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SARGE_VLLM_ENFORCE_EAGER", "1")
    monkeypatch.setenv("SARGE_VLLM_MAX_NUM_SEQS", "1")
    monkeypatch.setenv("SARGE_VLLM_MAX_NUM_BATCHED_TOKENS", "4096")
    backend = VllmGetmBackend(config=_config())

    metadata = backend.generation_metadata

    assert metadata["vllm_enforce_eager"] is True
    assert metadata["vllm_max_num_seqs"] == 1
    assert metadata["vllm_max_num_batched_tokens"] == 4096
    assert metadata["vllm_engine_config"] == {
        "dtype": "bfloat16",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 8192,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
    }


def test_vllm_engine_env_overrides_are_passed_to_llm(monkeypatch) -> None:
    monkeypatch.setenv("SARGE_VLLM_ENFORCE_EAGER", "1")
    monkeypatch.setenv("SARGE_VLLM_MAX_NUM_SEQS", "1")
    monkeypatch.setenv("SARGE_VLLM_MAX_NUM_BATCHED_TOKENS", "4096")
    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 2

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGuidedDecodingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_sampling_params = ModuleType("vllm.sampling_params")
    fake_sampling_params.GuidedDecodingParams = FakeGuidedDecodingParams
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = VllmGetmBackend(config=_config())
    backend._ensure_loaded()

    assert captured["enforce_eager"] is True
    assert captured["max_num_seqs"] == 1
    assert captured["max_num_batched_tokens"] == 4096


def test_vllm_cli_generation_config_records_sacd_strict_mode() -> None:
    from scripts.infer_checkpoint_vllm import _build_generation_config

    args = SimpleNamespace(
        sample=False,
        k=1,
        seed=13,
        temperature=0.7,
        sacd=True,
        sacd_strict=True,
    )

    generation = _build_generation_config(args)

    assert generation["sacd_strict"] is True


def test_vllm_cli_sacd_generation_disables_response_prefix() -> None:
    from scripts.infer_checkpoint_vllm import _build_generation_config

    base_args = SimpleNamespace(
        sample=False,
        k=1,
        seed=13,
        temperature=0.7,
        sacd=False,
        sacd_strict=False,
    )
    sacd_args = SimpleNamespace(
        sample=False,
        k=1,
        seed=13,
        temperature=0.7,
        sacd=True,
        sacd_strict=False,
    )

    base_generation = _build_generation_config(base_args)
    sacd_generation = _build_generation_config(sacd_args)

    assert base_generation["use_response_prefix"] is True
    assert base_generation["response_prefix"] == '{"events":'
    assert sacd_generation["use_response_prefix"] is False
    assert sacd_generation["response_prefix"] == ""


def test_vllm_cli_sacd_path_builds_schema_before_backend_init(monkeypatch, tmp_path) -> None:
    from scripts import infer_checkpoint_vllm as cli

    captured: dict[str, dict] = {}

    class FakeBackend:
        def __init__(self, config: dict) -> None:
            captured["config"] = config
            self.generation_metadata = config["getm"]["generation"]

    def fake_run_inference(**kwargs):
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text('{"doc_id":"doc-1","events":[]}\n', encoding="utf-8")
        return SimpleNamespace(prediction_path=pred_path)

    monkeypatch.setattr(cli, "stage_dataset", lambda **kwargs: None)
    monkeypatch.setattr(cli, "load_schema", lambda dataset, data_root: {"dataset": dataset})
    monkeypatch.setattr(
        cli,
        "build_dataset_json_schema",
        lambda schema, strict=False: {"type": "object", "strict": strict},
    )
    monkeypatch.setattr(cli, "VllmGetmBackend", FakeBackend)
    monkeypatch.setattr(cli, "run_inference", fake_run_inference)

    args = SimpleNamespace(
        dataset="DuEE-Fin-dev500",
        processed=str(tmp_path),
        split="dev",
        limit=1,
        slot_train_limit=1,
        merged=str(tmp_path),
        max_model_len=8192,
        gpu_memory_utilization=0.55,
        sample=False,
        k=1,
        seed=13,
        sacd=True,
        sacd_strict=False,
        sacd_backend="xgrammar",
        batch_mode="per_prompt",
        out=str(tmp_path / "runs"),
        source_commit="test",
    )

    cli._run_inference(args, tmp_path / "staging")

    generation = captured["config"]["getm"]["generation"]
    assert generation["sacd_backend"] == "xgrammar"
    assert generation["sacd_json_schema"] == {"type": "object", "strict": False}
