from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sarge.data.schema import DatasetSchema
from sarge.generation.parser import parse_getm_output
from sarge.generation.prompt import build_getm_prompt
from sarge.models.sft_dataset import (
    gold_events_to_getm_output,
    render_sft_answer_suffix,
    render_sft_user_assistant_prefix,
)
from sarge.record_planning import build_record_plan, events_to_record_plan_output


def _schema() -> DatasetSchema:
    return DatasetSchema(
        dataset_id="fixture",
        schema_dataset="fixture",
        schema_path=Path("/dev/null"),
        canonical_version=None,
        event_roles={
            "质押": ("质押方", "质押物", "质押股票/股份数量", "事件时间"),
            "中标": ("中标公司", "招标方", "中标标的", "中标金额"),
        },
        role_to_event_types={
            "质押方": ("质押",),
            "质押物": ("质押",),
            "质押股票/股份数量": ("质押",),
            "事件时间": ("质押",),
            "中标公司": ("中标",),
            "招标方": ("中标",),
            "中标标的": ("中标",),
            "中标金额": ("中标",),
        },
    )


def _event(event_type: str, **roles_to_values: list[str]) -> dict:
    return {
        "event_type": event_type,
        "arguments": {
            role: [{"text": value} for value in values]
            for role, values in roles_to_values.items()
        },
    }


def test_build_record_plan_selects_private_anchor_roles_for_same_type_records() -> None:
    schema = _schema()
    events = [
        _event("质押", 质押方=["甲公司"], 质押物=["股份"], 质押股票_股份数量=["100万股"]),
        _event("质押", 质押方=["甲公司"], 质押物=["股份"], 质押股票_股份数量=["200万股"]),
    ]
    events[0]["arguments"]["质押股票/股份数量"] = events[0]["arguments"].pop("质押股票_股份数量")
    events[1]["arguments"]["质押股票/股份数量"] = events[1]["arguments"].pop("质押股票_股份数量")

    plan = build_record_plan(events, schema)

    assert [item.record_id for item in plan] == ["R1", "R2"]
    assert [item.event_type for item in plan] == ["质押", "质押"]
    assert plan[0].anchors["质押股票/股份数量"] == ["100万股"]
    assert plan[1].anchors["质押股票/股份数量"] == ["200万股"]
    assert plan[0].anchor_signature != plan[1].anchor_signature


def test_record_plan_output_adds_record_ids_without_changing_arguments() -> None:
    schema = _schema()
    events = [
        _event("中标", 中标公司=["甲公司"], 招标方=["乙公司"], 中标金额=["100万元"]),
    ]

    output = events_to_record_plan_output(events, schema)

    assert output == {
        "record_plan": [
            {
                "record_id": "R1",
                "event_type": "中标",
                "anchors": {
                    "中标公司": ["甲公司"],
                },
            }
        ],
        "events": [
            {
                "record_id": "R1",
                "event_type": "中标",
                "arguments": {
                    "中标公司": ["甲公司"],
                    "招标方": ["乙公司"],
                    "中标金额": ["100万元"],
                },
            }
        ],
    }


def test_sft_record_plan_output_format_uses_planned_target() -> None:
    schema = _schema()

    output = gold_events_to_getm_output(
        [_event("中标", 中标公司=["甲公司"], 招标方=["乙公司"])],
        schema,
        surface_candidates=[],
        output_format="record_plan",
    )

    assert set(output) == {"record_plan", "events"}
    assert output["record_plan"][0]["record_id"] == "R1"
    assert output["events"][0]["record_id"] == "R1"


def test_record_plan_training_config_does_not_default_to_events_response_prefix() -> None:
    config = {"getm": {"output_format": "record_plan"}}

    prefix = render_sft_user_assistant_prefix(object(), "PROMPT", config)
    suffix = render_sft_answer_suffix(object(), {"record_plan": [], "events": []}, config)

    assert prefix == "PROMPT"
    assert suffix.startswith('{"record_plan":')


def test_train_script_config_exposes_record_plan_pilot_controls() -> None:
    from scripts.train_sft import build_config

    config = build_config(
        dataset="fixture",
        model_path=Path("/models/qwen"),
        epochs=1,
        max_train_docs=128,
        seed=13,
        output_format="record_plan",
        max_train_steps=64,
        max_new_tokens=768,
    )

    assert config["getm"]["output_format"] == "record_plan"
    assert config["getm"]["qwen"]["training"]["max_train_steps"] == 64
    assert config["getm"]["generation"]["max_new_tokens"] == 768
    assert config["getm"]["generation"]["use_response_prefix"] is False


def test_train_script_run_name_encodes_record_plan_pilot_scope() -> None:
    from scripts.train_sft import build_run_name

    run_name = build_run_name(
        SimpleNamespace(
            dataset="DuEE-Fin-dev500",
            seed=13,
            epochs=1,
            gpu=2,
            output_format="record_plan",
            max_train_docs=1024,
            max_train_steps=128,
            dev_limit=100,
            max_new_tokens=768,
            run_suffix=None,
        )
    )

    assert run_name == "sarge_sft_DuEE_Fin_dev500_s13_ep1_gpu2_record_plan_train1024_steps128_dev100_gen768"


def test_infer_checkpoint_config_exposes_record_plan_output_format() -> None:
    from scripts.infer_checkpoint import _build_generation_config

    generation = _build_generation_config(
        SimpleNamespace(
            sample=False,
            k=1,
            max_new_tokens=768,
            output_format="record_plan",
            temperature=0.7,
            seed=13,
        )
    )

    assert generation["max_new_tokens"] == 768
    assert generation["use_response_prefix"] is False
    assert generation["response_prefix"] == ""


def test_record_plan_prompt_describes_two_stage_output_contract() -> None:
    schema = _schema()

    prompt = build_getm_prompt(
        dataset="fixture",
        schema=schema,
        document={"doc_id": "doc-1", "content": "甲公司中标乙公司项目。"},
        surface_candidates=[],
        slot_plan=None,
        output_format="record_plan",
        baseline_mode="role_safe",
    )

    assert "`record_plan`" in prompt
    assert "`record_id`" in prompt
    assert '"record_plan"' in prompt


def test_parser_accepts_record_plan_output_and_exports_canonical_events() -> None:
    schema = _schema()
    raw = (
        '{"record_plan":[{"record_id":"R1","event_type":"中标","anchors":{"中标公司":["甲公司"]}}],'
        '"events":[{"record_id":"R1","event_type":"中标","arguments":{"中标公司":["甲公司"]}}]}'
    )

    candidate = parse_getm_output(
        raw,
        doc_id="doc-1",
        candidate_id="c1",
        schema=schema,
        output_format="record_plan",
    )

    assert candidate.parse_status == "ok"
    assert candidate.events[0].event_type == "中标"
    assert candidate.events[0].arguments["中标公司"][0].text == "甲公司"
