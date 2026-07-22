from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from sarge.data.schema import load_schema
from sarge.generation.schema_decoding import build_dataset_json_schema


def test_load_schema_normalizes_dict_roles_and_builds_strict_schema(tmp_path: Path) -> None:
    dataset_root = tmp_path / "fixture"
    dataset_root.mkdir(parents=True)
    (dataset_root / "schema.json").write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "event_types": [
                    {
                        "event_type": "质押",
                        "role_list": [{"role": "质押方"}, {"role": "质押物"}],
                    },
                    {
                        "event_type": "中标",
                        "roles": ["中标公司", "招标方"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = load_schema("fixture", data_root=tmp_path)
    strict = build_dataset_json_schema(schema, strict=True)
    lax = build_dataset_json_schema(schema, strict=False)

    assert schema.event_roles["质押"] == ("质押方", "质押物")
    assert strict["required"] == ["events"]
    assert strict["properties"]["events"]["items"]["oneOf"]
    assert lax["properties"]["events"]["items"]["properties"]["event_type"]["enum"] == ["中标", "质押"]


def test_sacd_schema_rejects_empty_event_arguments(tmp_path: Path) -> None:
    dataset_root = tmp_path / "fixture"
    dataset_root.mkdir(parents=True)
    (dataset_root / "schema.json").write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "event_types": [
                    {
                        "event_type": "质押",
                        "roles": ["质押方", "质押物"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    schema = load_schema("fixture", data_root=tmp_path)
    lax = build_dataset_json_schema(schema, strict=False)

    validate(
        {"events": [{"event_type": "质押", "arguments": {"质押方": [{"text": "甲公司"}]}}]},
        lax,
    )

    for payload in [
        {"events": [{"event_type": "质押", "arguments": {}}]},
        {"events": [{"event_type": "质押", "arguments": {"质押方": []}}]},
        {"events": [{"event_type": "质押", "arguments": {"质押方": [{"text": ""}]}}]},
    ]:
        with pytest.raises(ValidationError):
            validate(payload, lax)
