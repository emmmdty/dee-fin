from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from sarge.postprocess.rule_planner import EventRecord
from sarge.record_binding import (
    BindingAssembler,
    BindingDecision,
    BindingDiagnostics,
    BindingScoreProvider,
    bind_prediction_rows,
)
from sarge.record_binding.run import run_record_binding


@dataclass(frozen=True)
class _FakeSchema:
    event_roles: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self):
        if self.event_roles is None:
            object.__setattr__(
                self,
                "event_roles",
                {
                    "质押": ("质押方", "质押物", "质押股票/股份数量", "事件时间"),
                    "中标": ("中标公司", "招标方", "中标标的", "中标金额"),
                    "其他事件": ("其他角色",),
                },
            )

    def validate_event_type(self, event_type: str) -> str:
        if event_type not in self.event_roles:
            raise ValueError(f"unknown event type: {event_type}")
        return event_type

    def validate_role(self, event_type: str, role: str) -> str:
        if role not in self.event_roles[event_type]:
            raise ValueError(f"unknown role: {event_type}/{role}")
        return role


class _ScoreMap(BindingScoreProvider):
    def __init__(self, scores: dict[tuple[int, int], float]):
        self.scores = scores

    def score(self, left: EventRecord, right: EventRecord, *, left_index: int, right_index: int) -> float:
        return self.scores.get((left_index, right_index), self.scores.get((right_index, left_index), 0.0))


@pytest.fixture
def schema() -> _FakeSchema:
    return _FakeSchema()


def _record(event_type: str, **roles_to_texts: list[str]) -> EventRecord:
    return EventRecord(
        event_type=event_type,
        arguments={role: [{"text": value} for value in values] for role, values in roles_to_texts.items()},
    )


def test_binding_assembler_merges_high_confidence_complementary_records(schema) -> None:
    records = [
        _record("质押", 质押方=["甲公司"], 质押物=["A股"]),
        _record("质押", 质押方=["甲公司"], 事件时间=["2024-01-01"]),
    ]
    assembler = BindingAssembler(schema=schema, threshold=0.8)

    planned, diagnostics = assembler.bind_document(
        records,
        score_provider=_ScoreMap({(0, 1): 0.91}),
        doc_id="doc-1",
    )

    assert planned == [
        _record("质押", 质押方=["甲公司"], 质押物=["A股"], 事件时间=["2024-01-01"])
    ]
    assert diagnostics == BindingDiagnostics(
        doc_id="doc-1",
        events_before=2,
        events_after=1,
        merge_count=1,
        blocked_count=0,
        decisions=[
            BindingDecision(
                action="merge",
                event_type="质押",
                left_index=0,
                right_index=1,
                score=0.91,
                reason="score_above_threshold_and_compatible",
            )
        ],
    )


def test_binding_assembler_never_merges_different_event_types(schema) -> None:
    records = [
        _record("质押", 质押方=["甲公司"]),
        _record("中标", 中标公司=["甲公司"], 中标金额=["100万元"]),
    ]
    assembler = BindingAssembler(schema=schema, threshold=0.8)

    planned, diagnostics = assembler.bind_document(
        records,
        score_provider=_ScoreMap({(0, 1): 0.99}),
    )

    assert planned == records
    assert diagnostics.events_before == diagnostics.events_after == 2
    assert diagnostics.merge_count == 0
    assert diagnostics.blocked_count == 0


def test_binding_assembler_blocks_conflicting_anchor_values(schema) -> None:
    records = [
        _record("质押", 质押方=["甲公司"], 事件时间=["2024-01-01"]),
        EventRecord(
            event_type="质押",
            arguments={
                "质押方": [{"text": "乙公司"}],
                "质押股票/股份数量": [{"text": "100股"}],
            },
        ),
    ]
    assembler = BindingAssembler(schema=schema, threshold=0.8)

    planned, diagnostics = assembler.bind_document(
        records,
        score_provider=_ScoreMap({(0, 1): 0.95}),
    )

    assert planned == records
    assert diagnostics.merge_count == 0
    assert diagnostics.blocked_count == 1
    assert diagnostics.decisions == [
        BindingDecision(
            action="block",
            event_type="质押",
            left_index=0,
            right_index=1,
            score=0.95,
            reason="incompatible_anchor_values",
        )
    ]


def test_binding_assembler_rejects_schema_invalid_records(schema) -> None:
    assembler = BindingAssembler(schema=schema)

    with pytest.raises(ValueError, match="unknown role"):
        assembler.bind_document([_record("质押", 不存在=["x"])])


def test_bind_prediction_rows_preserves_canonical_contract(schema) -> None:
    rows = [
        {
            "doc_id": "doc-1",
            "events": [
                _record("质押", 质押方=["甲公司"], 质押物=["A股"]).to_canonical(),
                _record("质押", 质押方=["甲公司"], 事件时间=["2024-01-01"]).to_canonical(),
            ],
        }
    ]

    bound_rows, diagnostics = bind_prediction_rows(
        rows,
        schema=schema,
        threshold=0.8,
        score_provider=_ScoreMap({(0, 1): 0.91}),
    )

    assert bound_rows == [
        {
            "doc_id": "doc-1",
            "events": [
                _record("质押", 质押方=["甲公司"], 质押物=["A股"], 事件时间=["2024-01-01"]).to_canonical()
            ],
        }
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].doc_id == "doc-1"
    assert diagnostics[0].events_before == 2
    assert diagnostics[0].events_after == 1


def test_binding_diagnostics_serializes_to_json_ready_dict() -> None:
    diagnostics = BindingDiagnostics(
        doc_id="doc-1",
        events_before=2,
        events_after=1,
        merge_count=1,
        blocked_count=0,
        decisions=[
            BindingDecision(
                action="merge",
                event_type="质押",
                left_index=0,
                right_index=1,
                score=0.91,
                reason="score_above_threshold_and_compatible",
            )
        ],
    )

    assert diagnostics.to_dict() == {
        "doc_id": "doc-1",
        "events_before": 2,
        "events_after": 1,
        "merge_count": 1,
        "blocked_count": 0,
        "decisions": [
            {
                "action": "merge",
                "event_type": "质押",
                "left_index": 0,
                "right_index": 1,
                "score": 0.91,
                "reason": "score_above_threshold_and_compatible",
            }
        ],
    }


def test_run_record_binding_writes_run_style_prediction_and_summary(tmp_path: Path) -> None:
    input_root = tmp_path / "input_run"
    data_root = tmp_path / "data"
    output_root = tmp_path / "output_run"
    dataset = "toy_dataset"
    split = "dev"
    pred_path = input_root / "predictions" / dataset / f"{split}.canonical.pred.jsonl"
    schema_path = data_root / dataset / "schema.json"
    pred_path.parent.mkdir(parents=True)
    schema_path.parent.mkdir(parents=True)
    _write_jsonl(
        pred_path,
        [
            {
                "doc_id": "doc-1",
                "events": [
                    _record("质押", 质押方=["甲公司"], 质押物=["A股"]).to_canonical(),
                    _record("质押", 质押方=["甲公司"], 事件时间=["2024-01-01"]).to_canonical(),
                ],
            }
        ],
    )
    schema_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "event_types": [
                    {
                        "event_type": "质押",
                        "roles": ["质押方", "质押物", "质押股票/股份数量", "事件时间"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_record_binding(
        input_run_root=input_root,
        output_run_root=output_root,
        dataset=dataset,
        split=split,
        data_root=data_root,
        threshold=0.0,
    )

    output_pred_path = output_root / "predictions" / dataset / f"{split}.canonical.pred.jsonl"
    diagnostics_path = output_root / "diagnostics" / "record_binding.json"
    summary_path = output_root / "summary.json"
    assert summary["docs"] == 1
    assert summary["events_before"] == 2
    assert summary["events_after"] == 1
    assert summary["merge_count"] == 1
    assert summary["input_prediction_path"] == str(pred_path)
    assert summary["output_prediction_path"] == str(output_pred_path)
    assert output_pred_path.is_file()
    assert diagnostics_path.is_file()
    assert summary_path.is_file()
    assert _read_jsonl(output_pred_path) == [
        {
            "doc_id": "doc-1",
            "events": [
                _record("质押", 质押方=["甲公司"], 质押物=["A股"], 事件时间=["2024-01-01"]).to_canonical()
            ],
        }
    ]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["summary"] == summary
    assert diagnostics["documents"][0]["doc_id"] == "doc-1"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
