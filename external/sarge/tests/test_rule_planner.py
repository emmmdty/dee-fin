"""Unit tests for ``sarge.postprocess.rule_planner``.

Exercises the three supported modes (``pass_through``, ``dedup_only``,
``conservative_assembler``) on small handcrafted fixtures that cover
the exact-dedup / drop-empty / split / merge / near-dedup paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sarge.postprocess.rule_planner import (
    SUPPORTED_MODES,
    EventRecord,
    PlannerDiagnostics,
    anchor_signature,
    apply_planner,
    conservative_merge,
    conservative_split,
    exact_dedup,
    near_dedup_conservative,
)


@dataclass(frozen=True)
class _FakeSchema:
    """Minimal stand-in for sarge.data.schema.DatasetSchema in unit tests."""

    dataset_id: str = "fixture"
    event_roles: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self):
        if self.event_roles is None:
            object.__setattr__(
                self,
                "event_roles",
                {
                    # Anchor roles (per ANCHOR_ROLES_BY_EVENT_TYPE) plus a few
                    # non-anchor roles (中标日期, 项目编号) so near-dedup paths
                    # can be exercised: same anchor sig + different non-anchor args.
                    "质押": ("质押方", "质押物", "质押股票/股份数量", "事件时间"),
                    "解除质押": ("质押方", "质押物", "质押股票/股份数量", "事件时间"),
                    "股东减持": ("减持方", "交易股票/股份数量", "交易金额", "每股交易价格"),
                    "中标": ("中标公司", "招标方", "中标标的", "中标金额", "中标日期", "项目编号"),
                    "高管变动": ("高管姓名", "高管职位", "变动后职位", "变动类型", "事件时间", "披露日期"),
                    "亏损": ("公司名称", "财报周期", "净亏损"),
                    "其他事件": ("其他角色",),
                },
            )

    def validate_event_type(self, event_type: str) -> str:
        normalized = str(event_type).strip()
        if normalized not in self.event_roles:
            raise ValueError(f"unknown event_type: {normalized!r}")
        return normalized

    def validate_role(self, event_type: str, role: str) -> str:
        if role not in self.event_roles[event_type]:
            raise ValueError(f"unknown role {event_type}/{role!r}")
        return role


@pytest.fixture
def schema() -> _FakeSchema:
    return _FakeSchema()


def _record(event_type: str, **roles_to_texts) -> EventRecord:
    arguments = {role: [{"text": value} for value in values] for role, values in roles_to_texts.items()}
    return EventRecord(event_type=event_type, arguments=arguments)


def test_supported_modes_are_locked() -> None:
    assert SUPPORTED_MODES == frozenset({"pass_through", "dedup_only", "conservative_assembler"})


def test_pass_through_keeps_records_unchanged(schema) -> None:
    records = [_record("质押", 质押方=["甲公司"]), _record("中标", 中标公司=["乙公司"])]
    planned, diag = apply_planner(records, mode="pass_through", schema=schema)
    assert planned == records
    assert diag.events_before == diag.events_after == 2
    assert diag.applied_count == 0


def test_dedup_only_drops_exact_duplicates(schema) -> None:
    duplicate = _record("质押", 质押方=["甲公司"], 事件时间=["2024-01-01"])
    records = [duplicate, duplicate, _record("中标", 中标公司=["丙公司"])]
    planned, diag = apply_planner(records, mode="dedup_only", schema=schema)
    assert len(planned) == 2
    assert diag.dedup_count == 1


def test_conservative_assembler_drops_empty_target_events(schema) -> None:
    records = [_record("质押"), _record("中标", 中标公司=["丙公司"])]
    planned, diag = apply_planner(records, mode="conservative_assembler", schema=schema)
    assert len(planned) == 1
    assert planned[0].event_type == "中标"
    assert diag.dropped_count == 1


def test_conservative_split_creates_two_records_when_anchors_align(schema) -> None:
    record = _record("股东减持", 减持方=["甲", "乙"], 交易股票=["100", "200"], 交易股票_value=["x"])
    # use real anchor roles
    record = EventRecord(
        event_type="股东减持",
        arguments={
            "减持方": [{"text": "甲"}, {"text": "乙"}],
            "交易股票/股份数量": [{"text": "100"}, {"text": "200"}],
            "每股交易价格": [{"text": "1.0"}],
        },
    )
    planned, decisions = conservative_split([record])
    assert len(planned) == 2
    assert decisions and decisions[0].action == "split"
    assert {planned[0].arguments["减持方"][0]["text"], planned[1].arguments["减持方"][0]["text"]} == {"甲", "乙"}


def test_conservative_merge_combines_records_with_same_anchor(schema) -> None:
    left = _record("质押", 质押方=["甲"], 质押物=["A 公司股票"])
    right = _record("质押", 质押方=["甲"], 质押物=["A 公司股票"], 事件时间=["2024-01-01"])
    planned, decisions = conservative_merge([left, right])
    assert len(planned) == 1
    assert decisions and decisions[0].action == "merge"
    assert "事件时间" in planned[0].arguments


def test_near_dedup_keeps_superset_record(schema) -> None:
    """Same anchor (中标公司, 招标方) but superset adds a non-anchor role (中标日期)."""
    base = _record("中标", 中标公司=["甲"], 招标方=["乙"])
    superset = _record("中标", 中标公司=["甲"], 招标方=["乙"], 中标日期=["2024-01-01"])
    planned, decisions = near_dedup_conservative([base, superset])
    assert len(planned) == 1
    assert "中标日期" in planned[0].arguments
    assert decisions and decisions[0].action == "dedup"


def test_exact_dedup_only_targets_target_event_types(schema) -> None:
    non_target = _record("其他事件", 其他角色=["x"])
    records = [non_target, non_target]
    planned, decisions = exact_dedup(records)
    assert len(planned) == 2
    assert decisions == []


def test_apply_planner_rejects_unknown_mode(schema) -> None:
    with pytest.raises(ValueError, match="unsupported planner mode"):
        apply_planner([], mode="not_a_mode", schema=schema)


def test_anchor_signature_uses_event_type_specific_roles() -> None:
    record = _record("质押", 质押方=["甲"], 不在锚=["x"])
    sig = anchor_signature(record)
    assert sig == (("质押方", ("甲",)),)


def test_planner_diagnostics_records_per_event_effect(schema) -> None:
    duplicate = _record("中标", 中标公司=["甲"])
    records = [duplicate, duplicate]
    _, diag = apply_planner(records, mode="conservative_assembler", schema=schema)
    assert isinstance(diag, PlannerDiagnostics)
    assert "中标" in diag.per_event_type_effect
    assert diag.per_event_type_effect["中标"]["dedup"] >= 1
