"""Rule-based event-record planner: dedup, drop-empty, split, merge, near-dedup.

This module implements the deterministic post-processing stage that runs after
candidate selection and before canonical export. It is the production default
in SARGE; the learning-based replacement lives in :mod:`sarge.postprocess.lrd_planner`.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sarge.data.schema import DatasetSchema

TARGET_EVENT_TYPES = frozenset(
    {
        "亏损",
        "高管变动",
        "质押",
        "股东减持",
        "中标",
        "解除质押",
        "企业收购",
        "企业融资",
        "被约谈",
        "公司上市",
    }
)

ANCHOR_ROLES_BY_EVENT_TYPE: dict[str, tuple[str, ...]] = {
    "公司上市": ("上市公司", "证券代码", "发行价格", "募资金额"),
    "高管变动": ("高管姓名", "高管职位", "变动后职位", "变动类型", "事件时间", "披露日期"),
    "质押": ("质押方", "质押物", "质押股票/股份数量", "事件时间"),
    "解除质押": ("质押方", "质押物", "质押股票/股份数量", "事件时间"),
    "股东减持": ("减持方", "交易股票/股份数量", "交易金额", "每股交易价格"),
    "企业融资": ("被投资方", "融资轮次", "融资金额", "投资方", "领投方"),
    "企业收购": ("收购方", "被收购方", "收购标的", "交易金额"),
    "亏损": ("公司名称", "财报周期", "净亏损"),
    "中标": ("中标公司", "招标方", "中标标的", "中标金额"),
    "被约谈": ("公司名称", "约谈机构", "被约谈时间"),
}

SUPPORTED_MODES = frozenset({"pass_through", "dedup_only", "conservative_assembler"})
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class EventRecord:
    event_type: str
    arguments: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    @classmethod
    def from_canonical(cls, payload: dict[str, Any]) -> EventRecord:
        event_type = str(payload.get("event_type") or "").strip()
        arguments = _canonical_arguments(payload.get("arguments") or {})
        return cls(event_type=event_type, arguments=arguments)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "arguments": {
                role: [{"text": value["text"]} for value in values]
                for role, values in self.arguments.items()
                if values
            },
        }


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    event_type: str
    before_count: int
    after_count: int
    reason: str
    doc_id: str | None = None


@dataclass
class PlannerDiagnostics:
    mode: str
    events_before: int = 0
    events_after: int = 0
    applied_count: int = 0
    merge_count: int = 0
    split_count: int = 0
    dedup_count: int = 0
    dropped_count: int = 0
    affected_docs: set[str] = field(default_factory=set)
    decisions: list[PlannerDecision] = field(default_factory=list)
    per_event_type_effect: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def add_decision(self, decision: PlannerDecision) -> None:
        self.decisions.append(decision)
        self.applied_count += 1
        if decision.doc_id:
            self.affected_docs.add(decision.doc_id)
        self.per_event_type_effect[decision.event_type][decision.action] += 1
        if decision.action == "merge":
            self.merge_count += max(decision.before_count - decision.after_count, 1)
        elif decision.action == "split":
            self.split_count += max(decision.after_count - decision.before_count, 1)
        elif decision.action == "dedup":
            self.dedup_count += max(decision.before_count - decision.after_count, 1)
        elif decision.action == "drop_empty":
            self.dropped_count += max(decision.before_count - decision.after_count, 1)

    def merge_child(self, child: PlannerDiagnostics, *, doc_id: str) -> None:
        self.events_before += child.events_before
        self.events_after += child.events_after
        self.applied_count += child.applied_count
        self.merge_count += child.merge_count
        self.split_count += child.split_count
        self.dedup_count += child.dedup_count
        self.dropped_count += child.dropped_count
        if child.applied_count or child.events_before != child.events_after:
            self.affected_docs.add(doc_id)
        for decision in child.decisions:
            self.decisions.append(
                PlannerDecision(
                    action=decision.action,
                    event_type=decision.event_type,
                    before_count=decision.before_count,
                    after_count=decision.after_count,
                    reason=decision.reason,
                    doc_id=doc_id,
                )
            )
        for event_type, counts in child.per_event_type_effect.items():
            self.per_event_type_effect[event_type].update(counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "events_before": self.events_before,
            "events_after": self.events_after,
            "applied_count": self.applied_count,
            "merge_count": self.merge_count,
            "split_count": self.split_count,
            "dedup_count": self.dedup_count,
            "dropped_count": self.dropped_count,
            "affected_doc_count": len(self.affected_docs),
            "affected_docs": sorted(self.affected_docs),
            "per_event_type_effect": {
                event_type: dict(sorted(counts.items()))
                for event_type, counts in sorted(self.per_event_type_effect.items())
            },
            "decisions": [
                {
                    "action": decision.action,
                    "event_type": decision.event_type,
                    "before_count": decision.before_count,
                    "after_count": decision.after_count,
                    "reason": decision.reason,
                    "doc_id": decision.doc_id,
                }
                for decision in self.decisions
            ],
        }


def normalize_record_signature(record: EventRecord | dict[str, Any]) -> tuple[Any, ...]:
    event = record if isinstance(record, EventRecord) else EventRecord.from_canonical(record)
    role_signatures = []
    for role, values in event.arguments.items():
        texts = tuple(sorted({_normalize_text(value.get("text")) for value in values if value.get("text")}))
        role_signatures.append((role, texts))
    return event.event_type, tuple(sorted(role_signatures))


def exact_dedup(records: list[EventRecord]) -> tuple[list[EventRecord], list[PlannerDecision]]:
    seen: set[tuple[Any, ...]] = set()
    kept: list[EventRecord] = []
    decisions: list[PlannerDecision] = []
    dropped_by_event: Counter[str] = Counter()
    for record in records:
        if not _is_target(record):
            kept.append(record)
            continue
        signature = normalize_record_signature(record)
        if signature in seen:
            dropped_by_event[record.event_type] += 1
            continue
        seen.add(signature)
        kept.append(record)
    for event_type, count in dropped_by_event.items():
        decisions.append(
            PlannerDecision(
                action="dedup",
                event_type=event_type,
                before_count=count + 1,
                after_count=1,
                reason="identical canonical event record signature",
            )
        )
    return kept, decisions


def near_dedup_conservative(records: list[EventRecord]) -> tuple[list[EventRecord], list[PlannerDecision]]:
    kept: list[EventRecord] = []
    decisions: list[PlannerDecision] = []
    for record in records:
        if not _is_target(record):
            kept.append(record)
            continue
        replacement_index = None
        skip_record = False
        for index, existing in enumerate(kept):
            if not _is_target(existing) or existing.event_type != record.event_type:
                continue
            if anchor_signature(existing) != anchor_signature(record) or not anchor_signature(record):
                continue
            if _argument_signature(record) <= _argument_signature(existing):
                skip_record = True
                decisions.append(
                    PlannerDecision(
                        action="dedup",
                        event_type=record.event_type,
                        before_count=2,
                        after_count=1,
                        reason="near-duplicate anchor signature with subset arguments",
                    )
                )
                break
            if _argument_signature(existing) < _argument_signature(record):
                replacement_index = index
                break
        if skip_record:
            continue
        if replacement_index is not None:
            decisions.append(
                PlannerDecision(
                    action="dedup",
                    event_type=record.event_type,
                    before_count=2,
                    after_count=1,
                    reason="near-duplicate anchor signature with superset arguments",
                )
            )
            kept[replacement_index] = record
        else:
            kept.append(record)
    return kept, decisions


def anchor_signature(record: EventRecord) -> tuple[tuple[str, tuple[str, ...]], ...]:
    roles = ANCHOR_ROLES_BY_EVENT_TYPE.get(record.event_type, ())
    signature = []
    for role in roles:
        values = tuple(_argument_values(record, role))
        if values:
            signature.append((role, values))
    return tuple(signature)


def conservative_merge(records: list[EventRecord]) -> tuple[list[EventRecord], list[PlannerDecision]]:
    kept: list[EventRecord] = []
    decisions: list[PlannerDecision] = []
    for record in records:
        if not _is_target(record) or not anchor_signature(record):
            kept.append(record)
            continue
        merged = False
        for index, existing in enumerate(kept):
            if (
                _is_target(existing)
                and existing.event_type == record.event_type
                and _anchors_compatible(existing, record)
                and _records_compatible(existing, record)
            ):
                kept[index] = _merge_records(existing, record)
                decisions.append(
                    PlannerDecision(
                        action="merge",
                        event_type=record.event_type,
                        before_count=2,
                        after_count=1,
                        reason="same non-empty anchor signature with compatible arguments",
                    )
                )
                merged = True
                break
        if not merged:
            kept.append(record)
    return kept, decisions


def conservative_split(records: list[EventRecord]) -> tuple[list[EventRecord], list[PlannerDecision]]:
    planned: list[EventRecord] = []
    decisions: list[PlannerDecision] = []
    for record in records:
        split_records = _split_record(record)
        planned.extend(split_records)
        if len(split_records) > 1:
            decisions.append(
                PlannerDecision(
                    action="split",
                    event_type=record.event_type,
                    before_count=1,
                    after_count=len(split_records),
                    reason="multiple aligned anchor value groups",
                )
            )
    return planned, decisions


def apply_planner(
    records: list[EventRecord],
    *,
    mode: str,
    schema: DatasetSchema,
) -> tuple[list[EventRecord], PlannerDiagnostics]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported planner mode: {mode}")
    for record in records:
        _validate_record(record, schema)

    diagnostics = PlannerDiagnostics(mode=mode, events_before=len(records))
    if mode == "pass_through":
        diagnostics.events_after = len(records)
        return list(records), diagnostics

    planned, decisions = exact_dedup(records)
    for decision in decisions:
        diagnostics.add_decision(decision)
    if mode == "dedup_only":
        diagnostics.events_after = len(planned)
        return planned, diagnostics

    planned, drop_decisions = _drop_empty_target_events(planned)
    for decision in drop_decisions:
        diagnostics.add_decision(decision)
    planned, split_decisions = conservative_split(planned)
    for decision in split_decisions:
        diagnostics.add_decision(decision)
    planned, merge_decisions = conservative_merge(planned)
    for decision in merge_decisions:
        diagnostics.add_decision(decision)
    planned, near_dedup_decisions = near_dedup_conservative(planned)
    for decision in near_dedup_decisions:
        diagnostics.add_decision(decision)

    for record in planned:
        _validate_record(record, schema)
    diagnostics.events_after = len(planned)
    return planned, diagnostics


def _drop_empty_target_events(records: list[EventRecord]) -> tuple[list[EventRecord], list[PlannerDecision]]:
    kept: list[EventRecord] = []
    dropped: Counter[str] = Counter()
    for record in records:
        if _is_target(record) and not record.arguments:
            dropped[record.event_type] += 1
            continue
        kept.append(record)
    decisions = [
        PlannerDecision(
            action="drop_empty",
            event_type=event_type,
            before_count=count,
            after_count=0,
            reason="target event has no predicted arguments",
        )
        for event_type, count in sorted(dropped.items())
    ]
    return kept, decisions


def _split_record(record: EventRecord) -> list[EventRecord]:
    if not _is_target(record):
        return [record]
    anchor_roles = [role for role in ANCHOR_ROLES_BY_EVENT_TYPE.get(record.event_type, ()) if role in record.arguments]
    multi_lengths = [len(record.arguments[role]) for role in anchor_roles if len(record.arguments[role]) > 1]
    if len(multi_lengths) < 2 or len(set(multi_lengths)) != 1:
        return [record]
    split_count = multi_lengths[0]
    if split_count < 2 or split_count > 3:
        return [record]
    split_records = []
    for index in range(split_count):
        arguments: dict[str, list[dict[str, str]]] = {}
        for role, values in record.arguments.items():
            if len(values) == split_count:
                arguments[role] = [{"text": values[index]["text"]}]
            elif len(values) == 1:
                arguments[role] = [{"text": values[0]["text"]}]
            else:
                return [record]
        split_records.append(EventRecord(event_type=record.event_type, arguments=arguments))
    return split_records


def _records_compatible(left: EventRecord, right: EventRecord) -> bool:
    for role in set(left.arguments) & set(right.arguments):
        left_values = set(_argument_values(left, role))
        right_values = set(_argument_values(right, role))
        if left_values and right_values and not (left_values <= right_values or right_values <= left_values):
            return False
    return True


def _anchors_compatible(left: EventRecord, right: EventRecord) -> bool:
    if left.event_type != right.event_type:
        return False
    roles = ANCHOR_ROLES_BY_EVENT_TYPE.get(left.event_type, ())
    shared_anchor = False
    for role in roles:
        left_values = set(_argument_values(left, role))
        right_values = set(_argument_values(right, role))
        if not left_values or not right_values:
            continue
        shared_anchor = True
        if left_values != right_values:
            return False
    return shared_anchor


def _merge_records(left: EventRecord, right: EventRecord) -> EventRecord:
    arguments = {role: [{"text": value["text"]} for value in values] for role, values in left.arguments.items()}
    for role, values in right.arguments.items():
        existing = {_normalize_text(value["text"]) for value in arguments.get(role, [])}
        target_values = arguments.setdefault(role, [])
        for value in values:
            normalized = _normalize_text(value["text"])
            if normalized not in existing:
                target_values.append({"text": value["text"]})
                existing.add(normalized)
    return EventRecord(event_type=left.event_type, arguments=arguments)


def _argument_signature(record: EventRecord) -> set[tuple[str, str]]:
    return {
        (role, _normalize_text(value["text"]))
        for role, values in record.arguments.items()
        for value in values
        if value.get("text")
    }


def _argument_values(record: EventRecord, role: str) -> tuple[str, ...]:
    return tuple(_normalize_text(value["text"]) for value in record.arguments.get(role, []) if value.get("text"))


def _is_target(record: EventRecord) -> bool:
    return record.event_type in TARGET_EVENT_TYPES


def _validate_record(record: EventRecord, schema: DatasetSchema) -> None:
    schema.validate_event_type(record.event_type)
    for role in record.arguments:
        schema.validate_role(record.event_type, role)


def _canonical_arguments(arguments: Any) -> dict[str, list[dict[str, str]]]:
    if not isinstance(arguments, dict):
        return {}
    canonical: dict[str, list[dict[str, str]]] = {}
    for role, values in arguments.items():
        role_name = str(role).strip()
        if not role_name or not isinstance(values, list):
            continue
        role_values = []
        for value in values:
            if not isinstance(value, dict):
                continue
            text = str(value.get("text") or "").strip()
            if text:
                role_values.append({"text": text})
        if role_values:
            canonical[role_name] = role_values
    return canonical


def _normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", str(value).strip())
