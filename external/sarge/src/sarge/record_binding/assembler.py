from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sarge.postprocess.rule_planner import EventRecord


@dataclass(frozen=True)
class BindingDecision:
    action: str
    event_type: str
    left_index: int
    right_index: int
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "event_type": self.event_type,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BindingDiagnostics:
    doc_id: str | None = None
    events_before: int = 0
    events_after: int = 0
    merge_count: int = 0
    blocked_count: int = 0
    decisions: list[BindingDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "events_before": self.events_before,
            "events_after": self.events_after,
            "merge_count": self.merge_count,
            "blocked_count": self.blocked_count,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


class BindingScoreProvider(Protocol):
    def score(
        self,
        left: EventRecord,
        right: EventRecord,
        *,
        left_index: int,
        right_index: int,
    ) -> float:
        """Return compatibility score for two records from the same document."""


@dataclass(frozen=True)
class SurfaceOverlapScoreProvider:
    """Lightweight fallback scorer for CPU-only smoke paths."""

    def score(
        self,
        left: EventRecord,
        right: EventRecord,
        *,
        left_index: int,
        right_index: int,
    ) -> float:
        del left_index, right_index
        left_units = _argument_units(left)
        right_units = _argument_units(right)
        if not left_units or not right_units:
            return 0.0
        overlap = len(left_units & right_units)
        union = len(left_units | right_units)
        return overlap / union if union else 0.0


@dataclass
class _ActiveRecord:
    record: EventRecord
    source_indices: tuple[int, ...]

    @property
    def first_index(self) -> int:
        return self.source_indices[0]


class BindingAssembler:
    """Greedy schema-safe assembler for event records in one document.

    Pairwise scores may come from a learned scorer. The assembler owns the
    hard safety constraints: different event types never bind, conflicting
    anchor values never bind, and overlapping role values must be compatible.
    """

    def __init__(self, *, schema, threshold: float = 0.85):
        self.schema = schema
        self.threshold = float(threshold)

    def bind_document(
        self,
        records: list[EventRecord],
        *,
        score_provider: BindingScoreProvider | None = None,
        doc_id: str | None = None,
    ) -> tuple[list[EventRecord], BindingDiagnostics]:
        for record in records:
            self._validate_record(record)

        provider = score_provider or SurfaceOverlapScoreProvider()
        active = [_ActiveRecord(record=record, source_indices=(index,)) for index, record in enumerate(records)]
        decisions: list[BindingDecision] = []
        blocked_pairs: set[tuple[int, int, str]] = set()

        while True:
            best: tuple[float, int, int] | None = None
            for left_pos in range(len(active)):
                for right_pos in range(left_pos + 1, len(active)):
                    left = active[left_pos]
                    right = active[right_pos]
                    if left.record.event_type != right.record.event_type:
                        continue
                    score = float(
                        provider.score(
                            left.record,
                            right.record,
                            left_index=left.first_index,
                            right_index=right.first_index,
                        )
                    )
                    if score < self.threshold:
                        continue
                    compatible, reason = _records_bindable(left.record, right.record)
                    if not compatible:
                        key = _blocked_key(left, right, reason)
                        if key not in blocked_pairs:
                            blocked_pairs.add(key)
                            decisions.append(
                                BindingDecision(
                                    action="block",
                                    event_type=left.record.event_type,
                                    left_index=left.first_index,
                                    right_index=right.first_index,
                                    score=score,
                                    reason=reason,
                                )
                            )
                        continue
                    if best is None or score > best[0]:
                        best = (score, left_pos, right_pos)

            if best is None:
                break

            score, left_pos, right_pos = best
            left = active[left_pos]
            right = active[right_pos]
            merged = _merge_records(left.record, right.record)
            decisions.append(
                BindingDecision(
                    action="merge",
                    event_type=left.record.event_type,
                    left_index=left.first_index,
                    right_index=right.first_index,
                    score=score,
                    reason="score_above_threshold_and_compatible",
                )
            )
            active[left_pos] = _ActiveRecord(
                record=merged,
                source_indices=tuple(sorted((*left.source_indices, *right.source_indices))),
            )
            del active[right_pos]

        planned = [item.record for item in active]
        diagnostics = BindingDiagnostics(
            doc_id=doc_id,
            events_before=len(records),
            events_after=len(planned),
            merge_count=sum(1 for decision in decisions if decision.action == "merge"),
            blocked_count=sum(1 for decision in decisions if decision.action == "block"),
            decisions=decisions,
        )
        return planned, diagnostics

    def _validate_record(self, record: EventRecord) -> None:
        self.schema.validate_event_type(record.event_type)
        for role in record.arguments:
            self.schema.validate_role(record.event_type, role)


def _records_bindable(left: EventRecord, right: EventRecord) -> tuple[bool, str]:
    if _has_conflicting_anchor_values(left, right):
        return False, "incompatible_anchor_values"
    if _has_conflicting_role_values(left, right):
        return False, "incompatible_role_values"
    return True, "compatible"


def _has_conflicting_anchor_values(left: EventRecord, right: EventRecord) -> bool:
    anchor_roles = _anchor_roles(left.event_type)
    for role in anchor_roles:
        left_values = _role_values(left, role)
        right_values = _role_values(right, role)
        if left_values and right_values and left_values != right_values:
            return True
    return False


def _has_conflicting_role_values(left: EventRecord, right: EventRecord) -> bool:
    for role in set(left.arguments) & set(right.arguments):
        left_values = _role_values(left, role)
        right_values = _role_values(right, role)
        if left_values and right_values and not (left_values <= right_values or right_values <= left_values):
            return True
    return False


def _merge_records(left: EventRecord, right: EventRecord) -> EventRecord:
    arguments = {role: [{"text": value["text"]} for value in values] for role, values in left.arguments.items()}
    for role, values in right.arguments.items():
        target = arguments.setdefault(role, [])
        existing = {_normalize_text(value["text"]) for value in target if value.get("text")}
        for value in values:
            text = str(value.get("text") or "").strip()
            normalized = _normalize_text(text)
            if text and normalized not in existing:
                target.append({"text": text})
                existing.add(normalized)
    return EventRecord(event_type=left.event_type, arguments=arguments)


def _blocked_key(left: _ActiveRecord, right: _ActiveRecord, reason: str) -> tuple[int, int, str]:
    return min(left.source_indices), min(right.source_indices), reason


def _anchor_roles(event_type: str) -> tuple[str, ...]:
    # Keep this local and functional so record_binding does not depend on
    # rule-planner target-event gates. These are private record-binding anchors.
    anchors = {
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
    return anchors.get(event_type, ())


def _role_values(record: EventRecord, role: str) -> set[str]:
    return {_normalize_text(value["text"]) for value in record.arguments.get(role, []) if value.get("text")}


def _argument_units(record: EventRecord) -> set[tuple[str, str]]:
    return {
        (role, _normalize_text(value["text"]))
        for role, values in record.arguments.items()
        for value in values
        if value.get("text")
    }


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())
