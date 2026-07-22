"""Tests for evidence grounding (anti-fabrication)."""

from __future__ import annotations

from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.extractor.base import ExtractionContext
from finekg.relations.grounding import ground_relations


def _nodes() -> list[EventNode]:
    return [
        EventNode(event_id="e1", event_type="A", doc_id="d"),
        EventNode(event_id="e2", event_type="B", doc_id="d"),
    ]


def _edge(quote: str) -> RelationEdge:
    return RelationEdge(
        head_id="e1",
        tail_id="e2",
        relation_type=RelationType.CAUSAL,
        evidence=[EvidenceSpan(doc_id="d", char_start=0, char_end=0, text=quote)],
    )


def test_quote_found_in_text_is_kept_and_positioned() -> None:
    ctx = ExtractionContext(doc_text={"d": "因为甲事件，所以乙事件发生。"})
    result = ground_relations([_edge("所以乙事件")], _nodes(), ctx, require_evidence=True)
    assert len(result.kept) == 1
    assert result.kept[0].evidence[0].char_end > result.kept[0].evidence[0].char_start


def test_fabricated_quote_is_dropped() -> None:
    ctx = ExtractionContext(doc_text={"d": "因为甲事件，所以乙事件发生。"})
    result = ground_relations([_edge("子虚乌有的证据")], _nodes(), ctx, require_evidence=True)
    assert len(result.kept) == 0
    assert result.drop_rate == 1.0


def test_require_evidence_false_keeps_ungrounded() -> None:
    ctx = ExtractionContext(doc_text={"d": "无关文本"})
    result = ground_relations([_edge("不存在")], _nodes(), ctx, require_evidence=False)
    assert len(result.kept) == 1


def test_prepositioned_heuristic_evidence_is_kept() -> None:
    edge = RelationEdge(
        head_id="e1",
        tail_id="e2",
        relation_type=RelationType.TEMPORAL,
        subtype="BEFORE",
        evidence=[EvidenceSpan(doc_id="d", char_start=2, char_end=6, text="甲事件")],
    )
    result = ground_relations([edge], _nodes(), ExtractionContext(), require_evidence=True)
    assert len(result.kept) == 1
