from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from sarge.data.loader import V2DatasetDocument
from sarge.data.schema import DatasetSchema
from sarge.slot_planning.plan import EventSlot, SlotPlanDocument


@dataclass(frozen=True)
class SchemaEmptyPlanner:
    schema: DatasetSchema

    def predict_one(self, document: V2DatasetDocument) -> SlotPlanDocument:
        _require_predict_document(document)
        return SlotPlanDocument(doc_id=document.doc_id, dataset=self.schema.dataset_id, slots=[])

    def predict(self, documents: Iterable[V2DatasetDocument]) -> list[SlotPlanDocument]:
        return [self.predict_one(document) for document in documents]


@dataclass(frozen=True)
class EventTypePrior:
    event_type: str
    presence_rate: float
    positive_count_mode: int
    positive_count_confidence: float
    role_prior: dict[str, float]


@dataclass(frozen=True)
class TrainPriorPlanner:
    schema: DatasetSchema
    event_type_priors: dict[str, EventTypePrior]
    selected_event_type: str | None

    @classmethod
    def fit(cls, schema: DatasetSchema, documents: Iterable[V2DatasetDocument]) -> TrainPriorPlanner:
        document_list = list(documents)
        if not document_list:
            return cls(schema=schema, event_type_priors={}, selected_event_type=None)
        for document in document_list:
            if document.gold is None:
                raise ValueError("train_prior fit requires gold-visible train documents")

        count_by_event_type: dict[str, Counter[int]] = {
            event_type: Counter() for event_type in schema.event_roles
        }
        role_filled_by_event_type: dict[str, Counter[str]] = {
            event_type: Counter() for event_type in schema.event_roles
        }
        record_count_by_event_type: Counter[str] = Counter()

        for document in document_list:
            assert document.gold is not None
            doc_counts = Counter(str(event.get("event_type", "")).strip() for event in document.gold.events)
            for event_type in schema.event_roles:
                count_by_event_type[event_type][int(doc_counts.get(event_type, 0))] += 1
            for event in document.gold.events:
                event_type = schema.validate_event_type(str(event.get("event_type", "")))
                record_count_by_event_type[event_type] += 1
                arguments = event.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                for role in schema.event_roles[event_type]:
                    values = arguments.get(role) or []
                    if isinstance(values, list) and values:
                        role_filled_by_event_type[event_type][role] += 1

        event_type_priors: dict[str, EventTypePrior] = {}
        for event_type, roles in schema.event_roles.items():
            counter = count_by_event_type[event_type]
            positive_doc_total = sum(count for event_count, count in counter.items() if event_count > 0)
            if positive_doc_total:
                positive_mode, positive_mode_freq = _mode(
                    {event_count: count for event_count, count in counter.items() if event_count > 0}
                )
            else:
                positive_mode, positive_mode_freq = 0, 0
            record_total = record_count_by_event_type[event_type]
            event_type_priors[event_type] = EventTypePrior(
                event_type=event_type,
                presence_rate=positive_doc_total / len(document_list),
                positive_count_mode=positive_mode,
                positive_count_confidence=(positive_mode_freq / positive_doc_total) if positive_doc_total else 0.0,
                role_prior={
                    role: (role_filled_by_event_type[event_type][role] / record_total) if record_total else 0.0
                    for role in roles
                },
            )

        selected_event_type = _select_top_event_type(event_type_priors)
        return cls(schema=schema, event_type_priors=event_type_priors, selected_event_type=selected_event_type)

    def predict_one(self, document: V2DatasetDocument) -> SlotPlanDocument:
        _require_predict_document(document)
        if self.selected_event_type is None:
            return SlotPlanDocument(doc_id=document.doc_id, dataset=self.schema.dataset_id, slots=[])

        prior = self.event_type_priors[self.selected_event_type]
        slots = [
            EventSlot(
                event_type=prior.event_type,
                slot_id=slot_id,
                count_confidence=prior.positive_count_confidence,
                role_prior=dict(prior.role_prior),
                supporting_candidates=[],
            )
            for slot_id in range(prior.positive_count_mode)
        ]
        return SlotPlanDocument(doc_id=document.doc_id, dataset=self.schema.dataset_id, slots=slots)

    def predict(self, documents: Iterable[V2DatasetDocument]) -> list[SlotPlanDocument]:
        return [self.predict_one(document) for document in documents]


def _require_predict_document(document: V2DatasetDocument) -> None:
    if document.gold is not None:
        raise ValueError("predict document must not expose gold")


def _mode(counts: dict[int, int]) -> tuple[int, int]:
    if not counts:
        return 0, 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))


def _select_top_event_type(event_type_priors: dict[str, EventTypePrior]) -> str | None:
    if not event_type_priors:
        return None
    event_type, prior = max(
        event_type_priors.items(),
        key=lambda item: (item[1].presence_rate, item[1].positive_count_confidence, item[0]),
    )
    if prior.presence_rate <= 0.0:
        return None
    return event_type
