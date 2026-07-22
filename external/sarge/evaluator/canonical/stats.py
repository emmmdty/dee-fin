from __future__ import annotations

from dataclasses import dataclass

from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.types import CanonicalEventRecord


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def to_metrics(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def empty_metrics() -> dict[str, float | int]:
    return Counts().to_metrics()


def role_value_sets(record: CanonicalEventRecord) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for raw_role, raw_values in record.arguments.items():
        role = normalize_text(str(raw_role))
        if isinstance(raw_values, (str, bytes)):
            values = [raw_values]
        else:
            values = list(raw_values)
        value_set = {normalized for value in values if (normalized := normalize_optional_text(value))}
        if value_set:
            result.setdefault(role, set()).update(value_set)
    return result


def event_type(record: CanonicalEventRecord) -> str:
    return normalize_text(str(record.event_type))


def document_id(record: CanonicalEventRecord) -> str:
    return normalize_text(str(record.document_id))


def record_unit_count(record: CanonicalEventRecord) -> int:
    return sum(len(values) for values in role_value_sets(record).values())


def record_overlap(pred_record: CanonicalEventRecord, gold_record: CanonicalEventRecord) -> int:
    pred_sets = role_value_sets(pred_record)
    gold_sets = role_value_sets(gold_record)
    return sum(len(pred_sets.get(role, set()) & gold_sets.get(role, set())) for role in pred_sets.keys() | gold_sets.keys())


def count_record_pair(pred_record: CanonicalEventRecord, gold_record: CanonicalEventRecord) -> Counts:
    pred_sets = role_value_sets(pred_record)
    gold_sets = role_value_sets(gold_record)
    counts = Counts()
    for role in pred_sets.keys() | gold_sets.keys():
        pred_values = pred_sets.get(role, set())
        gold_values = gold_sets.get(role, set())
        counts.tp += len(pred_values & gold_values)
        counts.fp += len(pred_values - gold_values)
        counts.fn += len(gold_values - pred_values)
    return counts


def count_unmatched_pred(pred_record: CanonicalEventRecord) -> Counts:
    return Counts(fp=record_unit_count(pred_record))


def count_unmatched_gold(gold_record: CanonicalEventRecord) -> Counts:
    return Counts(fn=record_unit_count(gold_record))


def exact_record_match(pred_record: CanonicalEventRecord, gold_record: CanonicalEventRecord) -> bool:
    return role_value_sets(pred_record) == role_value_sets(gold_record) and record_unit_count(gold_record) > 0
