"""Canonical surface-prediction schema for SARGE pipelines.

This module defines the frozen output contract that the inference pipeline
emits and that the evaluator consumes. All downstream code must conform to
``CANONICAL_DOCUMENT_KEYS`` / ``CANONICAL_EVENT_RECORD_KEYS`` /
``CANONICAL_ARGUMENT_KEYS`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

CANONICAL_PREDICTION_FORMAT_VERSION = "sarge.canonical_prediction.v1"
CANONICAL_DOCUMENT_KEYS = frozenset({"doc_id", "events"})
CANONICAL_EVENT_RECORD_KEYS = frozenset({"event_type", "arguments"})
CANONICAL_ARGUMENT_KEYS = frozenset({"text"})


class CanonicalArgumentDict(TypedDict):
    text: str


class CanonicalEventRecordDict(TypedDict):
    event_type: str
    arguments: dict[str, list[CanonicalArgumentDict]]


class CanonicalPredictionDocumentDict(TypedDict):
    doc_id: str
    events: list[CanonicalEventRecordDict]


@dataclass(frozen=True)
class CanonicalArgument:
    text: str


@dataclass(frozen=True)
class CanonicalEventRecord:
    event_type: str
    arguments: dict[str, list[CanonicalArgument]] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalPredictionDocument:
    doc_id: str
    events: list[CanonicalEventRecord] = field(default_factory=list)
