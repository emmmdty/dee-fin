"""Relation extractor interface and registry.

A relation extractor turns a set of event nodes (optionally with their source
text) into typed, evidence-grounded relation edges. Heuristic and neural
implementations register here and are interchangeable behind this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from finekg.core.registry import Registry
from finekg.core.schema import EventNode, RelationEdge

__all__ = ["ExtractionContext", "RelationExtractor", "relation_extractors"]


@dataclass
class ExtractionContext:
    """Optional side information for extraction.

    `doc_text` maps doc_id -> raw text, used by neural extractors for grounding
    and reasoning. Heuristic extractors can ignore it and rely on node fields.
    """

    doc_text: dict[str, str] = field(default_factory=dict)


class RelationExtractor(ABC):
    """Maps event nodes to relation edges."""

    @abstractmethod
    def extract(
        self, nodes: list[EventNode], context: ExtractionContext | None = None
    ) -> list[RelationEdge]:
        """Return the relations predicted among `nodes`."""


relation_extractors: Registry[RelationExtractor] = Registry("relation_extractor")
