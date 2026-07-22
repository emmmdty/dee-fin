"""Relation extractors: interface, registry, and implementations.

Importing this package registers the available extractors. The neural extractor
imports torch lazily, so importing the package is safe without a GPU.
"""

from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)
from finekg.relations.extractor.heuristic import HeuristicRelationExtractor
from finekg.relations.extractor.llm import LLMRelationExtractor

__all__ = [
    "ExtractionContext",
    "RelationExtractor",
    "relation_extractors",
    "HeuristicRelationExtractor",
    "LLMRelationExtractor",
]
