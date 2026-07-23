"""Relation extractors: interface, registry, and implementations.

Importing this package registers the available extractors. The neural extractors
import torch lazily, so importing the package is safe without a GPU.
"""

from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)
from finekg.relations.extractor.heuristic import HeuristicRelationExtractor
from finekg.relations.extractor.llm import LLMRelationExtractor
from finekg.relations.extractor.supervised import SupervisedRelationExtractor

__all__ = [
    "ExtractionContext",
    "RelationExtractor",
    "relation_extractors",
    "HeuristicRelationExtractor",
    "LLMRelationExtractor",
    "SupervisedRelationExtractor",
]
