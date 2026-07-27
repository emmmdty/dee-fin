"""Relation extractors: interface, registry, and implementations.

Importing this package registers the available extractors. The neural extractors
import torch lazily, so importing the package is safe without a GPU.
"""

from ekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)
from ekg.relations.extractor.heuristic import HeuristicRelationExtractor
from ekg.relations.extractor.llm import LLMRelationExtractor
from ekg.relations.extractor.supervised import SupervisedRelationExtractor

__all__ = [
    "ExtractionContext",
    "RelationExtractor",
    "relation_extractors",
    "HeuristicRelationExtractor",
    "LLMRelationExtractor",
    "SupervisedRelationExtractor",
]
