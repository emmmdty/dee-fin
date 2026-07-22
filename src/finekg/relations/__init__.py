"""Event relation extraction and event-graph construction.

Public entry point is `RelationPipeline`, which turns event nodes into an
evidence-grounded, globally-consistent `EventGraph`. Extractors and consistency
solvers are pluggable via their registries.
"""

from finekg.relations.pipeline import RelationPipeline, RelationPipelineConfig

__all__ = ["RelationPipeline", "RelationPipelineConfig"]
