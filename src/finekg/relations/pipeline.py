"""Relation stage orchestrator: nodes -> graph.

Wires the swappable components into one flow:

    event nodes
      -> RelationExtractor.extract      (heuristic baseline | LoRA LLM)
      -> ground_relations               (drop fabricated / ungrounded edges)
      -> ConsistencySolver.solve        (acyclic causal, closed temporal, …)
      -> EventGraph

Components are selected by name from the registries, so an experiment config
chooses the method without any code change here. Importing this module imports
the implementation packages, which registers them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finekg.agents.protocol import Orchestrator, Stage, agent_roles
from finekg.core.config import load_config
from finekg.core.schema import EventGraph, EventNode

# Importing these packages registers their implementations.
from finekg.relations import agents  # noqa: F401  (registers the relation-stage agents)
from finekg.relations.consistency import consistency_solvers
from finekg.relations.extractor import relation_extractors
from finekg.relations.extractor.base import ExtractionContext
from finekg.relations.grounding import ground_relations

__all__ = [
    "RelationPipelineConfig",
    "RelationPipeline",
    "MultiAgentRelationConfig",
    "MultiAgentRelationPipeline",
]


@dataclass
class RelationPipelineConfig:
    extractor: str = "heuristic"
    extractor_kwargs: dict[str, Any] = field(default_factory=dict)
    require_evidence: bool = True
    consistency: str = "greedy"
    consistency_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationPipelineConfig:
        section = data.get("relations", data)
        return cls(
            extractor=section.get("extractor", "heuristic"),
            extractor_kwargs=section.get("extractor_kwargs", {}),
            require_evidence=section.get("require_evidence", True),
            consistency=section.get("consistency", "greedy"),
            consistency_kwargs=section.get("consistency_kwargs", {}),
        )


class RelationPipeline:
    """Build an evidence-grounded, globally-consistent event graph from nodes."""

    def __init__(self, config: RelationPipelineConfig | None = None) -> None:
        self.config = config or RelationPipelineConfig()
        self.extractor = relation_extractors.create(
            self.config.extractor, **self.config.extractor_kwargs
        )
        self.solver = consistency_solvers.create(
            self.config.consistency, **self.config.consistency_kwargs
        )

    @classmethod
    def from_config(cls, path: str | Path) -> RelationPipeline:
        return cls(RelationPipelineConfig.from_dict(load_config(path)))

    def build_graph(
        self, nodes: list[EventNode], context: ExtractionContext | None = None
    ) -> EventGraph:
        raw_edges = self.extractor.extract(nodes, context)
        grounding = ground_relations(
            raw_edges, nodes, context, require_evidence=self.config.require_evidence
        )
        graph = EventGraph(
            nodes={n.event_id: n for n in nodes},
            edges=grounding.kept,
            metadata={
                "extractor": self.config.extractor,
                "consistency": self.config.consistency,
                "edges_raw": str(len(raw_edges)),
                "edges_dropped_ungrounded": str(len(grounding.dropped)),
            },
        )
        return self.solver.solve(graph)


@dataclass
class MultiAgentRelationConfig:
    """Config for the multi-agent graph builder.

    A list of proposer specs (role + the relation types it owns), a verifier
    toggle/threshold, and the arbiter's consistency solver. `extractor` selects
    the engine every proposer wraps (`heuristic` on CPU, `llm` on the server).
    """

    proposers: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"role": "coref", "relation_types": ["coreference"]},
            {"role": "temporal", "relation_types": ["temporal"]},
            {"role": "causal", "relation_types": ["causal", "subevent"]},
        ]
    )
    extractor: str = "heuristic"
    extractor_kwargs: dict[str, Any] = field(default_factory=dict)
    use_verifier: bool = True
    verifier_kwargs: dict[str, Any] = field(default_factory=lambda: {"threshold": 0.5})
    consistency: str = "greedy"
    consistency_kwargs: dict[str, Any] = field(default_factory=lambda: {"close_temporal": True})
    debate_rounds: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiAgentRelationConfig:
        section = data.get("relations", data)
        default = cls()
        return cls(
            proposers=section.get("proposers", default.proposers),
            extractor=section.get("extractor", default.extractor),
            extractor_kwargs=section.get("extractor_kwargs", {}),
            use_verifier=section.get("use_verifier", default.use_verifier),
            verifier_kwargs=section.get("verifier_kwargs", default.verifier_kwargs),
            consistency=section.get("consistency", default.consistency),
            consistency_kwargs=section.get("consistency_kwargs", default.consistency_kwargs),
            debate_rounds=section.get("debate_rounds", default.debate_rounds),
        )


class MultiAgentRelationPipeline:
    """Build an evidence-grounded, globally-consistent event graph with a society
    of agents: a proposer committee, a grounding/faithfulness verifier (admits or
    abstains per edge), and a consistency arbiter.

    Same signature as `RelationPipeline` (`nodes -> EventGraph`), so the two are
    interchangeable in evaluation; the difference is *how* the graph is built and
    that every admitted edge carries a faithfulness score.
    """

    def __init__(self, config: MultiAgentRelationConfig | None = None) -> None:
        self.config = config or MultiAgentRelationConfig()
        self.proposers = [
            agent_roles.create(
                "relation_proposer",
                role=spec.get("role", "relation_proposer"),
                relation_types=spec.get("relation_types"),
                extractor=self.config.extractor,
                extractor_kwargs=self.config.extractor_kwargs,
            )
            for spec in self.config.proposers
        ]
        self.verifier = (
            agent_roles.create("grounding_verifier", **self.config.verifier_kwargs)
            if self.config.use_verifier
            else None
        )
        self.arbiter = agent_roles.create(
            "consistency_arbiter",
            solver=self.config.consistency,
            solver_kwargs=self.config.consistency_kwargs,
        )

    @classmethod
    def from_config(cls, path: str | Path) -> MultiAgentRelationPipeline:
        return cls(MultiAgentRelationConfig.from_dict(load_config(path)))

    def build_graph(
        self, nodes: list[EventNode], context: ExtractionContext | None = None
    ) -> EventGraph:
        stages = [Stage(self.proposers, rounds=self.config.debate_rounds)]
        if self.verifier is not None:
            stages.append(Stage([self.verifier]))
        stages.append(Stage([self.arbiter]))
        board = Orchestrator(stages).run(context={"nodes": nodes, "ext_context": context})
        result = board.latest("aggregate")
        if result is None:  # pragma: no cover - the arbiter always aggregates
            return EventGraph(nodes={n.event_id: n for n in nodes})
        return result.payload
