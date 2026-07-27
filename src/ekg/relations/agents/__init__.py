"""Relation-stage agents: a proposer committee, a grounding/faithfulness
verifier, and a consistency arbiter.

Importing this package registers the agents in `agent_roles`, so the relation
pipeline builds them by name from a config — exactly like the extractor and
consistency-solver registries.
"""

from ekg.relations.agents.consistency_arbiter import ConsistencyArbiterAgent
from ekg.relations.agents.grounding_verifier import (
    GroundingFaithfulnessVerifier,
    GroundingVerifierAgent,
    edge_grounding_faithfulness,
)
from ekg.relations.agents.proposer import ProposerAgent

__all__ = [
    "ProposerAgent",
    "GroundingVerifierAgent",
    "GroundingFaithfulnessVerifier",
    "edge_grounding_faithfulness",
    "ConsistencyArbiterAgent",
]
