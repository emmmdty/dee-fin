"""Multi-agent orchestration for evidence-grounded graph construction and
temporal reasoning.

The protocol here is stage-agnostic; concrete role agents live in
`finekg.relations.agents` (and per-stage agent packages) and register into
`agent_roles`.
"""

from finekg.agents.protocol import (
    Agent,
    Blackboard,
    Message,
    Orchestrator,
    Stage,
    Verifier,
    agent_roles,
)

__all__ = [
    "Agent",
    "Blackboard",
    "Message",
    "Orchestrator",
    "Stage",
    "Verifier",
    "agent_roles",
]
