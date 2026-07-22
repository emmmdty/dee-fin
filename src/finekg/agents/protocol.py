"""Multi-agent orchestration protocol (CPU-side; LLM agents run on the server).

The relation and forecasting stages are organised as a small society of
role-specialised agents that write to a shared `Blackboard`, coordinated by an
`Orchestrator` and gated by a `Verifier`. This module is the stage-agnostic
substrate — messages, blackboard, debate rounds, and the staged control flow.
Concrete agents (proposers, verifiers, arbiters, reasoners, calibrators) live in
the `relations/` and `forecasting/` domains and self-register in `agent_roles`,
so the orchestration never depends on a particular stage — mirroring the
extractor / forecaster registries.

Everything here is pure-Python and deterministic: the whole control flow runs
and is unit-tested on a CPU box with stub agents; only agents that wrap an LLM
need the `llm` extra and a GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from finekg.core.registry import Registry

__all__ = [
    "Message",
    "Blackboard",
    "Agent",
    "Verifier",
    "Stage",
    "Orchestrator",
    "agent_roles",
]


@dataclass
class Message:
    """One agent's contribution, posted to the blackboard.

    `kind` groups messages by purpose ("propose" / "critique" / "verify" /
    "aggregate") so later agents read exactly what they need; `payload` is the
    stage-specific content (edges, predictions, scores, ...).
    """

    role: str
    kind: str
    payload: Any = None
    rationale: str = ""
    confidence: float = 1.0


@dataclass
class Blackboard:
    """Shared working memory for one orchestration episode.

    Agents read prior messages and `context`, then post their own. Keeping all
    contributions (rather than overwriting) makes the debate auditable — the
    transcript itself is provenance. The same rule binds payloads: an agent
    that annotates or repairs another agent's output must post a copy, never
    mutate the original message's payload in place.
    """

    context: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)

    def post(self, message: Message) -> None:
        self.messages.append(message)

    def of_kind(self, kind: str) -> list[Message]:
        return [m for m in self.messages if m.kind == kind]

    def of_role(self, role: str) -> list[Message]:
        return [m for m in self.messages if m.role == role]

    def latest(self, kind: str) -> Message | None:
        found = self.of_kind(kind)
        return found[-1] if found else None


class Agent(ABC):
    """A role-specialised participant.

    `act` reads the blackboard and returns this agent's message; the
    orchestrator posts it. Implementations are deterministic when their
    underlying component is (stub / heuristic), and stochastic only when they
    wrap an LLM — so the control flow is identical on CPU and GPU.
    """

    role: str = "agent"

    @abstractmethod
    def act(self, board: Blackboard) -> Message:
        """Inspect the blackboard and produce this agent's contribution."""


class Verifier(ABC):
    """Scores a candidate's evidence faithfulness in [0, 1].

    The single "currency" of the system: a grounding/faithfulness verifier
    scores edges (relation stage) or forecast evidence chains (forecasting
    stage); callers admit, abstain or rank by this score. Kept separate from
    `Agent` so the scoring logic stays reusable and unit-testable outside any
    orchestration.
    """

    @abstractmethod
    def score(self, candidate: Any, board: Blackboard) -> float:
        """Return a faithfulness score in [0, 1] for `candidate`."""


@dataclass
class Stage:
    """One step of the orchestrated flow.

    A group of agents run together for `rounds` debate rounds. Order within a
    round does not imply precedence; stages run sequentially so a later stage
    sees everything earlier stages produced.
    """

    agents: list[Agent]
    rounds: int = 1


class Orchestrator:
    """Runs an ordered list of `Stage`s over a shared blackboard.

    Each stage's agents post for `rounds` rounds; later stages read everything
    earlier stages produced. Fully deterministic given deterministic agents, so
    the multi-agent control logic is testable with stubs and behaves identically
    on CPU and GPU.
    """

    def __init__(self, stages: list[Stage]) -> None:
        if not stages:
            raise ValueError("Orchestrator needs at least one stage")
        self.stages = stages

    def run(self, context: dict[str, Any] | None = None) -> Blackboard:
        board = Blackboard(context=dict(context or {}))
        for stage in self.stages:
            for _ in range(max(1, stage.rounds)):
                for agent in stage.agents:
                    board.post(agent.act(board))
        return board


# Concrete agents (relations / forecasting domains) self-register here so the
# pipelines build them by name from a config, like the other registries.
agent_roles: Registry[Agent] = Registry("agent_role")
