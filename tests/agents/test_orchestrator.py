"""The orchestration substrate must be deterministic and auditable with stub
agents — no torch, no LLM — so the whole multi-agent control flow is testable
on a CPU box.
"""

from __future__ import annotations

import pytest

from finekg.agents.protocol import Agent, Blackboard, Message, Orchestrator, Stage


class _Proposer(Agent):
    def __init__(self, role: str, value: int) -> None:
        self.role = role
        self.value = value

    def act(self, board: Blackboard) -> Message:
        return Message(role=self.role, kind="propose", payload=self.value)


class _Aggregator(Agent):
    role = "aggregator"

    def act(self, board: Blackboard) -> Message:
        total = sum(m.payload for m in board.of_kind("propose"))
        return Message(role=self.role, kind="aggregate", payload=total)


def test_orchestrator_runs_stages_in_order_and_aggregates() -> None:
    orch = Orchestrator(
        [
            Stage(agents=[_Proposer("a", 1), _Proposer("b", 2)]),
            Stage(agents=[_Aggregator()]),
        ]
    )
    board = orch.run(context={"doc": "x"})
    assert board.context["doc"] == "x"
    assert [m.payload for m in board.of_kind("propose")] == [1, 2]
    assert board.latest("aggregate").payload == 3


def test_debate_rounds_repeat_agents() -> None:
    orch = Orchestrator([Stage(agents=[_Proposer("a", 1)], rounds=3)])
    board = orch.run()
    assert len(board.of_kind("propose")) == 3


def test_run_is_deterministic_with_deterministic_agents() -> None:
    orch = Orchestrator(
        [Stage(agents=[_Proposer("a", 1), _Proposer("b", 2)]), Stage(agents=[_Aggregator()])]
    )
    first = [m.payload for m in orch.run().messages]
    second = [m.payload for m in orch.run().messages]
    assert first == second


def test_blackboard_filters_by_role_and_kind() -> None:
    board = Blackboard()
    board.post(Message(role="a", kind="propose", payload=1))
    board.post(Message(role="b", kind="verify", payload=0.5))
    assert len(board.of_role("a")) == 1
    assert board.latest("verify").payload == 0.5
    assert board.of_kind("nope") == []
    assert board.latest("nope") is None


def test_orchestrator_requires_a_stage() -> None:
    with pytest.raises(ValueError):
        Orchestrator([])
