"""Verifiable reward components for relation-extraction completions (pure CPU).

Each component scores one (completion, windowed document) pair in [0, 1] and
reuses the exact kernel that gates outputs at inference time, so training
optimizes what verification later checks:

- ``format``      the completion is well-formed JSON with a `relations` list,
                  scaled by the fraction of items the tolerant parser accepts;
- ``grounding``   fraction of predicted edges whose `evidence_quote` verifies
                  as a real substring of the source text (`ground_relations`);
- ``consistency`` fraction of predicted edges surviving contradiction repair by
                  the greedy solver (causal/temporal cycles, duplicate pairs);
- ``task_f1``     micro-F1 against the window's gold edges (`relation_prf`).

Anti-hacking invariants: on a window *with* gold edges an empty prediction
scores 0 on grounding / consistency / task (verifiability cannot be earned by
saying nothing); on a window *without* gold edges an empty prediction is the
correct answer and scores 1 (so hallucinated-but-grounded edges cannot outearn
honest silence there). A quote longer than `max_quote_chars` counts as
ungrounded (copying the whole document is not evidence).
"""

from __future__ import annotations

import json
import re

from finekg.core.eval.relation import relation_prf
from finekg.core.registry import Registry
from finekg.core.schema import EventGraph, RelationEdge
from finekg.relations.consistency import GreedyConsistencySolver
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.base import ExtractionContext
from finekg.relations.extractor.llm import parse_relation_json
from finekg.relations.grounding import ground_relations
from finekg.rl.reward import CompositeReward, RewardFn, build_composite

__all__ = [
    "relation_reward_components",
    "FormatReward",
    "GroundingReward",
    "ConsistencyReward",
    "TaskF1Reward",
    "build_relation_reward",
]

relation_reward_components: Registry[RewardFn] = Registry("relation_reward_component")


class _SharedParse:
    """Single-slot memo: the components score the same completion in sequence,
    so parse the JSON once instead of once per component. Keyed by
    (doc_id, completion) — doc ids are unique and content-stable within a run
    (DocStore enforces this). The trainer calls rewards from one thread; this
    is deliberately not a concurrency-safe cache."""

    def __init__(self) -> None:
        self._key: tuple[str, str] | None = None
        self._edges: list[RelationEdge] = []

    def __call__(self, completion: str, doc: RelationDocument) -> list[RelationEdge]:
        key = (doc.doc_id, completion)
        if key != self._key:
            self._key = key
            self._edges = parse_relation_json(completion, doc.nodes, doc_id=doc.doc_id)
        return self._edges


_parse_edges = _SharedParse()


def _empty_score(doc: RelationDocument) -> float:
    """Score of an empty prediction: 1.0 iff the window has no gold edges.

    Silence is the right answer on a no-gold window — without this, grounded
    hallucinations (format+grounding+consistency pay) would outearn honesty
    there. On gold-bearing windows emptiness stays at 0: verifiability cannot
    be earned by saying nothing.
    """
    return 1.0 if not doc.gold_edges else 0.0


@relation_reward_components.register("format")
class FormatReward:
    """JSON parses and carries a `relations` list; scaled by valid-item ratio.

    The tolerant parser drops malformed items, unknown types, out-of-range and
    self-loop indices — so hallucinated event indices cost format reward too.
    An explicit empty `relations` list is well-formed; whether emptiness is
    rewarded or starved overall is decided by the other components via
    `_empty_score` (gold-bearing window: starved; no-gold window: rewarded).
    """

    def __call__(self, completion: str, doc: RelationDocument) -> float:
        match = re.search(r"\{.*\}", completion, flags=re.DOTALL)
        if not match:
            return 0.0
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return 0.0
        items = payload.get("relations") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return 0.0
        if not items:
            return 1.0
        return len(_parse_edges(completion, doc)) / len(items)


@relation_reward_components.register("grounding")
class GroundingReward:
    """Fraction of predicted edges whose evidence quote verifies in the text.

    An empty prediction scores via `_empty_score`: 0 on gold-bearing windows,
    1 on no-gold windows (honest silence beats grounded hallucination there).
    """

    def __init__(self, max_quote_chars: int = 60) -> None:
        if max_quote_chars <= 0:
            raise ValueError("max_quote_chars must be positive")
        self.max_quote_chars = max_quote_chars

    def __call__(self, completion: str, doc: RelationDocument) -> float:
        edges = _parse_edges(completion, doc)
        if not edges:
            return _empty_score(doc)
        # Oversized quotes count as ungrounded instead of being passed along,
        # so "quote the whole document" cannot buy grounding reward.
        eligible = [
            e
            for e in edges
            if all(len(span.text) <= self.max_quote_chars for span in e.evidence)
        ]
        # Grounding resolves text via each *node's* doc_id, which a windowed
        # document (doc_id "<orig>#w<j>") no longer matches — key by node ids.
        doc_text = {node.doc_id: doc.doc_text for node in doc.nodes}
        doc_text[doc.doc_id] = doc.doc_text
        result = ground_relations(
            eligible, doc.nodes, ExtractionContext(doc_text=doc_text), require_evidence=True
        )
        return len(result.kept) / len(edges)


@relation_reward_components.register("consistency")
class ConsistencyReward:
    """Fraction of predicted edges surviving the greedy contradiction repair.

    A violation-rate formulation (closure is disabled) so the model cannot
    inflate the reward by emitting extra implied edges; causal/temporal cycles
    and duplicate pairs are exactly what the solver removes. Empty predictions
    score via `_empty_score`.
    """

    def __call__(self, completion: str, doc: RelationDocument) -> float:
        edges = _parse_edges(completion, doc)
        if not edges:
            return _empty_score(doc)
        graph = EventGraph(nodes={n.event_id: n for n in doc.nodes}, edges=edges)
        solved = GreedyConsistencySolver(close_temporal=False).solve(graph)
        return min(1.0, len(solved.edges) / len(edges))


@relation_reward_components.register("task_f1")
class TaskF1Reward:
    """Micro-F1 of predicted edges against the window's gold edges.

    Empty-vs-empty is exact agreement, so an empty prediction on a no-gold
    window earns 1.0 (`_empty_score`) rather than the 0.0 the bare F1
    convention would give.
    """

    def __call__(self, completion: str, doc: RelationDocument) -> float:
        edges = _parse_edges(completion, doc)
        if not edges:
            return _empty_score(doc)
        return float(relation_prf(edges, doc.gold_edges)["micro"]["f1"])


def build_relation_reward(specs: list[dict]) -> CompositeReward:
    """Composite verifiable reward from the `rewards:` config section."""
    return build_composite(relation_reward_components, specs)
