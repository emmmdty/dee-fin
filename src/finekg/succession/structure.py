"""M2 structural signal for EeCE's fourth stream: reachability to the anchor.

Per event token, one bit: does that event causally reach the query anchor over
the *directed* surviving edges? An event that does is *upstream causal evidence*
for what the mask predicts; one that does not is downstream, a sibling, or a
branch. This is the reachability currency of CS-CRP (SPEC 4.3) brought down to
the node level, and empirically the only per-node structural feature that carries
out-of-sample signal about a query's real difficulty (a 5-fold CV probe on the
gold ranks: reach-alone R^2 0.032, adding degree or proximity does not improve
it) -- so the stream is this single bit, not a wider vector.

Pure graph facts over `(head, relation, tail)` tuples: the relation label is
irrelevant to direction, and the gold successor is never among these nodes (it is
the mask), so no answer leaks in. CPU-only; the embedding that consumes the bit
lives in `model.EeCE`, behind the torch guard.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence

from finekg.succession.linearize import Edge

__all__ = ["event_reach_anchor", "reaches_anchor"]


def reaches_anchor(edges: Sequence[Edge], anchor: int) -> set[int]:
    """Nodes that reach `anchor` by following directed head->tail edges.

    Includes `anchor` itself (it trivially reaches itself). Implemented as a BFS
    over the reversed graph seeded at the anchor, so the cost is one traversal.
    """
    reverse: dict[int, set[int]] = defaultdict(set)
    for head, _, tail in edges:
        reverse[tail].add(head)
    seen = {anchor}
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        for predecessor in reverse[node]:
            if predecessor not in seen:
                seen.add(predecessor)
                queue.append(predecessor)
    return seen


def event_reach_anchor(
    edges: Sequence[Edge], event_nodes: Sequence[int], anchor: int
) -> list[int]:
    """1 if each event node reaches the anchor over `edges`, else 0.

    `event_nodes` is the template's per-token node list (`encode.event_token_nodes`),
    so the result aligns one-to-one with the event tokens -- repeats and all.
    """
    reachable = reaches_anchor(edges, anchor)
    return [1 if node in reachable else 0 for node in event_nodes]
