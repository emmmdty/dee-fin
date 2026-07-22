"""M3b: controlled cross-stage sweep of the CS-CRP composition on CGEP.

The cross-stage reachability budget (SPEC 4.3) only earns its keep when upstream
construction drops gold answers. On CGEP that construction stage is a MAVEN
causal+subevent extractor -- and ours recovers ~0.4% of gold causal and 0% of
subevent edges, so a *real* constructed ECG is degenerate (no >=4-node component
survives, reachability loss ~1.0). A single point at 100% loss demonstrates
nothing, so here reachability is a **controlled variable**: real SeDGPL reasoning
ranks over the gold ECG, with a target fraction of queries marked unreachable, and
the sweep traces coverage as that loss grows. A strong extractor that induces the
loss from real predictions is future work; this isolates the calibration
behaviour the reachability budget is responsible for.

The claim the sweep makes visible: as reachability loss rises, `naive` conformal
(whole budget on reasoning, nothing reserved) under-covers, while CS-CRP -- and
especially its conditional recycling (`cs_crp_cond`, which certifies the loss from
held-out reachability) -- holds composed coverage at ``1 - alpha_total``. All of
that math lives in `core.calibration.propagation`; this module only induces the
loss and sweeps it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from finekg.core.calibration.propagation import compare_cross_stage_methods

__all__ = ["DEFAULT_LOSSES", "cross_stage_sweep", "induce_reachability"]

# Reachability losses swept, from an intact graph up past a moderate budget.
DEFAULT_LOSSES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)

_METHODS = ("naive", "bonferroni", "cs_crp", "cs_crp_cond")


def induce_reachability(
    n: int,
    loss: float,
    rng: random.Random,
    ranks: Sequence[float] | None = None,
    mode: str = "random",
) -> list[bool]:
    """A length-`n` reachability mask with ``round(loss * n)`` queries dropped.

    ``mode="random"`` drops a uniform subset (loss independent of difficulty).
    ``mode="hardest"`` drops the queries with the largest ranks, modelling a weak
    extractor that misses exactly the edges its reasoner also finds hard -- the
    adversarial case for the composition, since the dropped mass is the tail the
    reasoning calibrator would have had to cover anyway.
    """
    k = round(loss * n)
    if k <= 0:
        return [True] * n
    if mode == "hardest" and ranks is not None:
        order = sorted(range(n), key=lambda i: (ranks[i], i), reverse=True)
        unreachable = set(order[:k])
    else:
        unreachable = set(rng.sample(range(n), min(k, n)))
    return [i not in unreachable for i in range(n)]


def cross_stage_sweep(
    cal_ranks: Sequence[float],
    test_ranks: Sequence[float],
    *,
    losses: Sequence[float] = DEFAULT_LOSSES,
    alpha_total: float = 0.2,
    edge_share: float = 0.5,
    mode: str = "random",
    seed: int = 209,
    window: int = 50,
) -> dict[str, object]:
    """Sweep reachability loss and report each method's composed coverage.

    For every ``loss`` a fresh reachability mask is drawn for both the calibration
    and test streams; the reasoning calibrator sees only reachable calibration
    ranks, unreachable test queries are charged as misses, and
    `compare_cross_stage_methods` returns the four methods' realised composed
    coverage and mean set size. `naive` is the strawman (no reachability budget),
    `cs_crp_cond` the conditional recycler.
    """
    rng = random.Random(seed)
    curve: list[dict[str, float]] = []
    for loss in losses:
        reachable_test = induce_reachability(len(test_ranks), loss, rng, test_ranks, mode)
        reachable_cal = induce_reachability(len(cal_ranks), loss, rng, cal_ranks, mode)
        masked_test = [
            r if reach else math.inf for r, reach in zip(test_ranks, reachable_test, strict=True)
        ]
        cal_reachable_ranks = [
            r for r, reach in zip(cal_ranks, reachable_cal, strict=True) if reach
        ]
        results = compare_cross_stage_methods(
            reachable_test, masked_test, cal_reachable_ranks,
            alpha_total=alpha_total, edge_share=edge_share,
            cal_reachable=reachable_cal, window=window,
        )
        row: dict[str, float] = {
            "loss": float(loss),
            "reachable_rate": sum(reachable_test) / len(reachable_test),
            "target": 1.0 - alpha_total,
        }
        for method in _METHODS:
            row[f"{method}_coverage"] = results[method].composed_coverage
            row[f"{method}_set_size"] = results[method].mean_set_size
        curve.append(row)
    return {"alpha_total": alpha_total, "edge_share": edge_share, "mode": mode, "curve": curve}
