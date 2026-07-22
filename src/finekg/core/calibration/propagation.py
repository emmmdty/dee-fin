"""Cross-stage conformal risk propagation (CS-CRP).

Composes the construction-stage edge-admission recall guarantee (FNR <= alpha_edge,
`relations.admission.CRCEdgeAdmission`) with the reasoning-stage drift-adaptive
coverage guarantee (miss <= alpha_pred, the `core.calibration` streaming
calibrators) into a single end-to-end selective predictor under one risk budget
``alpha_total = alpha_edge + alpha_pred``.

Why this is *not* just "conformal prediction on a graph": edge admission
**removes** candidates, so a dropped gold edge makes the answer unreachable — a
miss the reasoning calibrator never sees. CS-CRP reserves a slice of the budget
for that reachability loss and (unlike exchangeable multi-stage pipeline
conformal, e.g. PASC arXiv:2605.18812) keeps the reasoning side **drift-adaptive**,
so the *composed* coverage holds at ``1 - alpha_total`` even under regime shift.

The reasoning calibrator only ``observe``s reachable queries, so the construction
stage's FNR budget and the reasoning stage's coverage budget stay separate. The
guarantee is the union bound

    P(miss) <= P(unreachable) + P(reasoning miss | reachable) <= alpha_edge + alpha_pred,

a deliberately conservative end-to-end bound validated empirically against the
naive (no reachability budget) and Bonferroni (static, exchangeable) baselines.

The union bound is doubly conservative: the exact decomposition is

    P(miss) = P(unreachable) + P(reasoning miss | reachable) * P(reachable),

so budget reserved for reachability loss that admission does not actually spend
is wasted, and the reasoning miss only counts on the reachable fraction.
`allocate_budget_conditional` recycles both slacks: it certifies an upper bound
``u`` on the unreachability rate from held-out admission outcomes (exact
Clopper-Pearson, tightened by the CRC bound ``alpha_edge`` when available) and
runs the reasoning side at the corrected level

    alpha_pred' = (alpha_total - u) / (1 - u)  >=  alpha_total - alpha_edge,

which keeps ``P(miss) <= alpha_total`` (with probability ``1 - delta`` over the
calibration draw) while spending the recycled budget on strictly tighter sets.

Pure-Python / CPU. The construction stage that produces ``reachable`` lives in
`relations.admission` (CRC edge admission at ``alpha_edge``); this module consumes
its per-query reachability + the reasoning ranks over the admitted graph.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from finekg.core.calibration.base import build_calibrator
from finekg.core.calibration.metrics import drift_coverage_gap

__all__ = [
    "BudgetSplit",
    "allocate_budget",
    "allocate_budget_conditional",
    "binomial_upper_confidence",
    "CrossStageResult",
    "run_cross_stage",
    "compare_cross_stage_methods",
]


@dataclass(frozen=True)
class BudgetSplit:
    """An end-to-end risk budget split into construction and reasoning shares."""

    alpha_total: float
    alpha_edge: float
    alpha_pred: float


def allocate_budget(alpha_total: float, edge_share: float = 0.5) -> BudgetSplit:
    """Split the end-to-end budget into construction (edge) and reasoning (pred).

    ``alpha_edge = edge_share * alpha_total`` funds the edge-admission FNR bound;
    the remainder funds the reasoning coverage bound. Their sum is the union-bound
    end-to-end budget. ``edge_share = 0`` reserves *nothing* for reachability loss
    (the naive allocation that under-covers once admission drops gold edges).
    """
    if not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must be in (0, 1)")
    if not 0.0 <= edge_share <= 1.0:
        raise ValueError("edge_share must be in [0, 1]")
    alpha_edge = alpha_total * edge_share
    return BudgetSplit(alpha_total, alpha_edge, alpha_total - alpha_edge)


def binomial_upper_confidence(failures: int, trials: int, delta: float) -> float:
    """Exact Clopper-Pearson upper confidence bound for a binomial rate.

    Returns the smallest ``p`` such that observing ``<= failures`` events in
    ``trials`` draws has probability ``<= delta`` under ``Bin(trials, p)`` —
    i.e. with probability ``>= 1 - delta`` over the calibration draw the true
    rate is ``<= p``. Pure Python (log-space binomial CDF + bisection), so the
    calibration stack stays dependency-free.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not 0 <= failures <= trials:
        raise ValueError("failures must be in [0, trials]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if failures == trials:
        return 1.0
    if failures == 0:
        # P(Bin(n, p) = 0) = (1 - p)^n = delta solves in closed form
        return 1.0 - delta ** (1.0 / trials)

    log_choose = [
        math.lgamma(trials + 1) - math.lgamma(i + 1) - math.lgamma(trials - i + 1)
        for i in range(failures + 1)
    ]

    def log_cdf(p: float) -> float:
        log_p, log_q = math.log(p), math.log1p(-p)
        terms = [lc + i * log_p + (trials - i) * log_q for i, lc in enumerate(log_choose)]
        peak = max(terms)
        return peak + math.log(sum(math.exp(t - peak) for t in terms))

    log_delta = math.log(delta)
    lo, hi = failures / trials, 1.0 - 1e-15  # CDF is decreasing in p on (0, 1)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if log_cdf(mid) > log_delta:
            lo = mid
        else:
            hi = mid
    return hi


def allocate_budget_conditional(
    cal_reachable: Sequence[bool],
    alpha_total: float,
    *,
    alpha_edge: float | None = None,
    delta: float = 0.05,
    min_alpha_pred: float = 1e-4,
) -> BudgetSplit:
    """Reachability-corrected budget: recycle edge budget admission never spent.

    Certifies an upper bound ``u`` on the unreachability rate from held-out
    admission outcomes (``cal_reachable``) via `binomial_upper_confidence`,
    tightened by the construction-stage CRC bound ``alpha_edge`` when given
    (both are valid upper bounds, so their minimum is), and solves the exact
    composed-miss decomposition for the reasoning level:

        alpha_pred' = (alpha_total - u) / (1 - u),

    the largest level with ``u + alpha_pred' * (1 - u) <= alpha_total``. Holds
    with probability ``>= 1 - delta`` over the calibration draw; the reasoning
    side keeps its own (drift-adaptive) guarantee at ``alpha_pred'``. When the
    certified loss exhausts the whole budget the reasoning level clamps to
    ``min_alpha_pred`` (near admit-everything) instead of failing, so budget
    sweeps degrade gracefully. Returned as ``BudgetSplit(alpha_total, u,
    alpha_pred')`` — unlike `allocate_budget` the two shares need not sum to
    ``alpha_total``; that slack is exactly the recycled budget.
    """
    if not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must be in (0, 1)")
    if not cal_reachable:
        raise ValueError("cal_reachable must be non-empty")
    if alpha_edge is not None and not 0.0 < alpha_edge < 1.0:
        raise ValueError("alpha_edge must be in (0, 1)")
    unreachable = sum(1 for reach in cal_reachable if not reach)
    u_bound = binomial_upper_confidence(unreachable, len(cal_reachable), delta)
    if alpha_edge is not None:
        u_bound = min(u_bound, alpha_edge)
    reach_lcb = 1.0 - u_bound
    if reach_lcb <= 0.0:
        return BudgetSplit(alpha_total, u_bound, min_alpha_pred)
    corrected = (alpha_total - u_bound) / reach_lcb
    corrected = min(max(corrected, min_alpha_pred), 1.0 - 1e-9)
    return BudgetSplit(alpha_total, u_bound, corrected)


@dataclass(frozen=True)
class CrossStageResult:
    """Outcome of streaming one cross-stage method over a test series."""

    composed_coverage: float  # P(gold reachable AND in reasoning set)
    reasoning_coverage: float  # P(in reasoning set | reachable), realised
    reachable_rate: float  # P(gold reachable after admission)
    composed_drift_gap: float  # worst rolling deviation of composed coverage from target
    mean_set_size: float  # mean finite reasoning threshold (inf -> admit-all steps excluded)
    target: float  # 1 - alpha_total


def run_cross_stage(
    reachable: Sequence[bool],
    gold_ranks: Sequence[float],
    cal_ranks: Sequence[float],
    *,
    alpha_total: float,
    alpha_pred: float,
    reasoning: str = "aci",
    window: int = 50,
    **calibrator_kwargs: object,
) -> CrossStageResult:
    """Stream the reasoning calibrator over admitted-graph queries and compose.

    ``reachable[t]`` is whether gold survived edge admission for query ``t``;
    ``gold_ranks[t]`` is gold's rank in the reasoning over the admitted graph
    (``inf`` if unreachable). The reasoning calibrator (``reasoning`` = a
    `conformal_calibrators` registry name, e.g. ``aci`` / ``split``) is fit on
    ``cal_ranks`` at level ``alpha_pred`` and ``observe``s the realised reasoning
    outcome *only on reachable queries*, keeping its coverage budget isolated from
    the construction FNR budget. The composed event is
    ``reachable AND rank <= threshold``; ``alpha_total`` sets the drift-gap target
    ``1 - alpha_total``.
    """
    if len(reachable) != len(gold_ranks):
        raise ValueError("reachable and gold_ranks must be the same length")
    cal = build_calibrator(reasoning, alpha=alpha_pred, **calibrator_kwargs).fit(
        [float(s) for s in cal_ranks]
    )
    composed_stream: list[bool] = []
    finite_sizes: list[float] = []
    reasoning_hits = 0
    reachable_count = 0
    for reach, rank in zip(reachable, gold_ranks, strict=True):
        q = cal.threshold()
        if math.isfinite(q):
            finite_sizes.append(q)
        reasoning_hit = math.isfinite(rank) and rank <= q
        composed_stream.append(bool(reach) and reasoning_hit)
        if reach:
            reachable_count += 1
            reasoning_hits += int(reasoning_hit)
            cal.observe(
                miscovered=not reasoning_hit,
                score=float(rank) if math.isfinite(rank) else None,
            )
    n = len(composed_stream)
    target = 1.0 - alpha_total
    return CrossStageResult(
        composed_coverage=sum(composed_stream) / n if n else 0.0,
        reasoning_coverage=reasoning_hits / reachable_count if reachable_count else 0.0,
        reachable_rate=reachable_count / n if n else 0.0,
        composed_drift_gap=drift_coverage_gap(composed_stream, target, window),
        mean_set_size=(sum(finite_sizes) / len(finite_sizes)) if finite_sizes else math.inf,
        target=target,
    )


def compare_cross_stage_methods(
    reachable: Sequence[bool],
    gold_ranks: Sequence[float],
    cal_ranks: Sequence[float],
    *,
    alpha_total: float = 0.1,
    edge_share: float = 0.5,
    window: int = 50,
    gamma: float = 0.05,
    cal_reachable: Sequence[bool] | None = None,
    delta: float = 0.05,
) -> dict[str, CrossStageResult]:
    """The headline comparison for the CS-CRP experiment.

    - ``naive`` — spend the whole budget on reasoning (``alpha_pred = alpha_total``)
      with static split conformal: reserves nothing for reachability loss, so it
      under-covers once admission drops gold edges.
    - ``bonferroni`` — split the budget (PASC-style) but keep static split
      reasoning: valid under exchangeability, drifts under regime shift.
    - ``cs_crp`` — split the budget *and* keep the reasoning side drift-adaptive
      (ACI): composed coverage holds at ``1 - alpha_total`` under drift.
    - ``cs_crp_cond`` (only when ``cal_reachable`` is given) — replace the fixed
      split with `allocate_budget_conditional`: certify the actual reachability
      loss on held-out admission outcomes and recycle the slack into a higher
      reasoning level, i.e. strictly tighter sets under the same total budget.
    """
    split = allocate_budget(alpha_total, edge_share)
    common = dict(window=window)
    results = {
        "naive": run_cross_stage(
            reachable, gold_ranks, cal_ranks,
            alpha_total=alpha_total, alpha_pred=alpha_total, reasoning="split", **common,
        ),
        "bonferroni": run_cross_stage(
            reachable, gold_ranks, cal_ranks,
            alpha_total=alpha_total, alpha_pred=split.alpha_pred, reasoning="split", **common,
        ),
        "cs_crp": run_cross_stage(
            reachable, gold_ranks, cal_ranks,
            alpha_total=alpha_total, alpha_pred=split.alpha_pred, reasoning="aci",
            gamma=gamma, **common,
        ),
    }
    if cal_reachable is not None:
        conditional = allocate_budget_conditional(
            cal_reachable, alpha_total, alpha_edge=split.alpha_edge, delta=delta
        )
        results["cs_crp_cond"] = run_cross_stage(
            reachable, gold_ranks, cal_ranks,
            alpha_total=alpha_total, alpha_pred=conditional.alpha_pred, reasoning="aci",
            gamma=gamma, **common,
        )
    return results
