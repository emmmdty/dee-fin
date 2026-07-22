"""M3a: the selective conformal head over CGEP (the reasoning-stage predictor).

Turns a `SuccessorPredictor`'s candidate scores into a conformal *prediction set*
with a coverage guarantee and traces the risk-coverage curve -- the Ch3 headline
formal contribution, which holds whether or not the point-prediction MRR improves.

The bridge is small on purpose. `core.calibration.propagation.run_cross_stage`
already streams a conformal calibrator (split / ACI / ...) over per-query gold
ranks and reports coverage, drift, and set size; all that is missing is turning
CGEP instances into those ranks. `cgep_gold_ranks` does exactly that -- score the
candidates, read gold's rank -- and `selective_report` sweeps the risk budget over
one such pass. Being predictor-agnostic, the head runs on the torch-free `random`
and `frequency` baselines on CPU exactly as it will on SeDGPL's GPU scores.

Reachability: on the gold ECG every query is reachable (`reachable` all True), so
the composed guarantee reduces to the reasoning-stage coverage ``1 - alpha``. The
cross-stage reachability budget of CS-CRP (SPEC 4.3) only bites once a *constructed*
ECG prunes candidate edges -- the killer experiment, which supplies ``reachable``
from `relations.admission` and calls these same `propagation` primitives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from finekg.core.calibration.propagation import run_cross_stage
from finekg.succession.data.cgep import CgepInstance
from finekg.succession.metrics import sedgpl_rank, strict_rank
from finekg.succession.predictor import SuccessorPredictor, UnscorableInstance

__all__ = ["DEFAULT_ALPHAS", "cgep_gold_ranks", "selective_report"]

# Risk budgets swept to trace the risk-coverage curve, tight -> loose.
DEFAULT_ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.3)


def cgep_gold_ranks(
    predictor: SuccessorPredictor,
    instances: Sequence[CgepInstance],
    *,
    strict: bool = False,
) -> tuple[list[float], list[bool]]:
    """Gold's rank in each instance's candidate set, plus a reachability flag.

    The rank (SeDGPL-optimistic, or ``strict`` for the pessimistic tie-break) is
    the nonconformity score the calibrator consumes: lower means gold is easier to
    cover. An instance the predictor cannot score at all becomes ``inf`` -- a
    guaranteed miss, never a flat score that would win every optimistic tie. Every
    query is ``reachable`` here: reachability is what a constructed graph's edge
    admission removes, not something scoring decides.
    """
    rank_of = strict_rank if strict else sedgpl_rank
    ranks: list[float] = []
    for instance in instances:
        try:
            scores = predictor.score(instance)
        except UnscorableInstance:
            ranks.append(math.inf)
            continue
        ranks.append(float(rank_of(scores, instance.label)))
    return ranks, [True] * len(instances)


def selective_report(
    predictor: SuccessorPredictor,
    cal_instances: Sequence[CgepInstance],
    test_instances: Sequence[CgepInstance],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    reasoning: str = "aci",
    strict: bool = False,
    window: int = 50,
) -> dict[str, object]:
    """Risk-coverage curve for the selective head, computing gold ranks once.

    Ranks over the held-out calibration and test splits are read a single time,
    then every ``alpha`` reuses them -- so on GPU one SeDGPL inference pass yields
    the whole curve. Each point reports the realised coverage against its target
    ``1 - alpha`` and the mean rank threshold (~ prediction-set size); a predictor
    that cannot separate gold pays for coverage in set size, which is the
    risk-coverage trade the curve makes visible.
    """
    cal_ranks, _ = cgep_gold_ranks(predictor, cal_instances, strict=strict)
    test_ranks, reachable = cgep_gold_ranks(predictor, test_instances, strict=strict)
    curve: list[dict[str, float]] = []
    for alpha in alphas:
        result = run_cross_stage(
            reachable, test_ranks, cal_ranks,
            alpha_total=float(alpha), alpha_pred=float(alpha),
            reasoning=reasoning, window=window,
        )
        curve.append({
            "alpha": float(alpha),
            "target": result.target,
            "coverage": result.composed_coverage,
            "mean_set_size": result.mean_set_size,
            "drift_gap": result.composed_drift_gap,
        })
    return {
        "reasoning": reasoning,
        "strict": strict,
        "n_cal": len(cal_ranks),
        "n_test": len(test_ranks),
        "reachable_rate": (sum(reachable) / len(reachable)) if reachable else 0.0,
        "curve": curve,
    }
