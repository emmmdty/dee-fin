"""Evaluation metrics shared across stages.

Relation extraction (P/R/F1), event coreference (CoNLL: MUC/B3/CEAFe), global
consistency diagnostics, and temporal-link-prediction ranking (MRR/Hits@k).
"""

from finekg.core.eval.consistency import consistency_report
from finekg.core.eval.coreference import b_cubed, ceafe, conll_coref_f1, muc
from finekg.core.eval.faithfulness import (
    RiskCoveragePoint,
    aurc,
    expected_calibration_error,
    intervention_faithfulness,
    risk_coverage_curve,
    selective_risk_at_coverage,
)
from finekg.core.eval.forecasting import evaluate_predictions, mrr_hits, rank_of
from finekg.core.eval.relation import PRF, relation_prf

__all__ = [
    "relation_prf",
    "PRF",
    "muc",
    "b_cubed",
    "ceafe",
    "conll_coref_f1",
    "consistency_report",
    "mrr_hits",
    "rank_of",
    "evaluate_predictions",
    "intervention_faithfulness",
    "risk_coverage_curve",
    "RiskCoveragePoint",
    "selective_risk_at_coverage",
    "aurc",
    "expected_calibration_error",
]
