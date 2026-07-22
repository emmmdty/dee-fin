"""Streaming conformal calibrator: the verifier's third duty (risk controller).

A `Calibrator` turns a pool of calibration nonconformity scores into a per-step
threshold and, crucially, may *adapt* that threshold to the realised
(mis)coverage so the long-run coverage stays at 1 - alpha even when the score
distribution drifts — the financial regime-shift setting that plain split
conformal (exchangeability-only) cannot handle.

Nonconformity convention (forecasting rank setting): lower = more conforming. A
threshold ``q`` admits the candidates whose score ``<= q``; for ranked
candidates that means the coverage set is the top-``ceil(q)`` of the ranking.

Two responsibilities, deliberately split so adaptive calibration composes with
the multi-agent blackboard (the calibrator agent never sees gold):

- ``threshold()`` — produce the current admission threshold (read-only);
- ``observe(miscovered, score)`` — feed back the realised outcome of the
  just-served query so the *next* query adapts. Static calibrators ignore it.

Pluggable by name via the `conformal_calibrators` registry, mirroring the
forecaster / extractor registries, so swapping ``split`` -> ``aci`` is a config
change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from finekg.core.registry import Registry

__all__ = ["Calibrator", "conformal_calibrators", "build_calibrator"]

conformal_calibrators: Registry[Calibrator] = Registry("conformal_calibrator")


class Calibrator(ABC):
    """A streaming conformal threshold source (static or drift-adaptive)."""

    alpha: float = 0.1

    @abstractmethod
    def fit(self, cal_scores: Sequence[float]) -> Calibrator:
        """Seed state from calibration nonconformity scores; returns self."""

    @abstractmethod
    def threshold(self) -> float:
        """Current admission threshold (a rank/score; ``+inf`` admits all)."""

    def observe(self, miscovered: bool, score: float | None = None) -> None:
        """Feed back the realised outcome of the query just served.

        ``miscovered`` is True iff the gold answer fell outside the admitted
        set; ``score`` is its realised nonconformity (e.g. gold rank). Static
        calibrators are exchangeability-only and ignore this; adaptive ones
        update their state for the next query.
        """
        _ = (miscovered, score)  # static default: no adaptation


def build_calibrator(name: str, **kwargs: object) -> Calibrator:
    """Instantiate a calibrator by registry name (config-driven selection)."""
    return conformal_calibrators.create(name, **kwargs)
