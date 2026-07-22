"""Successor-event prediction over event causality graphs (CGEP).

The task: given an event causality graph (ECG) and an anchor event, rank
candidate events for the anchor's successor. This mirrors CGEP as introduced by
SeDGPL (EMNLP'24 Findings, arXiv:2409.17480), whose own MAVEN build was never
released -- see `finekg.succession.data.cgep` for the rebuild protocol.

Kept separate from `finekg.forecasting`, whose `Forecaster` ABC is shaped around
entity-centric temporal quads `(subject, relation, ?, t)`; CGEP ranks *events*
from an explicit candidate set and does not fit that contract. The I/O schema
types (`ForecastQuery.candidates`, `Prediction.coverage_set`) are shared.
"""

from __future__ import annotations

__all__: list[str] = []
