"""Successor-event prediction over event causality graphs (CGEP).

The task: given an event causality graph (ECG) and an anchor event, rank
candidate events for the anchor's successor. This mirrors CGEP as introduced by
SeDGPL (EMNLP'24 Findings, arXiv:2409.17480), whose own MAVEN build was never
released -- see `finekg.succession.data.cgep` for the rebuild protocol.

Kept separate from the legacy entity-centric temporal-quad contract: CGEP ranks
*events* from an explicit candidate set. This package is the v4 Ch4 downstream
reasoning domain; selective/conformal components are reliability modules rather
than a standalone chapter headline.
"""

from __future__ import annotations

__all__: list[str] = []
