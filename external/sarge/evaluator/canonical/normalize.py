from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Strict surface normalization used by all main metrics."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _SPACE_RE.sub(" ", normalized.strip())
    return normalized


def normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = normalize_text(value)
    return normalized if normalized else None
