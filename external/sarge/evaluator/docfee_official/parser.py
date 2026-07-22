from __future__ import annotations

import ast
import json
from typing import Any


def parse_model_output(text: str) -> tuple[list[dict[str, Any]] | None, bool]:
    """Parse DocFEE-style model output without unsafe eval()."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None, False
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else None, False
    except (SyntaxError, ValueError):
        return None, True
