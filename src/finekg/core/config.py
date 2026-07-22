"""Lightweight YAML config loading with shallow overrides.

Experiments are configured by YAML under `configs/`. We keep configs as plain
nested dicts (not rigid models) so adding a knob for a future method never
breaks older configs — consistent with the extend-don't-repurpose rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "merge_overrides"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overrides` into a copy of `base` (override wins)."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_overrides(out[key], value)
        else:
            out[key] = value
    return out
