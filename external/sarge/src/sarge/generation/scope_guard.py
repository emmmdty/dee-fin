from __future__ import annotations

from pathlib import Path
from typing import Any

FORMAT_STABILITY_CONFIG_NAME = "getm_format_stable.yaml"
FORMAT_STABILITY_PROFILE = "getm_format_stable_dev20_f1"
PROMPT_BASELINE_CONFIG_NAME = "prompt_baselines.yaml"
PROMPT_BASELINE_PROFILES = frozenset(
    {
        "direct_json",
        "schema_only",
        "role_safe",
        "role_safe_surface_memory",
    }
)
DEV20_LIMIT = 20
LIMIT50 = 50


def validate_getm_prediction_scope(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    profile: str,
    split: str,
    limit: int | None,
    allow_limit50: bool = False,
) -> None:
    split_name = str(split).strip().lower()
    if split_name == "test":
        raise ValueError("GETM prediction guard rejects test split; final-eval execution is not implemented")
    if split_name != "dev":
        raise ValueError(f"GETM prediction guard only permits dev split in this stage, got {split!r}")
    if limit is None:
        raise ValueError("GETM prediction guard rejects full dev; pass an explicit --limit")
    if int(limit) <= DEV20_LIMIT:
        return
    if not allow_limit50:
        raise ValueError("GETM prediction guard rejects limit > 20 without --allow-limit50")
    if int(limit) != LIMIT50:
        raise ValueError("GETM prediction guard with --allow-limit50 permits exactly 50 documents")
    if not (
        _is_format_stability_scope(config_path=config_path, config=config, profile=profile)
        or _is_prompt_baseline_scope(config_path=config_path, config=config, profile=profile)
    ):
        raise ValueError("--allow-limit50 requires the format-stability or prompt-baseline config/profile")


def _is_format_stability_scope(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    profile: str,
) -> bool:
    run_cfg = config.get("run") or {}
    return (
        Path(config_path).name == FORMAT_STABILITY_CONFIG_NAME
        and str(profile) == FORMAT_STABILITY_PROFILE
        and str(run_cfg.get("profile") or "") == FORMAT_STABILITY_PROFILE
    )


def _is_prompt_baseline_scope(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    profile: str,
) -> bool:
    run_cfg = config.get("run") or {}
    run_profile = str(run_cfg.get("profile") or "")
    return (
        Path(config_path).name == PROMPT_BASELINE_CONFIG_NAME
        and str(profile) in PROMPT_BASELINE_PROFILES
        and run_profile in PROMPT_BASELINE_PROFILES
    )
