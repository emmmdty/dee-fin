from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationProfile:
    name: str
    prompt_baseline_mode: str
    description: str


ABLATION_PROFILE_ROWS: tuple[AblationProfile, ...] = (
    AblationProfile(
        name="full",
        prompt_baseline_mode="role_safe_surface_memory",
        description="schema, role-safe instructions, surface candidates, and event-slot plan",
    ),
    AblationProfile(
        name="no_surface_memory",
        prompt_baseline_mode="role_safe_slot_plan_only",
        description="remove rendered surface candidates while keeping schema and event-slot plan",
    ),
    AblationProfile(
        name="no_slot_plan",
        prompt_baseline_mode="role_safe_surface_only",
        description="remove rendered event-slot plan while keeping schema and surface candidates",
    ),
    AblationProfile(
        name="no_surface_or_slot",
        prompt_baseline_mode="role_safe",
        description="keep role-safe schema instructions without surface candidates or event-slot plan",
    ),
    AblationProfile(
        name="schema_only",
        prompt_baseline_mode="schema_only",
        description="render schema text but remove role-safe closure, surface candidates, and slot plan",
    ),
    AblationProfile(
        name="direct_json",
        prompt_baseline_mode="direct_json",
        description="direct JSON generation from document text without schema or auxiliary grounding sections",
    ),
)

ABLATION_PROFILES: dict[str, AblationProfile] = {profile.name: profile for profile in ABLATION_PROFILE_ROWS}


def resolve_ablation_profile(profile: str | AblationProfile) -> AblationProfile:
    if isinstance(profile, AblationProfile):
        return profile
    name = str(profile).strip()
    try:
        return ABLATION_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown SARGE ablation profile: {profile!r}") from exc


def apply_ablation_profile(config: dict[str, Any], profile: str | AblationProfile) -> dict[str, Any]:
    resolved = resolve_ablation_profile(profile)
    updated = copy.deepcopy(config)
    getm = updated.setdefault("getm", {})
    if not isinstance(getm, dict):
        raise ValueError("config['getm'] must be a mapping")
    prompt = getm.setdefault("prompt", {})
    if not isinstance(prompt, dict):
        raise ValueError("config['getm']['prompt'] must be a mapping")
    prompt["baseline_mode"] = resolved.prompt_baseline_mode
    prompt["ablation_profile"] = resolved.name
    return updated


__all__ = [
    "ABLATION_PROFILES",
    "ABLATION_PROFILE_ROWS",
    "AblationProfile",
    "apply_ablation_profile",
    "resolve_ablation_profile",
]
