"""Dataset schema loader for SARGE.

Loads ``schema.json`` files describing event types and roles, exposing a
frozen :class:`DatasetSchema` for downstream validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSchema:
    dataset_id: str
    schema_dataset: str
    schema_path: Path
    canonical_version: str | None
    event_roles: dict[str, tuple[str, ...]]
    role_to_event_types: dict[str, tuple[str, ...]]

    @property
    def event_types(self) -> frozenset[str]:
        return frozenset(self.event_roles)

    @property
    def unique_roles(self) -> frozenset[str]:
        return frozenset(role for roles in self.event_roles.values() for role in roles)

    def validate_event_type(self, event_type: str) -> str:
        normalized = str(event_type).strip()
        if normalized not in self.event_roles:
            raise ValueError(f"Unknown event_type for dataset {self.dataset_id}: {normalized!r}")
        return normalized

    def validate_role(self, event_type: str, role: str) -> str:
        normalized_event_type = self.validate_event_type(event_type)
        normalized_role = str(role).strip()
        if normalized_role not in self.event_roles[normalized_event_type]:
            raise ValueError(
                f"Unknown role for dataset {self.dataset_id}/{normalized_event_type}: {normalized_role!r}"
            )
        return normalized_role

    def validate_event_record(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise ValueError("event record must be a mapping")
        event_type = self.validate_event_type(str(event.get("event_type", "")))
        arguments = event.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError(f"event arguments must be a mapping: {event_type}")
        for role, values in arguments.items():
            self.validate_role(event_type, str(role))
            if not isinstance(values, list):
                raise ValueError(f"event argument values must be a list: {event_type}/{role}")
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError(f"event argument value must be a mapping: {event_type}/{role}")


def load_schema(dataset: str, data_root: str | Path = "data") -> DatasetSchema:
    dataset_id = str(dataset).strip()
    if not dataset_id:
        raise ValueError("dataset is required")
    schema_path = Path(data_root) / dataset_id / "schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        # Normalise array-of-event-objects into the dict shape expected by
        # ``_event_roles``.  DuEE-Fin uses ``event_type/role_list`` where
        # each role entry is a dict like ``{"role": "..."}``; ChFinAnn uses
        # ``name/arguments`` with bare strings; both are tolerated here.
        normalised: list[dict] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            et = str(entry.get("event_type") or entry.get("name") or "").strip()
            roles = entry.get("role_list") or entry.get("arguments") or entry.get("roles") or []
            if not et or not isinstance(roles, list):
                continue
            extracted: list[str] = []
            for raw in roles:
                if isinstance(raw, dict):
                    name = str(raw.get("role") or raw.get("name") or "").strip()
                else:
                    name = str(raw).strip()
                if name:
                    extracted.append(name)
            normalised.append({"event_type": et, "roles": extracted})
        payload = {"event_types": normalised}
    if not isinstance(payload, dict):
        raise ValueError(f"{schema_path} must load as a mapping")

    event_roles = _event_roles(payload, schema_path=schema_path)
    return DatasetSchema(
        dataset_id=dataset_id,
        schema_dataset=str(payload.get("dataset") or dataset_id),
        schema_path=schema_path,
        canonical_version=_optional_str(payload.get("canonical_version")),
        event_roles=event_roles,
        role_to_event_types=_role_to_event_types(event_roles),
    )


def _event_roles(payload: dict[str, Any], *, schema_path: Path) -> dict[str, tuple[str, ...]]:
    raw_event_types = payload.get("event_types")
    if not isinstance(raw_event_types, list):
        raise ValueError(f"{schema_path}: event_types must be a list")

    event_roles: dict[str, tuple[str, ...]] = {}
    for index, raw_event in enumerate(raw_event_types, 1):
        if not isinstance(raw_event, dict):
            raise ValueError(f"{schema_path}: event_types[{index}] must be a mapping")
        event_type = str(raw_event.get("event_type", "")).strip()
        if not event_type:
            raise ValueError(f"{schema_path}: event_types[{index}] missing event_type")
        if event_type in event_roles:
            raise ValueError(f"{schema_path}: duplicate event_type {event_type!r}")
        raw_roles = raw_event.get("roles")
        if raw_roles is None:
            raw_roles = raw_event.get("role_list") or raw_event.get("arguments") or []
        if not isinstance(raw_roles, list):
            raise ValueError(f"{schema_path}: event_types[{event_type!r}].roles must be a list")
        roles: list[str] = []
        for role_index, raw_role in enumerate(raw_roles, 1):
            if isinstance(raw_role, dict):
                role = str(raw_role.get("role") or raw_role.get("name") or "").strip()
            else:
                role = str(raw_role).strip()
            if not role:
                raise ValueError(f"{schema_path}: event_types[{event_type!r}].roles[{role_index}] is empty")
            if role in roles:
                raise ValueError(f"{schema_path}: duplicate role {event_type}/{role}")
            roles.append(role)
        event_roles[event_type] = tuple(roles)
    return event_roles


def _role_to_event_types(event_roles: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    role_map: dict[str, list[str]] = {}
    for event_type, roles in event_roles.items():
        for role in roles:
            role_map.setdefault(role, []).append(event_type)
    return {role: tuple(event_types) for role, event_types in sorted(role_map.items())}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
