from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.canonical.normalize import normalize_text


@dataclass
class EventSchema:
    event_roles: dict[str, tuple[str, ...]]

    @classmethod
    def from_mapping(cls, mapping: dict[str, list[str] | tuple[str, ...]]) -> "EventSchema":
        normalized: dict[str, tuple[str, ...]] = {}
        for event_type, roles in mapping.items():
            normalized[normalize_text(str(event_type))] = tuple(normalize_text(str(role)) for role in roles)
        return cls(normalized)

    @classmethod
    def from_file(cls, path: str | Path) -> "EventSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_obj(data)

    @classmethod
    def from_json_obj(cls, data: Any) -> "EventSchema":
        if isinstance(data, list):
            mapping: dict[str, list[str]] = {}
            for item in data:
                if not isinstance(item, dict) or "event_type" not in item:
                    continue
                event_type = str(item["event_type"])
                if isinstance(item.get("arguments"), list):
                    roles = [str(role) for role in item["arguments"]]
                elif isinstance(item.get("role_list"), list):
                    roles = [str(role["role"]) for role in item["role_list"] if isinstance(role, dict) and "role" in role]
                else:
                    roles = []
                mapping[event_type] = roles
            return cls.from_mapping(mapping)

        if isinstance(data, dict) and isinstance(data.get("properties"), dict):
            mapping = {}
            for event_type, spec in data["properties"].items():
                roles = []
                if isinstance(spec, dict) and isinstance(spec.get("properties"), dict):
                    roles = list(spec["properties"].keys())
                mapping[event_type] = roles
            return cls.from_mapping(mapping)

        if isinstance(data, dict):
            mapping = {
                str(event_type): [str(role) for role in roles]
                for event_type, roles in data.items()
                if isinstance(roles, list)
            }
            return cls.from_mapping(mapping)

        return cls({})

    def has_event_type(self, event_type: str) -> bool:
        return normalize_text(event_type) in self.event_roles

    def has_role(self, event_type: str, role: str) -> bool:
        roles = self.event_roles.get(normalize_text(event_type))
        return roles is not None and normalize_text(role) in roles

    def roles_for(self, event_type: str) -> tuple[str, ...]:
        return self.event_roles.get(normalize_text(event_type), ())

    def to_report(self) -> dict[str, list[str]]:
        return {event_type: list(roles) for event_type, roles in self.event_roles.items()}
