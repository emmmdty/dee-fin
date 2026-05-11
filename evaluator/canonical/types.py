from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CanonicalEventRecord:
    document_id: str
    event_type: str
    arguments: dict[str, list[str]] = field(default_factory=dict)
    record_id: str | None = None


@dataclass
class CanonicalDocument:
    document_id: str
    records: list[CanonicalEventRecord] = field(default_factory=list)
