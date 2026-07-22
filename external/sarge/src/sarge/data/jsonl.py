from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[JsonObject]:
    jsonl_path = Path(path)
    count = 0
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid JSONL: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{jsonl_path}:{line_number}: JSONL row must be a mapping")
            yield payload
            count += 1
            if limit is not None and count >= limit:
                break


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[JsonObject]:
    return list(iter_jsonl(path, limit=limit))


def write_jsonl(path: str | Path, rows: Iterable[JsonObject]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return output_path
