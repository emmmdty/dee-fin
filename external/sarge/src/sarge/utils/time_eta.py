from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimingEvent:
    timestamp: str
    item_id: str | None
    item_type: str
    completed_items: int
    total_items: int | None
    duration_sec: float
    avg_sec_per_item: float | None
    moving_avg_sec_per_item: float | None
    eta_sec: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "item_id": self.item_id,
            "item_type": self.item_type,
            "completed_items": self.completed_items,
            "total_items": self.total_items,
            "duration_sec": self.duration_sec,
            "avg_sec_per_item": self.avg_sec_per_item,
            "moving_avg_sec_per_item": self.moving_avg_sec_per_item,
            "eta_sec": self.eta_sec,
        }


class TimingTracker:
    def __init__(
        self,
        *,
        total_items: int | None = None,
        moving_window_size: int = 20,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total_items = total_items
        self.moving_window_size = max(1, int(moving_window_size))
        self._clock = clock
        self._started_at = _utc_now()
        self._start_time = self._clock()
        self._last_mark = self._start_time
        self._ended_at: str | None = None
        self._durations: list[float] = []
        self._events: list[TimingEvent] = []

    def set_total_items(self, total_items: int | None) -> None:
        self.total_items = total_items

    def record_item(
        self,
        *,
        item_id: str | None = None,
        item_type: str = "item",
        duration_sec: float | None = None,
    ) -> TimingEvent:
        now = self._clock()
        duration = float(duration_sec) if duration_sec is not None else max(0.0, now - self._last_mark)
        self._last_mark = now
        self._durations.append(duration)
        event = TimingEvent(
            timestamp=_utc_now(),
            item_id=item_id,
            item_type=item_type,
            completed_items=len(self._durations),
            total_items=self.total_items,
            duration_sec=round(duration, 6),
            avg_sec_per_item=_round_optional(self._avg_duration()),
            moving_avg_sec_per_item=_round_optional(self._moving_avg_duration()),
            eta_sec=_round_optional(self._eta_sec()),
        )
        self._events.append(event)
        return event

    def finish(self, out_dir: str | Path) -> dict[str, Any]:
        self._ended_at = self._ended_at or _utc_now()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        events_path = out_dir / "timing_events.jsonl"
        with events_path.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        summary = self.summary()
        (out_dir / "timing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def summary(self) -> dict[str, Any]:
        elapsed = max(0.0, self._clock() - self._start_time)
        avg = self._avg_duration()
        moving_avg = self._moving_avg_duration()
        estimated_total = None
        if self.total_items is not None and moving_avg is not None:
            estimated_total = moving_avg * self.total_items
        return {
            "total_items": self.total_items,
            "completed_items": len(self._durations),
            "avg_sec_per_item": _round_optional(avg),
            "moving_avg_sec_per_item": _round_optional(moving_avg),
            "elapsed_sec": round(elapsed, 6),
            "estimated_total_sec": _round_optional(estimated_total),
            "eta_sec": _round_optional(self._eta_sec()),
            "started_at": self._started_at,
            "ended_at": self._ended_at,
        }

    def _avg_duration(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)

    def _moving_avg_duration(self) -> float | None:
        if not self._durations:
            return None
        window = self._durations[-self.moving_window_size :]
        return sum(window) / len(window)

    def _eta_sec(self) -> float | None:
        if self.total_items is None or not self._durations:
            return None
        remaining = max(0, self.total_items - len(self._durations))
        moving_avg = self._moving_avg_duration()
        if moving_avg is None:
            return None
        return moving_avg * remaining


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
