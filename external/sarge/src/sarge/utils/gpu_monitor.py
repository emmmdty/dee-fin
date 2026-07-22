from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class VramLimitExceeded(RuntimeError):
    """Raised when real-run telemetry exceeds the configured VRAM guard."""


@dataclass(frozen=True)
class ResourceMonitorConfig:
    enabled: bool = False
    sample_interval_sec: float = 1.0
    gpu_ids: tuple[int, ...] | None = None
    vram_soft_limit_gb: float = 23.0
    vram_target_min_gb: float = 20.0
    vram_target_max_gb: float = 23.0
    fail_on_vram_limit: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> ResourceMonitorConfig:
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            sample_interval_sec=float(raw.get("sample_interval_sec", 1.0)),
            gpu_ids=_parse_gpu_ids(raw.get("gpu_ids")),
            vram_soft_limit_gb=float(raw.get("vram_soft_limit_gb", 23.0)),
            vram_target_min_gb=float(raw.get("vram_target_min_gb", 20.0)),
            vram_target_max_gb=float(raw.get("vram_target_max_gb", 23.0)),
            fail_on_vram_limit=bool(raw.get("fail_on_vram_limit", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu_ids"] = list(self.gpu_ids) if self.gpu_ids is not None else None
        return payload


@dataclass(frozen=True)
class GpuMemorySample:
    timestamp: str
    gpu_index: int
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu_percent: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpu_index": self.gpu_index,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "utilization_gpu_percent": self.utilization_gpu_percent,
        }


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


class GpuMemoryMonitor:
    def __init__(
        self,
        *,
        out_dir: str | Path,
        config: ResourceMonitorConfig | None = None,
        run_command: RunCommand = subprocess.run,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.config = config or ResourceMonitorConfig()
        self._run_command = run_command
        self._samples: list[GpuMemorySample] = []
        self._warnings: list[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._ended_at: str | None = None
        self._last_summary: dict[str, Any] | None = None

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def samples_path(self) -> Path:
        return self.out_dir / "gpu_memory_samples.csv"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "gpu_memory_summary.json"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="sarge-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.config.sample_interval_sec + 1.0))
        self._thread = None

    def sample_once(self) -> list[GpuMemorySample]:
        timestamp = _utc_now()
        command = _nvidia_smi_command(self._effective_gpu_ids())
        try:
            completed = self._run_command(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            self._add_warning(f"nvidia-smi unavailable: {exc}")
            return []
        except Exception as exc:  # pragma: no cover - defensive telemetry path
            self._add_warning(f"nvidia-smi sample failed: {type(exc).__name__}: {exc}")
            return []

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            self._add_warning(f"nvidia-smi exited with code {completed.returncode}: {stderr}")
            return []

        samples = parse_nvidia_smi_csv(completed.stdout, timestamp=timestamp)
        with self._lock:
            self._samples.extend(samples)
        return samples

    def finish(self) -> dict[str, Any]:
        self.stop()
        self._ended_at = self._ended_at or _utc_now()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        samples = self._snapshot_samples()
        _write_samples_csv(self.samples_path, samples)
        summary = self._build_summary(samples)
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._last_summary = summary
        return summary

    def enforce_vram_limit(self, *, real_run: bool) -> None:
        summary = self._last_summary or self.finish()
        if real_run and self.config.fail_on_vram_limit and summary.get("exceeded_soft_limit"):
            raise VramLimitExceeded(
                "GPU memory peak exceeded soft limit: "
                f"peak={summary.get('max_peak_memory_used_gb')}GB, "
                f"limit={self.config.vram_soft_limit_gb}GB"
            )

    def _effective_gpu_ids(self) -> tuple[int, ...] | None:
        return self.config.gpu_ids or _parse_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(max(0.1, float(self.config.sample_interval_sec)))

    def _snapshot_samples(self) -> list[GpuMemorySample]:
        with self._lock:
            return list(self._samples)

    def _add_warning(self, message: str) -> None:
        with self._lock:
            if message not in self._warnings:
                self._warnings.append(message)

    def _build_summary(self, samples: list[GpuMemorySample]) -> dict[str, Any]:
        peak_mb: dict[str, int] = {}
        for sample in samples:
            key = str(sample.gpu_index)
            peak_mb[key] = max(peak_mb.get(key, 0), sample.memory_used_mb)
        peak_gb = {key: round(value / 1024.0, 4) for key, value in peak_mb.items()}
        max_peak_gb = max(peak_gb.values()) if peak_gb else None
        exceeded = bool(max_peak_gb is not None and max_peak_gb > self.config.vram_soft_limit_gb)
        within_band = None
        if max_peak_gb is not None:
            within_band = self.config.vram_target_min_gb <= max_peak_gb <= self.config.vram_target_max_gb
        return {
            "peak_memory_used_mb_by_gpu": dict(sorted(peak_mb.items())),
            "peak_memory_used_gb_by_gpu": dict(sorted(peak_gb.items())),
            "max_peak_memory_used_gb": max_peak_gb,
            "sample_count": len(samples),
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "duration_sec": round(max(0.0, time.monotonic() - self._started_monotonic), 6),
            "vram_soft_limit_gb": self.config.vram_soft_limit_gb,
            "vram_target_min_gb": self.config.vram_target_min_gb,
            "vram_target_max_gb": self.config.vram_target_max_gb,
            "exceeded_soft_limit": exceeded,
            "within_target_band": within_band,
            "warnings": list(self._warnings),
        }


def parse_nvidia_smi_csv(output: str, *, timestamp: str) -> list[GpuMemorySample]:
    samples: list[GpuMemorySample] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        utilization = _parse_optional_int(parts[3]) if len(parts) > 3 else None
        samples.append(
            GpuMemorySample(
                timestamp=timestamp,
                gpu_index=int(_strip_units(parts[0])),
                memory_used_mb=int(_strip_units(parts[1])),
                memory_total_mb=int(_strip_units(parts[2])),
                utilization_gpu_percent=utilization,
            )
        )
    return samples


def discover_gpu_ids(run_command: RunCommand = subprocess.run) -> tuple[int, ...]:
    env_ids = _parse_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if env_ids is not None:
        return env_ids
    try:
        completed = run_command(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ()
    if completed.returncode != 0:
        return ()
    ids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            ids.append(int(line))
    return tuple(ids)


def _nvidia_smi_command(gpu_ids: Sequence[int] | None) -> list[str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    if gpu_ids:
        command.insert(1, "--id=" + ",".join(str(gpu_id) for gpu_id in gpu_ids))
    return command


def _write_samples_csv(path: Path, samples: list[GpuMemorySample]) -> None:
    fieldnames = ["timestamp", "gpu_index", "memory_used_mb", "memory_total_mb", "utilization_gpu_percent"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())


def _parse_gpu_ids(value: Any) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.split(",") if part.strip()]
        if not raw_parts:
            return None
        if not all(part.isdigit() for part in raw_parts):
            return None
        return tuple(int(part) for part in raw_parts)
    if isinstance(value, Sequence):
        return tuple(int(part) for part in value)
    return None


def _parse_optional_int(value: str) -> int | None:
    stripped = _strip_units(value)
    if stripped in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    return int(stripped)


def _strip_units(value: str) -> str:
    return value.strip().replace("MiB", "").replace("%", "").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
