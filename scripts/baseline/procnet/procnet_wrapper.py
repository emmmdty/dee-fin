from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASETS = ("ChFinAnn-Doc2EDAG", "DuEE-Fin-dev500")


@dataclass(frozen=True)
class RunConfig:
    project_root: Path
    dataset: str
    experiment_name: str
    seed: int
    max_epochs: int
    patience: int
    gpu: str
    python_bin: str
    output_root: Path
    run_dir: Path
    server_mode: bool
    early_stopping: bool = True


def build_run_config(
    *,
    project_root: Path,
    dataset: str,
    experiment_name: str,
    seed: int,
    max_epochs: int,
    patience: int,
    gpu: str,
    python_bin: str,
    output_root: Path,
    server_mode: bool,
) -> RunConfig:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    run_dir = output_root / experiment_name
    return RunConfig(
        project_root=project_root,
        dataset=dataset,
        experiment_name=experiment_name,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        gpu=gpu,
        python_bin=python_bin,
        output_root=output_root,
        run_dir=run_dir,
        server_mode=server_mode,
    )


def ensure_run_layout(run_dir: Path) -> None:
    for relative in (
        "staged_data",
        "baseline_result",
        "native_event_tables",
        "canonical",
        "eval",
        "procnet_runtime/Data",
        "procnet_runtime/Result",
        "procnet_runtime/Checkpoint",
        "procnet_runtime/Cache/Transformers",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)


def expected_artifacts_for_split(run_dir: Path, split: str) -> dict[str, str]:
    del run_dir
    native_relative = f"native_event_tables/{split}.procnet_native_event_table_v1.json"
    return {
        "native_gold_path": f"{native_relative}#documents[].gold",
        "native_pred_path": f"{native_relative}#documents[].pred",
        "canonical_gold_path": f"canonical/{split}.canonical.gold.jsonl",
        "canonical_pred_path": f"canonical/{split}.canonical.pred.jsonl",
    }


def set_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    metadata: dict[str, Any] = {
        "seed": seed,
        "random_seed": seed,
        "numpy_seed": None,
        "torch_manual_seed": None,
        "torch_cuda_seed": None,
        "torch_deterministic_algorithms": False,
        "torch_cudnn_deterministic": None,
        "torch_cudnn_benchmark": None,
    }
    try:
        import numpy as np

        np.random.seed(seed)
        metadata["numpy_seed"] = seed
    except Exception as exc:  # pragma: no cover - environment-specific metadata
        metadata["numpy_seed_error"] = str(exc)

    try:
        import torch

        torch.manual_seed(seed)
        metadata["torch_manual_seed"] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            metadata["torch_cuda_seed"] = seed
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
            metadata["torch_deterministic_algorithms"] = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            metadata["torch_cudnn_deterministic"] = True
            metadata["torch_cudnn_benchmark"] = False
    except Exception as exc:  # pragma: no cover - environment-specific metadata
        metadata["torch_seed_error"] = str(exc)
    return metadata


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_run_metadata(config: RunConfig, *, command: list[str], seed_metadata: dict[str, Any]) -> None:
    ensure_run_layout(config.run_dir)
    (config.run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    write_json(
        config.run_dir / "config.json",
        {
            **_config_to_json(config),
            "early_stopping": {
                "enabled": config.early_stopping,
                "patience": config.patience,
                "max_epochs": config.max_epochs,
            },
        },
    )
    write_json(
        config.run_dir / "environment.json",
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "python_bin": config.python_bin,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "server_mode": config.server_mode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "seed_control": seed_metadata,
        },
    )


def _config_to_json(config: RunConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def format_command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)
