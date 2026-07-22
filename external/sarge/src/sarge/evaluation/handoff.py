from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_REPO_ROOT = REPO_ROOT / "data"
DEFAULT_PROFILES = ("unified_main", "record_level", "aux_basic", "paper_tables", "leaderboard")


@dataclass(frozen=True)
class EvaluatorHandoff:
    command: str
    argv: tuple[str, ...]
    data_repo_root: Path
    script_path: Path
    out_dir: Path
    script_exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "argv": list(self.argv),
            "data_repo_root": str(self.data_repo_root),
            "script_path": str(self.script_path),
            "out_dir": str(self.out_dir),
            "script_exists": self.script_exists,
        }


def build_evaluator_handoff(
    *,
    run_root: str | Path,
    dataset: str,
    split: str,
    data_repo_root: str | Path = DEFAULT_DATA_REPO_ROOT,
    out_dir: str | Path | None = None,
    profiles: tuple[str, ...] = DEFAULT_PROFILES,
    benchmark_root: str | Path = "processed",
    strict: bool = True,
) -> EvaluatorHandoff:
    data_root = Path(data_repo_root)
    artifact_out_dir = Path(out_dir) if out_dir is not None else Path(run_root) / "eval"
    project_root = REPO_ROOT
    script_path = project_root / "scripts" / "eval_three_tracks.py"
    python_bin = sys.executable
    argv = [
        python_bin,
        "-B",
        str(script_path),
        "--run-root",
        str(Path(run_root)),
        "--processed-root",
        str(data_root),
        "--project-root",
        str(project_root),
        "--python",
        python_bin,
        "--dataset",
        dataset,
        "--split",
        split,
    ]
    del profiles, benchmark_root, strict
    command = f"cd {quote(str(project_root))} && " + " ".join(
        ["UV_CACHE_DIR=/tmp/uv-cache", *[quote(arg) for arg in argv]]
    )
    return EvaluatorHandoff(
        command=command,
        argv=tuple(argv),
        data_repo_root=data_root,
        script_path=script_path,
        out_dir=artifact_out_dir,
        script_exists=script_path.is_file(),
    )


def run_evaluator_handoff(handoff: EvaluatorHandoff) -> dict[str, Any]:
    if not handoff.script_exists:
        return {
            "attempted": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"missing evaluator script: {handoff.script_path}",
        }
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = "/tmp/uv-cache"
    completed = subprocess.run(
        list(handoff.argv),
        cwd=handoff.script_path.parent.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "attempted": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
