# Setup

Package manager: **uv**. Python: **3.10** (matches the Chapter-1 / server stack).
Default index is the Tsinghua TUNA mirror (set in `pyproject.toml`).

## Local (CPU, no GPU)

Installs only the torch-free `core + dev` deps. Enough to run the contracts,
metrics, graph algorithms, heuristic baselines, the full test suite, and the
end-to-end smoke.

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"

uv run pytest                 # 全套件绿（2 个 torch-gated 测试 skip）
uv run finekg-smoke           # end-to-end on fixtures
uv run ruff check src scripts tests
```

## Server (CUDA 12.4)

Adds the GPU stack, pinned to the Chapter-1 (SARGE) verified versions for a
frictionless transition: `torch==2.6.0+cu124`, transformers/peft/accelerate,
PyTorch Geometric, and (optionally) vLLM.

```bash
uv venv --python 3.10
# torch+cu124 wheels come from the explicit pytorch-cu124 index in pyproject.toml
uv pip install -e ".[dev,llm,gnn]"      # add ,serve for vLLM; add ,rl for GRPO/path-RL training
```

For the RL stage (`trl`, pinned `>=0.17,<0.19` to stay co-tested with
vllm 0.8.5 / torch 2.6.0):

```bash
uv pip install -e ".[dev,llm,gnn,serve,rl]"
# before the first full GRPO run, smoke the TRL<->vLLM coupling for ~50 steps
# and check the adapter actually changes generations (docs/RL_DESIGN.md §6).
```

Current GPU operations are in [`docs/GPU_RUNBOOK.md`](GPU_RUNBOOK.md). The old
W3–4 milestone runbook (TRL↔vLLM smoke, easy-bucket GRPO) is archived at
[`docs/archive/SERVER_W3-4_RUNBOOK.md`](archive/SERVER_W3-4_RUNBOOK.md).

Notes:
- `torch` is sourced from `https://download.pytorch.org/whl/cu124` (declared as an
  `explicit` index, so the local CPU install never fetches it). If that host is
  slow, set a mirror, e.g.
  `UV_INDEX_PYTORCH_CU124_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu124`.
- `bitsandbytes` (4-bit) and `vllm` are GPU-only; keep them out of the local env.
- The neural code imports torch lazily, so a missing GPU only fails at the point
  of instantiation, never at import.

## Reproducibility

Generate a full lockfile on a machine that can reach the torch index:

```bash
uv lock                       # writes uv.lock for all extras
uv sync --extra dev --extra llm --extra gnn   # reproducible install from the lock
```

Match the Chapter-1 runtime flags for offline, deterministic runs
(`HF_HUB_OFFLINE=1`, fixed seeds 13/17/42, report mean±std).
