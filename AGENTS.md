# dee-fin Codex Instructions

## Repository Rules

- `./data` is the frozen dataset directory for this project. Do not delete or modify it unless the user explicitly authorizes a data phase.
- `./baseline` contains official GitHub repositories for comparison baselines. Do not delete or modify it.
- Keep CARVE claims evidence-bound: `docs/method/carve_method_design_v1_3.md` is a frozen proposal, not an implementation report.
- `docs/method/easv_v1.md` is deprecated and intentionally removed. Do not restore it.
- Record phase plans and acceptance results under `docs/phase/`.

## Environments

- Local Python: `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`
- Server Python: `/home/TJK/.conda/envs/tjk-feg/bin/python`
- Project server directory: `/data/TJK/DEE/dee-fin`
- `baseline/procnet` server Python: `/home/TJK/.conda/envs/procnet`

## CARVE Model Artifacts

- For CARVE P1 encoder measurement, use the project-local safetensors directories on `gpu-4090`:
  - `/data/TJK/DEE/dee-fin/models/chinese-roberta-wwm-ext_safetensors`
  - `/data/TJK/DEE/dee-fin/models/thunlp_Lawformer_safetensors`
  - `/data/TJK/DEE/dee-fin/models/schen_longformer-chinese-base-4096_safetensors_custom`
- P1 scripts and smoke checks must set offline mode and local-only loading, for example `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `local_files_only=True`, and `use_safetensors=True`.
- Do not point CARVE P1 commands at Hugging Face repo ids or the original `.bin` model directories.

## Execution Boundaries

- Develop locally by default.
- Never run local GPU training or local GPU inference.
- Unless a later phase request explicitly expands scope, CARVE planning work only allows local development plus `gpu-4090` sync and smoke checks when needed.
- Before any remote GPU command, report the exact command, working directory, and expected outputs.
- Do not start long-running remote jobs unless the user explicitly asks.
- Use additive/no-delete server sync. Do not use `rsync --delete`.

## Phase Gate

- Each phase must define acceptance criteria before implementation.
- After each accepted phase, run the relevant local validation commands, commit the phase, and confirm `git status --short` is empty.
- Do not promote templates in `docs/measurements/` to evidence. Write measured results only after the corresponding phase runs.

## Local Validation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/evaluator -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.data_split.test_split_utils -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/baseline/procnet -v
```
