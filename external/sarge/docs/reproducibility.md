# Reproducibility

## Environments

| Item | Local | Server |
|---|---|---|
| Project root | `/home/tjk/myProjects/masterProjects/DEE/SARGE/` | `/data/TJK/DEE/SARGE/` |
| Python | `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python` | `/data/TJK/envs/sarge_vllm_full/bin/python` |
| GPU | not used for training/inference | `gpu-4090`, 4 x 24 GB GPUs |
| Data | copied under `data/` | copied under `data/` |
| Models | copied under `models/` | copied under `models/` |
| Evaluator | copied under `evaluator/` | copied under `evaluator/` |

Local work is for editing, tests, documentation, and CPU-only checks. GPU training and inference run on the server only.

## Datasets

| Dataset | Path | Documents |
|---|---|---|
| ChFinAnn-Doc2EDAG | `data/ChFinAnn-Doc2EDAG/` | train 25,632 / dev 3,204 / test 3,204 |
| DuEE-Fin-dev500 | `data/DuEE-Fin-dev500/` | train 6,515 / dev 500 / test 1,171 |
| DocFEE-dev1000 | `data/DocFEE-dev1000/` | train 17,244 / dev 1,000 / test 800 |

## Models

| Model | Path | Purpose |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `models/Qwen/Qwen3-4B-Instruct-2507` | candidate generation backbone |
| Chinese-RoBERTa-wwm-ext (safetensors) | `models/chinese-roberta-wwm-ext_safetensors` | LRD encoder |
| Lawformer (safetensors) | `models/thunlp_Lawformer_safetensors` | long-document fallback |

Offline model loading:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Use `local_files_only=True` and `use_safetensors=True` when loading local Hugging Face models.

## Random Seeds

- Main seed: `13`
- Extension seeds currently active or queued: `17`, `42`
- Running or queued seed-extension assets remain status-only until eval JSON exists.

## Evaluation Metrics

Main comparison tables use Legacy-FS / `legacy_doc2edag` fixed-slot micro-F1. Unified-Strict, DocFEE, and ExactRec are diagnostic metric families and are reported separately.

ExactRec is computed from Unified-Strict diagnostics:

```text
ExactRec = 2 * record_exact_match_count / validated_record_count
```

## Validation Commands

Local invariant tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m pytest tests/ -v
```

Project-level dependency reference:

- `pyproject.toml` lists runtime dependencies, including DeepSeek/OpenAI-compatible API diagnostics via `openai`, `httpx`, and `python-dotenv`.
- `requirements.lock` is the project-level server-environment lock snapshot generated from `/data/TJK/envs/sarge_vllm_full/bin/python`; it records the authoritative server ML stack and already includes `openai==2.37.0`, `httpx==0.28.1`, and `python-dotenv==1.2.2`.
- `.env` and `.env.*` are ignored; API keys must never be committed.

Server CPU-only three-track evaluation:

```bash
cd /data/TJK/DEE/SARGE
/data/TJK/envs/sarge_vllm_full/bin/python -B scripts/eval_three_tracks.py \
  --run-root runs/<run_name>/<inner_run_name> \
  --dataset <DuEE-Fin-dev500|ChFinAnn-Doc2EDAG> \
  --split <dev|test>
```

## Paper Evidence Requirements

Main table or body performance numbers must come from traceable completed runs. The evidence chain should include:

- `run_manifest.json`
- `eval/eval_legacy_doc2edag.json`
- `eval/eval_unified_strict.json`
- `eval/eval_docfee_official.json`
- diagnostics such as `pipeline_summary.json` and `selection_summary.json` when available
- an entry in `paper/exp/data/asset_registry.json`

`MockGetmBackend`, smoke runs, running jobs, and invalid-contract diagnostics must not enter the paper main result table.

## Paper Assets

Experiment tables are generated from `paper/exp/data/asset_registry.json` and checked-in JSON snapshots:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B paper/exp/scripts/build_seed13_summary.py
```

The ACL-family draft lives in `paper/emnlp_aacl_draft/`:

```bash
cd /home/tjk/myProjects/masterProjects/DEE/SARGE/paper/emnlp_aacl_draft
./build.sh
```

`paper/emnlp_aacl_draft/source_manifest.json` records hashes for the registry, baseline constants, and generated summary inputs used by the draft asset builder.

## Artifact Pull Policy

Only pull small JSON evidence into Git:

```bash
rsync -av \
  --include='*/' \
  --include='*.json' \
  --exclude='*' \
  gpu-4090:/data/TJK/DEE/SARGE/runs/<run_name>/ \
  paper/exp/data/run_snapshots/<asset_id>/
```

Do not pull checkpoints, full prediction JSONL files, raw outputs, or parsed candidates into the repository.

## API Diagnostics

DeepSeek API diagnostics are CPU/API-only runs and are kept separate from GPU/main-table evidence:

- summary JSON: `paper/exp/data/api_diagnostics/deepseek_api_diagnostics_20260522.json`
- report: `docs/deepseek_api_diagnostics_20260522.md`
- runner: `src/sarge/experiments/deepseek_api_eval.py`

These diagnostics record model names, run roots, aggregate metrics, token counts, and whether an API key was recorded. They do not store `.env` contents or raw API outputs in Git.

## History

Historical code and old assets are recovered through Git history. Current paper/reporting entry points are `paper/exp/`, `paper/emnlp_aacl_draft/`, `docs/exp_result.md`, `docs/gpu_todo.md`, and `docs/handoff.md`.
