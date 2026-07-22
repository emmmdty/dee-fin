# W3.5 Evidence Hardening

> Date: 2026-05-19
> Local root: `/home/tjk/myProjects/masterProjects/DEE/SARGE/`
> Server root: `/data/TJK/DEE/SARGE/`
> Local Python: `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`
> Server Python: `/data/TJK/envs/sarge_vllm_full/bin/python`

## Purpose

Before using any number in a paper table, the run must be traceable to the
source code, model artifact, decoding configuration, and exact launch command.
The server repo is now git-initialized, but individual run roots still need an
explicit source commit when they are launched from copied artifacts or a
detached working tree.

## Evidence Rules

Use a run for model-performance evidence only if `run_manifest.json` records:

- `model_performance_evidence: true`
- non-mock `backend`
- `git_commit`
- `command_infer`
- `model_path`
- `adapter_path` or `merged_model_path`
- `generation.k_candidates`, `generation.do_sample`, `generation.seed`, and `generation.max_new_tokens`
- `limit` and `document_count`

Runs with `backend: "MockGetmBackend"` or `model_performance_evidence: false`
are pipeline smoke artifacts only. They must not be used in paper main tables.

When running from the server copy, pass the local source commit explicitly:

```bash
SOURCE_COMMIT=<committed_local_git_hash>
```

Use the commit after this hardening patch is committed and synced. Do not use an
uncommitted working tree as paper evidence.

## ChFinAnn Full-Dev Command Template

Working directory:

```bash
/data/TJK/DEE/SARGE
```

Command:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> PYTHONPATH=src \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
/data/TJK/envs/sarge_vllm_full/bin/python -u scripts/infer_checkpoint.py \
  --ckpt runs/sarge_sft_ChFinAnn_Doc2EDAG_s13_ep2_gpu1/artifacts/model/adapter \
  --dataset ChFinAnn-Doc2EDAG \
  --split dev \
  --k 1 \
  --slot-train-limit 50 \
  --source-commit "$SOURCE_COMMIT"
```

Expected outputs:

- new `runs/sarge_infer_ChFinAnn-Doc2EDAG_dev_<timestamp>/`
- `predictions/ChFinAnn-Doc2EDAG/dev.canonical.pred.jsonl` with 3,204 rows
- `run_manifest.json` containing the fields listed above
- `diagnostics/pipeline_summary.json` without `mock_backend_notice`

After inference, run the CPU evaluator:

```bash
/data/TJK/envs/sarge_vllm_full/bin/python -B scripts/eval_three_tracks.py \
  --run-root runs/<new_run_name> \
  --dataset ChFinAnn-Doc2EDAG \
  --split dev
```

Expected outputs:

- `eval/eval_legacy_doc2edag.json`
- `eval/eval_unified_strict.json`
- `eval/eval_docfee_official.json`
- `eval/summary.json` is not emitted by the current evaluator; the three per-track JSON files are the authoritative outputs.

## Launch Boundary

Do not start this GPU run until a free GPU is confirmed on `gpu-4090`.
Because the server is shared, check `nvidia-smi` and process ownership before
launching. Never kill processes owned by users other than `TJK`.
