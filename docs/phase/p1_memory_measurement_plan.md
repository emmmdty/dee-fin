# P1 Memory Measurement Plan

## Purpose

P1 measures whether the encoder and planned CARVE modules fit the available RTX 4090 budget before making GPU feasibility or paper-resource claims.

## Scope

P1 is a measurement phase, not a training phase. It may use `gpu-4090` only after the exact command is shown to the user.

Measure forward-pass memory for:

- ChFinAnn / DuEE-Fin baseline encoder route: `hfl/chinese-roberta-wwm-ext` via local safetensors path `/data/TJK/DEE/dee-fin/models/chinese-roberta-wwm-ext_safetensors`, with 512-token sliding windows.
- DocFEE long-document candidates:
  - `thunlp/Lawformer` via local safetensors path `/data/TJK/DEE/dee-fin/models/thunlp_Lawformer_safetensors`;
  - `schen/longformer-chinese-base-4096` via local safetensors path `/data/TJK/DEE/dee-fin/models/schen_longformer-chinese-base-4096_safetensors_custom`.
- CARVE head estimates or implemented lightweight heads only if those modules already exist in a later phase.

P1 commands must use the project-local safetensors paths above with offline/local-only loading: set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, and pass `local_files_only=True` and `use_safetensors=True` in model-loading code. Do not point P1 commands at Hugging Face repo ids or the original `.bin` model directories.

## Acceptance Criteria

P1 is accepted only if:

- The measured command, git commit, host, GPU, Python path, and model configuration are recorded.
- Peak memory and batch/window settings are written to `docs/measurements/p1_memory.md`.
- The report distinguishes measured numbers from estimates in CARVE v1.3.
- The fallback order in CARVE v1.3 is applied only after measurement:
  1. gradient checkpointing, smaller batch, gradient accumulation;
  2. freeze encoder and train CARVE heads on cached sentence representations;
  3. 512-window Chinese-RoBERTa with role-summary memory;
  4. measured lower-memory choice between Lawformer and Longformer-Chinese.

## Non-Goals

- No full training.
- No test-set evaluation.
- No Qwen verifier work.
- No claim that a template file is measured evidence.

## Closeout

Status: completed / measured.

P1 measured encoder forward-pass memory on `gpu-4090` using project-local safetensors paths and offline/local-only loading. It did not run training, inference evaluation, final-test scoring, or Qwen verifier work.

Measured evidence:

- Report: `docs/measurements/p1_memory.md`.
- Host: `ubuntu`.
- Server Python: `/home/TJK/.conda/envs/tjk-feg/bin/python`.
- Measured code commit recorded by the command: `3b5338790c3844e2e52a90776f32d99983e02dc8`.
- Evidence-file commit: `1251e8416d8249db5d2808326d2710ce83611c9a`.

Measured peak reserved memory at sequence length 512, batch size 1:

| Model | Status | Peak allocated GB | Peak reserved GB |
|---|---:|---:|---:|
| chinese-roberta-wwm-ext | ok | 0.407 | 0.455 |
| lawformer | ok | 0.531 | 0.945 |
| longformer-chinese | ok | 0.419 | 0.965 |

Fallback policy remains unchanged: apply CARVE v1.3 fallback decisions only after measured constraints are considered, and do not promote P1 memory measurements into training or performance claims.
