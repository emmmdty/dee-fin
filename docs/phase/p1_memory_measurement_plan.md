# P1 Memory Measurement Plan

## Purpose

P1 measures whether the encoder and planned CARVE modules fit the available RTX 4090 budget before making GPU feasibility or paper-resource claims.

## Scope

P1 is a measurement phase, not a training phase. It may use `gpu-4090` only after the exact command is shown to the user.

Measure forward-pass memory for:

- ChFinAnn / DuEE-Fin baseline encoder route: `hfl/chinese-roberta-wwm-ext` with 512-token sliding windows.
- DocFEE long-document candidates: `thunlp/Lawformer` and `schen/longformer-chinese-base-4096`.
- CARVE head estimates or implemented lightweight heads only if those modules already exist in a later phase.

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
