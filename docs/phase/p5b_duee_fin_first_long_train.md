# P5b DuEE-Fin First Long Training Plan

## Purpose

This phase starts the first long-running CARVE diagnostic training task on `DuEE-Fin-dev500`.

This is an explicit scope expansion from the minimal phase route. It does not claim that the full CARVE encoder, EDAG decoder, or verifier is complete. It trains a lightweight allocation/share diagnostic head and compares a no-allocation baseline against an allocation-aware route on the dev split.

## Scope

- Dataset: `DuEE-Fin-dev500`.
- Split used for diagnostics: dev only.
- Training source: train split only.
- Inference candidate construction: train-derived lexicon plus role regexes over document text; dev gold is not used for candidate construction.
- Evaluator: `unified-strict` dev report.
- Output root: `runs/carve/p5b_duee_fin_dev500_seed42`.

## Acceptance Criteria

The phase is accepted only if:

- local unit tests for CARVE allocation and P5b runner pass;
- existing evaluator, data split, and ProcNet wrapper validation commands still pass;
- P1 encoder memory measurement is run on `gpu-4090` with local safetensors and offline mode before the long job;
- a remote smoke run completes before the long job;
- the long job is launched detached, records PID/log path/run directory, and is observed until process, GPU memory, and logs are stable;
- outputs remain clearly labeled as dev diagnostics only and are not written as final test results.

## Non-Goals

- No hidden-test or final-test claim.
- No SOTA claim.
- No Qwen verifier work.
- No modification of `data/` or `baseline/`.
- No promotion of `docs/measurements/p5b_decision_table_template.md` to a final table until all required P5b rows are complete.

## Remote Commands

Working directory:

```bash
/data/TJK/DEE/dee-fin
```

P1 measurement:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/home/TJK/.conda/envs/tjk-feg/bin/python scripts/carve/measure_encoder_memory.py \
  --models chinese-roberta-wwm-ext,lawformer,longformer-chinese \
  --out runs/carve/p1_memory/result.json \
  --markdown-out docs/measurements/p1_memory.md
```

Smoke:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
/home/TJK/.conda/envs/tjk-feg/bin/python scripts/carve/run_p5b_diagnostic.py \
  --dataset DuEE-Fin-dev500 \
  --data-root data/processed/DuEE-Fin-dev500 \
  --schema data/processed/DuEE-Fin-dev500/schema.json \
  --run-dir runs/carve/p5b_duee_fin_dev500_seed42_smoke \
  --seed 42 --max-epochs 1 --patience 1 --batch-size 4 --grad-accum 8 \
  --routes baseline,carve --smoke --limit-docs 8
```

Detached long run:

```bash
mkdir -p runs/carve/p5b_duee_fin_dev500_seed42/logs && \
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
/home/TJK/.conda/envs/tjk-feg/bin/python scripts/carve/run_p5b_diagnostic.py \
  --dataset DuEE-Fin-dev500 \
  --data-root data/processed/DuEE-Fin-dev500 \
  --schema data/processed/DuEE-Fin-dev500/schema.json \
  --model-path /data/TJK/DEE/dee-fin/models/chinese-roberta-wwm-ext_safetensors \
  --run-dir runs/carve/p5b_duee_fin_dev500_seed42 \
  --seed 42 --max-epochs 30 --patience 5 --batch-size 4 --grad-accum 8 \
  --routes baseline,carve \
  > runs/carve/p5b_duee_fin_dev500_seed42/logs/train.log 2>&1 < /dev/null & echo $!
```
