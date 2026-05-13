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

## Acceptance Result

Status: completed / accepted as DuEE-Fin dev diagnostic evidence only.

Acceptance date: 2026-05-13.

This R5b run reran the DuEE-Fin first long diagnostic on `gpu-4090` from local commit `d03e3c5ba1b48a546c2ff21c20eef50cdd7eb6a4`. Source files used by the R5b runner, P1 measurement script, and phase docs were hash-checked between local and `/data/TJK/DEE/dee-fin` before running.

Remote evidence:

- P1 prerequisite rerun: `runs/carve/p1_memory/result.json` and `docs/measurements/p1_memory.md`.
- Smoke run: `runs/carve/p5b_duee_fin_dev500_seed42_smoke`.
- Detached long run: `runs/carve/p5b_duee_fin_dev500_seed42`.
- Long-run log: `runs/carve/p5b_duee_fin_dev500_seed42/logs/train.log`.
- Wrapper PID: `2486605`.
- Python PID: `2486607`.
- Initial stable observation: Python process running, GPU memory about 486 MB, log reached epoch 1 without errors.
- Completion: no R5b runner process remained after epoch 9; `summary.json` and `diagnostics/p5b_duee_fin_decision_row.json` were written.

Observed long-run data:

| Item | Value |
|---|---:|
| Train multi-event documents | 1,892 |
| Dev multi-event documents | 146 |
| Training groups | 19,322 |
| Elapsed seconds | 1,861.950 |
| Final logged epoch | 9 |
| Final logged loss | 0.965967 |

Unified-strict dev diagnostics:

| Route | Candidate count | Predicted records | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 148,651 | 1,892 | 202 | 8,599 | 1,825 | 0.022952 | 0.099655 | 0.037311 |
| carve | 148,651 | 3,082 | 225 | 39,098 | 1,802 | 0.005722 | 0.111001 | 0.010883 |

Runner decision row:

```json
{
  "dataset": "DuEE-Fin-dev500",
  "split": "dev",
  "status": "dev_diagnostic_only_not_final_test",
  "support_label": "No support"
}
```

Measured evidence file:

- `docs/measurements/p5b_duee_fin_dev500_seed42.md`

Interpretation:

- This completes only the authorized R5b DuEE-Fin first long run.
- It does not complete the full P5b three-dataset gate.
- It does not write or promote `docs/measurements/p5b_decision_table.md`.
- It is not a hidden-test result, final-test result, SOTA claim, Qwen verifier claim, or full CARVE implementation report.

Closeout validation passed:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/carve -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/evaluator -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.data_split.test_split_utils -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/baseline/procnet -v
git diff --check
```
