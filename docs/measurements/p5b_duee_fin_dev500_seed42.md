# P5b DuEE-Fin Dev Diagnostic Evidence

> Status: measured dev diagnostic evidence for the DuEE-Fin first long run only.  
> Scope: not hidden-test, not final-test, not SOTA, not Qwen verifier, and not the full three-dataset P5b decision table.

## Run Identity

| Item | Value |
|---|---|
| Date | 2026-05-13 |
| Dataset | `DuEE-Fin-dev500` |
| Split | dev diagnostics only |
| Training source | train split only |
| Local commit | `d03e3c5ba1b48a546c2ff21c20eef50cdd7eb6a4` |
| Server root | `/data/TJK/DEE/dee-fin` |
| Server Python | `/home/TJK/.conda/envs/tjk-feg/bin/python` |
| Long-run directory | `runs/carve/p5b_duee_fin_dev500_seed42` |
| Smoke-run directory | `runs/carve/p5b_duee_fin_dev500_seed42_smoke` |
| Long-run log | `runs/carve/p5b_duee_fin_dev500_seed42/logs/train.log` |
| Wrapper PID | `2486605` |
| Python PID | `2486607` |

## P1 Prerequisite

P1 encoder memory was rerun on `gpu-4090` before the R5b long diagnostic with:

```bash
DEE_FIN_GIT_COMMIT=d03e3c5ba1b48a546c2ff21c20eef50cdd7eb6a4 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/home/TJK/.conda/envs/tjk-feg/bin/python scripts/carve/measure_encoder_memory.py \
  --models chinese-roberta-wwm-ext,lawformer,longformer-chinese \
  --out runs/carve/p1_memory/result.json \
  --markdown-out docs/measurements/p1_memory.md
```

All three local safetensors model paths returned `status=ok`; refreshed evidence is recorded in `docs/measurements/p1_memory.md`.

## Smoke Result

The bounded smoke run completed before the long job.

| Route | Candidate count | Predicted records | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 4,961 | 103 | 13 | 391 | 132 | 0.032178 | 0.089655 | 0.047359 |
| carve | 4,961 | 110 | 3 | 1,997 | 142 | 0.001500 | 0.020690 | 0.002797 |

Smoke decision row:

```json
{
  "dataset": "DuEE-Fin-dev500",
  "split": "dev",
  "status": "dev_diagnostic_only_not_final_test",
  "support_label": "No support"
}
```

## Long-Run Result

The detached long diagnostic completed after early stopping at epoch 9.

| Item | Value |
|---|---:|
| Train multi-event documents | 1,892 |
| Dev multi-event documents | 146 |
| Training groups | 19,322 |
| Elapsed seconds | 1,861.950 |
| Final logged epoch | 9 |
| Final logged loss | 0.965967 |

Route metrics from `summary.json`:

| Route | Candidate count | Predicted records | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 148,651 | 1,892 | 202 | 8,599 | 1,825 | 0.022952 | 0.099655 | 0.037311 |
| carve | 148,651 | 3,082 | 225 | 39,098 | 1,802 | 0.005722 | 0.111001 | 0.010883 |

Observed F1 delta, `carve - baseline`: `-0.026428`.

Decision row:

```json
{
  "baseline_unified_strict_f1": 0.03731067602512006,
  "carve_unified_strict_f1": 0.010882708585247884,
  "dataset": "DuEE-Fin-dev500",
  "decision_rule": "first DuEE-Fin diagnostic row only; full P5b requires all datasets and final table",
  "split": "dev",
  "status": "dev_diagnostic_only_not_final_test",
  "support_label": "No support"
}
```

## Interpretation Boundary

This run completes the R5b DuEE-Fin first long diagnostic. It does not complete the full P5b mechanism gate because ChFinAnn and DocFEE rows were not run, and the current runner does not emit the full P5b misallocation/candidate-recall metric family required by `docs/measurements/p5b_decision_table_template.md`.

Therefore:

- Do not create or promote `docs/measurements/p5b_decision_table.md` from this single-dataset result.
- Keep the full P5b gate as planned/not run.
- Treat the DuEE-Fin result as dev diagnostic evidence only.
