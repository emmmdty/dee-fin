# P2 Evidence and Pointer Heads Plan

## Purpose

P2 replaces P5b hand-written allocation features with encoder-backed evidence and backward grounding heads. It is an internal diagnostic phase for CARVE v1.3 Section 3.2-3.3, not a final dev/test result.

## Components

P2 covers:

- deterministic sentence segmentation in `carve.text_segmentation.split_sentences`;
- weak evidence labels from gold role values;
- sentence-by-event-type labels defined as OR over role labels;
- sentence-by-event-type-by-role labels with schema role masking;
- unalignable gold role values recorded and masked from backward MI loss;
- role-value-conditioned sentence attention in `PointerHead`;
- local smoke and server smoke entry point `scripts/carve/run_p2_evidence_pointer.py`;
- checkpoint and diagnostics files under the selected run directory.

## Frozen Interface Decisions

- Cross-sentence role-value matching is not supported. A role value must occur inside one segmented sentence.
- Sentence splitting is newline-first, then punctuation-based with closing quotes attached.
- `y_ev[i, t]` is `OR_rho y_ev[i, t, rho]`.
- Gold values that cannot be aligned after strict normalization are recorded as `unalignable_args` and excluded from `L_ground^MI`.
- Repeated values may align to multiple sentences. Pointer MI uses marginal log-sum-exp over all positive sentences.
- Backward grounding is keyed by `(event_type, role, normalized_value)`, not by gold record id.
- First-match sentence is diagnostic-only; P2 does not implement Hard-EM as the main route.

## Acceptance Criteria

P2 is accepted only if:

- P2 unit tests pass;
- full `tests/carve` remains passing;
- server smoke loads the project-local safetensors model with offline/local-only settings;
- a non-smoke run records `diagnostics/p2_train_history.json`, `diagnostics/evidence_metrics.json`, and `checkpoints/p2.pt`;
- train `evidence_bce <= 0.30`;
- train `pointer_mi <= 1.5`;
- dev `unalignable_rate <= 0.30`.

## Validation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.carve.test_p2_heads -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/carve -v
```

Server smoke command must run from:

```bash
/data/TJK/DEE/dee-fin
```

with:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/TJK/.conda/envs/tjk-feg/bin/python -m scripts.carve.run_p2_evidence_pointer \
    --dataset DuEE-Fin-dev500 \
    --data-root data/processed/DuEE-Fin-dev500 \
    --schema data/processed/DuEE-Fin-dev500/schema.json \
    --model-path models/chinese-roberta-wwm-ext_safetensors \
    --run-dir runs/carve/p2_duee_fin_smoke \
    --smoke --max-epochs 2
```

Expected outputs:

- `runs/carve/p2_duee_fin_smoke/diagnostics/p2_train_history.json`
- `runs/carve/p2_duee_fin_smoke/diagnostics/evidence_metrics.json`
- `runs/carve/p2_duee_fin_smoke/checkpoints/p2.pt`
- `runs/carve/p2_duee_fin_smoke/summary.json`

## Acceptance Result

Status: implemented locally; server smoke completed; acceptance pending.

Local implementation, toy smoke tests, and the DuEE-Fin remote smoke were added on 2026-05-13. This is not yet accepted because the non-smoke threshold run has not been executed in this closeout.

Server smoke evidence:

- run directory: `runs/carve/p2_duee_fin_smoke`
- offline/local-only model path: `models/chinese-roberta-wwm-ext_safetensors`
- checkpoint: `runs/carve/p2_duee_fin_smoke/checkpoints/p2.pt`
- diagnostics: `runs/carve/p2_duee_fin_smoke/diagnostics/p2_train_history.json`
- diagnostics: `runs/carve/p2_duee_fin_smoke/diagnostics/evidence_metrics.json`

Observed 2-epoch smoke trend:

| Epoch | Loss | Evidence BCE | Pointer MI |
|---:|---:|---:|---:|
| 1 | 1.644822 | 0.556254 | 2.177135 |
| 2 | 1.313912 | 0.382694 | 1.862436 |

Observed smoke dev diagnostics:

| Metric | Value |
|---|---:|
| `evidence_bce` | 0.422835 |
| `pointer_mi` | 1.658796 |
| `unalignable_rate` | 0.005952 |
| `aligned_args` | 167 |
| `unalignable_args` | 1 |

## Non-Goals

- No unified-strict dev scoring.
- No hidden-test or final-test claim.
- No paper main-table claim.
- No P5b decision table update.
