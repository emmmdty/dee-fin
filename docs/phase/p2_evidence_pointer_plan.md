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

Status: **accepted 2026-05-14** based on `runs/carve/p2_duee_fin_seed42_acceptance`.

### Non-smoke acceptance evidence (2026-05-14)

Run directory: `runs/carve/p2_duee_fin_seed42_acceptance`. Command: `scripts.carve.run_p2_evidence_pointer --max-epochs 10 --seed 42` against `data/processed/DuEE-Fin-dev500` with `models/chinese-roberta-wwm-ext_safetensors`. 1892 train documents, 146 dev documents (`multi_event_subset` filter, matching the smoke and r3 runs). Elapsed 1184.7s (19.7 min, 10 epochs, encoder fine-tuned with `--encoder-lr 2e-5`, `--grad-accum 8`).

Train loss trajectory:

| Epoch | loss | evidence_bce | pointer_mi |
|---:|---:|---:|---:|
| 1 | 2.3379 | 1.0097 | 2.6563 |
| 2 | 0.8968 | 0.3553 | 1.0830 |
| 3 | 0.6381 | 0.2835 | 0.7093 |
| 4 | 0.5156 | 0.2441 | 0.5430 |
| 5 | 0.4419 | 0.2090 | 0.4658 |
| 6 | 0.3857 | 0.1682 | 0.4351 |
| 7 | 0.3497 | 0.1466 | 0.4062 |
| 8 | 0.3224 | 0.1308 | 0.3831 |
| 9 | 0.3015 | 0.1222 | 0.3586 |
| 10 | **0.2823** | **0.1132** | **0.3381** |

Dev evidence metrics (`diagnostics/evidence_metrics.json`):

| Metric | Value |
|---|---:|
| `aligned_args` | 1591 / 1613 |
| `unalignable_args` | 22 |
| `unalignable_rate` | 0.013639 |
| `evidence_bce` | 0.114657 |
| `pointer_mi` | 0.591746 |

Acceptance verdict against frozen thresholds:

| Criterion | Threshold | Final | Verdict |
|---|---|---|---|
| train `evidence_bce` | ≤ 0.30 | 0.1132 | **PASS** (cross at epoch 3) |
| train `pointer_mi` | ≤ 1.5 | 0.3381 | **PASS** (cross at epoch 2) |
| dev `unalignable_rate` | ≤ 0.30 | 0.013639 | **PASS** |

All three frozen acceptance criteria are satisfied with comfortable margins. The phase gate dependency from `docs/phase/p3_mention_planner_plan.md` ("all P2 acceptance checks remain passing") is closed.

### Historical smoke (2026-05-13, for reference only)

Earlier 2-epoch DuEE-Fin smoke at `runs/carve/p2_duee_fin_smoke`. Not used for acceptance; retained as the implementation-completion artifact:

| Epoch | Loss | Evidence BCE | Pointer MI |
|---:|---:|---:|---:|
| 1 | 1.644822 | 0.556254 | 2.177135 |
| 2 | 1.313912 | 0.382694 | 1.862436 |

Dev diagnostics from the smoke (16 docs only): `evidence_bce=0.422835`, `pointer_mi=1.658796`, `unalignable_rate=0.005952`, `aligned_args=167`, `unalignable_args=1`.

## Non-Goals

- No unified-strict dev scoring.
- No hidden-test or final-test claim.
- No paper main-table claim.
- No P5b decision table update.
