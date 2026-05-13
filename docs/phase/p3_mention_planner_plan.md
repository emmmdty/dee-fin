# P3 Mention CRF and Record Planner Plan

## Purpose

P3 replaces regex-dominated candidate generation and trigger-count record planning with a mention CRF and learned record-count planner. It implements CARVE v1.3 Section 3.4 interfaces needed before revisiting P5b.

## Components

P3 covers:

- character-offset tokenization helpers in `carve.tokenization`;
- BIO label construction over teacher-forced gold evidence sentences;
- a dependency-free PyTorch linear-chain CRF in `carve.p3_mention_crf`;
- per-document, per-event-type record-count planning in `carve.p3_planner`;
- a new `carve.candidates.generate_candidates` interface;
- lexicon/regex candidate generation preserved as optional fallback;
- runner entry point `scripts/carve/run_p3_mention_planner.py`;
- checkpoint and diagnostics files under the selected run directory.

## Frozen Interface Decisions

- During training, CRF sentences are selected by gold `y_ev[i, t] = 1`.
- During inference, evidence-head threshold and top-K sentence selection are the intended route; the runner keeps this configurable for later ablation.
- BIO labels mark all occurrences of each normalized gold role value inside a selected sentence.
- Overlapping labels use schema role order as the priority order.
- CRF decoding is per selected sentence, not document-global.
- Record planner input is `[global_repr; event_type_emb]`.
- Planner output is a softmax over `n = 0..K`; `n_t = 0` replaces P5b's string-based type gate.
- Default `K` is 10 until a dataset-specific train max plus margin is measured.
- Training uses gold `n_t`; inference uses predicted `n_t`.
- CRF token spans are converted to character spans before returning `CandidateMention`.
- Candidate values are normalized at the candidate-generation boundary; raw spans are retained for diagnostics.
- Oracle injection remains training-only and is disabled by default for inference.

## Acceptance Criteria

P3 is accepted only if:

- P3 unit tests pass;
- full `tests/carve` remains passing;
- all P2 acceptance checks remain passing;
- server smoke loads the project-local safetensors model with offline/local-only settings;
- a non-smoke run records `diagnostics/p3_mention_metrics.json`, `diagnostics/p3_planner_metrics.json`, and `checkpoints/p3.pt`;
- dev mention F1 is at least 0.40 under normalized gold-value matching;
- planner per-event-type MAE is at most 1.0;
- type-gate recall for `n_t > 0` is at least 0.90.

## Validation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.carve.test_p3_mention_crf tests.carve.test_p3_planner tests.carve.test_p3_candidates_integration -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/carve -v
```

Server smoke command must run from:

```bash
/data/TJK/DEE/dee-fin
```

with:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/TJK/.conda/envs/tjk-feg/bin/python -m scripts.carve.run_p3_mention_planner \
    --dataset DuEE-Fin-dev500 \
    --data-root data/processed/DuEE-Fin-dev500 \
    --schema data/processed/DuEE-Fin-dev500/schema.json \
    --model-path models/chinese-roberta-wwm-ext_safetensors \
    --run-dir runs/carve/p3_duee_fin_smoke \
    --smoke --max-epochs 2
```

Expected outputs:

- `runs/carve/p3_duee_fin_smoke/diagnostics/p3_train_history.json`
- `runs/carve/p3_duee_fin_smoke/diagnostics/p3_mention_metrics.json`
- `runs/carve/p3_duee_fin_smoke/diagnostics/p3_planner_metrics.json`
- `runs/carve/p3_duee_fin_smoke/checkpoints/p3.pt`
- `runs/carve/p3_duee_fin_smoke/summary.json`

## Acceptance Result

Status: implemented locally; server smoke completed; acceptance pending.

Local implementation, toy smoke tests, and the DuEE-Fin remote smoke were added on 2026-05-13. This is not yet accepted because P2 acceptance and non-smoke threshold runs have not been executed in this closeout.

Server smoke evidence:

- run directory: `runs/carve/p3_duee_fin_smoke`
- offline/local-only model path: `models/chinese-roberta-wwm-ext_safetensors`
- checkpoint: `runs/carve/p3_duee_fin_smoke/checkpoints/p3.pt`
- diagnostics: `runs/carve/p3_duee_fin_smoke/diagnostics/p3_train_history.json`
- diagnostics: `runs/carve/p3_duee_fin_smoke/diagnostics/p3_mention_metrics.json`
- diagnostics: `runs/carve/p3_duee_fin_smoke/diagnostics/p3_planner_metrics.json`

Observed 2-epoch smoke trend:

| Epoch | Loss | P2 Subloss | Mention Loss | Planner Loss |
|---:|---:|---:|---:|---:|
| 1 | 117.212563 | 1.967009 | 114.286585 | 0.958968 |
| 2 | 45.468727 | 1.276090 | 43.795659 | 0.396978 |

Observed smoke dev diagnostics:

| Metric | Value |
|---|---:|
| `mention_f1` | 0.000000 |
| `mention_precision` | 0.000000 |
| `mention_recall` | 0.000000 |
| `gold_mentions` | 325 |
| `predicted_mentions` | 0 |
| `planner_mae` | 0.197115 |
| `type_gate_recall` | 0.000000 |

## Non-Goals

- No unified-strict dev scoring.
- No hidden-test or final-test claim.
- No paper main-table claim.
- No P5b decision table update.
