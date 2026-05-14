# P3 Mention CRF and Record Planner Plan

## Purpose

P3 replaces regex-dominated candidate generation and trigger-count record planning with a mention CRF and learned record-count planner. It implements CARVE v1.3 Section 3.4 and Section 3.6 interfaces needed before revisiting P5b.

## Components

P3 covers:

- character-offset tokenization helpers in `carve.tokenization`;
- BIO label construction over teacher-forced gold evidence sentences;
- a dependency-free PyTorch linear-chain CRF in `carve.p3_mention_crf`;
- per-document, per-event-type type-gate and record-count planning in `carve.p3_planner`;
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
- ~~Planner output is a softmax over `n = 0..K`; `n_t = 0` replaces P5b's string-based type gate.~~
- 2026-05-14 R3 errata: the single softmax decision is withdrawn because it conflicts with CARVE v1.3 Section 3.6. P3 uses a two-stage planner: a binary type gate predicts `present(t, D)`, and a zero-truncated Poisson count head predicts `n_t` only for event types passing the gate.
- The type gate is trained with `BCEWithLogitsLoss(pos_weight=#negative/#positive)` measured from the actual train documents used by the run.
- The count planner is trained only on gold-positive `(document, event_type)` pairs with zero-truncated Poisson NLL.
- `K_clip` is the max train `n_t` measured from the actual train documents used by the run; it is recorded in diagnostics and checkpoint metadata.
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
- type gate dev AUC is at least 0.85 and Youden's J calibrated F1 is at least 0.55;
- count planner dev MAE on the `n_t > 0` subset is at most 0.5;
- `presence_loss` and `count_loss` each show at least five consecutive epochs of downward trend and at least 2x decrease;
- after integration into the current pipeline, DuEE-Fin-dev500 `unified_strict_f1` is not lower than the current R3 baseline by more than 0.005 absolute.

## Validation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.carve.test_p3_mention_crf tests.carve.test_p3_planner tests.carve.test_p3_candidates_integration -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/carve -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/evaluator -v
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
    --run-dir runs/carve/p3_duee_fin_smoke_r4 \
    --smoke --max-epochs 2
```

Expected outputs:

- `runs/carve/p3_duee_fin_smoke_r4/diagnostics/p3_train_history.json`
- `runs/carve/p3_duee_fin_smoke_r4/diagnostics/p3_mention_metrics.json`
- `runs/carve/p3_duee_fin_smoke_r4/diagnostics/p3_planner_metrics.json`
- `runs/carve/p3_duee_fin_smoke_r4/checkpoints/p3.pt`
- `runs/carve/p3_duee_fin_smoke_r4/summary.json`

The R3 implementation writes `presence_loss`, `count_loss`, `presence_pos_frac`, `presence_pos_weight`, and `k_clip` into `p3_train_history.json`. It writes `type_gate_auc`, `type_gate_f1_youden`, `presence_threshold`, `count_mae_positive`, `k_clip`, and `truncation_rate` into `p3_planner_metrics.json`.

## Acceptance Result

Status: implemented locally; server smoke completed; acceptance pending.

R3 note: the 2026-05-13 smoke evidence below is historical evidence for the withdrawn single-softmax planner. It remains useful as failure evidence, but it is not R3 two-stage planner acceptance evidence. R3 requires a fresh authorized smoke/non-smoke run before any acceptance claim.

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

---

## 2026-05-14 R3 v2 Errata (Acceptance Gate Rewrite)

The R3 acceptance criteria above were single-population and decoupled from baselines. The current external audit (`docs/diagnostics/current_stage_external_audit_20260514.md`) showed that an R3 run could simultaneously fail the absolute thresholds yet not be compared to the strongest baselines. Two changes apply to any R3 acceptance run starting from 2026-05-14:

1. **Dual-population**: the type gate AUC, type gate Youden F1, and count_mae_positive thresholds apply independently to both `multi_event_dev` and `all_dev`. All eight data checks plus the two loss-trend checks must pass for `accepted=true`.
2. **Baseline-relative margins** (in addition to absolute thresholds):
   - `type_gate_auc` must exceed `max(p5b_lexical_trigger, legacy_single_softmax)` by at least 0.05.
   - `type_gate_f1_youden` must exceed `max(p5b_lexical_trigger, legacy_single_softmax)` by at least 0.05.
   - `count_mae_positive` must be at least 0.05 below `min(predict_one, p5b_lexical_trigger, legacy_single_softmax)`.
   - The absolute thresholds from the original Acceptance Criteria section remain as floors. The gate combines absolute-and-baseline-relative with AND.

This errata is enforced by `carve/p3_planner_only_runner.py::_acceptance_checks` and the corresponding tests in `tests/carve/test_r3_planner_only_runner.py`. The full P3 path (`carve/p3_runner.py`) still emits v1 metrics, but its results do not constitute R3 acceptance evidence.

The original P2-acceptance prerequisite still applies. No P3 acceptance claim follows from R3 v2 alone.
