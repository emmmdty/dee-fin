# P3 R3 Planner Replan

Date: 2026-05-14

## Methodology Finding

The original P3 planner compressed CARVE v1.3 Section 3.6 into a single softmax over `n = 0..K`, with `n_t = 0` acting as the type gate. This conflicts with the frozen proposal route:

```text
for each event type t passing type gate:
    n_t = planner(t, D)
```

The R3 change restores the two-stage design: binary event-type presence first, then count planning only for active event types.

## Evidence

- In `data/processed/DuEE-Fin-dev500/train.jsonl`, the raw all-document `(document, event_type)` distribution is structurally zero-heavy: about 92.3% have `n_t = 0`, max train `n_t = 16`, and most documents have one active event type among 13.
- The actual P3 runner trains on `multi_event_subset`. Under that runner filter, read-only measurement gives train zero fraction about 88.9%, `pos_weight ~= 7.99`, and `k_clip = 16`.
- The previous smoke evidence recorded `type_gate_recall = 0.0`, which is consistent with collapse when a single majority-zero softmax controls both presence and count.

## Decision

- Add `TypeGate`: input `[global_repr; event_type_emb]`, scalar logit, trained with BCE-with-logits and train-only `pos_weight = #negative / #positive`.
- Add `CountPlanner`: input `[global_repr; event_type_emb]`, scalar `log_lambda`, trained only on gold-positive `(document, event_type)` pairs.
- Use zero-truncated Poisson NLL:

```text
L(n, lambda) = lambda - n * log(lambda) + log(n!) + log(1 - exp(-lambda))
```

The final term is implemented with a stable `log(-expm1(-lambda))` / `log1p` branch. `softplus(lambda) - lambda` is not used because it computes `log(1 + exp(-lambda))`, not `log(1 - exp(-lambda))`.

## Acceptance Criteria

R3 is accepted only if all of the following hold on the relevant DuEE-Fin-dev500 validation path:

- Type gate: dev `AUC >= 0.85` and Youden's J calibrated `F1 >= 0.55`.
- Count planner: dev `MAE <= 0.5` on the `n_t > 0` subset.
- Training trajectory: `presence_loss` and `count_loss` each show at least five consecutive epochs of downward trend and at least 2x decrease.
- Pipeline integration: DuEE-Fin-dev500 `unified_strict_f1` is not lower than the current R3 baseline by more than 0.005 absolute.

## Scope Boundaries

- No focal loss, class-balanced loss, or resampling in the main path.
- No P2, P4, P5a, or P5b behavior changes.
- No `data/`, `baseline/`, evaluator, hidden-test, final-test, or paper main-table updates.
- No remote GPU long run is started by this replan. A gpu-4090 smoke is allowed only after explicit authorization with command, working directory, and expected outputs stated first.

---

## R3 v2 (2026-05-14): Evidence-Conditioned Two-Stage Planner

### Why v2

External audit (`docs/diagnostics/current_stage_external_audit_20260514.md`) recorded the v1 non-smoke acceptance failure on `runs/carve/r3_planner_only_duee_fin_seed42`. The relevant numbers were:

- multi_event_dev `type_gate_auc = 0.710765`, `type_gate_f1_youden = 0.301989`, `count_mae_positive = 0.807692`.
- presence_loss decrease ratio 1.07; count_loss decrease ratio 1.24 across 30 epochs.
- multi_event_dev predict-1 count baseline `0.836538`; lexical trigger baseline `type_gate_f1 = 0.662539`. Lexical baseline outperformed the neural type gate by more than 30 absolute F1 points.
- all_dev `type_gate_auc = 0.589839`, `type_gate_f1_youden = 0.181432`; the gap between multi_event_dev and all_dev confirmed that v1 learned a multi-event-subset bias rather than a full-document presence signal.

The root cause was a method gap, not a coding bug: TypeGate only consumed `[global_repr; type_emb]`, which discarded the strongest local cue — whether the event-type string appears in sentences — that the lexical baseline already exploited. The acceptance gate was also decoupled from baselines, so the threshold `count_mae_positive <= 0.5` could be passed while predict-1 already sat at 0.348 on all_dev.

### v2 design

1. **Evidence pooling**: `TypeGate` and `CountPlanner` accept optional `sentence_repr [B, n_sent_max, H]` and `sentence_mask [B, n_sent_max]`. The forward path uses `type_emb` as a query against `sentence_repr` (scaled dot-product attention, additive `-1e9` masking on padded positions). The pooled `evidence_vec` is concatenated to `[global_repr; type_emb]`.
2. **Lexical feature**: a 1-d binary `lexical_hit` is precomputed per (doc, event_type) using `carve.p5b_runner._type_gate`. The model is free to weight, ignore, or invert this feature. This is a model input, not a rule override. The same lexical detector is used as the lexical_trigger baseline so the comparison stays apples-to-apples.
3. **Feature flag**: `--encoder-feature-mode {global_only, evidence, evidence_lexical}` selects which features the runner passes. `global_only` reproduces v1 behavior with the same architecture. `evidence_lexical` is the default. Three modes share one runner so ablation rows come out of the same code path.
4. **Training population**: default switches to `all_train`. `--train-population multi_event_train` is preserved as an explicit ablation. Evaluation still reports both `multi_event_dev` and `all_dev`.
5. **Baseline-relative acceptance gate**: each presence/count metric must pass both an absolute threshold and a baseline-relative margin. For AUC and Youden's F1 the model must exceed `max(lexical_trigger, legacy_single_softmax)` by at least 0.05. For count_mae_positive the model must beat `min(predict_one, lexical_trigger, legacy_single_softmax)` by at least 0.05. Loss-trend checks (presence_loss, count_loss) remain.
6. **Dual-population acceptance**: the same three data checks are applied independently on `multi_event_dev` and `all_dev`. `accepted=true` requires all eight data checks plus two trend checks to pass.

### Cache extension

`PlannerFeatureCache` now stores `sentence_repr [N, S_max, H]` padded, `sentence_mask [N, S_max]`, `lexical_hit [N, T]`, alongside the existing `global_repr [N, H]` and `counts [N, T]`. Sentence length per document is capped by `--max-sentences` (default 256); the truncation rate is recorded in `_population_stats` for transparency.

### Acceptance check format

`summary.json` now contains flat keys under `acceptance_checks`:

- `multi_event_dev/type_gate_auc`
- `multi_event_dev/type_gate_f1_youden`
- `multi_event_dev/count_mae_positive`
- `all_dev/type_gate_auc`
- `all_dev/type_gate_f1_youden`
- `all_dev/count_mae_positive`
- `training/presence_loss_trend`
- `training/count_loss_trend`

Each data entry exposes `value`, `absolute_threshold`, `best_baseline`, `best_baseline_sources`, `baseline_relative_threshold`, `passed_absolute`, `passed_baseline_relative`, and the combined `passed`.

### Risks recorded for paper review

- "Aren't you just a lexical baseline?" — `--encoder-feature-mode evidence` ablation row is the answer; if it does not beat lexical, the model is not over-relying on the feature.
- "Why train on all_train instead of multi_event_subset?" — multi_event_subset training led to all_dev AUC 0.59. all_train is the population whose acceptance the paper actually claims.
- "Why both populations as acceptance?" — Single-population thresholds let multi_event_dev pass while all_dev fails or vice versa. Both populations are reported and both must pass.
- "Global representation may still be insufficient" — v2 keeps `global_only` as a control. If `evidence` and `evidence_lexical` do not improve over `global_only`, the encoder representation itself is the bottleneck and further work needs to move upstream (richer encoder, more parameters, or sentence-level supervision rather than document-level).

### v2 acceptance precondition

This replan does not promote any R3 acceptance claim. R3 v2 acceptance still requires a non-smoke long run on `gpu-4090` with full DuEE-Fin-dev500 train docs. That run is not authorized in this round. A short server smoke is allowed once the local unit tests pass.

### v2 does not modify

- `carve/p5b_runner.py` — P5b integration of trained planner is left to `docs/phase/p5b_planner_integration_plan.md`.
- `carve/p3_runner.py` — the full P3 path retains v1 semantics; v2 only applies to the planner-only runner.
- The P2 acceptance prerequisite from `docs/phase/p3_mention_planner_plan.md`. R3 v2 acceptance still requires P2 acceptance closure as a separate phase gate.

### v2 smoke evidence (2026-05-14)

Local unit tests: all `tests/carve` (39) and `tests/evaluator` (39) tests pass, including 4 new `tests/carve/test_p3_planner.py` cases (evidence attention, lexical feature, padding mask, backward-compatible feature dims) and 3 new `tests/carve/test_r3_planner_only_runner.py` cases (dual-population checks present, feature-mode flag wired through, train-population switch).

Server smoke on `gpu-4090` with `evidence_lexical` feature mode, `all_train` training population, 2 epochs, batch 32, 16 docs (smoke mode), `runs/carve/r3_planner_only_duee_fin_seed42_v2_smoke/`. Exit 0, elapsed 4.421s. Artifacts present: `summary.json`, `diagnostics/r3_planner_{train_history,metrics,baselines}.json`, `checkpoints/r3_planner.pt`, `cache/{all_train,multi_event_dev,all_dev}.pt`.

Pipeline health checks (smoke only, not acceptance):

| Field | Value |
|---|---|
| `status` | `r3_planner_only_smoke` |
| `acceptance_population` | `["multi_event_dev", "all_dev"]` |
| `encoder_feature_mode` | `evidence_lexical` |
| `train_population.name` | `all_train` |
| `train_population.documents` | 16 |
| `train_population.zero_rate` | 0.913462 |
| `train_population.presence_pos_weight` | 10.555556 |
| `train_population.k_clip` | 3 |
| epoch 1 presence_loss | 1.3194 |
| epoch 2 presence_loss | 1.1224 (-14.9%) |
| epoch 1 count_loss | 0.9156 |
| epoch 2 count_loss | 0.7575 (-17.3%) |
| `acceptance_checks` keys | 8 data checks (multi_event_dev + all_dev × {auc, f1_youden, mae+}) + 2 trend checks |
| `accepted` | false (expected: 2 epochs cannot satisfy 5-consecutive-epoch trend check; 16-doc smoke is not the acceptance sample) |

What the smoke does and does not show:

- The smoke confirms that the v2 cache, runner, dual-population evaluation, baseline-relative gate, and checkpoint format are all wired correctly. It does not constitute an acceptance run.
- On 16 documents the neural type_gate F1 was 0.261 on multi_event_dev and 0.186 on all_dev, while lexical_trigger baselines were 0.647 and 0.381 respectively. This is the same shape as the v1 non-smoke failure recorded in the audit. With only 16 train documents and 2 epochs the model cannot be expected to exceed lexical; the smoke is only a pipeline check.
- Whether v2 actually beats lexical on full DuEE-Fin-dev500 (~1900 train documents, multiple epochs) requires a non-smoke run, which is not authorized in this phase.

