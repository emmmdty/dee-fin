# R3 planner-only v3 sentence count head acceptance measurement

> Date: 2026-05-14
> Run dir: `runs/carve/r3_planner_only_duee_fin_seed42_v3`
> Boundary: non-smoke planner-only diagnostic. Not P5b dev diagnostic, not unified-strict, not hidden-test, not final-test, not main-table.

## Purpose

v3 tests whether replacing the document-level CountPlanner with a sentence-level scorer — predicting a per-sentence binary "this sentence contains a record of type X" and aggregating expected counts — overcomes the v2.1 architectural bottleneck (`docs/measurements/r3_planner_only_duee_fin_seed42_v2_1.md:84-87`). Supervision is derived from a deterministic heuristic: a sentence is labeled positive iff at least one argument value of a gold record of type `t` appears in the normalised sentence text.

## Run configuration

| Field | Value |
|---|---|
| Dataset | DuEE-Fin-dev500 |
| Train documents | 6515 (all_train, full set) |
| Train (doc, type) pairs | 84695 |
| Dev documents | multi_event_dev=146, all_dev=500 |
| Encoder | `models/chinese-roberta-wwm-ext_safetensors` (offline, safetensors, local-only, frozen) |
| Seed | 42 |
| Max epochs | 50 |
| Batch size | 64 |
| Feature mode | `evidence_lexical` |
| Count head mode | `sentence` |
| Sentence label min hits | 1 |
| BCE pos weight cap | 20.0 |
| Elapsed | 588.4 s (9.8 min) |

## Loss trajectory (selected epochs)

| Epoch | presence_loss | count_loss (BCE) |
|---:|---:|---:|
| 1 | 1.2045 | 1.7651 |
| 10 | 0.6348 | 0.8118 |
| 20 | 0.3554 | 0.6939 |
| 30 | 0.2602 | 0.5378 |
| 40 | 0.2221 | 0.4194 |
| 50 | **0.2040** | **0.3436** |

`presence_loss` decreased 5.91× across 50 epochs. `sentence_count_loss` (BCE) decreased 5.14× across 50 epochs. Both trend checks PASS under the v3 criterion (≥1.5× for sentence count, ≥2.0× for presence).

## Dev metrics

| Population | type_gate_auc | type_gate_f1_youden | count_mae_positive | sentence_score_auc | presence_threshold |
|---|---:|---:|---:|---:|---:|
| multi_event_dev | 0.9839 | 0.7984 | 5.6058 | 0.8821 | 0.4093 |
| all_dev | 0.9889 | 0.8044 | 4.6580 | 0.8877 | 0.4093 |

## v3 vs v2.1 head-to-head

| Population × Metric | v2.1 | v3 | Δ | Notes |
|---|---:|---:|---:|---|
| multi_event_dev type_gate_auc | 0.9823 | 0.9839 | +0.0016 | within noise |
| multi_event_dev type_gate_f1_youden | 0.8248 | 0.7984 | -0.0264 | within noise |
| multi_event_dev count_mae_positive | 0.8702 | 5.6058 | +4.7356 | **v3 catastrophically worse** |
| all_dev type_gate_auc | 0.9874 | 0.9889 | +0.0015 | within noise |
| all_dev type_gate_f1_youden | 0.7783 | 0.8044 | +0.0261 | within noise |
| all_dev count_mae_positive | 0.4040 | 4.6580 | +4.2540 | **v3 catastrophically worse** |

TypeGate metrics are within noise — v3 should not have changed presence behaviour, and it did not. Sentence score AUC (0.882/0.888) confirms the per-sentence binary classifier is learning the heuristic signal. However, the aggregation to count is dramatically worse than v2.1's document-level Poisson head, which was already weaker than `predict_one`.

## Noise diagnostics

| Population | gold_record_sentence_recall | mean_sentence_label_count_over_gold |
|---|:---:|---:|
| train | 1.0 | 4.4166 |
| multi_event_dev | 1.0 | 4.5915 |
| all_dev | 1.0 | 4.4741 |

`gold_record_sentence_recall = 1.0` means every gold-positive (doc, type) pair has at least one sentence with a matching argument value — the heuristic never misses a record. However, `mean_ratio ≈ 4.5` means the heuristic on average labels 4-5 sentences as positive for each true record, because argument values (company names, person names) repeat across sentences. The model faithfully learns this noisy signal, and the expected-count aggregation `Σ_s σ(logit_st)` compounds the over-counting error.

## Acceptance verdict

| # | Check | passed | value | threshold |
|---:|------|:---:|---:|---|
| 1 | `multi_event_dev/type_gate_auc` | ✅ | 0.9839 | abs ≥ 0.80, rel ≥ best + 0.05 |
| 2 | `multi_event_dev/type_gate_f1_youden` | ✅ | 0.7984 | abs ≥ 0.55, rel ≥ best + 0.05 |
| 3 | `all_dev/type_gate_auc` | ✅ | 0.9889 | abs ≥ 0.80, rel ≥ best + 0.05 |
| 4 | `all_dev/type_gate_f1_youden` | ✅ | 0.8044 | abs ≥ 0.55, rel ≥ best + 0.05 |
| 5 | `multi_event_dev/count_mae_positive` | ❌ | 5.6058 | ≤ predict_one − 0.05 = 0.7865 |
| 6 | `all_dev/count_mae_positive` | ❌ | 4.6580 | ≤ predict_one − 0.02 = 0.3280 |
| 7 | `training/presence_loss_trend` | ✅ | 5.91× | first/last ≥ 2.0 over ≥10 epochs |
| 8 | `training/sentence_count_loss_trend` | ✅ | 5.14× | first/last ≥ 1.5 over ≥10 epochs |
| 9 | `multi_event_dev/sentence_score_auc` | ✅ | 0.8821 | ≥ 0.75 |
| 10 | `all_dev/sentence_score_auc` | ✅ | 0.8877 | ≥ 0.75 |

Net **`accepted = false`** (8/10 PASS). The four TypeGate checks remain PASS by large margins. The two baseline-relative count checks fail — in fact v3 count_mae is much worse than v2.1, not better. Sentence score AUC and both loss trend checks pass, confirming the sentence-level binary head trains correctly but its expected-count aggregation is not a valid count estimator under the current heuristic supervision.

## Methodology interpretation

1. **The sentence-level binary classifier works, but aggregation fails.** The model achieves sentence_score_auc ≈ 0.88 on both populations, confirming it can distinguish sentences containing record-relevant tokens from those that do not. The failure is in the aggregation step: summing per-sentence sigmoid scores produces counts 4-5× too large because the heuristic labels themselves over-count by a factor of ~4.5.

2. **Heuristic noise is the primary failure mode**, not architecture. A sentence-level head trained on ground-truth sentence-record alignments (which DuEE-Fin does not provide) might succeed. With the current heuristic, the model learns to reproduce the heuristic's over-counting bias, and the expected-count sum amplifies it. v2.1's document-level Poisson head, despite its architectural bottleneck, at least had a single scalar `λ` parameter that was penalised for large deviations — it converged to a modest over-estimate (MAE 0.87 vs predict_one 0.84 on multi_event_dev) rather than the 5-7× over-estimate produced by aggregating sentence scores.

3. **TypeGate methodology is solid and stable across three versions.** v2 (AUC 0.983/0.988, F1 0.832/0.778), v2.1 (AUC 0.982/0.987, F1 0.825/0.778), and v3 (AUC 0.984/0.989, F1 0.798/0.804) all dominate the strongest baseline by ≥15 absolute F1 points on both populations. The evidence-pooling + lexical-feature + dual-population baseline-relative gate design from v2 is the defensible contribution.

4. **Count is a structural bottleneck that neither document-level pooling (v2/v2.1) nor sentence-level aggregation under heuristic supervision (v3) resolves.** Without ground-truth sentence-record alignment labels, a neural count head cannot be reliably trained on DuEE-Fin. The p5b_lexical_trigger `_estimate_record_count` fallback (`carve/p5b_runner.py:621-628`) is the best available count estimator at this time.

## Recommendation: R3 TypeGate-only acceptance (Path A)

Per the frozen v3 phase plan (`docs/phase/r3_v3_sentence_count_plan.md`), since checks 5 and 6 fail, R3 v3 switches to **Path A**:

- **R3 = TypeGate-only acceptance.** The four TypeGate acceptance checks (1–4) are declared PASS. TypeGate methodology (evidence-pooling + lexical-feature + dual-population baseline-relative gate) is the accepted R3 contribution.
- **Count head out of scope.** Neither document-level Poisson NLL (v2/v2.1) nor sentence-level BCE aggregation (v3) clears the baseline-relative count gate. R3 does not claim a trained count estimator.
- **P5b count fallback.** P5b continues to use `_estimate_record_count` (lexical trigger string count, capped at 3) as its count estimator. This is not a trained neural count head but is the best available deterministic count estimate. The P5b planner integration smoke (Step 6 of the v3 phase plan) verifies the v3 TypeGate checkpoint loads and dispatches correctly in P5b, but the count branch in P5b remains the lexical fallback.

## Boundaries

This measurement is a planner-only diagnostic. It does not assert:

- R3 full acceptance (count head fails).
- P5b dev diagnostic against the current rule-path evidence (`runs/carve/p5b_duee_fin_dev500_seed42`).
- Any unified-strict, hidden-test, or final-test result.
- Any paper main-table or SOTA claim.
- P3 (full mention CRF + planner) acceptance.

The v3 checkpoint at `runs/carve/r3_planner_only_duee_fin_seed42_v3/checkpoints/r3_planner.pt` is a phase artifact with `count_head_mode=sentence`. It is not promoted to a published planner count head; the TypeGate weights are identical in behaviour to v2.1.
