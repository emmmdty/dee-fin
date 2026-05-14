# R3 v4 APCC Gold-Mention Measurement

Date: 2026-05-15
Run: `runs/carve/r3_planner_only_duee_fin_seed42_v4_gold/`
Status: **accepted=false** (10/13 checks)

## Purpose

R3 v4 replaces direct count regression (v2/v2.1 document-level Poisson, v3 sentence-level BCE) with
**Argument Pairwise Coreference Cluster (APCC)**. For each event type `t` in a document:

1. Extract candidate argument mentions from sentences
2. Predict pairwise coreference affinity `P(same_record | m_i, m_j, t)` via `ArgumentCoreferenceHead`
3. Threshold the affinity matrix and run connected-components clustering
4. `n_t = #clusters`

This measurement uses **gold-mention extraction** (oracle argument values matched in sentence text)
to establish the APCC upper bound. The CRF-mention path is pending a trained P3 MentionCRF checkpoint.

## Run Configuration

| Parameter | Value |
|---|---|
| Dataset | DuEE-Fin-dev500 |
| Train docs | 6515 (all_train) |
| Dev docs | 146 (multi_event_dev) / 500 (all_dev) |
| Encoder | chinese-roberta-wwm-ext (frozen) |
| Feature mode | evidence_lexical |
| Count head mode | coref |
| Mention source | gold |
| Max mentions | 64 |
| Coref threshold grid | 0.30, 0.40, 0.50, 0.60, 0.70 |
| Epochs | 50 |
| Batch size | 64 |
| Seed | 42 |
| Elapsed | ~18 min (gpu-4090) |

## Loss Trajectory

| Epoch | loss | presence_loss | count_loss (coref) |
|---:|---:|---:|---:|
| 1 | 3.819 | 1.209 | 2.622 |
| 5 | 2.199 | 1.015 | 1.189 |
| 10 | 0.988 | 0.642 | 0.349 |
| 20 | 0.492 | 0.337 | 0.155 |
| 30 | 0.398 | 0.263 | 0.135 |
| 40 | 0.338 | 0.232 | 0.107 |
| 50 | 0.303 | 0.213 | 0.090 |

presence_loss decrease: 5.68×. coref_loss decrease: 29.1×. Both trend checks PASS.

## Dev Metrics

| Metric | multi_event_dev | all_dev |
|---|---|---|
| TypeGate AUC | 0.9845 | 0.9891 |
| TypeGate F1 (Youden) | 0.8174 | 0.8051 |
| **count_mae_positive** | **0.8173** | **0.3880** |
| pair_auc | 0.8408 | 0.8556 |
| cluster_b3_precision | 0.3582 | 0.5938 |
| cluster_b3_recall | 0.7094 | 0.7412 |
| cluster_b3_f1 | 0.4760 | 0.6593 |
| ambiguous_pair_rate | 0.0462 | 0.0275 |

## Baselines (count_mae_positive)

| Baseline | multi_event_dev | all_dev |
|---|---|---|
| predict_one | 0.8365 | 0.3480 |
| p5b_lexical_trigger | 1.6875 | 1.4220 |
| legacy_single_softmax | 1.8365 | 1.3480 |
| v2.1 Poisson (static) | 0.8702 | 0.4040 |
| v3 Sentence BCE (static) | 5.6058 | 4.6580 |
| **v4 Gold APCC** | **0.8173** | **0.3880** |

### Threshold Grid Search (train MAE)

| τ | 0.10 | 0.15 | 0.20 | **0.25** | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train MAE | 0.306 | 0.293 | 0.285 | **0.275** | 0.278 | 0.292 | 0.322 | 0.376 | 0.471 |

Best τ = 0.25 (extended grid). At τ = 0.25, count_mae = 0.808 (multi) / 0.380 (all).

## Coreference Diagnostics

| Population | Matched | Ambiguous | Ambiguity % | Pos pairs | Neg pairs | Pos ratio |
|---|---|---|---|---|---|---|
| train | 66,471 | 11,870 | 15.2% | 346,705 | 56,466 | 86.0% |
| multi_event_dev | 2,506 | 927 | 27.0% | 14,478 | 4,643 | 75.7% |
| all_dev | 5,182 | 927 | 15.2% | 28,091 | 4,643 | 85.8% |

Ambiguity is manageable: 15% on train/all, 27% on multi_event_dev. Positive pair ratio is high (76-86%),
meaning most eligible mention pairs within the same event type belong to the same record.

## Acceptance Verdict

### 10/13 PASS, 3 FAIL

| # | Check | Value | Threshold | Result |
|---:|---|---|---|---|
| 1 | multi_event_dev type_gate_auc | 0.9845 | >= 0.80 + baseline+0.05 | ✅ |
| 2 | multi_event_dev type_gate_f1_youden | 0.8174 | >= 0.55 + baseline+0.05 | ✅ |
| 3 | all_dev type_gate_auc | 0.9891 | >= 0.80 + baseline+0.05 | ✅ |
| 4 | all_dev type_gate_f1_youden | 0.8051 | >= 0.55 + baseline+0.05 | ✅ |
| 5 | **multi_event_dev count_mae_positive** | **0.8173** | <= min(baselines) - 0.05 = 0.7865 | ❌ |
| 6 | **all_dev count_mae_positive** | **0.3880** | <= min(baselines) - 0.02 = 0.3280 | ❌ |
| 7 | multi_event_dev pair_auc | 0.8408 | >= 0.75 | ✅ |
| 8 | all_dev pair_auc | 0.8556 | >= 0.75 | ✅ |
| 9 | **multi_event_dev cluster_b3_f1** | **0.4760** | >= 0.65 | ❌ |
| 10 | all_dev cluster_b3_f1 | 0.6593 | >= 0.65 | ✅ |
| 11 | training/presence_loss_trend | 5.68× over 50 epochs | >= 2.0 over >= 10 epochs | ✅ |
| 12 | training/coref_loss_trend | 29.1× over 50 epochs | >= 1.5 over >= 10 epochs | ✅ |
| 13 | ambiguity_audit | rates present (0.046, 0.028) | field required | ✅ |

### Failed checks

- **count_mae_positive (both populations)**: v4 count MAE (0.808/0.380 at best τ=0.25) beats v2.1 (0.870/0.404, -6.1%/-5.9%) and crushes v3 (5.606/4.658), but the gap to `predict_one` (0.836/0.348) is only 0.028/0.032 — below the acceptance margin of 0.05/0.02. On multi_event_dev, the model improves count by 2.8% over predict_one; on all_dev, it is 0.032 worse.

- **cluster_b3_f1 (multi_event_dev)**: 0.476 vs threshold 0.65. The B³ precision of 0.358 on multi_event_dev reveals systematic over-merging — the low clustering threshold (τ=0.25) creates false edges between mentions belonging to different records of the same event type. Connected components amplifies these errors through transitive closure.

## Ablation: global_only Feature Mode

To test whether sentence-level encoder features contribute to coreference beyond span
representations, E2.2 was run with `--encoder-feature-mode global_only` (no evidence pooling,
no lexical hit feature, TypeGate sees only `[global_repr; type_emb]`).

### Results (runs/carve/r3_planner_only_duee_fin_seed42_v4_gold_globalonly/)

| Metric | evidence_lexical | global_only | Δ |
|---|---|---|---|
| TypeGate AUC (multi) | 0.9845 | 0.6100 | -0.375 |
| TypeGate AUC (all) | 0.9891 | 0.7048 | -0.284 |
| count_mae (multi) | 0.8173 | 1.0625 | +0.245 |
| count_mae (all) | 0.3880 | 0.5960 | +0.208 |
| pair_auc (multi) | 0.8408 | 0.7954 | -0.045 |
| pair_auc (all) | 0.8556 | 0.8159 | -0.040 |
| cluster_b3_f1 (multi) | 0.4760 | 0.1367 | -0.339 |
| cluster_b3_f1 (all) | 0.6593 | 0.1696 | -0.490 |
| presence_loss trend | 5.68× | 1.04× | — |
| coref_loss trend | 29.1× | 29.1× | same |

TypeGate collapses to v1 level without evidence features (AUC 0.61 → matches the original
v1 failure documented in `docs/phase/p3_replan.md`). The coreference head retains most
pair-discrimination ability (pair_auc -0.04 to -0.05) but cluster B³ F1 collapses — likely
because the broken TypeGate creates false negatives that cascade into count=0 predictions
for gold-positive (doc, type) pairs, removing them from the B³ sample. The coref_loss
trend is identical (29.1×), confirming that the coreference head converges equally
regardless of TypeGate feature quality.

**Conclusion**: sentence-level evidence features are critical for the full pipeline.
Coreference head benefits modestly from them (pair_auc +0.04-0.05) but TypeGate is
completely dependent on them.

## Methodology Interpretation

### What works

1. **Pairwise coreference is learnable.** pair_auc 0.84-0.86 under gold mentions proves that
   the model can distinguish same-record from different-record mention pairs. The
   `ArgumentCoreferenceHead` architecture (scaled dot-product affinity + same-role/
   same-sentence structural biases) captures the coreference signal.

2. **Count prediction improves over prior neural approaches.** v4 gold APCC beats v2.1
   Poisson by 6.1% (multi) / 5.9% (all) on count_mae_positive, and crushes v3's
   catastrophic heuristic-noise amplification (5.606 → 0.808).

3. **TypeGate is unchanged and solid.** AUC ~0.98-0.99, F1 ~0.78-0.82 across all
   evidence_lexical runs, consistent with v2/v2.1/v3.

### What doesn't work

1. **Clustering precision is the bottleneck.** B³ precision of 0.358 on multi_event_dev
   means the model over-merges: when it predicts two mentions belong to the same record,
   it is wrong 64% of the time. The root cause is the connected-components algorithm,
   which amplifies individual false edges through transitive closure. A single false
   positive edge between two otherwise-correct clusters merges them into one.

2. **Count does not beat predict_one by a convincing margin.** The improvement over
   always-predict-1 is only 2.8% on multi_event_dev (0.8365 → 0.8173 at τ=0.3,
   or 0.808 at τ=0.25). On all_dev, v4 is actually 0.032 worse than predict_one.

### Why clustering precision is low

With gold mentions at τ=0.25, the model partitions 2506 non-ambiguous mentions across
208 positive (doc, type) pairs into ~14,478 positive and ~4,643 negative eligible pairs
on multi_event_dev. The positive ratio is 75.7% — most eligible pairs are same-record.
At low threshold (τ=0.25, corresponding to sigmoid−1(0.25) ≈ −1.1 logit), the model
creates edges for most moderately positive pairs. Even though pair_auc 0.84 means ranking
is good, the absolute sigmoid values are not calibrated: the model assigns moderate
probabilities (~0.5-0.7) to negative pairs, which at τ=0.25 become false edges. These
merge distinct-record clusters, producing B³ precision 0.36.

### Comparison to prior versions

| Method | Count mechanism | multi count_mae | all count_mae | Count gate |
|---|---|---|---|---|
| predict_one | Always n=1 | 0.8365 | 0.3480 | baseline |
| v2.1 Poisson | Document-level λ prediction | 0.8702 | 0.4040 | FAIL |
| v3 Sentence BCE | Per-sentence binary + sum | 5.6058 | 4.6580 | CATASTROPHIC |
| **v4 Gold APCC** | Mention coreference → clusters | **0.8080** | **0.3800** | FAIL (marginal) |

v4 is the best neural count estimator, but the improvement over a trivial baseline is
insufficient for acceptance. The key architectural insight — count emerges from clustering
rather than regression — is validated, but connected components with a single global
threshold is too crude for the precision required.

### What would close the count gap

- **Cluster-level loss**: Train with a loss that directly penalizes clustering errors
  (e.g., B³ loss or cluster F1 loss) rather than pairwise BCE + threshold search.
- **Higher threshold with precision optimization**: Accept lower recall to gain precision.
  The threshold search currently optimizes train count MAE, which implicitly trades
  precision for recall. Optimizing for B³ precision directly would help.
- **Post-hoc cluster splitting**: Apply a second-stage splitter that breaks large clusters
  using role-consistency constraints or mention proximity.

## Boundaries

- This measurement uses **gold mention extraction** (oracle). The CRF-mention realistic
  path requires a trained P3 MentionCRF checkpoint (not yet produced).
- Only DuEE-Fin-dev500 measured. No ChFinAnn, DocFEE, or other datasets.
- No P5b pipeline integration tested with v4 checkpoint.
- No hidden-test, final-test, or paper main-table updates.
- The extended threshold grid (E2.1) used a separate run with τ={0.10..0.70}; results
  are from a fresh training with the same seed 42.

## Artifacts

- `runs/carve/r3_planner_only_duee_fin_seed42_v4_gold/` — E1.1 primary run
- `runs/carve/r3_planner_only_duee_fin_seed42_v4_gold_threshold9/` — E2.1 extended grid
- `runs/carve/r3_planner_only_duee_fin_seed42_v4_gold_globalonly/` — E2.2 global_only ablation
- `runs/carve/r3_planner_only_duee_fin_seed42_v4_gold_smoke/` — E0.2 local smoke
