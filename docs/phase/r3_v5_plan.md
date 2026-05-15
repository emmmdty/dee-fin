# R3 v5 Acceptance Plan: Agglomerative APCC

Date: 2026-05-15
Predecessor: R3 v4 (10/13 checks PASS; failed count_mae both populations, multi_event cluster_b3_f1)
Baseline reference: `docs/measurements/r3_planner_only_duee_fin_seed42_v4.md`

## Motivation

v4 established that pairwise coreference is learnable (pair_auc 0.84–0.86 under gold mentions),
but connected-components clustering amplifies individual false positive edges through transitive
closure, yielding B³ precision 0.358 on multi_event_dev. A single false edge between two
otherwise-correct clusters merges them irreversibly. This precision bottleneck causes over-merging
that propagates into count MAE.

## Architecture Changes vs v4

| Component | v4 | v5 |
|---|---|---|
| Clustering algorithm | Connected-components (Union-Find) | Agglomerative, average linkage |
| Probability calibration | None (raw sigmoid) | Temperature scaling (T learned on dev) |
| Post-hoc precision | None | Role-conflict splitter (schema rule) |
| Pair training labels | eligible mask (ambiguity excluded) | Same — no change |
| ArgumentCoreferenceHead | Unchanged | Unchanged |
| TypeGate | Unchanged | Unchanged |

### Agglomerative Clustering (average linkage)

Replace `predict_clusters()` in `carve/p3_planner.py` with an agglomerative variant:

```python
# v5: average linkage — cluster merge requires *average* inter-cluster affinity >= tau
# A single false positive edge cannot trigger a cascade merge
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

probs = sigmoid(affinity)                    # [M, M]
dist  = 1.0 - probs                          # distance matrix
Z     = linkage(squareform(dist), method='average')
labels = fcluster(Z, t=(1 - tau), criterion='distance')
```

### Temperature Scaling

After training, fit scalar T on dev NLL:
```
T* = argmin_T  -sum_{eligible i<j} [ y_ij * log sigma(logit_ij / T)
                                    + (1-y_ij) * log(1 - sigma(logit_ij / T)) ]
```
Stored in checkpoint metadata as `coref_temperature`. Applied at inference before tau search.
Tau search runs on calibrated probabilities; best tau range is expected to shift to 0.40–0.60.

### Role-Conflict Splitter

Applied after agglomerative clustering, before count = #clusters:

```
for each cluster C:
    for each role rho marked unique in schema:
        values = {m.surface_value : m in C, m.role == rho}
        if |values| > 1:
            split C into sub-clusters by value  # one sub-cluster per unique value
```

No training required. Uses `EventSchema.roles_for()` and per-mention role assignments
from `MentionSpan.role_id`.

## Acceptance Checks (14 checks, all must pass)

| # | Check | Threshold | New in v5? |
|---:|---|---|---|
| 1 | multi_event_dev type_gate_auc | >= 0.80 + baseline+0.05 | No |
| 2 | multi_event_dev type_gate_f1_youden | >= 0.55 + baseline+0.05 | No |
| 3 | all_dev type_gate_auc | >= 0.80 + baseline+0.05 | No |
| 4 | all_dev type_gate_f1_youden | >= 0.55 + baseline+0.05 | No |
| 5 | **multi_event_dev count_mae_positive** | <= min(all_baselines) - 0.05 | No |
| 6 | **all_dev count_mae_positive** | <= min(all_baselines) - 0.02 | No |
| 7 | multi_event_dev pair_auc | >= 0.75 | No |
| 8 | all_dev pair_auc | >= 0.75 | No |
| 9 | **multi_event_dev cluster_b3_f1** | >= 0.65 | No (v4 failed) |
| 10 | all_dev cluster_b3_f1 | >= 0.65 | No |
| 11 | training/presence_loss_trend | >= 2.0× over >= 10 epochs | No |
| 12 | training/coref_loss_trend | >= 1.5× over >= 10 epochs | No |
| 13 | ambiguity_audit | ambiguous_pair_rate field present | No |
| 14 | **multi_event_dev cluster_b3_precision** | >= 0.55 | **YES — new** |

Check #14 targets the root cause of v4's failure: over-merging (B³ precision 0.358).
The agglomerative + splitter combination must recover precision to >= 0.55.

Baseline pool for count_mae checks (same as v4):
- predict_one, p5b_lexical_trigger, legacy_single_softmax, v2.1_poisson_static, v3_sentence_static

## Path A Fallback

If v5 fails count_mae checks (#5, #6) but passes TypeGate checks (#1–4):
Path A is confirmed active. TypeGate accepted; count uses lexical fallback in P5b.
This is consistent with the v3 Path A trigger already on record.

If v5 fails cluster_b3_precision check #14: record the precision value and diagnose
whether the role-conflict splitter is being applied correctly before declaring failure.

## Run Configuration

Same as v4 except `--count-head-mode coref_v5` (new mode name to distinguish from v4).
Encoder frozen, gold mentions (`--mention-source gold`), seed 42, 50 epochs, batch 64.

## Artifacts

- `runs/carve/r3_planner_only_duee_fin_seed42_v5_gold/` — primary run
- `runs/carve/r3_planner_only_duee_fin_seed42_v5_gold_smoke/` — smoke (3 epoch)
- `docs/measurements/r3_planner_only_duee_fin_seed42_v5.md` — measurement record

## Boundaries

- Gold mention extraction only. CRF-mention path requires trained P3 MentionCRF (not yet produced).
- DuEE-Fin-dev500 only. No ChFinAnn, DocFEE.
- No P5b pipeline integration tested with v5 checkpoint (separate P5b track).
