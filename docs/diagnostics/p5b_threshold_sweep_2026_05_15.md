# P5b PlannerGate Threshold Sweep

Date: 2026-05-15
Inputs: R3 v4 PlannerGate, fixed allocation checkpoint from `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate`
Question: does lowering `--planner-presence-threshold` close the gap between carve F1 and baseline F1?

## Setup

- Allocation checkpoint reused (`--alloc-checkpoint`); training fully skipped per run (~5 min wall).
- Planner checkpoint: R3 v4 (`r3_planner_only_duee_fin_seed42_v4_gold/checkpoints/r3_planner.pt`).
- 4 runs in parallel on 4 GPUs (thresholds 0.05/0.10/0.15/0.20). Baseline run at 0.27 from earlier.
- All other parameters identical (share_threshold=0.50, batch_size=4, etc.).

## Results (unified strict, multi-event dev = 146 docs)

| threshold | route | TP | FP | FN | precision | recall | F1 | pred_records |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | baseline | 199 | 2,271 | 1,828 | 0.0806 | 0.0982 | 0.0885 | 438 |
| 0.05 | carve    | 105 | 1,567 | 1,922 | 0.0628 | 0.0518 | 0.0568 | 544 |
| 0.10 | baseline | 196 | 1,723 | 1,831 | 0.1021 | 0.0967 | 0.0993 | 335 |
| 0.10 | carve    | 103 | 1,334 | 1,924 | 0.0717 | 0.0508 | 0.0595 | 458 |
| 0.15 | baseline | 195 | 1,495 | 1,832 | 0.1154 | 0.0962 | 0.1049 | 291 |
| 0.15 | carve    | 103 | 1,221 | 1,924 | 0.0778 | 0.0508 | 0.0615 | 423 |
| 0.20 | baseline | 193 | 1,371 | 1,834 | 0.1234 | 0.0952 | 0.1075 | 267 |
| 0.20 | carve    | 103 | 1,160 | 1,924 | 0.0816 | 0.0508 | 0.0626 | 401 |
| **0.27** | **baseline** | **192** | **1,244** | **1,835** | **0.1337** | **0.0947** | **0.1109** | **241** |
| **0.27** | **carve**    | **103** | **1,106** | **1,924** | **0.0852** | **0.0508** | **0.0637** | **381** |

## Findings

1. **F1 increases monotonically with threshold for both routes.** No sweet spot exists in [0.05, 0.27].
   The Youden-optimized threshold 0.27 (from R3 v4 checkpoint metadata) is also near-optimal for
   P5b downstream F1.

2. **Carve F1 never closes the gap to baseline F1.** The gap is widest at 0.27 (-0.047) and
   narrowest at 0.05 (-0.032), but carve is *worse than baseline at every threshold*.

3. **CRITICAL: carve TP is flat at 103–105 across all thresholds.** Carve recall is structurally
   capped at 0.051. Lowering PlannerGate admits more (doc, event_type) pairs, but carve catches
   no additional gold values. Baseline TP rises 192→199 with broader admission.

4. The 90% candidate-miss FN from Phase 1 (`docs/diagnostics/p5b_carve_alloc_breakdown.md`)
   is therefore *not* attributable to PlannerGate being too strict — it is a downstream
   allocation problem.

## Root Cause: Sinkhorn NULL-column drop

In `carve/p5b_runner.py:_predict_route`:

```python
null_score = row[-1]
best_index = int(torch.argmax(row[:-1]).item()) if record_count else 0
if row[best_index] <= null_score:
    continue  # <-- candidate dropped
```

When Sinkhorn predicts that no record column scores higher than the NULL column, the candidate
is discarded entirely. Baseline has no equivalent check — it always assigns the first candidate
to record 0. Carve's conservative drop rule is what produces the recall ceiling at TP=103.

Removing the NULL-drop check would inflate carve recall but tank precision (most dropped
candidates are correctly dropped — they really are noise). The fundamental issue is the
allocation model is over-confident in NULL assignment, possibly because:

- Allocation training labels assign most candidates to NULL (multi-event docs have many
  irrelevant candidates per role).
- The training loss plateaus at 0.966 from epoch 1 — the model is not learning much beyond
  a constant baseline.

## Conclusion

The threshold sweep cleanly refutes the hypothesis that PlannerGate threshold tuning can close
the carve-vs-baseline gap on DuEE-Fin. The bottleneck is the Sinkhorn NULL-column dominance,
not the presence gate. Fixing this requires either:

- Re-training the allocation head with better supervision (deeper retraining work).
- Replacing the NULL-drop rule with a softer admission policy (architectural change).
- Accepting that CARVE allocation is not viable on DuEE-Fin and pivoting the venue narrative.

See `docs/phase/venue_decision_2026_05_15.md` for the updated decision.

## Artifacts

- `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate_th{005,010,015,020}/` — 4 sweep runs
- `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate/` — original 0.27 run (reused as baseline row)
- `carve/p5b_runner.py:--alloc-checkpoint` — new flag enabling inference-only re-runs
