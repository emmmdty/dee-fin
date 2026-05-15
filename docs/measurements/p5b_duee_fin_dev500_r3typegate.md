# P5b DuEE-Fin Dev + R3 v4 PlannerGate (Path A)

> Status: measured dev diagnostic evidence — first DuEE-Fin row with the R3 PlannerGate active.
> Compared head-to-head against the original DuEE-Fin dev run (no PlannerGate) recorded in
> `docs/measurements/p5b_duee_fin_dev500_seed42.md`.
> Scope: dev diagnostic only. Not hidden-test, not final-test, not the full three-dataset P5b decision table.

## Run Identity

| Item | Value |
|---|---|
| Date | 2026-05-15 |
| Dataset | `DuEE-Fin-dev500` |
| Split | dev diagnostics only |
| Training source | train split (multi-event subset) only |
| Server root | `/data/TJK/DEE/dee-fin` |
| Server Python | `/home/TJK/.conda/envs/tjk-feg/bin/python` |
| Run directory | `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate` |
| Long-run log | `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate/logs/train.log` |
| Planner checkpoint | `runs/carve/r3_planner_only_duee_fin_seed42_v4_gold/checkpoints/r3_planner.pt` |
| Planner feature mode | `evidence_lexical` |
| Planner presence threshold | 0.269895 (from v4 metadata) |
| Planner count head mode | `coref` (v4) — Path A: record count falls back to lexical heuristic |
| Elapsed seconds | 1693.8 |
| Final logged epoch | 8 (early stop, patience=5) |

## Path A Semantics

`PlannerGate.predict(document, event_type)` returns `(present, count_or_None)`:
- `present`: from the trained R3 v4 presence head (sigmoid > 0.270 threshold).
- `count`: returned as `None` because the v4 checkpoint's count head is in `coref` mode and
  needs mention extraction at inference; no MentionCRF was available. The caller falls back to
  the lexical `_estimate_record_count` heuristic — this is the v3 Path A pattern reaffirmed.

This makes the experiment a clean **TypeGate-only** test: does replacing the rule-based
`_type_gate` with the trained R3 TypeGate cut P5b's FP explosion (39,098 in the original run)?

## Long-Run Result

Route metrics from `summary.json` (unified strict):

| Route | Predicted records | Candidates | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 241 | 33,371 | 192 | 1,244 | 1,835 | 0.1337 | 0.0947 | 0.1109 |
| carve | 381 | 33,371 | 103 | 1,106 | 1,924 | 0.0852 | 0.0508 | 0.0637 |

## Head-to-Head with Original P5b Run

Original P5b (no PlannerGate, rule `_type_gate` + lexical `_estimate_record_count`):

| Route | Predicted records | TP | FP | F1 |
|---|---:|---:|---:|---:|
| baseline | 1,892 | 202 | 8,599 | 0.0373 |
| carve | 3,082 | 225 | 39,098 | 0.0109 |

With R3 v4 PlannerGate (this run):

| Route | Predicted records | TP | FP | F1 |
|---|---:|---:|---:|---:|
| baseline | 241 (-87%) | 192 (-5%) | 1,244 (-86%) | 0.1109 (+3.0×) |
| carve | 381 (-88%) | 103 (-54%) | 1,106 (-97%) | 0.0637 (+5.8×) |

**Headline**: the R3 v4 TypeGate cuts the carve route's FP count by 97% (from 39,098 to 1,106).
Recall drops as expected (carve 0.111 → 0.051, baseline 0.100 → 0.095) because the strict
TypeGate also filters out some true (doc, event_type) pairs. F1 improves dramatically on both
routes, with carve gaining 5.8× and baseline gaining 3.0×.

## H2 Diagnosis Validated

The P5b FP decomposition diagnostic (`docs/diagnostics/p5b_fp_breakdown.md`, original run)
identified record count over-prediction (H2) as the dominant FP source: 2,775 excess records
out of 3,082 predicted (90%). After applying PlannerGate:

| Indicator | Original P5b | + R3 PlannerGate | Δ |
|---|---:|---:|---:|
| Total carve FP | 39,098 | 1,106 | **-97%** |
| Predicted record count | 3,082 | 381 | **-88%** |
| Excess records (n_pred > n_gold) | 2,775 | 154 | **-94%** |
| Hallucination rate | 0.1% | 0.4% | +0.3pp (still negligible) |

(See `docs/diagnostics/p5b_fp_breakdown_r3typegate.md` for the post-PlannerGate per-event
table.) The H2 hypothesis is now confirmed empirically: gating (doc, event_type) admissions
with a learned presence head removes ~85% of the (doc, type) pairs that the rule-based
TypeGate had been admitting, which in turn removes ~94% of the excess records and ~97%
of the false positive arguments.

## Carve vs Baseline Gap

Original P5b had carve − baseline F1 = -0.0264 (carve worse by 26pp relative).
With PlannerGate, carve − baseline F1 = -0.0472 (carve worse by 43pp relative).

The gap **widened** because the PlannerGate helped baseline more than carve in absolute terms.
This is consistent with the diagnostic: most FPs were attributable to spurious (doc, type)
admissions, which TypeGate now blocks regardless of allocation strategy. The Sinkhorn
allocation + share gate mechanism on top of a fixed candidate pool does not, on this dataset,
recover the recall it costs versus baseline's "first candidate per role" rule.

This is a **negative methodology signal for the CARVE allocation mechanism on DuEE-Fin**,
even though the engineering integration of R3 TypeGate into P5b is a clear win.

## Decision Row

```json
{
  "baseline_unified_strict_f1": 0.1109,
  "carve_unified_strict_f1": 0.0637,
  "dataset": "DuEE-Fin-dev500",
  "decision_rule": "first DuEE-Fin diagnostic row with PlannerGate; full P5b requires all datasets and final table",
  "split": "dev",
  "status": "dev_diagnostic_only_not_final_test",
  "support_label": "No support"
}
```

Per the P5b decision table (`docs/measurements/p5b_decision_table_template.md`):
- The `misallocated_rate_eligible` metric is not computed by the current runner → cannot
  classify Strong/Weak/No support by the primary rule.
- Δ record F1 (carve − baseline) is -0.0472, which is worse than the original run's -0.0264 →
  by the secondary rule on record F1, this is consistent with **No support** for the carve
  mechanism on DuEE-Fin (the decline in carve-relative-to-baseline F1 exceeds the 0.5
  absolute-point tolerance interpreted on the scale of these F1 values).

## Interpretation Boundary

- This run only validates the **R3 TypeGate integration into P5b** (an engineering fix for
  the FP explosion), not the **CARVE allocation mechanism**.
- The CARVE allocation mechanism (Sinkhorn prior + share gate) still does not beat baseline
  on this dataset; the gap actually widens once TypeGate is correctly applied.
- DuEE-Fin alone cannot complete the P5b decision table; ChFinAnn and DocFEE rows are missing.
- `misallocated_rate_eligible` is not measured by the runner → the primary P5b metric is
  unavailable.
- The Path A pattern is confirmed valid: TypeGate from R3 is strong enough to be used in
  downstream pipelines even when the count head fails acceptance.

## Artifacts

- `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate/summary.json` (synced to local)
- `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate/canonical/{dev.gold,dev.baseline.pred,dev.carve.pred}.jsonl`
- `runs/carve/p5b_duee_fin_dev500_seed42_r3typegate/eval/dev.{baseline,carve}.unified_strict.json`
- `docs/diagnostics/p5b_fp_breakdown_r3typegate.md` — post-PlannerGate FP decomposition
- `carve/p5b_runner.py:PlannerGate.predict` — Path A graceful degradation for coref count heads
