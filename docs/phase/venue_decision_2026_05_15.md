# Venue Decision: DuEE-Fin Evidence as of 2026-05-15

> Closing decision on Track A (R3) and Track B (P5b) for DuEE-Fin-dev500 only.
> Hidden-test, final-test, and the full three-dataset P5b decision table are out of scope.

## Inputs

| Track | Measurement | Verdict |
|---|---|---|
| R3 v5 + affinity-aware role splitter | `docs/measurements/r3_planner_only_duee_fin_seed42_v5.md` §"Role-Conflict Splitter" | **12/14** — first time both count gates pass |
| R3 v5 (agglomerative APCC + temperature, no splitter) | same doc, main section | accepted=false (11/14) |
| P5b + R3 v4 PlannerGate (Path A) | `docs/measurements/p5b_duee_fin_dev500_r3typegate.md` | dev_diagnostic_only |
| P5b FP decomposition (original) | `docs/diagnostics/p5b_fp_breakdown.md` | H2 confirmed |
| P5b FP decomposition (PlannerGate) | `docs/diagnostics/p5b_fp_breakdown_r3typegate.md` | H2 fix validated |
| P5b CARVE allocation FP breakdown | `docs/diagnostics/p5b_carve_alloc_breakdown.md` | 80% FP = noise records; 0% share-gate excess |
| P5b PlannerGate threshold sweep | `docs/diagnostics/p5b_threshold_sweep_2026_05_15.md` | **no threshold closes carve-baseline gap**; carve TP flat at 103 |

## Confirmed Path A

Path A (TypeGate from R3, record count from lexical fallback) was the v3 decision and remains
the operative mode for downstream P5b integration. Evidence:

- R3 v5 TypeGate AUC = 0.985 (multi) / 0.989 (all); F1_youden = 0.817 / 0.805 — stable across
  v2.1, v3, v4, v5.
- R3 v5 count gate now passes on `multi_event_dev` (0.755 ≤ 0.787) for the first time, but
  fails on `all_dev` (0.376 > 0.328) — Path A still required for the all_dev population.
- P5b with R3 v4 PlannerGate (Path A: TypeGate-only, count via lexical) cuts CARVE-route FPs
  by 97% (39,098 → 1,106) — confirms TypeGate-only is operationally sufficient for the FP
  bottleneck regardless of count gate status.

`carve/p5b_runner.py:PlannerGate.predict` now returns `(present, None)` whenever the
checkpoint's `count_head_mode` is `coref` or `coref_v5`, letting the caller fall back to
the lexical estimator. This is the mechanical implementation of Path A.

## Methodology Wins

1. **R3 v5 = first neural count head to beat predict_one by margin on multi_event_dev**
   (0.755 vs 0.787 threshold). Improvement is from inference-time changes only: connected
   components → agglomerative average linkage, plus dev-NLL temperature calibration. Both
   are zero-cost retrofits on a trained pairwise affinity model.

1b. **R3 v5 + role-conflict splitter = first method to pass BOTH count gates** (multi
   0.298 ≪ 0.787; all 0.306 < 0.328). The all_dev count gate had failed on every prior
   version (v2/v2.1/v3/v4 and v5-without-splitter). The splitter is also gold-mention-only;
   it requires per-mention role assignments that are unavailable from a trained MentionCRF.
   For the paper, the splitter is best framed as the gold-mention upper bound, with the
   pure-v5 path as the realistic-mention-pipeline contribution.

2. **R3 PlannerGate diagnostic validates the H2 hypothesis** for P5b's FP explosion. The
   original P5b had 90% of predicted records as excess; after PlannerGate this collapses to
   40% (154 / 381). Engineering integration of a learned presence head into the P5b pipeline
   is a clear win.

## Methodology Limits

1. **R3 v5 + affinity splitter still fails 2/14 checks**: multi cluster_b3_f1 (0.525 vs
   0.65) and multi cluster_b3_precision (0.443 vs 0.55). The splitter brings B³ precision
   from 0.358 (v4) to 0.443 (+0.085, ~50% of the way to threshold); the remaining gap is a
   representation-quality bound. The splitter is gold-mention-only — would need CRF mention
   + role tagging to use in realistic pipeline.

2. **CARVE allocation mechanism does not beat baseline on DuEE-Fin**, even with PlannerGate.
   - Original run: carve F1 0.011 vs baseline F1 0.037 — gap -0.026.
   - With PlannerGate: carve F1 0.064 vs baseline F1 0.111 — gap **widened to -0.047**.
   - Threshold sweep ∈ {0.05, 0.10, 0.15, 0.20, 0.27}: no threshold closes the gap. Best gap
     achieved is -0.032 at threshold=0.05 (worst threshold for both routes individually).
     **Carve TP is structurally capped at 103** across all thresholds while baseline ranges
     from 192 to 199.
   - Root cause identified: Sinkhorn allocation has a NULL-column dominance check that drops
     candidates the simpler baseline rule keeps. This is an architectural property of the
     CARVE allocation, not a tunable parameter.

3. **`misallocated_rate_eligible` is not measured** by the current P5b runner — the primary
   metric in `docs/measurements/p5b_decision_table_template.md` is unavailable, so the
   strong/weak/no-support classification cannot be issued cleanly.

4. **ChFinAnn and DocFEE rows have not been run**. The full P5b decision table is one row of
   three.

5. **Role-conflict splitter deferred** (MentionSpan lacks `role_id` field; requires data
   pipeline augmentation). Could plausibly push B³ precision past 0.55 if implemented.

## Decision

Apply the P5b decision rules to what we have on DuEE-Fin:

| Rule | Applies? | Result |
|---|---|---|
| Three strong → EMNLP main stretch | No (one dataset, missing primary metric) | — |
| Two strong + one weak → EMNLP main or Findings | No (same) | — |
| Three weak → Findings/COLING with stable F1 | No | — |
| One strong + two weak, or any no-support → COLING / SCI Q2 / simplify to ECPD-CRV | DuEE-Fin row is no-support on Δ F1 | **Applies** |
| Two+ no-support → CARVE mechanism fails; no main framing | Pending other datasets | — |

**Decision (DuEE-Fin alone, reinforced by threshold sweep evidence)**: **COLING / SCI Q2 framing**,
or simplify the methodology narrative away from the full CARVE allocation mechanism. The
contribution worth defending is the **two-stage planner (R3) as a precondition for argument
extraction**, not the allocation mechanism. The R3 PlannerGate-driven FP collapse is the
cleanest result.

The threshold sweep evidence is the deciding factor: at the cheapest possible knob (inference-time
threshold), there is no setting that makes carve competitive. The bottleneck is structural
(Sinkhorn NULL-drop), not tunable. Fixing it would require allocation-head retraining or
architectural changes, both of which are larger investments than the current DuEE-Fin
signal justifies.

This decision is provisional on the single DuEE-Fin row; ChFinAnn or DocFEE may shift it,
but the structural property identified (NULL-column dominance) is dataset-independent and
likely transfers.

## Recommended Next Steps (ranked, updated 2026-05-15 evening)

1. **Pivot venue narrative to "R3 planner as precondition"**. The defensible technical claims
   are: (a) R3 v5 is the first neural count head to pass the multi-event count gate, (b)
   integrating a trained TypeGate into P5b eliminates 97% of its FP explosion. These are
   both real and measured. The CARVE allocation mechanism is *not* part of the defensible
   claim set on DuEE-Fin.
2. **Write up R3 v5 + PlannerGate integration as the main contribution**. Self-contained,
   reproducible, beats prior methods (v2/v2.1/v3/v4) on count_mae, and demonstrates clear
   downstream value (P5b FP collapse).
3. **Do NOT pursue Sinkhorn allocation tuning on DuEE-Fin further**. The threshold sweep
   evidence shows the bottleneck is architectural (NULL-column dominance), not a tunable
   parameter. Fixing it would be a paper-on-its-own scope.
4. **Defer ChFinAnn and DocFEE P5b runs**. Even if those rows somehow showed carve > baseline
   (unlikely given the structural diagnosis), the DuEE-Fin no-support row would still kill
   the "three strong" pattern. GPU time is better spent on the R3 v5 write-up.
5. **~~(Optional, if reviewers push back)~~ DONE 2026-05-15**: role-conflict splitter
   implemented with both naive and affinity-aware untagged-mention assignment. v5+splitter
   reaches 12/14, passing both count_mae gates (first ever on this project). B³ precision
   reached 0.443 — meaningful improvement but did not clear 0.55. The splitter is gold-
   mention-only and is framed in the paper as the gold-mention upper bound rather than the
   pipeline acceptance path.
6. **(Optional, only if pivoting to a CARVE-mechanism paper)** Retrain the allocation head
   with a recall-weighted loss or remove the NULL-drop check at inference. Both require
   substantial work and would only matter if the venue narrative includes the allocation
   mechanism — which the current recommendation says it should not.

## Boundaries

- This decision is **DuEE-Fin dev only**.
- No hidden-test or final-test evidence is used.
- No SOTA claim is made.
- The paper main table is unchanged.
- P5b decision table is incomplete (1 of 3 dataset rows; primary metric missing).
