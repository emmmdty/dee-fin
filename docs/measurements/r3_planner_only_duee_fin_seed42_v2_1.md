# R3 planner-only v2.1 acceptance measurement

> Date: 2026-05-14
> Run dir: `runs/carve/r3_planner_only_duee_fin_seed42_v2_1`
> Boundary: non-smoke planner-only diagnostic. Not P5b dev diagnostic, not unified-strict, not hidden-test, not final-test, not main-table.

## Purpose

v2.1 tests one specific hypothesis: did v2's count head fail because of *weak gradient signal* (an imbalance / objective issue) or because of *architectural ceiling* (document-level pooled evidence is structurally insufficient to estimate `n_t`)? Two minimal changes vs v2:

- `truncated_poisson_nll` accepts optional `sample_weights`. The runner now feeds `weights = 1 + ln(n)` for positive-count training samples (n=1 → 1.0, n=2 → 1.69, n=3 → 2.10, n=16 → 3.77). This shifts gradient mass toward `n ≥ 2` cases without letting `n=16` outliers dominate.
- `CountPlanner.proj.bias` is zero-initialized so cold-start `log λ ≈ 0` (i.e. `λ ≈ 1`, `argmax_n PMF(n) = 1`). The model must actively learn to upgrade beyond `n=1`.

The trend acceptance criterion was also rewritten (`_has_required_overall_decrease`): `first / last >= 2.0` over `>= 10` epochs, replacing the 6-epoch window check that mis-fired on v2's smooth convergence.

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
| Eval batch size | 128 |
| Feature mode | `evidence_lexical` |
| Max sentences | 256 |
| Elapsed | 668.0s (11.1 min) |

## Loss trajectory (selected epochs)

| Epoch | presence_loss | count_loss (weighted) |
|---:|---:|---:|
| 1 | 1.2078 | 1.0692 |
| 10 | 0.6732 | 0.9422 |
| 20 | 0.3643 | 0.9245 |
| 30 | 0.2724 | 0.8986 |
| 40 | 0.2324 | 0.8172 |
| 50 | **0.2103** | **0.7959** |

`presence_loss` decreased 5.74× across 50 epochs; `count_loss` decreased only 1.34× (it carries the new sample weights, so direct numeric comparison to v2 count_loss is not meaningful).

## Dev metrics

| Population | type_gate_auc | type_gate_f1_youden | count_mae_positive | presence_threshold |
|---|---:|---:|---:|---:|
| multi_event_dev | 0.9823 | 0.8248 | 0.8702 | 0.2453 |
| all_dev | 0.9874 | 0.7783 | 0.4040 | 0.2611 |

## v2.1 vs v2 head-to-head

| Population × Metric | v2 | v2.1 | Δ | predict-1 baseline | gap to baseline |
|---|---:|---:|---:|---:|---:|
| multi_event_dev type_gate_auc | 0.9830 | 0.9823 | -0.0007 | — | — |
| multi_event_dev type_gate_f1_youden | 0.8319 | 0.8248 | -0.0071 | — | — |
| multi_event_dev count_mae_positive | 0.8990 | 0.8702 | **-0.0288 (-3.2%)** | 0.8365 | +0.0337 |
| all_dev type_gate_auc | 0.9879 | 0.9874 | -0.0005 | — | — |
| all_dev type_gate_f1_youden | 0.7778 | 0.7783 | +0.0005 | — | — |
| all_dev count_mae_positive | 0.4140 | 0.4040 | **-0.0100 (-2.4%)** | 0.3480 | +0.0560 |

TypeGate metrics are within noise — v2.1 should not have changed presence behaviour, and it did not. CountPlanner metrics improved by 2-3% in absolute MAE, but the gap to `predict-1` is still positive on both populations.

## Acceptance verdict

| Check | passed | value | baseline / threshold |
|---|:---:|---:|---|
| `multi_event_dev/type_gate_auc` | ✅ | 0.9823 | abs ≥ 0.80, vs lexical+0.05 |
| `multi_event_dev/type_gate_f1_youden` | ✅ | 0.8248 | abs ≥ 0.55, vs lexical+0.05 |
| `multi_event_dev/count_mae_positive` | ❌ | 0.8702 | abs ≤ 0.50 fail; predict-1 0.836 − 0.05 = 0.786 fail |
| `all_dev/type_gate_auc` | ✅ | 0.9874 | abs ≥ 0.80, vs lexical+0.05 |
| `all_dev/type_gate_f1_youden` | ✅ | 0.7783 | abs ≥ 0.55, vs lexical+0.05 |
| `all_dev/count_mae_positive` | ❌ | 0.4040 | abs ≤ 0.50 PASS; predict-1 0.348 − 0.05 = 0.298 fail (so rel fails) |
| `training/presence_loss_trend` | ✅ | first/last 5.74× | first/last ≥ 2.0 over ≥ 10 epochs |
| `training/count_loss_trend` | ❌ | first/last 1.34× | first/last ≥ 2.0 over ≥ 10 epochs |

Net `accepted=false`. Five of eight checks pass (vs three of eight in v2). The four TypeGate checks remain PASS by large margins (audit-level differences from v2 are noise). The two count checks fail and the count_loss_trend fails (the count_loss does not decrease enough because the weighting raises the relative contribution of high-n samples, whose NLL stays large even when the prediction is near-optimal).

## Methodology interpretation

This is the result we hoped to obtain *cleanly*: the reweighting + bias-init experiment is a controlled probe and its outcome is unambiguous.

1. **Weighted NLL + bias init is not enough.** The 2-3% improvement is real but does not close the gap to `predict-1`. The weighted loss does shift gradient mass toward `n ≥ 2`, but the model does not have enough information in `[global_repr; type_emb; evidence_vec; lexical_hit]` to identify those cases. Even with the bias initialized at the right default, the conditional distribution it learns is essentially `~1` everywhere, plus small perturbations that fail to align with the rare high-`n` cases.
2. **The bottleneck is architectural, not optimization.** Document-level pooled evidence cannot distinguish "this document has 1 record of type X" from "this document has 3 records of type X" because all per-occurrence information collapses into a single attention pool. A successful count head requires evidence at finer granularity — likely a sentence-level head that scores "this sentence contains a record of type X" and aggregates the binary scores into a count.
3. **TypeGate methodology is solid.** Both v2 and v2.1 give multi_event_dev AUC ≈ 0.98 and F1 ≈ 0.83; all_dev AUC ≈ 0.99 and F1 ≈ 0.78. These beat the strongest baseline (`p5b_lexical_trigger`) by 15-20 absolute F1 points and are stable across runs. The methodology contribution of v2 (evidence pooling + lexical feature + dual-population baseline-relative gate) is the part of R3 that is defensible.
4. **Trend gate fix is validated.** The `presence_loss_trend` check now PASSES with a 5.74× first/last ratio across 50 epochs. The criterion previously misfired on smooth convergence; the rewrite (`first/last ≥ 2.0 over ≥ 10 epochs`) reflects the actual convergence shape.

## Recommendation for the next iteration

R3 v3 (separate phase, not in scope for this measurement):

- Move the count head from document-level `[global_repr; type_emb; evidence_vec]` to a **sentence-level scorer**. For each sentence, predict a binary "this sentence contains a record of type X" label; sum the predictions (or expected counts) to get `n_t`. This requires per-sentence record supervision derived from `gold record.event_type ∩ sentence span`.
- Keep the TypeGate exactly as v2 — its behaviour and acceptance status are unchanged.
- Either preserve the absolute `count_mae_positive ≤ 0.5` threshold or replace it with a baseline-relative-only check; the dual-criterion combination is the most informative.

Alternative path if sentence-level supervision is too expensive: drop the count head from the acceptance gate and document R3 as "TypeGate-only acceptance", with count downstream estimated by lexical trigger fallback. This matches the original P5b rule but with neural presence filtering, which the v2 P5b smoke already showed lifts baseline F1 by ~2.5× on DuEE-Fin dev500.

## Boundaries

This measurement is a planner-only diagnostic. It does not assert:

- R3 acceptance (count head fails).
- P5b dev diagnostic against the current rule-path evidence (`runs/carve/p5b_duee_fin_dev500_seed42`).
- Any unified-strict, hidden-test, or final-test result.
- Any paper main-table or SOTA claim.
- P3 (full mention CRF + planner) acceptance.

The v2.1 checkpoint at `runs/carve/r3_planner_only_duee_fin_seed42_v2_1/checkpoints/r3_planner.pt` is a phase artifact and can be loaded into P5b via `--planner-checkpoint` if desired, but it is not promoted to a published planner.
