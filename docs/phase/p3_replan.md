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
