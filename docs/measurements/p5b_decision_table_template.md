# P5b Decision Table Template

> **Status**: template. Fill after P5b is executed.  
> **Purpose**: decide whether CARVE's core mechanism is viable by measuring misallocation reduction on dev multi-event subsets.

## Protocol Freeze

Compare:

```text
with Sinkhorn allocation prior + L_alloc + share gate
vs.
no-allocation-prior baseline
```

on the dev multi-event subset of each dataset.

No test data is used.

## Metrics

Primary:

```text
misallocated_rate_eligible
```

Secondary:

```text
record F1
inference candidate recall
misallocated_rate_total
ambiguous_excluded_rate
hallucinated_argument_rate
ungrounded_argument_rate
```

## Decision Table

| Dataset | Δ misallocated_rate_eligible | Δ record F1 | Inference candidate recall | Signal | Notes |
|---|---:|---:|---:|---|---|
| ChFinAnn | TBD | TBD | TBD | TBD | TBD |
| DuEE-Fin | TBD | TBD | TBD | TBD | TBD |
| DocFEE | TBD | TBD | TBD | TBD | TBD |

## Signal Rules

- **Strong**: `misallocated_rate_eligible` decreases by at least 1.0 absolute point and record F1 does not drop by more than 0.5 absolute.
- **Weak**: `misallocated_rate_eligible` decreases but by less than 1.0, or record F1 drops by 0.5–1.0 absolute.
- **No support**: `misallocated_rate_eligible` does not decrease or record F1 drops by more than 1.0 absolute.

## Venue Decision

| Signal Pattern | Decision |
|---|---|
| Three strong | EMNLP main stretch justified |
| Two strong + one weak | EMNLP main or Findings depending on final baseline strength |
| Three weak | Findings/COLING only if record F1 is stable and diagnostics are visually consistent; otherwise COLING / SCI Q2 |
| One strong + two weak, or any no-support | COLING / SCI Q2, or simplify back to ECPD-CRV |
| Two or more no-support | CARVE mechanism fails; do not force main-conference framing |

## Raw Commands

```bash
# paste commands here
```

## Diagnostic Plots / Tables

TBD

## Final Decision

TBD
