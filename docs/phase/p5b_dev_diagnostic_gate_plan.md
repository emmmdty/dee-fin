# P5b Dev Diagnostic Gate Plan

## Purpose

P5b is the first CARVE mechanism viability gate on real dev multi-event subsets. It decides whether the allocation mechanism supports the intended paper route.

## Comparison

For each dataset's dev multi-event subset, compare:

- baseline route without Sinkhorn allocation prior, `L_alloc`, and share gate;
- CARVE route with Sinkhorn allocation prior, `L_alloc`, and share gate.

Keep all other factors fixed: split, seed, encoder, candidate construction, decoder route, evaluator track, and normalization policy.

## Metrics

Record:

- `misallocated_rate_eligible`;
- `misallocated_rate_total`;
- `ambiguous_excluded_rate`;
- record F1 under the applicable evaluator family;
- inference candidate recall;
- failed-run diagnostics, if any.

## Acceptance Criteria

P5b is accepted only if:

- measured results are written to `docs/measurements/p5b_decision_table.md`;
- each dataset row is labeled Strong, Weak, or No support using CARVE v1.3 thresholds;
- the paper-route decision follows the CARVE v1.3 venue decision profile;
- dev diagnostics are not promoted into final test results.

## Non-Goals

- No hidden-test or final-test claim.
- No SOTA claim.
- No Qwen verifier claim unless a separate appendix phase was authorized and measured.
- No fallback to EASV/ECPD-CRV without recording the P5b failure reason.
