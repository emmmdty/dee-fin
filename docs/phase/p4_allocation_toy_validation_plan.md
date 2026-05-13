# P4 Allocation Toy Validation Plan

## Purpose

P4 validates the CARVE allocation mechanism on controlled examples before using real dev data.

## Components

P4 covers:

- deterministic gold-record column sorting;
- construction of multi-positive `Y_{t,rho}`;
- NULL-column assignment;
- training-only oracle-injection flag behavior;
- inference-time forced `oracle_inject=False`;
- Sinkhorn prior toy behavior;
- `L_alloc` multi-positive marginal loss;
- optional positive-coverage regularizer;
- learned share-gate label construction.

## Acceptance Criteria

P4 is accepted only if toy tests cover:

- repeated surface value shared by two records of the same event type;
- non-shared role value assigned to a single positive column;
- candidate mention with no matching gold role value assigned to NULL;
- deterministic output under a fixed seed and repeated execution;
- oracle-injected candidate present in training construction but absent in inference construction;
- `mu=0` disables positive coverage while preserving marginal allocation loss;
- share-gate labels mark surface-level sharing without coreference or entity-linking assumptions.

## Acceptance Result

Status: completed / accepted as toy behavior only.

Acceptance date: 2026-05-13.

P4 was hardened with local toy tests and a JSON toy-summary runner. The accepted output is not dev-set scoring, test-set scoring, model training, or a paper main-table result.

Evidence commands:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.carve.test_allocation -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B scripts/carve/run_p4_p5a_toy_validation.py --out /tmp/dee-fin-p4-p5a-toy-validation.json
```

Observed P4 toy-summary status:

- `status`: `toy_behavior_only`
- `accepted`: `true`
- `record_order`: `["r1", "r2"]`
- shared surface target row: `[1.0, 1.0, 0.0]`
- single-positive target row: `[1.0, 0.0, 0.0]`
- NULL target row: `[0.0, 0.0, 1.0]`
- oracle-injected training values: `["丙银行", "乙银行"]`
- inference candidates for the same missing role: `[]`
- `mu=0` loss equals marginal-only loss; positive coverage increases the toy loss when enabled.

Accepted P4 checks:

- repeated surface value shared by two records of the same event type: pass;
- non-shared role value assigned to a single positive column: pass;
- candidate mention with no matching gold role value assigned to NULL: pass;
- deterministic output under repeated execution: pass;
- oracle-injected candidate present in training construction but absent in inference construction: pass;
- `mu=0` disables positive coverage while preserving marginal allocation loss: pass;
- share-gate labels mark surface-level sharing only: pass.

## Non-Goals

- No full encoder training.
- No full EDAG decoder.
- No dev or test scoring.
- No schema-derived hard exclusions unless separately specified as experimental assumptions.
