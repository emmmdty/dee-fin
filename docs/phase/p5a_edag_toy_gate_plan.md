# P5a EDAG Toy Gate Plan

## Purpose

P5a checks whether conflict-aware EDAG scoring changes record-binding behavior on a controlled misallocation example.

## Toy Setup

Construct a small document/event-type case where:

- two records share at least one surface role value;
- one role value is grounded but can be bound to the wrong record;
- the same candidate set is evaluated with and without the allocation prior and share gate;
- oracle injection is disabled because P5a tests inference behavior.

## Acceptance Criteria

P5a is accepted only if:

- the no-allocation baseline and allocation-aware route are run on the same toy inputs;
- the expected misallocation case is explicit and deterministic;
- the allocation-aware route changes candidate preference in the intended direction;
- the result is reported as toy behavior only, not model performance.

## Acceptance Result

Status: completed / accepted as toy behavior only.

Acceptance date: 2026-05-13.

R5a hardened the P5a toy comparison so the no-allocation baseline and allocation-aware route share one deterministic toy input. The accepted output is not dev-set scoring, model training, verifier pruning, or a paper main-table result.

Evidence commands:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.carve.test_allocation -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B scripts/carve/run_p4_p5a_toy_validation.py --out /tmp/dee-fin-p4-p5a-toy-validation.json
```

Observed P5a toy-summary status:

- `status`: `toy_behavior_only`
- `accepted`: `true`
- `oracle_inject`: `false`
- expected misallocation: `质权方=丙银行` is grounded in `r1` but the baseline route binds it to `r2`
- no-allocation baseline choice: `wrong-record`
- allocation-aware route choice: `correct-record`
- baseline scores: `correct-record=0.65`, `wrong-record=0.72`
- allocation-aware scores: `correct-record=0.6028446602643793`, `wrong-record=-0.13739921404596334`
- allocation-aware margin: `0.7402438743103427`
- shared surface gate: `质押方=甲公司` has share label `true`
- disputed value gate: `质权方=丙银行` has share label `false`

Accepted P5a checks:

- same toy inputs for the no-allocation and allocation-aware routes: pass;
- expected misallocation case is explicit and deterministic: pass;
- allocation-aware route changes candidate preference in the intended direction: pass;
- result is reported as toy behavior only: pass;
- oracle injection is disabled for inference behavior: pass.

## Non-Goals

- No dev-set scoring.
- No model training.
- No verifier pruning.
- No paper main-table claim.
