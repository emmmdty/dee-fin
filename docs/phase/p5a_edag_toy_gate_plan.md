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

## Non-Goals

- No dev-set scoring.
- No model training.
- No verifier pruning.
- No paper main-table claim.
