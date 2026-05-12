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

## Non-Goals

- No full encoder training.
- No full EDAG decoder.
- No dev or test scoring.
- No schema-derived hard exclusions unless separately specified as experimental assumptions.
