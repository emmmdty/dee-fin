# CARVE Phase Documents

This directory records staged CARVE experiment design and acceptance criteria for `dee-fin`.

## Current Status

- Active proposal: CARVE v1.3 in `docs/method/carve_method_design_v1_3.md`
- Implementation status: P0/P1 closed; later diagnostic CARVE code remains dev-only and is not a full CARVE implementation report
- Evidence currently available in this repository: data splits, evaluator tracks, baseline wrappers, and P1 encoder memory measurement
- Deprecated proposal: `docs/method/easv_v1.md` was intentionally removed

## Phase Route

The minimal validation chain is:

```text
P0 -> P1 -> P4 -> P5a -> P5b
```

Later phases P2, P3, P6, P7, P8, and P9 should not be expanded into implementation work until the P5b diagnostic gate supports continuing CARVE.

## Phase Gate Rules

- Each phase must define acceptance criteria before implementation.
- Each phase must record whether it changed docs only, local code, server smoke artifacts, or measured experimental evidence.
- Templates in `docs/measurements/` are not evidence until a phase writes measured results into non-template files.
- After each accepted phase, run relevant validation, commit the phase, and verify `git status --short` is empty.
- Local development is the default. `gpu-4090` is allowed only for sync and smoke unless the user explicitly authorizes heavier work.

## Current Phase Index

| Phase | Document | Status |
|---|---|---|
| P0 | `p0_documentation_freeze.md` | Completed / accepted |
| P1 | `p1_memory_measurement_plan.md` | Completed / measured; evidence in `docs/measurements/p1_memory.md` |
| P4 | `p4_allocation_toy_validation_plan.md` | Completed / accepted as toy behavior only |
| P5a | `p5a_edag_toy_gate_plan.md` | Completed / accepted as toy behavior only |
| P5b | `p5b_dev_diagnostic_gate_plan.md` | Planned, not run |
| P5b DuEE-Fin first long run | `p5b_duee_fin_first_long_train.md` | Implementation in progress; dev diagnostic only |
