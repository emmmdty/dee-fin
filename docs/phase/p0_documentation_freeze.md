# P0 Documentation Freeze

## Purpose

P0 freezes the CARVE v1.3 research contract before model implementation. It converts the method proposal into repository-operational guidance without claiming that CARVE code, measurements, training, or inference already exist.

## Scope

P0 is docs-only:

- Update root `AGENTS.md` with concise Codex instructions, repository boundaries, environments, and phase gates.
- Fill `README.md` with method status, data boundary, evaluator usage, and staged experiment route.
- Fix method-document index facts after the intentional deletion of `docs/method/easv_v1.md`.
- Record the phase plan under `docs/phase/`.

P0 must not:

- Modify `data/` or `baseline/`.
- Implement CARVE model modules.
- Run training, inference, full dev, test, or Qwen experiments.
- Convert measurement templates into evidence files.

## Frozen Method Points

Future implementation phases must preserve these CARVE v1.3 decisions:

- Weak evidence labels are deterministic projections from gold records, not human evidence annotations.
- `PosSent(r, rho)` includes all sentences containing the normalized role value.
- Gold-record column ordering for allocation targets must be deterministic.
- Multi-positive `Y_{t,rho}` must allow one mention to map to multiple positive record columns.
- Oracle injection is training-only and must be forced off at inference.
- Stage B hard-negative mining must use train predictions only and must not use oracle injection.
- `misallocated_rate_eligible` is the primary misallocation diagnostic denominator.
- Qwen-4B verifier is appendix-only if later authorized; it is not the headline method.

## Acceptance Criteria

P0 is accepted only if:

- `README.md`, `AGENTS.md`, `docs/method/README.md`, and `docs/phase/` all state that CARVE is not implemented yet.
- No document says `docs/method/easv_v1.md` is present.
- No new code, model, data, or baseline file is introduced.
- Existing local unit checks pass.
- The phase is committed and `git status --short` is empty.

## Closeout

Status: completed / accepted.

P0 was a documentation-freeze phase. Its accepted scope was to freeze the CARVE v1.3 repository contract and method-document boundaries without modifying `data/`, `baseline/`, model artifacts, or run artifacts.

Closeout evidence:

- Phase document: `docs/phase/p0_documentation_freeze.md`.
- Method proposal: `docs/method/carve_method_design_v1_3.md`.
- Method index: `docs/method/README.md`.
- Phase index: `docs/phase/README.md`.

Boundary note: later diagnostic CARVE code and P1 measurement evidence are outside P0. They do not change the fact that P0 itself was accepted as a docs-only freeze.

## Validation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/evaluator -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.data_split.test_split_utils -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/baseline/procnet -v
rg -n "[e]asv_v1\\.md\\s*\\|\\s*Present|Keep for design history [o]nly|docs/method/[e]asv_v1\\.md$" README.md AGENTS.md docs/method docs/phase
git status --short
```
