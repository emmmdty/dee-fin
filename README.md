# dee-fin

`dee-fin` is a reproducible Chinese financial document-level event extraction workspace. The repository currently contains frozen dataset splits, evaluator tracks, baseline wrappers, and method planning documents. It does not yet contain CARVE model components or CARVE training/inference code.

## Method Status

The current method proposal is CARVE v1.3:

- Method document: `docs/method/carve_method_design_v1_3.md`
- Status: method proposal, not implemented
- Repository evidence: data splits and evaluator tracks only
- Schema assumption: event types, roles, and arguments only by default
- Deprecated proposal: `docs/method/easv_v1.md` was intentionally removed

CARVE frames Chinese financial DEE around cross-record argument allocation and role-grounded verification. Its future implementation path is staged, with P5b as the decision gate for whether the allocation mechanism is strong enough for the intended paper framing.

## Data

The project uses frozen local data under `data/`. Do not delete or modify this directory unless a task explicitly authorizes a data phase.

The processed datasets are:

- ChFinAnn-Doc2EDAG
- DuEE-Fin-dev500
- DocFEE-dev1000

`baseline/` contains official GitHub repositories for comparison methods and is not part of normal implementation scope.

## Evaluators

The evaluator package provides strict surface-form metrics only. It does not use alias expansion, semantic matching, embedding similarity, LLM judging, event-type guessing, role guessing, or gold-based repair.

Evaluator tracks:

- ChFinAnn: `legacy-doc2edag` + `unified-strict`
- DuEE-Fin: historical-compatible / Doc2EDAG-style if supported + `unified-strict`
- DocFEE: `docfee-official` + `unified-strict`

Example commands:

```bash
/home/tjk/miniconda3/envs/feg-dev-py310/bin/python -m evaluator --help
/home/tjk/miniconda3/envs/feg-dev-py310/bin/python -m evaluator inspect-gold --dataset DuEE-Fin-dev500 --gold data/processed/DuEE-Fin-dev500/dev.jsonl --schema data/processed/DuEE-Fin-dev500/schema.json
/home/tjk/miniconda3/envs/feg-dev-py310/bin/python -m evaluator legacy-doc2edag --input-format canonical-jsonl --dataset ChFinAnn-Doc2EDAG --gold data/processed/ChFinAnn-Doc2EDAG/test.json --pred predictions.jsonl --schema data/processed/ChFinAnn-Doc2EDAG/schema.json --out report.json
/home/tjk/miniconda3/envs/feg-dev-py310/bin/python -m evaluator docfee-official --dataset DocFEE-dev1000 --gold data/processed/DocFEE-dev1000/test.jsonl --pred predictions.jsonl --out report.json
/home/tjk/miniconda3/envs/feg-dev-py310/bin/python -m evaluator unified-strict --dataset DuEE-Fin-dev500 --gold data/processed/DuEE-Fin-dev500/test.jsonl --pred predictions.jsonl --schema data/processed/DuEE-Fin-dev500/schema.json --out report.json
```

See `evaluator/README.md` for metric definitions, input contracts, and track-specific limitations.

## Experiment Design

Phase documents live under `docs/phase/`.

Current staged route:

```text
P0 -> P1 -> P4 -> P5a -> P5b
```

- P0 freezes the weak-evidence and allocation-target design before code.
- P1 measures encoder and module memory feasibility before making GPU claims.
- P4 validates allocation-target construction, Sinkhorn behavior, share gating, and oracle-injection boundaries on toy cases.
- P5a validates conflict-aware EDAG behavior on a controlled misallocation toy case.
- P5b measures dev multi-event diagnostic support and records the paper-route decision profile.

Do not run full training, full dev, test, Qwen inference, or long remote jobs without an explicit phase request.

## Local Validation

Use the project Python unless a task explicitly chooses another environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/evaluator -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.data_split.test_split_utils -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/baseline/procnet -v
```

After each accepted phase, commit the phase and verify:

```bash
git status --short
```
