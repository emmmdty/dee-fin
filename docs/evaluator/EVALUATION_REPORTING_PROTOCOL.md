# Evaluation Reporting Protocol

This document defines the paper-facing metric protocol for reporting financial
document-level event extraction results in this project. The purpose is to keep
historical native replay, exported fixed-slot evaluation, unified canonical
role-value evaluation, and DocFEE official-style evaluation clearly separated.

The protocol is reporting-only. It does not change evaluator scoring logic,
dataset splits, model outputs, training, inference, or data conversion.

## Source Position

This protocol consolidates the evaluator design already documented in:

- `docs/evaluator/MULTI_VALUE_ROLE_AUDIT.md`
- `docs/evaluator/EVALUATION_PROTOCOL_RATIONALE.md`
- `evaluator/README.md`

The relevant conclusion is that fixed-slot native metrics and canonical
role-value metrics answer different scientific questions. Fixed-slot scores
remain useful for historical reproduction, while `unified-strict` is the main
canonical metric for cross-dataset comparison.

## Reporting Layers

### Layer A: Native Replay / Historical Official-Style Scores

Purpose:

- Reproduce historical baseline scores.
- Use native event-table artifacts where available.
- Support ProcNet / Doc2EDAG-style historical comparison.

Evaluator:

```bash
python -m evaluator legacy-doc2edag --input-format native-event-table
```

Input:

- Native event-table matrices.
- Gold/prediction artifacts before canonical JSONL export.

Use for:

- ProcNet native training-report F1.
- Doc2EDAG/ProcNet-style historical comparison.
- Appendix or historical reproduction tables.

Do not use for:

- Cross-dataset unified comparison.
- DocFEE official comparison.
- LLM canonical output main metric.
- Claiming canonical role-value extraction quality.

Reporting rule:

Native replay scores belong in a historical native replay table or a clearly
marked native replay column. They must not be merged into the same metric column
as exported canonical or `unified-strict` scores.

### Layer B: Exported Legacy Fixed-Slot Scores

Purpose:

- Evaluate exported canonical JSONL predictions under native-style fixed-slot
  counting.
- Diagnose export-level compatibility.
- Measure how much a score changes after native outputs are exported to
  canonical JSONL.

Evaluator:

```bash
python -m evaluator legacy-doc2edag --input-format canonical-jsonl
```

Input:

- Canonical JSONL prediction files.
- Processed gold files.
- Dataset schema.

Use for:

- Exported prediction comparison.
- Canonical export drift analysis.
- Checking score changes after export.

Important warning:

- This is not bit-exact ProcNet native replay.
- Do not mix this score with native training-report scores in the same
  leaderboard column.
- This metric still uses fixed-slot semantics; it does not preserve every
  canonical multi-value role-value unit.

Reporting rule:

Exported legacy fixed-slot scores may be shown next to native replay and
`unified-strict` scores, but the columns must be separately named. The caption
must say that exported legacy scores evaluate canonical JSONL under fixed-slot
semantics.

### Layer C: Unified Strict Role-Value Scores

Purpose:

- Provide cross-dataset scientific comparison.
- Preserve multi-value roles.
- Evaluate ChFinAnn, DuEE-Fin, and DocFEE under one strict canonical metric.

Evaluator:

```bash
python -m evaluator unified-strict
```

Input:

- Canonical JSONL prediction files.
- Processed gold files.
- Dataset schema when applicable.

Use for:

- Main cross-dataset metric.
- Comparing this project's model against exported baselines.
- Multi-value role-sensitive evaluation.
- ChFinAnn, DuEE-Fin, and DocFEE canonical role-value comparison.

Do not use for:

- Claiming exact reproduction of ProcNet/Doc2EDAG native scores.
- Replacing DocFEE official-style reporting.
- Historical fixed-slot leaderboard reproduction.

Reporting rule:

`unified-strict` is the main cross-dataset canonical role-value metric. It may
be used in the main scientific comparison table, but it must not be described as
a native replay metric.

### Layer D: DocFEE Official-Style Scores

Purpose:

- Align with DocFEE official baseline/evaluator style.
- Preserve compatibility with DocFEE-specific reporting expectations.

Evaluator:

```bash
python -m evaluator docfee-official
```

Use for:

- DocFEE baseline comparison.
- Official-style DocFEE reporting.
- Side-by-side DocFEE compatibility analysis with `unified-strict`.

Do not use for:

- ChFinAnn or DuEE-Fin cross-dataset main metric reporting.
- Replacing `unified-strict` in the cross-dataset canonical table.
- ProcNet/Doc2EDAG native replay.

Reporting rule:

DocFEE official-style scores are reported separately from DocFEE
`unified-strict` scores. If both are shown, they must use distinct columns.

## Table Policy

### Table 1: Historical Native Replay Table

Purpose:

- Report historical native replay or paper-reported official-style scores.

Columns:

```text
Model
Dataset
Split
Evaluator
Input Artifact
P
R
F1
Source
Notes
```

Rows may include:

- ProcNet s42/s43/s44 with native-event-table replay.
- Doc2EDAG / GIT / DE-PPN / SEELE reported scores if only paper scores are
  available.
- This project's method only if it has a comparable native artifact.

Do not include:

- `unified-strict` scores.
- Exported canonical JSONL fixed-slot scores unless they are placed in a
  separately named column or table.
- DocFEE official-style scores unless the table is explicitly scoped to DocFEE.

### Table 2: Exported Canonical Evaluation Table

Purpose:

- Compare exported canonical JSONL predictions under fixed-slot and
  `unified-strict` metrics.

Columns:

```text
Model
Dataset
Split
legacy-doc2edag canonical-jsonl F1
unified-strict F1
Input prediction file
Notes
```

Rows:

- ProcNet exported canonical predictions.
- This project's exported predictions.
- LLM/Qwen exported predictions if available.

Rule:

The `legacy-doc2edag canonical-jsonl F1` column is an exported fixed-slot score,
not a native training-report replay score. It must not be merged with Table 1
native replay F1.

### Table 3: Unified Strict Main Table

Purpose:

- Provide the main cross-dataset scientific comparison.

Columns:

```text
Model
ChFinAnn unified-strict F1
DuEE-Fin unified-strict F1
DocFEE unified-strict F1
Average or macro summary if justified
Notes
```

Rule:

This table is for `unified-strict` only. Native replay, exported legacy
fixed-slot, and DocFEE official-style scores should not appear in this table
unless they are clearly outside the main metric columns.

### Table 4: DocFEE Official-Style Table

Purpose:

- Report DocFEE official-style compatibility while preserving the canonical
  `unified-strict` view.

Columns:

```text
Model
DocFEE official-style P/R/F1
DocFEE unified-strict P/R/F1
Notes
```

Rule:

The DocFEE official-style score is a DocFEE compatibility score. The
`unified-strict` score is the canonical role-value score. They must be reported
as separate columns.

### Table 5: Export Drift Table

Purpose:

- Explain why ProcNet native training-report scores can differ from canonical
  exported scores.

Columns:

```text
Seed
Native replay F1
Exported legacy F1
Unified-strict F1
Native-to-exported gap
Exported-to-unified gap
Reason
```

Rule:

This table is diagnostic. It may show gaps across metric layers, but it must not
imply that the layers are interchangeable.

### Table 6: Multi-Value Role Audit Table

Purpose:

- Justify `unified-strict` by showing when fixed-slot units and canonical
  role-value units differ.

Columns:

```text
Dataset
Split
Documents
Event records
Fixed-slot units
Canonical role-value units
Extra role-value units
Multi-value role docs
Top affected roles
```

Rule:

This table supports the metric choice. It is not a model performance table.

## Required Wording Policy

The paper and reports must explicitly avoid the following misleading statements:

1. "unified-strict reproduces ProcNet native scores"
2. "legacy-doc2edag canonical-jsonl is bit-exact ProcNet native"
3. "native replay and exported canonical evaluation are the same metric column"
4. "fixed-slot metrics fully represent multi-value roles"
5. "LLM judge / embedding / semantic match is part of the main metric"

Recommended correct statements:

1. Native replay scores are used for historical baseline reproduction.
2. Exported legacy scores evaluate canonical JSONL predictions under fixed-slot semantics.
3. Unified-strict scores are used for cross-dataset canonical role-value comparison.
4. DocFEE official-style scores are reported separately for DocFEE compatibility.
5. Fixed-slot and unified-strict scores should be reported in separate columns.

## Main Metric Boundary

The main metric is strict and deterministic. It does not use:

- LLM judge.
- Embedding similarity.
- Semantic matching.
- Alias expansion.
- Amount/date reasoning.
- Event-type guessing.
- Role guessing.
- Gold-based repair.

Any exploratory semantic, embedding, or LLM-based analysis must be reported as a
separate auxiliary analysis and must not be described as part of the main metric.

## Paper-Safe Summary

Use native replay scores to discuss historical baseline reproduction. Use
exported legacy fixed-slot scores to diagnose canonical export compatibility.
Use `unified-strict` as the main cross-dataset canonical role-value metric. Use
DocFEE official-style scores only for DocFEE compatibility. These layers may be
shown side by side, but they must remain separate metric columns with captions
that explain what each layer measures.
