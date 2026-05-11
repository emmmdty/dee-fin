---
description: Audit and improve Chinese financial document-level event extraction methodology documents against repository facts, evaluator constraints, baselines, and EMNLP reviewer expectations.
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
  - WebFetch
---

# DEE Method Audit Skill

Use this skill when auditing or revising methodology documents under docs/method/.

Process:
1. Read AGENTS.md, CLAUDE.md, docs/method/*.md, evaluator docs, scripts, tests, and schema/data layout files.
2. Treat repository files as source of truth.
3. Mark every unverified claim.
4. Separate method contribution from evaluator/reproducibility contribution.
5. Preserve three evaluator tracks:
   - Doc2EDAG-style evaluator
   - DocFEE official-style evaluator
   - Unified Strict evaluator
6. Check whether the proposed method is feasible under 1–2 RTX 4090 GPUs.
7. Check whether Qwen-4B is main method, verifier/reranker, weak-label generator, or baseline only.
8. Output concrete acceptance criteria.

Never edit ./data or ./baseline.
Never tune on test data.
Never claim SOTA without fair baseline and metric alignment.
