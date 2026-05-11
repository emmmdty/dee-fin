---
description: Convert a DEE methodology proposal into executable experiment, baseline, ablation, diagnostic, and acceptance matrices.
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
  - WebFetch
---

# DEE Experiment Plan Skill

Use this skill when converting a method proposal into a reproducible experiment plan.

Required outputs:
1. Dataset-to-evaluator mapping.
2. Must-run, recommended, and reported-only baselines.
3. Main metrics and supplementary metrics.
4. Ablation matrix with expected movement and falsification criteria.
5. Diagnostics:
   - long-document bucket
   - single vs multi-event
   - per-event F1
   - per-role F1
   - record grouping
   - unsupported argument rate
   - evidence grounding rate
   - split/merge error
6. Minimum viable submission package.
7. Phase-by-phase implementation roadmap with PASS/FAIL acceptance checks.

Never allow LLM judge, embedding semantic matching, or soft matching into the main leaderboard.
