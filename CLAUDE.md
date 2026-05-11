@AGENTS.md

## Claude Code project rules

This repository is for Chinese financial document-level event extraction research.

Hard constraints:
- Do not delete, rewrite, move, or regenerate ./data unless explicitly instructed.
- Do not delete, rewrite, move, or modify ./baseline unless explicitly instructed.
- For the current task, only inspect repository files and improve documentation under docs/method/.
- Treat docs/method/easv_v1.md as a draft report, not as ground truth.
- Verify every repository-specific claim against actual local files.
- Use the local Python interpreter: /home/tjk/miniconda3/envs/feg-dev-py310/bin/python.
- Never tune on test data.
- Keep Doc2EDAG-style, DocFEE official-style, and Unified Strict evaluator tracks conceptually separated.
