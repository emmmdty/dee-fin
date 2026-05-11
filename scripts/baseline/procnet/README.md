# ProcNet Reproduction Wrappers

Script-side wrappers for reproducing `baseline/procnet` without modifying the
baseline source tree.

Dry-run examples:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/baseline/procnet/run_procnet_repro.py \
  --project-root . \
  --dataset ChFinAnn-Doc2EDAG \
  --experiment-name dryrun_chfinann_seed42 \
  --seed 42 \
  --max-epochs 100 \
  --patience 8 \
  --gpu 0 \
  --dry-run

bash scripts/baseline/procnet/run_two_gpu_seed42.sh --dry-run
bash scripts/baseline/procnet/sync_to_server.sh --dry-run
```

Server defaults:

- project root: `/data/TJK/DEE/dee-fin`
- ProcNet Python: `/home/TJK/.conda/envs/procnet/bin/python`

The wrappers write experiment artifacts under
`runs/baseline/procnet/{experiment_name}/`, including native replay JSON,
run-local canonical gold/pred JSONL, evaluator reports, and summaries.
