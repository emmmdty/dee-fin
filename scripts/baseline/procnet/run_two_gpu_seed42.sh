#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."
PYTHON_BIN="python"
GPU_CHFINANN="0"
GPU_DUEE="1"
MAX_EPOCHS="100"
PATIENCE="8"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --gpu-chfinann)
      GPU_CHFINANN="$2"
      shift 2
      ;;
    --gpu-duee)
      GPU_DUEE="$2"
      shift 2
      ;;
    --max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cmd_chfinann=(
  "$PYTHON_BIN" scripts/baseline/procnet/run_procnet_repro.py
  --project-root "$PROJECT_ROOT"
  --dataset ChFinAnn-Doc2EDAG
  --experiment-name chfinann_doc2edag_procnet_seed42
  --seed 42
  --max-epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --gpu "$GPU_CHFINANN"
  --python-bin "$PYTHON_BIN"
)

cmd_duee=(
  "$PYTHON_BIN" scripts/baseline/procnet/run_procnet_repro.py
  --project-root "$PROJECT_ROOT"
  --dataset DuEE-Fin-dev500
  --experiment-name duee_fin_dev500_procnet_seed42
  --seed 42
  --max-epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --gpu "$GPU_DUEE"
  --python-bin "$PYTHON_BIN"
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: two separate single-GPU ProcNet commands"
  printf 'CUDA_VISIBLE_DEVICES=%s ' "$GPU_CHFINANN"
  printf '%q ' "${cmd_chfinann[@]}"
  printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%s ' "$GPU_DUEE"
  printf '%q ' "${cmd_duee[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_CHFINANN" "${cmd_chfinann[@]}" &
pid_chfinann="$!"
CUDA_VISIBLE_DEVICES="$GPU_DUEE" "${cmd_duee[@]}" &
pid_duee="$!"

wait "$pid_chfinann"
wait "$pid_duee"
