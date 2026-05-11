#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."
REMOTE="gpu-4090:/data/TJK/DEE/dee-fin"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
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

args=(-avz --exclude "__pycache__/" --exclude "*.pyc")
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

rsync "${args[@]}" "$PROJECT_ROOT/scripts/baseline/procnet/" "$REMOTE/scripts/baseline/procnet/"
rsync "${args[@]}" "$PROJECT_ROOT/tests/baseline/procnet/" "$REMOTE/tests/baseline/procnet/"
