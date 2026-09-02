#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export DP_ADAM_IID_REPOSITORY_ROOT="$PROJECT_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to activate the curve environment" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate curve

exec python scripts/train.py "$@"
