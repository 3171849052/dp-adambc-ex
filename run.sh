#!/usr/bin/env bash

# This launcher uses Bash features below, but users commonly invoke it as
# `sh run.sh`.  In that form the shebang is ignored and /bin/sh would abort
# on Bash-only syntax before training can start.
# Unified launcher for qnli_roberta_base* and qnli_bert_base* configurations.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DP_ADAMBC_EX_REPOSITORY_ROOT="$ROOT"
DEFAULT_CONFIG="$ROOT/config/qnli_roberta_base_dpadambc.yaml"

if [[ $# -eq 0 ]]; then
  CONFIG="$DEFAULT_CONFIG"
elif [[ $# -eq 2 && $1 == "--config" ]]; then
  CONFIG="$2"
elif [[ $# -eq 1 && $1 != "--config" ]]; then
  CONFIG="$1"
else
  echo "usage: $0 [--config CONFIG]" >&2
  echo "       $0 [CONFIG]" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "config file does not exist: $CONFIG" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
else
  echo "conda is required to activate the adamex environment" >&2
  exit 1
fi
if command -v tmux >/dev/null 2>&1; then
  HAS_TMUX=1
else
  HAS_TMUX=0
  echo "tmux is unavailable; running training in the foreground" >&2
fi

# Some site-wide Conda activation hooks reference optional shell variables.
# Suspend nounset only while Conda evaluates those hooks.
set +u
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate adamex
set -u

CONFIG="$(realpath "$CONFIG")"
GPU="$(cd "$ROOT" && env -u CUDA_VISIBLE_DEVICES python -u scripts/train.py --config "$CONFIG" --print-gpu)"
if ! (cd "$ROOT" && env -u CUDA_VISIBLE_DEVICES python -u scripts/train.py --config "$CONFIG" --validate-gpu >/dev/null); then
  echo "GPU validation failed; refusing to start tmux training" >&2
  exit 1
fi

RUN_DIR="$(cd "$ROOT" && CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/train.py --config "$CONFIG" --prepare-run)"
RUN_DIR="$(realpath "$RUN_DIR")"
TRAIN_LOG="$RUN_DIR/train.log"

if [[ "$HAS_TMUX" -eq 0 ]]; then
  echo "physical GPU: $GPU"
  echo "run directory: $RUN_DIR"
  echo "log: $TRAIN_LOG"
  export TOKENIZERS_PARALLELISM=false
  export RAYON_NUM_THREADS=1
  export PYTHONFAULTHANDLER=1
  set -o pipefail
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/train.py \
    --config "$CONFIG" --run-dir "$RUN_DIR" 2>&1 | tee -a "$TRAIN_LOG"
  exit $?
fi

SESSION="$(cd "$ROOT" && python -u scripts/train.py --tmux-session-name "$RUN_DIR")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

printf -v COMMAND \
  'cd %q && eval "$(%q shell.bash hook)" && conda activate adamex && export TOKENIZERS_PARALLELISM=false && export RAYON_NUM_THREADS=1 && export PYTHONFAULTHANDLER=1 && GPU=%q && set -o pipefail && CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/train.py --config %q --run-dir %q 2>&1 | tee -a %q' \
  "$ROOT" "$CONDA_BIN" "$GPU" "$CONFIG" "$RUN_DIR" "$TRAIN_LOG"
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "physical GPU: $GPU"
echo "tmux session: $SESSION"
echo "run directory: $RUN_DIR"
echo "log: $TRAIN_LOG"
echo "attach: tmux attach -t $SESSION"
echo "tail: tail -f $TRAIN_LOG"
echo "kill: tmux kill-session -t $SESSION"
