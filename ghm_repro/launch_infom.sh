#!/usr/bin/env bash
# GHM/InFOM launch helper (per-worker). Recreated on each GPU box at /root/ghm/launch_infom.sh
# Usage: ENV=<env_name> SEED=<n> NAME=<group> CUDA=<gpu_idx> MEMF=<frac> ./launch_infom.sh [extra flags...]
set -euo pipefail

GHM_ROOT=/root/ghm
VENV="${VENV:-$GHM_ROOT/.venv}"

# activate venv
source "$VENV/bin/activate"

export PYTHONPATH="$GHM_ROOT/infom"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA:-0}"
if [ -n "${MEMF:-}" ]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEMF"
fi

ENV="${ENV:?set ENV}"
SEED="${SEED:-0}"
NAME="${NAME:-ghm}"

cd "$GHM_ROOT/infom"
exec python main.py \
  --agent=agents/ghm.py \
  --env_name="$ENV" \
  --seed="$SEED" \
  --enable_wandb=0 \
  --save_dir="$GHM_ROOT/exp" \
  --wandb_run_group="$NAME" \
  "$@"
