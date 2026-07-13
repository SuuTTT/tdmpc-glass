#!/usr/bin/env bash
set -u
REPO=/root/tdmpc_glass/helios-rl
cd /root/tdmpc_glass || exit 1
source /root/tdmpc_glass/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/tdmpc_glass/mujoco_playground_repo
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl MJPG_IMPL=jax ABLATE=none VAC_LAM=0.0 URC_LAM=0.0
export CUDA_VISIBLE_DEVICES=${GPU:?}
export VBN_DIM=${VBN_DIM:?}
TASK=${TASK:?}; SEED=${SEED:?}; TOTAL_STEPS=${TOTAL_STEPS:-5000000}
TAG=vbn${VBN_DIM}_${TASK}_s${SEED}
export TDMPC_GLASS_OUTPUT_TAG=$TAG
OUT=/root/tdmpc_glass/exp/vac; mkdir -p "$OUT/logs"
LOG=$OUT/logs/${TAG}.log
echo "[vbn] START VBN_DIM=$VBN_DIM task=$TASK seed=$SEED gpu=$GPU $(date -u +%FT%TZ)" | tee -a "$LOG"
python -u $REPO/scripts/run_benchmark.py --algos tdmpc2 --tasks "$TASK" --total_steps "$TOTAL_STEPS" --seed "$SEED" --k_update 128 --mppi_n_samples 2048 --mppi_horizon 3 --expl_until 25000 --no_plot >> "$LOG" 2>&1
echo "[vbn] DONE VBN_DIM=$VBN_DIM task=$TASK seed=$SEED rc=$? $(date -u +%FT%TZ)" | tee -a "$LOG"
