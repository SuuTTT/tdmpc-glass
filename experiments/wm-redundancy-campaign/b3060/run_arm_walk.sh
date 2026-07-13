#!/usr/bin/env bash
# WM-head-ablation single-arm runner, HopperHop variant (task-suffixed tags so it
# cannot collide with the CheetahRun ablation files). Env: ABLATE, SEED, GPU, TOTAL_STEPS
set -u
REPO=/root/helios_wmablate
cd "$REPO" || exit 1
PY=/root/helios-rl/.venv/bin/python
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}
export MUJOCO_GL=egl
export MJPG_IMPL=jax
export CUDA_VISIBLE_DEVICES=${GPU:?}
export ABLATE=${ABLATE:-none}
TASK=WalkerRun
SEED=${SEED:-1}
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
TAG=wmablwalk_${ABLATE}_s${SEED}
export TDMPC_GLASS_OUTPUT_TAG=$TAG
OUT=$REPO/exp/wm_head_ablation_walk
mkdir -p "$OUT/logs" "$OUT/jsonl"
export A2_JSONL=$OUT/jsonl/${TAG}.jsonl
LOG=$OUT/logs/${TAG}.log
echo "[run_arm_walk] START ablate=$ABLATE seed=$SEED gpu=$GPU steps=$TOTAL_STEPS task=$TASK tag=$TAG $(date -u +%FT%TZ)" | tee -a "$LOG"
$PY -u scripts/run_benchmark.py --algos tdmpc2 --tasks "$TASK" \
  --total_steps "$TOTAL_STEPS" --seed "$SEED" \
  --k_update 128 --mppi_n_samples 2048 --mppi_horizon 3 --expl_until 25000 \
  --no_plot >> "$LOG" 2>&1
rc=$?
echo "[run_arm_walk] DONE ablate=$ABLATE seed=$SEED rc=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $rc
