#!/usr/bin/env bash
# EXP#13: orientation-aware option controller + in-loop residual (alpha=1.0 fixed,
# the #5b-best authority), trained with brax PPO exactly like #5b. Plus a TOP-DOWN
# control+residual reproduce arm (fair baseline). One job/GPU, 45M env-steps.
#
# Orientation scheme (validated by orient_sweep + controller-alone sanity):
#   ORIENT_TILT_GAIN=2.0, ORIENT_TILT_R0=0.55, ORIENT_TILT_MAX=0.9,
#   ORIENT_RELEVEL_FROM=4 (TRANSPORT): tilt-grasp far box, relevel cube in carry.
# Top-down baseline = ORIENT_TILT_GAIN=0.0 (recovers exp#5b controller).
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/hl_orient
source /root/tdmpc_glass/venv/bin/activate
LOGROOT=/root/tdmpc_glass/exp/tdmpc_glass/hl_orient/logs
mkdir -p "$LOGROOT"
NT=${NT:-45000000}
COMMON="--env_name PandaPickCube --num_timesteps $NT --num_evals 24 --num_videos 0 --logdir $LOGROOT"
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo

launch () { # gpu suffix seed extra_env...
  local GPU=$1 SUF=$2 SEED=$3 ; shift 3
  local LOG="$LOGROOT/run_${SUF}.log"
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 "$@" \
    setsid nohup python3 run_ppo_residual_orient.py $COMMON --seed $SEED --suffix $SUF \
    > "$LOG" 2>&1 &
  echo "launched $SUF on GPU$GPU pid=$! log=$LOG"
}

# orientation-aware + residual (alpha=1.0), seed 1 and 2
launch 0 orient_a1_s1 1 env RES_ALPHA_FIXED=1.0 ORIENT_TILT_GAIN=2.0 ORIENT_TILT_R0=0.55 ORIENT_TILT_MAX=0.9 ORIENT_RELEVEL_FROM=4
launch 1 orient_a1_s2 2 env RES_ALPHA_FIXED=1.0 ORIENT_TILT_GAIN=2.0 ORIENT_TILT_R0=0.55 ORIENT_TILT_MAX=0.9 ORIENT_RELEVEL_FROM=4
# top-down control + residual (alpha=1.0), seed 1 = fair baseline / reproduce 0.78
launch 2 topdown_a1_s1 1 env RES_ALPHA_FIXED=1.0 ORIENT_TILT_GAIN=0.0
echo ALL_LAUNCHED
