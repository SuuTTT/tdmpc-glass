#!/usr/bin/env bash
# Exp#8 CURRICULUM arm: prioritized resets (oversample failed configs), alpha=1.0,
# 40M steps, 2 seeds, sharing GPU0/1 with the no-curriculum control runs.
set -u
cd /root/helios-rl/exp/tdmpc_glass/hl_curriculum
source /root/helios-rl/.venv/bin/activate
LOGROOT=/root/helios-rl/exp/tdmpc_glass/hl_curriculum/curr_logs; mkdir -p "$LOGROOT"
NT=${NT:-40000000}
COMMON="--env_name PandaPickCube --num_timesteps $NT --num_evals 21 --num_videos 0 --logdir $LOGROOT"
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/mujoco_playground_repo
launch () { local GPU=$1 SUF=$2 SEED=$3
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  RES_ALPHA_FIXED=1.0 RES_CURRICULUM=1 RES_REPLAY_P=0.5 RES_FAILBUF_K=64 RES_SUCC_THRESH=0.9 \
    setsid nohup python3 run_ppo_curriculum.py $COMMON --seed $SEED --suffix $SUF \
    > "$LOGROOT/run_${SUF}.log" 2>&1 &
  echo "launched $SUF on GPU$GPU pid=$!"; }
launch 0 curr_s1 1
launch 1 curr_s2 2
echo ALL_CURR_LAUNCHED
