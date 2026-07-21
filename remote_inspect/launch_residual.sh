#!/usr/bin/env bash
# Exp#5: live analytic option-controller + learned residual (alpha-annealed),
# phase z fed into obs (Markov in (s,z)), milestone shaping. brax PPO via
# run_ppo_residual.py (registry.load swap). One job per GPU.
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac
source /root/tdmpc_glass/venv/bin/activate
LOGROOT=/root/tdmpc_glass/exp/tdmpc_glass/hl_subgoal/logs
mkdir -p "$LOGROOT"
COMMON="--env_name PandaPickCube --num_timesteps 40000000 --num_evals 24 --num_videos 0 --logdir $LOGROOT"
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo

launch () { # gpu suffix seed extra_env
  local GPU=$1 SUF=$2 SEED=$3 ; shift 3
  local LOG="$LOGROOT/run_${SUF}.log"
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 "$@" \
    setsid nohup python3 run_ppo_residual.py $COMMON --seed $SEED --suffix $SUF \
    > "$LOG" 2>&1 &
  echo "launched $SUF on GPU$GPU pid=$! log=$LOG"
}

# GPU0: main seed1, anneal per-env 10000 -> alpha->1.0 @ 20.48M total (50% of 40M)
launch 0 s1_anneal10k 1 env RES_ANNEAL_STEPS=10000 RES_ALPHA_MIN=0.0 RES_ALPHA_MAX=1.0
# GPU1: main seed2, same schedule
launch 1 s2_anneal10k 2 env RES_ANNEAL_STEPS=10000 RES_ALPHA_MIN=0.0 RES_ALPHA_MAX=1.0
# GPU2: ablation fast anneal per-env 5000 -> alpha->1.0 @ ~10.24M total (25%)
launch 2 s1_anneal5k 1 env RES_ANNEAL_STEPS=5000 RES_ALPHA_MIN=0.0 RES_ALPHA_MAX=1.0
# GPU3: ablation fixed alpha=0.5 (constant partial authority)
launch 3 s1_fixed0p5 1 env RES_ALPHA_FIXED=0.5
echo "ALL LAUNCHED"
