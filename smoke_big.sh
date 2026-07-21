#!/usr/bin/env bash
# Exp#6 SMOKE: verify (a) bigger residual policy/value net trains, and
# (b) alpha=0 reproduces the analytic controller baseline (~0.24 reached/0.0x success).
# Two short jobs on GPU0/GPU1. ~3M steps each. Net: policy(512x4), value(512x5).
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac
source /root/tdmpc_glass/venv/bin/activate
LOGROOT=/root/tdmpc_glass/exp/tdmpc_glass/hl_beatppo_big/logs; mkdir -p "$LOGROOT"
NT=${NT:-3000000}
POL=${POL:-512,512,512,512}
VAL=${VAL:-512,512,512,512,512}
COMMON="--env_name PandaPickCube --num_timesteps $NT --num_evals 4 --num_videos 0 --logdir $LOGROOT --policy_hidden_layer_sizes $POL --value_hidden_layer_sizes $VAL"
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo
launch () { local GPU=$1 SUF=$2 SEED=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 "$@" \
    setsid nohup python3 run_ppo_residual.py $COMMON --seed $SEED --suffix $SUF > "$LOGROOT/run_${SUF}.log" 2>&1 &
  echo "launched $SUF on GPU$GPU pid=$! pol=$POL val=$VAL"; }
# smoke arm 1: alpha=1.0 big-net (does it train?)
launch 0 smoke_big_a1p0 1 env RES_ALPHA_FIXED=1.0
# smoke arm 2: alpha=0.0 big-net (sanity: controller baseline, residual off)
launch 1 smoke_big_a0p0 1 env RES_ALPHA_FIXED=0.0
echo ALL_SMOKE_LAUNCHED
