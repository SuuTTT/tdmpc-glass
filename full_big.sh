#!/usr/bin/env bash
# Exp#6 FULL: bigger residual policy/value net + longer training + dense eval
# (keep-best via per-ckpt eval). 2 arms: big-net seed1 (GPU0), big-net seed2 (GPU1).
# alpha=1.0 fixed (the #5b headline). ~70M env-steps, eval/ckpt every ~2M.
# Net: policy(512x4) vs #5b/default policy(32x4); value(512x5) vs default value(256x5).
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac
source /root/tdmpc_glass/venv/bin/activate
LOGROOT=/root/tdmpc_glass/exp/tdmpc_glass/hl_beatppo_big/logs; mkdir -p "$LOGROOT"
NT=${NT:-70000000}
NEVALS=${NEVALS:-35}     # ~70M/35 = eval+ckpt every 2M
POL=${POL:-512,512,512,512}
VAL=${VAL:-512,512,512,512,512}
COMMON="--env_name PandaPickCube --num_timesteps $NT --num_evals $NEVALS --num_videos 0 --logdir $LOGROOT --policy_hidden_layer_sizes $POL --value_hidden_layer_sizes $VAL"
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo
launch () { local GPU=$1 SUF=$2 SEED=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 "$@" \
    setsid nohup python3 run_ppo_residual.py $COMMON --seed $SEED --suffix $SUF > "$LOGROOT/run_${SUF}.log" 2>&1 &
  echo "launched $SUF on GPU$GPU pid=$! pol=$POL val=$VAL nt=$NT nevals=$NEVALS"; }
launch 0 big_a1p0_s1 1 env RES_ALPHA_FIXED=1.0
launch 1 big_a1p0_s2 2 env RES_ALPHA_FIXED=1.0
echo ALL_FULL_LAUNCHED
