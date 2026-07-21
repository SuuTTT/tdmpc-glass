#!/usr/bin/env bash
# Exp#6 EVAL: alpha=1.0, n=256, 1000-step real success over EVERY checkpoint
# (=> keep-best/peak + final + steps_to_cross_0.66). Two arms on GPU0/GPU1.
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac
source /root/tdmpc_glass/venv/bin/activate
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo
LR=/root/tdmpc_glass/exp/tdmpc_glass/hl_beatppo_big/logs
OUT=/root/tdmpc_glass/exp/tdmpc_glass/hl_beatppo_big
N=${N:-256}
ck () { ls -d $LR/PandaPickCube-*-$1/checkpoints 2>/dev/null | head -1; }
ev () { local GPU=$1 SUF=$2 OUTN=$3; local C=$(ck $SUF)
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 setsid nohup \
    python3 eval_residual_curve.py --ckpt_root "$C" --n $N --steps 1000 --alpha_override 1.0 \
    --out $OUT/$OUTN.json > $LR/eval_$OUTN.log 2>&1 &
  echo "eval $SUF -> $OUTN pid=$! ckpt=$C"; }
ev 0 big_a1p0_s1 curve_big_a1p0_s1
ev 1 big_a1p0_s2 curve_big_a1p0_s2
echo ALL_EVALS_LAUNCHED
