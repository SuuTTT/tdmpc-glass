#!/usr/bin/env bash
# EXP#13 eval: all 3 arms with TRAINING-MATCHED alpha=1.0 (fixed-authority), n=256,
# 1000-step, real success box_target>=0.9, reported with NEAR/FAR (r0=0.78) split.
# The orient env is used for ALL arms; ORIENT_TILT_GAIN selects orient (2.0) vs
# topdown (0.0) so each policy is evaluated under the controller it trained on.
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/hl_orient
source /root/tdmpc_glass/venv/bin/activate
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo
LR=/root/tdmpc_glass/exp/tdmpc_glass/hl_orient/logs
OUT=/root/tdmpc_glass/exp/tdmpc_glass/hl_orient
N=${N:-256}
ck () { ls -d $LR/PandaPickCube-*-$1/checkpoints 2>/dev/null | head -1; }

ev () { # gpu suffix outname tilt_gain relevel
  local GPU=$1 SUF=$2 OUTN=$3 TG=$4 RL=$5
  local C=$(ck $SUF)
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    ORIENT_TILT_GAIN=$TG ORIENT_TILT_R0=0.55 ORIENT_TILT_MAX=0.9 ORIENT_RELEVEL_FROM=$RL \
    setsid nohup python3 eval_residual_curve_orient.py --ckpt_root "$C" \
    --n $N --steps 1000 --far_r0 0.78 --alpha_override 1.0 \
    --out $OUT/$OUTN.json > $LR/eval_$OUTN.log 2>&1 &
  echo "eval $SUF -> $OUTN pid=$! ckpt=$C tilt=$TG relevel=$RL"
}

ev 0 orient_a1_s1  curve_orient_a1_s1  2.0 4
ev 1 orient_a1_s2  curve_orient_a1_s2  2.0 4
ev 2 topdown_a1_s1 curve_topdown_a1_s1 0.0 99
echo ALL_EVALS_LAUNCHED
