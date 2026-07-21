#!/usr/bin/env bash
# Exp#8 HONEST eval: UNIFORM held-out config distribution (NOT the curriculum
# distribution). eval_residual_curve.py imports the PLAIN residual_env (uniform
# reset, RES_CURRICULUM unset) -> the reported real success is on the held-out
# uniform distribution. alpha fixed=1.0 (matches training), n=256, 1000-step.
set -u
cd /root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac
source /root/helios-rl/.venv/bin/activate
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
       XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 PYTHONPATH=/root/mujoco_playground_repo
# make SURE curriculum is OFF for eval (uniform held-out)
unset RES_CURRICULUM
OUTDIR=/root/helios-rl/exp/tdmpc_glass/hl_curriculum
GPU=${GPU:-0}
evalone () { local NAME=$1 CKPT=$2
  echo "=== EVAL $NAME (uniform held-out, alpha=1.0, n=256, 1000-step) ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 eval_residual_curve.py \
    --ckpt_root "$CKPT" --n 256 --steps 1000 --alpha_override 1.0 \
    --out "$OUTDIR/eval_${NAME}.json" 2>&1 | grep -E "STEP|SUMMARY|FINAL|PEAK|Found" | tail -30
}
# args: pairs of name ckptroot
while [ $# -ge 2 ]; do evalone "$1" "$2"; shift 2; done
