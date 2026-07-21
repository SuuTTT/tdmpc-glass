#!/usr/bin/env bash
# Part 34 fuller-parametrization ori-aware residual launcher (b3060b).
# Trains AND evals one variant on one GPU. Levers via env:
#   RES_ROT_FROM (APPROACH|DESCEND|GRASP), RES_ROTERR_OBS (0|1),
#   POLHID/VALHID (comma hidden sizes). All default to Part 33 baseline.
# Usage: GPU=<g> SEED=<s> TAG=<name> [RES_ROT_FROM=..] [RES_ROTERR_OBS=1] \
#        [POLHID=128,128,128,128] launch_oriaware_v2_b3060b.sh
set -u
GPU="${GPU:?gpu}"; SEED="${SEED:-1}"; TAG="${TAG:?tag}"
BASE=/root/tdmpc_glass/exp/tdmpc_glass/generalization_PandaPickCubeOrientation_oriaware
cd "$BASE"
source /root/tdmpc_glass/venv/bin/activate
NT=${NT:-35000000}
NE=${NE:-18}
POLHID=${POLHID:-64,64,64}
VALHID=${VALHID:-64,64,64}
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
       PYTHONPATH=/root/tdmpc_glass/mujoco_playground_repo \
       XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES="$GPU" \
       RES_ORI_AWARE=1 RES_ALPHA_FIXED=1.0 \
       RES_ROT_FROM="${RES_ROT_FROM:-GRASP}" RES_ROTERR_OBS="${RES_ROTERR_OBS:-0}"

LOGD="$BASE/v2_logs_${TAG}_s${SEED}"; mkdir -p "$LOGD"
echo "[$(date -u +%FT%TZ)] V2 ${TAG} GPU$GPU seed$SEED NT=$NT ROT_FROM=${RES_ROT_FROM:-GRASP} ROTERR=${RES_ROTERR_OBS:-0} POLHID=$POLHID -> $LOGD" | tee -a "$BASE/launch.v2.b3060b.log"
python3 run_ppo_orientation_residual_oriaware_v2.py \
    --env_name PandaPickCubeOrientation --num_timesteps "$NT" --num_evals "$NE" \
    --num_videos 0 --seed "$SEED" --suffix oriAwareV2_${TAG}_s${SEED} \
    --policy_hidden_layer_sizes "$POLHID" --value_hidden_layer_sizes "$VALHID" \
    --logdir "$LOGD" >> "$LOGD/train.log" 2>&1
RC=$?
echo "[$(date -u +%FT%TZ)] V2 ${TAG} train done rc=$RC seed$SEED" | tee -a "$BASE/launch.v2.b3060b.log"
CK=$(find "$LOGD" -type d -name checkpoints | head -1)
echo "ckpt_root=$CK" | tee -a "$BASE/launch.v2.b3060b.log"
RES_ORI_AWARE=1 RES_ROT_FROM="${RES_ROT_FROM:-GRASP}" RES_ROTERR_OBS="${RES_ROTERR_OBS:-0}" \
  python3 eval_residual_orientation_oriaware_v2.py --ckpt_root "$CK" --n 256 --steps 1000 \
    --residual 1 --alpha_override 1.0 \
    --out "$BASE/v2_eval_${TAG}_s${SEED}.json" >> "$LOGD/eval.log" 2>&1
echo "[$(date -u +%FT%TZ)] V2 ${TAG} eval done rc=$? seed$SEED -> v2_eval_${TAG}_s${SEED}.json" | tee -a "$BASE/launch.v2.b3060b.log"
