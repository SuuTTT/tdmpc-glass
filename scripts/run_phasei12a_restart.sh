#!/usr/bin/env bash
# Phase i12a — Iteration 11 PIVOT: basin-entry robustness via CALIBRATED restart.
#
# Lesson from iteration-11: every clean Glass variant (off@1M/off@2M+temp/one-level/
# K2) gives only a modest mean edge over TD-MPC2 K256 but NONE is robust (>=3/5 G1).
# Per B1, the bottleneck is BASIN ENTRY: 1-2 seeds find the hopping basin (>=500),
# the rest plateau at ~300-470. restart-on-plateau (re-init pi+q+opt, keep
# encoder/dynamics/reward/replay) gives stuck seeds fresh basin shots — but the old
# phasear trigger (threshold=100) only caught DEAD seeds (<100), never the
# plateau-at-mediocre ones. This recalibrates the trigger to the real failure mode.
#
# Base: vanilla TD-MPC2 (no Glass) + the proven post-basin levers — cleanest test of
# whether restart ALONE buys robustness vs the K256 baseline (mean 362.1, 1/5 G1).
# One seed per queue task. Clean PROBE_ID + CODE_SHA tagging (no clobber of old phasear).
set -u; set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}
[[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

PROBE_ID=${PROBE_ID:-phasei12a_restart}
SEEDS=${SEEDS:-1}
K_UPDATE=${K_UPDATE:-128}
MPPI_NS=${MPPI_NS:-2048}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
TOTAL_STEPS=${TOTAL_STEPS:-10000000}
# Calibrated restart trigger (the fix): catch plateau-at-mediocre seeds, multiple shots.
RESTART_CHECK_AT=${RESTART_CHECK_AT:-2000000}
RESTART_THRESHOLD=${RESTART_THRESHOLD:-430}
RESTART_MAX_ATTEMPTS=${RESTART_MAX_ATTEMPTS:-4}
LATENT_SMOOTH=${LATENT_SMOOTH:-0.001}
LATENT_SMOOTH_WARMUP=${LATENT_SMOOTH_WARMUP:-250000}
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}
TAG_PROBE=${PROBE_ID//[^A-Za-z0-9_.-]/_}
export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-${TAG_PROBE}_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"
status=0

{
  echo "[${PROBE_ID}] start $(date -u +%FT%TZ) seeds=${SEEDS} sha=${CODE_SHA}"
  echo "[${PROBE_ID}] output_tag=${TDMPC_GLASS_OUTPUT_TAG}"
  echo "[${PROBE_ID}] RESTART check_at=${RESTART_CHECK_AT} threshold=${RESTART_THRESHOLD} max=${RESTART_MAX_ATTEMPTS} (vanilla tdmpc2)"
} | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[${PROBE_ID}] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 \
    --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" \
    --seed "$seed" \
    --k_update "$K_UPDATE" \
    --mppi_n_samples "$MPPI_NS" \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$LATENT_SMOOTH" \
    --latent_smooth_warmup_env_steps "$LATENT_SMOOTH_WARMUP" \
    --restart_on_plateau \
    --restart_check_at "$RESTART_CHECK_AT" \
    --restart_threshold "$RESTART_THRESHOLD" \
    --restart_max_attempts "$RESTART_MAX_ATTEMPTS" \
    --no_plot 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && status=$rc
  echo "[${PROBE_ID}] === seed=${seed} done rc=${rc} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[${PROBE_ID}] all done status=${status} at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
exit $status
