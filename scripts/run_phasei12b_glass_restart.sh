#!/usr/bin/env bash
# Phase i12b — basin-entry robustness: GLASS off@1M + CALIBRATED restart-on-plateau.
# Stacks the two best iteration-11 levers: Glass's modest mean edge (off@1M, N32/K8)
# + restart-on-plateau (re-init pi+q+opt, keep encoder/dynamics/reward/replay) to
# convert plateau-at-mediocre seeds into fresh basin shots. Hypothesis: Glass lifts
# the mean while restart fixes the robustness -> >=3/5 G1. One seed per queue task.
set -u; set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}
[[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

PROBE_ID=${PROBE_ID:-phasei12b_glass_restart}
SEEDS=${SEEDS:-1}
K_UPDATE=${K_UPDATE:-128}
MPPI_NS=${MPPI_NS:-2048}
EXPL_UNTIL=${EXPL_UNTIL:-25000}
TOTAL_STEPS=${TOTAL_STEPS:-10000000}
TEMP_STABILITY=${TEMP_STABILITY:-0.0}
GLASS_WARMUP=${GLASS_WARMUP:-100000}
GLASS_DECAY=${GLASS_DECAY:-1000000}
PROTO_TEMP=${PROTO_TEMP:-0.7}
ASSIGN_SCALE=${ASSIGN_SCALE:-0.5}
STOPGRAD=${STOPGRAD:-true}
NUM_PROTOTYPES=${NUM_PROTOTYPES:-32}
NUM_CLUSTERS=${NUM_CLUSTERS:-8}
LATENT_SMOOTH=${LATENT_SMOOTH:-0.0}
LATENT_SMOOTH_WARMUP=${LATENT_SMOOTH_WARMUP:-0}
RESTART_CHECK_AT=${RESTART_CHECK_AT:-2000000}
RESTART_THRESHOLD=${RESTART_THRESHOLD:-430}
RESTART_MAX_ATTEMPTS=${RESTART_MAX_ATTEMPTS:-4}
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}; TAG_PROBE=${PROBE_ID//[^A-Za-z0-9_.-]/_}
export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-${TAG_PROBE}_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"; status=0
{
  echo "[${PROBE_ID}] start $(date -u +%FT%TZ) seeds=${SEEDS} sha=${CODE_SHA} tag=${TDMPC_GLASS_OUTPUT_TAG}"
  echo "[${PROBE_ID}] glass off@1M N${NUM_PROTOTYPES}/K${NUM_CLUSTERS} + RESTART thr=${RESTART_THRESHOLD}@${RESTART_CHECK_AT} max=${RESTART_MAX_ATTEMPTS}"
} | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[${PROBE_ID}] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --k_update "$K_UPDATE" --mppi_n_samples "$MPPI_NS" --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$LATENT_SMOOTH" --latent_smooth_warmup_env_steps "$LATENT_SMOOTH_WARMUP" \
    --glass_warmup_env_steps "$GLASS_WARMUP" --glass_decay_steps "$GLASS_DECAY" \
    --glass_proto_temperature "$PROTO_TEMP" --glass_assign_logits_init_scale "$ASSIGN_SCALE" \
    --glass_lambda_temp_stability "$TEMP_STABILITY" --glass_stopgrad_graph "$STOPGRAD" \
    --glass_num_prototypes "$NUM_PROTOTYPES" --glass_num_clusters "$NUM_CLUSTERS" \
    --restart_on_plateau --restart_check_at "$RESTART_CHECK_AT" \
    --restart_threshold "$RESTART_THRESHOLD" --restart_max_attempts "$RESTART_MAX_ATTEMPTS" \
    --no_plot 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && status=$rc
  echo "[${PROBE_ID}] === seed=${seed} done rc=${rc} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[${PROBE_ID}] all done status=${status} at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
exit $status
