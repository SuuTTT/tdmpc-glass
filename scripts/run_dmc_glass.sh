#!/usr/bin/env bash
# run_dmc_glass.sh — Iteration 14 Stage 1: TD-MPC-Glass off@1M on a DMC task, with an
# optional behavior-aware coefficient (GLASS_LAMBDA_BEHAV). LAMBDA_BEHAV=0 -> geometric
# Glass (latent-similarity prototypes); >0 -> behavioral Glass (reward-grounded prototypes,
# the iter-14 contribution). Fair single-variable protocol: identical to run_dmc_baseline
# except --algos tdmpc-glass + glass flags. NO procedure tricks. One seed per queue task.
set -u; set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}
[[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

PROBE_ID=${PROBE_ID:-phasei14_dmc_glass}
TASK=${TASK:-HumanoidWalk}
SEEDS=${SEEDS:-1}
K_UPDATE=${K_UPDATE:-128}
MPPI_NS=${MPPI_NS:-2048}
EXPL_UNTIL=${EXPL_UNTIL:-25000}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
GLASS_WARMUP=${GLASS_WARMUP:-100000}
GLASS_DECAY=${GLASS_DECAY:-1000000}
PROTO_TEMP=${PROTO_TEMP:-0.7}
ASSIGN_SCALE=${ASSIGN_SCALE:-0.5}
STOPGRAD=${STOPGRAD:-true}
NUM_PROTOTYPES=${NUM_PROTOTYPES:-32}
NUM_CLUSTERS=${NUM_CLUSTERS:-8}
TEMP_STABILITY=${TEMP_STABILITY:-0.0}
LAMBDA_BEHAV=${LAMBDA_BEHAV:-0.0}     # iter-14 behavior-aware coefficient (0=geometric Glass)
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}; TAG_PROBE=${PROBE_ID//[^A-Za-z0-9_.-]/_}; TAG_TASK=${TASK//[^A-Za-z0-9_.-]/_}
export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-${TAG_PROBE}_${TAG_TASK}_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"; status=0
{
  echo "[${PROBE_ID}] start $(date -u +%FT%TZ) task=${TASK} seeds=${SEEDS} behav=${LAMBDA_BEHAV} sha=${CODE_SHA} tag=${TDMPC_GLASS_OUTPUT_TAG}"
} | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/${TASK}_seed_${seed}.log"
  echo "[${PROBE_ID}] === ${TASK} seed=${seed} behav=${LAMBDA_BEHAV} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks "$TASK" \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --k_update "$K_UPDATE" --mppi_n_samples "$MPPI_NS" --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef 0.0 --latent_smooth_warmup_env_steps 0 \
    --glass_warmup_env_steps "$GLASS_WARMUP" --glass_decay_steps "$GLASS_DECAY" \
    --glass_proto_temperature "$PROTO_TEMP" --glass_assign_logits_init_scale "$ASSIGN_SCALE" \
    --glass_lambda_temp_stability "$TEMP_STABILITY" --glass_stopgrad_graph "$STOPGRAD" \
    --glass_num_prototypes "$NUM_PROTOTYPES" --glass_num_clusters "$NUM_CLUSTERS" \
    --glass_lambda_behav "$LAMBDA_BEHAV" \
    --no_plot 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && status=$rc
  echo "[${PROBE_ID}] === ${TASK} seed=${seed} done rc=${rc} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[${PROBE_ID}] all done status=${status} at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
exit $status
