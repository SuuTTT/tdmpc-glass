#!/usr/bin/env bash
# run_dreamer4_baseline.sh — queue launcher for the standalone DreamerV4 driver
# (scripts/run_dreamer4.py), a small block-causal Transformer world model.
# Mirrors run_dreamer_baseline.sh's env-var convention so the task_queue_daemon
# can launch DreamerV4-vs-TD-MPC2 comparison runs with the same env-string shape
# (PROBE_ID, ALGO, TASK, SEEDS, TOTAL_STEPS, CODE_SHA).
#
# Does NOT touch the hot path (run_benchmark.py / tdmpc2.py / tdmpc_glass.py /
# task_queue_daemon.py).
# CSVs land under exp/tdmpc_glass/<TASK>[_<TAG>]/seed_<seed>.csv (eval_type=dreamer4)
# plus a rollup at exp/benchmark/dreamer4_<TASK>[_<TAG>].csv.
set -u; set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}
[[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

PROBE_ID=${PROBE_ID:-dreamer4_baseline}
ALGO=${ALGO:-dreamer4}           # accepted for parity; this launcher only runs dreamer4
TASK=${TASK:-PandaPickCube}
SEEDS=${SEEDS:-1}
NUM_ENVS=${NUM_ENVS:-16}
WARMUP_ENV_STEPS=${WARMUP_ENV_STEPS:-25000}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}; TAG_PROBE=${PROBE_ID//[^A-Za-z0-9_.-]/_}; TAG_TASK=${TASK//[^A-Za-z0-9_.-]/_}
export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-${TAG_PROBE}_${TAG_TASK}_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"; status=0
{
  echo "[${PROBE_ID}] start $(date -u +%FT%TZ) algo=dreamer4 task=${TASK} seeds=${SEEDS} sha=${CODE_SHA} tag=${TDMPC_GLASS_OUTPUT_TAG}"
} | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/${TASK}_seed_${seed}.log"
  echo "[${PROBE_ID}] === ${TASK} seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_dreamer4.py \
    --task "$TASK" --seeds "$seed" \
    --total_steps "$TOTAL_STEPS" \
    --num_envs "$NUM_ENVS" --warmup_env_steps "$WARMUP_ENV_STEPS" \
    --no_plot 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && status=$rc
  echo "[${PROBE_ID}] === ${TASK} seed=${seed} done rc=${rc} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[${PROBE_ID}] all done status=${status} at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
exit $status
