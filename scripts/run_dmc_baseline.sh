#!/usr/bin/env bash
# run_dmc_baseline.sh — Iteration 14 Stage 0: clean DMC baseline on the new (non-HopperHop)
# benchmark. Parameterized by TASK + ALGO so the same launcher serves the vanilla-TD-MPC2
# reproduction AND later the behavior-aware-Glass arm, under a fair single-variable protocol.
# NO procedure tricks (no restart, no PBT). One seed per queue task. Clean PROBE_ID+CODE_SHA.
set -u; set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}
[[ -n "${TMPDIR:-}" ]] && mkdir -p "$TMPDIR"

PROBE_ID=${PROBE_ID:-phasei14_dmc_baseline}
ALGO=${ALGO:-tdmpc2}
TASK=${TASK:-HumanoidWalk}
SEEDS=${SEEDS:-1}
K_UPDATE=${K_UPDATE:-128}
MPPI_NS=${MPPI_NS:-2048}
EXPL_UNTIL=${EXPL_UNTIL:-25000}
BISIM_COEF=${BISIM_COEF:-0}            # iter-14: BS-MPC bisimulation aux weight (0=vanilla)
DISTRACTOR_DIMS=${DISTRACTOR_DIMS:-0}  # iter-14 Stage-2a: OU nuisance obs dims (0=off)
LATENT_NORM=${LATENT_NORM:-simnorm}    # iter-16: simnorm (vanilla) | fsq (codebook swap)
FSQ_LEVELS=${FSQ_LEVELS:-5}            # iter-16: FSQ levels/dim (8 = pre-registered retune)
MPPI_H=${MPPI_H:-3}                    # iter-18: MPPI planning horizon (H-sweep gate for temporal abstraction)
RHO=${RHO:-}                           # iter-20: consistency-horizon decay (empty=default 0.5; 0.9=deep-planning lever)
RHO_FLAG=""; [[ -n "$RHO" ]] && RHO_FLAG="--rho $RHO"
INTRINSIC=${INTRINSIC:-none}           # iter-21: none | rnd | laplacian (exploration intrinsic reward)
INTRINSIC_COEF=${INTRINSIC_COEF:-0.0}  # iter-21: weight on normalized intrinsic reward
JUMPY_K=${JUMPY_K:-0}                   # iter-22: k-step jumpy latent head (0=off; needs MPPI_H>=2k)
JUMPY_COEF=${JUMPY_COEF:-1.0}          # iter-22: jumpy consistency + horizon-consistency weight
JUMPY_PLAN=${JUMPY_PLAN:-0}            # iter-22: 1 = eval with jumpy-MPPI (writes 'jumpy' rows)
JUMPY_NMACRO=${JUMPY_NMACRO:-3}        # iter-22: jumpy macro-steps (eff horizon = k*n_macro)
JUMPY_PLAN_FLAG=""; [[ "$JUMPY_PLAN" == "1" ]] && JUMPY_PLAN_FLAG="--jumpy_plan"
TOTAL_STEPS=${TOTAL_STEPS:-3000000}   # Stage-0 reproduction budget (sample-efficiency window)
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}; TAG_PROBE=${PROBE_ID//[^A-Za-z0-9_.-]/_}; TAG_TASK=${TASK//[^A-Za-z0-9_.-]/_}
export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-${TAG_PROBE}_${TAG_TASK}_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"; status=0
{
  echo "[${PROBE_ID}] start $(date -u +%FT%TZ) algo=${ALGO} task=${TASK} seeds=${SEEDS} sha=${CODE_SHA} tag=${TDMPC_GLASS_OUTPUT_TAG}"
} | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/${TASK}_seed_${seed}.log"
  echo "[${PROBE_ID}] === ${TASK} seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos "$ALGO" --tasks "$TASK" \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --k_update "$K_UPDATE" --mppi_n_samples "$MPPI_NS" --mppi_horizon "$MPPI_H" --expl_until "$EXPL_UNTIL" \
    $RHO_FLAG \
    --intrinsic "$INTRINSIC" --intrinsic_coef "$INTRINSIC_COEF" \
    --jumpy_k "$JUMPY_K" --jumpy_coef "$JUMPY_COEF" --jumpy_n_macro "$JUMPY_NMACRO" $JUMPY_PLAN_FLAG \
    --bisim_coef "$BISIM_COEF" \
    --distractor_dims "$DISTRACTOR_DIMS" \
    --latent_norm "$LATENT_NORM" --fsq_levels "$FSQ_LEVELS" \
    --no_plot 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && status=$rc
  echo "[${PROBE_ID}] === ${TASK} seed=${seed} done rc=${rc} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[${PROBE_ID}] all done status=${status} at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
exit $status
