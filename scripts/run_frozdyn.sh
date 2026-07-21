#!/usr/bin/env bash
# Frozen-random-dynamics experiment (H4): FREEZE_DYN=1 zeros the dynamics-net gradients, leaving it
# at random init. Tests whether TD-MPC2's LEARNED dynamics is redundant given a value-sufficient latent.
# Compare to D2 vanilla+mlp (trained dyn). SELF-VERIFYING: runs one smoke job first; aborts if it crashes.
set -u
cd /root/helios-rl && . .venv/bin/activate
export PYTHONPATH=/root/helios-rl/src:/root/mujoco_playground_repo MUJOCO_GL=egl MJPG_IMPL=jax FREEZE_DYN=1
export JAX_COMPILATION_CACHE_DIR=/root/helios-rl/.jaxcache JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.30}
SEEDS=${SEEDS:-0,1,2}; NGPU=${NGPU:-2}

echo "=== SMOKE: FREEZE_DYN CheetahRun 60k ==="
CUDA_VISIBLE_DEVICES=0 TDMPC_GLASS_OUTPUT_TAG=frz_smoke \
  python3 -u scripts/run_benchmark.py --algos tdmpc2 --tasks CheetahRun --total_steps 60000 --seed 0 --no_plot > /tmp/frz_smoke.log 2>&1
if grep -q ",mppi," exp/tdmpc_glass/CheetahRun_frz_smoke/seed_0.csv 2>/dev/null; then
  echo "FROZDYN_SMOKE_OK"
else
  echo "FROZDYN_SMOKE_FAIL — aborting (will not run suite)"; tail -15 /tmp/frz_smoke.log; exit 1
fi

echo "=== SUITE: FREEZE_DYN vanilla, 6 tasks x seeds=$SEEDS ==="
i=0
for t in PandaPickCube PandaPickCubeOrientation PandaOpenCabinet CheetahRun HopperHop CartpoleSwingupSparse; do
  for s in ${SEEDS//,/ }; do
    g=$((i%NGPU)); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g TDMPC_GLASS_OUTPUT_TAG=frz_${t}_s${s} \
      python3 -u scripts/run_benchmark.py --algos tdmpc2 --tasks $t --total_steps 500000 --seed $s --no_plot \
        > /tmp/frz_${t}_s${s}.log 2>&1 &
    [ $((i % (NGPU*3) )) -eq 0 ] && wait
  done
done
wait; echo FROZDYN_ALL_DONE
