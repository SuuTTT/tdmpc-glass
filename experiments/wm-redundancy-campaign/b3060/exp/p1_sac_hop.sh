#!/usr/bin/env bash
cd /root/helios-rl
PY=/root/helios-rl/.venv/bin/python
export PYTHONPATH=/root/helios_wmablate/src:/root/mujoco_playground_repo
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
for i in 0 1 2; do
  S=$((50+i))
  CUDA_VISIBLE_DEVICES=$i TDMPC_GLASS_OUTPUT_TAG=p1sacbase_s$S nohup $PY -u scripts/run_benchmark.py --algos sac --tasks HopperHop --total_steps 5000000 --seed $S > /root/helios_wmablate/exp/wm_head_ablation/logs/p1_sac_hop_s$S.log 2>&1 &
done
wait; echo DONE > /root/helios_wmablate/exp/P1_SAC_HOP_DONE
