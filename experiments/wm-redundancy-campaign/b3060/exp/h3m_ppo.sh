#!/usr/bin/env bash
cd /root/helios-rl
PY=/root/helios-rl/.venv/bin/python
export PYTHONPATH=/root/helios_wmablate/src:/root/mujoco_playground_repo
export MJPG_IMPL=jax MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export HOP_SPEED=1.0 HOP_MARGIN=1.0
CUDA_VISIBLE_DEVICES=2 TDMPC_GLASS_OUTPUT_TAG=h3m_speed1margin1_s50 nohup $PY -u scripts/run_benchmark.py --algos ppo --tasks HopperHop --total_steps 20000000 --seed 50 > /root/helios_wmablate/exp/wm_head_ablation/logs/h3m_ppo_s50.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 TDMPC_GLASS_OUTPUT_TAG=h3m_speed1margin1_s51 nohup $PY -u scripts/run_benchmark.py --algos ppo --tasks HopperHop --total_steps 20000000 --seed 51 > /root/helios_wmablate/exp/wm_head_ablation/logs/h3m_ppo_s51.log 2>&1 &
wait; echo DONE > /root/helios_wmablate/exp/H3_MARGIN_DONE
