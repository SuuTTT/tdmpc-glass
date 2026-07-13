#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=0 VBN_DIM=64  TASK=WalkerRun SEED=54 TOTAL_STEPS=5000000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 bash run_vbn.sh &
GPU=0 VBN_DIM=128 TASK=WalkerRun SEED=54 TOTAL_STEPS=5000000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 bash run_vbn.sh &
wait; echo DONE > /root/tdmpc_glass/exp/vac/A1_WALKER_S54CD_DONE
