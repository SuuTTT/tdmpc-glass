#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=3 VBN_DIM=16  TASK=WalkerRun SEED=53 TOTAL_STEPS=5000000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 bash run_vbn.sh &
GPU=3 VBN_DIM=32  TASK=WalkerRun SEED=53 TOTAL_STEPS=5000000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 bash run_vbn.sh &
wait; echo DONE > /root/tdmpc_glass/exp/vac/A1_WALKER_S53AB_DONE
