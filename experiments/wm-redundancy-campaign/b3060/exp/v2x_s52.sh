#!/usr/bin/env bash
cd /root/helios_wmablate
GPU=0 ABLATE=none TASK=HopperHop SEED=52 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
GPU=1 ABLATE=consistency TASK=HopperHop SEED=52 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
GPU=2 ABLATE=none TASK=WalkerRun SEED=52 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
GPU=3 ABLATE=consistency TASK=WalkerRun SEED=52 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
wait; echo DONE > /root/helios_wmablate/exp/V2X_S52_DONE
