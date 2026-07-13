#!/usr/bin/env bash
cd /root/helios_wmablate
GPU=3 ABLATE=none TASK=WalkerRun SEED=53 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
GPU=3 ABLATE=consistency TASK=WalkerRun SEED=53 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
wait; echo DONE > /root/helios_wmablate/exp/V2W_S53_DONE
