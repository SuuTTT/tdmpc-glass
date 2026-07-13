#!/usr/bin/env bash
cd /root/helios_wmablate
GPU=0 ABLATE=none TASK=HopperHop SEED=54 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
GPU=0 ABLATE=consistency TASK=HopperHop SEED=54 TOTAL_STEPS=2500000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 bash run_arm_v2.sh &
wait; echo DONE > /root/helios_wmablate/exp/V2H_S54_DONE
