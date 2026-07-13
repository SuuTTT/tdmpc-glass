#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=0 ARM=vac VAC_LAM=1.0 TASK=CheetahRun SEED=50 TOTAL_STEPS=5000000 bash run_vac.sh &
GPU=1 ARM=vac VAC_LAM=1.0 TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_vac.sh &
GPU=2 ARM=van VAC_LAM=0.0 TASK=CheetahRun SEED=50 TOTAL_STEPS=5000000 bash run_vac.sh &
GPU=3 ARM=van VAC_LAM=0.0 TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_vac.sh &
wait; echo DONE > /root/tdmpc_glass/exp/vac/VAC_CHEETAH_DONE
