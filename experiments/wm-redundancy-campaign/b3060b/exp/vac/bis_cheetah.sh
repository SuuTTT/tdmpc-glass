#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=0 BISIM=0.1 TASK=CheetahRun SEED=50 TOTAL_STEPS=5000000 bash run_bisim.sh &
GPU=1 BISIM=0.1 TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_bisim.sh &
GPU=2 BISIM=0.5 TASK=CheetahRun SEED=50 TOTAL_STEPS=5000000 bash run_bisim.sh &
GPU=3 BISIM=0.5 TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_bisim.sh &
wait; echo DONE > /root/tdmpc_glass/exp/vac/BIS_CHEETAH_DONE
