#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=0 VBN_DIM=16  TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
GPU=1 VBN_DIM=32  TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
GPU=2 VBN_DIM=64  TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
GPU=3 VBN_DIM=128 TASK=CheetahRun SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
wait; echo DONE > /root/tdmpc_glass/exp/vac/A1_CHEETAH_S51_DONE
