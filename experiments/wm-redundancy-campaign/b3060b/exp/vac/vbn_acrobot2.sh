#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=0 VBN_DIM=16  TASK=AcrobotSwingup SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
GPU=1 VBN_DIM=32  TASK=AcrobotSwingup SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
GPU=2 VBN_DIM=64  TASK=AcrobotSwingup SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
GPU=3 VBN_DIM=128 TASK=AcrobotSwingup SEED=51 TOTAL_STEPS=5000000 bash run_vbn.sh &
wait; echo DONE > /root/tdmpc_glass/exp/vac/VBN_ACROBOT2_DONE
