#!/usr/bin/env bash
cd /root/tdmpc_glass
GPU=2 VBN_DIM=64 TASK=AcrobotSwingup SEED=52 TOTAL_STEPS=5000000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 bash run_vbn.sh
echo DONE > /root/tdmpc_glass/exp/vac/A1_ACROBOT_S52_W64_DONE
