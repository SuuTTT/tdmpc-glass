#!/usr/bin/env python3
"""Register ResidualAcrobot into the dm_control_suite registry, then run the
official brax train_jax_ppo main on it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")
import jax_compat  # noqa: F401

from mujoco_playground._src import dm_control_suite
import residual_acrobot as RA

dm_control_suite.register_environment(
    "AcrobotSwingupResidual", RA.ResidualAcrobot, RA.default_config
)

sys.argv[0] = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
