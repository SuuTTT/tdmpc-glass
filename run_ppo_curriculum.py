"""Exp#8 launcher: jax_compat shim + residual_patch_curriculum (swaps
PandaPickCube -> CurriculumResidualPickCube and forces full_reset=True when
RES_CURRICULUM=1), then run the official train_jax_ppo.py main. Reuses ALL brax PPO
infra (checkpointing, eval, logging). Pass --env_name PandaPickCube."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import jax_compat  # noqa: F401
import residual_patch_curriculum  # noqa: F401  (curriculum patch, gated by RES_CURRICULUM)
sys.argv[0] = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
