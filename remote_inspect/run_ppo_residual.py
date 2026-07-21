"""Exp#5 launcher: jax_compat shim + residual_patch (swaps PandaPickCube ->
ResidualPickCube), then run the official train_jax_ppo.py main. Reuses ALL brax
PPO infra (checkpointing, eval, logging). Pass --env_name PandaPickCube."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import jax_compat  # noqa: F401
import residual_patch  # noqa: F401  (patches registry.load -> ResidualPickCube)
sys.argv[0] = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
