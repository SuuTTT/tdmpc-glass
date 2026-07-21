"""EXP#13 launcher: jax_compat shim + residual_patch_orient (swaps PandaPickCube
-> orientation-aware ResidualPickCube), then run the official train_jax_ppo.py.
Reuses ALL brax PPO infra. Pass --env_name PandaPickCube."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/hl_orient")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")  # jax_compat
import jax_compat  # noqa: F401
import residual_patch_orient  # noqa: F401
sys.argv[0] = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
