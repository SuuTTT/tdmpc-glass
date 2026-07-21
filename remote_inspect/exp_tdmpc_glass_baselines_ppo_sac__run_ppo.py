"""Launcher: apply jax_compat shim, then run the official train_jax_ppo.py main."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax_compat  # noqa: F401  (monkeypatches jax.device_put_replicated)
sys.argv[0] = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
