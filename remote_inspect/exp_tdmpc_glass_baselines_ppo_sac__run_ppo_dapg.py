"""DAPG launcher: jax_compat shim + dapg_patch (BC-augmented loss), then run the
official train_jax_ppo.py main. Gated by DAPG_DEMOS env var."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax_compat  # noqa: F401
import dapg_patch  # noqa: F401  (patches compute_ppo_loss when DAPG_DEMOS set)
sys.argv[0] = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
import runpy
runpy.run_path("/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py",
               run_name="__main__")
