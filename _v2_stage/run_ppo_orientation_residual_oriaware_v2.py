"""Runner for the Part 34 orientation-AWARE residual (fuller parametrization)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import jax_compat  # noqa: F401
import residual_patch_orientation_oriaware_v2  # noqa: F401
TJP = "/root/tdmpc_glass/mujoco_playground_repo/learning/train_jax_ppo.py"
sys.argv[0] = TJP
import runpy
runpy.run_path(TJP, run_name="__main__")
