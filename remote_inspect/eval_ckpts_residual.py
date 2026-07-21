"""Eval launcher for exp#5: import residual_patch (swap PandaPickCube ->
ResidualPickCube) then run the standard eval_ckpts.py real-success protocol
(box_target>=0.9, n>=256, 1000-step eval). The residual env's .step applies
a = clip(a_option + alpha*pi_res, -1, 1) with the SAME alpha schedule used in
training (alpha->1 by ANNEAL_STEPS per-env steps; at eval the per-env ctrl_gsteps
starts at 0 each episode-... actually ctrl_gsteps starts at 0 at eval reset, so
alpha would be ~0 during eval unless we force it).

IMPORTANT: at evaluation we want the policy AS TRAINED, i.e. with FULL residual
authority the checkpoint was trained under. We therefore force RES_ALPHA_FIXED to
the alpha the policy experienced at that checkpoint's training step. Pass
--res_alpha (added below) OR set RES_ALPHA_FIXED; default 1.0 (full authority,
the late-training regime where the residual must carry the policy past 0.24)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
# default: evaluate at full residual authority unless overridden
os.environ.setdefault("RES_ALPHA_FIXED", os.environ.get("EVAL_ALPHA", "1.0"))
import jax_compat  # noqa: F401
import residual_patch  # noqa: F401
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ckpts.py")
import runpy
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ckpts.py"),
               run_name="__main__")
