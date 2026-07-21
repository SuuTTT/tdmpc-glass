"""EXP#13: monkeypatch registry.load -> orientation-aware ResidualPickCube
(residual_env_orient.py, which uses param_controller_orient with reach-dependent
grasp tilt). Same 77-d obs / action size as exp#5, so all brax infra is unchanged.

ORIENT_TILT_GAIN=0 (env var) recovers the exact top-down controller (fair baseline).
"""
import sys, os
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/hl_orient")
from mujoco_playground import registry
import residual_env_orient as RE

_ENV_CLS = RE.ResidualPickCube
_orig_load = registry.load


def _patched_load(env_name, config=None, config_overrides=None):
    if env_name == "PandaPickCube":
        print("[RES-ORIENT] loading orientation-aware ResidualPickCube", flush=True)
        return _ENV_CLS(config=config, config_overrides=config_overrides)
    return _orig_load(env_name, config, config_overrides)


registry.load = _patched_load
import param_controller_orient as PCO
print(f"[RES-ORIENT] patched. TILT gain={PCO.ORIENT_TILT_GAIN} "
      f"r0={PCO.ORIENT_TILT_R0} max={PCO.ORIENT_TILT_MAX} "
      f"ANNEAL={RE.ANNEAL_STEPS} ALPHA=[{RE.ALPHA_MIN},{RE.ALPHA_MAX}] "
      f"FIXED='{RE.ALPHA_FIXED}' shaping(grasp={RE.W_GRASP},lift={RE.W_LIFT})",
      flush=True)
