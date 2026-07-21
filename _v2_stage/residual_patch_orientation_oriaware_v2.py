"""Part 34 orientation-AWARE residual patch (fuller parametrization).

Same as Part 33's residual_patch_orientation_oriaware.py but imports
residual_env_oriaware_v2 (which imports param_controller_oriaware_v2), adding two
env-var-gated levers on top of the Part 33 baseline:

  * RES_ROT_FROM in {APPROACH,DESCEND,GRASP} (default GRASP = Part 33): apply the
    R_target rotation EARLIER so the gripper pre-aligns before contact.
  * RES_ROTERR_OBS=1: append the so(3) cube->target rotation error (3) to the obs.

Both default OFF -> exactly reproduces Part 33. Observation_size adapts.
"""
import sys
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
from mujoco_playground import registry
import residual_env_oriaware_v2 as RE


_orig_load = registry.load


def _patched_load(env_name, config=None, config_overrides=None):
    if env_name == "PandaPickCubeOrientation":
        print("[RES-ORI-AWARE-V2] loading ResidualPickCube(sample_orientation=True) "
              "v2 for 'PandaPickCubeOrientation'", flush=True)
        return RE.ResidualPickCube(config=config,
                                   config_overrides=config_overrides,
                                   sample_orientation=True)
    return _orig_load(env_name, config, config_overrides)


registry.load = _patched_load
print(f"[RES-ORI-AWARE-V2] patched; ORI_AWARE={RE.ORI_AWARE} obs_ori+={RE._N_ORI} "
      f"roterr+={RE._N_ROTERR} ROT_FROM={RE.PC.ROT_FROM} "
      f"ANNEAL={RE.ANNEAL_STEPS} ALPHA_FIXED='{RE.ALPHA_FIXED}'", flush=True)
