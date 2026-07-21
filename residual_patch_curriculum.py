"""Exp#8 monkeypatch: registry.load('PandaPickCube') -> CurriculumResidualPickCube
(in-loop residual + prioritized resets), AND force the brax training wrapper to use
full_reset=True so env.reset is called on every autoreset (required for the
curriculum to re-sample / replay configs per episode).

Gated by RES_CURRICULUM=1. If unset, falls back to the plain exp#5 residual env
(so eval scripts that import this by accident stay honest). EVAL never sets
RES_CURRICULUM, so eval uses the plain uniform-distribution residual env.
"""
import os
import sys

sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_playground import registry
from mujoco_playground._src import wrapper as _wrapper

import residual_env as RE  # noqa: F401  (shared exp#5)

_CURRICULUM = os.environ.get("RES_CURRICULUM", "0") == "1"

if _CURRICULUM:
    import residual_env_curriculum as REC

    _orig_load = registry.load

    def _patched_load(env_name, config=None, config_overrides=None):
        if env_name == "PandaPickCube":
            print("[CURR] loading CurriculumResidualPickCube (exp#8)", flush=True)
            return REC.CurriculumResidualPickCube(
                config=config, config_overrides=config_overrides)
        return _orig_load(env_name, config, config_overrides)

    registry.load = _patched_load

    # force full_reset=True so reset() is called per-episode autoreset
    _orig_wrap = _wrapper.wrap_for_brax_training

    def _wrap_full_reset(*args, **kwargs):
        kwargs["full_reset"] = True  # force per-episode reset for curriculum
        return _orig_wrap(*args, **kwargs)

    _wrapper.wrap_for_brax_training = _wrap_full_reset
    # train_jax_ppo passes wrap_env_fn=wrapper.wrap_for_brax_training captured at
    # import; patch the module attr BEFORE train_jax_ppo imports it (run launcher
    # imports this patch first), so the bound reference picks up the override.
    print(f"[CURR] curriculum ON: replay_p={REC.P_REPLAY} K={REC.FAILBUF_K} "
          f"succ_thresh={REC.SUCCESS_THRESH}; full_reset forced TRUE", flush=True)
else:
    import residual_patch  # noqa: F401  (plain exp#5 patch)
    print("[CURR] RES_CURRICULUM unset -> plain residual env (exp#5)", flush=True)
