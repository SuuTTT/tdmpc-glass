"""Monkeypatch registry.load so that loading 'PandaPickCube' returns the
experiment-#5 ResidualPickCube (live analytic option-controller + learned residual
with phase fed into obs). train_jax_ppo.py, eval_ckpts.py, and the brax eval/infer
envs all call registry.load('PandaPickCube', ...), so patching here makes the whole
pipeline (train, checkpoint, eval) consistently use the residual env.

The PPO config (get_rl_config) and default config still key on the name
'PandaPickCube', which remains valid; only the env CLASS that gets instantiated is
swapped. observation_size is 77 (66 base + 7 phase one-hot + 3 sub-goal + 1 grip).

EXP#7: if RES_RETRY=1, swap in ResidualRetryPickCube (residual_env_retry.py) which
adds closed-loop retry (failure->re-approach) to the live phase machine. Same 77-d
obs / action size, so all brax infra is unchanged.
"""
import sys, os
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
from mujoco_playground import registry

_RETRY = os.environ.get("RES_RETRY", "0") == "1"
if _RETRY:
    import residual_env_retry as RER
    import residual_env as RE
    _ENV_CLS = RER.ResidualRetryPickCube
    _TAG = "ResidualRetryPickCube (exp#7 closed-loop retry)"
else:
    import residual_env as RE
    _ENV_CLS = RE.ResidualPickCube
    _TAG = "ResidualPickCube (exp#5)"

_orig_load = registry.load


def _patched_load(env_name, config=None, config_overrides=None):
    if env_name == "PandaPickCube":
        print(f"[RES] loading {_TAG} for 'PandaPickCube'", flush=True)
        return _ENV_CLS(config=config, config_overrides=config_overrides)
    return _orig_load(env_name, config, config_overrides)


registry.load = _patched_load
print(f"[RES] registry.load patched ({_TAG}); RES_RETRY={_RETRY} "
      f"ANNEAL_STEPS={RE.ANNEAL_STEPS} "
      f"ALPHA=[{RE.ALPHA_MIN},{RE.ALPHA_MAX}] FIXED='{RE.ALPHA_FIXED}' "
      f"shaping(grasp={RE.W_GRASP},lift={RE.W_LIFT})", flush=True)
