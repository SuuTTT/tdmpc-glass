#!/usr/bin/env python3
"""Smoke: register ResidualAcrobot, verify obs size, and check that a ZERO residual
reproduces controller-alone return (plumbing check)."""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("MUJOCO_GL", "egl")
import jax, jax.numpy as jp, numpy as np
from mujoco_playground._src import dm_control_suite
from mujoco_playground import wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residual_acrobot as RA

dm_control_suite.register_environment(
    "AcrobotSwingupResidual", RA.ResidualAcrobot, RA.default_config)

EPLEN = 1000; N = 128
env = dm_control_suite.load("AcrobotSwingupResidual", config_overrides={"impl": "jax"})
print("observation_size:", env.observation_size, " action_size:", env.action_size)
wenv = wrapper.wrap_for_brax_training(env, episode_length=EPLEN, action_repeat=1)
_step = jax.jit(wenv.step)

@jax.jit
def rollout(key):
  st = wenv.reset(jax.random.split(key, N))
  def body(carry, _):
    s, ep = carry
    a = jp.zeros((N, 1))           # ZERO residual -> pure controller
    ns = _step(s, a)
    return (ns, ep + ns.reward), None
  (_, ep), _ = jax.lax.scan(body, (st, jp.zeros(N)), None, length=EPLEN)
  return ep

ep = np.array(rollout(jax.random.PRNGKey(0)))
print(f"ZERO-residual (alpha={RA.RES_ALPHA}) RETURN mean={ep.mean():.1f} "
      f"std={ep.std():.1f} min={ep.min():.0f} max={ep.max():.0f}")
print("EXPECT == controller-alone (controller_default.json mean) -> plumbing OK")
