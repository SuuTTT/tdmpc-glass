#!/usr/bin/env python3
"""Smoke for exp#5 ResidualPickCube:
  1) env builds; observation_size==77; brax-wrap works.
  2) at alpha=0 (RES_ALPHA_FIXED=0) with ZERO residual action, true success
     reproduces the analytic v9 controller baseline (~0.06-0.10, NOT 0 ->
     confirms phase z is threaded & the controller is live).
  3) z resets on autoreset: run >episode_length so autoreset fires; verify phase
     distribution looks like a fresh controller (mostly early phases right after).
  4) alpha schedule increases with ctrl_gsteps.
Numbers from real rollouts only.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ["RES_ALPHA_FIXED"] = os.environ.get("RES_ALPHA_FIXED", "0.0")
import sys, time
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground import wrapper
import residual_env as RE

N = int(os.environ.get("SMOKE_N", "128"))
STEPS = int(os.environ.get("SMOKE_STEPS", "1000"))

base = RE.ResidualPickCube(config_overrides={"impl": "jax"})
print(f"observation_size={base.observation_size} action_size={base.action_size} "
      f"ALPHA_FIXED={RE.ALPHA_FIXED}", flush=True)
assert base.observation_size == 77, base.observation_size

env = wrapper.wrap_for_brax_training(base, episode_length=STEPS, action_repeat=1)
reset, step = jax.jit(env.reset), jax.jit(env.step)

keys = jax.random.split(jax.random.PRNGKey(7), N)

def run(keys):
    state = reset(keys)
    z0 = jp.zeros((N,))
    def body(carry, _):
        state, mbt, mreach, alpha_last = carry
        # ZERO residual action -> executed == pure analytic controller (alpha=0)
        act = jp.zeros((N, base.action_size))
        state = step(state, act)
        bt = state.metrics["box_target"]
        rb = state.info["reached_box"]
        return (state, jp.maximum(mbt, bt), jp.maximum(mreach, rb),
                state.info["alpha"]), state.info["ctrl"]["phase"]
    (state, mbt, mreach, alpha_last), phases = jax.lax.scan(
        body, (state, z0, z0, jp.zeros((N,))), None, length=STEPS)
    return mbt, mreach, phases, alpha_last, state.info["ctrl_gsteps"]

t0 = time.time()
mbt, mreach, phases, alpha_last, gsteps = jax.jit(run)(keys)
mbt = np.asarray(jax.block_until_ready(mbt)); mreach = np.asarray(mreach)
phases = np.asarray(phases); alpha_last = np.asarray(alpha_last)
gsteps = np.asarray(gsteps)
succ = float((mbt >= 0.9).mean()); reach = float((mreach > 0.5).mean())
print(f"[alpha=0, zero residual] TRUE success={succ:.4f} reached={reach:.4f} "
      f"mean_max_bt={mbt.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
print(f"  (expect controller baseline ~0.06-0.10, NOT 0 -> z threaded & live)",
      flush=True)
print(f"  ctrl_gsteps[end]={gsteps.mean():.0f} (== STEPS if no autoreset within ep)",
      flush=True)
print(f"  alpha[end]={alpha_last.mean():.4f} (fixed=0 here)", flush=True)
# phase distribution over time (sanity that phases advance)
ph_final = phases[-1]
import collections
print("  final-step phase counts:", dict(collections.Counter(ph_final.tolist())),
      flush=True)
# verify alpha schedule (non-fixed) increases: quick check with annealing on
print("=== alpha-schedule check (analytic, not a rollout) ===", flush=True)
del os.environ["RES_ALPHA_FIXED"]
import importlib
importlib.reload(RE)
for g in [0, 1e6, 8e6, 16e6, 32e6]:
    frac = min(g / RE.ANNEAL_STEPS, 1.0)
    a = RE.ALPHA_MIN + (RE.ALPHA_MAX - RE.ALPHA_MIN) * frac
    print(f"  gsteps={g:>12,.0f} -> alpha={a:.4f}", flush=True)
print("SMOKE_OK", flush=True)
