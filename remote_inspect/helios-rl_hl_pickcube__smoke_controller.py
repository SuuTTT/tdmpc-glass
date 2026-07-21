#!/usr/bin/env python3
"""Smoke: verify imports + param controller d==0 reproduces HL ~0.0625 success,
and that hl_options seed0 best_theta (MLP) reproduces ~0.24 success.
Numbers from real rollouts only."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jp

HL = "/root/tdmpc_glass/helios-rl/hl_pickcube"
sys.path.insert(0, HL)
import param_controller as PC
import options_train as OT
from mujoco_playground import registry

env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
print(f"env obs={env.observation_size} act={env.action_size} KV_DIM={PC.KV_DIM}", flush=True)

EVAL = int(os.environ.get("SMOKE_EVAL", "128"))
STEPS = int(os.environ.get("SMOKE_STEPS", "150"))

# d==0 (global mode) -> HL controller baseline
rollout_g, _ = OT.build_rollout(env, STEPS, "global")
theta0_g, aux_g = OT.flatten(OT.init_theta_global(jax.random.PRNGKey(0)))
keys = jax.random.split(jax.random.PRNGKey(1234), EVAL)
t0 = time.time()
summ = jax.jit(lambda th, ks: rollout_g(th, aux_g, ks))(theta0_g, keys)
succ0 = float(jp.mean(summ["success"])); reach0 = float(jp.mean(summ["reached"]))
print(f"[d==0 global HL] success={succ0:.4f} reached={reach0:.4f} ({time.time()-t0:.0f}s) "
      f"(expect ~0.06-0.10 HL ceiling)", flush=True)

# hl_options seed0 best_theta (MLP, 1049 params) -> ~0.24
theta = np.load("/root/tdmpc_glass/exp/tdmpc_glass/hl_options/seed0/best_theta.npy")
print("loaded best_theta shape", theta.shape, flush=True)
rollout_m, _ = OT.build_rollout(env, STEPS, "mlp")
_, aux_m = OT.flatten(OT.init_theta_mlp(jax.random.PRNGKey(0)))
t0 = time.time()
summ = jax.jit(lambda th, ks: rollout_m(th, aux_m, ks))(jp.asarray(theta), keys)
succ_m = float(jp.mean(summ["success"])); reach_m = float(jp.mean(summ["reached"]))
print(f"[options seed0 MLP] success={succ_m:.4f} reached={reach_m:.4f} ({time.time()-t0:.0f}s) "
      f"(expect ~0.22-0.24)", flush=True)
print("SMOKE_OK", flush=True)
