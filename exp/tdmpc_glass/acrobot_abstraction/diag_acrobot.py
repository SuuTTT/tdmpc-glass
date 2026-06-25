#!/usr/bin/env python3
"""Empirically probe AcrobotSwingup dynamics: obs layout, angle/energy convention,
torque sign at the elbow, and the reward at the true upright. This informs the
Spong energy-shaping controller. Pure measurement, prints to stdout."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground import registry

env = registry.load("AcrobotSwingup", config_overrides={"impl": "jax"})
print("obs size:", env.observation_size, " act size:", env.action_size)
print("nq:", env.mjx_model.nq, " nv:", env.mjx_model.nv)

reset = jax.jit(env.reset)
step = jax.jit(env.step)

# Inspect raw obs vs qpos at a few hand-set states by stepping from reset and
# reading data. We can't set qpos directly via public API easily, so we probe
# by examining the relationship over a rollout under zero torque from many resets.

# 1) obs layout: orientations = [horiz_upper, horiz_lower, vert_upper, vert_lower], qvel(2)
# Let's confirm the upright reward by driving qpos through env internals.
import mujoco
from mujoco import mjx

mj_model = env.mj_model
shoulder_id = mj_model.joint("shoulder").qposadr[0]
elbow_id = mj_model.joint("elbow").qposadr[0]
print("shoulder qposadr:", shoulder_id, " elbow qposadr:", elbow_id)

# Build an mjx data with custom qpos to read obs & reward at known configs.
from mujoco_playground._src import mjx_env
def probe(q_shoulder, q_elbow, v_shoulder=0.0, v_elbow=0.0, action=0.0):
    data = mjx_env.make_data(
        env.mj_model,
        qpos=jp.array([q_shoulder, q_elbow]),
        impl=env.mjx_model.impl.value,
        naconmax=env._config.naconmax,
        njmax=env._config.njmax,
    )
    data = data.replace(qvel=jp.array([v_shoulder, v_elbow]))
    data = mjx.forward(env.mjx_model, data)
    obs = env._get_obs(data, {})
    rew = env._get_reward(data, jp.array([action]), {}, {})
    tip = data.site_xpos[env._tip_site_id]
    return np.array(obs), float(rew), np.array(tip)

print("\n=== config probes (q_shoulder, q_elbow) ===")
for (qs, qe) in [(0,0),(np.pi,0),(0,np.pi),(np.pi/2,0),(np.pi,np.pi),(0.1,0.1)]:
    obs, rew, tip = probe(qs, qe)
    print(f" qs={qs:+.3f} qe={qe:+.3f} -> tip={tip}  rew={rew:.4f}")
    print(f"    obs(orient4+qvel2)={np.round(obs,3)}")

# Energy probe: total mechanical energy via mjx? use data.energy if available
print("\n=== torque sign probe (apply +1 / -1 from rest at qs=pi (hanging-ish)) ===")
# 'hanging' for acrobot with arms down: shoulder angle measured from upright config?
# We'll step from reset states under constant torque to see elbow vel sign.
# step from a custom near-hanging state under constant torque to read elbow accel.
def step_probe(qs, qe, u, nsteps=10):
    data = mjx_env.make_data(env.mj_model, qpos=jp.array([qs, qe]),
        impl=env.mjx_model.impl.value, naconmax=env._config.naconmax,
        njmax=env._config.njmax)
    data = mjx.forward(env.mjx_model, data)
    st = mjx_env.State(data, env._get_obs(data, {}), jp.zeros(()), jp.zeros(()),
                       {"distance": jp.zeros(())}, {"rng": jax.random.PRNGKey(0)})
    a = jp.array([u])
    vs = []
    for _ in range(nsteps):
        st = step(st, a)
        vs.append((float(st.obs[4]), float(st.obs[5])))
    return vs

for u in [1.0, -1.0]:
    vs = step_probe(np.pi, 0.0, u, 10)  # hanging-down-ish
    print(f" u={u:+.1f} @qs=pi,qe=0: (vshoulder,velbow) after 10 steps = ({vs[-1][0]:+.4f},{vs[-1][1]:+.4f})")
for u in [1.0, -1.0]:
    vs = step_probe(0.0, 0.0, u, 5)  # at upright, elbow torque effect
    print(f" u={u:+.1f} @upright: (vshoulder,velbow) after 5 = ({vs[-1][0]:+.4f},{vs[-1][1]:+.4f})")

# energy at upright vs hanging (use mjx energy if present)
print("\n=== potential energy probe (relative) ===")
for (qs,qe,name) in [(0,0,"upright"),(np.pi,0,"both-down-folded"),(np.pi,np.pi,"shoulder-down lower-up")]:
    data = mjx_env.make_data(env.mj_model, qpos=jp.array([float(qs),float(qe)]),
        impl=env.mjx_model.impl.value, naconmax=env._config.naconmax, njmax=env._config.njmax)
    data = mjx.forward(env.mjx_model, data)
    # com height of both bodies
    zc = data.subtree_com
    print(f" {name}: subtree_com z (link coms)= {np.round(np.array(data.xipos[:,2]),3)}")
