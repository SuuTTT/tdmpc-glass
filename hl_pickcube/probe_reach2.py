import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C

env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
reset = jax.jit(jax.vmap(env.reset))
step = jax.vmap(env.step)
N = 256
keys = jax.random.split(jax.random.PRNGKey(0), N)
st = reset(keys)
st = step(st, jp.zeros((N, 8)))
tgt = np.array(st.info["target_pos"])
q0 = jp.array(st.data.qpos[:, jp.array(env._robot_arm_qposadr)])
lo = jp.array(env._lowers)[:7]
up = jp.array(env._uppers)[:7]


def reach(q, t):
    qf = C.ik_step(q, t, lo, up, 1.0, 60)
    return jp.linalg.norm(C.site_fk(qf) - t)


errs = np.array(jax.jit(jax.vmap(reach))(q0, jp.array(tgt)))
print("IK reach error to TARGET (site), percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(errs, p):.4f}")
print("targets reachable (err<0.02):", int((errs < 0.02).sum()), "/", N)
print("targets reachable (err<0.05):", int((errs < 0.05).sum()), "/", N)
print("target z range:", round(float(tgt[:, 2].min()), 3), round(float(tgt[:, 2].max()), 3))
r = np.linalg.norm(tgt, axis=1)
print("target radius range:", round(float(r.min()), 3), round(float(r.max()), 3))
far = r > 0.85
if far.sum():
    print(f"targets radius>0.85: {int(far.sum())}, reach err p50 {np.percentile(errs[far],50):.3f}")
# correlation: reachable targets that we should be able to place
print("median reach err of the 90% closest-radius targets:",
      round(float(np.percentile(errs[r < np.percentile(r, 90)], 50)), 4))
