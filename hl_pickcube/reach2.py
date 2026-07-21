import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.3")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics as pk
from mujoco_playground import registry
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
lo=jp.array(env._lowers)[:7]; hi=jp.array(env._uppers)[:7]
fk=jax.jit(jax.vmap(lambda q: pk.compute_franka_fk(q)[:3,3]))
rng=np.random.default_rng(1)
Q=jp.asarray(lo+(hi-lo)*rng.random((20000,7)))
P=np.asarray(fk(Q))
r=np.linalg.norm(P[:,:2],axis=1); z=P[:,2]
print("reachable site r:",r.min().round(3),r.max().round(3)," z:",z.min().round(3),z.max().round(3))
for zb in [0.25,0.30,0.35,0.40]:
    m=np.abs(z-zb)<0.025
    print("  z~%.2f: max reach r=%.2f median r=%.2f n=%d"%(zb, r[m].max() if m.sum() else 0, np.median(r[m]) if m.sum() else 0, m.sum()))
# target dist: r in[0.5,0.92] z in[0.23,0.43]; frac of (r,z) target box that is reachable-ish
# reachable if exists sampled site within 3cm
