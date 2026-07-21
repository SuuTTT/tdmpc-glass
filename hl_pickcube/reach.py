import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.2")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics as pk
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
lo=np.array(env._lowers)[:7]; hi=np.array(env._uppers)[:7]
def fk(q): return np.asarray(pk.compute_franka_fk(jp.asarray(q))[:3,3])
# base position: where is site at home?
home=np.array(env._init_q)[np.asarray(env._robot_arm_qposadr)]
print("site at home:",fk(home).round(3))
# Sample joint space, get reachable site positions; find max radius at z bins
rng=np.random.default_rng(0)
Q=lo+(hi-lo)*rng.random((40000,7))
P=np.array([fk(q) for q in Q[:4000]])  # subsample for speed
r=np.linalg.norm(P[:,:2],axis=1); z=P[:,2]
print("reachable site: r range",r.min().round(2),r.max().round(2)," z range",z.min().round(2),z.max().round(2))
# Target distribution: r in [0.5,0.92], z in [0.23,0.43]. What frac reachable (within 2cm of some sampled pt)?
# Better: for target z bins, max reachable r
for zb in [0.25,0.30,0.35,0.40]:
    m=np.abs(z-zb)<0.03
    print(f"  at z~{zb}: max reachable r = {r[m].max() if m.sum() else 0:.2f} (n={m.sum()})")
