import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco.mjx._src import math
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
k=C.Knobs(); k.up_bias=0.05; k.place_cart_step=0.04; k.ik_iters=40
ctrl=C.make_controller(env,k)
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
def rot_err(s):
    bm=np.asarray(s.data.xmat[env._obj_body]).reshape(-1)[:6]
    tm=np.asarray(math.quat_to_mat(s.data.mocap_quat[env._mocap_target])).reshape(-1)[:6]
    return float(np.linalg.norm(tm-bm))
for sd in [0,6,11,15]:
    s=reset(jax.random.PRNGKey(sd)); s=step(s,jp.zeros(8)); cs=ctrl.init_state()
    print("--- seed",sd,"---")
    for t in range(200):
        a,cs=act(s,cs); s=step(s,a); ph=int(cs["phase"])
        if t in (18,30,45,60,90,140,199):
            print(" t=%3d ph=%d rot_err=%.3f"%(t,ph,rot_err(s)))
