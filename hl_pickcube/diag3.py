import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
k=C.Knobs(); ctrl=C.make_controller(env,k)
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
s=reset(jax.random.PRNGKey(3)); s=step(s,jp.zeros(8)); cs=ctrl.init_state()
tgt=np.asarray(s.info["target_pos"]); box0=np.asarray(s.data.xpos[env._obj_body])
print("target_pos:",tgt," box_start:",box0)
for t in range(160):
    a,cs=act(s,cs); s=step(s,a)
    ph=int(cs["phase"])
    if t in (70,73,76,80,90,100,120):
        box=np.asarray(s.data.xpos[env._obj_body]); site=np.asarray(s.data.site_xpos[env._gripper_site])
        pe=float(np.linalg.norm(tgt-box))
        print("t=%d box=%s site=%s poserr=%.3f ph=%d"%(t,box.round(3),site.round(3),pe,ph))
