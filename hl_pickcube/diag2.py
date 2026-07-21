import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
k=C.Knobs(); ctrl=C.make_controller(env,k)
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
s=reset(jax.random.PRNGKey(3)); s=step(s,jp.zeros(8)); cs=ctrl.init_state()
rows=[]
for t in range(220):
    a,cs=act(s,cs); s=step(s,a)
    box=np.asarray(s.data.xpos[env._obj_body]); site=np.asarray(s.data.site_xpos[env._gripper_site])
    fw=float(s.data.qpos[7]); ph=int(cs["phase"])
    rows.append((t,ph,box[2],float(np.linalg.norm(box-site)),fw,float(s.metrics["box_target"])))
import numpy as np
prevph=-1
for r in rows:
    if r[1]!=prevph or r[0]%20==0:
        print(f"t={r[0]:3d} ph={r[1]} box_z={r[2]:.3f} d_box_site={r[3]:.3f} fw={r[4]:.4f} bt={r[5]:.3f}")
        prevph=r[1]
