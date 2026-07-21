import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.5")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
ctrl = C.make_controller(env, C.Knobs())  # grasp_width default 0
reset=jax.jit(jax.vmap(env.reset)); step=jax.vmap(env.step); actv=jax.vmap(ctrl.act)
N=256; keys=jax.random.split(jax.random.PRNGKey(0),N)
st=reset(keys); st=step(st,jp.zeros((N,8)))
cs=jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape), ctrl.init_state())
b0=np.array(st.data.xpos[:,env._obj_body,2])
maxlift=np.zeros(N); lift_at=np.full(N,-1); dropped=np.zeros(N,bool); held_d=np.zeros(N)
peak=np.zeros(N)
for t in range(149):
    a,cs=actv(st,cs); st=step(st,a)
    box=np.array(st.data.xpos[:,env._obj_body]); site=np.array(st.data.site_xpos[:,env._gripper_site])
    lift=box[:,2]-b0; d=np.linalg.norm(box-site,axis=1)
    grasped_now=(d<0.05)
    peak=np.maximum(peak,np.where(grasped_now,lift,peak))
    maxlift=np.maximum(maxlift,lift)
# classify
got_high=peak>0.06  # ever lifted >6cm while grasped
print("ever lifted>6cm while grasped:", int(got_high.sum()),"/",N, f"= {got_high.mean():.3f}")
print("max lift over all:", round(float(maxlift.max()),3))
print("mean peak(grasped lift):", round(float(peak.mean()),3))
print("envs peak>0.15:", int((peak>0.15).sum()), " peak>0.25:", int((peak>0.25).sum()))
