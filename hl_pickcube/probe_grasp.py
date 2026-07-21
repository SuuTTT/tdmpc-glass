import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
ctrl = C.make_controller(env, C.Knobs())
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
# find a seed where grasp happens; trace one episode
st=reset(jax.random.PRNGKey(3)); st=step(st,jp.zeros(8)); cs=ctrl.init_state()
b0=float(np.array(st.data.xpos[env._obj_body])[2])
print("step phase fw ctrl7 d_box_site box_z lift")
for t in range(149):
    a,cs=act(st,cs); st=step(st,a)
    box=np.array(st.data.xpos[env._obj_body]); site=np.array(st.data.site_xpos[env._gripper_site])
    fw=float(np.array(st.data.qpos)[7]); c7=float(np.array(st.data.ctrl)[7]); d=np.linalg.norm(box-site)
    ph=int(np.array(cs["phase"]))
    if t%5==0 or ph in (2,3):
        print(f"{t:3d} {ph} fw={fw:.4f} c7={c7:.4f} d={d:.4f} bz={box[2]:.4f} lift={box[2]-b0:.4f}")
    if bool(st.done>0.5): print("DONE"); break
