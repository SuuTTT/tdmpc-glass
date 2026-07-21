import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
ctrl = C.make_controller(env, C.Knobs())
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
st=reset(jax.random.PRNGKey(3)); st=step(st,jp.zeros(8)); cs=ctrl.init_state()
b0=float(np.array(st.data.xpos[env._obj_body])[2])
print("step phase fw d_box_site box_z site_z goalz dist_to_tgt")
for t in range(149):
    a,cs=act(st,cs); st=step(st,a)
    box=np.array(st.data.xpos[env._obj_body]); site=np.array(st.data.site_xpos[env._gripper_site])
    tgt=np.array(st.info["target_pos"]); fw=float(np.array(st.data.qpos)[7]); d=np.linalg.norm(box-site)
    ph=int(np.array(cs["phase"]))
    if 50<=t<=92:
        print(f"{t:3d} {ph} fw={fw:.4f} d={d:.4f} bz={box[2]:.4f} sz={site[2]:.4f} dtgt={np.linalg.norm(tgt-box):.4f}")
    if bool(st.done>0.5): print("DONE at",t); break
