import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.5")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
ctrl = C.make_controller(env, C.Knobs())
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
for sd in [0,1,5]:
  st=reset(jax.random.PRNGKey(sd)); st=step(st,jp.zeros(8)); cs=ctrl.init_state()
  tgt=np.array(st.info["target_pos"])
  print(f"=== seed {sd} target={np.round(tgt,3)} ===")
  for t in range(149):
    a,cs=act(st,cs); st=step(st,a)
    box=np.array(st.data.xpos[env._obj_body]); site=np.array(st.data.site_xpos[env._gripper_site])
    ph=int(np.array(cs["phase"])); d=np.linalg.norm(box-site)
    if ph>=4 and t%6==0:
      print(f"  t{t} ph={ph} site={np.round(site,3)} box={np.round(box,3)} d={d:.3f} boxtgt={np.linalg.norm(box-tgt):.3f}")
    if bool(st.done>0.5): print("  DONE",t); break
