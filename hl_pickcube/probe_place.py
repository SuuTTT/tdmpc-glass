import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.5")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
ctrl = C.make_controller(env, C.Knobs())
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
for sd in [1,5,7]:
  st=reset(jax.random.PRNGKey(sd)); st=step(st,jp.zeros(8)); cs=ctrl.init_state()
  tgt=np.array(st.info["target_pos"]); ph_enter={}
  print(f"=== seed {sd} target={np.round(tgt,3)} ===")
  for t in range(149):
    a,cs=act(st,cs); st=step(st,a)
    ph=int(np.array(cs["phase"]))
    ph_enter.setdefault(ph,t)
    box=np.array(st.data.xpos[env._obj_body]); site=np.array(st.data.site_xpos[env._gripper_site])
    d=np.linalg.norm(box-site); bt=float(st.metrics["box_target"])
    if ph>=5 and t%4==0:
      print(f"  t{t} ph={ph} box={np.round(box,3)} d={d:.3f} boxtgt={np.linalg.norm(box-tgt):.3f} bt={bt:.2f}")
    if bool(st.done>0.5): print("  DONE",t); break
  print("  phase entry times:",{["A","D","G","L","T","P","H"][k]:v for k,v in sorted(ph_enter.items())})
