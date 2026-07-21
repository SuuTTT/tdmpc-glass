import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=C.make_controller(env,C.Knobs(use_aik=1))
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
for sd in [7,1]:
  st=reset(jax.random.PRNGKey(sd)); st=step(st,jp.zeros(8)); cs=ctrl.init_state()
  b0=float(np.array(st.data.xpos[env._obj_body])[2]); prevq=None
  print(f"=== seed {sd} ===")
  for t in range(149):
    a,cs=act(st,cs); st=step(st,a)
    ph=int(np.array(cs["phase"])); q=np.array(st.data.qpos[np.array(env._robot_arm_qposadr)])
    box=np.array(st.data.xpos[env._obj_body]); site=np.array(st.data.site_xpos[env._gripper_site])
    dq = 0 if prevq is None else np.abs(q-prevq).max(); prevq=q
    if ph in (2,3,4) and t%2==0:
      print(f"  t{t} ph={ph} q7={q[6]:.2f} q5={q[4]:.2f} maxdq={dq:.3f} bz={box[2]:.3f} lift={box[2]-b0:.3f} d={np.linalg.norm(box-site):.3f}")
    if bool(st.done>0.5): print("  DONE",t); break
