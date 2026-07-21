import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.5")
from pathlib import Path
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
import mediapy as media
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=C.make_controller(env,C.Knobs())
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
out=Path("videos"); out.mkdir(exist_ok=True); fps=int(round(1.0/env.dt))
results=[]
for sd in range(120):
  st=reset(jax.random.PRNGKey(sd)); st=step(st,jp.zeros(8)); cs=ctrl.init_state()
  traj=[jax.device_get(st)]; maxbt=0.0; succ=False
  for t in range(149):
    a,cs=act(st,cs); st=step(st,a); traj.append(jax.device_get(st))
    bt=float(st.metrics["box_target"]); maxbt=max(maxbt,bt)
    if bt>=0.9: succ=True
    if bool(st.done>0.5): break
  results.append((maxbt,sd,traj,succ))
  if succ: print(f"  SUCCESS seed {sd} maxbt={maxbt:.3f}",flush=True)
results.sort(key=lambda x:x[0],reverse=True)
n=sum(r[3] for r in results); print(f"successes {n}/120 (serial seeds)",flush=True)
for maxbt,sd,traj,succ in results[:3]:
  frames=[np.asarray(f) for f in env.render(traj,height=440,width=440)]
  tag="SUCCESS" if succ else f"bt{maxbt:.2f}"
  p=out/f"HL_v8_{tag}_seed{sd}.mp4"; media.write_video(str(p),frames,fps=fps)
  print(f"  wrote {p} maxbt={maxbt:.3f}",flush=True)
