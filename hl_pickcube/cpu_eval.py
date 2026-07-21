import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.3")
import sys; sys.path.insert(0,"/root/helios-rl/scripts")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from scripted_pick import ScriptedPickController
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
# CPU controller needs env.mj_model etc. Check attrs.
ctrl=ScriptedPickController(env)
step=jax.jit(env.step); reset=jax.jit(env.reset)
res=[]
for sd in range(16):
    s=reset(jax.random.PRNGKey(sd)); s=step(s,jp.zeros(8)); ctrl.reset()
    tgt=np.asarray(s.info["target_pos"]); rad=float(np.linalg.norm(tgt[:2]))
    maxbt=0.0; minpe=9.0
    for t in range(300):
        a=ctrl(s); s=step(s,jp.asarray(a))
        box=np.asarray(s.data.xpos[env._obj_body])
        maxbt=max(maxbt,float(s.metrics["box_target"]))
        minpe=min(minpe,float(np.linalg.norm(tgt-box)))
    res.append((sd,rad,tgt[2],maxbt,minpe))
    print("sd=%2d tgt_rad=%.2f tgt_z=%.2f maxbt=%.3f minpe=%.3f"%(sd,rad,tgt[2],maxbt,minpe))
import numpy as np
arr=np.array([(r[3],r[4]) for r in res])
print("CPU scripted: mean maxbt=%.3f, success(>=0.9)=%.3f, mean minpe=%.3f"%(arr[:,0].mean(),(arr[:,0]>=0.9).mean(),arr[:,1].mean()))
