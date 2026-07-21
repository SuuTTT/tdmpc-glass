import os,sys
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.3")
sys.path.insert(0,"/root/helios-rl/scripts")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from scripted_pick import ScriptedPickController
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=ScriptedPickController(env)
step=jax.jit(env.step); reset=jax.jit(env.reset)
# Override target to a REACHABLE near point by editing state.info AND mocap.
def force_target(s,tp):
    info=dict(s.info); info["target_pos"]=jp.asarray(tp)
    d=s.data.replace(mocap_pos=s.data.mocap_pos.at[env._mocap_target,:].set(jp.asarray(tp)))
    return s.replace(info=info,data=d)
res=[]
rng=np.random.default_rng(7)
for sd in range(16):
    s=reset(jax.random.PRNGKey(sd)); 
    # reachable target: radial in [0.40,0.60], z in [0.20,0.32]
    ang=rng.uniform(-0.5,0.5); rr=rng.uniform(0.40,0.58); zz=rng.uniform(0.20,0.30)
    tp=np.array([rr*np.cos(ang), rr*np.sin(ang), zz])
    s=force_target(s,tp); s=step(s,jp.zeros(8)); s=force_target(s,tp); ctrl.reset()
    maxbt=0.0;minpe=9.0
    for t in range(300):
        a=ctrl(s); s=step(s,jp.asarray(a)); s=force_target(s,tp)
        box=np.asarray(s.data.xpos[env._obj_body])
        maxbt=max(maxbt,float(s.metrics["box_target"])); minpe=min(minpe,float(np.linalg.norm(tp-box)))
    res.append((maxbt,minpe))
    print("sd=%2d tgt_rad=%.2f z=%.2f maxbt=%.3f minpe=%.3f"%(sd,rr,zz,maxbt,minpe))
arr=np.array(res)
print("REACHABLE-target CPU scripted: mean maxbt=%.3f success(>=0.9)=%.3f minpe=%.3f"%(arr[:,0].mean(),(arr[:,0]>=0.9).mean(),arr[:,1].mean()))
