import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
k=C.Knobs(); ctrl=C.make_controller(env,k)
reset=jax.jit(jax.vmap(env.reset)); step=jax.vmap(env.step); act=jax.vmap(ctrl.act)
N=128; STEPS=300
keys=jax.random.split(jax.random.PRNGKey(0),N)
s=reset(keys); s=step(s,jp.zeros((N,8)))
cs=jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape),ctrl.init_state())
tgt=np.asarray(s.info["target_pos"])
def body(carry,_):
    s,cs=carry; a,cs=act(s,cs); s=step(s,a)
    box=s.data.xpos[:,env._obj_body]; site=s.data.site_xpos[:,env._gripper_site]
    pe=jp.linalg.norm(s.info["target_pos"]-box,axis=-1)
    db=jp.linalg.norm(box-site,axis=-1)
    return (s,cs),{"pe":pe,"db":db,"bt":s.metrics["box_target"]}
(s,cs),tele=jax.lax.scan(body,(s,cs),None,length=STEPS)
pe=np.asarray(tele["pe"]); db=np.asarray(tele["db"]); bt=np.asarray(tele["bt"])
held=db.max(0)<0.05   # cube held throughout
minpe=pe.min(0)
rad=np.linalg.norm(tgt[:,:2],axis=1)  # radial xy of target
print("held(cube never slipped):",held.mean())
print("among HELD envs: min poserr mean/median:",minpe[held].mean(),np.median(minpe[held]))
for thr in [0.022,0.03,0.04,0.05,0.07]:
    print(f"  frac HELD envs with minpe<{thr}: {(minpe[held]<thr).mean():.3f}")
print("target radial xy: min/mean/max",rad.min(),rad.mean(),rad.max())
print("target z: min/mean/max",tgt[:,2].min(),tgt[:,2].mean(),tgt[:,2].max())
# reachable proxy: corr between minpe and (rad + z)
import numpy as np
reach=rad+tgt[:,2]
print("minpe by target-difficulty quartile (rad+z):")
q=np.argsort(reach)
for i,name in enumerate(["easy","q2","q3","hard"]):
    idx=q[i*N//4:(i+1)*N//4]
    print(f"  {name}: reach~{reach[idx].mean():.2f} minpe={minpe[idx].mean():.3f} succ(bt>=.9)={ (bt[:,idx].max(0)>=0.9).mean():.3f}")
