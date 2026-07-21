import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
k=C.Knobs(); k.up_bias=0.05; k.place_cart_step=0.04; k.ik_iters=40
ctrl=C.make_controller(env,k)
reset=jax.jit(jax.vmap(env.reset)); step=jax.vmap(env.step); act=jax.vmap(ctrl.act)
N=128; STEPS=300
keys=jax.random.split(jax.random.PRNGKey(0),N)
s=reset(keys); s=step(s,jp.zeros((N,8)))
cs=jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape),ctrl.init_state())
def body(carry,_):
    s,cs=carry; a,cs=act(s,cs); s=step(s,a)
    box=s.data.xpos[:,env._obj_body]; site=s.data.site_xpos[:,env._gripper_site]
    pe=jp.linalg.norm(s.info["target_pos"]-box,axis=-1)
    return (s,cs),{"pe":pe,"bt":s.metrics["box_target"],"rb":s.info["reached_box"],"db":jp.linalg.norm(box-site,axis=-1)}
(s,cs),tele=jax.lax.scan(body,(s,cs),None,length=STEPS)
pe=np.asarray(tele["pe"]);bt=np.asarray(tele["bt"]);rb=np.asarray(tele["rb"]);db=np.asarray(tele["db"])
print("reached_box latched(any):",(rb>0.5).any(0).mean())
print("minpe mean/median:",pe.min(0).mean(),np.median(pe.min(0)))
for thr in [0.0223,0.03,0.04,0.05]:
    print(" frac minpe<%.3f: %.3f"%(thr,(pe.min(0)<thr).mean()))
print("max box_target>=0.9:",(bt.max(0)>=0.9).mean()," >=0.8:",(bt.max(0)>=0.8).mean()," >=0.5:",(bt.max(0)>=0.5).mean())
# among envs that got pe<0.0223, was reached_box latched?
close=(pe.min(0)<0.0223)
print("envs with pe<0.0223:",close.sum(),"of those reached_box latched:",(rb[:,close]>0.5).any(0).mean() if close.sum() else None)
# at the timestep of min pe, what was bt and rb?
ti=pe.argmin(0)
btatmin=bt[ti,np.arange(N)]; rbatmin=rb[ti,np.arange(N)]
print("at min-pe step: mean bt=%.3f mean reached_box=%.3f"%(btatmin.mean(),rbatmin.mean()))
print("db at grasp (min over t) mean:",db.min(0).mean())
