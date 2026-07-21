import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco.mjx._src import math
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
    box=s.data.xpos[:,env._obj_body]
    box_mat=s.data.xmat[:,env._obj_body]
    tmat=jax.vmap(lambda q: math.quat_to_mat(q))(s.data.mocap_quat[:,env._mocap_target])
    rot_err=jp.linalg.norm(tmat.reshape(N,-1)[:,:6]-box_mat.reshape(N,-1)[:,:6],axis=-1)
    pe=jp.linalg.norm(s.info["target_pos"]-box,axis=-1)
    pos_only=jp.linalg.norm(s.info["target_pos"]-box,axis=-1)
    return (s,cs),{"pe":pe,"rot":rot_err,"bt":s.metrics["box_target"],"rb":s.info["reached_box"]}
(s,cs),tele=jax.lax.scan(body,(s,cs),None,length=STEPS)
pe=np.asarray(tele["pe"]);rot=np.asarray(tele["rot"]);bt=np.asarray(tele["bt"]);rb=np.asarray(tele["rb"])
# at min-pe step per env
ti=pe.argmin(0); idx=np.arange(N)
print("at min-pe step: pos_err mean=%.3f  rot_err mean=%.3f  bt mean=%.3f"%(pe[ti,idx].mean(),rot[ti,idx].mean(),bt[ti,idx].mean()))
# what bt WOULD be if rot_err=0:
pe_min=pe[ti,idx]; rb_min=rb[ti,idx]
bt_norot=(1-np.tanh(5*0.9*pe_min))*rb_min
print("hypothetical bt if rot_err=0 (at min-pe step): mean=%.3f  frac>=0.9=%.3f"%(bt_norot.mean(),(bt_norot>=0.9).mean()))
# rot_err distribution at min-pe step
print("rot_err at min-pe: min/median/max",rot[ti,idx].min(),np.median(rot[ti,idx]),rot[ti,idx].max())
