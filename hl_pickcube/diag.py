import os
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.4")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
k = C.Knobs()
ctrl = C.make_controller(env, k)
reset = jax.jit(jax.vmap(env.reset)); step = jax.vmap(env.step); act = jax.vmap(ctrl.act)
N=64; STEPS=300
keys = jax.random.split(jax.random.PRNGKey(0), N)
state = reset(keys); state = step(state, jp.zeros((N,8)))
cs = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape), ctrl.init_state())
def body(carry,_):
    state,cs=carry
    a,cs=act(state,cs); state=step(state,a)
    box=state.data.xpos[:,env._obj_body]; site=state.data.site_xpos[:,env._gripper_site]
    tgt=state.info["target_pos"]
    poserr=jp.linalg.norm(tgt-box,axis=-1)
    site_box=jp.linalg.norm(box-site,axis=-1)
    return (state,cs),{"phase":cs["phase"],"poserr":poserr,"site_box":site_box,"box_target":state.metrics["box_target"]}
(state,cs),tele=jax.lax.scan(body,(state,cs),None,length=STEPS)
ph=np.asarray(tele["phase"]); pe=np.asarray(tele["poserr"]); sb=np.asarray(tele["site_box"]); bt=np.asarray(tele["box_target"])
print("final phase dist:", np.bincount(ph[-1],minlength=7))
print("final poserr  mean/min/median:", pe[-1].mean(), pe[-1].min(), np.median(pe[-1]))
print("BEST poserr per env (min over t) mean/min:", pe.min(0).mean(), pe.min(0).min())
print("site_box at final mean:", sb[-1].mean(), " (grasp drift; >0.05 = cube dropped)")
print("max box_target ever, mean:", bt.max(0).mean(), " >=0.9 count:", (bt.max(0)>=0.9).sum())
# of envs that ever reach PLACE/HOLD, what is min poserr?
reachedplace=(ph>=5).any(0)
print("envs reaching PLACE/HOLD:", reachedplace.sum(), "their best poserr:", pe[:,reachedplace].min(0).mean() if reachedplace.sum() else None)
