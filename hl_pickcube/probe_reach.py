import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.5")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
ctrl = C.make_controller(env, C.Knobs())
reset=jax.jit(jax.vmap(env.reset)); step=jax.vmap(env.step); actv=jax.vmap(ctrl.act)
N=256; keys=jax.random.split(jax.random.PRNGKey(0),N)
def run(keys):
    st=reset(keys); st=step(st,jp.zeros((N,8)))
    cs=jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape), ctrl.init_state())
    def body(carry,_):
        st,cs=carry; a,cs=actv(st,cs); st=step(st,a)
        box=st.data.xpos[:,env._obj_body]; site=st.data.site_xpos[:,env._gripper_site]
        return (st,cs), dict(d=jp.linalg.norm(box-site,axis=1), reached=st.info["reached_box"], ph=cs["phase"])
    (st,cs),tele=jax.lax.scan(body,(st,cs),None,length=150)
    return tele
tele=jax.block_until_ready(run(keys))
d=np.array(tele["d"]); reached=np.array(tele["reached"]); ph=np.array(tele["ph"])
mind=d.min(axis=0)
print("min d (box-site) over episode, percentiles:")
for p in [10,25,50,75,90]: print(f"  p{p}: {np.percentile(mind,p):.4f}")
print("envs min_d<0.012:", int((mind<0.012).sum()),"/",N)
print("reached_box latched:", int((reached.max(axis=0)>0.5).sum()),"/",N)
# min d during grasp/descend phases only
grasp_mask=(ph>=1)&(ph<=2)
dg=np.where(grasp_mask, d, 1.0); mindg=dg.min(axis=0)
print("min d during DESCEND/GRASP, p50:", round(float(np.percentile(mindg,50)),4), "envs<0.012:", int((mindg<0.012).sum()))
