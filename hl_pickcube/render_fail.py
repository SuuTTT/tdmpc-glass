import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.6")
from pathlib import Path
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C, mediapy as media
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=C.make_controller(env,C.Knobs()); N=256; keys=jax.random.split(jax.random.PRNGKey(0),N)
reset=jax.jit(jax.vmap(env.reset)); step=jax.vmap(env.step); actv=jax.vmap(ctrl.act)
def run(keys):
    st=reset(keys); st=step(st,jp.zeros((N,8)))
    cs=jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape), ctrl.init_state())
    b0=st.data.xpos[:,env._obj_body,2]
    def body(carry,_):
        st,cs=carry; a,cs=actv(st,cs); st=step(st,a)
        return (st,cs),(st.metrics["box_target"], st.data.xpos[:,env._obj_body,2]-b0)
    _,(bt,lift)=jax.lax.scan(body,(st,cs),None,length=150); return bt,lift
bt,lift=jax.block_until_ready(run(keys)); bt=np.array(bt); lift=np.array(lift)
maxbt=bt.max(axis=0); maxlift=lift.max(axis=0)
# a "drop" failure: lifted >0.1 then box_target stayed low (slipped during carry)
fail=np.where((maxlift>0.10)&(maxbt<0.3))[0]
print(f"drop-failure candidates: {len(fail)}; idx(first)={fail[:4].tolist()}",flush=True)
idx = int(fail[0]) if len(fail) else int(np.argsort(maxbt)[0])
keys_np=np.array(keys); reset1=jax.jit(env.reset); step1=jax.jit(env.step); act1=jax.jit(ctrl.act)
st=reset1(jp.array(keys_np[idx])); st=step1(st,jp.zeros(8)); cs=ctrl.init_state(); traj=[jax.device_get(st)]; mbt=0.0
for t in range(149):
    a,cs=act1(st,cs); st=step1(st,a); traj.append(jax.device_get(st)); mbt=max(mbt,float(st.metrics["box_target"]))
    if bool(st.done>0.5): break
frames=[np.asarray(f) for f in env.render(traj,height=440,width=440)]
out=Path("videos"); out.mkdir(exist_ok=True); p=out/f"HL_v9_FAIL_drop_env{idx}_bt{mbt:.2f}.mp4"
media.write_video(str(p),frames,fps=int(round(1.0/env.dt))); print(f"  wrote {p} maxbt={mbt:.3f}",flush=True)
