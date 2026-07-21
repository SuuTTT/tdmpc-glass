import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.6")
from pathlib import Path
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
import mediapy as media
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=C.make_controller(env,C.Knobs())
N=256; keys=jax.random.split(jax.random.PRNGKey(0),N)
reset=jax.jit(jax.vmap(env.reset)); step=jax.vmap(env.step); actv=jax.vmap(ctrl.act)
# batch run to find success env indices
def run(keys):
    st=reset(keys); st=step(st,jp.zeros((N,8)))
    cs=jax.tree_util.tree_map(lambda x: jp.broadcast_to(x,(N,)+x.shape), ctrl.init_state())
    def body(carry,_):
        st,cs=carry; a,cs=actv(st,cs); st=step(st,a)
        return (st,cs), st.metrics["box_target"]
    _,bt=jax.lax.scan(body,(st,cs),None,length=150)
    return bt
bt=np.array(jax.block_until_ready(run(keys)))  # (150,N)
maxbt=bt.max(axis=0); succ=np.where(maxbt>=0.9)[0]
print(f"batch successes: {len(succ)}/{N}; idxs(first8)={succ[:8].tolist()}",flush=True)
print(f"top maxbt: {np.sort(maxbt)[::-1][:6].round(3).tolist()}",flush=True)
# render the single env for each success key (re-run single env with that exact key)
reset1=jax.jit(env.reset); step1=jax.jit(env.step); act1=jax.jit(ctrl.act)
keys_np=np.array(keys)
out=Path("videos"); out.mkdir(exist_ok=True); fps=int(round(1.0/env.dt))
render_idx = list(succ[:2]) if len(succ)>=2 else list(np.argsort(maxbt)[::-1][:2])
for i in render_idx:
    k=jp.array(keys_np[i])
    st=reset1(k); st=step1(st,jp.zeros(8)); cs=ctrl.init_state()
    traj=[jax.device_get(st)]; mbt=0.0; sc=False
    for t in range(149):
        a,cs=act1(st,cs); st=step1(st,a); traj.append(jax.device_get(st))
        v=float(st.metrics["box_target"]); mbt=max(mbt,v); sc=sc or v>=0.9
        if bool(st.done>0.5): break
    frames=[np.asarray(f) for f in env.render(traj,height=440,width=440)]
    tag="SUCCESS" if sc else f"bt{mbt:.2f}"
    p=out/f"HL_v9_{tag}_env{i}.mp4"; media.write_video(str(p),frames,fps=fps)
    print(f"  wrote {p} maxbt={mbt:.3f} (single-env replay of batch key)",flush=True)
