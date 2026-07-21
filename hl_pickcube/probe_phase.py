import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C

env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
ctrl = C.make_controller(env, C.Knobs())
reset = jax.jit(jax.vmap(env.reset)); step = jax.vmap(env.step); actv = jax.vmap(ctrl.act)
N = 256; keys = jax.random.split(jax.random.PRNGKey(0), N)


def run(keys):
    st = reset(keys); st = step(st, jp.zeros((N, 8)))
    cs = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (N,) + x.shape), ctrl.init_state())
    def body(carry, _):
        st, cs = carry; a, cs = actv(st, cs); st = step(st, a)
        box = st.data.xpos[:, env._obj_body]; site = st.data.site_xpos[:, env._gripper_site]
        tgt = st.info["target_pos"]
        return (st, cs), dict(ph=cs["phase"], d=jp.linalg.norm(box - site, axis=1),
                              bt3=jp.linalg.norm(box - tgt, axis=1),
                              btxy=jp.linalg.norm(box[:, :2] - tgt[:, :2], axis=1),
                              bt=st.metrics["box_target"], reached=st.info["reached_box"])
    (st, cs), tele = jax.lax.scan(body, (st, cs), None, length=150)
    return tele


tele = jax.block_until_ready(run(keys))
ph = np.array(tele["ph"]); d = np.array(tele["d"]); bt = np.array(tele["bt"])
bt3 = np.array(tele["bt3"]); reached = np.array(tele["reached"])
maxbt = bt.max(axis=0); reach_l = reached.max(axis=0) > 0.5
print("box_target peak distribution (per env):")
for p in [50, 75, 90, 95]:
    print(f"  p{p}: {np.percentile(maxbt, p):.3f}")
print("envs maxbt>=0.9:", int((maxbt >= 0.9).sum()), " >=0.7:", int((maxbt >= 0.7).sum()),
      " >=0.5:", int((maxbt >= 0.5).sum()))
print("reached_box latched:", int(reach_l.sum()), "/", N)
# among reached-box envs, what is the best box_target?
print("among reached-box envs: maxbt p50 %.3f p90 %.3f, count>=0.9 %d" % (
    np.percentile(maxbt[reach_l], 50), np.percentile(maxbt[reach_l], 90),
    int((maxbt[reach_l] >= 0.9).sum())))
# min box-target-3d among reached envs
bt3min = bt3.min(axis=0)
print("among reached-box: min box-target-3d p50 %.3f p90 %.3f, <0.022 count %d" % (
    np.percentile(bt3min[reach_l], 50), np.percentile(bt3min[reach_l], 90),
    int((bt3min[reach_l] < 0.022).sum())))
