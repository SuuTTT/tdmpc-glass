import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from mujoco.mjx._src import math
import controller as C

env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
# pos-only controller (ori_w=0) — measure the cube rotation it produces.
ctrl = C.make_controller(env, C.Knobs(ori_w=0.0))
reset = jax.jit(jax.vmap(env.reset)); step = jax.vmap(env.step); actv = jax.vmap(ctrl.act)
N = 256; keys = jax.random.split(jax.random.PRNGKey(0), N)
mocap_id = env._mocap_target


def run(keys):
    st = reset(keys); st = step(st, jp.zeros((N, 8)))
    cs = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (N,) + x.shape), ctrl.init_state())
    def body(carry, _):
        st, cs = carry; a, cs = actv(st, cs); st = step(st, a)
        box = st.data.xpos[:, env._obj_body]; tgt = st.info["target_pos"]
        box_mat = st.data.xmat[:, env._obj_body].reshape(box.shape[0], 9)
        tmat = jax.vmap(math.quat_to_mat)(st.data.mocap_quat[:, mocap_id])
        tmat = tmat.reshape(box.shape[0], 9)
        rot_err = jp.linalg.norm(tmat[:, :6] - box_mat[:, :6], axis=1)
        pos_err = jp.linalg.norm(tgt - box, axis=1)
        bt = (1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))) * st.info["reached_box"]
        return (st, cs), dict(pos=pos_err, rot=rot_err, bt=bt, reached=st.info["reached_box"])
    (st, cs), tele = jax.lax.scan(body, (st, cs), None, length=150)
    return tele


tele = jax.block_until_ready(run(keys))
pos = np.array(tele["pos"]); rot = np.array(tele["rot"]); bt = np.array(tele["bt"])
reached = np.array(tele["reached"]).max(axis=0) > 0.5
# at the step where pos_err is minimized (per env), what is rot_err?
amin = pos.argmin(axis=0)
pos_at = pos[amin, np.arange(N)]; rot_at = rot[amin, np.arange(N)]; bt_at = bt[amin, np.arange(N)]
m = reached & (pos_at < 0.03)
print("among reached & pos_err<3cm (count %d):" % int(m.sum()))
print("  pos_err at closest: p50 %.3f p90 %.3f" % (np.percentile(pos_at[m], 50), np.percentile(pos_at[m], 90)))
print("  rot_err at closest: p50 %.3f p90 %.3f" % (np.percentile(rot_at[m], 50), np.percentile(rot_at[m], 90)))
print("  box_target at closest: p50 %.3f p90 %.3f max %.3f" % (
    np.percentile(bt_at[m], 50), np.percentile(bt_at[m], 90), bt_at[m].max()))
# what box_target WOULD be if rot_err=0:
bt_norot = (1 - np.tanh(5 * (0.9 * pos_at[m]))) * 1.0
print("  box_target IF rot_err=0: p50 %.3f p90 %.3f, count>=0.9 %d" % (
    np.percentile(bt_norot, 50), np.percentile(bt_norot, 90), int((bt_norot >= 0.9).sum())))
