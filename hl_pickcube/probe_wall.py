import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
import json, sys
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C

kn = C.Knobs()
if len(sys.argv) > 1:
    for k, v in json.loads(sys.argv[1]).items():
        setattr(kn, k, type(getattr(kn, k))(v))
env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
ctrl = C.make_controller(env, kn)
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
                              bt=st.metrics["box_target"], reached=st.info["reached_box"],
                              boxz=box[:,2])
    (st, cs), tele = jax.lax.scan(body, (st, cs), None, length=300)
    return tele

tele = jax.block_until_ready(run(keys))
ph = np.array(tele["ph"]); d = np.array(tele["d"]); bt = np.array(tele["bt"])
bt3 = np.array(tele["bt3"]); reached = np.array(tele["reached"]); boxz=np.array(tele["boxz"])
maxbt = bt.max(axis=0); reach_l = reached.max(axis=0) > 0.5
# phase reach: did env ever reach each phase
def everph(p): return (ph==p).any(axis=0)
print("=== phase reach counts (of %d) ===" % N)
for nm,p in [("APPROACH",0),("DESCEND",1),("GRASP",2),("LIFT",3),("TRANSPORT",4),("PLACE",5),("HOLD",6)]:
    print(f"  reached {nm}: {int(everph(p).sum())}")
print("reached_box latched:", int(reach_l.sum()))
print("success (bt>=0.9):", int((maxbt>=0.9).sum()), " bt>=0.7:", int((maxbt>=0.7).sum()))
# Among envs that got box within 3cm of target at any point:
near = (bt3 < 0.03).any(axis=0)
print("\n=== envs that brought box within 3cm of target: %d ===" % int(near.sum()))
print("  of those, reached_box latched:", int((near & reach_l).sum()))
print("  of those, success:", int((near & (maxbt>=0.9)).sum()))
# For near-but-not-success envs: at the timestep of min bt3, what was reached & bt?
bt3min_t = bt3.argmin(axis=0)
near_fail = near & (maxbt < 0.9)
idxs = np.where(near_fail)[0][:12]
print("\n=== sample near-target FAILURES (min-dist timestep) ===")
print("  env | min_bt3 | reached@that | bt@that | reach_latched")
for e in idxs:
    t = bt3min_t[e]
    print(f"  {e:4d} | {bt3[t,e]:.3f}   | {reached[t,e]:.2f}        | {bt[t,e]:.3f}  | {reach_l[e]}")
# Lift loss: envs that reached LIFT but box fell back (boxz dropped below 0.1 after lifting)
reached_lift = everph(3)
lifted_high = (boxz>0.15).any(axis=0)
print("\n=== lift analysis ===")
print("  reached LIFT phase:", int(reached_lift.sum()), " box ever >0.15z:", int(lifted_high.sum()))
