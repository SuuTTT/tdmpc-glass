#!/usr/bin/env python3
"""GOAL B: return-vs-success mapping. For each REGIME, run episodes and log per
episode the TOTAL env return AND the per-COMPONENT reward breakdown (raw metric
* config weight, summed over the episode). Reward components + weights read from
pick.py default_config: gripper_box=4, box_target=8, no_floor_collision=0.25,
robot_target_qpos=0.3. box_target is GATED by reached_box (latched). Everything
is read from REAL rollouts; nothing fabricated.

Regimes:
  1 hover : TD-MPC2 checkpoint (bare pi, deterministic tanh(mu))
  2 grasp_no_place / 3 success : SCRIPTED controller; classify each episode by
    whether it crossed box_target>=0.9 (success) or grasped+lifted but missed.
"""
import os, pickle, sys, json
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.35")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
sys.path.insert(0,"/root/helios-rl/hl_pickcube")
import controller as C

WEIGHTS={"gripper_box":4.0,"box_target":8.0,"no_floor_collision":0.25,"robot_target_qpos":0.3}
COMPS=list(WEIGHTS.keys())

env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
reset=jax.jit(env.reset); step=jax.jit(env.step)

def run_episode_record(act_fn, key, ep_len):
    st=reset(key)
    comp_raw={c:0.0 for c in COMPS}; ret=0.0; maxbt=0.0; lifted=False
    box0=float(np.asarray(st.data.xpos[env._obj_body])[2])
    for t in range(ep_len-1):
        a=act_fn(st,t); st=step(st,a)
        ret+=float(st.reward)
        for c in COMPS: comp_raw[c]+=float(st.metrics[c])
        bt=float(st.metrics["box_target"]); maxbt=max(maxbt,bt)
        bz=float(np.asarray(st.data.xpos[env._obj_body])[2])
        if bz-box0>0.06: lifted=True
        if bool(st.done>0.5): break
    comp_w={c:comp_raw[c]*WEIGHTS[c] for c in COMPS}
    return {"return":ret,"maxbt":maxbt,"lifted":lifted,
            "comp_raw":comp_raw,"comp_weighted":comp_w,
            "success":maxbt>=0.9,"steps":t+1}

def regime_hover(ckpt, n_ep, ep_len):
    from helios.algorithms.tdmpc2 import Encoder, Pi
    enc=Encoder(latent_dim=512,hidden=(512,512),V=8); pi=Pi(action_dim=env.action_size,hidden=(512,512))
    params=pickle.load(open(ckpt,"rb"))["params"]
    @jax.jit
    def act_of(o):
        z=enc.apply(params["enc"],o[None]); mu,_=pi.apply(params["pi"],z); return jp.tanh(mu)[0]
    eps=[]
    for ep in range(n_ep):
        eps.append(run_episode_record(lambda st,t: act_of(jp.asarray(st.obs)), jax.random.PRNGKey(1000+ep), ep_len))
    return eps

def regime_scripted(n_ep, ep_len, knobs):
    ctrl=C.make_controller(env,knobs); act=jax.jit(ctrl.act)
    eps=[]
    for ep in range(n_ep):
        cs=ctrl.init_state()
        # need cstate threaded; wrap
        state_holder={"cs":cs}
        def af(st,t):
            a,state_holder["cs"]=act(st,state_holder["cs"]); return a
        eps.append(run_episode_record(af, jax.random.PRNGKey(2000+ep), ep_len))
    return eps

def summarize(name, eps):
    arr=np.array([e["return"] for e in eps])
    bt=np.array([e["maxbt"] for e in eps])
    cw={c:np.mean([e["comp_weighted"][c] for e in eps]) for c in COMPS}
    print(f"\n## {name}  (n={len(eps)})")
    print(f"  return: mean={arr.mean():.1f} min={arr.min():.1f} max={arr.max():.1f}")
    print(f"  max_box_target: mean={bt.mean():.3f}  success(>=0.9)={np.mean([e[\"success\"] for e in eps]):.2f}")
    print(f"  weighted comp sums (mean/ep): "+" ".join(f"{c}={cw[c]:.1f}" for c in COMPS))
    return {"name":name,"n":len(eps),"return_mean":float(arr.mean()),"return_min":float(arr.min()),
            "return_max":float(arr.max()),"maxbt_mean":float(bt.mean()),
            "success_rate":float(np.mean([e["success"] for e in eps])),
            "comp_weighted_mean":{c:float(cw[c]) for c in COMPS},
            "per_ep":[{"return":e["return"],"maxbt":e["maxbt"],"success":e["success"],
                      "comp_weighted":e["comp_weighted"]} for e in eps]}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--n",type=int,default=12)
    ap.add_argument("--ep_len",type=int,default=150); ap.add_argument("--out",default="goalB_results.json")
    ap.add_argument("--knobs",default="{}")
    a=ap.parse_args()
    knobs=C.Knobs()
    for k,v in json.loads(a.knobs).items(): setattr(knobs,k,type(getattr(knobs,k))(v))
    print("=== REGIME 1: HOVER (TD-MPC2 bare pi) ===")
    hov=regime_hover(a.ckpt,a.n,a.ep_len)
    print("=== REGIME 2/3: SCRIPTED (split by success) ===")
    scr=regime_scripted(a.n*2,a.ep_len,knobs)
    succ=[e for e in scr if e["success"]]
    fail=[e for e in scr if (not e["success"]) and e["lifted"]]
    out={}
    out["hover"]=summarize("REGIME1 reward-hack HOVER (TD-MPC2)",hov)
    if fail: out["grasp_no_place"]=summarize("REGIME2 scripted GRASP-NO-PLACE",fail)
    if succ: out["success"]=summarize("REGIME3 scripted SUCCESS",succ)
    out["scripted_overall"]=summarize("scripted ALL",scr)
    json.dump(out,open(a.out,"w"),indent=2)
    print("\nwrote",a.out)
