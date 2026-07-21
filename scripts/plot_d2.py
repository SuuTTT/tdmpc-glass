#!/usr/bin/env python3
"""Render D2 learning-curve plots: vanilla(mppi) vs jumpy(jumpy) per task + summary bars.
Outputs PNGs to docs/assets/d2/."""
import glob, os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ubuntu/tdmpc-glass"
G = f"{ROOT}/exp/tdmpc_glass"
OUT = f"{ROOT}/docs/assets/d2"; os.makedirs(OUT, exist_ok=True)
TASKS = [("PandaPickCube","Pick (contact)"),("PandaPickCubeOrientation","Pick-Orient (contact)"),
         ("PandaOpenCabinet","Open-Cabinet (contact)"),("CheetahRun","Cheetah (locomotion)"),
         ("HopperHop","Hopper (locomotion)"),("CartpoleSwingupSparse","Cartpole-Sparse (sparse)")]

def curves(arm, task):
    et = "mppi" if arm=="van" else "jumpy"
    per_seed={}
    for d in glob.glob(f"{G}/{task}_d2_{arm}_{task}_s*"):
        s=int(re.search(r"_s(\d+)$",d).group(1))
        cs=[c for c in glob.glob(d+"/seed_*.csv") if "_diag" not in c and "_arb" not in c]
        if not cs: continue
        pts={}
        for l in open(cs[0]):
            p=l.split(",")
            if not p[0].isdigit() or p[2]!=et: continue
            step=round(int(p[0])/50000)*50000; pts[step]=float(p[1])
        if pts: per_seed[s]=pts
    return per_seed

def agg(per_seed):
    steps=sorted({k for d in per_seed.values() for k in d})
    mean=[]; sem=[]; xs=[]
    for st in steps:
        vals=[d[st] for d in per_seed.values() if st in d and np.isfinite(d[st])]
        if len(vals)<1: continue
        xs.append(st/1000); mean.append(np.mean(vals)); sem.append(np.std(vals)/max(1,np.sqrt(len(vals))))
    return np.array(xs),np.array(mean),np.array(sem)

# ---- learning curves grid ----
fig,axes=plt.subplots(2,3,figsize=(15,8))
for ax,(task,title) in zip(axes.flat,TASKS):
    for arm,color,lab in [("van","#1f77b4","vanilla TD-MPC2 (MPPI)"),("jum","#d62728","jumpy k=8 (macro-MPPI)")]:
        ps=curves(arm,task)
        if not ps: continue
        x,m,e=agg(ps)
        ax.plot(x,m,color=color,lw=2,label=f"{lab} (n={len(ps)})")
        ax.fill_between(x,m-e,m+e,color=color,alpha=0.2)
    ax.set_title(title,fontsize=11,fontweight="bold"); ax.set_xlabel("env steps (k)"); ax.set_ylabel("return")
    ax.grid(alpha=0.3); ax.legend(fontsize=8,loc="best")
fig.suptitle("D2: jumpy vs vanilla TD-MPC2 — learning curves (mean ± SEM across seeds, 500k steps)",
             fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{OUT}/learning_curves.png",dpi=110); plt.close(fig)
print("saved learning_curves.png")

# ---- summary bars (peak & final deltas + CI) ----
d=json.load(open(f"{G}/mechcheck/d2_suite_results.json"))
order=[t for t,_ in TASKS if t in d]; labels=[dict(TASKS)[t].split(" (")[0] for t in order]
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,key,ttl in [(axes[0],"peak","Δpeak return (jumpy − vanilla)"),(axes[1],"final","Δfinal return (jumpy − vanilla)")]:
    vals=[d[t][f"delta_{key}"] for t in order]; cis=[d[t][f"delta_{key}_ci95"] for t in order]
    lo=[v-c[0] for v,c in zip(vals,cis)]; hi=[c[1]-v for v,c in zip(vals,cis)]
    cols=["#2ca02c" if c[0]>0 else ("#d62728" if c[1]<0 else "#999999") for c in cis]
    ax.barh(labels,vals,xerr=[lo,hi],color=cols,capsize=4)
    ax.axvline(0,color="k",lw=0.8); ax.set_title(ttl,fontweight="bold"); ax.grid(alpha=0.3,axis="x")
    ax.invert_yaxis()
fig.suptitle("D2 summary: green=jumpy wins (CI>0), red=jumpy hurts (CI<0), grey=null",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/summary_bars.png",dpi=110); plt.close(fig)
print("saved summary_bars.png")
print("done ->",OUT)
