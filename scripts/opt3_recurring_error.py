#!/usr/bin/env python3
"""Option-3 precondition: is converged-WM error RECURRING + STRUCTURED beyond displacement?

If error is i.i.d. noise or pure motion-magnitude (displacement), then error-prioritized
replay / test-time adaptation has no targetable headroom (= lever-11/D1 redux). If train-
derived per-region difficulty predicts HELD-OUT error beyond displacement, the precondition
passes (necessary, not sufficient — conversion to return still needs GPU training).

Train/test split BY EPISODE (even=train, odd=test). KMeans regions fit on train only.
"""
import numpy as np, json, glob, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT/"exp"/"tdmpc_glass"/"se_dump"; OUT = ROOT/"exp"/"tdmpc_glass"/"mechcheck"
RNG = np.random.default_rng(0); N_EP=12; G,V=8,64

def rankdata(x):
    _,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt); avg=((csum-cnt)+(csum-1))/2.0; return avg[inv]
def spear(a,b):
    ra,rb=rankdata(a)-rankdata(a).mean(),rankdata(b)-rankdata(b).mean()
    d=np.sqrt((ra*ra).sum()*(rb*rb).sum()); return float((ra*rb).sum()/d) if d>0 else 0.0
def partial(y,x,c):
    ry,rx=rankdata(y),rankdata(x); C=np.column_stack([np.ones(len(y)),rankdata(c)])
    ry=ry-C@np.linalg.lstsq(C,ry,rcond=None)[0]; rx=rx-C@np.linalg.lstsq(C,rx,rcond=None)[0]
    ry-=ry.mean(); rx-=rx.mean(); d=np.sqrt((ry*ry).sum()*(rx*rx).sum())
    return float((ry*rx).sum()/d) if d>0 else 0.0
def episodes(Zt):
    dist=np.linalg.norm(np.diff(Zt,axis=0),axis=1); cuts=np.sort(np.argsort(dist)[-(N_EP-1):])+1
    b=[0,*cuts.tolist(),len(Zt)]; return [(b[i],b[i+1]) for i in range(len(b)-1)]
def kmeans(X,K,it=40):
    C=X[RNG.choice(len(X),K,replace=False)].copy()
    for _ in range(it):
        lab=((X[:,None,:]-C[None])**2).sum(-1).argmin(1)
        nC=np.array([X[lab==j].mean(0) if (lab==j).any() else C[j] for j in range(K)])
        if np.allclose(nC,C): break
        C=nC
    return C
def assign(X,C): return ((X[:,None,:]-C[None])**2).sum(-1).argmin(1)

def analyze(f,K=8):
    d=np.load(f,allow_pickle=True)
    Zt=d["Zt"].astype(np.float32); Ztk=d["Ztk"].astype(np.float32); err=d["err"].astype(np.float32)
    n=min(len(Zt),len(err)); Zt,Ztk,err=Zt[:n],Ztk[:n],err[:n]; disp=np.linalg.norm(Ztk-Zt,axis=1)
    eps=episodes(Zt); tr=np.zeros(n,bool); te=np.zeros(n,bool)
    for i,(s,e) in enumerate(eps): (tr if i%2==0 else te)[s:e]=True
    C=kmeans(Zt[tr],K); lab=assign(Zt,C)
    cl_err=np.array([np.median(err[tr&(lab==j)]) if (tr&(lab==j)).any() else np.median(err[tr]) for j in range(K)])
    pred=cl_err[lab]
    yt,pt,dt=err[te],pred[te],disp[te]
    rho_recur=spear(pt,yt)                          # train-region difficulty -> held-out error
    rho_shuf=np.percentile([spear(RNG.permutation(cl_err)[lab[te]],yt) for _ in range(200)],[2.5,50,97.5])
    rho_disp=spear(dt,yt)                            # how much is just displacement
    rho_beyond=partial(yt,pt,dt)                     # region difficulty beyond displacement
    # episode-block bootstrap CI for rho_beyond
    te_eps=[i for i in range(len(eps)) if i%2==1]; epid=np.full(n,-1)
    for i,(s,e) in enumerate(eps): epid[s:e]=i
    epid_te=epid[te]
    boot=[]
    for _ in range(1000):
        ch=RNG.choice(te_eps,len(te_eps),replace=True)
        sel=np.concatenate([np.where(epid_te==c)[0] for c in ch])
        boot.append(partial(yt[sel],pt[sel],dt[sel]))
    ci=np.percentile(boot,[2.5,50,97.5])
    return {"file":os.path.basename(f),"env":str(d["env"]),"k":int(d["k"]),
            "rho_recurring":round(rho_recur,3),"rho_shuffle_ci":[round(v,3) for v in rho_shuf],
            "rho_displacement":round(rho_disp,3),
            "rho_beyond_disp":round(rho_beyond,3),"beyond_ci":[round(v,3) for v in ci]}

def main():
    groups={"Pick":sorted(glob.glob(str(DUMP/"ks_Pick_k8_s*_seed1_k8_mech.npz"))),
            "Ori":sorted(glob.glob(str(DUMP/"ks_Ori_k8_s*_seed1_k8_mech.npz"))),
            "Cab":sorted(glob.glob(str(DUMP/"ks_Cab_k8_s*_seed1_k8_mech.npz")))}
    res={}
    for name,fs in groups.items():
        res[name]=[analyze(f) for f in fs]
        for r in res[name]:
            print(f"[{name}] {r['file']}: recur={r['rho_recurring']} (shuf {r['rho_shuffle_ci'][1]}) "
                  f"disp={r['rho_displacement']} BEYOND-disp={r['rho_beyond_disp']} CI{r['beyond_ci']}")
    json.dump(res,open(OUT/"opt3_recurring_error.json","w"),indent=2)
    print("\nsaved",OUT/"opt3_recurring_error.json")
if __name__=="__main__": main()
