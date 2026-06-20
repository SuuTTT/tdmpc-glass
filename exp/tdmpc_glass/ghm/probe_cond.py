"""Diagnostic: does the GHM future depend on the CONDITIONING ACTION at all?
If futures are ~identical for opposite actions, the model ignores conditioning ->
policy-conditioning cannot help. Loads GHM with flags.json-merged config (norm-aware via dataset).
"""
import importlib, json, os, sys
import numpy as np
sys.path.insert(0, "/root/ghm/infom")
INFOM="/root/ghm/infom"; IMPLS="/root/ghm/ogbench_impls"

def clear(*ns):
    for m in list(sys.modules):
        for n in ns:
            if m==n or m.startswith(n+"."): del sys.modules[m]; break

ghm_dir=sys.argv[1]; epoch=int(sys.argv[2]); plan_gamma=float(sys.argv[3]) if len(sys.argv)>3 else None

# env + dataset stats
clear("utils","agents"); sys.path.insert(0,IMPLS)
from utils.env_utils import make_env_and_datasets
env, ds, _ = make_env_and_datasets("antmaze-medium-navigate-singletask-task1-v0")
obs_all=np.asarray(ds["observations"]); obs_dim=obs_all.shape[1]; action_dim=np.asarray(ds["actions"]).shape[1]
obs_mean=obs_all.mean(0).astype(np.float32); obs_var=obs_all.var(0).astype(np.float32); eps=1e-8

# load GHM with merged config
clear("utils","agents"); sys.path.insert(0,INFOM)
import jax, jax.numpy as jnp
from agents.ghm import GHMAgent, get_config
from utils.flax_utils import restore_agent
cfg=get_config()
fj=os.path.join(ghm_dir,"flags.json")
norm="none"
if os.path.exists(fj):
    fl=json.load(open(fj)); norm=fl.get("obs_norm_type","none")
    for k,v in fl.get("agent",{}).items():
        if k in cfg: cfg[k]=v
agent=GHMAgent.create(0,jnp.zeros((4,obs_dim),jnp.float32),jnp.zeros((4,action_dim),jnp.float32),cfg)
agent=restore_agent(agent,ghm_dir,epoch)
g=plan_gamma if plan_gamma is not None else float(cfg["gamma_max"])
def to_g(x):
    return (x-obs_mean)/np.sqrt(obs_var+eps) if norm=="normal" else x
def from_g(x):
    return x*np.sqrt(obs_var+eps)+obs_mean if norm=="normal" else x

M=256; latent_dim=int(cfg["latent_dim"])
omin=jnp.asarray(np.minimum(to_g(obs_all.min(0)),to_g(obs_all.max(0))))
omax=jnp.asarray(np.maximum(to_g(obs_all.min(0)),to_g(obs_all.max(0))))
def jump(state_raw, action, seed):
    sg=to_g(state_raw)
    rng=jax.random.PRNGKey(seed); nr,lr=jax.random.split(rng)
    noises=jax.random.normal(nr,(M,obs_dim),jnp.float32)
    latents=jax.random.normal(lr,(M,latent_dim),jnp.float32)
    fut=agent.compute_fwd_flow_goals(noises, jnp.broadcast_to(jnp.asarray(sg)[None],(M,obs_dim)),
        jnp.broadcast_to(jnp.asarray(action,jnp.float32)[None],(M,action_dim)), latents,
        horizons=jnp.full((M,),g,jnp.float32), observation_min=omin, observation_max=omax)
    return from_g(np.asarray(fut))

# pick a few real states from dataset
idxs=np.random.RandomState(0).randint(0,len(obs_all),5)
print(f"plan_gamma={g} norm={norm}")
for ix in idxs:
    s=obs_all[ix].astype(np.float32)
    a_pos=np.ones(action_dim,np.float32); a_neg=-np.ones(action_dim,np.float32); a_zero=np.zeros(action_dim,np.float32)
    fp=jump(s,a_pos,1); fn=jump(s,a_neg,1); fz=jump(s,a_zero,1)
    # mean future xy under each action
    mp,mn,mz=fp[:,:2].mean(0),fn[:,:2].mean(0),fz[:,:2].mean(0)
    print(f"state xy={s[:2].round(2)}  mean-future-xy: +1act={mp.round(2)} -1act={mn.round(2)} 0act={mz.round(2)}  |+1 - -1|={np.linalg.norm(mp-mn):.3f}  spread(std)={fp[:,:2].std(0).round(2)}")
