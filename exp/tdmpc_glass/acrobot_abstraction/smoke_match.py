"""Exact plumbing check: same reset keys, compare controller-alone (on raw env)
vs zero-residual (on residual env). They must match per-episode (the residual env
just appends phase + executes a_ctrl+0)."""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.3")
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry, wrapper
from mujoco_playground._src import dm_control_suite
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acrobot_controller as AC
import residual_acrobot as RA
dm_control_suite.register_environment("AcrobotSwingupResidual", RA.ResidualAcrobot, RA.default_config)
EPLEN=1000; N=128
keys=jax.random.split(jax.random.PRNGKey(0),N)

# raw env, controller-alone
renv=wrapper.wrap_for_brax_training(registry.load("AcrobotSwingup",config_overrides={"impl":"jax"}),episode_length=EPLEN,action_repeat=1)
rstep=jax.jit(renv.step)
@jax.jit
def raw(keys):
    st=renv.reset(keys)
    def body(c,_):
        s,ep=c
        a,_=AC.controller(s.obs)
        ns=rstep(s,a.reshape(N,1))
        return (ns,ep+ns.reward),None
    (_,ep),_=jax.lax.scan(body,(st,jp.zeros(N)),None,length=EPLEN)
    return ep
ep_raw=np.array(raw(keys))

# residual env, zero residual
denv=wrapper.wrap_for_brax_training(dm_control_suite.load("AcrobotSwingupResidual",config_overrides={"impl":"jax"}),episode_length=EPLEN,action_repeat=1)
dstep=jax.jit(denv.step)
@jax.jit
def res(keys):
    st=denv.reset(keys)
    def body(c,_):
        s,ep=c
        ns=dstep(s,jp.zeros((N,1)))
        return (ns,ep+ns.reward),None
    (_,ep),_=jax.lax.scan(body,(st,jp.zeros(N)),None,length=EPLEN)
    return ep
ep_res=np.array(res(keys))
print(f"controller-alone(raw)  mean={ep_raw.mean():.2f}")
print(f"zero-residual(res env) mean={ep_res.mean():.2f}")
print(f"max abs per-ep diff = {np.abs(ep_raw-ep_res).max():.4f}  (should be ~0 -> plumbing OK)")
