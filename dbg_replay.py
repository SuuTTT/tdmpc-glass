import os
os.environ.update(MUJOCO_GL="egl", MJPG_IMPL="jax", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                  XLA_PYTHON_CLIENT_MEM_FRACTION="0.3", RES_ALPHA_FIXED="1.0",
                  RES_CURRICULUM="1", RES_REPLAY_P="0.5", RES_FAILBUF_K="64", RES_SUCC_THRESH="0.9")
import sys
for p in ["/root/helios-rl/exp/tdmpc_glass/hl_curriculum",
          "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac",
          "/root/tdmpc_glass/helios-rl/hl_pickcube"]:
    sys.path.insert(0,p)
import jax_compat, numpy as np, jax, jax.numpy as jp
from mujoco_playground._src import wrapper as W
import residual_env_curriculum as REC
N=64; EP=30; T=900
base=REC.CurriculumResidualPickCube(config_overrides={"impl":"jax"})
env=W.wrap_for_brax_training(base, episode_length=EP, action_repeat=1, full_reset=True)
reset,step=jax.jit(env.reset),jax.jit(env.step)
PI="AutoResetWrapper_preserve_info"
st=reset(jax.random.split(jax.random.PRNGKey(11),N))
def body(carry,_):
    st=carry
    buf_before=st.info[PI]["fail_box0"]; n_before=st.info[PI]["fail_n"]
    st=step(st, jp.zeros((N,base.action_size)))
    return st,(st.info["steps"], st.info[PI]["cur_box0"], buf_before, n_before)
st,(steps_t,curbox_t,buf_t,n_t)=jax.lax.scan(body, st, None, length=T)
steps_t=np.asarray(steps_t); curbox_t=np.asarray(curbox_t); buf_t=np.asarray(buf_t); n_t=np.asarray(n_t)
# new episode = post-step steps==1 (EpisodeWrapper sets steps=action_repeat=1 on first step)
replay=0; tot=0
for t in range(T):
    for n in range(N):
        if steps_t[t,n]==1:  # first step of a new episode
            sel=np.round(curbox_t[t,n],5)
            k=int(n_t[t,n])
            if k==0: continue
            buf=np.round(buf_t[t,n,:k],5)
            tot+=1
            if np.any(np.all(buf==sel,axis=1)): replay+=1
rate=replay/max(tot,1)
print(f"new-episodes w/ non-empty buffer: {tot}; selected==a-buffer-slot (REPLAY): {replay}; rate={rate:.3f}")
print(f"final fail_n mean={np.asarray(st.info[PI]['fail_n']).mean():.1f}")
print("EXPECT rate ~= RES_REPLAY_P = 0.5")
print("REPLAY_OK" if 0.30<rate<0.70 else "REPLAY_CHECK")
