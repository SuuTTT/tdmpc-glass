import os
os.environ.update(MUJOCO_GL="egl", MJPG_IMPL="jax", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                  XLA_PYTHON_CLIENT_MEM_FRACTION="0.3", RES_ALPHA_FIXED="1.0",
                  RES_CURRICULUM="1", RES_REPLAY_P="0.5", RES_FAILBUF_K="64", RES_SUCC_THRESH="0.9")
import sys
for p in ["/root/helios-rl/exp/tdmpc_glass/hl_curriculum",
          "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac",
          "/root/tdmpc_glass/helios-rl/hl_pickcube"]:
    sys.path.insert(0, p)
import jax_compat, numpy as np, jax, jax.numpy as jp
from mujoco_playground._src import wrapper as W
import residual_env_curriculum as REC
N=8; EP=30; T=120
base=REC.CurriculumResidualPickCube(config_overrides={"impl":"jax"})
env=W.wrap_for_brax_training(base, episode_length=EP, action_repeat=1, full_reset=True)
reset,step=jax.jit(env.reset),jax.jit(env.step)
PI="AutoResetWrapper_preserve_info"
st=reset(jax.random.split(jax.random.PRNGKey(0),N))
print("keys in info:", sorted(st.info.keys()))
print("keys in PI:", sorted(st.info[PI].keys()))
for t in range(T):
    st=step(st, jp.zeros((N,base.action_size)))
    if t in (28,29,30,31,58,59,60,61):
        dn=np.asarray(st.info.get("steps")) if "steps" in st.info else None
        print(f"t={t} done={np.asarray(st.done).astype(int)} steps={dn} "
              f"ep_max={np.round(np.asarray(st.info[PI]['ep_max_bt']),3)} "
              f"fail_n={np.asarray(st.info[PI]['fail_n'])}")
