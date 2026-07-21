#!/usr/bin/env python3
"""Smoke for exp#8 curriculum mechanism (NOT a training run):
  1) brax-wrap with full_reset=True; CurriculumResidualPickCube builds; obs==77.
  2) run many short episodes (episode_length small) so autoresets fire repeatedly
     with ZERO residual action (pure controller) -> ~90% fail at short horizon, so
     the failed-config ring buffer FILLS (fail_n grows).
  3) verify that after the buffer is non-empty, a measurable fraction of resets
     draw configs that EXACTLY match a previously-seen (failed) config (replay),
     vs uniform. We measure replay by checking the placed box0 matches a buffer
     slot to float precision.
All numbers from real rollouts.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
os.environ["RES_ALPHA_FIXED"] = "1.0"
os.environ["RES_CURRICULUM"] = "1"
os.environ["RES_REPLAY_P"] = os.environ.get("RES_REPLAY_P", "0.5")
os.environ["RES_FAILBUF_K"] = os.environ.get("RES_FAILBUF_K", "64")
import sys
sys.path.insert(0, "/root/helios-rl/exp/tdmpc_glass/hl_curriculum")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import jax_compat  # noqa
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground._src import wrapper as W
import residual_env_curriculum as REC

N = int(os.environ.get("SMOKE_N", "64"))
EP = int(os.environ.get("SMOKE_EP", "30"))   # short episode -> controller fails -> buffer fills
T  = int(os.environ.get("SMOKE_T", "600"))   # total steps -> ~T/EP episodes per env

base = REC.CurriculumResidualPickCube(config_overrides={"impl": "jax"})
print(f"obs={base.observation_size} act={base.action_size} replay_p={REC.P_REPLAY} K={REC.FAILBUF_K}", flush=True)
assert base.observation_size == 77

env = W.wrap_for_brax_training(base, episode_length=EP, action_repeat=1, full_reset=True)
reset, step = jax.jit(env.reset), jax.jit(env.step)
keys = jax.random.split(jax.random.PRNGKey(3), N)

PI = "AutoResetWrapper_preserve_info"

def run(keys):
    st = reset(keys)
    def body(carry, _):
        st = carry
        st = step(st, jp.zeros((N, base.action_size)))
        return st, (st.info[PI]["fail_n"],
                    st.info["cur_box0_reset"],
                    st.done,
                    st.info[PI]["ep_max_bt"])
    st, (fail_n, box0, done, epmax) = jax.lax.scan(body, st, None, length=T)
    return fail_n, box0, done, st.info[PI]["fail_box0"], st.info[PI]["fail_n"]

fail_n_t, box0_t, done_t, fail_buf, fail_n_final = jax.jit(run)(keys)
fail_n_t = np.asarray(jax.block_until_ready(fail_n_t))  # (T,N)
done_t = np.asarray(done_t)
fail_n_final = np.asarray(fail_n_final)
print(f"fail_n per env over time: start mean={fail_n_t[0].mean():.2f} "
      f"end mean={fail_n_t[-1].mean():.2f} max={fail_n_t[-1].max()}", flush=True)
print(f"total dones across run: {int(done_t.sum())} "
      f"(~{int(done_t.sum())//N} episodes/env)", flush=True)
print(f"final fail_n (>0 means buffer filled): mean={fail_n_final.mean():.2f} "
      f"min={fail_n_final.min()} max={fail_n_final.max()}", flush=True)

# replay check: count resets whose placed box0 exactly equals a buffer slot.
box0_t = np.asarray(box0_t)        # (T,N,3)
fail_buf = np.asarray(fail_buf)    # (N,K,3) final buffers
# at the LAST reset boundary, check fraction of envs whose new box0 matches a slot
# in its own final buffer (proxy for replay rate; not exact since buffer evolves).
# Better: count, over all done transitions in 2nd half, how many next-episode box0
# match SOME earlier-seen box0 for that env.
half = T // 2
replay_hits = 0; resets_2h = 0
for n in range(N):
    seen = set()
    for t in range(T):
        b = tuple(np.round(box0_t[t, n], 6))
        if t >= half and done_t[t, n] > 0.5:
            # the box0 visible AFTER this done is next episode's config
            nb = tuple(np.round(box0_t[min(t+1, T-1), n], 6))
            resets_2h += 1
            if nb in seen:
                replay_hits += 1
        seen.add(b)
rate = replay_hits / max(resets_2h, 1)
print(f"REPLAY RATE (2nd half): {replay_hits}/{resets_2h} = {rate:.3f} "
      f"(expect ~{REC.P_REPLAY:.2f} once buffer non-empty)", flush=True)
ok = (fail_n_final.mean() > 0) and (0.2 < rate < 0.8)
print("CURR_SMOKE_OK" if ok else "CURR_SMOKE_CHECK_VALUES", flush=True)
